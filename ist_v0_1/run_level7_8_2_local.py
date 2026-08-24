"""Level 7.8.2: oracle-marked selective L3 write gating and midstream rewrite."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from run_level7_1_local import atomic_save
from run_level7_4_1_local import atomic_torch_save
from run_level7_6_local import SEEDS
from run_level7_7_local import paired_exact
from run_level7_8_local import (BATCH, CHANCE, CHUNK_SIZE, SAMPLES,
                                balanced_targets, build, clone_memory,
                                forward_memory, query_chunk, random_chunk,
                                write_chunk)


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("normal_update", "freeze_l3_after_first", "oracle_marker_l3_gate")
MILESTONES = (1, 16, 128, 511, 512, 513, 640, 1000)
REWRITE_CHUNK = 512
VOCAB = 16


def route_l3(condition: str, previous, candidate, marked: bool):
    if condition == "normal_update":
        return candidate
    routed = list(candidate)
    if condition == "freeze_l3_after_first" or not marked:
        routed[2] = previous[2]
    return routed


def score(model, memory, old_target, new_target, query_seed, chunks, device, dtype):
    tokens = query_chunk(query_seed, device)
    logits, _ = forward_memory(model, tokens, clone_memory(memory), dtype)
    prediction = logits[:, -1, :VOCAB].argmax(-1)
    expected = old_target if chunks < REWRITE_CHUNK else new_target
    return {"chunks": chunks, "expected_version": "A" if chunks < REWRITE_CHUNK else "B",
            "expected_correctness": (prediction == expected).int().cpu().tolist(),
            "old_correctness": (prediction == old_target).int().cpu().tolist(),
            "new_correctness": (prediction == new_target).int().cpu().tolist(),
            "predictions": prediction.cpu().tolist(),
            "old_targets": old_target.cpu().tolist(), "new_targets": new_target.cpu().tolist()}


def l3_geometry(memory, first_anchor, rewrite_anchor):
    current = memory[2].float()
    first = first_anchor[2].float()
    first_cosine = F.cosine_similarity(current.flatten(1), first.flatten(1), dim=-1)
    row = {"l3_first_anchor_cosine_mean": float(first_cosine.mean().cpu()),
           "l3_first_anchor_cosine_by_sample": first_cosine.cpu().tolist()}
    if rewrite_anchor is not None:
        rewrite = rewrite_anchor[2].float()
        rewrite_cosine = F.cosine_similarity(current.flatten(1), rewrite.flatten(1), dim=-1)
        row["l3_rewrite_anchor_cosine_mean"] = float(rewrite_cosine.mean().cpu())
        row["l3_rewrite_anchor_cosine_by_sample"] = rewrite_cosine.cpu().tolist()
    else:
        row["l3_rewrite_anchor_cosine_mean"] = None
        row["l3_rewrite_anchor_cosine_by_sample"] = None
    return row


@torch.no_grad()
def run_replicate(model, condition: str, seed: int, replicate: int, folder: Path,
                  device: torch.device, dtype: torch.dtype, force: bool):
    final = folder / f"replicate_{replicate}.json"
    progress = folder / f"replicate_{replicate}_progress.pt"
    if final.exists() and not force:
        return json.loads(final.read_text(encoding="utf-8"))
    old_target = balanced_targets(replicate, device)
    new_target = (old_target + 7) % VOCAB
    memory = None; first_anchor = None; rewrite_anchor = None; rows = []; start_chunk = 0
    if progress.exists() and not force:
        state = torch.load(progress, map_location=device, weights_only=False)
        start_chunk = int(state["chunk"]); memory = state["memory"]
        first_anchor = state["first_anchor"]; rewrite_anchor = state["rewrite_anchor"]
        rows = state["rows"]
        print(f"resume condition={condition} seed={seed} replicate={replicate} "
              f"chunk={start_chunk}", flush=True)
    started = time.perf_counter()
    for chunk_index in range(start_chunk + 1, MILESTONES[-1] + 1):
        stream_seed = 782000000 + seed * 100000 + replicate * 2000 + chunk_index
        marked = chunk_index in (1, REWRITE_CHUNK)
        if chunk_index == 1:
            tokens = write_chunk(old_target, stream_seed, device)
            _, memory = forward_memory(model, tokens, None, dtype)
            first_anchor = clone_memory(memory)
        else:
            tokens = (write_chunk(new_target, stream_seed, device) if marked
                      else random_chunk(stream_seed, device))
            previous = memory
            _, candidate = forward_memory(model, tokens, memory, dtype)
            memory = route_l3(condition, previous, candidate, marked)
            if chunk_index == REWRITE_CHUNK:
                rewrite_anchor = clone_memory(memory)
        if chunk_index in MILESTONES:
            row = score(model, memory, old_target, new_target,
                        stream_seed + 900000, chunk_index, device, dtype)
            row.update(l3_geometry(memory, first_anchor, rewrite_anchor))
            row["condition"] = condition
            rows.append(row)
            atomic_torch_save(progress, {"chunk": chunk_index, "memory": memory,
                                         "first_anchor": first_anchor,
                                         "rewrite_anchor": rewrite_anchor, "rows": rows})
            accuracy = sum(row["expected_correctness"]) / BATCH
            print(f"condition={condition} seed={seed} replicate={replicate} "
                  f"chunks={chunk_index} expected={row['expected_version']} "
                  f"accuracy={accuracy:.2%}", flush=True)
    torch.cuda.synchronize()
    result = {"condition": condition, "seed": seed, "replicate": replicate,
              "samples": BATCH, "seconds": time.perf_counter() - started, "rows": rows}
    atomic_save(final, result)
    return result


def wilson(correct: int, samples: int, z: float = 1.959963984540054):
    p = correct / samples; scale = 1 + z*z/samples
    middle = (p + z*z/(2*samples))/scale
    half = z*math.sqrt(p*(1-p)/samples + z*z/(4*samples*samples))/scale
    return [middle-half, middle+half]


def summarize(runs):
    output = {}
    for condition in CONDITIONS:
        condition_rows = []
        for chunks in MILESTONES:
            selected = [row for run in runs if run["condition"] == condition
                        for row in run["rows"] if row["chunks"] == chunks]
            row = {"chunks": chunks, "expected_version": selected[0]["expected_version"]}
            for metric in ("expected", "old", "new"):
                values = [v for item in selected for v in item[f"{metric}_correctness"]]
                correct, samples = sum(values), len(values); interval = wilson(correct, samples)
                row[metric] = {"correct": correct, "samples": samples,
                               "accuracy": correct / samples, "wilson95": interval,
                               "above_chance": interval[0] > CHANCE}
            row["l3_first_anchor_cosine_mean"] = sum(
                item["l3_first_anchor_cosine_mean"] for item in selected) / len(selected)
            rewrite_values = [item["l3_rewrite_anchor_cosine_mean"] for item in selected
                              if item["l3_rewrite_anchor_cosine_mean"] is not None]
            row["l3_rewrite_anchor_cosine_mean"] = (
                sum(rewrite_values) / len(rewrite_values) if rewrite_values else None)
            condition_rows.append(row)
        output[condition] = condition_rows
    return output


def comparisons(runs):
    keyed = {(run["condition"], run["seed"], run["replicate"]): run for run in runs}
    rows = []
    for comparator in ("normal_update", "freeze_l3_after_first"):
        for chunks in MILESTONES:
            treatment, control = [], []
            for seed in SEEDS:
                for replicate in range(SAMPLES // BATCH):
                    adaptive = next(row for row in keyed[("oracle_marker_l3_gate", seed, replicate)]["rows"]
                                    if row["chunks"] == chunks)
                    baseline = next(row for row in keyed[(comparator, seed, replicate)]["rows"]
                                    if row["chunks"] == chunks)
                    treatment += adaptive["expected_correctness"]
                    control += baseline["expected_correctness"]
            result = paired_exact(treatment, control)
            result.update({"comparison": f"oracle_vs_{comparator}", "chunks": chunks,
                           "expected_version": "A" if chunks < REWRITE_CHUNK else "B",
                           "oracle_accuracy": sum(treatment) / len(treatment),
                           "comparator_accuracy": sum(control) / len(control)})
            rows.append(result)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level7_8_2/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "checkpoint": "level7_6_4/stage_4096.pt",
                "conditions": list(CONDITIONS), "seeds": list(SEEDS),
                "chunk_size": CHUNK_SIZE, "samples_per_seed": SAMPLES,
                "milestones": list(MILESTONES), "writes": {"A": 1, "B": REWRITE_CHUNK},
                "target_relation": "B=(A+7) mod 16",
                "oracle_gate": "L3 updates iff the current stream chunk contains marker token 17",
                "l1_l2": "always update", "identical_streams_and_queries": True,
                "primary": "before rewrite retrieve A; from chunk 512 retrieve B",
                "controls": {"normal_update": "L3 always updates",
                             "freeze_l3_after_first": "L3 never accepts B"}}
    if args.dry_run:
        print(json.dumps(protocol, indent=2)); return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol,
                "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "dtype": str(dtype)})
    runs = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            model = build(seed, device)
            folder = root / condition / f"seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
            for replicate in range(SAMPLES // BATCH):
                runs.append(run_replicate(model, condition, seed, replicate, folder,
                                          device, dtype, args.force))
                atomic_save(root / "runs.partial.json", runs)
            del model; torch.cuda.empty_cache()
    summary = summarize(runs); paired = comparisons(runs)
    result = {"protocol": protocol, "condition_summaries": summary,
              "paired_comparisons": paired, "runs": runs}
    atomic_save(root / "result.json", result)
    compact = [row for row in paired if row["chunks"] in (511, 512, 640, 1000)]
    print(json.dumps({"key_comparisons": compact}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
