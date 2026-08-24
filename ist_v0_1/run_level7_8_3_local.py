"""Level 7.8.3: repeated selective L3 rewrites across a 1000-chunk stream."""
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
from run_level7_8_2_local import CONDITIONS, route_l3


ROOT = Path(__file__).resolve().parent
WRITE_CHUNKS = tuple([1] + list(range(100, 1000, 100)))
MILESTONES = tuple([1] + [value for write in WRITE_CHUNKS[1:]
                           for value in (write - 1, write)] + [1000])
VOCAB = 16


def target_for_version(base: torch.Tensor, version: int):
    return (base + 7 * version) % VOCAB


def version_at(chunks: int):
    return max(index for index, write in enumerate(WRITE_CHUNKS) if write <= chunks)


def score(model, memory, base_target, query_seed, chunks, device, dtype):
    logits, _ = forward_memory(model, query_chunk(query_seed, device),
                               clone_memory(memory), dtype)
    prediction = logits[:, -1, :VOCAB].argmax(-1)
    version = version_at(chunks)
    expected = target_for_version(base_target, version)
    correctness_by_version = []
    for candidate in range(len(WRITE_CHUNKS)):
        target = target_for_version(base_target, candidate)
        correctness_by_version.append((prediction == target).int().cpu().tolist())
    return {"chunks": chunks, "version": version, "write_chunk": WRITE_CHUNKS[version],
            "phase": "immediate" if chunks == WRITE_CHUNKS[version] else "retention",
            "expected_correctness": (prediction == expected).int().cpu().tolist(),
            "correctness_by_version": correctness_by_version,
            "predictions": prediction.cpu().tolist(), "expected_targets": expected.cpu().tolist()}


def geometry(memory, first_anchor, latest_anchor):
    current = memory[2].float()
    first = F.cosine_similarity(current.flatten(1), first_anchor[2].float().flatten(1), dim=-1)
    latest = F.cosine_similarity(current.flatten(1), latest_anchor[2].float().flatten(1), dim=-1)
    return {"l3_first_anchor_cosine_mean": float(first.mean().cpu()),
            "l3_latest_anchor_cosine_mean": float(latest.mean().cpu())}


@torch.no_grad()
def run_replicate(model, condition: str, seed: int, replicate: int, folder: Path,
                  device: torch.device, dtype: torch.dtype, force: bool):
    final = folder / f"replicate_{replicate}.json"
    progress = folder / f"replicate_{replicate}_progress.pt"
    if final.exists() and not force:
        return json.loads(final.read_text(encoding="utf-8"))
    base_target = balanced_targets(replicate, device)
    memory = None; first_anchor = None; latest_anchor = None; rows = []; start_chunk = 0
    if progress.exists() and not force:
        state = torch.load(progress, map_location=device, weights_only=False)
        start_chunk = int(state["chunk"]); memory = state["memory"]
        first_anchor = state["first_anchor"]; latest_anchor = state["latest_anchor"]
        rows = state["rows"]
        print(f"resume condition={condition} seed={seed} replicate={replicate} "
              f"chunk={start_chunk}", flush=True)
    started = time.perf_counter()
    for chunk_index in range(start_chunk + 1, MILESTONES[-1] + 1):
        stream_seed = 783000000 + seed * 100000 + replicate * 2000 + chunk_index
        marked = chunk_index in WRITE_CHUNKS
        if marked:
            version = WRITE_CHUNKS.index(chunk_index)
            tokens = write_chunk(target_for_version(base_target, version), stream_seed, device)
        else:
            tokens = random_chunk(stream_seed, device)
        if chunk_index == 1:
            _, memory = forward_memory(model, tokens, None, dtype)
            first_anchor = clone_memory(memory); latest_anchor = clone_memory(memory)
        else:
            previous = memory
            _, candidate = forward_memory(model, tokens, memory, dtype)
            memory = route_l3(condition, previous, candidate, marked)
            if marked:
                latest_anchor = clone_memory(memory)
        if chunk_index in MILESTONES:
            row = score(model, memory, base_target, stream_seed + 900000,
                        chunk_index, device, dtype)
            row.update(geometry(memory, first_anchor, latest_anchor)); row["condition"] = condition
            rows.append(row)
            atomic_torch_save(progress, {"chunk": chunk_index, "memory": memory,
                                         "first_anchor": first_anchor,
                                         "latest_anchor": latest_anchor, "rows": rows})
            accuracy = sum(row["expected_correctness"]) / BATCH
            print(f"condition={condition} seed={seed} replicate={replicate} "
                  f"chunks={chunk_index} version={row['version']} phase={row['phase']} "
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
        rows = []
        for chunks in MILESTONES:
            selected = [row for run in runs if run["condition"] == condition
                        for row in run["rows"] if row["chunks"] == chunks]
            values = [value for row in selected for value in row["expected_correctness"]]
            correct, samples = sum(values), len(values); interval = wilson(correct, samples)
            rows.append({"chunks": chunks, "version": selected[0]["version"],
                         "phase": selected[0]["phase"], "correct": correct, "samples": samples,
                         "accuracy": correct / samples, "wilson95": interval,
                         "above_chance": interval[0] > CHANCE,
                         "l3_first_anchor_cosine_mean": sum(
                             row["l3_first_anchor_cosine_mean"] for row in selected) / len(selected),
                         "l3_latest_anchor_cosine_mean": sum(
                             row["l3_latest_anchor_cosine_mean"] for row in selected) / len(selected)})
        output[condition] = rows
    return output


def paired(runs):
    keyed = {(run["condition"], run["seed"], run["replicate"]): run for run in runs}
    output = []
    for comparator in ("normal_update", "freeze_l3_after_first"):
        for chunks in MILESTONES:
            treatment, control = [], []
            for seed in SEEDS:
                for replicate in range(SAMPLES // BATCH):
                    for condition, destination in (("oracle_marker_l3_gate", treatment),
                                                   (comparator, control)):
                        row = next(item for item in keyed[(condition, seed, replicate)]["rows"]
                                   if item["chunks"] == chunks)
                        destination += row["expected_correctness"]
            result = paired_exact(treatment, control)
            result.update({"comparison": f"oracle_vs_{comparator}", "chunks": chunks,
                           "version": version_at(chunks),
                           "phase": "immediate" if chunks in WRITE_CHUNKS else "retention",
                           "oracle_accuracy": sum(treatment) / len(treatment),
                           "comparator_accuracy": sum(control) / len(control)})
            output.append(result)
    return output


def endurance(summary):
    rows = summary["oracle_marker_l3_gate"]
    immediate = [row["accuracy"] for row in rows if row["phase"] == "immediate"]
    retained = [row["accuracy"] for row in rows if row["phase"] == "retention"]
    return {"immediate_by_version": immediate, "retention_by_version": retained,
            "first_half_immediate_mean": sum(immediate[:5]) / 5,
            "second_half_immediate_mean": sum(immediate[5:]) / 5,
            "first_half_retention_mean": sum(retained[:5]) / 5,
            "second_half_retention_mean": sum(retained[5:]) / len(retained[5:])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level7_8_3/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "checkpoint": "level7_6_4/stage_4096.pt",
                "conditions": list(CONDITIONS), "seeds": list(SEEDS),
                "chunk_size": CHUNK_SIZE, "samples_per_seed": SAMPLES,
                "write_chunks": list(WRITE_CHUNKS), "milestones": list(MILESTONES),
                "target_schedule": "target_v=(base+7*v) mod 16; first 10 versions are distinct",
                "query": "retrieve the most recently marked target",
                "oracle_gate": "L3 updates only on the 10 marked write chunks; L1/L2 always update",
                "primary": "immediate acquisition and pre-next-write retention across rewrite ordinal",
                "controls": ["normal L3 update", "L3 frozen after first write"],
                "identical_streams_and_queries": True}
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
    summary = summarize(runs); comparisons = paired(runs); endurance_row = endurance(summary)
    result = {"protocol": protocol, "condition_summaries": summary,
              "oracle_endurance": endurance_row, "paired_comparisons": comparisons, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"oracle_endurance": endurance_row,
                      "oracle_summary": summary["oracle_marker_l3_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
