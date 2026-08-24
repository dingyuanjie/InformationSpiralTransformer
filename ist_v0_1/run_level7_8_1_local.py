"""Level 7.8.1: causal confirmation of cross-chunk Memory overwrite."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_level7_1_local import atomic_save
from run_level7_4_1_local import atomic_torch_save
from run_level7_6_local import SEEDS
from run_level7_7_local import paired_exact
from run_level7_8_local import (BATCH, CHANCE, CHUNK_SIZE, MILESTONES, SAMPLES,
                                balanced_targets, build, clone_memory,
                                forward_memory, random_chunk, score_at_milestone,
                                summarize, write_chunk)


ROOT = Path(__file__).resolve().parent
LOCKED = ROOT / "experiments/level7_8/formal/result.json"
CONDITIONS = {
    "freeze_all": (0, 1, 2),
    "freeze_l1": (0,),
    "freeze_l2": (1,),
    "freeze_l3": (2,),
}


def retain_frozen(previous, candidate, frozen_layers):
    return [previous[layer] if layer in frozen_layers else candidate[layer]
            for layer in range(len(candidate))]


@torch.no_grad()
def run_replicate(model, condition: str, frozen_layers: tuple[int, ...], seed: int,
                  replicate: int, folder: Path, device: torch.device,
                  dtype: torch.dtype, force: bool):
    final = folder / f"replicate_{replicate}.json"
    progress = folder / f"replicate_{replicate}_progress.pt"
    if final.exists() and not force:
        return json.loads(final.read_text(encoding="utf-8"))
    target = balanced_targets(replicate, device)
    memory = None; anchor = None; rows = []; start_chunk = 0
    if progress.exists() and not force:
        state = torch.load(progress, map_location=device, weights_only=False)
        start_chunk = int(state["chunk"]); memory = state["memory"]
        anchor = state["anchor"]; rows = state["rows"]
        print(f"resume condition={condition} seed={seed} replicate={replicate} "
              f"chunk={start_chunk}", flush=True)
    started = time.perf_counter()
    for chunk_index in range(start_chunk + 1, MILESTONES[-1] + 1):
        stream_seed = 780000000 + seed * 100000 + replicate * 2000 + chunk_index
        tokens = (write_chunk(target, stream_seed, device) if chunk_index == 1
                  else random_chunk(stream_seed, device))
        if chunk_index == 1:
            _, memory = forward_memory(model, tokens, None, dtype)
            if anchor is None:
                anchor = clone_memory(memory)
        else:
            previous = memory
            _, candidate = forward_memory(model, tokens, memory, dtype)
            memory = retain_frozen(previous, candidate, frozen_layers)
        if chunk_index in MILESTONES:
            row = score_at_milestone(model, memory, anchor, target, stream_seed,
                                     chunk_index, device, dtype)
            row["condition"] = condition
            rows.append(row)
            atomic_torch_save(progress, {"chunk": chunk_index, "memory": memory,
                                         "anchor": anchor, "rows": rows})
            early = sum(row["early_correctness"]) / BATCH
            print(f"condition={condition} seed={seed} replicate={replicate} "
                  f"chunks={chunk_index} early={early:.2%}", flush=True)
    torch.cuda.synchronize()
    result = {"condition": condition, "frozen_layers": list(frozen_layers),
              "seed": seed, "replicate": replicate, "samples": BATCH,
              "seconds": time.perf_counter() - started, "rows": rows}
    atomic_save(final, result)
    return result


def paired_comparisons(new_runs, locked_runs):
    output = []
    normal = {(run["seed"], run["replicate"]): run for run in locked_runs}
    for condition in CONDITIONS:
        selected = [run for run in new_runs if run["condition"] == condition]
        for chunks in MILESTONES:
            treatment, control = [], []
            for run in selected:
                key = (run["seed"], run["replicate"])
                new_row = next(row for row in run["rows"] if row["chunks"] == chunks)
                old_row = next(row for row in normal[key]["rows"] if row["chunks"] == chunks)
                treatment += new_row["early_correctness"]
                control += old_row["early_correctness"]
            row = paired_exact(treatment, control)
            row.update({"condition": condition, "chunks": chunks,
                        "treatment_accuracy": sum(treatment) / len(treatment),
                        "normal_accuracy": sum(control) / len(control)})
            output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level7_8_1/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "checkpoint": "level7_6_4/stage_4096.pt",
                "locked_normal": "level7_8/formal/result.json",
                "conditions": {name: list(layers) for name, layers in CONDITIONS.items()},
                "intervention": "after chunk 1, run every filler forward but discard updates from frozen layers",
                "seeds": list(SEEDS), "chunk_size": CHUNK_SIZE,
                "milestones": list(MILESTONES), "samples_per_seed": SAMPLES,
                "identical_streams_and_queries": True,
                "primary": "paired early-write accuracy versus normal update",
                "interpretation": "recovery under freezing causally identifies overwrite; layer freezes localize it"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2)); return 0
    if not LOCKED.exists():
        raise FileNotFoundError(f"Locked Level 7.8 result missing: {LOCKED}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    locked = json.loads(LOCKED.read_text(encoding="utf-8"))
    if locked["protocol"]["milestones"] != list(MILESTONES):
        raise ValueError("Level 7.8 milestone protocol mismatch")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol,
                "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "dtype": str(dtype)})
    runs = []
    for condition, frozen_layers in CONDITIONS.items():
        for seed in SEEDS:
            model = build(seed, device)
            folder = root / condition / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            for replicate in range(SAMPLES // BATCH):
                runs.append(run_replicate(model, condition, frozen_layers, seed,
                                          replicate, folder, device, dtype, args.force))
                atomic_save(root / "runs.partial.json", runs)
            del model; torch.cuda.empty_cache()
    condition_summaries = {condition: summarize([run for run in runs
                                                  if run["condition"] == condition])
                           for condition in CONDITIONS}
    comparisons = paired_comparisons(runs, locked["runs"])
    result = {"protocol": protocol, "locked_normal_summary": locked["summary"],
              "condition_summaries": condition_summaries,
              "paired_vs_normal": comparisons, "runs": runs}
    atomic_save(root / "result.json", result)
    key_chunks = {32, 128, 512, 1000}
    compact = [{"condition": row["condition"], "chunks": row["chunks"],
                "accuracy": row["treatment_accuracy"], "normal": row["normal_accuracy"],
                "difference": row["difference"], "p": row["mcnemar_exact_p"]}
               for row in comparisons if row["chunks"] in key_chunks]
    print(json.dumps({"key_comparisons": compact}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
