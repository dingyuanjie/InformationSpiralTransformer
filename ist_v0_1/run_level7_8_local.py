"""Level 7.8: fixed-state Memory lifetime curve through 1000 chunks."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level7_1_local import atomic_save
from run_level7_4_1_local import atomic_torch_save
from run_level7_6_local import SEEDS


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
CHUNK_SIZE = 512
MILESTONES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)
SAMPLES = 32
BATCH = 16
VOCAB = 16
CHANCE = 1 / VOCAB


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, CHUNK_SIZE, "rope", True).to(device).eval()
    path = PARENT / f"ist-full_seed{seed}" / "stage_4096.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model


def balanced_targets(replicate: int, device: torch.device):
    # Each replicate is a full permutation of all 16 classes.
    offset = (replicate * 7) % VOCAB
    return (torch.arange(BATCH, device=device) + offset) % VOCAB


def random_chunk(seed: int, device: torch.device):
    set_seed(seed)
    return torch.randint(VOCAB, (BATCH, CHUNK_SIZE), device=device)


def write_chunk(target: torch.Tensor, seed: int, device: torch.device):
    tokens = random_chunk(seed, device)
    rows = torch.arange(BATCH, device=device)
    # Stagger positions so the result is not tied to one fixed location.
    positions = 32 + (rows * 29 + seed % 17) % (CHUNK_SIZE - 66)
    tokens[rows, positions] = 17
    tokens[rows, positions + 1] = target
    return tokens


def query_chunk(seed: int, device: torch.device):
    tokens = random_chunk(seed, device)
    tokens[:, -2] = 18
    tokens[:, -1] = 16
    return tokens


def forward_memory(model, tokens, memory, dtype):
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits, memory = model(tokens, memory=memory, return_memory=True,
                               detach_memory=True, per_layer_memory=True)
    return logits, memory


def clone_memory(memory):
    return [item.clone() for item in memory]


def memory_geometry(memory, anchor):
    rows = []
    for layer, (current, initial) in enumerate(zip(memory, anchor)):
        current_f = current.float(); initial_f = initial.float()
        normalized = F.normalize(current_f, dim=-1)
        cosine = normalized @ normalized.transpose(-1, -2)
        mask = ~torch.eye(current.size(1), device=current.device, dtype=torch.bool).unsqueeze(0)
        redundancy = cosine.abs().masked_select(mask.expand(current.size(0), -1, -1))
        drift = F.cosine_similarity(current_f.flatten(1), initial_f.flatten(1), dim=-1)
        rows.append({"layer": layer,
                     "mean_norm": float(current_f.norm(dim=-1).mean().cpu()),
                     "mean_abs_offdiag_cosine": float(redundancy.mean().cpu()),
                     "anchor_cosine_mean": float(drift.mean().cpu()),
                     "anchor_cosine_by_sample": drift.cpu().tolist()})
    return rows


@torch.no_grad()
def score_at_milestone(model, memory, anchor, early_target, seed: int,
                       milestone: int, device, dtype):
    query = query_chunk(seed + 900000 + milestone, device)
    early_logits, _ = forward_memory(model, query, clone_memory(memory), dtype)
    early_pred = early_logits[:, -1, :VOCAB].argmax(-1)

    late_target = (early_target + 7) % VOCAB
    late_tokens = write_chunk(late_target, seed + 800000 + milestone, device)
    _, late_memory = forward_memory(model, late_tokens, clone_memory(memory), dtype)
    late_logits, _ = forward_memory(model, query, late_memory, dtype)
    late_pred = late_logits[:, -1, :VOCAB].argmax(-1)

    reset_logits, _ = forward_memory(model, query, None, dtype)
    reset_pred = reset_logits[:, -1, :VOCAB].argmax(-1)
    return {"chunks": milestone, "total_stream_tokens": milestone * CHUNK_SIZE,
            "early_correctness": (early_pred == early_target).int().cpu().tolist(),
            "late_correctness": (late_pred == late_target).int().cpu().tolist(),
            "reset_correctness": (reset_pred == early_target).int().cpu().tolist(),
            "early_predictions": early_pred.cpu().tolist(),
            "late_predictions": late_pred.cpu().tolist(),
            "early_targets": early_target.cpu().tolist(),
            "late_targets": late_target.cpu().tolist(),
            "memory_geometry": memory_geometry(memory, anchor)}


@torch.no_grad()
def run_replicate(model, seed: int, replicate: int, folder: Path,
                  device: torch.device, dtype: torch.dtype, force: bool):
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
        print(f"resume seed={seed} replicate={replicate} chunk={start_chunk}", flush=True)
    started = time.perf_counter()
    for chunk_index in range(start_chunk + 1, MILESTONES[-1] + 1):
        stream_seed = 780000000 + seed * 100000 + replicate * 2000 + chunk_index
        tokens = (write_chunk(target, stream_seed, device) if chunk_index == 1
                  else random_chunk(stream_seed, device))
        _, memory = forward_memory(model, tokens, memory, dtype)
        if chunk_index == 1 and anchor is None:
            anchor = clone_memory(memory)
        if chunk_index in MILESTONES:
            row = score_at_milestone(model, memory, anchor, target,
                                     stream_seed, chunk_index, device, dtype)
            rows.append(row)
            atomic_torch_save(progress, {"chunk": chunk_index, "memory": memory,
                                         "anchor": anchor, "rows": rows})
            early = sum(row["early_correctness"]) / BATCH
            late = sum(row["late_correctness"]) / BATCH
            print(f"seed={seed} replicate={replicate} chunks={chunk_index} "
                  f"early={early:.2%} late={late:.2%}", flush=True)
    torch.cuda.synchronize()
    result = {"seed": seed, "replicate": replicate, "samples": BATCH,
              "seconds": time.perf_counter() - started, "rows": rows}
    atomic_save(final, result)
    return result


def wilson(correct: int, samples: int, z: float = 1.959963984540054):
    p = correct / samples; scale = 1 + z*z/samples
    middle = (p + z*z/(2*samples))/scale
    half = z*math.sqrt(p*(1-p)/samples + z*z/(4*samples*samples))/scale
    return [middle-half, middle+half]


def summarize(runs):
    summary = []
    for chunks in MILESTONES:
        selected = [row for run in runs for row in run["rows"] if row["chunks"] == chunks]
        output = {"chunks": chunks, "total_stream_tokens": chunks * CHUNK_SIZE}
        for condition in ("early", "late", "reset"):
            values = [v for row in selected for v in row[f"{condition}_correctness"]]
            correct, samples = sum(values), len(values)
            output[condition] = {"correct": correct, "samples": samples,
                                 "accuracy": correct / samples,
                                 "wilson95": wilson(correct, samples),
                                 "above_chance": wilson(correct, samples)[0] > CHANCE}
        output["early_minus_late"] = output["early"]["accuracy"] - output["late"]["accuracy"]
        output["mean_anchor_cosine_by_layer"] = [
            sum(row["memory_geometry"][layer]["anchor_cosine_mean"] for row in selected) / len(selected)
            for layer in range(3)]
        summary.append(output)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level7_8/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "checkpoint": "level7_6_4/stage_4096.pt",
                "seeds": list(SEEDS), "chunk_size": CHUNK_SIZE,
                "milestones": list(MILESTONES), "samples_per_seed": SAMPLES,
                "balanced_targets": True, "single_write": "chunk 1 only",
                "conditions": {"early": "query the chunk-1 target",
                               "late": "write a different target immediately before query",
                               "reset": "query with no incoming Memory"},
                "streaming": "one forward trajectory; queries do not mutate the main trajectory",
                "primary": "early-write accuracy and Wilson interval over chunk lifetime",
                "diagnostics": "late-write viability, reset prior, per-layer anchor cosine and redundancy"}
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
    replicates = SAMPLES // BATCH
    for seed in SEEDS:
        model = build(seed, device)
        folder = root / f"seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        for replicate in range(replicates):
            runs.append(run_replicate(model, seed, replicate, folder, device, dtype, args.force))
            atomic_save(root / "runs.partial.json", runs)
        del model; torch.cuda.empty_cache()
    summary = summarize(runs)
    result = {"protocol": protocol, "summary": summary, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
