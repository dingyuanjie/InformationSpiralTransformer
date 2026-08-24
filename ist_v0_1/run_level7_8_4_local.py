"""Level 7.8.4: keyed multi-fact capacity under selective L3 writes."""
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
from run_level7_7_local import paired_exact


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
SEEDS = (2026, 7)  # Preregistered successful persistent-Memory group.
LOADS = (1, 2, 4, 8, 16)
CHUNK_SIZE = 128
FILLER_CHUNKS = 16
TRAIN_STEPS = {1: 200, 2: 200, 4: 200, 8: 200, 16: 200}
TRAIN_BATCH = {1: 8, 2: 8, 4: 4, 8: 2, 16: 1}
EVAL_SAMPLES = 64
EVAL_BATCH = 8
CONDITIONS = ("oracle_marker_l3_gate", "normal_update", "freeze_l3_after_first")
VOCAB = 16
CHANCE = 1 / VOCAB


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, CHUNK_SIZE, "rope", True).to(device)
    path = PARENT / f"ist-full_seed{seed}" / "stage_4096.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model


def make_values(batch: int, load: int, device: torch.device):
    return torch.stack([torch.randperm(VOCAB, device=device)[:load] for _ in range(batch)])


def base_chunk(batch: int, device: torch.device):
    return torch.randint(VOCAB, (batch, CHUNK_SIZE), device=device)


def write_tokens(keys: torch.Tensor, values: torch.Tensor, write_index: int,
                 device: torch.device):
    batch = values.size(0); tokens = base_chunk(batch, device)
    rows = torch.arange(batch, device=device)
    positions = 8 + (rows * 13 + write_index * 17) % (CHUNK_SIZE - 20)
    tokens[rows, positions] = 17
    tokens[rows, positions + 1] = keys
    tokens[rows, positions + 2] = values
    return tokens, positions


def query_tokens(query_keys: torch.Tensor, device: torch.device):
    tokens = base_chunk(len(query_keys), device)
    tokens[:, -3] = 18
    tokens[:, -2] = query_keys
    tokens[:, -1] = 16
    return tokens


def route_memory(condition: str, previous, candidate, marked: bool, first_write: bool):
    if previous is None or condition == "normal_update":
        return candidate
    routed = list(candidate)
    if condition == "freeze_l3_after_first" or (condition == "oracle_marker_l3_gate" and not marked):
        routed[2] = previous[2]
    return routed


def forward_one(model, tokens, memory, dtype, detach: bool):
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits, candidate = model(tokens, memory=memory, return_memory=True,
                                  detach_memory=detach, per_layer_memory=True)
    return logits, candidate


def run_stream(model, load: int, batch: int, condition: str, device: torch.device,
               dtype: torch.dtype, detach: bool, query_offset: int = 0):
    keys = torch.arange(load, device=device)
    values = make_values(batch, load, device)
    query_indices = (torch.arange(batch, device=device) + query_offset) % load
    query_keys = keys[query_indices]
    rows = torch.arange(batch, device=device)
    targets = values[rows, query_indices]
    memory = None; local_losses = []
    for write_index in range(load):
        tokens, positions = write_tokens(keys[write_index].expand(batch),
                                         values[:, write_index], write_index, device)
        logits, candidate = forward_one(model, tokens, memory, dtype, detach)
        memory = route_memory(condition, memory, candidate, True, write_index == 0)
        local_losses.append(F.cross_entropy(logits[rows, positions, :VOCAB], values[:, write_index]))
    for _ in range(FILLER_CHUNKS):
        tokens = base_chunk(batch, device)
        _, candidate = forward_one(model, tokens, memory, dtype, detach)
        memory = route_memory(condition, memory, candidate, False, False)
    logits, _ = forward_one(model, query_tokens(query_keys, device), memory, dtype, detach)
    return logits[:, -1, :VOCAB], targets, torch.stack(local_losses).mean()


def checkpoint(path: Path, model, optimizer, load: int, step: int, history: list):
    atomic_torch_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "load": load, "step": step, "history": history})


def train_seed(model, seed: int, folder: Path, device: torch.device,
               dtype: torch.dtype, force: bool):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    history = []
    for load in LOADS:
        final = folder / f"stage_load{load}.pt"
        resume = folder / f"stage_load{load}_resume.pt"
        if final.exists() and not force:
            state = torch.load(final, map_location=device, weights_only=False)
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
            print(f"seed={seed} load={load} already complete", flush=True)
            continue
        start = 0
        candidates = [p for p in (resume, resume.with_suffix(".pt.tmp")) if p.exists()]
        if candidates and not force:
            loaded = [(torch.load(p, map_location=device, weights_only=False), p) for p in candidates]
            state, selected = max(loaded, key=lambda item: int(item[0]["step"]))
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start = int(state["step"]); history = state["history"]
            print(f"resume seed={seed} load={load} step={start} source={selected.name}", flush=True)
        for step in range(start + 1, TRAIN_STEPS[load] + 1):
            set_seed(784000000 + seed * 10000 + load * 1000 + step)
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, targets, local_loss = run_stream(
                model, load, TRAIN_BATCH[load], "oracle_marker_l3_gate",
                device, dtype, detach=False, query_offset=step)
            query_loss = F.cross_entropy(logits, targets)
            loss = query_loss + 0.2 * local_loss + 0.1 * model.memory_diversity_loss()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step == 1 or step % 25 == 0:
                row = {"load": load, "step": step, "loss": float(loss.detach()),
                       "query_loss": float(query_loss.detach()),
                       "local_loss": float(local_loss.detach()),
                       "train_accuracy": float((logits.argmax(-1) == targets).float().mean().detach())}
                history.append(row); checkpoint(resume, model, optimizer, load, step, history)
                print(f"seed={seed} load={load} step={step} loss={row['loss']:.4f} "
                      f"accuracy={row['train_accuracy']:.2%}", flush=True)
        checkpoint(final, model, optimizer, load, TRAIN_STEPS[load], history)
    return history


@torch.no_grad()
def evaluate(model, seed: int, load: int, condition: str,
             device: torch.device, dtype: torch.dtype):
    model.eval(); correctness = []; predictions = []; targets_out = []
    started = time.perf_counter()
    for start in range(0, EVAL_SAMPLES, EVAL_BATCH):
        set_seed(784900000 + seed * 10000 + load * 100 + start)
        logits, targets, _ = run_stream(model, load, EVAL_BATCH, condition,
                                         device, dtype, detach=True, query_offset=start)
        predicted = logits.argmax(-1)
        correctness += (predicted == targets).int().cpu().tolist()
        predictions += predicted.cpu().tolist(); targets_out += targets.cpu().tolist()
    torch.cuda.synchronize()
    return {"seed": seed, "load": load, "condition": condition,
            "samples": EVAL_SAMPLES, "correct": sum(correctness),
            "accuracy": sum(correctness) / EVAL_SAMPLES,
            "correctness": correctness, "predictions": predictions,
            "targets": targets_out, "seconds": time.perf_counter() - started}


def wilson(correct: int, samples: int, z: float = 1.959963984540054):
    p = correct / samples; scale = 1 + z*z/samples
    middle = (p + z*z/(2*samples))/scale
    half = z*math.sqrt(p*(1-p)/samples + z*z/(4*samples*samples))/scale
    return [middle-half, middle+half]


def summarize(runs):
    rows = []
    for condition in CONDITIONS:
        for load in LOADS:
            selected = [run for run in runs if run["condition"] == condition and run["load"] == load]
            correct = sum(run["correct"] for run in selected); samples = sum(run["samples"] for run in selected)
            interval = wilson(correct, samples)
            rows.append({"condition": condition, "load": load, "correct": correct,
                         "samples": samples, "accuracy": correct / samples,
                         "wilson95": interval, "above_chance": interval[0] > CHANCE,
                         "seed_accuracies": {str(run["seed"]): run["accuracy"] for run in selected}})
    return rows


def comparisons(runs):
    keyed = {(run["condition"], run["seed"], run["load"]): run for run in runs}
    rows = []
    for comparator in ("normal_update", "freeze_l3_after_first"):
        for load in LOADS:
            treatment, control = [], []
            for seed in SEEDS:
                treatment += keyed[("oracle_marker_l3_gate", seed, load)]["correctness"]
                control += keyed[(comparator, seed, load)]["correctness"]
            result = paired_exact(treatment, control)
            result.update({"comparison": f"oracle_vs_{comparator}", "load": load,
                           "oracle_accuracy": sum(treatment) / len(treatment),
                           "comparator_accuracy": sum(control) / len(control)})
            rows.append(result)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level7_8_4/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full keyed finetune", "source": "level7_6_4/stage_4096.pt",
                "seeds": list(SEEDS), "selection": "preregistered successful persistent-Memory group",
                "chunk_size": CHUNK_SIZE, "loads": list(LOADS), "filler_chunks": FILLER_CHUNKS,
                "write_format": "[17,key,value]", "query_format": "[18,key,16]",
                "train_steps": TRAIN_STEPS, "train_batch": TRAIN_BATCH, "learning_rate": 5e-4,
                "training_gate": "oracle marker L3 gate", "conditions": list(CONDITIONS),
                "eval_samples_per_seed_load_condition": EVAL_SAMPLES,
                "primary": "maximum keyed load whose pooled Wilson lower bound exceeds 1/16 chance",
                "scope": "latest checkpoint is finetuned; this is not zero-shot capacity"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2)); return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol,
                "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "dtype": str(dtype)})
    runs = []; training = []
    for seed in SEEDS:
        folder = root / f"seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
        model = build(seed, device)
        history = train_seed(model, seed, folder, device, dtype, args.force)
        training.append({"seed": seed, "history": history})
        for condition in CONDITIONS:
            for load in LOADS:
                output = folder / f"eval_{condition}_load{load}.json"
                if output.exists() and not args.force:
                    row = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row = evaluate(model, seed, load, condition, device, dtype)
                    atomic_save(output, row)
                runs.append(row)
                print(f"seed={seed} condition={condition} load={load} "
                      f"accuracy={row['accuracy']:.2%}", flush=True)
                atomic_save(root / "runs.partial.json", runs)
        del model; torch.cuda.empty_cache()
    summary = summarize(runs); paired = comparisons(runs)
    oracle_rows = [row for row in summary if row["condition"] == "oracle_marker_l3_gate"]
    max_supported = max((row["load"] for row in oracle_rows if row["above_chance"]), default=0)
    result = {"protocol": protocol, "maximum_supported_load": max_supported,
              "summary": summary, "paired_comparisons": paired,
              "training": training, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"maximum_supported_load": max_supported,
                      "oracle_capacity_curve": oracle_rows, "paired": paired}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
