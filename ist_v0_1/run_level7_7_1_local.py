"""Level 7.7.1: topology-aware adaptive Memory-bank dropout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from long_context_test import set_seed
from marked_retrieval_level2 import make_batch
from run_level7_1_local import atomic_save
from run_level7_4_1_local import atomic_torch_save
from run_level7_6_local import SEEDS
from run_level7_7_local import (CHANCE, EVAL_LENGTH, TRAIN_LENGTH, WINDOWS,
                                build, evaluate, paired_exact, wilson)


ROOT = Path(__file__).resolve().parent
LEVEL77 = ROOT / "experiments/level7_7/formal/result.json"
BRANCH = "adaptive_bankdrop_k8_p50"
TRAIN_STEPS = 200
EVAL_SAMPLES = 100
# Locked before Level 7.7.1 from Level 7.6.6.1 near/far high-vs-low
# slot_abs_cosine_offdiag midpoints. These are not fitted on Level 7.7 outcomes.
REDUNDANCY_THRESHOLDS = (0.189, 0.165, 0.169)


def configure(model) -> None:
    if len(model.blocks) != len(REDUNDANCY_THRESHOLDS):
        raise ValueError("Level 7.7.1 protocol requires exactly three Memory layers")
    for block, threshold in zip(model.blocks, REDUNDANCY_THRESHOLDS):
        block.memory_bank_dropout_probability = 0.5
        block.memory_bank_dropout_size = 8
        block.memory_bank_dropout_redundancy_threshold = threshold


def save_training(path: Path, model, optimizer, step: int, history: list,
                  topology: dict) -> None:
    atomic_torch_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "step": step, "history": history, "topology": topology})


def train(model, optimizer, seed: int, folder: Path, device: torch.device,
          dtype: torch.dtype, force: bool) -> tuple[list, dict]:
    final, resume = folder / "trained.pt", folder / "resume.pt"
    if final.exists() and not force:
        state = torch.load(final, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        return state["history"], state["topology"]
    start, history = 0, []
    topology = {str(layer): {"observations": 0, "eligible": 0, "activated": 0,
                             "redundancy_sum": 0.0}
                for layer in range(len(REDUNDANCY_THRESHOLDS))}
    candidates = [p for p in (resume, resume.with_suffix(".pt.tmp")) if p.exists()]
    if candidates and not force:
        loaded = [(torch.load(p, map_location=device, weights_only=False), p) for p in candidates]
        state, selected = max(loaded, key=lambda item: int(item[0]["step"]))
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start, history = int(state["step"]), state["history"]
        topology = state.get("topology", topology)
        print(f"resume branch={BRANCH} seed={seed} step={start} source={selected.name}", flush=True)
    configure(model)
    for step in range(start + 1, TRAIN_STEPS + 1):
        # Same schedule as Level 7.7: adaptive and locked controls see identical examples.
        set_seed(770000000 + seed * 1000 + step)
        model.train()
        tokens, target, position = make_batch(1, TRAIN_LENGTH, TRAIN_LENGTH - 3, 16, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(tokens)[..., :16]
            query_loss = F.cross_entropy(logits[:, -1], target)
            local_loss = F.cross_entropy(logits[torch.arange(len(target), device=device), position], target)
            loss = query_loss + 0.5 * local_loss + 0.1 * model.memory_diversity_loss()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        for layer, block in enumerate(model.blocks):
            row = topology[str(layer)]
            redundancy = block.last_training_bank_dropout_redundancy
            eligible = block.last_training_bank_dropout_eligible
            mask = block.last_training_bank_dropout_mask
            row["observations"] += int(redundancy.numel())
            row["redundancy_sum"] += float(redundancy.float().sum().cpu())
            row["eligible"] += int(eligible.sum().cpu())
            row["activated"] += int(mask.any(dim=1).sum().cpu())
        if step == 1 or step % 25 == 0:
            row = {"step": step, "loss": float(loss.detach()),
                   "query_loss": float(query_loss.detach()), "local_loss": float(local_loss.detach())}
            history.append(row); save_training(resume, model, optimizer, step, history, topology)
            rates = [topology[str(i)]["activated"] / topology[str(i)]["observations"]
                     for i in range(len(model.blocks))]
            print(f"branch={BRANCH} seed={seed} step={step} loss={row['loss']:.4f} "
                  f"activation={','.join(f'{rate:.1%}' for rate in rates)}", flush=True)
    save_training(final, model, optimizer, TRAIN_STEPS, history, topology)
    return history, topology


def compact_topology(topology: dict) -> list[dict]:
    rows = []
    for layer, threshold in enumerate(REDUNDANCY_THRESHOLDS):
        source = topology[str(layer)]; n = source["observations"]
        rows.append({"layer": layer, "threshold": threshold,
                     "mean_redundancy": source["redundancy_sum"] / n,
                     "eligible_rate": source["eligible"] / n,
                     "activation_rate": source["activated"] / n})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level7_7_1/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"branch": BRANCH, "seeds": list(SEEDS),
                "source": "level7_6_4/formal/ist-full_seed*/stage_4096.pt",
                "locked_comparators": "level7_7/formal/result.json",
                "dropout": {"size": 8, "conditional_probability": 0.5,
                            "metric": "mean absolute off-diagonal slot cosine",
                            "layer_thresholds": list(REDUNDANCY_THRESHOLDS)},
                "train": {"length": TRAIN_LENGTH, "steps": TRAIN_STEPS, "batch": 1,
                          "identical_examples_to_level7_7": True},
                "eval": {"length": EVAL_LENGTH, "windows": WINDOWS,
                         "samples_per_seed_window": EVAL_SAMPLES,
                         "identical_examples_to_level7_7": True, "dropout_disabled": True},
                "primary": "adaptive vs control paired accuracy and successful-seed count",
                "secondary": "adaptive vs fixed bankdrop and per-layer activation rates"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2)); return 0
    if not LEVEL77.exists():
        raise FileNotFoundError(f"Locked Level 7.7 comparator missing: {LEVEL77}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    locked = json.loads(LEVEL77.read_text(encoding="utf-8"))
    if locked["protocol"]["seeds"] != list(SEEDS):
        raise ValueError("Level 7.7 seed protocol does not match Level 7.7.1")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol,
                "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "dtype": str(dtype)})
    runs = []
    for seed in SEEDS:
        folder = root / f"{BRANCH}_seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
        model, optimizer = build(seed, device)
        history, topology = train(model, optimizer, seed, folder, device, dtype, args.force)
        tests = []
        for window_name in WINDOWS:
            output = folder / f"eval_{window_name}.json"
            if output.exists() and not args.force:
                row = json.loads(output.read_text(encoding="utf-8"))
            else:
                row = evaluate(model, BRANCH, seed, window_name, device, dtype)
                atomic_save(output, row)
            tests.append(row)
            print(f"branch={BRANCH} seed={seed} window={window_name} accuracy={row['accuracy']:.2%}", flush=True)
        runs.append({"branch": BRANCH, "seed": seed, "history": history,
                     "topology": compact_topology(topology), "tests": tests})
        atomic_save(root / "runs.partial.json", runs)
        del model, optimizer; torch.cuda.empty_cache()
    seed_results = []
    for run in runs:
        correct = sum(x["correct"] for x in run["tests"]); samples = sum(x["samples"] for x in run["tests"])
        interval = wilson(correct, samples)
        seed_results.append({"seed": run["seed"], "correct": correct, "samples": samples,
                             "accuracy": correct / samples, "wilson95": interval,
                             "above_chance": interval[0] > CHANCE})
    adaptive_summary = {"branch": BRANCH, "seed_results": seed_results,
                        "successful_seed_count": sum(x["above_chance"] for x in seed_results),
                        "mean_seed_accuracy": sum(x["accuracy"] for x in seed_results) / len(seed_results)}
    locked_runs = {(r["branch"], r["seed"]): r for r in locked["runs"]}
    comparisons = {}
    adaptive = {r["seed"]: [v for t in r["tests"] for v in t["correctness"]] for r in runs}
    for comparator in ("control_continue", "bankdrop_k8_p50"):
        treatment, control = [], []
        for seed in SEEDS:
            treatment += adaptive[seed]
            control += [v for t in locked_runs[(comparator, seed)]["tests"] for v in t["correctness"]]
        comparisons[f"adaptive_vs_{comparator}"] = paired_exact(treatment, control)
    result = {"protocol": protocol, "adaptive_summary": adaptive_summary,
              "locked_level7_7_summary": locked["summary"], "comparisons": comparisons, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"adaptive_summary": adaptive_summary, "comparisons": comparisons,
                      "topology": {str(r['seed']): r["topology"] for r in runs}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
