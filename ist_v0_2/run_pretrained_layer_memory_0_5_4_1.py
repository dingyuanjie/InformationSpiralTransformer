"""Frozen Memory 0.5.4.1: de-collapse and sparse-read causal interventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiment_utils import ROOT, atomic_json, run_metadata
from pretrained_layer_memory_adapter import FrozenLayerInjectedIST
from pretrained_memory_adapter import load_qwen
from run_pretrained_base_smoke import MODEL_ID, candidate_ids
from run_pretrained_frozen_memory_0_4 import CHUNK, SEEDS, make_batch, paired_exact, wilson


DEFAULT_SOURCE = ROOT / "experiments/pretrained_base/layer_memory_0_5_3/formal"
METHODS = {
    "baseline": {"transform": "normal", "temperature": 1.0, "top_k": None},
    "prototype_center": {"transform": "prototype", "temperature": 1.0, "top_k": None},
    "slot_center": {"transform": "slot_center", "temperature": 1.0, "top_k": None},
    "prototype_pc1": {"transform": "prototype_pc1", "temperature": 1.0, "top_k": None},
    "prototype_temp025": {"transform": "prototype", "temperature": .25, "top_k": None},
    "prototype_topk4": {"transform": "prototype", "temperature": 1.0, "top_k": 4},
    "prototype_pc1_topk4": {"transform": "prototype_pc1", "temperature": 1.0, "top_k": 4},
}


@torch.no_grad()
def collect_fast(adapter, tokenizer, seeds, device, batch):
    states = []
    for start in range(0, len(seeds), batch):
        ids, _ = make_batch(tokenizer, seeds[start:start + batch], "validation", device)
        first = ids[:, :CHUNK]
        _, state = adapter(first, None, detach_state=True)
        states.append(state["fast"].float().cpu())
    return torch.cat(states)


def calibrate(adapter, tokenizer, seed, args, device):
    seeds = [args.calibration_seed_base + seed * 10000 + i for i in range(args.calibration_samples)]
    fast = collect_fast(adapter, tokenizer, seeds, device, args.batch)
    prototype = fast.mean(0)
    centered = fast.flatten(1) - prototype.flatten()[None]
    # Right singular vector of the largest centered variation component.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    pc1 = vh[0].reshape_as(prototype)
    explained = float(
        (centered @ vh[0]).square().sum() / centered.square().sum().clamp_min(1e-12)
    )
    return prototype, pc1, {
        "calibration_samples": len(seeds),
        "prototype_norm": float(prototype.norm()),
        "pc1_explained_fraction": explained,
    }


def configure(adapter, method):
    settings = METHODS[method]
    adapter.memory_transform = settings["transform"]
    adapter.read_temperature = settings["temperature"]
    adapter.read_top_k = settings["top_k"]


@torch.no_grad()
def evaluate_seed(backbone, adapter, tokenizer, labels, seed, args, device, methods):
    seeds = [args.heldout_seed_base + seed * 10000 + i for i in range(args.samples)]
    correctness = {method: {"normal": [], "swap": []} for method in methods}
    predictions = {method: {"normal": [], "swap": []} for method in methods}
    base = []
    labels = labels.to(device)
    for start in range(0, len(seeds), args.batch):
        ids, target = make_batch(tokenizer, seeds[start:start + args.batch], "held_out", device)
        full = backbone(ids, use_cache=False).logits[:, -1, labels].argmax(-1)
        base.extend((full == target).int().cpu().tolist())
        first, second = ids.split(CHUNK, dim=1)
        adapter.memory_transform = "normal"
        adapter.read_temperature = 1.0
        adapter.read_top_k = None
        _, state = adapter(first, None, detach_state=True)
        for method in methods:
            configure(adapter, method)
            for condition in ("normal", "swap"):
                intervention = "normal" if condition == "normal" else "swap_fast"
                logits, _ = adapter(second, state, intervention=intervention, detach_state=True)
                prediction = logits[:, -1, labels].argmax(-1)
                predictions[method][condition].extend(prediction.cpu().tolist())
                correctness[method][condition].extend(
                    (prediction == target).int().cpu().tolist()
                )
    results = {}
    baseline_predictions = predictions["baseline"]["normal"]
    for method in methods:
        normal = correctness[method]["normal"]
        swap = correctness[method]["swap"]
        paired = paired_exact(normal, swap)
        results[method] = {
            "settings": METHODS[method],
            "normal_accuracy": sum(normal) / len(normal),
            "swap_accuracy": sum(swap) / len(swap),
            "normal_minus_swap": (sum(normal) - sum(swap)) / len(normal),
            "normal_vs_swap": paired,
            "prediction_change_vs_baseline": sum(
                a != b for a, b in zip(predictions[method]["normal"], baseline_predictions)
            ) / len(normal),
            "normal_correctness": normal,
            "swap_correctness": swap,
        }
    return {"base_accuracy": sum(base) / len(base), "methods": results}


def aggregate(runs, methods):
    output = {}
    baseline_normal = [
        value for run in runs
        for value in run["evaluation"]["methods"]["baseline"]["normal_correctness"]
    ]
    for method in methods:
        normal = [value for run in runs for value in run["evaluation"]["methods"][method]["normal_correctness"]]
        swap = [value for run in runs for value in run["evaluation"]["methods"][method]["swap_correctness"]]
        correct = sum(normal)
        output[method] = {
            "settings": METHODS[method],
            "normal_accuracy": correct / len(normal),
            "normal_wilson95": wilson(correct, len(normal)),
            "swap_accuracy": sum(swap) / len(swap),
            "normal_vs_swap": paired_exact(normal, swap),
            "normal_vs_baseline": paired_exact(normal, baseline_normal),
            "samples": len(normal),
        }
    baseline = output["baseline"]["normal_accuracy"]
    for method in output:
        output[method]["normal_gain_vs_baseline"] = output[method]["normal_accuracy"] - baseline
    return output


def main(default_methods=None, default_calibration_samples=64, default_samples=64,
         default_output="experiments/pretrained_base/layer_memory_0_5_4_1/formal",
         stage="Frozen Memory 0.5.4.1", calibration_seed_base=450000000,
         heldout_seed_base=460000000, primary_method=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--calibration-samples", type=int, default=default_calibration_samples)
    parser.add_argument("--samples", type=int, default=default_samples)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--methods", nargs="+", default=default_methods or list(METHODS))
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    unknown = [name for name in args.methods if name not in METHODS]
    if unknown:
        parser.error("unknown methods: " + ", ".join(unknown))
    if "baseline" not in args.methods:
        parser.error("methods must include baseline")
    args.calibration_seed_base = calibration_seed_base
    args.heldout_seed_base = heldout_seed_base
    if args.smoke_test:
        args.seeds = [2026]
        args.calibration_samples = 4
        args.samples = 4
        args.batch = 2
        if args.output.endswith("formal"):
            args.output = args.output[:-6] + "smoke"
    protocol = {
        "stage": stage,
        "task": "frozen Memory de-collapse and sparse-read interventions",
        "source": str(args.source),
        "seeds": args.seeds,
        "calibration_samples_per_seed": args.calibration_samples,
        "heldout_samples_per_seed": args.samples,
        "methods": {name: METHODS[name] for name in args.methods},
        "primary_method": primary_method,
        "calibration_seed_base": calibration_seed_base,
        "heldout_seed_base": heldout_seed_base,
        "training": False,
        "calibration_split_separate_from_heldout": True,
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    missing = [str(args.source / f"seed{seed}" / "best.pt") for seed in args.seeds
               if not (args.source / f"seed{seed}" / "best.pt").exists()]
    if missing:
        raise FileNotFoundError("missing Level 0.5.3 checkpoints: " + ", ".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    tokenizer, backbone = load_qwen(args.model_id, dtype, device, args.local_files_only)
    labels = candidate_ids(tokenizer)
    runs = []
    for seed in args.seeds:
        adapter = FrozenLayerInjectedIST(
            backbone, injection_layer=-4, heads=8, layer_matched_write=True
        ).to(device=device, dtype=dtype)
        adapter.injection_scale.data = adapter.injection_scale.data.float()
        checkpoint = torch.load(args.source / f"seed{seed}" / "best.pt", map_location=device, weights_only=False)
        adapter.load_trainable_state_dict(checkpoint["adapter"])
        adapter.eval()
        prototype, pc1, calibration = calibrate(adapter, tokenizer, seed, args, device)
        adapter.memory_prototype = prototype.to(device=device, dtype=dtype)
        adapter.memory_pc1 = pc1.to(device=device, dtype=dtype)
        evaluation = evaluate_seed(
            backbone, adapter, tokenizer, labels, seed, args, device, args.methods
        )
        row = {"seed": seed, "calibration": calibration, "evaluation": evaluation}
        runs.append(row)
        print(f"seed={seed} " + json.dumps({"calibration": calibration, "methods": {
            key: {k: v for k, v in value.items() if not k.endswith("correctness")}
            for key, value in evaluation["methods"].items()}}, default=str), flush=True)
        del adapter
        torch.cuda.empty_cache()
    summary = aggregate(runs, args.methods)
    ranked = sorted(summary, key=lambda key: summary[key]["normal_gain_vs_baseline"], reverse=True)
    best = ranked[0]
    confirmation_passed = None
    if primary_method is not None:
        primary = summary[primary_method]
        confirmation_passed = (
            primary["normal_wilson95"][0] > .25
            and primary["normal_vs_baseline"]["difference"] > 0
            and primary["normal_vs_baseline"]["mcnemar_exact_p"] < .05
            and primary["normal_vs_swap"]["difference"] > 0
            and primary["normal_vs_swap"]["mcnemar_exact_p"] < .05
        )
    result = {
        "status": "complete",
        "confirmation_passed": confirmation_passed,
        "best_method": best,
        "best_method_summary": summary[best],
        "ranking": ranked,
        "summary": summary,
        "runs": runs,
        "protocol": protocol,
    }
    atomic_json(root / "config.json", protocol)
    atomic_json(root / "run_metadata.json", run_metadata(device, args.seeds))
    atomic_json(root / "raw_results.json", result)
    atomic_json(root / "result.json", result)
    lines = ["# Frozen Memory 0.5.4.1", "", f"Best frozen intervention: `{best}`.", "", "## Ranking", ""]
    lines += [f"- {name}: gain {summary[name]['normal_gain_vs_baseline']:+.4f}" for name in ranked]
    (root / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
