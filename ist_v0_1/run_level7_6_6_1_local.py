"""Level 7.6.6.1: 32K successful/failed-seed Memory-state contrast."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level7_1_local import atomic_save


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
LENGTH = 32768
WINDOWS = {"near": (16, 4095), "far": (16384, 32766)}
SEEDS = (313, 42, 2026, 7, 1234)
GROUPS = {"high": (2026, 7), "intermediate": (42,), "low": (313, 1234)}
LAYERS = 3
SLOTS = 32


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, LAYERS, LENGTH, "rope", True).to(device).eval()
    checkpoint = PARENT / f"ist-full_seed{seed}" / "stage_4096.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    for block in model.blocks:
        block.capture_memory_read_weights = True
        block.memory.capture_memory_attention_weights = True
    return model


def make_example(window: tuple[int, int], seed: int, device: torch.device):
    set_seed(seed)
    target = torch.randint(16, (1,), device=device)
    tokens = torch.randint(16, (1, LENGTH), device=device)
    distance = int(torch.randint(window[0], window[1] + 1, (1,), device=device).item())
    position = LENGTH - 2 - distance
    tokens[0, position] = 17
    tokens[0, position + 1] = target
    tokens[0, -2] = 18
    tokens[0, -1] = 16
    return tokens, target, distance


def entropy(distribution: torch.Tensor) -> float:
    safe = distribution.float().clamp_min(1e-12)
    return float((-(safe * safe.log()).sum()).cpu())


def extract_layer(block, layer: int) -> dict:
    diagnostics = block.memory.last_diagnostics
    memory = diagnostics["new_memory"][0].float()
    normalized = F.normalize(memory, dim=-1)
    cosine = normalized @ normalized.T
    mask = ~torch.eye(SLOTS, dtype=torch.bool, device=cosine.device)
    read_mass = block.last_memory_read_weights[0].float().mean(dim=(0, 1))
    propagation_mass = diagnostics["memory_attention_weights"][0].float().mean(dim=(0, 1))
    read_mass = read_mass / read_mass.sum().clamp_min(1e-12)
    propagation_mass = propagation_mass / propagation_mass.sum().clamp_min(1e-12)
    return {"layer": layer,
            "diversity_loss": float(diagnostics["diversity_loss"].cpu()),
            "write_entropy_mean": float(diagnostics["attention_entropy"].float().mean().cpu()),
            "write_entropy_by_slot": diagnostics["attention_entropy"][0].float().cpu().tolist(),
            "update_gate_mean": float(diagnostics["update_gate"].float().mean().cpu()),
            "update_gate_by_slot": diagnostics["update_gate"][0].float().mean(-1).cpu().tolist(),
            "slot_norm_mean": float(memory.norm(dim=-1).mean().cpu()),
            "slot_norm_by_slot": memory.norm(dim=-1).cpu().tolist(),
            "slot_abs_cosine_offdiag": float(cosine[mask].abs().mean().cpu()),
            "read_entropy": entropy(read_mass), "read_mass_by_slot": read_mass.cpu().tolist(),
            "propagation_read_entropy": entropy(propagation_mass),
            "propagation_mass_by_slot": propagation_mass.cpu().tolist(),
            "fusion_gate_mean": float(block.last_fusion_gate.float().mean().cpu()),
            "propagation_ratio": float(diagnostics["propagation_ratio"].float().mean().cpu())}


@torch.no_grad()
def evaluate(model, seed: int, window_name: str, samples: int,
             device: torch.device, dtype: torch.dtype) -> dict:
    rows = []
    for sample in range(samples):
        example_seed = 766100000 + seed * 1000 + (0 if window_name == "near" else 500) + sample
        tokens, target, distance = make_example(WINDOWS[window_name], example_seed, device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(tokens)
        prediction = logits[..., :16][:, -1].argmax(-1)
        rows.append({"sample": sample, "distance": distance,
                     "correct": int(prediction.item() == target.item()),
                     "layers": [extract_layer(block, index) for index, block in enumerate(model.blocks)]})
        print(f"seed={seed} window={window_name} sample={sample + 1}/{samples} "
              f"running_accuracy={sum(row['correct'] for row in rows) / len(rows):.2%}", flush=True)
    return {"seed": seed, "window": window_name, "range": list(WINDOWS[window_name]),
            "samples": samples, "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / samples, "rows": rows}


def mean(values):
    return statistics.mean(values) if values else None


def standardized_difference(high: list[float], low: list[float]) -> float | None:
    if len(high) < 2 or len(low) < 2:
        return None
    pooled = math.sqrt((statistics.variance(high) + statistics.variance(low)) / 2)
    return (mean(high) - mean(low)) / pooled if pooled else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-seed-window", type=int, default=20)
    parser.add_argument("--output", default="experiments/level7_6_6_1/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "length": LENGTH, "windows": WINDOWS,
                "seeds": SEEDS, "preregistered_groups": GROUPS,
                "samples_per_seed_window": args.samples_per_seed_window,
                "captured": ["write_entropy", "update_gate", "slot_norm", "slot_redundancy",
                             "read_mass", "propagation_read_mass", "fusion_gate", "propagation_ratio"]}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if args.samples_per_seed_window != 20:
        raise ValueError("Formal protocol locks --samples-per-seed-window=20")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol, "gpu": torch.cuda.get_device_name(device),
                                           "torch": torch.__version__, "dtype": str(dtype)})
    runs = []
    for seed in SEEDS:
        model = build(seed, device)
        for window_name in WINDOWS:
            output = root / f"seed{seed}_{window_name}.json"
            if output.exists() and not args.force:
                run = json.loads(output.read_text(encoding="utf-8"))
            else:
                run = evaluate(model, seed, window_name, args.samples_per_seed_window, device, dtype)
                atomic_save(output, run)
            runs.append(run)
            atomic_save(root / "runs.partial.json", runs)
        del model
        torch.cuda.empty_cache()

    scalar_metrics = ("diversity_loss", "write_entropy_mean", "update_gate_mean", "slot_norm_mean",
                      "slot_abs_cosine_offdiag", "read_entropy", "propagation_read_entropy",
                      "fusion_gate_mean", "propagation_ratio")
    summary = []
    contrasts = []
    for window_name in WINDOWS:
        for layer in range(LAYERS):
            for group_name, group_seeds in GROUPS.items():
                # Build one compact row per group/layer without duplicating sample tensors.
                row = {"window": window_name, "layer": layer, "group": group_name,
                       "seeds": list(group_seeds)}
                matching = [sample["layers"][layer] for run in runs if run["window"] == window_name
                            and run["seed"] in group_seeds for sample in run["rows"]]
                for metric in scalar_metrics:
                    row[metric] = mean([item[metric] for item in matching])
                summary.append(row)
            for metric in scalar_metrics:
                high = [sample["layers"][layer][metric] for run in runs if run["window"] == window_name
                        and run["seed"] in GROUPS["high"] for sample in run["rows"]]
                low = [sample["layers"][layer][metric] for run in runs if run["window"] == window_name
                       and run["seed"] in GROUPS["low"] for sample in run["rows"]]
                contrasts.append({"window": window_name, "layer": layer, "metric": metric,
                                  "high_mean": mean(high), "low_mean": mean(low),
                                  "difference": mean(high) - mean(low),
                                  "standardized_difference": standardized_difference(high, low)})
    accuracies = [{"seed": run["seed"], "window": run["window"], "accuracy": run["accuracy"]} for run in runs]
    result = {"protocol": protocol, "accuracies": accuracies, "group_layer_summary": summary,
              "high_vs_low_contrasts": contrasts, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"accuracies": accuracies,
                      "largest_absolute_contrasts": sorted(contrasts, key=lambda row: abs(row["standardized_difference"] or 0), reverse=True)[:12]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
