"""Level 7.6.6.3: targeted layer-3 read concentration and fusion rescue at 32K."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level7_1_local import atomic_save


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
LENGTH = 32768
WINDOWS = {"near": (16, 4095), "far": (16384, 32766)}
LOW_SEEDS = (313, 1234)
HIGH_SEEDS = (2026, 7)
SEEDS = LOW_SEEDS + HIGH_SEEDS
CONDITIONS = {
    "intact": {"topk": None, "fusion_floor": None},
    "fusion_floor_0_35": {"topk": None, "fusion_floor": 0.35},
    "fusion_floor_0_50": {"topk": None, "fusion_floor": 0.50},
    "read_top24": {"topk": 24, "fusion_floor": None},
    "read_top16": {"topk": 16, "fusion_floor": None},
    "read_top16_fusion_0_35": {"topk": 16, "fusion_floor": 0.35},
    "read_top16_fusion_0_50": {"topk": 16, "fusion_floor": 0.50},
}
SAMPLES = 30


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, LENGTH, "rope", True).to(device).eval()
    checkpoint = PARENT / f"ist-full_seed{seed}" / "stage_4096.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    model.blocks[2].capture_memory_read_weights = True
    return model


def configure(model, condition: str) -> None:
    for block in model.blocks:
        block.memory_read_topk = None
        block.fusion_gate_floor = None
    model.blocks[2].memory_read_topk = CONDITIONS[condition]["topk"]
    model.blocks[2].fusion_gate_floor = CONDITIONS[condition]["fusion_floor"]


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
    return tokens, target


def read_entropy(block) -> float:
    weights = block.last_memory_read_weights[0].float().mean(dim=(0, 1))
    distribution = weights / weights.sum().clamp_min(1e-12)
    safe = distribution.clamp_min(1e-12)
    return float((-(safe * safe.log()).sum()).cpu())


@torch.no_grad()
def evaluate(model, seed: int, window_name: str, condition: str,
             device: torch.device, dtype: torch.dtype) -> dict:
    configure(model, condition)
    correctness = []
    entropies = []
    fusion_means = []
    for sample in range(SAMPLES):
        example_seed = 766300000 + seed * 1000 + (0 if window_name == "near" else 500) + sample
        tokens, target = make_example(WINDOWS[window_name], example_seed, device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(tokens)
        correctness.append(int(logits[..., :16][:, -1].argmax(-1).item() == target.item()))
        entropies.append(read_entropy(model.blocks[2]))
        fusion_means.append(float(model.blocks[2].last_fusion_gate.float().mean().cpu()))
    return {"seed": seed, "seed_group": "low" if seed in LOW_SEEDS else "high",
            "window": window_name, "range": list(WINDOWS[window_name]),
            "condition": condition, **CONDITIONS[condition], "samples": SAMPLES,
            "correct": sum(correctness), "accuracy": sum(correctness) / SAMPLES,
            "correctness": correctness, "mean_l3_read_entropy": sum(entropies) / len(entropies),
            "mean_l3_fusion_gate": sum(fusion_means) / len(fusion_means)}


def paired_exact(treatment: list[int], intact: list[int]) -> dict:
    rescued = sum(a == 1 and b == 0 for a, b in zip(treatment, intact))
    harmed = sum(a == 0 and b == 1 for a, b in zip(treatment, intact))
    discordant = rescued + harmed
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(rescued, harmed) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {"difference": (sum(treatment) - sum(intact)) / len(intact),
            "rescued": rescued, "harmed": harmed, "ties": len(intact) - discordant,
            "mcnemar_exact_p": p_value}


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["mcnemar_exact_p"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["mcnemar_exact_p"])
        running = max(running, adjusted)
        rows[index]["holm_p"] = running
        rows[index]["holm_significant"] = running < 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-seed-window", type=int, default=SAMPLES)
    parser.add_argument("--output", default="experiments/level7_6_6_3/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "length": LENGTH, "windows": WINDOWS,
                "low_seeds_primary_rescue": LOW_SEEDS, "high_seeds_specificity_control": HIGH_SEEDS,
                "conditions": CONDITIONS, "samples_per_seed_window": args.samples_per_seed_window,
                "paired_examples": True, "layer_intervened": 3,
                "primary_endpoint": "low-group combined accuracy change vs intact"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if args.samples_per_seed_window != SAMPLES:
        raise ValueError(f"Formal protocol locks --samples-per-seed-window={SAMPLES}")
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
            for condition in CONDITIONS:
                output = root / f"seed{seed}_{window_name}_{condition}.json"
                if output.exists() and not args.force:
                    row = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row = evaluate(model, seed, window_name, condition, device, dtype)
                    atomic_save(output, row)
                runs.append(row)
                atomic_save(root / "runs.partial.json", runs)
                print(f"seed={seed} group={row['seed_group']} window={window_name} "
                      f"condition={condition} accuracy={row['accuracy']:.2%} "
                      f"entropy={row['mean_l3_read_entropy']:.3f} fusion={row['mean_l3_fusion_gate']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    aggregate = []
    comparisons = []
    for group, seeds in (("low", LOW_SEEDS), ("high", HIGH_SEEDS)):
        for window_scope in ("near", "far", "combined"):
            selected_windows = tuple(WINDOWS) if window_scope == "combined" else (window_scope,)
            intact = [value for run in runs if run["seed"] in seeds and run["window"] in selected_windows
                      and run["condition"] == "intact" for value in run["correctness"]]
            for condition in CONDITIONS:
                selected = [run for run in runs if run["seed"] in seeds and run["window"] in selected_windows
                            and run["condition"] == condition]
                treatment = [value for run in selected for value in run["correctness"]]
                aggregate.append({"group": group, "window": window_scope, "condition": condition,
                                  "correct": sum(treatment), "samples": len(treatment),
                                  "accuracy": sum(treatment) / len(treatment),
                                  "mean_l3_read_entropy": sum(run["mean_l3_read_entropy"] for run in selected) / len(selected),
                                  "mean_l3_fusion_gate": sum(run["mean_l3_fusion_gate"] for run in selected) / len(selected)})
                if condition != "intact":
                    comparisons.append({"group": group, "window": window_scope, "condition": condition,
                                        **paired_exact(treatment, intact)})
    for group in ("low", "high"):
        for window_scope in ("near", "far", "combined"):
            holm([row for row in comparisons if row["group"] == group and row["window"] == window_scope])
    low_combined = [row for row in comparisons if row["group"] == "low" and row["window"] == "combined"]
    winner = max(low_combined, key=lambda row: (row["difference"], -row["holm_p"]))
    result = {"protocol": protocol, "aggregate": aggregate, "paired_comparisons": comparisons,
              "primary_winner": winner, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"primary_winner": winner,
                      "low_combined": low_combined,
                      "high_combined": [row for row in comparisons if row["group"] == "high" and row["window"] == "combined"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
