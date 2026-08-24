"""Level 7.6.6.4: causal slot-geometry and write-content rescue at 32K."""
from __future__ import annotations

import argparse
import json
import math
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
LOW_SEEDS = (313, 1234)
HIGH_SEEDS = (2026, 7)
SEEDS = LOW_SEEDS + HIGH_SEEDS
CONDITIONS = {
    "intact": (0.0, 0.0, 0.0),
    "l3_decor_0_25": (0.0, 0.0, 0.25),
    "l3_decor_0_50": (0.0, 0.0, 0.50),
    "l3_decor_1_00": (0.0, 0.0, 1.00),
    "l1_decor_0_25": (0.25, 0.0, 0.0),
    "l2_l3_decor_0_25": (0.0, 0.25, 0.25),
    "all_layers_decor_0_25": (0.25, 0.25, 0.25),
}
SAMPLES = 30


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, LENGTH, "rope", True).to(device).eval()
    checkpoint = PARENT / f"ist-full_seed{seed}" / "stage_4096.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model


def configure(model, condition: str) -> None:
    for block, strength in zip(model.blocks, CONDITIONS[condition]):
        block.memory.slot_decorrelation_strength = strength


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


def redundancy(memory: torch.Tensor) -> float:
    normalized = F.normalize(memory[0].float(), dim=-1)
    cosine = normalized @ normalized.T
    mask = ~torch.eye(cosine.size(0), dtype=torch.bool, device=cosine.device)
    return float(cosine[mask].abs().mean().cpu())


@torch.no_grad()
def evaluate(model, seed: int, window_name: str, condition: str,
             device: torch.device, dtype: torch.dtype) -> dict:
    configure(model, condition)
    correctness = []
    target_probabilities = []
    margins = []
    redundancies = []
    for sample in range(SAMPLES):
        example_seed = 766400000 + seed * 1000 + (0 if window_name == "near" else 500) + sample
        tokens, target = make_example(WINDOWS[window_name], example_seed, device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(tokens)[..., :16][:, -1].float()
        target_index = int(target.item())
        prediction = int(logits.argmax(-1).item())
        correctness.append(int(prediction == target_index))
        probability = logits.softmax(-1)[0, target_index]
        target_probabilities.append(float(probability.cpu()))
        target_logit = logits[0, target_index]
        competitor = logits[0].clone()
        competitor[target_index] = -torch.inf
        margins.append(float((target_logit - competitor.max()).cpu()))
        redundancies.append([redundancy(block.memory.last_diagnostics["new_memory"])
                             for block in model.blocks])
    return {"seed": seed, "seed_group": "low" if seed in LOW_SEEDS else "high",
            "window": window_name, "range": list(WINDOWS[window_name]),
            "condition": condition, "strengths": list(CONDITIONS[condition]),
            "samples": SAMPLES, "correct": sum(correctness),
            "accuracy": sum(correctness) / SAMPLES, "correctness": correctness,
            "target_probabilities": target_probabilities, "margins": margins,
            "mean_target_probability": sum(target_probabilities) / SAMPLES,
            "mean_margin": sum(margins) / SAMPLES,
            "mean_slot_redundancy_by_layer": [sum(row[layer] for row in redundancies) / SAMPLES
                                               for layer in range(3)]}


def paired_exact(treatment: list[int], intact: list[int]) -> dict:
    rescued = sum(a == 1 and b == 0 for a, b in zip(treatment, intact))
    harmed = sum(a == 0 and b == 1 for a, b in zip(treatment, intact))
    discordant = rescued + harmed
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(rescued, harmed) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {"accuracy_difference": (sum(treatment) - sum(intact)) / len(intact),
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
    parser.add_argument("--output", default="experiments/level7_6_6_4/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "length": LENGTH, "windows": WINDOWS,
                "low_seeds_primary_rescue": LOW_SEEDS, "high_seeds_specificity_control": HIGH_SEEDS,
                "conditions": CONDITIONS, "samples_per_seed_window": args.samples_per_seed_window,
                "paired_examples": True, "geometry_operation": "norm-preserving sign-aligned QR interpolation",
                "primary_endpoint": "low-group combined accuracy change vs intact",
                "secondary_endpoints": ["target_probability", "target_logit_margin", "slot_redundancy"]}
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
                print(f"seed={seed} group={row['seed_group']} window={window_name} condition={condition} "
                      f"accuracy={row['accuracy']:.2%} margin={row['mean_margin']:.3f} "
                      f"redundancy={row['mean_slot_redundancy_by_layer']}", flush=True)
        del model
        torch.cuda.empty_cache()

    aggregate = []
    comparisons = []
    for group, seeds in (("low", LOW_SEEDS), ("high", HIGH_SEEDS)):
        for window_scope in ("near", "far", "combined"):
            selected_windows = tuple(WINDOWS) if window_scope == "combined" else (window_scope,)
            intact_runs = [run for run in runs if run["seed"] in seeds and run["window"] in selected_windows
                           and run["condition"] == "intact"]
            intact = [value for run in intact_runs for value in run["correctness"]]
            intact_probability = [value for run in intact_runs for value in run["target_probabilities"]]
            intact_margin = [value for run in intact_runs for value in run["margins"]]
            for condition in CONDITIONS:
                selected = [run for run in runs if run["seed"] in seeds and run["window"] in selected_windows
                            and run["condition"] == condition]
                treatment = [value for run in selected for value in run["correctness"]]
                probabilities = [value for run in selected for value in run["target_probabilities"]]
                margins = [value for run in selected for value in run["margins"]]
                aggregate.append({"group": group, "window": window_scope, "condition": condition,
                                  "correct": sum(treatment), "samples": len(treatment),
                                  "accuracy": sum(treatment) / len(treatment),
                                  "mean_target_probability": sum(probabilities) / len(probabilities),
                                  "mean_margin": sum(margins) / len(margins),
                                  "mean_slot_redundancy_by_layer": [
                                      sum(run["mean_slot_redundancy_by_layer"][layer] for run in selected) / len(selected)
                                      for layer in range(3)]})
                if condition != "intact":
                    comparisons.append({"group": group, "window": window_scope, "condition": condition,
                                        **paired_exact(treatment, intact),
                                        "mean_target_probability_change": sum(p - q for p, q in zip(probabilities, intact_probability)) / len(probabilities),
                                        "mean_margin_change": sum(p - q for p, q in zip(margins, intact_margin)) / len(margins)})
    for group in ("low", "high"):
        for window_scope in ("near", "far", "combined"):
            holm([row for row in comparisons if row["group"] == group and row["window"] == window_scope])
    low_combined = [row for row in comparisons if row["group"] == "low" and row["window"] == "combined"]
    winner = max(low_combined, key=lambda row: (row["accuracy_difference"], row["mean_margin_change"], -row["holm_p"]))
    result = {"protocol": protocol, "aggregate": aggregate, "paired_comparisons": comparisons,
              "primary_winner": winner, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"primary_winner": winner,
                      "low_combined": low_combined,
                      "high_combined": [row for row in comparisons if row["group"] == "high" and row["window"] == "combined"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
