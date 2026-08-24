"""Level 7.6.6.7: independent confirmation of seed-7 probe-selected L3 top-4 causality."""
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
PARENT = ROOT / "experiments/level7_6_4/formal/ist-full_seed7/stage_4096.pt"
LENGTH = 32768
WINDOWS = {"near": (16, 4095), "far": (16384, 32766)}
SEED = 7
PROBE_TOP4 = (0, 20, 24, 26)
RANDOM_CONTROLS = {
    "random4_a_l3_ablate": (1, 5, 12, 29),
    "random4_b_l3_ablate": (2, 8, 17, 30),
    "random4_c_l3_ablate": (3, 10, 18, 27),
}
CONDITIONS = ("intact", "probe_top4_l3_ablate", *RANDOM_CONTROLS,
              "probe_indices_l2_ablate", "probe_top4_l3_boost_2")
SAMPLES = 128


def build(device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, LENGTH, "rope", True).to(device).eval()
    model.load_state_dict(torch.load(PARENT, map_location=device, weights_only=False)["model"])
    return model


def configure(model, condition: str) -> None:
    for block in model.blocks:
        block.memory_read_keep_slots = None
        block.memory_read_ablate_slots = None
        block.memory_read_slot_scales = None
        block.memory_read_topk = None
    if condition == "probe_top4_l3_ablate":
        model.blocks[2].memory_read_ablate_slots = PROBE_TOP4
    elif condition in RANDOM_CONTROLS:
        model.blocks[2].memory_read_ablate_slots = RANDOM_CONTROLS[condition]
    elif condition == "probe_indices_l2_ablate":
        model.blocks[1].memory_read_ablate_slots = PROBE_TOP4
    elif condition == "probe_top4_l3_boost_2":
        model.blocks[2].memory_read_slot_scales = {slot: 2.0 for slot in PROBE_TOP4}


def make_example(window: tuple[int, int], sample: int, seed: int, device: torch.device):
    set_seed(seed)
    target_value = sample % 16
    target = torch.tensor([target_value], device=device)
    tokens = torch.randint(16, (1, LENGTH), device=device)
    distance = int(torch.randint(window[0], window[1] + 1, (1,), device=device).item())
    position = LENGTH - 2 - distance
    tokens[0, position] = 17
    tokens[0, position + 1] = target
    tokens[0, -2] = 18
    tokens[0, -1] = 16
    return tokens, target


@torch.no_grad()
def evaluate(model, window_name: str, condition: str,
             device: torch.device, dtype: torch.dtype) -> dict:
    configure(model, condition)
    correctness, probabilities, margins = [], [], []
    for sample in range(SAMPLES):
        example_seed = 766700000 + (0 if window_name == "near" else 1000) + sample
        tokens, target = make_example(WINDOWS[window_name], sample, example_seed, device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(tokens)[..., :16][:, -1].float()
        target_index = int(target.item())
        correctness.append(int(logits.argmax(-1).item() == target_index))
        probabilities.append(float(logits.softmax(-1)[0, target_index].cpu()))
        competitor = logits[0].clone(); competitor[target_index] = -torch.inf
        margins.append(float((logits[0, target_index] - competitor.max()).cpu()))
        if (sample + 1) % 32 == 0:
            print(f"window={window_name} condition={condition} sample={sample + 1}/{SAMPLES}", flush=True)
    return {"seed": SEED, "window": window_name, "range": list(WINDOWS[window_name]),
            "condition": condition, "samples": SAMPLES, "correct": sum(correctness),
            "accuracy": sum(correctness) / SAMPLES, "correctness": correctness,
            "target_probabilities": probabilities, "margins": margins,
            "mean_target_probability": sum(probabilities) / SAMPLES,
            "mean_margin": sum(margins) / SAMPLES}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-window", type=int, default=SAMPLES)
    parser.add_argument("--output", default="experiments/level7_6_6_7/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "seed": SEED, "length": LENGTH, "windows": WINDOWS,
                "samples_per_window": args.samples_per_window, "balanced_targets": True,
                "probe_top4_l3": PROBE_TOP4, "random_l3_controls": RANDOM_CONTROLS,
                "conditions": CONDITIONS, "paired_examples": True,
                "single_primary_contrast": "probe_top4_l3_ablate vs intact, combined windows",
                "primary_alpha": 0.05, "primary_multiplicity_correction": "not required; one preregistered test"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if args.samples_per_window != SAMPLES:
        raise ValueError(f"Formal protocol locks --samples-per-window={SAMPLES}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol, "gpu": torch.cuda.get_device_name(device),
                                           "torch": torch.__version__, "dtype": str(dtype)})
    model = build(device)
    runs = []
    for window_name in WINDOWS:
        for condition in CONDITIONS:
            output = root / f"{window_name}_{condition}.json"
            if output.exists() and not args.force:
                row = json.loads(output.read_text(encoding="utf-8"))
            else:
                row = evaluate(model, window_name, condition, device, dtype)
                atomic_save(output, row)
            runs.append(row); atomic_save(root / "runs.partial.json", runs)
            print(f"window={window_name} condition={condition} accuracy={row['accuracy']:.2%}", flush=True)
    comparisons = []
    aggregate = []
    for window_scope in ("near", "far", "combined"):
        selected_windows = tuple(WINDOWS) if window_scope == "combined" else (window_scope,)
        intact_runs = [run for run in runs if run["window"] in selected_windows and run["condition"] == "intact"]
        intact = [value for run in intact_runs for value in run["correctness"]]
        intact_p = [value for run in intact_runs for value in run["target_probabilities"]]
        intact_m = [value for run in intact_runs for value in run["margins"]]
        for condition in CONDITIONS:
            selected = [run for run in runs if run["window"] in selected_windows and run["condition"] == condition]
            values = [value for run in selected for value in run["correctness"]]
            probabilities = [value for run in selected for value in run["target_probabilities"]]
            margins = [value for run in selected for value in run["margins"]]
            aggregate.append({"window": window_scope, "condition": condition,
                              "correct": sum(values), "samples": len(values),
                              "accuracy": sum(values) / len(values),
                              "mean_target_probability": sum(probabilities) / len(probabilities),
                              "mean_margin": sum(margins) / len(margins)})
            if condition != "intact":
                comparisons.append({"window": window_scope, "condition": condition,
                                    **paired_exact(values, intact),
                                    "mean_target_probability_change": sum(a-b for a,b in zip(probabilities,intact_p))/len(probabilities),
                                    "mean_margin_change": sum(a-b for a,b in zip(margins,intact_m))/len(margins)})
    primary = next(row for row in comparisons if row["window"] == "combined" and row["condition"] == "probe_top4_l3_ablate")
    random_effects = [row["accuracy_difference"] for row in comparisons
                      if row["window"] == "combined" and row["condition"] in RANDOM_CONTROLS]
    specificity = {"probe_top4_accuracy_effect": primary["accuracy_difference"],
                   "mean_random4_accuracy_effect": sum(random_effects) / len(random_effects),
                   "probe_minus_mean_random_effect": primary["accuracy_difference"] - sum(random_effects) / len(random_effects)}
    result = {"protocol": protocol, "primary_result": primary,
              "primary_confirmed": primary["mcnemar_exact_p"] < 0.05 and primary["accuracy_difference"] < 0,
              "specificity": specificity, "aggregate": aggregate,
              "secondary_comparisons": comparisons, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"primary_result": primary, "primary_confirmed": result["primary_confirmed"],
                      "specificity": specificity,
                      "combined_secondary": [row for row in comparisons if row["window"] == "combined"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
