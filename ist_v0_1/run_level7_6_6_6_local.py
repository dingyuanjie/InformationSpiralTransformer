"""Level 7.6.6.6: validation-probe-guided causal Memory-slot routing at 32K."""
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
PROBES = ROOT / "experiments/level7_6_6_5/formal"
LENGTH = 32768
WINDOWS = {"near": (16, 4095), "far": (16384, 32766)}
SEEDS = (313, 42, 2026, 7, 1234)
GROUPS = {313: "low", 42: "intermediate", 2026: "high", 7: "high", 1234: "low"}
CONDITIONS = ("intact", "selected_keep", "selected_ablate", "selected_boost_2",
              "top4_keep", "top4_ablate", "top4_boost_2")
SAMPLES = 30


def selection(seed: int) -> dict:
    path = PROBES / f"seed{seed}_probes.json"
    if not path.exists():
        raise FileNotFoundError(f"Level 7.6.6.5 probe result missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    ranked = sorted(result["slot_linear"],
                    key=lambda row: (-row["validation_accuracy"], row["layer"], row["slot"]))
    selected = result["selected_slot_by_validation"]
    return {"selected": {"layer": selected["layer"], "slot": selected["slot"],
                          "validation_accuracy": selected["validation_accuracy"]},
            "top4": [{"layer": row["layer"], "slot": row["slot"],
                       "validation_accuracy": row["validation_accuracy"]} for row in ranked[:4]]}


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, LENGTH, "rope", True).to(device).eval()
    checkpoint = PARENT / f"ist-full_seed{seed}" / "stage_4096.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    for block in model.blocks:
        block.capture_memory_read_weights = True
    return model


def configure(model, chosen: dict, condition: str) -> None:
    for block in model.blocks:
        block.memory_read_topk = None
        block.memory_read_keep_slots = None
        block.memory_read_ablate_slots = None
        block.memory_read_slot_scales = None
    targets = [chosen["selected"]] if condition.startswith("selected_") else chosen["top4"]
    by_layer = {layer: [] for layer in range(3)}
    for target in targets:
        by_layer[int(target["layer"])].append(int(target["slot"]))
    operation = condition.split("_")[-1]
    for layer, slots in by_layer.items():
        if not slots:
            continue
        if operation == "keep":
            model.blocks[layer].memory_read_keep_slots = tuple(sorted(set(slots)))
        elif operation == "ablate":
            model.blocks[layer].memory_read_ablate_slots = tuple(sorted(set(slots)))
        elif operation == "2":
            model.blocks[layer].memory_read_slot_scales = {slot: 2.0 for slot in slots}


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


def selected_read_mass(model, chosen: dict) -> float:
    masses = []
    for target in chosen["top4"]:
        weights = model.blocks[int(target["layer"])].last_memory_read_weights[0].float().mean(dim=(0, 1))
        distribution = weights / weights.sum().clamp_min(1e-12)
        masses.append(float(distribution[int(target["slot"])].cpu()))
    return sum(masses)


@torch.no_grad()
def evaluate(model, seed: int, window_name: str, condition: str, chosen: dict,
             device: torch.device, dtype: torch.dtype) -> dict:
    configure(model, chosen, condition)
    correctness, probabilities, margins, read_masses = [], [], [], []
    for sample in range(SAMPLES):
        example_seed = 766600000 + seed * 1000 + (0 if window_name == "near" else 500) + sample
        tokens, target = make_example(WINDOWS[window_name], example_seed, device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(tokens)[..., :16][:, -1].float()
        target_index = int(target.item())
        correctness.append(int(logits.argmax(-1).item() == target_index))
        probabilities.append(float(logits.softmax(-1)[0, target_index].cpu()))
        competitor = logits[0].clone(); competitor[target_index] = -torch.inf
        margins.append(float((logits[0, target_index] - competitor.max()).cpu()))
        read_masses.append(selected_read_mass(model, chosen))
    return {"seed": seed, "group": GROUPS[seed], "window": window_name,
            "condition": condition, "selection": chosen, "samples": SAMPLES,
            "correct": sum(correctness), "accuracy": sum(correctness) / SAMPLES,
            "correctness": correctness, "target_probabilities": probabilities, "margins": margins,
            "mean_target_probability": sum(probabilities) / SAMPLES,
            "mean_margin": sum(margins) / SAMPLES,
            "mean_top4_read_mass_sum_across_layers": sum(read_masses) / SAMPLES}


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
    parser.add_argument("--output", default="experiments/level7_6_6_6/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selections = {str(seed): selection(seed) for seed in SEEDS}
    protocol = {"variant": "ist-full", "length": LENGTH, "windows": WINDOWS,
                "seeds": SEEDS, "groups": GROUPS, "conditions": CONDITIONS,
                "samples_per_seed_window": args.samples_per_seed_window,
                "slot_selection_source": "level7_6_6_5 validation accuracy only",
                "frozen_selections": selections, "paired_examples": True,
                "primary_seed": 1234, "negative_control_seed": 313}
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
        chosen = selections[str(seed)]
        for window_name in WINDOWS:
            for condition in CONDITIONS:
                output = root / f"seed{seed}_{window_name}_{condition}.json"
                if output.exists() and not args.force:
                    row = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row = evaluate(model, seed, window_name, condition, chosen, device, dtype)
                    atomic_save(output, row)
                runs.append(row); atomic_save(root / "runs.partial.json", runs)
                print(f"seed={seed} window={window_name} condition={condition} accuracy={row['accuracy']:.2%} "
                      f"read_mass={row['mean_top4_read_mass_sum_across_layers']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    aggregate, comparisons = [], []
    for seed in SEEDS:
        for window_scope in ("near", "far", "combined"):
            selected_windows = tuple(WINDOWS) if window_scope == "combined" else (window_scope,)
            intact_runs = [run for run in runs if run["seed"] == seed and run["window"] in selected_windows
                           and run["condition"] == "intact"]
            intact = [value for run in intact_runs for value in run["correctness"]]
            intact_p = [value for run in intact_runs for value in run["target_probabilities"]]
            intact_m = [value for run in intact_runs for value in run["margins"]]
            for condition in CONDITIONS:
                selected_runs = [run for run in runs if run["seed"] == seed and run["window"] in selected_windows
                                 and run["condition"] == condition]
                values = [value for run in selected_runs for value in run["correctness"]]
                probabilities = [value for run in selected_runs for value in run["target_probabilities"]]
                margins = [value for run in selected_runs for value in run["margins"]]
                aggregate.append({"seed": seed, "group": GROUPS[seed], "window": window_scope,
                                  "condition": condition, "correct": sum(values), "samples": len(values),
                                  "accuracy": sum(values) / len(values),
                                  "mean_target_probability": sum(probabilities) / len(probabilities),
                                  "mean_margin": sum(margins) / len(margins),
                                  "mean_top4_read_mass_sum_across_layers": sum(run["mean_top4_read_mass_sum_across_layers"] for run in selected_runs) / len(selected_runs)})
                if condition != "intact":
                    comparisons.append({"seed": seed, "group": GROUPS[seed], "window": window_scope,
                                        "condition": condition, **paired_exact(values, intact),
                                        "mean_target_probability_change": sum(a-b for a,b in zip(probabilities,intact_p))/len(probabilities),
                                        "mean_margin_change": sum(a-b for a,b in zip(margins,intact_m))/len(margins)})
    for seed in SEEDS:
        for window_scope in ("near", "far", "combined"):
            holm([row for row in comparisons if row["seed"] == seed and row["window"] == window_scope])
    primary = [row for row in comparisons if row["seed"] == 1234 and row["window"] == "combined"]
    winner = max(primary, key=lambda row: (row["accuracy_difference"], row["mean_margin_change"], -row["holm_p"]))
    result = {"protocol": protocol, "aggregate": aggregate, "paired_comparisons": comparisons,
              "primary_winner": winner, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"primary_winner": winner,
                      "seed1234_combined": primary,
                      "seed313_combined": [row for row in comparisons if row["seed"] == 313 and row["window"] == "combined"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
