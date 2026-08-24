"""Level 7.6.6.9: seed-2026 distributed Memory redundancy deletion-dose curve."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level7_1_local import atomic_save


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "experiments/level7_6_4/formal/ist-full_seed2026/stage_4096.pt"
PROBE_RESULT = ROOT / "experiments/level7_6_6_5/formal/seed2026_probes.json"
LENGTH = 32768
SEED = 2026
WINDOWS = {"near": (16, 4095), "far": (16384, 32766)}
DOSES = (4, 8, 12, 16, 24)
SAMPLES = 96


def frozen_ladders() -> dict:
    probes = json.loads(PROBE_RESULT.read_text(encoding="utf-8"))
    ranked = sorted(probes["slot_linear"],
                    key=lambda row: (-row["validation_accuracy"], row["layer"], row["slot"]))
    ranked_slots = [(int(row["layer"]), int(row["slot"])) for row in ranked]
    universe = [(layer, slot) for layer in range(3) for slot in range(32)]
    random_a = universe.copy(); random.Random(766901).shuffle(random_a)
    random_b = universe.copy(); random.Random(766902).shuffle(random_b)
    return {"ranked": ranked_slots, "random_a": random_a, "random_b": random_b}


def conditions(ladders: dict) -> dict:
    result = {"intact": []}
    for dose in DOSES:
        result[f"ranked_top{dose:02d}_ablate"] = ladders["ranked"][:dose]
        result[f"random_a_{dose:02d}_ablate"] = ladders["random_a"][:dose]
        result[f"random_b_{dose:02d}_ablate"] = ladders["random_b"][:dose]
    return result


def build(device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, LENGTH, "rope", True).to(device).eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=False)["model"])
    return model


def configure(model, targets: list[tuple[int, int]]) -> None:
    by_layer = {layer: [] for layer in range(3)}
    for layer, slot in targets:
        by_layer[layer].append(slot)
    for layer, block in enumerate(model.blocks):
        block.memory_read_keep_slots = None
        block.memory_read_slot_scales = None
        block.memory_read_topk = None
        block.memory_read_ablate_slots = tuple(sorted(by_layer[layer])) if by_layer[layer] else None


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
def evaluate(model, window_name: str, condition: str, targets: list[tuple[int, int]],
             device: torch.device, dtype: torch.dtype) -> dict:
    configure(model, targets)
    correctness, probabilities, margins = [], [], []
    for sample in range(SAMPLES):
        example_seed = 766900000 + (0 if window_name == "near" else 1000) + sample
        tokens, target = make_example(WINDOWS[window_name], sample, example_seed, device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(tokens)[..., :16][:, -1].float()
        target_index = int(target.item())
        correctness.append(int(logits.argmax(-1).item() == target_index))
        probabilities.append(float(logits.softmax(-1)[0, target_index].cpu()))
        competitor = logits[0].clone(); competitor[target_index] = -torch.inf
        margins.append(float((logits[0, target_index] - competitor.max()).cpu()))
        if (sample + 1) % 24 == 0:
            print(f"window={window_name} condition={condition} sample={sample + 1}/{SAMPLES}", flush=True)
    return {"seed": SEED, "window": window_name, "condition": condition,
            "ablated_layer_slots": [list(item) for item in targets], "samples": SAMPLES,
            "correct": sum(correctness), "accuracy": sum(correctness) / SAMPLES,
            "correctness": correctness, "target_probabilities": probabilities, "margins": margins,
            "mean_target_probability": sum(probabilities) / SAMPLES,
            "mean_margin": sum(margins) / SAMPLES}


def paired_exact(treatment: list[int], control: list[int]) -> dict:
    improved = sum(a == 1 and b == 0 for a, b in zip(treatment, control))
    harmed = sum(a == 0 and b == 1 for a, b in zip(treatment, control))
    discordant = improved + harmed
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(improved, harmed) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {"accuracy_difference": (sum(treatment) - sum(control)) / len(control),
            "improved": improved, "harmed": harmed, "ties": len(control) - discordant,
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
    parser.add_argument("--samples-per-window", type=int, default=SAMPLES)
    parser.add_argument("--output", default="experiments/level7_6_6_9/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ladders = frozen_ladders()
    condition_map = conditions(ladders)
    protocol = {"variant": "ist-full", "seed": SEED, "length": LENGTH, "windows": WINDOWS,
                "samples_per_window": args.samples_per_window, "balanced_targets": True,
                "doses": DOSES, "conditions": condition_map,
                "ranking_source": "level7_6_6_5 validation accuracy only",
                "random_ladder_seeds": (766901, 766902), "nested_deletion_ladders": True,
                "primary_family": "five ranked-dose vs intact contrasts with Holm correction",
                "collapse_threshold": "smallest ranked dose with negative Holm-significant effect"}
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
        for condition, targets in condition_map.items():
            output = root / f"{window_name}_{condition}.json"
            if output.exists() and not args.force:
                row = json.loads(output.read_text(encoding="utf-8"))
            else:
                row = evaluate(model, window_name, condition, targets, device, dtype)
                atomic_save(output, row)
            runs.append(row); atomic_save(root / "runs.partial.json", runs)
            print(f"window={window_name} condition={condition} accuracy={row['accuracy']:.2%}", flush=True)
    aggregate, vs_intact, ranked_vs_random = [], [], []
    for window_scope in ("near", "far", "combined"):
        selected_windows = tuple(WINDOWS) if window_scope == "combined" else (window_scope,)
        keyed = {}
        for condition in condition_map:
            selected = [run for run in runs if run["window"] in selected_windows and run["condition"] == condition]
            values = [value for run in selected for value in run["correctness"]]
            probabilities = [value for run in selected for value in run["target_probabilities"]]
            margins = [value for run in selected for value in run["margins"]]
            keyed[condition] = (values, probabilities, margins)
            aggregate.append({"window": window_scope, "condition": condition,
                              "correct": sum(values), "samples": len(values),
                              "accuracy": sum(values) / len(values),
                              "mean_target_probability": sum(probabilities) / len(probabilities),
                              "mean_margin": sum(margins) / len(margins)})
        intact, intact_p, intact_m = keyed["intact"]
        for dose in DOSES:
            ranked_name = f"ranked_top{dose:02d}_ablate"
            ranked, ranked_p, ranked_m = keyed[ranked_name]
            vs_intact.append({"window": window_scope, "dose": dose, "condition": ranked_name,
                              **paired_exact(ranked, intact),
                              "mean_target_probability_change": sum(a-b for a,b in zip(ranked_p,intact_p))/len(ranked_p),
                              "mean_margin_change": sum(a-b for a,b in zip(ranked_m,intact_m))/len(ranked_m)})
            for ladder in ("random_a", "random_b"):
                random_name = f"{ladder}_{dose:02d}_ablate"
                random_values, _, _ = keyed[random_name]
                ranked_vs_random.append({"window": window_scope, "dose": dose,
                                         "random_ladder": ladder,
                                         **paired_exact(ranked, random_values)})
    for window_scope in ("near", "far", "combined"):
        holm([row for row in vs_intact if row["window"] == window_scope])
        for ladder in ("random_a", "random_b"):
            holm([row for row in ranked_vs_random if row["window"] == window_scope and row["random_ladder"] == ladder])
    combined_ranked = [row for row in vs_intact if row["window"] == "combined"]
    significant = [row for row in combined_ranked if row["holm_significant"] and row["accuracy_difference"] < 0]
    threshold = min((row["dose"] for row in significant), default=None)
    result = {"protocol": protocol, "collapse_threshold": threshold,
              "ranked_vs_intact": vs_intact, "ranked_vs_random": ranked_vs_random,
              "aggregate": aggregate, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"collapse_threshold": threshold,
                      "combined_ranked_vs_intact": combined_ranked,
                      "combined_ranked_vs_random": [row for row in ranked_vs_random if row["window"] == "combined"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
