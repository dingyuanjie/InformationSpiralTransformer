"""Level 7.6.6.10: seed-2026 two-bank Memory redundancy and synergy analysis."""
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
LENGTH = 32768
SEED = 2026
WINDOWS = {"near": (16, 4095), "far": (16384, 32766)}
BANK_A = (31, 3, 5, 12)
BANK_B = (28, 4, 13, 24)
SAMPLES = 128
CONDITIONS = (
    "intact", "a_ablate", "b_ablate", "ab_ablate",
    "a_keep_only", "b_keep_only", "a_boost_2", "b_boost_2",
    "a_ablate_b_boost_2", "b_ablate_a_boost_2",
)


def build(device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, LENGTH, "rope", True).to(device).eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=False)["model"])
    return model


def configure(model, condition: str) -> None:
    for block in model.blocks:
        block.memory_read_keep_slots = None
        block.memory_read_ablate_slots = None
        block.memory_read_slot_scales = None
        block.memory_read_topk = None
    layer = model.blocks[2]
    if condition == "a_ablate": layer.memory_read_ablate_slots = BANK_A
    elif condition == "b_ablate": layer.memory_read_ablate_slots = BANK_B
    elif condition == "ab_ablate": layer.memory_read_ablate_slots = tuple(sorted(BANK_A + BANK_B))
    elif condition == "a_keep_only": layer.memory_read_keep_slots = BANK_A
    elif condition == "b_keep_only": layer.memory_read_keep_slots = BANK_B
    elif condition == "a_boost_2": layer.memory_read_slot_scales = {slot: 2.0 for slot in BANK_A}
    elif condition == "b_boost_2": layer.memory_read_slot_scales = {slot: 2.0 for slot in BANK_B}
    elif condition == "a_ablate_b_boost_2":
        layer.memory_read_ablate_slots = BANK_A
        layer.memory_read_slot_scales = {slot: 2.0 for slot in BANK_B}
    elif condition == "b_ablate_a_boost_2":
        layer.memory_read_ablate_slots = BANK_B
        layer.memory_read_slot_scales = {slot: 2.0 for slot in BANK_A}


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
        example_seed = 767000000 + (0 if window_name == "near" else 1000) + sample
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
    return {"seed": SEED, "window": window_name, "condition": condition,
            "samples": SAMPLES, "correct": sum(correctness),
            "accuracy": sum(correctness) / SAMPLES, "correctness": correctness,
            "target_probabilities": probabilities, "margins": margins,
            "mean_target_probability": sum(probabilities) / SAMPLES,
            "mean_margin": sum(margins) / SAMPLES}


def paired_exact(treatment: list[int], intact: list[int]) -> dict:
    improved = sum(a == 1 and b == 0 for a, b in zip(treatment, intact))
    harmed = sum(a == 0 and b == 1 for a, b in zip(treatment, intact))
    discordant = improved + harmed
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(improved, harmed) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    else: p_value = 1.0
    return {"accuracy_difference": (sum(treatment) - sum(intact)) / len(intact),
            "improved": improved, "harmed": harmed, "ties": len(intact) - discordant,
            "mcnemar_exact_p": p_value}


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["mcnemar_exact_p"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["mcnemar_exact_p"])
        running = max(running, adjusted); rows[index]["holm_p"] = running
        rows[index]["holm_significant"] = running < 0.05


def interaction(values: dict[str, list[int]], bootstrap_seed: int) -> dict:
    per_sample = [ab - a - b + intact for intact, a, b, ab in zip(
        values["intact"], values["a_ablate"], values["b_ablate"], values["ab_ablate"])]
    estimate = sum(per_sample) / len(per_sample)
    rng = random.Random(bootstrap_seed)
    boot = []
    for _ in range(10000):
        boot.append(sum(per_sample[rng.randrange(len(per_sample))] for _ in per_sample) / len(per_sample))
    boot.sort()
    return {"additive_interaction": estimate,
            "bootstrap95": [boot[249], boot[9749]],
            "definition": "AB - A - B + intact; negative means super-additive damage"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-window", type=int, default=SAMPLES)
    parser.add_argument("--output", default="experiments/level7_6_6_10/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "seed": SEED, "length": LENGTH, "windows": WINDOWS,
                "samples_per_window": args.samples_per_window, "balanced_targets": True,
                "bank_a": BANK_A, "bank_b": BANK_B, "conditions": CONDITIONS,
                "paired_examples": True,
                "primary_family": "A ablate, B ablate, A+B ablate vs intact with Holm correction",
                "interaction": "paired additive interaction with 10000 bootstrap resamples"}
    if args.dry_run: print(json.dumps(protocol, indent=2)); return 0
    if args.samples_per_window != SAMPLES:
        raise ValueError(f"Formal protocol locks --samples-per-window={SAMPLES}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol, "gpu": torch.cuda.get_device_name(device),
                                           "torch": torch.__version__, "dtype": str(dtype)})
    model = build(device); runs = []
    for window_name in WINDOWS:
        for condition in CONDITIONS:
            output = root / f"{window_name}_{condition}.json"
            if output.exists() and not args.force:
                row = json.loads(output.read_text(encoding="utf-8"))
            else:
                row = evaluate(model, window_name, condition, device, dtype); atomic_save(output, row)
            runs.append(row); atomic_save(root / "runs.partial.json", runs)
            print(f"window={window_name} condition={condition} accuracy={row['accuracy']:.2%}", flush=True)
    aggregate, comparisons, interactions = [], [], []
    for index, window_scope in enumerate(("near", "far", "combined")):
        selected_windows = tuple(WINDOWS) if window_scope == "combined" else (window_scope,)
        values = {}
        for condition in CONDITIONS:
            selected = [run for run in runs if run["window"] in selected_windows and run["condition"] == condition]
            correct = [value for run in selected for value in run["correctness"]]
            probabilities = [value for run in selected for value in run["target_probabilities"]]
            margins = [value for run in selected for value in run["margins"]]
            values[condition] = correct
            aggregate.append({"window": window_scope, "condition": condition,
                              "correct": sum(correct), "samples": len(correct),
                              "accuracy": sum(correct) / len(correct),
                              "mean_target_probability": sum(probabilities) / len(probabilities),
                              "mean_margin": sum(margins) / len(margins)})
        for condition in CONDITIONS[1:]:
            comparisons.append({"window": window_scope, "condition": condition,
                                **paired_exact(values[condition], values["intact"])})
        interactions.append({"window": window_scope, **interaction(values, 767010000 + index)})
    for window_scope in ("near", "far", "combined"):
        holm([row for row in comparisons if row["window"] == window_scope
              and row["condition"] in ("a_ablate", "b_ablate", "ab_ablate")])
    combined_primary = [row for row in comparisons if row["window"] == "combined"
                        and row["condition"] in ("a_ablate", "b_ablate", "ab_ablate")]
    result = {"protocol": protocol, "aggregate": aggregate, "comparisons": comparisons,
              "interactions": interactions, "combined_primary": combined_primary, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"combined_primary": combined_primary,
                      "combined_interaction": next(row for row in interactions if row["window"] == "combined"),
                      "combined_secondary": [row for row in comparisons if row["window"] == "combined"
                                             and row["condition"] not in ("a_ablate", "b_ablate", "ab_ablate")]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
