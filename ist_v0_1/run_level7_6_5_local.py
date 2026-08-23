"""Level 7.6.5: paired 8192-token stability and fair-efficiency replication."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from long_context_test import set_seed
from run_level7_1_local import atomic_save
from run_level7_6_local import SEEDS, build, matched_width, params


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
VARIANTS = ("transformer-matched", "ist-full", "ist-stable")
LENGTH = 8192
WINDOWS = (
    (16, 63), (64, 127), (128, 255), (256, 511), (512, 1023),
    (1024, 2047), (2048, 4095), (4096, 6143), (6144, 8190),
)
CHANCE = 1 / 16


def wilson(correct: int, samples: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / samples
    scale = 1 + z * z / samples
    middle = (p + z * z / (2 * samples)) / scale
    half = z * math.sqrt(p * (1 - p) / samples + z * z / (4 * samples * samples)) / scale
    return [middle - half, middle + half]


def paired_exact(left: list[int], right: list[int]) -> dict:
    improved = sum(a == 1 and b == 0 for a, b in zip(left, right))
    harmed = sum(a == 0 and b == 1 for a, b in zip(left, right))
    discordant = improved + harmed
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(improved, harmed) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    return {"difference": (sum(left) - sum(right)) / len(left), "improved": improved,
            "harmed": harmed, "ties": len(left) - discordant, "mcnemar_exact_p": p_value}


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["mcnemar_exact_p"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["mcnemar_exact_p"])
        running = max(running, adjusted)
        rows[index]["holm_p"] = running
        rows[index]["holm_significant"] = running < 0.05


def make_example(seed: int, window: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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


@torch.no_grad()
def evaluate(model, seed: int, window_index: int, samples: int, device: torch.device,
             dtype: torch.dtype) -> dict:
    model.eval()
    correctness = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for sample in range(samples):
        # Identical seed mapping across variants makes every comparison paired.
        example_seed = 765500000 + seed * 10000 + window_index * samples + sample
        tokens, target = make_example(example_seed, WINDOWS[window_index], device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            prediction = model(tokens)[..., :16][:, -1].argmax(-1)
        correctness.append(int(prediction.item() == target.item()))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    correct = sum(correctness)
    return {"window": list(WINDOWS[window_index]), "samples": samples, "correct": correct,
            "accuracy": correct / samples, "wilson95": wilson(correct, samples),
            "correctness": correctness, "seconds": seconds,
            "tokens_per_second": samples * LENGTH / seconds,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576}


def load_model(variant: str, seed: int, width: int, device: torch.device):
    checkpoint = PARENT / f"{variant}_seed{seed}" / "stage_4096.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Level 7.6.4 checkpoint missing: {checkpoint}")
    model = build(variant, width).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-seed-window", type=int, default=100)
    parser.add_argument("--output", default="experiments/level7_6_5/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variants": VARIANTS, "seeds": SEEDS, "length": LENGTH, "windows": WINDOWS,
                "samples_per_seed_window": args.samples_per_seed_window,
                "paired_examples": True, "chance_accuracy": CHANCE,
                "source_checkpoints": "level7_6_4/stage_4096.pt"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if args.samples_per_seed_window != 100:
        raise ValueError("Formal protocol locks --samples-per-seed-window=100")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target = params(build("ist-full", 96))
    width, matched_parameters = matched_width(target)
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {**protocol, "matched_width": width,
                                           "matched_transformer_parameters": matched_parameters})
    runs = []
    for variant in VARIANTS:
        for seed in SEEDS:
            model = load_model(variant, seed, width, device)
            tests = []
            for index, window in enumerate(WINDOWS):
                output = root / f"{variant}_seed{seed}_window{index}.json"
                if output.exists() and not args.force:
                    row = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row = evaluate(model, seed, index, args.samples_per_seed_window, device, dtype)
                    atomic_save(output, row)
                tests.append(row)
                print(f"{variant} seed={seed} window={window} accuracy={row['accuracy']:.2%} "
                      f"speed={row['tokens_per_second']:.0f} tok/s", flush=True)
            runs.append({"variant": variant, "seed": seed, "tests": tests})
            atomic_save(root / "runs.partial.json", runs)
            del model
            torch.cuda.empty_cache()

    keyed = {(run["variant"], run["seed"]): run for run in runs}
    summaries = []
    comparisons = []
    seed_stability = []
    for variant in VARIANTS:
        seed_accuracies = []
        for seed in SEEDS:
            rows = keyed[(variant, seed)]["tests"]
            correct = sum(row["correct"] for row in rows)
            samples = sum(row["samples"] for row in rows)
            interval = wilson(correct, samples)
            seed_accuracies.append(correct / samples)
            seed_stability.append({"variant": variant, "seed": seed, "correct": correct,
                                   "samples": samples, "accuracy": correct / samples,
                                   "wilson95": interval, "above_chance": interval[0] > CHANCE})
        all_rows = [row for seed in SEEDS for row in keyed[(variant, seed)]["tests"]]
        correct = sum(row["correct"] for row in all_rows)
        samples = sum(row["samples"] for row in all_rows)
        summaries.append({"variant": variant, "correct": correct, "samples": samples,
                          "accuracy": correct / samples, "wilson95": wilson(correct, samples),
                          "seed_accuracy_mean": statistics.mean(seed_accuracies),
                          "seed_accuracy_stdev": statistics.stdev(seed_accuracies),
                          "mean_tokens_per_second": statistics.mean(row["tokens_per_second"] for row in all_rows),
                          "mean_peak_memory_mb": statistics.mean(row["peak_memory_mb"] for row in all_rows)})
    for variant in ("ist-full", "ist-stable"):
        for index, window in enumerate(WINDOWS):
            left = [value for seed in SEEDS for value in keyed[(variant, seed)]["tests"][index]["correctness"]]
            right = [value for seed in SEEDS for value in keyed[("transformer-matched", seed)]["tests"][index]["correctness"]]
            comparisons.append({"variant": variant, "window": list(window), **paired_exact(left, right)})
        holm([row for row in comparisons if row["variant"] == variant])
    result = {"protocol": protocol, "matched_width": width, "summary": summaries,
              "seed_stability": seed_stability, "paired_window_comparisons": comparisons,
              "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"summary": summaries, "seed_stability": seed_stability}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
