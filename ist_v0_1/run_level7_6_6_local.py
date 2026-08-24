"""Level 7.6.6: frozen 16K/32K extrapolation, accuracy, OOM and efficiency boundary."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from baseline_transformer import StandardTransformer
from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level7_1_local import atomic_save
from run_level7_6_local import SEEDS, matched_width, params


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
VARIANTS = ("transformer-matched", "ist-full", "ist-stable")
LENGTHS = (16384, 32768)
MAX_LENGTH = max(LENGTHS)
SAMPLES = 50
CHANCE = 1 / 16


def windows(length: int) -> tuple[tuple[int, int], ...]:
    return ((16, length // 8 - 1), (length // 8, length // 4 - 1),
            (length // 4, length // 2 - 1), (length // 2, length - 2))


def build_extended(variant: str, matched: int):
    if variant == "transformer-matched":
        return StandardTransformer(19, matched, 3, 8, MAX_LENGTH, 0.0, "rope")
    model = InformationSpiralTransformer(19, 64, 3, MAX_LENGTH, "rope", True)
    if variant == "ist-stable":
        model.blocks[2].memory.slot_queries.requires_grad_(False)
    return model


def wilson(correct: int, samples: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / samples
    scale = 1 + z * z / samples
    middle = (p + z * z / (2 * samples)) / scale
    half = z * math.sqrt(p * (1 - p) / samples + z * z / (4 * samples * samples)) / scale
    return [middle - half, middle + half]


def make_example(length: int, window: tuple[int, int], seed: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    target = torch.randint(16, (1,), device=device)
    tokens = torch.randint(16, (1, length), device=device)
    distance = int(torch.randint(window[0], window[1] + 1, (1,), device=device).item())
    position = length - 2 - distance
    tokens[0, position] = 17
    tokens[0, position + 1] = target
    tokens[0, -2] = 18
    tokens[0, -1] = 16
    return tokens, target


@torch.no_grad()
def evaluate(model, variant: str, seed: int, length: int, window_index: int,
             device: torch.device, dtype: torch.dtype) -> dict:
    window = windows(length)[window_index]
    model.eval()
    correctness = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        for sample in range(SAMPLES):
            example_seed = 766000000 + seed * 100000 + length * 10 + window_index * SAMPLES + sample
            tokens, target = make_example(length, window, example_seed, device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                prediction = model(tokens)[..., :16][:, -1].argmax(-1)
            correctness.append(int(prediction.item() == target.item()))
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as error:
        del correctness[:]
        torch.cuda.empty_cache()
        return {"variant": variant, "seed": seed, "length": length, "window": list(window),
                "status": "oom", "error": str(error).splitlines()[0]}
    seconds = time.perf_counter() - started
    correct = sum(correctness)
    return {"variant": variant, "seed": seed, "length": length, "window": list(window),
            "status": "ok", "samples": SAMPLES, "correct": correct,
            "accuracy": correct / SAMPLES, "wilson95": wilson(correct, SAMPLES),
            "correctness": correctness, "seconds": seconds,
            "tokens_per_second": SAMPLES * length / seconds,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-seed-window", type=int, default=SAMPLES)
    parser.add_argument("--output", default="experiments/level7_6_6/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variants": VARIANTS, "seeds": SEEDS, "lengths": LENGTHS,
                "windows": {str(length): windows(length) for length in LENGTHS},
                "samples_per_seed_window": args.samples_per_seed_window,
                "frozen_zero_shot": True, "source": "level7_6_4/stage_4096.pt",
                "chance_accuracy": CHANCE}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if args.samples_per_seed_window != SAMPLES:
        raise ValueError(f"Formal protocol locks --samples-per-seed-window={SAMPLES}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target = params(InformationSpiralTransformer(19, 64, 3, MAX_LENGTH, "rope", True))
    width, matched_parameters = matched_width(target)
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"protocol": protocol, "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__, "dtype": str(dtype), "matched_width": width,
                "matched_transformer_parameters": matched_parameters}
    atomic_save(root / "protocol.json", metadata)
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            model = build_extended(variant, width).to(device)
            checkpoint = PARENT / f"{variant}_seed{seed}" / "stage_4096.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
            stop_variant_seed = False
            for length in LENGTHS:
                for window_index, window in enumerate(windows(length)):
                    output = root / f"{variant}_seed{seed}_length{length}_window{window_index}.json"
                    if output.exists() and not args.force:
                        row = json.loads(output.read_text(encoding="utf-8"))
                    elif stop_variant_seed:
                        row = {"variant": variant, "seed": seed, "length": length,
                               "window": list(window), "status": "skipped_after_oom"}
                        atomic_save(output, row)
                    else:
                        row = evaluate(model, variant, seed, length, window_index, device, dtype)
                        atomic_save(output, row)
                    rows.append(row)
                    atomic_save(root / "runs.partial.json", rows)
                    print(f"{variant} seed={seed} length={length} window={window} status={row['status']}"
                          + (f" accuracy={row['accuracy']:.2%}" if row["status"] == "ok" else ""), flush=True)
                    if row["status"] == "oom":
                        stop_variant_seed = True
            del model
            torch.cuda.empty_cache()

    summary = []
    for variant in VARIANTS:
        for length in LENGTHS:
            selected = [row for row in rows if row["variant"] == variant and row["length"] == length]
            valid = [row for row in selected if row["status"] == "ok"]
            correct = sum(row["correct"] for row in valid)
            samples = sum(row["samples"] for row in valid)
            summary.append({"variant": variant, "length": length,
                            "ok_windows": len(valid), "oom_windows": sum(row["status"] == "oom" for row in selected),
                            "skipped_windows": sum(row["status"] == "skipped_after_oom" for row in selected),
                            "correct": correct, "samples": samples,
                            "accuracy": correct / samples if samples else None,
                            "wilson95": wilson(correct, samples) if samples else None,
                            "mean_tokens_per_second": sum(row["tokens_per_second"] for row in valid) / len(valid) if valid else None,
                            "mean_peak_memory_mb": sum(row["peak_memory_mb"] for row in valid) / len(valid) if valid else None})
    result = {**metadata, "summary": summary, "runs": rows}
    atomic_save(root / "result.json", result)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
