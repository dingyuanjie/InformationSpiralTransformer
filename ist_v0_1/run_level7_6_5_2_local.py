"""Level 7.6.5.2: controlled 1024-8192 latency and memory scaling audit."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from run_level7_1_local import atomic_save
from run_level7_6_local import build, matched_width, params


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
VARIANTS = ("transformer-matched", "ist-full", "ist-stable")
SEED = 313
LENGTHS = (1024, 2048, 4096, 8192)
MEASURED = {1024: 120, 2048: 80, 4096: 50, 8192: 30}


def load_model(variant: str, width: int, device: torch.device):
    path = PARENT / f"{variant}_seed{SEED}" / "stage_4096.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model = build(variant, width).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model


def fixed_inputs(length: int, count: int, repeat: int, device: torch.device) -> list[torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(765520000 + length * 10 + repeat)
    return [torch.randint(19, (1, length), generator=generator, device=device) for _ in range(count)]


@torch.no_grad()
def benchmark(model, variant: str, length: int, repeat: int, warmup: int,
              device: torch.device, dtype: torch.dtype) -> dict:
    measured = MEASURED[length]
    inputs = fixed_inputs(length, measured, repeat, device)
    for index in range(warmup):
        with torch.autocast(device_type="cuda", dtype=dtype):
            model(inputs[index % measured])
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    for tokens in inputs:
        with torch.autocast(device_type="cuda", dtype=dtype):
            model(tokens)
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    cuda_seconds = start_event.elapsed_time(end_event) / 1000
    return {"variant": variant, "length": length, "repeat": repeat, "warmup": warmup,
            "measured": measured, "wall_seconds": wall_seconds, "cuda_seconds": cuda_seconds,
            "wall_latency_ms": wall_seconds * 1000 / measured,
            "cuda_latency_ms": cuda_seconds * 1000 / measured,
            "wall_tokens_per_second": measured * length / wall_seconds,
            "cuda_tokens_per_second": measured * length / cuda_seconds,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576}


def slope(points: list[tuple[float, float]]) -> float:
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum((x - x_mean) ** 2 for x in xs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="experiments/level7_6_5_2/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variants": VARIANTS, "seed": SEED, "lengths": LENGTHS,
                "measured_forwards": MEASURED, "warmup": args.warmup,
                "repeats": args.repeats, "batch_size": 1, "dtype": "bf16-if-supported",
                "fixed_replay": True, "source_checkpoint": "level7_6_4/stage_4096.pt"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if (args.warmup, args.repeats) != (10, 3):
        raise ValueError("Formal protocol locks warmup=10 and repeats=3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target = params(build("ist-full", 96))
    width, matched_parameters = matched_width(target)
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {**protocol, "gpu": torch.cuda.get_device_name(device),
                                           "torch": torch.__version__, "dtype_used": str(dtype),
                                           "matched_width": width,
                                           "matched_transformer_parameters": matched_parameters})
    rows = []
    for variant in VARIANTS:
        model = load_model(variant, width, device)
        for length in LENGTHS:
            for repeat in range(args.repeats):
                output = root / f"{variant}_length{length}_repeat{repeat}.json"
                if output.exists() and not args.force:
                    row = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row = benchmark(model, variant, length, repeat, args.warmup, device, dtype)
                    atomic_save(output, row)
                rows.append(row)
                atomic_save(root / "runs.partial.json", rows)
                print(f"{variant} length={length} repeat={repeat} latency={row['wall_latency_ms']:.3f}ms "
                      f"speed={row['wall_tokens_per_second']:.0f} tok/s memory={row['peak_memory_mb']:.1f}MB",
                      flush=True)
        del model
        torch.cuda.empty_cache()

    curves = []
    for variant in VARIANTS:
        points = []
        for length in LENGTHS:
            selected = [row for row in rows if row["variant"] == variant and row["length"] == length]
            points.append({"length": length,
                           "median_latency_ms": statistics.median(row["wall_latency_ms"] for row in selected),
                           "median_tokens_per_second": statistics.median(row["wall_tokens_per_second"] for row in selected),
                           "median_peak_memory_mb": statistics.median(row["peak_memory_mb"] for row in selected),
                           "latency_cv": statistics.stdev(row["wall_latency_ms"] for row in selected) /
                                         statistics.mean(row["wall_latency_ms"] for row in selected)})
        curves.append({"variant": variant, "points": points,
                       "latency_scaling_exponent": slope([(p["length"], p["median_latency_ms"]) for p in points]),
                       "memory_scaling_exponent": slope([(p["length"], p["median_peak_memory_mb"]) for p in points])})
    by_variant = {curve["variant"]: curve for curve in curves}
    ratios = []
    for index, length in enumerate(LENGTHS):
        transformer = by_variant["transformer-matched"]["points"][index]
        full = by_variant["ist-full"]["points"][index]
        ratios.append({"length": length,
                       "ist_full_speedup": transformer["median_latency_ms"] / full["median_latency_ms"],
                       "transformer_over_ist_full_memory": transformer["median_peak_memory_mb"] /
                                                           full["median_peak_memory_mb"]})
    result = {"protocol": protocol, "matched_width": width, "curves": curves,
              "lengthwise_ratios": ratios, "runs": rows}
    atomic_save(root / "result.json", result)
    print(json.dumps({"curves": curves, "lengthwise_ratios": ratios}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
