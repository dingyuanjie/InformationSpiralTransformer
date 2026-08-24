"""Level 7.6.5.4: allocation-clean balanced 1024-8192 scaling replication."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from run_level7_1_local import atomic_save
from run_level7_6_local import build, matched_width, params


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
VARIANTS = ("transformer-matched", "ist-full", "ist-stable")
LENGTHS = (1024, 2048, 4096, 8192)
SEED = 313


def load_model(variant: str, width: int, device: torch.device):
    checkpoint = PARENT / f"{variant}_seed{SEED}" / "stage_4096.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = build(variant, width).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def slope(points: list[tuple[float, float]]) -> float:
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum((x - x_mean) ** 2 for x in xs)


@torch.no_grad()
def measure(model, tokens: torch.Tensor, variant: str, length: int, round_index: int,
            position: int, forwards: int, device: torch.device, dtype: torch.dtype) -> dict:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    for _ in range(forwards):
        with torch.autocast(device_type="cuda", dtype=dtype):
            model(tokens)
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    cuda_seconds = start_event.elapsed_time(end_event) / 1000
    return {"variant": variant, "length": length, "round": round_index, "position": position,
            "forwards": forwards, "wall_seconds": wall_seconds, "cuda_seconds": cuda_seconds,
            "wall_latency_ms": wall_seconds * 1000 / forwards,
            "cuda_latency_ms": cuda_seconds * 1000 / forwards,
            "wall_tokens_per_second": forwards * length / wall_seconds,
            "cuda_tokens_per_second": forwards * length / cuda_seconds}


@torch.no_grad()
def memory_probe(model, tokens: torch.Tensor, variant: str, length: int,
                 device: torch.device, dtype: torch.dtype) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=dtype):
        model(tokens)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {"variant": variant, "length": length, "baseline_memory_mb": baseline / 1048576,
            "peak_memory_mb": peak / 1048576, "incremental_peak_memory_mb": (peak - baseline) / 1048576}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--forwards-per-block", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="experiments/level7_6_5_4/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    conditions = [(variant, length) for variant in VARIANTS for length in LENGTHS]
    schedule_rng = random.Random(76554)
    base = conditions.copy()
    schedule_rng.shuffle(base)
    schedule = [base[offset:] + base[:offset] for offset in range(len(base))]
    protocol = {"variants": VARIANTS, "lengths": LENGTHS, "seed": SEED,
                "rounds": args.rounds, "forwards_per_block": args.forwards_per_block,
                "warmup_per_condition": args.warmup, "schedule_seed": 76554,
                "cyclic_position_balancing": True, "conditions_per_round": len(conditions),
                "no_empty_cache_inside_timing_phase": True}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if (args.rounds, args.forwards_per_block, args.warmup) != (12, 30, 10):
        raise ValueError("Formal protocol locks rounds=12, forwards-per-block=30, warmup=10")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target = params(build("ist-full", 96))
    width, matched_parameters = matched_width(target)
    models = {variant: load_model(variant, width, device) for variant in VARIANTS}
    generator = torch.Generator(device=device)
    generator.manual_seed(765540000)
    inputs = {length: torch.randint(19, (1, length), generator=generator, device=device) for length in LENGTHS}
    for variant, length in conditions:
        for _ in range(args.warmup):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                models[variant](inputs[length])
    torch.cuda.synchronize()

    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"protocol": protocol, "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__, "dtype": str(dtype), "matched_width": width,
                "matched_transformer_parameters": matched_parameters}
    atomic_save(root / "protocol.json", metadata)
    rows = []
    for round_index, order in enumerate(schedule):
        for position, (variant, length) in enumerate(order):
            output = root / f"round{round_index:02d}_position{position:02d}_{variant}_length{length}.json"
            if output.exists() and not args.force:
                row = json.loads(output.read_text(encoding="utf-8"))
            else:
                row = measure(models[variant], inputs[length], variant, length, round_index, position,
                              args.forwards_per_block, device, dtype)
                atomic_save(output, row)
            rows.append(row)
            atomic_save(root / "runs.partial.json", rows)
            print(f"round={round_index} pos={position} {variant} L={length} "
                  f"latency={row['wall_latency_ms']:.3f}ms speed={row['wall_tokens_per_second']:.0f} tok/s",
                  flush=True)

    memory = []
    for variant, length in conditions:
        output = root / f"memory_{variant}_length{length}.json"
        if output.exists() and not args.force:
            row = json.loads(output.read_text(encoding="utf-8"))
        else:
            row = memory_probe(models[variant], inputs[length], variant, length, device, dtype)
            atomic_save(output, row)
        memory.append(row)

    curves = []
    for variant in VARIANTS:
        points = []
        for length in LENGTHS:
            selected = [row for row in rows if row["variant"] == variant and row["length"] == length]
            latencies = [row["wall_latency_ms"] for row in selected]
            mem = next(row for row in memory if row["variant"] == variant and row["length"] == length)
            points.append({"length": length, "blocks": len(selected),
                           "median_latency_ms": statistics.median(latencies),
                           "latency_iqr_ms": percentile(latencies, 0.75) - percentile(latencies, 0.25),
                           "latency_cv": statistics.stdev(latencies) / statistics.mean(latencies),
                           "median_tokens_per_second": statistics.median(row["wall_tokens_per_second"] for row in selected),
                           **{key: mem[key] for key in ("peak_memory_mb", "incremental_peak_memory_mb")}})
        curves.append({"variant": variant, "points": points,
                       "latency_scaling_exponent": slope([(p["length"], p["median_latency_ms"]) for p in points]),
                       "incremental_memory_scaling_exponent": slope(
                           [(p["length"], p["incremental_peak_memory_mb"]) for p in points])})
    keyed = {curve["variant"]: curve for curve in curves}
    ratios = []
    for index, length in enumerate(LENGTHS):
        transformer = keyed["transformer-matched"]["points"][index]
        full = keyed["ist-full"]["points"][index]
        paired = []
        for round_index in range(args.rounds):
            t = next(row for row in rows if row["round"] == round_index and row["variant"] == "transformer-matched" and row["length"] == length)
            i = next(row for row in rows if row["round"] == round_index and row["variant"] == "ist-full" and row["length"] == length)
            paired.append(t["wall_latency_ms"] / i["wall_latency_ms"])
        ratios.append({"length": length, "speedup_from_medians": transformer["median_latency_ms"] / full["median_latency_ms"],
                       "paired_speedup_median": statistics.median(paired),
                       "paired_speedup_iqr": percentile(paired, 0.75) - percentile(paired, 0.25),
                       "incremental_memory_ratio": transformer["incremental_peak_memory_mb"] /
                                                   full["incremental_peak_memory_mb"]})
    result = {**metadata, "curves": curves, "lengthwise_ratios": ratios,
              "memory_probes": memory, "runs": rows}
    atomic_save(root / "result.json", result)
    print(json.dumps({"curves": curves, "lengthwise_ratios": ratios}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
