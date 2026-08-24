"""Level 7.6.5.3: randomized interleaved 8192 steady-state performance replication."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch

from run_level7_1_local import atomic_save
from run_level7_6_local import build, matched_width, params


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
VARIANTS = ("transformer-matched", "ist-full", "ist-stable")
SEED = 313
LENGTH = 8192


def load_model(variant: str, width: int, device: torch.device):
    checkpoint = PARENT / f"{variant}_seed{SEED}" / "stage_4096.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = build(variant, width).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model


def gpu_snapshot() -> dict | None:
    fields = ("temperature.gpu", "power.draw", "clocks.sm", "clocks.mem", "utilization.gpu")
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        values = [value.strip() for value in result.stdout.strip().split(",")]
        return {field: float(value) for field, value in zip(fields, values)}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@torch.no_grad()
def measure_block(model, inputs: list[torch.Tensor], variant: str, round_index: int,
                  device: torch.device, dtype: torch.dtype) -> dict:
    before = gpu_snapshot()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    for tokens in inputs:
        with torch.autocast(device_type="cuda", dtype=dtype):
            model(tokens)
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    cuda_seconds = start_event.elapsed_time(end_event) / 1000
    after = gpu_snapshot()
    return {"round": round_index, "variant": variant, "forwards": len(inputs),
            "wall_seconds": wall_seconds, "cuda_seconds": cuda_seconds,
            "wall_latency_ms": wall_seconds * 1000 / len(inputs),
            "cuda_latency_ms": cuda_seconds * 1000 / len(inputs),
            "wall_tokens_per_second": len(inputs) * LENGTH / wall_seconds,
            "cuda_tokens_per_second": len(inputs) * LENGTH / cuda_seconds,
            "gpu_before": before, "gpu_after": after}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--forwards-per-block", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="experiments/level7_6_5_3/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    schedule_rng = random.Random(76553)
    # Six permutations repeated twice give every model exactly four visits to
    # each position; shuffling their order removes monotonic warm/thermal bias.
    schedule = [list(order) for order in itertools.permutations(VARIANTS)] * 2
    schedule_rng.shuffle(schedule)
    protocol = {"variants": VARIANTS, "seed": SEED, "length": LENGTH,
                "rounds": args.rounds, "forwards_per_block": args.forwards_per_block,
                "warmup_per_model": args.warmup, "schedule_seed": 76553,
                "randomized_interleaved_schedule": schedule, "batch_size": 1}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if (args.rounds, args.forwards_per_block, args.warmup) != (12, 20, 10):
        raise ValueError("Formal protocol locks rounds=12, forwards-per-block=20, warmup=10")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target = params(build("ist-full", 96))
    width, matched_parameters = matched_width(target)
    models = {variant: load_model(variant, width, device) for variant in VARIANTS}
    generator = torch.Generator(device=device)
    generator.manual_seed(765530000)
    inputs = [torch.randint(19, (1, LENGTH), generator=generator, device=device)
              for _ in range(args.forwards_per_block)]
    for variant, model in models.items():
        for index in range(args.warmup):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                model(inputs[index % len(inputs)])
        torch.cuda.synchronize()
        print(f"warmed {variant}", flush=True)

    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"protocol": protocol, "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__, "dtype": str(dtype), "matched_width": width,
                "matched_transformer_parameters": matched_parameters}
    atomic_save(root / "protocol.json", metadata)
    rows = []
    for round_index, order in enumerate(schedule):
        for position, variant in enumerate(order):
            output = root / f"round{round_index:02d}_position{position}_{variant}.json"
            if output.exists() and not args.force:
                row = json.loads(output.read_text(encoding="utf-8"))
            else:
                row = measure_block(models[variant], inputs, variant, round_index, device, dtype)
                row["position"] = position
                atomic_save(output, row)
            rows.append(row)
            atomic_save(root / "runs.partial.json", rows)
            print(f"round={round_index} position={position} {variant} "
                  f"latency={row['wall_latency_ms']:.3f}ms speed={row['wall_tokens_per_second']:.0f} tok/s",
                  flush=True)

    summary = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        latencies = [row["wall_latency_ms"] for row in selected]
        speeds = [row["wall_tokens_per_second"] for row in selected]
        summary.append({"variant": variant, "blocks": len(selected),
                        "median_latency_ms": statistics.median(latencies),
                        "latency_iqr_ms": percentile(latencies, 0.75) - percentile(latencies, 0.25),
                        "latency_cv": statistics.stdev(latencies) / statistics.mean(latencies),
                        "median_tokens_per_second": statistics.median(speeds),
                        "wall_cuda_relative_difference": statistics.median(
                            abs(row["wall_seconds"] - row["cuda_seconds"]) / row["cuda_seconds"]
                            for row in selected),
                        "median_speed_by_position": {
                            str(position): statistics.median(row["wall_tokens_per_second"] for row in selected
                                                             if row["position"] == position)
                            for position in range(len(VARIANTS))}})
    keyed = {row["variant"]: row for row in summary}
    ratios = []
    for round_index in range(args.rounds):
        round_rows = {row["variant"]: row for row in rows if row["round"] == round_index}
        ratios.append({"round": round_index,
                       "ist_full_speedup": round_rows["transformer-matched"]["wall_latency_ms"] /
                                           round_rows["ist-full"]["wall_latency_ms"],
                       "ist_stable_speedup": round_rows["transformer-matched"]["wall_latency_ms"] /
                                             round_rows["ist-stable"]["wall_latency_ms"]})
    full_ratios = [row["ist_full_speedup"] for row in ratios]
    diagnosis = {"ist_full_speedup_median": statistics.median(full_ratios),
                 "ist_full_speedup_iqr": percentile(full_ratios, 0.75) - percentile(full_ratios, 0.25),
                 "ist_full_speedup_range": [min(full_ratios), max(full_ratios)],
                 "summary_speedup_from_medians": keyed["ist-full"]["median_tokens_per_second"] /
                                                   keyed["transformer-matched"]["median_tokens_per_second"]}
    atomic_save(root / "result.json", {**metadata, "summary": summary,
                                         "paired_round_ratios": ratios,
                                         "diagnosis": diagnosis, "runs": rows})
    print(json.dumps({"summary": summary, "diagnosis": diagnosis}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
