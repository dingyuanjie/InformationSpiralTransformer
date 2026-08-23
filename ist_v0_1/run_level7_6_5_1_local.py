"""Level 7.6.5.1: audit the Level 7.6.4/7.6.5 8192-token timing discrepancy."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from long_context_test import set_seed
from run_level7_1_local import atomic_save
from run_level7_6_local import build, matched_width, params


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
VARIANTS = ("transformer-matched", "ist-full", "ist-stable")
SEED = 313
LENGTH = 8192
WINDOW = (2048, 4095)


def make_example(generator: torch.Generator, device: torch.device) -> torch.Tensor:
    target = torch.randint(16, (1,), generator=generator, device=device)
    tokens = torch.randint(16, (1, LENGTH), generator=generator, device=device)
    distance = int(torch.randint(WINDOW[0], WINDOW[1] + 1, (1,), generator=generator, device=device).item())
    position = LENGTH - 2 - distance
    tokens[0, position] = 17
    tokens[0, position + 1] = target
    tokens[0, -2] = 18
    tokens[0, -1] = 16
    return tokens


def load_model(variant: str, width: int, device: torch.device):
    path = PARENT / f"{variant}_seed{SEED}" / "stage_4096.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model = build(variant, width).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model


@torch.no_grad()
def benchmark(model, variant: str, mode: str, repeat: int, warmup: int, measured: int,
              device: torch.device, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device=device)
    generator.manual_seed(765510000 + repeat)
    fixed = [make_example(generator, device) for _ in range(measured)]

    def input_for(index: int) -> torch.Tensor:
        if mode == "fixed_replay":
            return fixed[index % measured]
        if mode == "streaming_rng":
            return make_example(generator, device)
        if mode == "reseed_each_sample":
            set_seed(765520000 + repeat * measured + index)
            # This mirrors the historical per-sample seed path without changing the model.
            local = torch.Generator(device=device)
            local.manual_seed(765520000 + repeat * measured + index)
            return make_example(local, device)
        raise ValueError(mode)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    cold_input = input_for(0)
    cold_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=dtype):
        model(cold_input)
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - cold_started

    for index in range(warmup):
        with torch.autocast(device_type="cuda", dtype=dtype):
            model(input_for(index % measured))
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_started = time.perf_counter()
    start_event.record()
    for index in range(measured):
        with torch.autocast(device_type="cuda", dtype=dtype):
            model(input_for(index))
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    event_seconds = start_event.elapsed_time(end_event) / 1000
    return {"variant": variant, "mode": mode, "repeat": repeat, "warmup": warmup,
            "measured": measured, "cold_seconds": cold_seconds,
            "wall_seconds": wall_seconds, "cuda_event_seconds": event_seconds,
            "wall_tokens_per_second": measured * LENGTH / wall_seconds,
            "cuda_tokens_per_second": measured * LENGTH / event_seconds,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576}


def environment(device: torch.device, dtype: torch.dtype) -> dict:
    cudnn = getattr(torch.backends, "cudnn", None)
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)), "dtype": str(dtype),
            "cudnn_version": torch.backends.cudnn.version() if cudnn else None,
            "cudnn_benchmark": bool(cudnn.benchmark) if cudnn else None,
            "cudnn_deterministic": bool(cudnn.deterministic) if cudnn else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="experiments/level7_6_5_1/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variants": VARIANTS, "seed": SEED, "length": LENGTH, "window": WINDOW,
                "modes": ("fixed_replay", "streaming_rng", "reseed_each_sample"),
                "warmup": args.warmup, "measured": args.measured, "repeats": args.repeats}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if (args.warmup, args.measured, args.repeats) != (10, 30, 3):
        raise ValueError("Formal protocol locks warmup=10, measured=30, repeats=3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target = params(build("ist-full", 96))
    width, matched_parameters = matched_width(target)
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"protocol": protocol, "environment": environment(device, dtype),
                "matched_width": width, "matched_transformer_parameters": matched_parameters}
    atomic_save(root / "protocol.json", metadata)
    rows = []
    for variant in VARIANTS:
        model = load_model(variant, width, device)
        for mode in protocol["modes"]:
            for repeat in range(args.repeats):
                output = root / f"{variant}_{mode}_repeat{repeat}.json"
                if output.exists() and not args.force:
                    row = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row = benchmark(model, variant, mode, repeat, args.warmup, args.measured,
                                    device, dtype)
                    atomic_save(output, row)
                rows.append(row)
                atomic_save(root / "runs.partial.json", rows)
                print(f"{variant} mode={mode} repeat={repeat} cold={row['cold_seconds']:.3f}s "
                      f"wall={row['wall_tokens_per_second']:.0f} cuda={row['cuda_tokens_per_second']:.0f} tok/s",
                      flush=True)
        del model
        torch.cuda.empty_cache()

    summary = []
    for variant in VARIANTS:
        for mode in protocol["modes"]:
            selected = [row for row in rows if row["variant"] == variant and row["mode"] == mode]
            summary.append({"variant": variant, "mode": mode,
                            "median_cold_seconds": statistics.median(row["cold_seconds"] for row in selected),
                            "median_wall_tokens_per_second": statistics.median(row["wall_tokens_per_second"] for row in selected),
                            "median_cuda_tokens_per_second": statistics.median(row["cuda_tokens_per_second"] for row in selected),
                            "median_peak_memory_mb": statistics.median(row["peak_memory_mb"] for row in selected),
                            "wall_cv": statistics.stdev(row["wall_tokens_per_second"] for row in selected) /
                                       statistics.mean(row["wall_tokens_per_second"] for row in selected)})
    transformer = next(row for row in summary if row["variant"] == "transformer-matched" and row["mode"] == "fixed_replay")
    full = next(row for row in summary if row["variant"] == "ist-full" and row["mode"] == "fixed_replay")
    diagnosis = {"steady_state_speedup_ist_full_over_transformer":
                 full["median_wall_tokens_per_second"] / transformer["median_wall_tokens_per_second"],
                 "steady_state_memory_ratio_transformer_over_ist_full":
                 transformer["median_peak_memory_mb"] / full["median_peak_memory_mb"]}
    atomic_save(root / "result.json", {**metadata, "summary": summary, "diagnosis": diagnosis, "runs": rows})
    print(json.dumps({"summary": summary, "diagnosis": diagnosis}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
