import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from run_level6_5_local import run, save


DEFAULT_MODEL_SEEDS = [313, 42, 2026, 7, 1234]


def main():
    parser = argparse.ArgumentParser(description="Level 6.5.4 independent-initialization confirmation")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_MODEL_SEEDS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--stage1-steps", type=int, default=3000)
    parser.add_argument("--later-steps", type=int, default=1000)
    parser.add_argument("--maintenance-steps", type=int, default=500)
    parser.add_argument("--stage4-lr", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--output", default="experiments/level6_5_4/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.profiles = ["hard400"]
    args.allow_nondeterministic = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        results.append(run("hard400", seed, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()

    reached = {
        str(chunks): sum(
            any(stage["chunks"] == chunks and stage["passed"] for stage in result["stages"])
            for result in results
        )
        for chunks in [2, 4, 8, 16]
    }
    completed = [result for result in results if result["maintenance"] is not None]
    summary = {
        "runs": len(results),
        "pipeline_successes": sum(result["passed"] for result in results),
        "pipeline_success_rate": statistics.mean(result["passed"] for result in results),
        "stage_pass_counts": reached,
        "maintenance_runs": len(completed),
        "mean_maintenance_query": (
            statistics.mean(result["maintenance"]["query"] for result in completed)
            if completed
            else None
        ),
        "worst_maintenance_query": (
            min(result["maintenance"]["query"] for result in completed)
            if completed
            else None
        ),
    }
    protocol = vars(args).copy()
    save(root / "summary.json", {"protocol": protocol, "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
