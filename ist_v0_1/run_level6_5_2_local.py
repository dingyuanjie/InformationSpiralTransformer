import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from run_level6_5_1_local import run, save


DEFAULT_LRS = [5e-5, 1e-4]
DEFAULT_DATA_SEEDS = [71313, 42042, 22026, 70007, 51234]


def main():
    parser = argparse.ArgumentParser(description="Level 6.5.2 cross-stream LR confirmation")
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=DEFAULT_DATA_SEEDS)
    parser.add_argument("--checkpoint", default="experiments/level6_5/deterministic/hard400_seed313/stage3.pt")
    parser.add_argument("--model-seed", type=int, default=313)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--maintenance-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--output", default="experiments/level6_5_2/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

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
    for data_seed in args.data_seeds:
        args.data_seed = data_seed
        stream_root = root / f"stream_seed{data_seed}"
        stream_root.mkdir(parents=True, exist_ok=True)
        for lr in args.lrs:
            result = run(lr, args, device, dtype, stream_root)
            results.append(result)
            save(root / "runs.partial.json", results)
            torch.cuda.empty_cache()

    summaries = []
    for lr in args.lrs:
        selected = [result for result in results if result["lr"] == lr]
        summaries.append(
            {
                "lr": lr,
                "successes": sum(result["passed"] for result in selected),
                "runs": len(selected),
                "success_rate": statistics.mean(result["passed"] for result in selected),
                "mean_best_query": statistics.mean(result["best_query"] for result in selected),
                "mean_final_query": statistics.mean(
                    result["final_maintenance"]["query"] for result in selected
                ),
                "worst_final_query": min(
                    result["final_maintenance"]["query"] for result in selected
                ),
                "worst_tail_query": min(
                    result["maintenance_tail_min_query"] for result in selected
                ),
                "mean_seconds": statistics.mean(result["seconds"] for result in selected),
            }
        )
    protocol = vars(args).copy()
    protocol["data_seeds"] = args.data_seeds
    protocol.pop("data_seed", None)
    save(root / "summary.json", {"protocol": protocol, "summary": summaries, "runs": results})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
