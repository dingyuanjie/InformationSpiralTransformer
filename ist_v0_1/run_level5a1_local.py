import argparse
import json
import statistics
from pathlib import Path

import torch

from run_level5a_local import VARIANTS, run_one, save_json


def main():
    parser = argparse.ArgumentParser(description="Level 5A.1 five-seed stability confirmation")
    parser.add_argument("--variants", nargs="+", choices=["transformer", "ist-c"],
                        default=["transformer", "ist-c"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[313, 42, 2026, 7, 1234])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--minimum-stage-steps", type=int, default=500)
    parser.add_argument("--stage1-max-steps", type=int, default=3000)
    parser.add_argument("--later-max-steps", type=int, default=1000)
    parser.add_argument("--gate-accuracy", type=float, default=0.90)
    parser.add_argument("--consecutive-passes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="experiments/level5a1/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Install CUDA-enabled PyTorch.")
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    print(f"GPU={torch.cuda.get_device_name(0)} AMP={amp_dtype}", flush=True)
    results = []
    for name in args.variants:
        for seed in args.seeds:
            results.append(run_one(name, VARIANTS[name], seed, args, device, amp_dtype, root))
            save_json(root / "runs.partial.json", results)
            torch.cuda.empty_cache()
    summary = []
    for name in args.variants:
        selected = [item for item in results if item["variant"] == name]
        completed = [item for item in selected if item["completed_all_stages"]]
        stage1_steps = [item["stages"][0]["steps"] for item in selected]
        summary.append({
            "variant": name,
            "parameters": selected[0]["parameters"],
            "successes": len(completed),
            "runs": len(selected),
            "success_rate": len(completed) / len(selected),
            "mean_stage1_steps": statistics.mean(stage1_steps),
            "std_stage1_steps": statistics.stdev(stage1_steps) if len(stage1_steps) > 1 else 0.0,
            "mean_seconds": statistics.mean(item["seconds"] for item in selected),
            "mean_peak_memory_mb": statistics.mean(item["peak_memory_mb"] for item in selected),
        })
    payload = {"protocol": vars(args), "summary": summary, "runs": results}
    save_json(root / "summary.json", payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DONE: {root / 'summary.json'}")


if __name__ == "__main__":
    main()
