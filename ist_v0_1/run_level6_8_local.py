import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from long_context_test import set_seed
from run_level6_6_local import build, curriculum, fixed_stage, save, withdrawal


FRESH_SEEDS = [606, 707, 808, 909, 1001]


def run_seed(seed, args, device, dtype, root):
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    set_seed(seed)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()), lr=1e-3)
    started = time.perf_counter()
    fixed = fixed_stage(model, probe, optimizer, args, device, dtype, folder, seed)
    if not fixed["passed"]:
        result = {"seed": seed, "passed": False, "probe_diagnostic_passed": False,
                  "failed_phase": "fixed", "fixed": fixed,
                  "seconds": time.perf_counter() - started}
        save(result_path, result); return result
    set_seed(seed + 20000)
    stages, history = curriculum(model, probe, optimizer, args, device, dtype, folder, seed)
    curriculum_passed = len(stages) == 4 and all(stage["passed"] for stage in stages)
    if not curriculum_passed:
        result = {"seed": seed, "passed": False, "probe_diagnostic_passed": False,
                  "failed_phase": "curriculum", "fixed": fixed, "stages": stages,
                  "curriculum_history": history, "seconds": time.perf_counter() - started}
        save(result_path, result); return result
    withdrawal_history, final = withdrawal(model, probe, optimizer, args, device, dtype, folder, seed)
    passed = final["query"] >= 0.95
    probe_passed = final["probe_min"] >= 0.90
    result = {"seed": seed, "passed": passed, "probe_diagnostic_passed": probe_passed,
              "failed_phase": None if passed else "withdrawal", "fixed": fixed,
              "stages": stages, "withdrawal_history": withdrawal_history,
              "final": final, "seconds": time.perf_counter() - started}
    save(result_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description="Level 6.8 behavior-first unified protocol")
    p.add_argument("--seeds", nargs="+", type=int, default=FRESH_SEEDS)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--fixed-steps", type=int, default=2500)
    p.add_argument("--fixed-batch-size", type=int, default=16)
    p.add_argument("--fixed-eval-batch-size", type=int, default=16)
    p.add_argument("--random-stage1-steps", type=int, default=2500)
    p.add_argument("--later-steps", type=int, default=1500)
    p.add_argument("--probe-weight", type=float, default=0.5)
    p.add_argument("--stage3-lr", type=float, default=5e-5)
    p.add_argument("--stage4-lr", type=float, default=1e-5)
    p.add_argument("--withdrawal-lr", type=float, default=5e-6)
    p.add_argument("--maintenance-steps", type=int, default=750)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--final-eval-batches", type=int, default=50)
    p.add_argument("--output", default="experiments/level6_8/formal")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.behavior_only_gate = True
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda"); dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True); results = []
    for seed in args.seeds:
        results.append(run_seed(seed, args, device, dtype, root)); save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    stage_counts = {str(count): sum(any(stage["chunks"] == count and stage["passed"]
                                          for stage in result.get("stages", [])) for result in results)
                    for count in [2, 4, 8, 16]}
    completed = [result for result in results if "final" in result]
    summary = {"runs": len(results), "behavior_successes": sum(r["passed"] for r in results),
               "behavior_success_rate": statistics.mean(r["passed"] for r in results),
               "probe_diagnostic_successes": sum(r.get("probe_diagnostic_passed", False) for r in results),
               "stage_pass_counts": stage_counts,
               "mean_final_query": statistics.mean(r["final"]["query"] for r in completed) if completed else None,
               "worst_final_query": min(r["final"]["query"] for r in completed) if completed else None,
               "mean_final_probe_min": statistics.mean(r["final"]["probe_min"] for r in completed) if completed else None}
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
