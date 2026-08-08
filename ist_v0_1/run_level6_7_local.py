import argparse
import copy
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from long_context_test import set_seed
from run_level6_2_local import evaluate
from run_level6_6_local import build, checkpoint, curriculum, fixed_stage, random_step, save


FRESH_SEEDS = [101, 202, 303, 404, 505]


def update_ema(ema_model, model, ema_probe, probe, decay):
    with torch.no_grad():
        for ema_value, value in zip(ema_model.state_dict().values(), model.state_dict().values()):
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)
        for ema_value, value in zip(ema_probe.state_dict().values(), probe.state_dict().values()):
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)


def withdrawal_with_ema(model, probe, optimizer, args, device, dtype, folder, seed):
    evaluator = argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    ema_model = copy.deepcopy(model).eval()
    ema_probe = copy.deepcopy(probe).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    for parameter in ema_probe.parameters():
        parameter.requires_grad_(False)
    for group in optimizer.param_groups:
        group["lr"] = args.withdrawal_lr
    history = []
    schedule = [(0.2, 300), (0.1, 300), (0.0, args.maintenance_steps)]
    global_step = 0
    for phase, (weight, steps) in enumerate(schedule, 1):
        for parameter in probe.parameters():
            parameter.requires_grad_(weight > 0)
        for step in range(1, steps + 1):
            global_step += 1
            model.train(); probe.train(weight > 0)
            random_step(model, probe, optimizer, args, 16, 2, weight, device, dtype)
            update_ema(ema_model, model, ema_probe, probe, args.ema_decay)
            if step == 1 or step % args.eval_every == 0:
                raw = evaluate(model, probe, evaluator, 16, device, dtype)
                history.append({"phase": phase, "weight": weight, "step": step,
                                "global_step": global_step, "raw": raw})
                save(folder / "withdrawal_progress.json", history)
                print(
                    f"seed={seed} withdraw={weight} step={step} "
                    f"raw_query={raw['query']:.2%}", flush=True
                )
        checkpoint(folder / f"withdrawal_phase{phase}.pt", model, probe, optimizer,
                   {"withdrawal_history": history, "ema_model": ema_model.state_dict(),
                    "ema_probe": ema_probe.state_dict()})
    heldout_seed = seed + args.heldout_seed_offset
    set_seed(heldout_seed)
    raw_final = evaluate(model, probe, evaluator, 16, device, dtype, args.final_eval_batches)
    set_seed(heldout_seed)
    ema_final = evaluate(ema_model, ema_probe, evaluator, 16, device, dtype, args.final_eval_batches)
    return history, raw_final, ema_final, ema_model, ema_probe


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
        result = {"seed": seed, "passed": False, "raw_passed": False,
                  "failed_phase": "fixed", "fixed": fixed,
                  "seconds": time.perf_counter() - started}
        save(result_path, result); return result
    set_seed(seed + 20000)
    stages, curriculum_history = curriculum(model, probe, optimizer, args, device, dtype, folder, seed)
    curriculum_passed = len(stages) == 4 and all(stage["passed"] for stage in stages)
    if not curriculum_passed:
        result = {"seed": seed, "passed": False, "raw_passed": False,
                  "failed_phase": "curriculum", "fixed": fixed, "stages": stages,
                  "curriculum_history": curriculum_history,
                  "seconds": time.perf_counter() - started}
        save(result_path, result); return result
    history, raw_final, ema_final, ema_model, ema_probe = withdrawal_with_ema(
        model, probe, optimizer, args, device, dtype, folder, seed
    )
    raw_passed = raw_final["query"] >= 0.95 and raw_final["probe_min"] >= 0.90
    passed = ema_final["query"] >= 0.95 and ema_final["probe_min"] >= 0.90
    result = {"seed": seed, "passed": passed, "raw_passed": raw_passed,
              "failed_phase": None if passed else "withdrawal", "fixed": fixed,
              "stages": stages, "withdrawal_history": history,
              "raw_final": raw_final, "ema_final": ema_final,
              "seconds": time.perf_counter() - started}
    save(result_path, result)
    torch.save({"model": ema_model.state_dict(), "probe": ema_probe.state_dict()}, folder / "ema_final.pt")
    return result


def main():
    p = argparse.ArgumentParser(description="Level 6.7 unified robust formation protocol")
    p.add_argument("--seeds", nargs="+", type=int, default=FRESH_SEEDS)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--fixed-steps", type=int, default=2500)
    p.add_argument("--fixed-batch-size", type=int, default=16)
    p.add_argument("--fixed-eval-batch-size", type=int, default=16)
    p.add_argument("--random-stage1-steps", type=int, default=2500)
    p.add_argument("--later-steps", type=int, default=1500)
    p.add_argument("--probe-weight", type=float, default=0.5)
    p.add_argument("--stage4-lr", type=float, default=1e-5)
    p.add_argument("--withdrawal-lr", type=float, default=5e-6)
    p.add_argument("--maintenance-steps", type=int, default=750)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--heldout-seed-offset", type=int, default=950000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--final-eval-batches", type=int, default=50)
    p.add_argument("--output", default="experiments/level6_7/formal")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not 0.0 < args.ema_decay < 1.0: raise ValueError("ema-decay must be in (0, 1)")
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
    fixed = sum(result["fixed"]["passed"] for result in results)
    curriculum_count = sum(len(result.get("stages", [])) == 4 and
                           all(stage["passed"] for stage in result.get("stages", [])) for result in results)
    completed = [result for result in results if "ema_final" in result]
    summary = {"runs": len(results), "fixed_passes": fixed, "curriculum_passes": curriculum_count,
               "raw_successes": sum(result.get("raw_passed", False) for result in results),
               "ema_successes": sum(result["passed"] for result in results),
               "ema_success_rate": statistics.mean(result["passed"] for result in results),
               "mean_raw_query": statistics.mean(r["raw_final"]["query"] for r in completed) if completed else None,
               "mean_ema_query": statistics.mean(r["ema_final"]["query"] for r in completed) if completed else None}
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
