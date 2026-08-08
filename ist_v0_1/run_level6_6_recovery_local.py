import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from run_level6_2_local import evaluate
from run_level6_6_local import build, checkpoint, random_step, restore, save


CASES = ["seed7_budget", "seed42_lr", "seed2026_withdrawal"]


def eval_args(args):
    return argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )


def stage_passed(metric):
    return metric["query"] >= 0.95 and metric["probe_min"] >= 0.90


def train_stage(model, probe, optimizer, args, device, dtype, folder, label,
                count, steps, lr, batch, initial_metric=None):
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = lr
    history = []
    consecutive = 1 if initial_metric is not None and stage_passed(initial_metric) else 0
    metric = initial_metric
    if initial_metric is not None:
        history.append({"step": 0, **initial_metric})
    for step in range(1, steps + 1):
        model.train(); probe.train()
        random_step(model, probe, optimizer, args, count, batch, args.probe_weight, device, dtype)
        if step == 1 or step % args.eval_every == 0:
            metric = evaluate(model, probe, eval_args(args), count, device, dtype)
            history.append({"step": step, **metric})
            save(folder / f"{label}_progress.json", history)
            consecutive = consecutive + 1 if stage_passed(metric) else 0
            print(
                f"{label} chunks={count} step={step} query={metric['query']:.2%} "
                f"probe={metric['probe_min']:.2%}", flush=True
            )
            if consecutive >= 2:
                break
    result = {"chunks": count, "steps": step, "passed": consecutive >= 2,
              "validation": metric, "history": history}
    checkpoint(folder / f"{label}.pt", model, probe, optimizer, {"stage_result": result})
    return result


def withdraw(model, probe, optimizer, args, device, dtype, folder, schedule, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr
    history = []
    for phase, (weight, steps) in enumerate(schedule, 1):
        for parameter in probe.parameters():
            parameter.requires_grad_(weight > 0)
        for step in range(1, steps + 1):
            model.train(); probe.train(weight > 0)
            random_step(model, probe, optimizer, args, 16, 2, weight, device, dtype)
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(model, probe, eval_args(args), 16, device, dtype)
                history.append({"phase": phase, "weight": weight, "step": step, **metric})
                save(folder / "withdrawal_progress.json", history)
                print(
                    f"withdraw weight={weight} step={step} query={metric['query']:.2%} "
                    f"probe={metric['probe_min']:.2%}", flush=True
                )
        checkpoint(folder / f"withdrawal_phase{phase}.pt", model, probe, optimizer,
                   {"withdrawal_history": history})
    final = evaluate(model, probe, eval_args(args), 16, device, dtype, args.final_eval_batches)
    return history, final


def load_source(path, args, device):
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()), lr=1e-3)
    state = restore(path, model, probe, optimizer, device)
    return model, probe, optimizer, state


def seed7_budget(args, device, dtype, folder):
    source = Path(args.level6_6_root) / "seed7" / "curriculum_stage2.pt"
    model, probe, optimizer, state = load_source(source, args, device)
    initial = state["stages"][-1]["validation"]
    stages = [train_stage(model, probe, optimizer, args, device, dtype, folder,
                          "extended_4chunk", 4, args.extension_steps, 1e-3, 4, initial)]
    if stages[-1]["passed"]:
        stages.append(train_stage(model, probe, optimizer, args, device, dtype, folder,
                                  "recovery_8chunk", 8, args.later_steps, 2.5e-4, 4))
    if stages[-1]["passed"] and len(stages) == 2:
        stages.append(train_stage(model, probe, optimizer, args, device, dtype, folder,
                                  "recovery_16chunk", 16, args.later_steps, 5e-5, 2))
    final = None; withdrawal_history = []
    if len(stages) == 3 and all(stage["passed"] for stage in stages):
        withdrawal_history, final = withdraw(
            model, probe, optimizer, args, device, dtype, folder,
            [(0.2, 300), (0.1, 300), (0.0, 500)], 1e-5
        )
    return {"case": "seed7_budget", "source": str(source), "stages": stages,
            "withdrawal_history": withdrawal_history, "final": final,
            "passed": final is not None and stage_passed(final)}


def seed42_lr(args, device, dtype, folder):
    source = Path(args.level6_6_root) / "seed42" / "curriculum_stage3.pt"
    model, probe, optimizer, _ = load_source(source, args, device)
    baseline = evaluate(model, probe, eval_args(args), 16, device, dtype)
    stage = train_stage(model, probe, optimizer, args, device, dtype, folder,
                        "lower_lr_16chunk", 16, args.later_steps, 1e-5, 2, baseline)
    final = None; withdrawal_history = []
    if stage["passed"]:
        withdrawal_history, final = withdraw(
            model, probe, optimizer, args, device, dtype, folder,
            [(0.2, 300), (0.1, 300), (0.0, 500)], 1e-5
        )
    return {"case": "seed42_lr", "source": str(source), "baseline": baseline,
            "stages": [stage], "withdrawal_history": withdrawal_history, "final": final,
            "passed": final is not None and stage_passed(final)}


def seed2026_withdrawal(args, device, dtype, folder):
    source = Path(args.level6_6_root) / "seed2026" / "curriculum_stage4.pt"
    model, probe, optimizer, _ = load_source(source, args, device)
    schedule = [(0.3, 300), (0.2, 300), (0.1, 300), (0.05, 300), (0.0, 750)]
    history, final = withdraw(model, probe, optimizer, args, device, dtype, folder,
                              schedule, args.slow_withdrawal_lr)
    return {"case": "seed2026_withdrawal", "source": str(source), "schedule": schedule,
            "withdrawal_history": history, "final": final, "passed": stage_passed(final)}


def main():
    p = argparse.ArgumentParser(description="Post-hoc Level 6.6 targeted recovery controls")
    p.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    p.add_argument("--level6-6-root", default="experiments/level6_6/formal")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--extension-steps", type=int, default=300)
    p.add_argument("--later-steps", type=int, default=1000)
    p.add_argument("--probe-weight", type=float, default=0.5)
    p.add_argument("--slow-withdrawal-lr", type=float, default=5e-6)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--final-eval-batches", type=int, default=50)
    p.add_argument("--output", default="experiments/level6_6_recovery/formal")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda"); dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True); results = []
    functions = {"seed7_budget": seed7_budget, "seed42_lr": seed42_lr,
                 "seed2026_withdrawal": seed2026_withdrawal}
    for case in args.cases:
        folder = root / case; folder.mkdir(parents=True, exist_ok=True); result_path = folder / "result.json"
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            started = time.perf_counter(); result = functions[case](args, device, dtype, folder)
            result["seconds"] = time.perf_counter() - started; save(result_path, result)
        results.append(result); save(root / "runs.partial.json", results); torch.cuda.empty_cache()
    summary = {"post_hoc": True, "cases": len(results),
               "recoveries": sum(result["passed"] for result in results),
               "results": {result["case"]: result["passed"] for result in results}}
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
