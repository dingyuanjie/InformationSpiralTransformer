import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_1_local import evaluate as evaluate_fixed
from run_level6_1_local import forward_pair, make_batch
from run_level6_2_local import evaluate, forward_chunks, make_chunks


SEEDS = [313, 42, 2026, 7, 1234]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build(device, size):
    return (
        InformationSpiralTransformer(19, 64, 3, size, "rope", True).to(device),
        nn.Linear(192, 16).to(device),
    )


def checkpoint(path, model, probe, optimizer, payload):
    torch.save(
        {
            "model": model.state_dict(),
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            **payload,
        },
        path,
    )


def restore(path, model, probe, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    probe.load_state_dict(state["probe"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
    return state


def fixed_stage(model, probe, optimizer, args, device, dtype, folder, seed):
    path = folder / "level6_1.pt"
    if path.exists() and not args.force:
        state = restore(path, model, probe, optimizer, device)
        return state["fixed_result"]
    set_seed(seed + 10000)
    history = []
    consecutive = 0
    eval_args = argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.fixed_eval_batch_size,
        chunk_size=args.chunk_size,
    )
    for step in range(1, args.fixed_steps + 1):
        model.train()
        probe.train()
        first, second, target = make_batch(args.fixed_batch_size, args.chunk_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            a, b, p1, p2, _, _ = forward_pair(model, probe, first, second)
            loss = (
                F.cross_entropy(b[:, -1, :16], target)
                + 0.5 * F.cross_entropy(a[:, 0, :16], target)
                + 0.5 * F.cross_entropy(p1, target)
                + 0.5 * F.cross_entropy(p2, target)
                + 0.1 * model.memory_diversity_loss()
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            metric = evaluate_fixed(model, probe, eval_args, device, dtype)
            history.append({"step": step, **metric})
            save(folder / "level6_1_progress.json", history)
            ok = metric["query"] >= 0.95 and metric["probe1"] >= 0.95 and metric["probe2"] >= 0.95
            consecutive = consecutive + 1 if ok else 0
            print(
                f"seed={seed} fixed step={step} query={metric['query']:.2%} "
                f"probe={min(metric['probe1'], metric['probe2']):.2%}",
                flush=True,
            )
            if consecutive >= 2:
                break
    result = {"passed": consecutive >= 2, "steps": step, "final": metric, "history": history}
    checkpoint(path, model, probe, optimizer, {"fixed_result": result})
    return result


def random_step(model, probe, optimizer, args, count, batch, weight, device, dtype):
    chunks, target, pos = make_chunks(batch, count, args.chunk_size, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        last, first, probes, _ = forward_chunks(model, probe, chunks)
        rows = torch.arange(batch, device=device)
        query = F.cross_entropy(last[:, -1, :16], target)
        local = F.cross_entropy(first[rows, pos, :16], target)
        probe_loss = torch.stack([F.cross_entropy(item, target) for item in probes]).mean()
        loss = query + 0.5 * local + weight * probe_loss + 0.1 * model.memory_diversity_loss()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), 1.0)
    optimizer.step()


def curriculum(model, probe, optimizer, args, device, dtype, folder, seed):
    eval_args = argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    specs = [(2, args.random_stage1_steps, 1e-3, 8), (4, args.later_steps, 1e-3, 4),
             (8, args.later_steps, getattr(args, "stage3_lr", 2.5e-4), 4),
             (16, args.later_steps, args.stage4_lr, 2)]
    stages = []
    history = []
    start = 0
    for index in range(4, 0, -1):
        path = folder / f"curriculum_stage{index}.pt"
        if path.exists() and not args.force:
            state = restore(path, model, probe, optimizer, device)
            stages, history, start = state["stages"], state["history"], index
            break
    for stage_index, (count, steps, lr, batch) in enumerate(specs[start:], start + 1):
        for parameter in probe.parameters():
            parameter.requires_grad_(True)
        for group in optimizer.param_groups:
            group["lr"] = lr
        consecutive = 0
        metric = None
        for step in range(1, steps + 1):
            model.train(); probe.train()
            random_step(model, probe, optimizer, args, count, batch, args.probe_weight, device, dtype)
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(model, probe, eval_args, count, device, dtype)
                history.append({"stage": stage_index, "step": step, **metric})
                save(folder / "curriculum_progress.json", history)
                ok = metric["query"] >= 0.95 and (
                    getattr(args, "behavior_only_gate", False) or metric["probe_min"] >= 0.90
                )
                consecutive = consecutive + 1 if ok else 0
                print(
                    f"seed={seed} random chunks={count} step={step} "
                    f"query={metric['query']:.2%} probe={metric['probe_min']:.2%}", flush=True
                )
                if consecutive >= 2:
                    break
        stages.append({"chunks": count, "steps": step, "passed": consecutive >= 2, "validation": metric})
        checkpoint(folder / f"curriculum_stage{stage_index}.pt", model, probe, optimizer,
                   {"stages": stages, "history": history})
        if consecutive < 2:
            break
    return stages, history


def withdrawal(model, probe, optimizer, args, device, dtype, folder, seed):
    eval_args = argparse.Namespace(eval_batches=args.eval_batches, eval_batch_size=args.eval_batch_size,
                                   chunk_size=args.chunk_size)
    schedule = [(0.2, 300), (0.1, 300), (0.0, args.maintenance_steps)]
    history = []
    for group in optimizer.param_groups:
        group["lr"] = args.withdrawal_lr
    for phase, (weight, steps) in enumerate(schedule, 1):
        for parameter in probe.parameters():
            parameter.requires_grad_(weight > 0)
        for step in range(1, steps + 1):
            model.train(); probe.train(weight > 0)
            random_step(model, probe, optimizer, args, 16, 2, weight, device, dtype)
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(model, probe, eval_args, 16, device, dtype)
                history.append({"phase": phase, "weight": weight, "step": step, **metric})
                save(folder / "withdrawal_progress.json", history)
                print(f"seed={seed} withdraw={weight} step={step} query={metric['query']:.2%}", flush=True)
        checkpoint(folder / f"withdrawal_phase{phase}.pt", model, probe, optimizer,
                   {"withdrawal_history": history})
    final = evaluate(model, probe, eval_args, 16, device, dtype, args.final_eval_batches)
    return history, final


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
        result = {"seed": seed, "passed": False, "failed_phase": "fixed", "fixed": fixed,
                  "seconds": time.perf_counter() - started}
        save(result_path, result); return result
    set_seed(seed + 20000)
    stages, curriculum_history = curriculum(model, probe, optimizer, args, device, dtype, folder, seed)
    curriculum_passed = len(stages) == 4 and all(stage["passed"] for stage in stages)
    if not curriculum_passed:
        result = {"seed": seed, "passed": False, "failed_phase": "curriculum", "fixed": fixed,
                  "stages": stages, "curriculum_history": curriculum_history,
                  "seconds": time.perf_counter() - started}
        save(result_path, result); return result
    withdrawal_history, final = withdrawal(model, probe, optimizer, args, device, dtype, folder, seed)
    passed = final["query"] >= 0.95 and final["probe_min"] >= 0.90
    result = {"seed": seed, "passed": passed, "failed_phase": None if passed else "withdrawal",
              "fixed": fixed, "stages": stages, "final": final,
              "withdrawal_history": withdrawal_history, "seconds": time.perf_counter() - started}
    save(result_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description="Level 6.6 full independent formation validation")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--fixed-steps", type=int, default=2000)
    p.add_argument("--fixed-batch-size", type=int, default=16)
    p.add_argument("--fixed-eval-batch-size", type=int, default=16)
    p.add_argument("--random-stage1-steps", type=int, default=2000)
    p.add_argument("--later-steps", type=int, default=1000)
    p.add_argument("--probe-weight", type=float, default=0.5)
    p.add_argument("--stage4-lr", type=float, default=5e-5)
    p.add_argument("--withdrawal-lr", type=float, default=1e-5)
    p.add_argument("--maintenance-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--final-eval-batches", type=int, default=50)
    p.add_argument("--output", default="experiments/level6_6/formal")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
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
    phases = {name: sum(result.get("failed_phase") == name for result in results)
              for name in ["fixed", "curriculum", "withdrawal"]}
    summary = {"runs": len(results), "successes": sum(result["passed"] for result in results),
               "success_rate": statistics.mean(result["passed"] for result in results), "failures": phases}
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
