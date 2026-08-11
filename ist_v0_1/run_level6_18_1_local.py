import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from long_context_test import set_seed
from run_level6_2_local import evaluate
from run_level6_6_local import build, checkpoint, random_step, restore
from run_level6_9_local import CONDITIONS, evaluate_condition


CALIBRATION_SEED = 707
LOCKED_TRANSFER_SEED = 909
FORMAL_SEEDS = [CALIBRATION_SEED, LOCKED_TRANSFER_SEED]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def atomic_checkpoint(path, model, probe, optimizer, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    checkpoint(temporary, model, probe, optimizer, payload)
    last_error = None
    for _ in range(10):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.1)
    raise last_error


def preserved_rng_evaluate(model, probe, args, count, device, dtype, seed, batches):
    """Evaluate on a fixed panel without changing the subsequent training RNG."""
    python_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    try:
        set_seed(seed)
        eval_args = argparse.Namespace(
            eval_batches=batches,
            eval_batch_size=args.eval_batch_size,
            chunk_size=args.chunk_size,
        )
        return evaluate(model, probe, eval_args, count, device, dtype, batches)
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)


def screen(model, probe, args, seed, count, device, dtype):
    eval_seed = args.validation_seed_base + seed * 1000 + count * 10
    return preserved_rng_evaluate(
        model, probe, args, count, device, dtype, eval_seed,
        args.screen_eval_batches,
    )


def confirm(model, probe, args, seed, count, device, dtype):
    panels = []
    for panel in range(2):
        eval_seed = (
            args.validation_seed_base + seed * 1000 + count * 10 + panel + 1
        )
        panels.append(preserved_rng_evaluate(
            model, probe, args, count, device, dtype, eval_seed,
            args.confirm_eval_batches,
        ))
    total = sum(item["samples"] for item in panels)
    query_mean = sum(item["query"] * item["samples"] for item in panels) / total
    local_mean = sum(item["local"] * item["samples"] for item in panels) / total
    result = {
        "samples": total,
        "query_mean": query_mean,
        "query_worst_panel": min(item["query"] for item in panels),
        "local_mean": local_mean,
        "probe_min": min(item["probe_min"] for item in panels),
        "panels": panels,
    }
    result["behavior_passed"] = (
        result["query_mean"] >= args.confirm_query_threshold
        and result["query_worst_panel"] >= args.confirm_panel_floor
    )
    result["probe_diagnostic_passed"] = (
        result["probe_min"] >= args.confirm_probe_diagnostic_threshold
    )
    return result


def phase_cycle(base, midpoint, target, phase):
    if phase == "bridge":
        # 25% rehearsal at the last proven length, 75% at the bridge length.
        return [base, midpoint, midpoint, midpoint]
    if phase == "target":
        # Keep both previously proven lengths active while the target dominates.
        return [base, midpoint, target, target, target, target]
    raise ValueError(phase)


def phase_paths(folder, base, target, phase):
    stem = f"transition_{base}_to_{target}_{phase}"
    return {
        "latest": folder / f"{stem}_latest.pt",
        "best": folder / f"{stem}_best.pt",
        "stable": folder / f"{stem}_stable.pt",
        "progress": folder / f"{stem}_progress.json",
    }


def train_phase(model, probe, optimizer, args, seed, base, midpoint, target,
                phase, device, dtype, folder):
    paths = phase_paths(folder, base, target, phase)
    eval_count = midpoint if phase == "bridge" else target
    max_steps = args.bridge_steps if phase == "bridge" else args.target_steps
    lr = args.bridge_lr if phase == "bridge" else args.target_lr
    cycle = phase_cycle(base, midpoint, target, phase)
    start_step = 0
    stable_streak = 0
    history = []
    best_score = None

    if paths["stable"].exists() and not args.force:
        state = restore(paths["stable"], model, probe, optimizer, device)
        return state["phase_result"]
    if paths["latest"].exists() and not args.force:
        state = restore(paths["latest"], model, probe, optimizer, device)
        meta = state["phase_result"]
        start_step = meta["step"]
        stable_streak = meta["stable_streak"]
        history = meta["history"]
        best_score = meta.get("best_score")

    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = lr

    final_screen = history[-1]["screen"] if history else None
    final_confirmation = history[-1].get("confirmation") if history else None
    for step in range(start_step + 1, max_steps + 1):
        count = cycle[(step - 1) % len(cycle)]
        batch = max(1, args.chunk_batch_budget // count)
        model.train()
        probe.train()
        random_step(
            model, probe, optimizer, args, count, batch,
            args.probe_weight, device, dtype,
        )
        should_evaluate = step == 1 or step % args.eval_every == 0 or step == max_steps
        if not should_evaluate:
            continue

        final_screen = screen(model, probe, args, seed, eval_count, device, dtype)
        final_confirmation = None
        if final_screen["query"] >= args.screen_query_threshold:
            final_confirmation = confirm(
                model, probe, args, seed, eval_count, device, dtype,
            )
        confirmed = bool(
            final_confirmation and final_confirmation["behavior_passed"]
        )
        stable_streak = stable_streak + 1 if confirmed else 0
        row = {
            "step": step,
            "trained_chunks": count,
            "screen": final_screen,
            "confirmation": final_confirmation,
            "confirmed": confirmed,
            "stable_streak": stable_streak,
        }
        history.append(row)
        save(paths["progress"], history)

        score = [
            final_confirmation["query_mean"] if final_confirmation else final_screen["query"],
            final_confirmation["query_worst_panel"] if final_confirmation else final_screen["query"],
            final_confirmation["probe_min"] if final_confirmation else final_screen["probe_min"],
        ]
        is_best = best_score is None or tuple(score) > tuple(best_score)
        if is_best:
            best_score = score

        meta = {
            "seed": seed,
            "phase": phase,
            "base_chunks": base,
            "midpoint_chunks": midpoint,
            "target_chunks": target,
            "eval_chunks": eval_count,
            "step": step,
            "stable_streak": stable_streak,
            "passed": stable_streak >= args.stable_confirmations,
            "best_score": best_score,
            "history": history,
            "final_screen": final_screen,
            "final_confirmation": final_confirmation,
        }
        atomic_checkpoint(
            paths["latest"], model, probe, optimizer, {"phase_result": meta}
        )
        if is_best:
            atomic_checkpoint(
                paths["best"], model, probe, optimizer, {"phase_result": meta}
            )

        confirm_text = "not-run"
        if final_confirmation:
            confirm_text = (
                f"mean={final_confirmation['query_mean']:.2%} "
                f"worst={final_confirmation['query_worst_panel']:.2%}"
            )
        print(
            f"seed={seed} {base}->{target} phase={phase} step={step} "
            f"screen={final_screen['query']:.2%} confirm={confirm_text} "
            f"stable={stable_streak}/{args.stable_confirmations}",
            flush=True,
        )
        if stable_streak >= args.stable_confirmations:
            atomic_checkpoint(
                paths["stable"], model, probe, optimizer, {"phase_result": meta}
            )
            return meta

    return {
        "seed": seed,
        "phase": phase,
        "base_chunks": base,
        "midpoint_chunks": midpoint,
        "target_chunks": target,
        "eval_chunks": eval_count,
        "step": max_steps,
        "stable_streak": stable_streak,
        "passed": False,
        "best_score": best_score,
        "history": history,
        "final_screen": final_screen,
        "final_confirmation": final_confirmation,
    }


def last_passed_source(seed, args):
    source_folder = Path(args.level6_8_root) / f"seed{seed}"
    result_path = source_folder / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("passed"):
        raise RuntimeError(f"seed={seed} is not a failed Level 6.8 initialization")
    passed = [
        (index, stage) for index, stage in enumerate(result.get("stages", []), 1)
        if stage.get("passed")
    ]
    if not passed:
        raise RuntimeError(f"seed={seed} has no passed random curriculum stage")
    index, stage = passed[-1]
    path = source_folder / f"curriculum_stage{index}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "checkpoint": path,
        "stage_index": index,
        "chunks": stage["chunks"],
        "original_failure": result.get("failed_phase"),
    }


def run_withdrawal(model, probe, optimizer, args, seed, device, dtype, folder):
    schedule = [
        (0.2, args.withdrawal_ramp_steps),
        (0.1, args.withdrawal_ramp_steps),
        (0.0, args.maintenance_steps),
    ]
    history = []
    for group in optimizer.param_groups:
        group["lr"] = args.withdrawal_lr

    for phase_index, (weight, steps) in enumerate(schedule, 1):
        phase_path = folder / f"withdrawal_phase{phase_index}.pt"
        if phase_path.exists() and not args.force:
            state = restore(phase_path, model, probe, optimizer, device)
            history = state["withdrawal_history"]
            continue
        for parameter in probe.parameters():
            parameter.requires_grad_(weight > 0)
        for step in range(1, steps + 1):
            model.train()
            probe.train(weight > 0)
            random_step(
                model, probe, optimizer, args, 16, 2, weight, device, dtype
            )
            if step == 1 or step % args.eval_every == 0 or step == steps:
                metric = screen(model, probe, args, seed + phase_index * 100, 16,
                                device, dtype)
                history.append({
                    "phase": phase_index,
                    "weight": weight,
                    "step": step,
                    **metric,
                })
                save(folder / "withdrawal_progress.json", history)
                print(
                    f"seed={seed} withdrawal={weight} step={step} "
                    f"query={metric['query']:.2%}",
                    flush=True,
                )
        atomic_checkpoint(
            phase_path, model, probe, optimizer,
            {"withdrawal_history": history},
        )

    final = preserved_rng_evaluate(
        model, probe, args, 16, device, dtype,
        args.final_eval_seed_base + seed,
        args.final_eval_batches,
    )
    result = {
        "passed": final["query"] >= args.final_query_threshold,
        "probe_diagnostic_passed": (
            final["probe_min"] >= args.final_probe_diagnostic_threshold
        ),
        "history": history,
        "final": final,
    }
    save(folder / "withdrawal_result.json", result)
    return result


def causal_validation(model, probe, args, seed, device, dtype, folder):
    rows = []
    causal_args = argparse.Namespace(
        eval_batches=args.causal_eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    for condition in CONDITIONS:
        set_seed(args.causal_eval_seed_base + seed)
        metric = evaluate_condition(
            model, probe, causal_args, 16, condition, device, dtype
        )
        rows.append(metric)
        print(
            f"seed={seed} causal={condition} query={metric['query']:.2%} "
            f"local={metric['local']:.2%}",
            flush=True,
        )
    selected = {row["condition"]: row for row in rows}
    result = {
        "intact_query": selected["intact"]["query"],
        "max_intervened_query": max(
            selected[name]["query"] for name in CONDITIONS[1:]
        ),
        "min_local": min(selected[name]["local"] for name in CONDITIONS),
        "query_drop": selected["intact"]["query"] - max(
            selected[name]["query"] for name in CONDITIONS[1:]
        ),
        "metrics": rows,
    }
    result["passed"] = (
        result["intact_query"] >= args.causal_intact_threshold
        and result["max_intervened_query"] <= args.causal_intervention_threshold
        and result["min_local"] >= args.causal_local_threshold
    )
    save(folder / "causal_validation.json", result)
    return result


def recover_seed(seed, args, device, dtype, root):
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    final_path = folder / "withdrawal_phase3.pt"
    if result_path.exists() and not args.force:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("passed") or final_path.exists():
            return result

    source = last_passed_source(seed, args)
    set_seed(seed)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.bridge_lr
    )
    restore(source["checkpoint"], model, probe, optimizer, device)
    set_seed(args.training_seed_base + seed)
    started = time.perf_counter()
    transitions = []
    base = source["chunks"]

    while base < args.target_chunks:
        target = min(base * 2, args.target_chunks)
        midpoint = (base + target) // 2
        transition = {
            "base_chunks": base,
            "midpoint_chunks": midpoint,
            "target_chunks": target,
        }
        bridge = train_phase(
            model, probe, optimizer, args, seed, base, midpoint, target,
            "bridge", device, dtype, folder,
        )
        transition["bridge"] = bridge
        if not bridge["passed"]:
            transitions.append(transition)
            result = {
                "seed": seed,
                "role": "calibration" if seed == CALIBRATION_SEED else "locked_transfer",
                "passed": False,
                "failed_phase": f"bridge_{base}_to_{midpoint}",
                "source": {**source, "checkpoint": str(source["checkpoint"])},
                "transitions": transitions,
                "seconds": time.perf_counter() - started,
            }
            save(result_path, result)
            return result

        target_phase = train_phase(
            model, probe, optimizer, args, seed, base, midpoint, target,
            "target", device, dtype, folder,
        )
        transition["target"] = target_phase
        transitions.append(transition)
        if not target_phase["passed"]:
            result = {
                "seed": seed,
                "role": "calibration" if seed == CALIBRATION_SEED else "locked_transfer",
                "passed": False,
                "failed_phase": f"target_{midpoint}_to_{target}",
                "source": {**source, "checkpoint": str(source["checkpoint"])},
                "transitions": transitions,
                "seconds": time.perf_counter() - started,
            }
            save(result_path, result)
            return result
        base = target

    withdrawal = run_withdrawal(
        model, probe, optimizer, args, seed, device, dtype, folder
    )
    causal = None
    if withdrawal["passed"]:
        causal = causal_validation(model, probe, args, seed, device, dtype, folder)
    passed = withdrawal["passed"] and bool(causal and causal["passed"])
    result = {
        "seed": seed,
        "role": "calibration" if seed == CALIBRATION_SEED else "locked_transfer",
        "passed": passed,
        "formation_passed": True,
        "withdrawal_passed": withdrawal["passed"],
        "causal_passed": bool(causal and causal["passed"]),
        "probe_diagnostic_passed": withdrawal["probe_diagnostic_passed"],
        "failed_phase": None if passed else (
            "causal_validation" if withdrawal["passed"] else "withdrawal"
        ),
        "source": {**source, "checkpoint": str(source["checkpoint"])},
        "transitions": transitions,
        "withdrawal": withdrawal,
        "causal": causal,
        "seconds": time.perf_counter() - started,
    }
    save(result_path, result)
    return result


def protocol(args):
    return {
        "level": "6.18.1",
        "hypothesis": (
            "Failed initializations require rollback plus intermediate-length "
            "rehearsal, not more optimization at the failed length."
        ),
        "roles": {
            "707": "calibration; must pass before seed909 is opened",
            "909": "locked transfer; not used to alter this protocol",
        },
        "rollback": "last passed Level 6.8 curriculum checkpoint",
        "normalized_transitions": {
            "707": "4 -> 6 -> 8 -> 12 -> 16",
            "909": "8 -> 12 -> 16",
        },
        "mixtures": {
            "bridge": "[base, midpoint, midpoint, midpoint]",
            "target": "[base, midpoint, target, target, target, target]",
        },
        "gate": {
            "screen_query": args.screen_query_threshold,
            "confirm_query_mean": args.confirm_query_threshold,
            "confirm_worst_panel": args.confirm_panel_floor,
            "fixed_panels": 2,
            "samples_per_panel": args.confirm_eval_batches * args.eval_batch_size,
            "successive_confirmed_checkpoints": args.stable_confirmations,
            "probe_is_diagnostic_only": True,
        },
        "post_formation": {
            "withdrawal": "0.2 -> 0.1 -> 0.0, matched to Level 6.8",
            "behavior_gate": f"query >= {args.final_query_threshold}",
            "causal_gate": {
                "intact_query": f">= {args.causal_intact_threshold}",
                "max_reset_zero_roll_query": f"<= {args.causal_intervention_threshold}",
                "min_local": f">= {args.causal_local_threshold}",
            },
        },
        "protocol": vars(args),
    }


def validate_inputs(args):
    if args.seeds != FORMAL_SEEDS:
        raise ValueError("Formal roles are fixed as seed707 then seed909")
    if args.target_chunks != 16:
        raise ValueError("Level 6.18.1 target is preregistered at 16 chunks")
    if args.eval_batch_size < 2:
        raise ValueError("batch-roll causal validation requires batch size >= 2")
    for seed in args.seeds:
        last_passed_source(seed, args)


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.1 rollback-and-bridge initialization recovery"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--output", default="experiments/level6_18_1/formal")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--target-chunks", type=int, default=16)
    parser.add_argument("--bridge-steps", type=int, default=1500)
    parser.add_argument("--target-steps", type=int, default=3000)
    parser.add_argument("--bridge-lr", type=float, default=2e-5)
    parser.add_argument("--target-lr", type=float, default=1e-5)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--chunk-batch-budget", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--screen-eval-batches", type=int, default=10)
    parser.add_argument("--confirm-eval-batches", type=int, default=25)
    parser.add_argument("--screen-query-threshold", type=float, default=0.90)
    parser.add_argument("--confirm-query-threshold", type=float, default=0.95)
    parser.add_argument("--confirm-panel-floor", type=float, default=0.93)
    parser.add_argument("--confirm-probe-diagnostic-threshold", type=float, default=0.90)
    parser.add_argument("--stable-confirmations", type=int, default=2)
    parser.add_argument("--validation-seed-base", type=int, default=6181000)
    parser.add_argument("--training-seed-base", type=int, default=6181100)
    parser.add_argument("--withdrawal-lr", type=float, default=5e-6)
    parser.add_argument("--withdrawal-ramp-steps", type=int, default=300)
    parser.add_argument("--maintenance-steps", type=int, default=750)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--final-eval-seed-base", type=int, default=6181200)
    parser.add_argument("--final-query-threshold", type=float, default=0.95)
    parser.add_argument("--final-probe-diagnostic-threshold", type=float, default=0.90)
    parser.add_argument("--causal-eval-batches", type=int, default=50)
    parser.add_argument("--causal-eval-seed-base", type=int, default=6181300)
    parser.add_argument("--causal-intact-threshold", type=float, default=0.90)
    parser.add_argument("--causal-intervention-threshold", type=float, default=0.20)
    parser.add_argument("--causal-local-threshold", type=float, default=0.90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.behavior_only_gate = True
    validate_inputs(args)

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    preregistration = protocol(args)
    save(root / "preregistration.json", preregistration)
    if args.dry_run:
        print(json.dumps(preregistration, indent=2))
        return

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

    results = []
    calibration = recover_seed(CALIBRATION_SEED, args, device, dtype, root)
    results.append(calibration)
    save(root / "runs.partial.json", results)
    torch.cuda.empty_cache()
    if not calibration["passed"]:
        summary = {
            "preregistration": preregistration,
            "runs": results,
            "success": {
                "passed": False,
                "reason": "calibration_recovery_failed; seed909_not_opened",
            },
        }
        save(root / "summary.json", summary)
        print("Calibration recovery failed; locked seed909 was not opened.")
        return

    transfer = recover_seed(LOCKED_TRANSFER_SEED, args, device, dtype, root)
    results.append(transfer)
    save(root / "runs.partial.json", results)
    passed = calibration["passed"] and transfer["passed"]
    causal_drops = [
        item["causal"]["query_drop"] for item in results if item.get("causal")
    ]
    summary = {
        "preregistration": preregistration,
        "runs": results,
        "aggregate": {
            "recovered": sum(item["passed"] for item in results),
            "recovery_rate": statistics.mean(item["passed"] for item in results),
            "mean_causal_query_drop": (
                statistics.mean(causal_drops) if causal_drops else None
            ),
        },
        "success": {"passed": passed},
    }
    save(root / "summary.json", summary)
    print(json.dumps(summary["aggregate"] | summary["success"], indent=2))


if __name__ == "__main__":
    main()
