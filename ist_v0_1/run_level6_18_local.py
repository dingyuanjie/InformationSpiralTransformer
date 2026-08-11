import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from long_context_test import set_seed
from run_level6_6_local import (
    build, checkpoint, evaluate, random_step, restore,
)
from run_level6_13_1_local import bootstrap_mean_ci, mcnemar, save
from run_level6_16_local import collect_detector_data, load_model, probe_scores
from run_level6_16_1_local import (
    clean_trigger_rate, decision_metrics, filter_split, filter_utility_split,
    fit_utility_classifier, select_utility_threshold, train_time_detectors,
)
from run_level6_16_1_local import collect_utility_data


RECOVERY_SEEDS = [707, 909]
CALIBRATION_SEED = 707
HELD_OUT_SEED = 909
SWAP_TIMES = [4, 8]
ACTIONS = [0.20, 0.25]
FIXED_CANDIDATES = [
    {"key": "13-28", "slots": [13, 28], "category": "cross_model_robust"},
    {"key": "2-7", "slots": [2, 7], "category": "fixed_external"},
    {"key": "10-17", "slots": [10, 17], "category": "fixed_external"},
]


def atomic_checkpoint(path, model, probe, optimizer, extra):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    checkpoint(temporary, model, probe, optimizer, extra)
    last_error = None
    for _ in range(10):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.1)
    raise last_error


def continue_stage(model, probe, optimizer, seed, count, lr, batch, args,
                   device, dtype, folder):
    path = folder / f"extended_stage{count}.pt"
    history = []
    start_step = 0
    consecutive = 0
    if path.exists() and not args.force:
        state = restore(path, model, probe, optimizer, device)
        meta = state["extension"]
        history = meta["history"]
        start_step = meta["step"]
        consecutive = meta["consecutive"]
        if meta.get("passed"):
            return meta
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = lr
    eval_args = argparse.Namespace(eval_batches=args.eval_batches,
                                   eval_batch_size=args.eval_batch_size,
                                   chunk_size=args.chunk_size)
    metric = history[-1]["metric"] if history else None
    for step in range(start_step + 1, args.extension_steps + 1):
        model.train(); probe.train()
        random_step(model, probe, optimizer, args, count, batch,
                    args.probe_weight, device, dtype)
        if step == 1 or step % args.eval_every == 0:
            metric = evaluate(model, probe, eval_args, count, device, dtype)
            ok = metric["query"] >= 0.95
            consecutive = consecutive + 1 if ok else 0
            history.append({"step": step, "metric": metric, "ok": ok})
            meta = {"seed": seed, "chunks": count, "step": step,
                    "consecutive": consecutive, "passed": consecutive >= 2,
                    "history": history, "final": metric}
            atomic_checkpoint(path, model, probe, optimizer, {"extension": meta})
            print(f"seed={seed} extend chunks={count} step={step} "
                  f"query={metric['query']:.2%} probe={metric['probe_min']:.2%}",
                  flush=True)
            if consecutive >= 2:
                return meta
    return {"seed": seed, "chunks": count, "step": args.extension_steps,
            "consecutive": consecutive, "passed": False,
            "history": history, "final": metric}


def run_withdrawal(model, probe, optimizer, args, device, dtype, folder, seed):
    eval_args = argparse.Namespace(eval_batches=args.eval_batches,
                                   eval_batch_size=args.eval_batch_size,
                                   chunk_size=args.chunk_size)
    schedule = [(0.2, args.withdrawal_ramp_steps),
                (0.1, args.withdrawal_ramp_steps),
                (0.0, args.maintenance_steps)]
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
                history.append({"phase": phase, "weight": weight,
                                "step": step, **metric})
                save(folder / "withdrawal_progress.json", history)
                print(f"seed={seed} withdraw={weight} step={step} "
                      f"query={metric['query']:.2%}", flush=True)
        atomic_checkpoint(folder / f"withdrawal_phase{phase}.pt",
                          model, probe, optimizer,
                          {"withdrawal_history": history})
    final = evaluate(model, probe, eval_args, 16, device, dtype,
                     args.final_eval_batches)
    return history, final


def recover_seed(seed, args, device, dtype, root):
    folder = root / "checkpoints" / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    final_path = folder / "withdrawal_phase3.pt"
    if result_path.exists() and not args.force:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("passed") and final_path.exists():
            return result
    source_folder = Path(args.level6_8_root) / f"seed{seed}"
    original = json.loads((source_folder / "result.json").read_text(encoding="utf-8"))
    if original.get("passed"):
        raise RuntimeError(f"seed{seed} is not a failed external seed")
    failed_stage = original["stages"][-1]["chunks"]
    source_checkpoint = source_folder / f"curriculum_stage{len(original['stages'])}.pt"
    set_seed(seed)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()), lr=1e-3)
    restore(source_checkpoint, model, probe, optimizer, device)
    started = time.perf_counter()
    extensions = []
    if failed_stage == 8:
        stage8 = continue_stage(model, probe, optimizer, seed, 8, args.stage3_lr, 4,
                                args, device, dtype, folder)
        extensions.append(stage8)
        if not stage8["passed"]:
            result = {"seed": seed, "passed": False, "failed_phase": "extended_8",
                      "original_failure": original["failed_phase"],
                      "extensions": extensions, "seconds": time.perf_counter() - started}
            save(result_path, result); return result
        stage16 = continue_stage(model, probe, optimizer, seed, 16, args.stage4_lr, 2,
                                 args, device, dtype, folder)
        extensions.append(stage16)
    elif failed_stage == 16:
        stage16 = continue_stage(model, probe, optimizer, seed, 16, args.stage4_lr, 2,
                                 args, device, dtype, folder)
        extensions.append(stage16)
    else:
        raise RuntimeError(f"Unexpected failed stage {failed_stage} for seed{seed}")
    if not extensions[-1]["passed"]:
        result = {"seed": seed, "passed": False,
                  "failed_phase": f"extended_{extensions[-1]['chunks']}",
                  "original_failure": original["failed_phase"],
                  "extensions": extensions, "seconds": time.perf_counter() - started}
        save(result_path, result); return result
    # Withdrawal is deliberately identical to Level 6.8 and writes into Level 6.18.
    history, final = run_withdrawal(
        model, probe, optimizer, args, device, dtype, folder, seed)
    passed = final["query"] >= 0.95
    result = {"seed": seed, "passed": passed,
              "probe_diagnostic_passed": final["probe_min"] >= 0.90,
              "failed_phase": None if passed else "withdrawal",
              "original_failure": original["failed_phase"], "extensions": extensions,
              "withdrawal_history": history, "final": final,
              "seconds": time.perf_counter() - started}
    save(result_path, result)
    return result


def clean_only_threshold(probe, detector_data, swap_after, max_fpr):
    validation = filter_split(detector_data["splits"]["validation"], swap_after)
    x = np.asarray(validation["x"], dtype=np.float32)
    y = np.asarray(validation["y"], dtype=np.int8)
    scores = probe_scores(probe, x[y == 0])
    candidates = np.unique(scores)
    valid = [float(value) for value in candidates
             if float((scores >= value).mean()) <= max_fpr + 1e-12]
    return min(valid) if valid else float(np.nextafter(scores.max(), np.inf))


def fit_calibration_policy(detector_data, utility_data, args):
    detectors = train_time_detectors({CALIBRATION_SEED: detector_data[CALIBRATION_SEED]}, args)
    policies = {}
    for swap_after in args.swap_times:
        train = filter_utility_split(
            utility_data[CALIBRATION_SEED]["splits"]["train"], swap_after)
        x = np.asarray(train["x"], dtype=np.float32)
        utilities = {action: fit_utility_classifier(
            x, np.asarray(train[f"utility_{action}"], dtype=np.int8), args)
            for action in args.actions}
        validation = filter_utility_split(
            utility_data[CALIBRATION_SEED]["splits"]["validation"], swap_after)
        detector = detectors[CALIBRATION_SEED][swap_after]
        selection = select_utility_threshold(
            validation, detector_data[CALIBRATION_SEED], detector, utilities,
            args.actions, swap_after, args)
        policies[swap_after] = {"detector": detector, "utilities": utilities,
                                "utility_threshold": selection["threshold"],
                                "selection": selection}
    return policies


def evaluate_transfer(detector_data, utility_data, policies, seed, mode, args):
    results = []
    for swap_after in args.swap_times:
        policy = policies[swap_after]
        detector = dict(policy["detector"])
        if mode == "clean_recalibrated":
            detector["threshold"] = clean_only_threshold(
                detector["probe"], detector_data[seed], swap_after, args.max_clean_fpr)
        test = filter_utility_split(utility_data[seed]["splits"]["test"], swap_after)
        metric = decision_metrics(test, detector, policy["utilities"], args.actions,
                                  policy["utility_threshold"])
        clean_fpr = clean_trigger_rate(
            detector_data[seed], detector, policy["utilities"], args.actions,
            policy["utility_threshold"], swap_after, "test")
        selected = metric.pop("selected_utility")
        metric["accuracy_gain_ci"] = bootstrap_mean_ci(
            selected, args.bootstrap_seed + seed * 10 + swap_after
            + (10000 if mode == "clean_recalibrated" else 0),
            args.bootstrap_iterations)
        metric["mcnemar"] = mcnemar(selected.clip(min=0), (-selected).clip(min=0))
        results.append({"seed": seed, "mode": mode, "swap_after": swap_after,
                        "detector_threshold": detector["threshold"],
                        "clean_fpr": clean_fpr, **metric,
                        "selected_utility": selected.tolist()})
    return results


def aggregate_transfer(rows, args, offset):
    selected = np.concatenate([np.asarray(row["selected_utility"], dtype=np.int8)
                               for row in rows])
    return {"samples": int(len(selected)),
            "accuracy_gain": bootstrap_mean_ci(
                selected, args.bootstrap_seed + offset, args.bootstrap_iterations),
            "corrected": int((selected == 1).sum()),
            "harmed": int((selected == -1).sum()),
            "mcnemar": mcnemar(selected.clip(min=0), (-selected).clip(min=0)),
            "max_clean_fpr": max(row["clean_fpr"] for row in rows)}


def serialize_policy(policies):
    output = {}
    for time_key, item in policies.items():
        detector = item["detector"]
        output[str(time_key)] = {
            "detector": {"threshold": detector["threshold"],
                         "probe": {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                   for key, value in detector["probe"].items()}},
            "utilities": {str(action): {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in probe.items()}
                for action, probe in item["utilities"].items()},
            "utility_threshold": item["utility_threshold"],
            "selection": item["selection"]}
    return output


def strip_arrays(rows):
    return [{key: value for key, value in row.items() if key != "selected_utility"}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Level 6.18 external initialization validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=RECOVERY_SEEDS)
    parser.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--held-out-seed", type=int, default=HELD_OUT_SEED)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--actions", nargs="+", type=float, default=ACTIONS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--extension-steps", type=int, default=3000)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)
    parser.add_argument("--stage4-lr", type=float, default=1e-5)
    parser.add_argument("--withdrawal-lr", type=float, default=5e-6)
    parser.add_argument("--withdrawal-ramp-steps", type=int, default=300)
    parser.add_argument("--maintenance-steps", type=int, default=750)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--detector-samples-per-seed", type=int, default=200)
    parser.add_argument("--detector-train-seeds", type=int, default=2)
    parser.add_argument("--detector-validation-seeds", type=int, default=1)
    parser.add_argument("--detector-test-seeds", type=int, default=1)
    parser.add_argument("--detector-seed-base", type=int, default=8950000)
    parser.add_argument("--utility-samples-per-seed", type=int, default=200)
    parser.add_argument("--utility-train-seeds", type=int, default=2)
    parser.add_argument("--utility-validation-seeds", type=int, default=1)
    parser.add_argument("--utility-test-seeds", type=int, default=1)
    parser.add_argument("--utility-seed-base", type=int, default=9150000)
    parser.add_argument("--eval-batch-size-policy", dest="eval_batch_size", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--probe-lr", type=float, default=0.03)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-clean-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=618000)
    parser.add_argument("--output", default="experiments/level6_18/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.behavior_only_gate = True
    if args.seeds != [args.calibration_seed, args.held_out_seed]:
        raise ValueError("Roles are preregistered as seed707 calibration, seed909 held-out")
    if args.detector_samples_per_seed % args.eval_batch_size:
        raise ValueError("detector samples must be divisible by policy eval batch size")
    if args.utility_samples_per_seed % args.eval_batch_size:
        raise ValueError("utility samples must be divisible by policy eval batch size")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "roles": {"707": "calibration", "909": "fully_held_out"},
        "fixed_slot_pairs": FIXED_CANDIDATES,
        "checkpoint_recovery": {"extra_steps_per_failed_stage": args.extension_steps,
                                "gate": "two consecutive query >= 0.95",
                                "withdrawal": "identical to Level 6.8"},
        "held_out_modes": ["zero_shot", "clean_recalibrated"],
        "held_out_success": {"clean_recalibrated_gain_ci_lower": "> 0",
                             "max_clean_fpr": "<= 0.05"},
        "protocol": vars(args)}
    save(root / "preregistration.json", preregistration)
    recovery = []
    for seed in args.seeds:
        recovery.append(recover_seed(seed, args, device, dtype, root))
        save(root / "recovery.partial.json", recovery)
        torch.cuda.empty_cache()
    if not all(item["passed"] for item in recovery):
        save(root / "summary.json", {"preregistration": preregistration,
                                     "recovery": recovery,
                                     "external_validation": None,
                                     "success": {"passed": False,
                                                 "reason": "checkpoint_recovery_failed"}})
        print("Checkpoint recovery failed; external validation was not run.")
        return
    detector_path = root / "detector_datasets.json"
    utility_path = root / "utility_datasets.json"
    if detector_path.exists() and utility_path.exists() and not args.force:
        detector_data = {int(k): v for k, v in
                         json.loads(detector_path.read_text(encoding="utf-8")).items()}
        utility_data = {int(k): v for k, v in
                        json.loads(utility_path.read_text(encoding="utf-8")).items()}
    else:
        detector_data, utility_data = {}, {}
        for seed in args.seeds:
            model = load_model(seed, argparse.Namespace(
                level6_8_root=str(root / "checkpoints"), chunk_size=args.chunk_size), device)
            detector_data[seed] = collect_detector_data(
                model, seed, FIXED_CANDIDATES, args, device, dtype)
            utility_data[seed] = collect_utility_data(
                model, seed, FIXED_CANDIDATES, args, device, dtype)
            save(detector_path, detector_data); save(utility_path, utility_data)
            torch.cuda.empty_cache()
    policies = fit_calibration_policy(detector_data, utility_data, args)
    save(root / "calibration_policy.json", serialize_policy(policies))
    calibration = evaluate_transfer(detector_data, utility_data, policies,
                                    args.calibration_seed, "in_model", args)
    zero_shot = evaluate_transfer(detector_data, utility_data, policies,
                                 args.held_out_seed, "zero_shot", args)
    clean_recal = evaluate_transfer(detector_data, utility_data, policies,
                                   args.held_out_seed, "clean_recalibrated", args)
    aggregates = {"calibration": aggregate_transfer(calibration, args, 1000),
                  "zero_shot": aggregate_transfer(zero_shot, args, 2000),
                  "clean_recalibrated": aggregate_transfer(clean_recal, args, 3000)}
    held = aggregates["clean_recalibrated"]
    passed = held["accuracy_gain"]["ci95"][0] > 0 and held["max_clean_fpr"] <= args.max_clean_fpr
    summary = {"preregistration": preregistration, "recovery": recovery,
               "external_validation": {"calibration": strip_arrays(calibration),
                                       "zero_shot": strip_arrays(zero_shot),
                                       "clean_recalibrated": strip_arrays(clean_recal),
                                       "aggregate": aggregates},
               "success": {"passed": passed}}
    save(root / "summary.json", summary)
    print(json.dumps({"recovery": [{"seed": x["seed"], "passed": x["passed"]}
                                    for x in recovery],
                      "external": aggregates, "passed": passed}, indent=2))


if __name__ == "__main__": main()
