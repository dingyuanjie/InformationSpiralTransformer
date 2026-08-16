"""Level 7.2 validation-selected zero-Probe retention checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from long_context_test import set_seed
from run_level6_2_local import evaluate
from run_level6_6_local import (
    build,
    checkpoint,
    curriculum,
    fixed_stage,
    random_step,
    restore,
)
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import (
    CONDITIONS,
    DISRUPTED,
    atomic_save,
    evaluate_causal_condition,
    model_fingerprint,
)


FORMAL_SEEDS = [1601, 1879]
SMOKE_SEED = 23
CANDIDATE_STEPS = [300, 450, 600, 750]
ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_2"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--candidate-steps", nargs="+", type=int, default=CANDIDATE_STEPS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--fixed-steps", type=int, default=2500)
    parser.add_argument("--fixed-batch-size", type=int, default=16)
    parser.add_argument("--fixed-eval-batch-size", type=int, default=16)
    parser.add_argument("--random-stage1-steps", type=int, default=2500)
    parser.add_argument("--later-steps", type=int, default=1500)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)
    parser.add_argument("--stage4-lr", type=float, default=1e-5)
    parser.add_argument("--withdrawal-lr", type=float, default=5e-6)
    parser.add_argument("--maintenance-steps", type=int, default=750)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--protected-samples", type=int, default=4096)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--diagnostic-batch-size", type=int, default=16)
    parser.add_argument("--selection-query-threshold", type=float, default=0.95)
    parser.add_argument("--protected-query-threshold", type=float, default=0.95)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--keep-l3-threshold", type=float, default=0.90)
    parser.add_argument("--eval-seed-base", type=int, default=7200000)
    parser.add_argument("--output", default="experiments/level7_2/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.behavior_only_gate = True
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def metric_with_interval(metric: dict[str, Any]) -> dict[str, Any]:
    query_correct = round(metric["query"] * metric["samples"])
    local_correct = round(metric["local"] * metric["samples"])
    return {
        **metric,
        "query_correct": query_correct,
        "query_wilson95": wilson_interval(query_correct, metric["samples"]),
        "local_correct": local_correct,
        "local_wilson95": wilson_interval(local_correct, metric["samples"]),
    }


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "seeds": FORMAL_SEEDS,
        "candidate_steps": CANDIDATE_STEPS,
        "chunk_size": 128,
        "fixed_steps": 2500,
        "fixed_batch_size": 16,
        "fixed_eval_batch_size": 16,
        "random_stage1_steps": 2500,
        "later_steps": 1500,
        "probe_weight": 0.5,
        "stage3_lr": 5e-5,
        "stage4_lr": 1e-5,
        "withdrawal_lr": 5e-6,
        "maintenance_steps": 750,
        "eval_every": 100,
        "eval_batch_size": 8,
        "eval_batches": 10,
        "validation_samples": 1024,
        "protected_samples": 4096,
        "causal_samples": 1024,
        "diagnostic_batch_size": 16,
        "selection_query_threshold": 0.95,
        "protected_query_threshold": 0.95,
        "local_threshold": 0.90,
        "disruption_threshold": 0.20,
        "keep_l3_threshold": 0.90,
        "eval_seed_base": 7200000,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.2 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_2/formal":
        raise ValueError("Formal output path is locked")


def run_withdrawal_candidates(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    folder: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    eval_args = argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    for group in optimizer.param_groups:
        group["lr"] = args.withdrawal_lr

    completed_phase = 0
    if not args.force:
        for phase in (2, 1):
            path = folder / f"withdrawal_phase{phase}.pt"
            if path.exists():
                state = restore(path, model, probe, optimizer, device)
                history = state.get("withdrawal_history", [])
                completed_phase = phase
                print(f"seed={seed} resumed withdrawal phase={phase}", flush=True)
                break
    schedule = [(0.2, 300), (0.1, 300)]
    for phase_index in range(completed_phase, len(schedule)):
        phase = phase_index + 1
        weight, steps = schedule[phase_index]
        for parameter in probe.parameters():
            parameter.requires_grad_(True)
        for step in range(1, steps + 1):
            model.train()
            probe.train()
            random_step(model, probe, optimizer, args, 16, 2, weight, device, dtype)
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(model, probe, eval_args, 16, device, dtype)
                history.append({
                    "phase": phase, "weight": weight, "step": step, **metric,
                })
                atomic_save(folder / "withdrawal_progress.json", history)
                print(
                    f"seed={seed} withdraw={weight} step={step} "
                    f"query={metric['query']:.2%}", flush=True,
                )
        checkpoint(
            folder / f"withdrawal_phase{phase}.pt",
            model,
            probe,
            optimizer,
            {"withdrawal_history": history},
        )

    start_step = 1
    if not args.force:
        for candidate_step in reversed(args.candidate_steps):
            path = folder / f"zero_probe_step{candidate_step:04d}.pt"
            if path.exists():
                state = restore(path, model, probe, optimizer, device)
                history = state.get("withdrawal_history", [])
                start_step = candidate_step + 1
                print(f"seed={seed} resumed zero-Probe step={candidate_step}", flush=True)
                break
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    candidate_rows = []
    for step in range(start_step, args.maintenance_steps + 1):
        model.train()
        probe.eval()
        random_step(model, probe, optimizer, args, 16, 2, 0.0, device, dtype)
        if step == 1 or step % args.eval_every == 0:
            metric = evaluate(model, probe, eval_args, 16, device, dtype)
            history.append({
                "phase": 3, "weight": 0.0, "step": step, **metric,
            })
            atomic_save(folder / "withdrawal_progress.json", history)
            print(
                f"seed={seed} zero-Probe step={step} "
                f"query={metric['query']:.2%}", flush=True,
            )
        if step in args.candidate_steps:
            path = folder / f"zero_probe_step{step:04d}.pt"
            checkpoint(
                path,
                model,
                probe,
                optimizer,
                {"withdrawal_history": history, "zero_probe_step": step},
            )
    for step in args.candidate_steps:
        path = folder / f"zero_probe_step{step:04d}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        candidate_rows.append({"step": step, "checkpoint": str(path)})
    return history, candidate_rows


def train_seed(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    folder: Path,
) -> dict[str, Any]:
    result_path = folder / "training_result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=1e-3
    )
    started = time.perf_counter()
    fixed = fixed_stage(model, probe, optimizer, args, device, dtype, folder, seed)
    if not fixed["passed"]:
        result = {
            "seed": seed,
            "reached_candidates": False,
            "failed_phase": "fixed",
            "fixed": fixed,
            "seconds": time.perf_counter() - started,
            "budget_extended": False,
        }
        atomic_save(result_path, result)
        return result
    set_seed(seed + 20000)
    stages, curriculum_history = curriculum(
        model, probe, optimizer, args, device, dtype, folder, seed
    )
    curriculum_passed = len(stages) == 4 and all(stage["passed"] for stage in stages)
    if not curriculum_passed:
        result = {
            "seed": seed,
            "reached_candidates": False,
            "failed_phase": "curriculum",
            "fixed": fixed,
            "stages": stages,
            "curriculum_history": curriculum_history,
            "seconds": time.perf_counter() - started,
            "budget_extended": False,
        }
        atomic_save(result_path, result)
        return result
    withdrawal_history, candidates = run_withdrawal_candidates(
        model, probe, optimizer, args, device, dtype, folder, seed
    )
    result = {
        "seed": seed,
        "reached_candidates": True,
        "failed_phase": None,
        "fixed": fixed,
        "stages": stages,
        "curriculum_history": curriculum_history,
        "withdrawal_history": withdrawal_history,
        "candidates": candidates,
        "seconds": time.perf_counter() - started,
        "budget_extended": False,
    }
    atomic_save(result_path, result)
    return result


def load_candidate(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, probe = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    probe.load_state_dict(state["probe"])
    for module in (model, probe):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, probe


def evaluate_seed(
    seed: int,
    training: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    folder: Path,
) -> dict[str, Any]:
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    base_integrity = {
        "budget_extended": training.get("budget_extended", False),
        "seed_replaced": False,
        "old_checkpoint_used": False,
        "seed909_used": False,
        "protected_used_for_selection": False,
        "training_changed_from_Level_7_1": False,
    }
    if not training["reached_candidates"]:
        result = {
            "seed": seed,
            "training": training,
            "reached_candidates": False,
            "selection": None,
            "protected": None,
            "causal": None,
            "primary_passed": False,
            "integrity": base_integrity,
        }
        atomic_save(result_path, result)
        return result

    validation_seed = args.eval_seed_base + seed * 10
    selection_path = folder / "selection.json"
    if selection_path.exists() and not args.force:
        selection = read_json(selection_path)
        selected = selection["selected"]
        print(f"seed={seed} reused frozen validation selection", flush=True)
    else:
        validation_rows = []
        for candidate in training["candidates"]:
            checkpoint_path = Path(candidate["checkpoint"])
            model, _ = load_candidate(checkpoint_path, args, device)
            metric = evaluate_causal_condition(
                model, args, 16, "intact", args.validation_samples,
                validation_seed, device, dtype,
            )
            metric = metric_with_interval(metric)
            validation_rows.append({
                "step": candidate["step"],
                "checkpoint": candidate["checkpoint"],
                "metric": metric,
                "eligible": (
                    metric["query"] >= args.selection_query_threshold
                    and metric["local"] >= args.local_threshold
                ),
            })
            print(
                f"seed={seed} validation step={candidate['step']} "
                f"query={metric['query']:.2%} local={metric['local']:.2%}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
        eligible = [row for row in validation_rows if row["eligible"]]
        selected = max(
            eligible, key=lambda row: (row["metric"]["query"], row["step"])
        ) if eligible else None
        selection = {
            "validation_seed": validation_seed,
            "candidates": validation_rows,
            "eligible_count": len(eligible),
            "selected": selected,
            "passed": selected is not None,
        }
        atomic_save(selection_path, selection)
    if selected is None:
        result = {
            "seed": seed,
            "training": training,
            "reached_candidates": True,
            "selection": selection,
            "protected": None,
            "causal": None,
            "primary_passed": False,
            "integrity": base_integrity,
        }
        atomic_save(result_path, result)
        return result

    model, _ = load_candidate(Path(selected["checkpoint"]), args, device)
    fingerprint_before = model_fingerprint(model)
    protected_seed = validation_seed + 1
    protected_path = folder / "protected.json"
    if protected_path.exists() and not args.force:
        protected = read_json(protected_path)
        print(f"seed={seed} protected result already frozen", flush=True)
    else:
        protected = metric_with_interval(evaluate_causal_condition(
            model, args, 16, "intact", args.protected_samples,
            protected_seed, device, dtype,
        ))
        protected["dataset_seed"] = protected_seed
        protected["opened"] = True
        protected["passed"] = bool(
            protected["query"] >= args.protected_query_threshold
            and protected["local"] >= args.local_threshold
        )
        atomic_save(protected_path, protected)
    print(
        f"seed={seed} protected query={protected['query']:.2%} "
        f"local={protected['local']:.2%}", flush=True,
    )
    causal = None
    if protected["passed"]:
        causal_seed = validation_seed + 2
        causal_progress_path = folder / "causal_progress.json"
        causal_rows = (
            read_json(causal_progress_path)
            if causal_progress_path.exists() and not args.force else {}
        )
        for condition in CONDITIONS:
            if condition in causal_rows:
                continue
            metric = evaluate_causal_condition(
                model, args, 16, condition, args.causal_samples,
                causal_seed, device, dtype,
            )
            causal_rows[condition] = metric
            atomic_save(causal_progress_path, causal_rows)
            print(
                f"seed={seed} causal={condition} query={metric['query']:.2%}",
                flush=True,
            )
        max_disrupted = max(causal_rows[name]["query"] for name in DISRUPTED)
        min_local = min(row["local"] for row in causal_rows.values())
        causal = {
            "dataset_seed": causal_seed,
            "conditions": causal_rows,
            "max_disrupted_query": max_disrupted,
            "minimum_local": min_local,
            "passed": bool(
                causal_rows["intact"]["query"] >= args.protected_query_threshold
                and min_local >= args.local_threshold
                and max_disrupted <= args.disruption_threshold
                and causal_rows["keep_l3"]["query"] >= args.keep_l3_threshold
            ),
        }
        atomic_save(folder / "causal.json", causal)
    fingerprint_after = model_fingerprint(model)
    integrity = {
        **base_integrity,
        "selected_step_registered": selected["step"] in CANDIDATE_STEPS,
        "single_checkpoint_selected": True,
        "protected_opened_once": True,
        "model_fingerprint_before": fingerprint_before,
        "model_fingerprint_after": fingerprint_after,
        "model_fingerprint_unchanged": fingerprint_before == fingerprint_after,
        "all_model_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
    }
    integrity["passed"] = all([
        not integrity["budget_extended"],
        not integrity["seed_replaced"],
        not integrity["old_checkpoint_used"],
        not integrity["seed909_used"],
        not integrity["protected_used_for_selection"],
        not integrity["training_changed_from_Level_7_1"],
        integrity["selected_step_registered"],
        integrity["single_checkpoint_selected"],
        integrity["protected_opened_once"],
        integrity["model_fingerprint_unchanged"],
        integrity["all_model_parameters_frozen"],
    ])
    result = {
        "seed": seed,
        "training": training,
        "reached_candidates": True,
        "selection": selection,
        "protected": protected,
        "causal": causal,
        "primary_passed": bool(
            protected["passed"] and causal is not None and causal["passed"]
            and integrity["passed"]
        ),
        "integrity": integrity,
    }
    atomic_save(result_path, result)
    return result


def classify(results: list[dict[str, Any]]) -> dict[str, Any]:
    reached = [row for row in results if row["reached_candidates"]]
    passed = [row for row in results if row["primary_passed"]]
    protected_passed = [
        row for row in results if row["protected"] and row["protected"]["passed"]
    ]
    causal_failed = [
        row for row in protected_passed
        if row["causal"] is not None and not row["causal"]["passed"]
    ]
    if causal_failed:
        classification = "causal_gate_failed"
    elif len(passed) == 2:
        classification = "strong_stable_causal_formation"
    elif len(passed) == 1:
        classification = "conditional_stable_causal_formation"
    elif not reached:
        classification = "formation_curriculum_failed"
    else:
        classification = "retention_checkpoint_selection_failed"
    return {
        "classification": classification,
        "seeds_total": len(results),
        "seeds_reached_candidates": len(reached),
        "seeds_with_eligible_selection": sum(
            bool(row["selection"] and row["selection"]["passed"]) for row in results
        ),
        "seeds_protected_passed": len(protected_passed),
        "seeds_primary_passed": len(passed),
        "strong_passed": classification == "strong_stable_causal_formation",
        "conditional_supported": classification in {
            "strong_stable_causal_formation",
            "conditional_stable_causal_formation",
        },
        "registered_stop_boundary": (
            "Do not add candidates, rerun protected data, extend training, "
            "replace a seed, repair an output head/router, or open seed909."
        ),
    }


def plot_results(results: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for result in results:
        if result["selection"]:
            rows = result["selection"]["candidates"]
            axes[0].plot(
                [row["step"] for row in rows],
                [100 * row["metric"]["query"] for row in rows],
                marker="o", label=f"seed {result['seed']}",
            )
    axes[0].axhline(95, color="#333333", linestyle="--")
    axes[0].set_xlabel("Zero-Probe maintenance step")
    axes[0].set_ylabel("Validation query accuracy (%)")
    axes[0].set_ylim(70, 101)
    axes[0].set_title("Registered candidate panel")
    axes[0].legend()

    labels = [str(row["seed"]) for row in results]
    protected = [
        100 * row["protected"]["query"] if row["protected"] else 0.0
        for row in results
    ]
    axes[1].bar(labels, protected, color="#2e6fbb")
    axes[1].axhline(95, color="#333333", linestyle="--")
    axes[1].set_ylim(0, 101)
    axes[1].set_ylabel("Protected query accuracy (%)")
    axes[1].set_title("One-time protected test")

    causal_results = [row for row in results if row["causal"]]
    if causal_results:
        x = np.arange(len(CONDITIONS))
        width = 0.8 / len(causal_results)
        for index, result in enumerate(causal_results):
            values = [
                100 * result["causal"]["conditions"][name]["query"]
                for name in CONDITIONS
            ]
            axes[2].bar(
                x - 0.4 + width / 2 + index * width, values, width,
                label=f"seed {result['seed']}",
            )
        axes[2].set_xticks(
            x, ["intact", "reset", "zero", "roll", "zero L3", "roll L3", "keep L3"],
            rotation=35, ha="right",
        )
        axes[2].legend()
    else:
        axes[2].text(
            0.5, 0.5, "Not opened\n(upstream gate failed)",
            ha="center", va="center", transform=axes[2].transAxes,
        )
    axes[2].axhline(20, color="#b23a48", linestyle=":")
    axes[2].axhline(90, color="#333333", linestyle="--")
    axes[2].set_ylim(0, 101)
    axes[2].set_ylabel("Causal query accuracy (%)")
    axes[2].set_title("Conditional causal audit")
    fig.suptitle("IST Level 7.2: Validation-Selected Retention", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_smoke(args: argparse.Namespace) -> int:
    args.output = "experiments/level7_2/smoke"
    args.chunk_size = 32
    args.diagnostic_batch_size = 4
    args.validation_samples = 8
    args.protected_samples = 8
    args.causal_samples = 8
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    set_seed(SMOKE_SEED)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=1e-3
    )
    random_step(model, probe, optimizer, args, 2, 2, 0.5, device, dtype)
    checkpoint_path = root / "candidate_roundtrip.pt"
    checkpoint(
        checkpoint_path, model, probe, optimizer,
        {"zero_probe_step": CANDIDATE_STEPS[0]},
    )
    restored_model, _ = load_candidate(checkpoint_path, args, device)
    roundtrip = model_fingerprint(model) == model_fingerprint(restored_model)
    validation = metric_with_interval(evaluate_causal_condition(
        restored_model, args, 2, "intact", args.validation_samples,
        7200023, device, dtype,
    ))
    protected = metric_with_interval(evaluate_causal_condition(
        restored_model, args, 2, "intact", args.protected_samples,
        7200024, device, dtype,
    ))
    causal = [
        evaluate_causal_condition(
            restored_model, args, 2, condition, args.causal_samples,
            7200025, device, dtype,
        )
        for condition in CONDITIONS
    ]
    checkpoint_path.unlink(missing_ok=True)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "seed": SMOKE_SEED,
        "candidate_roundtrip_passed": roundtrip,
        "validation_exercised": validation["samples"] == args.validation_samples,
        "protected_exercised": protected["samples"] == args.protected_samples,
        "causal_conditions_exercised": [row["condition"] for row in causal],
        "passed": bool(
            roundtrip and len(causal) == len(CONDITIONS)
            and validation["samples"] == args.validation_samples
            and protected["samples"] == args.protected_samples
        ),
    }
    atomic_save(result_path, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return run_smoke(args)
    formal_protocol_check(args)
    formal_root = ROOT / args.output
    if args.force and any(formal_root.glob("seed*/protected.json")):
        raise ValueError(
            "Protected test has already been opened; formal --force is prohibited."
        )
    protocol = read_json(STATIC_PREREGISTRATION)
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = formal_root
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "preregistration.json", protocol)
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        result = read_json(result_path)
        print(json.dumps(result["diagnosis"], indent=2))
        return 0
    progress = {
        "stage": "training",
        "formal_seeds": FORMAL_SEEDS,
        "completed_seeds": [],
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    results = []
    for seed in FORMAL_SEEDS:
        folder = root / f"seed{seed}"
        training = train_seed(seed, args, device, dtype, folder)
        progress["stage"] = "selection_and_protected_test"
        progress["active_seed"] = seed
        atomic_save(root / "progress.json", progress)
        result = evaluate_seed(seed, training, args, device, dtype, folder)
        results.append(result)
        progress["completed_seeds"].append(seed)
        progress.pop("active_seed", None)
        atomic_save(root / "progress.json", progress)
        torch.cuda.empty_cache()
    diagnosis = classify(results)
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "formal_seeds_exact": [row["seed"] for row in results] == FORMAL_SEEDS,
        "candidate_steps_exact": CANDIDATE_STEPS == args.candidate_steps,
        "seed909_used": False,
        "old_seed_rescue_used": False,
        "training_changed_from_Level_7_1": False,
        "protected_used_for_selection": False,
        "all_opened_seed_integrity_passed": all(
            row["integrity"].get("passed", True) for row in results
        ),
    }
    integrity["passed"] = all([
        integrity["formal_seeds_exact"],
        integrity["candidate_steps_exact"],
        not integrity["seed909_used"],
        not integrity["old_seed_rescue_used"],
        not integrity["training_changed_from_Level_7_1"],
        not integrity["protected_used_for_selection"],
        integrity["all_opened_seed_integrity_passed"],
    ])
    result = {
        "protocol": protocol,
        "integrity": integrity,
        "runs": results,
        "diagnosis": diagnosis,
    }
    summary = {
        "integrity": integrity,
        "diagnosis": diagnosis,
        "seeds": [
            {
                "seed": row["seed"],
                "reached_candidates": row["reached_candidates"],
                "selection_passed": bool(row["selection"] and row["selection"]["passed"]),
                "selected_step": (
                    row["selection"]["selected"]["step"]
                    if row["selection"] and row["selection"]["selected"] else None
                ),
                "protected_query": row["protected"]["query"] if row["protected"] else None,
                "protected_passed": row["protected"]["passed"] if row["protected"] else None,
                "causal_passed": row["causal"]["passed"] if row["causal"] else None,
                "primary_passed": row["primary_passed"],
                "seconds": row["training"].get("seconds"),
            }
            for row in results
        ],
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_results(results, root / "retention_selection.png")
    atomic_save(root / "progress.json", {
        "stage": "complete",
        "completed_seeds": FORMAL_SEEDS,
        "classification": diagnosis["classification"],
        "integrity_passed": integrity["passed"],
        "seed909_locked": True,
    })
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
