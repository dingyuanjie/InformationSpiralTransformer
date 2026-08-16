"""Level 7.5 prospective cross-initialization C2-to-C4 formation dynamics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from run_level6_6_local import build, fixed_stage, random_step, restore
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save, model_fingerprint
from run_level7_3_local import (
    CONDITIONS,
    evaluate_condition,
    layer_profile,
    read_json,
    sha256_file,
)
from run_level7_4_1_local import (
    atomic_torch_save,
    model_displacement,
    state_dict_fingerprint,
)
from run_level7_4_local import classify_checkpoint


FORMAL_SEEDS = [2203, 2551, 2909]
SMOKE_SEED = 23
ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_5"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
OLD_MODEL_SEEDS = {313, 42, 2026, 7, 1234, 606, 808, 909, 1001, 1217, 1429, 1601, 1879}
TARGET_ROUTE = "l2_core_l3_supported"
WHOLE_CONTROLS = ("reset_all", "zero_all", "batch_roll_all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--fixed-steps", type=int, default=2500)
    parser.add_argument("--fixed-batch-size", type=int, default=16)
    parser.add_argument("--fixed-eval-batch-size", type=int, default=16)
    parser.add_argument("--c2-steps", type=int, default=2500)
    parser.add_argument("--c2-batch-size", type=int, default=8)
    parser.add_argument("--c4-steps", type=int, default=1500)
    parser.add_argument("--c4-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--training-gate", type=float, default=0.95)
    parser.add_argument("--causal-chunks", type=int, default=16)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--causal-eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=7500000)
    parser.add_argument("--formed-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--core-retention-threshold", type=float, default=0.80)
    parser.add_argument("--pair-sufficiency-threshold", type=float, default=0.90)
    parser.add_argument("--pair-gain-threshold", type=float, default=0.03)
    parser.add_argument("--roll-drop-threshold", type=float, default=0.05)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--precursor-intact-threshold", type=float, default=0.75)
    parser.add_argument("--precursor-retention-threshold", type=float, default=0.70)
    parser.add_argument("--output", default="experiments/level7_5/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.behavior_only_gate = True
    # Names expected by the frozen Level 7.3 layer-atlas helper.
    args.intact_threshold = args.formed_threshold
    args.sufficiency_threshold = args.pair_sufficiency_threshold
    return args


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "seeds": FORMAL_SEEDS,
        "chunk_size": 128,
        "fixed_steps": 2500,
        "fixed_batch_size": 16,
        "fixed_eval_batch_size": 16,
        "c2_steps": 2500,
        "c2_batch_size": 8,
        "c4_steps": 1500,
        "c4_batch_size": 4,
        "learning_rate": 1e-3,
        "probe_weight": 0.5,
        "eval_every": 100,
        "eval_batches": 10,
        "eval_batch_size": 8,
        "training_gate": 0.95,
        "causal_chunks": 16,
        "causal_samples": 1024,
        "causal_eval_batch_size": 16,
        "dataset_seed": 7500000,
        "formed_threshold": 0.90,
        "disruption_threshold": 0.20,
        "core_retention_threshold": 0.80,
        "pair_sufficiency_threshold": 0.90,
        "pair_gain_threshold": 0.03,
        "roll_drop_threshold": 0.05,
        "local_threshold": 0.90,
        "precursor_intact_threshold": 0.75,
        "precursor_retention_threshold": 0.70,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.5 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_5/formal":
        raise ValueError("Formal output path is locked")


def validate_static_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("formal_model_seeds") != FORMAL_SEEDS:
        raise RuntimeError("Static Level 7.5 seeds changed")
    if set(FORMAL_SEEDS) & OLD_MODEL_SEEDS:
        raise RuntimeError("A Level 7.5 seed overlaps a previous model seed")
    panel = protocol["fresh_shared_causal_panel"]
    if panel.get("conditions") != CONDITIONS:
        raise RuntimeError("Static Level 7.5 condition order changed")
    if panel.get("dataset_seed") != 7500000:
        raise RuntimeError("Static Level 7.5 dataset seed changed")


def full_checkpoint(
    path: Path,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any],
) -> None:
    atomic_torch_save(
        path,
        {
            "model": model.state_dict(),
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            **payload,
        },
    )


def training_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )


def run_c2_stage(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    folder: Path,
    seed: int,
) -> dict[str, Any]:
    path = folder / "curriculum_stage1.pt"
    if path.exists() and not args.force:
        state = restore(path, model, probe, optimizer, device)
        return state["stage_result"]
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    history: list[dict[str, Any]] = []
    consecutive = 0
    metric = None
    eval_args = training_eval_args(args)
    for step in range(1, args.c2_steps + 1):
        model.train()
        probe.train()
        random_step(
            model,
            probe,
            optimizer,
            args,
            2,
            args.c2_batch_size,
            args.probe_weight,
            device,
            dtype,
        )
        if step == 1 or step % args.eval_every == 0:
            metric = evaluate(model, probe, eval_args, 2, device, dtype)
            record = {"stage": 1, "step": step, **metric}
            history.append(record)
            atomic_save(folder / "c2_progress.json", history)
            ok = metric["query"] >= args.training_gate
            consecutive = consecutive + 1 if ok else 0
            print(
                f"seed={seed} C2 step={step} query={metric['query']:.2%} "
                f"probe={metric['probe_min']:.2%} consecutive={consecutive}",
                flush=True,
            )
            if consecutive >= 2:
                break
    result = {
        "stage": 1,
        "chunks": 2,
        "passed": consecutive >= 2,
        "steps": step,
        "consecutive_behavior_passes": consecutive,
        "validation": metric,
        "history": history,
        "budget_extended": False,
    }
    full_checkpoint(
        path,
        model,
        probe,
        optimizer,
        {"stage_result": result, "stages": [result], "history": history},
    )
    return result


def stage2_snapshot_path(folder: Path, step: int) -> Path:
    return folder / "stage2" / f"model_step{step:04d}.pt"


def save_stage2_state(
    folder: Path,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
    step: int,
    consecutive: int,
    history: list[dict[str, Any]],
) -> None:
    stage2 = folder / "stage2"
    stage2.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        stage2_snapshot_path(folder, step),
        {
            "model": model.state_dict(),
            "model_seed": seed,
            "stage": 2,
            "chunks": 4,
            "stage_step": step,
            "validation": history[-1],
        },
    )
    full_checkpoint(
        stage2 / "resume.pt",
        model,
        probe,
        optimizer,
        {
            "model_seed": seed,
            "stage": 2,
            "stage_step": step,
            "consecutive_behavior_passes": consecutive,
            "history": history,
        },
    )


def run_c4_stage(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    folder: Path,
    seed: int,
) -> dict[str, Any]:
    result_path = folder / "stage2_training_result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    resume_path = folder / "stage2" / "resume.pt"
    if resume_path.exists() and not args.force:
        state = restore(resume_path, model, probe, optimizer, device)
        last_step = int(state["stage_step"])
        consecutive = int(state["consecutive_behavior_passes"])
        history = state["history"]
        print(f"seed={seed} resumed C4 at step={last_step}", flush=True)
    else:
        last_step = 0
        consecutive = 0
        history = []
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    eval_args = training_eval_args(args)
    stop_step = last_step if consecutive >= 2 else None
    if consecutive < 2:
        for step in range(last_step + 1, args.c4_steps + 1):
            model.train()
            probe.train()
            random_step(
                model,
                probe,
                optimizer,
                args,
                4,
                args.c4_batch_size,
                args.probe_weight,
                device,
                dtype,
            )
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(model, probe, eval_args, 4, device, dtype)
                history.append({"stage": 2, "step": step, **metric})
                ok = metric["query"] >= args.training_gate
                consecutive = consecutive + 1 if ok else 0
                save_stage2_state(
                    folder, model, probe, optimizer, seed, step, consecutive, history
                )
                atomic_save(
                    folder / "stage2_progress.json",
                    {
                        "model_seed": seed,
                        "completed_step": step,
                        "consecutive_behavior_passes": consecutive,
                        "history": history,
                        "milestone_steps": [row["step"] for row in history],
                    },
                )
                print(
                    f"seed={seed} C4 step={step} query={metric['query']:.2%} "
                    f"probe={metric['probe_min']:.2%} consecutive={consecutive}",
                    flush=True,
                )
                if consecutive >= 2:
                    stop_step = step
                    break
    if stop_step is None:
        stop_step = args.c4_steps
    result = {
        "stage": 2,
        "chunks": 4,
        "passed": consecutive >= 2,
        "stop_step": stop_step,
        "consecutive_behavior_passes": consecutive,
        "milestone_steps": [row["step"] for row in history],
        "history": history,
        "final_validation": history[-1] if history else None,
        "budget_extended": False,
        "continued_beyond_C4": False,
    }
    atomic_save(result_path, result)
    return result


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
    started = time.perf_counter()
    set_seed(seed)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    fixed = fixed_stage(model, probe, optimizer, args, device, dtype, folder, seed)
    if not fixed["passed"]:
        result = {
            "seed": seed,
            "passed_prerequisites": False,
            "failed_phase": "fixed",
            "fixed": fixed,
            "seconds": time.perf_counter() - started,
            "budget_extended": False,
        }
        atomic_save(result_path, result)
        del model, probe, optimizer
        torch.cuda.empty_cache()
        return result
    set_seed(seed + 20000)
    c2 = run_c2_stage(model, probe, optimizer, args, device, dtype, folder, seed)
    if not c2["passed"]:
        result = {
            "seed": seed,
            "passed_prerequisites": False,
            "failed_phase": "C2",
            "fixed": fixed,
            "C2": c2,
            "seconds": time.perf_counter() - started,
            "budget_extended": False,
        }
        atomic_save(result_path, result)
        del model, probe, optimizer
        torch.cuda.empty_cache()
        return result
    c4 = run_c4_stage(model, probe, optimizer, args, device, dtype, folder, seed)
    result = {
        "seed": seed,
        "passed_prerequisites": True,
        "failed_phase": None,
        "fixed": fixed,
        "C2": c2,
        "C4": c4,
        "seconds": time.perf_counter() - started,
        "budget_extended": False,
        "continued_beyond_C4": False,
    }
    atomic_save(result_path, result)
    del model, probe, optimizer
    torch.cuda.empty_cache()
    return result


def causal_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        chunks=args.causal_chunks,
        chunk_size=args.chunk_size,
        samples=args.causal_samples,
        eval_batch_size=args.causal_eval_batch_size,
        dataset_seed=args.dataset_seed,
    )


def milestone_sources(
    seed: int, training: dict[str, Any], folder: Path
) -> list[dict[str, Any]]:
    if not training.get("passed_prerequisites"):
        return []
    specs = [(0, folder / "curriculum_stage1.pt", "C2_endpoint")]
    specs.extend(
        (
            int(step),
            stage2_snapshot_path(folder, int(step)),
            "prospective_C4_training",
        )
        for step in training["C4"]["milestone_steps"]
    )
    sources = []
    for step, path, origin in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(
            {
                "id": f"step_{step:04d}",
                "seed": seed,
                "step": step,
                "origin": origin,
                "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256_file(path),
            }
        )
    return sources


def validate_resumed_metrics(
    metrics: dict[str, dict[str, Any]], args: argparse.Namespace
) -> None:
    unexpected = set(metrics) - set(CONDITIONS)
    if unexpected:
        raise RuntimeError(f"Unexpected resumed conditions: {sorted(unexpected)}")
    for name, metric in metrics.items():
        if metric.get("condition") != name:
            raise RuntimeError(f"Resumed condition label mismatch: {name}")
        if (
            metric.get("samples") != args.causal_samples
            or metric.get("chunks") != args.causal_chunks
        ):
            raise RuntimeError(f"Resumed causal protocol mismatch: {name}")


def profile_checkpoint(
    metrics: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    targeted = classify_checkpoint(metrics, args)
    atlas_args = argparse.Namespace(
        disruption_threshold=args.disruption_threshold,
        sufficiency_threshold=args.pair_sufficiency_threshold,
        intact_threshold=args.formed_threshold,
        local_threshold=args.local_threshold,
    )
    atlas = layer_profile(metrics, atlas_args)
    query = {name: metrics[name]["query"] for name in CONDITIONS}
    pair_gain = query["keep_l2_l3"] - query["keep_l2"]
    roll_l3_drop = query["intact"] - query["batch_roll_l3"]
    precursor = bool(
        targeted["route_class"] != TARGET_ROUTE
        and query["intact"] >= args.precursor_intact_threshold
        and atlas["minimum_local"] >= args.local_threshold
        and max(query[name] for name in WHOLE_CONTROLS)
        <= args.disruption_threshold
        and query["zero_l2"] <= args.disruption_threshold
        and query["batch_roll_l2"] <= args.disruption_threshold
        and query["keep_l2"] >= args.precursor_retention_threshold
        and query["zero_l3"] >= args.precursor_retention_threshold
        and query["keep_l3"] <= args.disruption_threshold
        and query["keep_l2_l3"] >= args.precursor_intact_threshold
        and pair_gain >= args.pair_gain_threshold
        and roll_l3_drop >= args.roll_drop_threshold
    )
    return {
        **targeted,
        "l2_causal_precursor": precursor,
        "layer_atlas": atlas,
    }


def run_causal_milestone(
    source: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    seed_root: Path,
    base_model_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    folder = seed_root / "causal" / source["id"]
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        result = read_json(result_path)
        if result["source"]["checkpoint_sha256"] != source["checkpoint_sha256"]:
            raise RuntimeError(f"Cached source mismatch: seed={source['seed']} {source['id']}")
        return result
    state = torch.load(ROOT / source["checkpoint"], map_location="cpu", weights_only=False)
    expected_fingerprint = state_dict_fingerprint(state["model"])
    displacement = model_displacement(state["model"], base_model_state)
    model, probe = build(device, args.chunk_size)
    del probe
    model.load_state_dict(state["model"])
    del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fingerprint_before = model_fingerprint(model)
    if fingerprint_before != expected_fingerprint:
        raise RuntimeError(f"Model fingerprint mismatch: {source['id']}")
    progress_path = folder / "condition_progress.json"
    metrics = read_json(progress_path) if progress_path.exists() and not args.force else {}
    validate_resumed_metrics(metrics, args)
    panel_args = causal_args(args)
    for condition in CONDITIONS:
        if condition in metrics:
            continue
        metric = evaluate_condition(model, panel_args, condition, device, dtype)
        metrics[condition] = metric
        atomic_save(progress_path, metrics)
        print(
            f"seed={source['seed']} milestone={source['id']} "
            f"condition={condition} query={metric['query']:.2%} "
            f"local={metric['local']:.2%}",
            flush=True,
        )
    fingerprint_after = model_fingerprint(model)
    profile = profile_checkpoint(metrics, args)
    integrity = {
        "checkpoint_sha256": source["checkpoint_sha256"],
        "expected_model_fingerprint": expected_fingerprint,
        "model_fingerprint_before": fingerprint_before,
        "model_fingerprint_after": fingerprint_after,
        "model_fingerprint_unchanged": fingerprint_before == fingerprint_after,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "all_conditions_present": list(metrics) == CONDITIONS,
        "fixed_samples_every_condition": all(
            row["samples"] == args.causal_samples for row in metrics.values()
        ),
        "shared_dataset_seed": args.dataset_seed,
    }
    integrity["passed"] = bool(
        integrity["model_fingerprint_unchanged"]
        and integrity["all_parameters_frozen"]
        and integrity["all_conditions_present"]
        and integrity["fixed_samples_every_condition"]
    )
    result = {
        "id": source["id"],
        "seed": source["seed"],
        "step": source["step"],
        "source": source,
        "model_displacement_from_C2": displacement,
        "metrics": metrics,
        "profile": profile,
        "integrity": integrity,
    }
    atomic_save(result_path, result)
    del model
    torch.cuda.empty_cache()
    return result


def first_step(results: list[dict[str, Any]], key: str) -> int | None:
    row = next((item for item in results if item["profile"].get(key)), None)
    return int(row["step"]) if row is not None else None


def diagnose_seed(
    seed: int, training: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    if not training.get("passed_prerequisites"):
        return {
            "seed": seed,
            "classification": "training_prerequisite_failed",
            "failed_phase": training.get("failed_phase"),
            "milestones": 0,
            "first_precursor_step": None,
            "first_target_step": None,
            "endpoint_route": None,
        }
    expected_steps = [0, *training["C4"]["milestone_steps"]]
    if [row["step"] for row in results] != expected_steps:
        raise RuntimeError(f"Seed {seed} milestone trajectory is incomplete")
    target_flags = [row["profile"]["route_class"] == TARGET_ROUTE for row in results]
    precursor_flags = [row["profile"]["l2_causal_precursor"] for row in results]
    formed_causal_flags = [
        row["profile"]["behavior_formed"] and row["profile"]["whole_memory_causal"]
        for row in results
    ]
    first_target = next((row["step"] for row, flag in zip(results, target_flags) if flag), None)
    first_precursor = next(
        (row["step"] for row, flag in zip(results, precursor_flags) if flag), None
    )
    endpoint_target = target_flags[-1]
    target_regression = any(
        left and not right for left, right in zip(target_flags, target_flags[1:])
    )
    if target_flags[0]:
        classification = "l2_target_present_at_C2_endpoint"
    elif (
        endpoint_target
        and not target_regression
        and first_precursor is not None
        and 0 < first_precursor < first_target
    ):
        classification = "l2_two_stage_replication_within_C4"
    elif endpoint_target and target_regression:
        classification = "l2_target_recovered_after_regression"
    elif endpoint_target and first_precursor == 0:
        classification = "l2_precursor_present_at_C2_then_target_by_C4"
    elif endpoint_target:
        classification = "l2_target_without_registered_precursor"
    elif any(target_flags):
        classification = "l2_target_regressed_before_C4_endpoint"
    elif any(precursor_flags):
        classification = "l2_precursor_without_target"
    elif any(formed_causal_flags):
        classification = "alternative_formed_causal_route"
    else:
        classification = "no_registered_16chunk_formation"
    route_transitions = []
    for previous, current in zip(results, results[1:]):
        old_class = previous["profile"]["route_class"]
        new_class = current["profile"]["route_class"]
        if old_class != new_class:
            route_transitions.append(
                {
                    "from_step": previous["step"],
                    "to_step": current["step"],
                    "from_class": old_class,
                    "to_class": new_class,
                }
            )
    endpoint = results[-1]
    return {
        "seed": seed,
        "classification": classification,
        "milestones": len(results),
        "C4_training_gate_passed": training["C4"]["passed"],
        "C4_stop_step": training["C4"]["stop_step"],
        "first_precursor_step": first_precursor,
        "first_target_step": first_target,
        "endpoint_route": endpoint["profile"]["route_class"],
        "endpoint_intact_query": endpoint["metrics"]["intact"]["query"],
        "endpoint_whole_memory_causal": endpoint["profile"]["whole_memory_causal"],
        "endpoint_layer_signature": endpoint["profile"]["layer_atlas"]["signature_key"],
        "target_membership_changes": sum(
            left != right for left, right in zip(target_flags, target_flags[1:])
        ),
        "target_regression_observed": target_regression,
        "route_transitions": route_transitions,
    }


def diagnose_cohort(seed_diagnoses: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in seed_diagnoses:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    two_stage = counts.get("l2_two_stage_replication_within_C4", 0)
    l2_endpoint = sum(
        row.get("endpoint_route") == TARGET_ROUTE for row in seed_diagnoses
    )
    causal_endpoint = sum(
        bool(row.get("endpoint_whole_memory_causal")) for row in seed_diagnoses
    )
    if two_stage >= 2:
        classification = "strong_prospective_two_stage_replication"
    elif two_stage == 1:
        classification = "conditional_prospective_two_stage_replication"
    elif l2_endpoint:
        classification = "L2_formation_without_two_stage_replication"
    elif causal_endpoint:
        classification = "alternative_route_formation_observed"
    else:
        classification = "prospective_formation_dynamics_not_replicated"
    signatures = sorted(
        {
            row["endpoint_layer_signature"]
            for row in seed_diagnoses
            if row.get("endpoint_whole_memory_causal")
        }
    )
    return {
        "classification": classification,
        "formal_seeds": FORMAL_SEEDS,
        "seeds_total": len(seed_diagnoses),
        "within_C4_two_stage_replications": two_stage,
        "L2_target_endpoints": l2_endpoint,
        "whole_memory_causal_endpoints": causal_endpoint,
        "per_seed_outcome_counts": counts,
        "endpoint_signature_count_among_causal_seeds": len(signatures),
        "endpoint_route_heterogeneity_observed": len(signatures) >= 2,
        "registered_stop_boundary": (
            "Report this fixed three-seed classification; do not replace, rescue, "
            "extend, or selectively continue a seed."
        ),
    }


def plot_trajectories(
    seed_runs: list[dict[str, Any]], path: Path
) -> None:
    fig, axes = plt.subplots(len(seed_runs), 2, figsize=(17, 5 * len(seed_runs)))
    if len(seed_runs) == 1:
        axes = np.array([axes])
    for row_index, seed_run in enumerate(seed_runs):
        seed = seed_run["seed"]
        results = seed_run["milestones"]
        if not results:
            for axis in axes[row_index]:
                axis.text(0.5, 0.5, "training prerequisite failed", ha="center")
                axis.set_axis_off()
            continue
        steps = [row["step"] for row in results]
        left = axes[row_index, 0]
        for condition, label in [
            ("intact", "intact"),
            ("reset_all", "reset all"),
            ("zero_all", "zero all"),
            ("batch_roll_all", "roll all"),
        ]:
            left.plot(
                steps,
                [100 * row["metrics"][condition]["query"] for row in results],
                marker="o",
                label=label,
            )
        left.axhline(20, color="#b23a48", linestyle=":")
        left.axhline(75, color="#8c6d31", linestyle="-.")
        left.axhline(90, color="#333333", linestyle="--")
        left.set_ylim(0, 105)
        left.set_title(f"seed {seed}: behavior and whole-Memory controls")
        left.set_ylabel("Query accuracy (%)")
        left.legend(ncol=2, fontsize=8)
        left.grid(alpha=0.2)

        right = axes[row_index, 1]
        for condition, label in [
            ("keep_l1", "keep L1"),
            ("keep_l2", "keep L2"),
            ("keep_l3", "keep L3"),
            ("keep_l1_l2", "keep L1+L2"),
            ("keep_l1_l3", "keep L1+L3"),
            ("keep_l2_l3", "keep L2+L3"),
        ]:
            right.plot(
                steps,
                [100 * row["metrics"][condition]["query"] for row in results],
                marker="o",
                label=label,
            )
        right.axhline(70, color="#8c6d31", linestyle=":")
        right.axhline(90, color="#333333", linestyle="--")
        right.set_ylim(0, 105)
        right.set_title(f"seed {seed}: single/pair retention")
        right.legend(ncol=2, fontsize=8)
        right.grid(alpha=0.2)
        for axis in axes[row_index]:
            axis.set_xlabel("C4 training step (0 = C2 endpoint)")
    fig.suptitle("IST Level 7.5: Prospective Cross-Initialization Formation Dynamics")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_integrity(
    args: argparse.Namespace,
    seed_runs: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    all_milestones = [
        milestone for seed_run in seed_runs for milestone in seed_run["milestones"]
    ]
    expected_evaluated = sum(
        1 + len(seed_run["training"].get("C4", {}).get("milestone_steps", []))
        for seed_run in seed_runs
        if seed_run["training"].get("passed_prerequisites")
    )
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "formal_seeds_exact": [row["seed"] for row in seed_runs] == FORMAL_SEEDS,
        "all_seeds_new": not bool(set(FORMAL_SEEDS) & OLD_MODEL_SEEDS),
        "no_seed_replacement": len(seed_runs) == len(FORMAL_SEEDS),
        "evaluated_milestones_expected": expected_evaluated,
        "evaluated_milestones_completed": len(all_milestones),
        "all_milestone_integrity_passed": all(
            row["integrity"]["passed"] for row in all_milestones
        ),
        "all_conditions_exact": all(
            list(row["metrics"]) == CONDITIONS for row in all_milestones
        ),
        "fixed_N_completed": all(
            metric["samples"] == args.causal_samples
            for row in all_milestones
            for metric in row["metrics"].values()
        ),
        "shared_fresh_dataset_seed": args.dataset_seed,
        "no_training_beyond_C4": all(
            not row["training"].get("continued_beyond_C4", False)
            for row in seed_runs
        ),
        "no_budget_extension": all(
            not row["training"].get("budget_extended", False)
            for row in seed_runs
        ),
        "seed909_used": False,
        "static_protocol_level": protocol["level"],
    }
    integrity["passed"] = bool(
        integrity["formal_seeds_exact"]
        and integrity["all_seeds_new"]
        and integrity["no_seed_replacement"]
        and integrity["evaluated_milestones_expected"]
        == integrity["evaluated_milestones_completed"]
        and integrity["all_milestone_integrity_passed"]
        and integrity["all_conditions_exact"]
        and integrity["fixed_N_completed"]
        and integrity["no_training_beyond_C4"]
        and integrity["no_budget_extension"]
        and not integrity["seed909_used"]
    )
    return integrity


def run_smoke(args: argparse.Namespace) -> int:
    args.output = "experiments/level7_5/smoke"
    args.seeds = [SMOKE_SEED]
    args.chunk_size = 32
    args.c4_steps = 3
    args.c4_batch_size = 2
    args.eval_every = 1
    args.eval_batches = 1
    args.eval_batch_size = 4
    args.training_gate = 2.0
    args.causal_chunks = 4
    args.causal_samples = 32
    args.causal_eval_batch_size = 8
    args.dataset_seed = 7599999
    args.force = True
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output
    seed_root = root / f"seed{SMOKE_SEED}"
    seed_root.mkdir(parents=True, exist_ok=True)
    set_seed(SMOKE_SEED)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    random_step(model, probe, optimizer, args, 2, 2, 0.5, device, dtype)
    c2_path = seed_root / "curriculum_stage1.pt"
    full_checkpoint(
        c2_path,
        model,
        probe,
        optimizer,
        {"stage_result": {"passed": True}, "stages": [], "history": []},
    )
    c4 = run_c4_stage(model, probe, optimizer, args, device, dtype, seed_root, SMOKE_SEED)
    training = {
        "seed": SMOKE_SEED,
        "passed_prerequisites": True,
        "C4": c4,
        "budget_extended": False,
        "continued_beyond_C4": False,
    }
    sources = milestone_sources(SMOKE_SEED, training, seed_root)
    base_state = torch.load(c2_path, map_location="cpu", weights_only=False)["model"]
    milestones = [
        run_causal_milestone(source, args, device, dtype, seed_root, base_state)
        for source in sources
    ]
    diagnosis = diagnose_seed(SMOKE_SEED, training, milestones)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "milestone_steps": [row["step"] for row in milestones],
        "conditions_per_milestone": len(CONDITIONS),
        "all_milestone_integrity_passed": all(
            row["integrity"]["passed"] for row in milestones
        ),
        "diagnosis_exercised": diagnosis["classification"],
        "resume_checkpoint_present": (seed_root / "stage2" / "resume.pt").is_file(),
    }
    result["passed"] = bool(
        result["milestone_steps"] == [0, 1, 2, 3]
        and result["conditions_per_milestone"] == 16
        and result["all_milestone_integrity_passed"]
        and result["resume_checkpoint_present"]
    )
    atomic_save(root / "result.json", result)
    if seed_root.resolve().parent != root.resolve():
        raise RuntimeError("Refusing to clean an unexpected smoke directory")
    shutil.rmtree(seed_root)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return run_smoke(args)
    formal_protocol_check(args)
    protocol = read_json(STATIC_PREREGISTRATION)
    validate_static_protocol(protocol)
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0
    started = time.perf_counter()
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "preregistration.json", protocol)
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        result = read_json(result_path)
        print(json.dumps(result["diagnosis"], indent=2))
        return 0
    progress: dict[str, Any] = {
        "stage": "prospective_training_and_causal_trajectory",
        "formal_seeds": FORMAL_SEEDS,
        "completed_seeds": [],
        "active_seed": None,
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    seed_runs: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        progress["active_seed"] = seed
        atomic_save(root / "progress.json", progress)
        seed_root = root / f"seed{seed}"
        training = train_seed(seed, args, device, dtype, seed_root)
        sources = milestone_sources(seed, training, seed_root)
        milestones: list[dict[str, Any]] = []
        if sources:
            c2_state = torch.load(
                seed_root / "curriculum_stage1.pt",
                map_location="cpu",
                weights_only=False,
            )
            base_model_state = c2_state["model"]
            del c2_state
            for source in sources:
                milestones.append(
                    run_causal_milestone(
                        source, args, device, dtype, seed_root, base_model_state
                    )
                )
                progress["active_milestone"] = source["id"]
                atomic_save(root / "progress.json", progress)
            del base_model_state
        diagnosis = diagnose_seed(seed, training, milestones)
        seed_result = {
            "seed": seed,
            "training": training,
            "milestones": milestones,
            "diagnosis": diagnosis,
        }
        atomic_save(seed_root / "result.json", seed_result)
        seed_runs.append(seed_result)
        progress["completed_seeds"].append(seed)
        progress.pop("active_milestone", None)
        atomic_save(root / "progress.json", progress)
        torch.cuda.empty_cache()
    seed_diagnoses = [row["diagnosis"] for row in seed_runs]
    diagnosis = diagnose_cohort(seed_diagnoses)
    integrity = build_integrity(args, seed_runs, protocol)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "protocol": protocol,
        "runs": seed_runs,
        "diagnosis": diagnosis,
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    summary = {
        "diagnosis": diagnosis,
        "integrity": integrity,
        "seeds": [
            {
                "seed": row["seed"],
                "diagnosis": row["diagnosis"],
                "trajectory": [
                    {
                        "step": milestone["step"],
                        "route_class": milestone["profile"]["route_class"],
                        "l2_causal_precursor": milestone["profile"]["l2_causal_precursor"],
                        "behavior_formed": milestone["profile"]["behavior_formed"],
                        "whole_memory_causal": milestone["profile"]["whole_memory_causal"],
                        "intact_query": milestone["metrics"]["intact"]["query"],
                        "layer_atlas": milestone["profile"]["layer_atlas"],
                        "effects": milestone["profile"]["effects"],
                        "model_displacement_from_C2": milestone[
                            "model_displacement_from_C2"
                        ],
                    }
                    for milestone in row["milestones"]
                ],
            }
            for row in seed_runs
        ],
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_trajectories(seed_runs, root / "prospective_formation_dynamics.png")
    atomic_save(
        root / "progress.json",
        {
            "stage": "complete",
            "formal_seeds": FORMAL_SEEDS,
            "completed_seeds": FORMAL_SEEDS,
            "classification": diagnosis["classification"],
            "integrity_passed": integrity["passed"],
            "seed909_locked": True,
        },
    )
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
