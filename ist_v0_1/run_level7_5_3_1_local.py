"""Level 7.5.3.1: unsuppressed recovery dynamics from all counterfactual endpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_level6_2_local import evaluate
from run_level6_6_local import build, random_step, restore
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save, model_fingerprint
from run_level7_3_local import CONDITIONS, evaluate_condition, read_json, sha256_file
from run_level7_4_1_local import atomic_torch_save, canonical_fingerprint, state_dict_fingerprint
from run_level7_5_local import profile_checkpoint
from run_level7_5_3_local import save_resume


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_5_3_1"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
PARENT_RESULT = ROOT / "experiments/level7_5_3/formal/result.json"
PARENT_RESULT_SHA256 = "b972ce9cb9b7136b3126bebbf1ab151d4eb923147b8b27170ee052f40a8d5716"
FORMAL_SEEDS = [1879, 2203, 2551, 2909]
BRANCH_NAMES = [
    "intact_replay",
    "selected_layer_suppression",
    "other_layer_suppression",
]
RECOVERY_MILESTONES = [0, 100, 300, 600, 1000]
SCREEN_CONDITIONS = ["intact", "zero_l2", "zero_l3", "keep_l2", "keep_l3"]
SCREEN_DATASET_SEED = 7531000
CONFIRM_DATASET_SEED = 7531001


def source(
    seed: int,
    branch: str,
    expected_route: str,
    baseline_route: str,
    baseline_intact_query: float,
    sha256: str,
    size: int,
) -> dict[str, Any]:
    return {
        "id": f"seed{seed}_{branch}",
        "seed": seed,
        "branch": branch,
        "expected_route": expected_route,
        "baseline_route": baseline_route,
        "baseline_intact_query": baseline_intact_query,
        "checkpoint": f"experiments/level7_5_3/formal/seed{seed}/{branch}/C4_endpoint.pt",
        "checkpoint_sha256": sha256,
        "checkpoint_size_bytes": size,
    }


SOURCE_SPECS = [
    source(1879, "intact_replay", "l2_core_l3_supported", "l2_core_l3_supported", 0.947265625, "4330acaf0cb3100bb2595cd1852fab24c9dc0c071b1f3007123bd1e0862196ed", 4325455),
    source(1879, "selected_layer_suppression", "l2_core_l3_supported", "unformed_behavior", 0.8359375, "a87914418a5e3b03a49cb86b3391ee54d48f53566126d198e66429b3302a5071", 4325455),
    source(1879, "other_layer_suppression", "l2_core_l3_supported", "unformed_behavior", 0.82763671875, "c612b514698e65ade7f5b7110bf5ce6407a586d37a43de6b96d5f18bd0748740", 4325455),
    source(2203, "intact_replay", "l3_core", "l3_core", 0.97119140625, "bfb5cca5098dd5a33b01694fab2673a67e612d029dcd5f7508a1d2976145a35b", 4321743),
    source(2203, "selected_layer_suppression", "l3_core", "unformed_behavior", 0.77587890625, "0da36751fac997921584685d5a0c0b4ea08697308de7bdc427c00fd9fb0e812d", 4321743),
    source(2203, "other_layer_suppression", "l3_core", "unformed_behavior", 0.86181640625, "587bf4109ae77d8e2e6045d653bc1e55cce21dcc14c0d1c636762abb70fee7df", 4321743),
    source(2551, "intact_replay", "l3_core", "l3_core", 0.96435546875, "97ed384d857395e93816fb5a0fe87189573cdee3cbb2728957103d8e3a54ff5a", 4320911),
    source(2551, "selected_layer_suppression", "l3_core", "l3_core", 0.93017578125, "14c776f78821f968b0caf909312b863352b8fd9d887d970426245f3cf41560a2", 4320911),
    source(2551, "other_layer_suppression", "l3_core", "unformed_behavior", 0.67578125, "f515e6a85d22f6932584f78c07edaef8e93aaaa5a5ec2a08dd9347eebcc5b854", 4320911),
    source(2909, "intact_replay", "l3_core", "l3_core", 0.91796875, "9862476ac927196b36da89ce25500b4a70de4347b80bf7bedc23b3b0f2358fe8", 4321167),
    source(2909, "selected_layer_suppression", "l3_core", "l3_core", 0.92626953125, "07238dcc5e475a0146d8ad56c3e00ab2f236b9bf86a546af7ba2ad8d05131b62", 4321167),
    source(2909, "other_layer_suppression", "l3_core", "unformed_behavior", 0.89892578125, "913f00d97b5851b862ad1c086158394b45da1158851967564c9ed29c8397482f", 4321167),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--recovery-steps", type=int, default=1000)
    parser.add_argument("--training-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--causal-chunks", type=int, default=16)
    parser.add_argument("--screen-samples", type=int, default=1024)
    parser.add_argument("--screen-eval-batch-size", type=int, default=16)
    parser.add_argument("--screen-dataset-seed", type=int, default=SCREEN_DATASET_SEED)
    parser.add_argument("--confirm-samples", type=int, default=2048)
    parser.add_argument("--confirm-eval-batch-size", type=int, default=16)
    parser.add_argument("--confirm-dataset-seed", type=int, default=CONFIRM_DATASET_SEED)
    parser.add_argument("--formed-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--core-retention-threshold", type=float, default=0.80)
    parser.add_argument("--pair-sufficiency-threshold", type=float, default=0.90)
    parser.add_argument("--pair-gain-threshold", type=float, default=0.03)
    parser.add_argument("--roll-drop-threshold", type=float, default=0.05)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--precursor-intact-threshold", type=float, default=0.75)
    parser.add_argument("--precursor-retention-threshold", type=float, default=0.70)
    parser.add_argument("--output", default="experiments/level7_5_3_1/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.behavior_only_gate = True
    args.intact_threshold = args.formed_threshold
    args.sufficiency_threshold = args.pair_sufficiency_threshold
    return args


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "chunk_size": 128,
        "recovery_steps": 1000,
        "training_batch_size": 4,
        "learning_rate": 1e-3,
        "probe_weight": 0.5,
        "eval_every": 100,
        "eval_batches": 10,
        "eval_batch_size": 8,
        "causal_chunks": 16,
        "screen_samples": 1024,
        "screen_eval_batch_size": 16,
        "screen_dataset_seed": SCREEN_DATASET_SEED,
        "confirm_samples": 2048,
        "confirm_eval_batch_size": 16,
        "confirm_dataset_seed": CONFIRM_DATASET_SEED,
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
        raise ValueError(f"Formal Level 7.5.3.1 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_5_3_1/formal":
        raise ValueError("Formal Level 7.5.3.1 output path is locked")


def validate_static_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("frozen_recovery_sources") != SOURCE_SPECS:
        raise RuntimeError("Static Level 7.5.3.1 recovery sources changed")
    if protocol.get("recovery_milestones") != RECOVERY_MILESTONES:
        raise RuntimeError("Static recovery milestones changed")
    screen = protocol.get("trajectory_screen_panel", {})
    if screen.get("conditions") != SCREEN_CONDITIONS:
        raise RuntimeError("Static screen conditions changed")
    if screen.get("dataset_seed") != SCREEN_DATASET_SEED:
        raise RuntimeError("Static screen dataset changed")
    confirm = protocol.get("final_confirmation_panel", {})
    if confirm.get("conditions") != CONDITIONS:
        raise RuntimeError("Static confirmation conditions changed")
    if confirm.get("dataset_seed") != CONFIRM_DATASET_SEED:
        raise RuntimeError("Static confirmation dataset changed")


def validate_sources(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_static_protocol(protocol)
    if not PARENT_RESULT.is_file() or sha256_file(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise RuntimeError("Frozen Level 7.5.3 result changed")
    parent = read_json(PARENT_RESULT)
    if not parent["integrity"]["passed"]:
        raise RuntimeError("Level 7.5.3 parent integrity failed")
    if parent["diagnosis"]["classification"] != "transient_suppression_disrupts_routes_nonspecifically":
        raise RuntimeError("Unexpected Level 7.5.3 parent classification")
    parent_panels = {
        (int(row["seed"]), row["branch"]): row for row in parent["endpoint_panels"]
    }
    parent_training = {
        (int(row["seed"]), row["branch"]): row for row in parent["training_branches"]
    }
    audit = []
    for spec in SOURCE_SPECS:
        path = ROOT / spec["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != spec["checkpoint_sha256"] or size != spec["checkpoint_size_bytes"]:
            raise RuntimeError(f"Frozen recovery source changed: {spec['id']}")
        key = (spec["seed"], spec["branch"])
        parent_panel = parent_panels[key]
        parent_train = parent_training[key]
        if parent_panel["profile"]["route_class"] != spec["baseline_route"]:
            raise RuntimeError(f"Baseline route changed: {spec['id']}")
        if parent_panel["metrics"]["intact"]["query"] != spec["baseline_intact_query"]:
            raise RuntimeError(f"Baseline query changed: {spec['id']}")
        if parent_train["endpoint_checkpoint_sha256"] != digest:
            raise RuntimeError(f"Parent endpoint hash mismatch: {spec['id']}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model", "probe", "optimizer", "cpu_rng", "cuda_rng"}
        if not required.issubset(state):
            raise RuntimeError(f"Recovery state incomplete: {spec['id']}")
        audit.append(
            {
                **spec,
                "observed_checkpoint_sha256": digest,
                "observed_checkpoint_size_bytes": size,
                "model_fingerprint": state_dict_fingerprint(state["model"]),
                "probe_fingerprint": state_dict_fingerprint(state["probe"]),
                "optimizer_fingerprint": canonical_fingerprint(state["optimizer"]),
                "source_validation_passed": True,
            }
        )
        del state
    return audit, {
        "result": str(PARENT_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": PARENT_RESULT_SHA256,
        "integrity_passed": parent["integrity"]["passed"],
        "classification": parent["diagnosis"]["classification"],
        "baseline_unformed_branches": sum(
            row["baseline_route"] == "unformed_behavior" for row in SOURCE_SPECS
        ),
        "baseline_formed_branches": sum(
            row["baseline_route"] != "unformed_behavior" for row in SOURCE_SPECS
        ),
    }


def snapshot_path(folder: Path, step: int) -> Path:
    return folder / "recovery" / f"model_step{step:04d}.pt"


def save_snapshot(
    folder: Path,
    spec: dict[str, Any],
    step: int,
    model: torch.nn.Module,
    validation: dict[str, Any],
) -> None:
    path = snapshot_path(folder, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "model": model.state_dict(),
            "seed": spec["seed"],
            "branch": spec["branch"],
            "recovery_step": step,
            "validation": validation,
            "suppression_active": False,
        },
    )


def run_recovery(
    spec: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    folder = root / f"seed{spec['seed']}" / spec["branch"]
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "recovery_result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    resume_path = folder / "recovery" / "resume.pt"
    if resume_path.exists() and not args.force:
        state = restore(resume_path, model, probe, optimizer, device)
        last_step = int(state["recovery_step"])
        history = state["recovery_history"]
        print(
            f"seed={spec['seed']} branch={spec['branch']} resumed recovery step={last_step}",
            flush=True,
        )
    else:
        restore(ROOT / spec["checkpoint"], model, probe, optimizer, device)
        last_step = 0
        history = []
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    eval_args = argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    for step in range(last_step + 1, args.recovery_steps + 1):
        model.train()
        probe.train()
        random_step(
            model,
            probe,
            optimizer,
            args,
            4,
            args.training_batch_size,
            args.probe_weight,
            device,
            dtype,
        )
        if step == 1 or step % args.eval_every == 0:
            metric = evaluate(model, probe, eval_args, 4, device, dtype)
            history.append({"recovery_step": step, **metric})
            if step in RECOVERY_MILESTONES:
                save_snapshot(folder, spec, step, model, metric)
            save_resume(
                resume_path,
                model,
                probe,
                optimizer,
                {
                    "seed": spec["seed"],
                    "branch": spec["branch"],
                    "recovery_step": step,
                    "recovery_history": history,
                    "suppression_active_steps": 0,
                },
            )
            print(
                f"seed={spec['seed']} branch={spec['branch']} recovery={step} "
                f"query={metric['query']:.2%} probe={metric['probe_min']:.2%}",
                flush=True,
            )
    milestone_rows = [
        {
            "recovery_step": 0,
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
        }
    ]
    for step in RECOVERY_MILESTONES[1:]:
        path = snapshot_path(folder, step)
        if not path.is_file():
            raise FileNotFoundError(path)
        milestone_rows.append(
            {
                "recovery_step": step,
                "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256_file(path),
            }
        )
    result = {
        "id": spec["id"],
        "seed": spec["seed"],
        "branch": spec["branch"],
        "baseline_route": spec["baseline_route"],
        "expected_route": spec["expected_route"],
        "recovery_steps_completed": args.recovery_steps,
        "suppression_active_steps": 0,
        "history": history,
        "milestones": milestone_rows,
        "training_complete": True,
        "training_beyond_registered_recovery": False,
    }
    atomic_save(result_path, result)
    del model, probe, optimizer
    torch.cuda.empty_cache()
    return result


def panel_args(
    args: argparse.Namespace, samples: int, batch_size: int, dataset_seed: int
) -> argparse.Namespace:
    return argparse.Namespace(
        chunks=args.causal_chunks,
        chunk_size=args.chunk_size,
        samples=samples,
        eval_batch_size=batch_size,
        dataset_seed=dataset_seed,
    )


def run_panel(
    spec: dict[str, Any],
    milestone: dict[str, Any],
    panel_name: str,
    conditions: list[str],
    samples: int,
    eval_batch_size: int,
    dataset_seed: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    step = int(milestone["recovery_step"])
    folder = (
        root
        / f"seed{spec['seed']}"
        / spec["branch"]
        / panel_name
        / f"step_{step:04d}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    checkpoint = ROOT / milestone["checkpoint"]
    if sha256_file(checkpoint) != milestone["checkpoint_sha256"]:
        raise RuntimeError(f"Recovery milestone hash changed: {spec['id']} step={step}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_fingerprint = state_dict_fingerprint(state["model"])
    model, probe = build(device, args.chunk_size)
    del probe
    model.load_state_dict(state["model"])
    del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = model_fingerprint(model)
    progress_path = folder / "condition_progress.json"
    metrics = read_json(progress_path) if progress_path.exists() and not args.force else {}
    if set(metrics) - set(conditions):
        raise RuntimeError("Unexpected resumed recovery condition")
    for name, metric in metrics.items():
        if metric.get("condition") != name or metric.get("samples") != samples:
            raise RuntimeError(f"Resumed recovery panel mismatch: {name}")
    evaluation_args = panel_args(args, samples, eval_batch_size, dataset_seed)
    for condition in conditions:
        if condition in metrics:
            continue
        metric = evaluate_condition(model, evaluation_args, condition, device, dtype)
        metrics[condition] = metric
        atomic_save(progress_path, metrics)
        print(
            f"seed={spec['seed']} branch={spec['branch']} recovery={step} "
            f"panel={panel_name} condition={condition} query={metric['query']:.2%}",
            flush=True,
        )
    after = model_fingerprint(model)
    integrity = {
        "checkpoint_sha256": milestone["checkpoint_sha256"],
        "expected_model_fingerprint": expected_fingerprint,
        "model_fingerprint_before": before,
        "model_fingerprint_after": after,
        "model_fingerprint_unchanged": before == after,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "all_conditions_in_registered_order": list(metrics) == conditions,
        "fixed_samples_every_condition": all(
            row["samples"] == samples for row in metrics.values()
        ),
        "dataset_seed": dataset_seed,
    }
    integrity["passed"] = bool(
        integrity["model_fingerprint_unchanged"]
        and integrity["all_parameters_frozen"]
        and integrity["all_conditions_in_registered_order"]
        and integrity["fixed_samples_every_condition"]
    )
    result = {
        "id": spec["id"],
        "seed": spec["seed"],
        "branch": spec["branch"],
        "baseline_route": spec["baseline_route"],
        "expected_route": spec["expected_route"],
        "recovery_step": step,
        "panel": panel_name,
        "metrics": metrics,
        "integrity": integrity,
    }
    if conditions == CONDITIONS:
        result["profile"] = profile_checkpoint(metrics, args)
    else:
        keep_l2 = metrics["keep_l2"]["query"]
        keep_l3 = metrics["keep_l3"]["query"]
        result["screen_profile"] = {
            "dominant_retention_layer": 2 if keep_l2 > keep_l3 else 3,
            "retention_gap_l2_minus_l3": keep_l2 - keep_l3,
            "intact_query": metrics["intact"]["query"],
            "minimum_local": min(row["local"] for row in metrics.values()),
        }
    atomic_save(result_path, result)
    del model
    torch.cuda.empty_cache()
    return result


def opposite_route(spec: dict[str, Any], route: str) -> bool:
    if spec["seed"] == 1879:
        return route == "l3_core"
    return route in {"l2_core_l3_supported", "l2_core_minimal_l3_support"}


def diagnose(
    training: list[dict[str, Any]],
    screens: list[dict[str, Any]],
    confirmations: list[dict[str, Any]],
    integrity_passed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    screens_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in screens:
        screens_by_id.setdefault(row["id"], []).append(row)
    confirms = {row["id"]: row for row in confirmations}
    per_branch = []
    for spec in SOURCE_SPECS:
        screen_rows = sorted(screens_by_id[spec["id"]], key=lambda row: row["recovery_step"])
        expected_layer = 2 if spec["seed"] == 1879 else 3
        first_screen_recovery = next(
            (
                row["recovery_step"]
                for row in screen_rows
                if row["screen_profile"]["intact_query"] >= 0.90
                and row["screen_profile"]["minimum_local"] >= 0.90
                and row["screen_profile"]["dominant_retention_layer"] == expected_layer
            ),
            None,
        )
        final = confirms[spec["id"]]
        final_route = final["profile"]["route_class"]
        per_branch.append(
            {
                "id": spec["id"],
                "seed": spec["seed"],
                "branch": spec["branch"],
                "baseline_route": spec["baseline_route"],
                "expected_route": spec["expected_route"],
                "baseline_unformed": spec["baseline_route"] == "unformed_behavior",
                "first_screen_recovery_step": first_screen_recovery,
                "final_route": final_route,
                "final_intact_query": final["metrics"]["intact"]["query"],
                "original_route_recovered_or_preserved": final_route == spec["expected_route"],
                "opposite_route_observed": opposite_route(spec, final_route),
                "screen_trajectory": [
                    {
                        "recovery_step": row["recovery_step"],
                        **row["screen_profile"],
                    }
                    for row in screen_rows
                ],
            }
        )
    initially_unformed = [row for row in per_branch if row["baseline_unformed"]]
    initially_formed = [row for row in per_branch if not row["baseline_unformed"]]
    recovered = sum(
        row["original_route_recovered_or_preserved"] for row in initially_unformed
    )
    stable = sum(
        row["original_route_recovered_or_preserved"] for row in initially_formed
    )
    migrations = [row["id"] for row in per_branch if row["opposite_route_observed"]]
    if not integrity_passed:
        classification = "formal_integrity_failed_recovery_interpretation_closed"
    elif migrations:
        classification = "opposite_layer_route_migration_observed"
    elif stable < len(initially_formed):
        classification = "continued_C4_destabilizes_preformed_routes"
    elif recovered == len(initially_unformed):
        classification = "complete_unsuppressed_route_recovery"
    elif recovered > 0:
        classification = "partial_unsuppressed_route_recovery"
    else:
        classification = "persistent_long_context_generalization_deficit"
    cohort = {
        "classification": classification,
        "initially_unformed_branches": len(initially_unformed),
        "initially_unformed_recovered_original_route": recovered,
        "initially_formed_branches": len(initially_formed),
        "initially_formed_preserved_original_route": stable,
        "opposite_layer_migrations": migrations,
        "registered_stop_boundary": (
            "Report the fixed +1000-step recovery trajectories; do not add recovery "
            "steps, change panels, or reclassify the frozen baseline groups."
        ),
    }
    return per_branch, cohort


def build_integrity(
    source_audit: list[dict[str, Any]],
    training: list[dict[str, Any]],
    screens: list[dict[str, Any]],
    confirmations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    integrity = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "parent_result_sha256": sha256_file(PARENT_RESULT),
        "all_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "expected_recovery_branches": len(SOURCE_SPECS),
        "completed_recovery_branches": len(training),
        "all_recovery_steps_exact": all(
            row["recovery_steps_completed"] == args.recovery_steps for row in training
        ),
        "no_recovery_suppression": all(
            row["suppression_active_steps"] == 0 for row in training
        ),
        "no_training_beyond_recovery_budget": all(
            not row["training_beyond_registered_recovery"] for row in training
        ),
        "all_registered_milestones_saved": all(
            [row["recovery_step"] for row in branch["milestones"]]
            == RECOVERY_MILESTONES
            for branch in training
        ),
        "expected_screen_panels": len(SOURCE_SPECS) * len(RECOVERY_MILESTONES),
        "completed_screen_panels": len(screens),
        "all_screen_integrity_passed": all(
            row["integrity"]["passed"] for row in screens
        ),
        "screen_samples_fixed": all(
            metric["samples"] == args.screen_samples
            for row in screens
            for metric in row["metrics"].values()
        ),
        "screen_dataset_seed": args.screen_dataset_seed,
        "expected_confirmation_panels": len(SOURCE_SPECS),
        "completed_confirmation_panels": len(confirmations),
        "all_confirmation_integrity_passed": all(
            row["integrity"]["passed"] for row in confirmations
        ),
        "confirmation_conditions_exact": all(
            list(row["metrics"]) == CONDITIONS for row in confirmations
        ),
        "confirmation_samples_fixed": all(
            metric["samples"] == args.confirm_samples
            for row in confirmations
            for metric in row["metrics"].values()
        ),
        "confirmation_dataset_seed": args.confirm_dataset_seed,
        "seed909_used": False,
    }
    integrity["passed"] = bool(
        integrity["all_source_hashes_validated"]
        and integrity["expected_recovery_branches"]
        == integrity["completed_recovery_branches"]
        and integrity["all_recovery_steps_exact"]
        and integrity["no_recovery_suppression"]
        and integrity["no_training_beyond_recovery_budget"]
        and integrity["all_registered_milestones_saved"]
        and integrity["expected_screen_panels"] == integrity["completed_screen_panels"]
        and integrity["all_screen_integrity_passed"]
        and integrity["screen_samples_fixed"]
        and integrity["expected_confirmation_panels"]
        == integrity["completed_confirmation_panels"]
        and integrity["all_confirmation_integrity_passed"]
        and integrity["confirmation_conditions_exact"]
        and integrity["confirmation_samples_fixed"]
        and not integrity["seed909_used"]
    )
    return integrity


def plot_trajectories(per_branch: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    colors = {
        "intact_replay": "#333333",
        "selected_layer_suppression": "#d1495b",
        "other_layer_suppression": "#0077b6",
    }
    labels = {
        "intact_replay": "intact source",
        "selected_layer_suppression": "selected-layer source",
        "other_layer_suppression": "other-layer source",
    }
    for axis, seed in zip(axes.flat, FORMAL_SEEDS):
        for row in [item for item in per_branch if item["seed"] == seed]:
            steps = [item["recovery_step"] for item in row["screen_trajectory"]]
            values = [100 * item["intact_query"] for item in row["screen_trajectory"]]
            axis.plot(
                steps,
                values,
                marker="o",
                linewidth=2,
                color=colors[row["branch"]],
                label=labels[row["branch"]],
            )
        axis.axhline(90, color="#666666", linestyle="--")
        axis.set_title(f"seed{seed}")
        axis.set_xlabel("Additional unsuppressed C4 steps")
        axis.set_ylabel("Fresh 16-chunk query accuracy (%)")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Level 7.5.3.1 unsuppressed recovery dynamics", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def state_after_one_step(
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, str]:
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    restore(checkpoint, model, probe, optimizer, device)
    before = model_fingerprint(model)
    random_step(
        model,
        probe,
        optimizer,
        args,
        4,
        args.training_batch_size,
        args.probe_weight,
        device,
        dtype,
    )
    result = {
        "before": before,
        "model": model_fingerprint(model),
        "probe": state_dict_fingerprint(probe.state_dict()),
        "optimizer": canonical_fingerprint(optimizer.state_dict()),
        "CPU_RNG": canonical_fingerprint(torch.get_rng_state().cpu()),
        "CUDA_RNG": canonical_fingerprint(
            [item.cpu() for item in torch.cuda.get_rng_state_all()]
        ),
    }
    del model, probe, optimizer
    torch.cuda.empty_cache()
    return result


def run_smoke(args: argparse.Namespace) -> int:
    protocol = read_json(STATIC_PREREGISTRATION)
    source_audit, parent_audit = validate_sources(protocol)
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    checkpoint = ROOT / SOURCE_SPECS[0]["checkpoint"]
    first = state_after_one_step(checkpoint, args, device, dtype)
    second = state_after_one_step(checkpoint, args, device, dtype)
    model, probe = build(device, args.chunk_size)
    del probe
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = model_fingerprint(model)
    smoke_args = argparse.Namespace(
        chunks=4,
        chunk_size=args.chunk_size,
        samples=32,
        eval_batch_size=8,
        dataset_seed=7531999,
    )
    metrics = {
        condition: evaluate_condition(model, smoke_args, condition, device, dtype)
        for condition in SCREEN_CONDITIONS
    }
    after = model_fingerprint(model)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "all_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "parent_audit": parent_audit,
        "recovery_step_deterministic": first == second,
        "recovery_step_changes_model": first["before"] != first["model"],
        "no_suppression_path_used": True,
        "screen_conditions_evaluated": list(metrics) == SCREEN_CONDITIONS,
        "evaluation_fingerprint_unchanged": before == after,
    }
    result["passed"] = all(
        result[key]
        for key in (
            "all_source_hashes_validated",
            "recovery_step_deterministic",
            "recovery_step_changes_model",
            "no_suppression_path_used",
            "screen_conditions_evaluated",
            "evaluation_fingerprint_unchanged",
        )
    )
    atomic_save(ROOT / "experiments/level7_5_3_1/smoke/result.json", result)
    print(json.dumps(result, indent=2))
    del model
    torch.cuda.empty_cache()
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
    source_audit, parent_audit = validate_sources(protocol)
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
    progress = {
        "stage": "unsuppressed_recovery_training",
        "completed_recovery_branches": [],
        "completed_screen_panels": [],
        "completed_confirmation_panels": [],
        "active": None,
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    training = []
    screens = []
    confirmations = []
    for spec in SOURCE_SPECS:
        progress["active"] = spec["id"]
        atomic_save(root / "progress.json", progress)
        recovered = run_recovery(spec, args, device, dtype, root)
        training.append(recovered)
        progress["completed_recovery_branches"].append(spec["id"])
        atomic_save(root / "progress.json", progress)
        progress["stage"] = "trajectory_screen_panels"
        for milestone in recovered["milestones"]:
            row = run_panel(
                spec,
                milestone,
                "screen",
                SCREEN_CONDITIONS,
                args.screen_samples,
                args.screen_eval_batch_size,
                args.screen_dataset_seed,
                args,
                device,
                dtype,
                root,
            )
            screens.append(row)
            progress["completed_screen_panels"].append(
                f"{spec['id']}/step{milestone['recovery_step']}"
            )
            atomic_save(root / "progress.json", progress)
        progress["stage"] = "final_confirmation_panels"
        final_milestone = recovered["milestones"][-1]
        confirmation = run_panel(
            spec,
            final_milestone,
            "confirmation",
            CONDITIONS,
            args.confirm_samples,
            args.confirm_eval_batch_size,
            args.confirm_dataset_seed,
            args,
            device,
            dtype,
            root,
        )
        confirmations.append(confirmation)
        progress["completed_confirmation_panels"].append(spec["id"])
        atomic_save(root / "progress.json", progress)
    integrity = build_integrity(source_audit, training, screens, confirmations, args)
    per_branch, diagnosis = diagnose(training, screens, confirmations, integrity["passed"])
    elapsed = time.perf_counter() - started
    result = {
        "protocol": protocol,
        "parent_audit": parent_audit,
        "source_audit": source_audit,
        "recovery_training": training,
        "screen_panels": screens,
        "confirmation_panels": confirmations,
        "branch_diagnoses": per_branch,
        "diagnosis": diagnosis,
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed,
    }
    summary = {
        "diagnosis": diagnosis,
        "integrity": integrity,
        "branches": per_branch,
        "final_endpoints": [
            {
                "id": row["id"],
                "seed": row["seed"],
                "branch": row["branch"],
                "route_class": row["profile"]["route_class"],
                "intact_query": row["metrics"]["intact"]["query"],
                "layer_atlas": row["profile"]["layer_atlas"],
            }
            for row in confirmations
        ],
        "elapsed_seconds_this_invocation": elapsed,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_trajectories(per_branch, root / "unsuppressed_recovery_dynamics.png")
    atomic_save(
        root / "progress.json",
        {
            "stage": "complete",
            "completed_recovery_branches": len(training),
            "completed_screen_panels": len(screens),
            "completed_confirmation_panels": len(confirmations),
            "classification": diagnosis["classification"],
            "integrity_passed": integrity["passed"],
            "seed909_locked": True,
        },
    )
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

