"""Level 7.5.2: independent frozen-panel confirmation of the weak L2 precursor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_level6_6_local import build
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save, model_fingerprint
from run_level7_3_local import CONDITIONS, evaluate_condition, read_json, sha256_file
from run_level7_4_1_local import state_dict_fingerprint


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_5_2"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
PARENT_RESULT = ROOT / "experiments/level7_5_1/formal/result.json"
PARENT_RESULT_SHA256 = "c9f03418d5337e67f9b7ed853db55b741e2e5f760e1760e8facf91d20360c6e8"
PARENT_DATASET_SEED = 7510000
FORMAL_DATASET_SEED = 7520000
TARGET_SEED = 1879
CONTROL_SEEDS = [2203, 2551, 2909]
WHOLE_CONTROLS = ("reset_all", "zero_all", "batch_roll_all")


def source(
    seed: int,
    step: int,
    sha256: str,
    role: str,
    l3_calibration: bool = False,
) -> dict[str, Any]:
    return {
        "id": f"seed{seed}_step{step:04d}",
        "seed": seed,
        "step": step,
        "checkpoint": (
            f"experiments/level7_5_1/formal/seed{seed}/replay/"
            f"model_step{step:04d}.pt"
        ),
        "checkpoint_sha256": sha256,
        "checkpoint_size_bytes": 1420803,
        "registered_role": role,
        "registered_L3_calibration_positive": l3_calibration,
    }


SOURCE_SPECS = [
    source(1879, 1200, "4d20eb5f607523423fc2b293aabec45e55bfb88eb42f3a9bcec5b4071f628cdb", "target_window_context"),
    source(1879, 1300, "b35212723c8f87e318b527466187d1207ec14adfff7865fddab094f7b2282036", "target_window_context"),
    source(1879, 1400, "de5487b9f3fcb6b6a33020f19375186f16de97b47ac660513c6e0f0ad69f2f0c", "registered_L2_positive"),
    source(1879, 1500, "43dad953389ba233981c0c1e818e14ab4e955aaf95ec05ea9fc586a807a10cc9", "target_window_context"),
    source(1879, 1600, "5feaebd5d0248ecc6131aecf7e72e356bbec80e90dffc8f4a6ca1f672eae2a3a", "registered_L2_positive"),
    source(1879, 1700, "6141d669f244ace997c261e1206c33a9ab37fc47c9e68dcec70af91786bdbf6d", "target_window_context"),
    source(1879, 1800, "a96f5596531c12ab0cc72c70de159b848fbe78141397899c5c4908bb38370aa2", "target_window_context"),
    source(2203, 800, "18afbced72c103a20fd6dbc0467b700b0afb4df6e3c6a289d22581981d2f6020", "default_L3_negative_control"),
    source(2203, 900, "8787240aec2e8418bb75d83053e22d22280aec3f0a316954b0e153239d516ee2", "default_L3_negative_control"),
    source(2203, 1000, "79ba1d6782a78333adf922ed418b2b5066369e0e2264dc5445dfb05bc379b849", "default_L3_negative_control", True),
    source(2551, 600, "e347e58084993a878e3d8723b2be163c47212f10cf6699f0c2019f4eba0718c1", "default_L3_negative_control"),
    source(2551, 700, "e179f6beaea8fd6e62b4033ed57f1efbc54ae1909dd6de4e7b1ac34eba609d9e", "default_L3_negative_control", True),
    source(2551, 800, "c80aeff87f45db4159deae63debf81fba2bcbcb1c5546a953fdd5896401ccdec", "default_L3_negative_control", True),
    source(2909, 600, "26f1ccd12f2dfc9e60ea8d14359710f732b1cc6ef404cda1dc546d062f75e788", "default_L3_negative_control"),
    source(2909, 700, "9fb7a66b4314b1b67fc6a8b0366b13461ea397f9dba7ea81481239545fd07ad2", "default_L3_negative_control", True),
    source(2909, 800, "93f65660c4f8205660dc17b0ad4d230fd5ed4f55e7a1aceb157fdd24d7d0d1d1", "default_L3_negative_control", True),
]

REGISTERED_L2_POSITIVES = [
    "seed1879_step1400",
    "seed1879_step1600",
]
REGISTERED_L2_NEGATIVE_CONTROLS = [
    row["id"] for row in SOURCE_SPECS if row["seed"] in CONTROL_SEEDS
]
REGISTERED_L3_CALIBRATION_POSITIVES = [
    row["id"] for row in SOURCE_SPECS if row["registered_L3_calibration_positive"]
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=FORMAL_DATASET_SEED)
    parser.add_argument("--formed-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--strict-preservation-threshold", type=float, default=0.80)
    parser.add_argument("--strict-retention-threshold", type=float, default=0.90)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--weak-intact-threshold", type=float, default=0.20)
    parser.add_argument("--weak-retention-threshold", type=float, default=0.20)
    parser.add_argument("--weak-preservation-margin", type=float, default=0.05)
    parser.add_argument("--weak-selectivity-gap", type=float, default=0.10)
    parser.add_argument("--output", default="experiments/level7_5_2/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "chunks": 16,
        "chunk_size": 128,
        "samples": 4096,
        "eval_batch_size": 16,
        "dataset_seed": FORMAL_DATASET_SEED,
        "formed_threshold": 0.90,
        "disruption_threshold": 0.20,
        "strict_preservation_threshold": 0.80,
        "strict_retention_threshold": 0.90,
        "local_threshold": 0.90,
        "weak_intact_threshold": 0.20,
        "weak_retention_threshold": 0.20,
        "weak_preservation_margin": 0.05,
        "weak_selectivity_gap": 0.10,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.5.2 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_5_2/formal":
        raise ValueError("Formal Level 7.5.2 output path is locked")


def validate_static_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("frozen_milestones") != SOURCE_SPECS:
        raise RuntimeError("Static Level 7.5.2 milestone registration changed")
    panel = protocol.get("independent_causal_panel", {})
    if panel.get("conditions") != CONDITIONS:
        raise RuntimeError("Static Level 7.5.2 condition order changed")
    if panel.get("dataset_seed") != FORMAL_DATASET_SEED:
        raise RuntimeError("Static Level 7.5.2 dataset seed changed")
    outcomes = protocol.get("registered_primary_test", {})
    if outcomes.get("positive_milestones") != REGISTERED_L2_POSITIVES:
        raise RuntimeError("Static Level 7.5.2 L2 positives changed")
    if outcomes.get("negative_control_milestones") != REGISTERED_L2_NEGATIVE_CONTROLS:
        raise RuntimeError("Static Level 7.5.2 L2 negative controls changed")


def validate_sources(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_static_protocol(protocol)
    if not PARENT_RESULT.is_file():
        raise FileNotFoundError(PARENT_RESULT)
    observed_parent_hash = sha256_file(PARENT_RESULT)
    if observed_parent_hash != PARENT_RESULT_SHA256:
        raise RuntimeError("Frozen Level 7.5.1 result hash changed")
    parent = read_json(PARENT_RESULT)
    if not parent["integrity"]["passed"]:
        raise RuntimeError("Level 7.5.1 parent integrity did not pass")
    if parent["diagnosis"]["classification"] != "default_L3_precursor_divergence_confirmed":
        raise RuntimeError("Unexpected Level 7.5.1 parent classification")
    parent_runs = {int(row["seed"]): row for row in parent["runs"]}
    if not all(parent_runs[seed]["replay_gate"]["passed"] for seed in [TARGET_SEED, *CONTROL_SEEDS]):
        raise RuntimeError("One or more Level 7.5.1 exact replay gates failed")
    audit = []
    for spec in SOURCE_SPECS:
        path = ROOT / spec["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        observed_size = path.stat().st_size
        if observed_hash != spec["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint hash changed: {spec['id']}")
        if observed_size != spec["checkpoint_size_bytes"]:
            raise RuntimeError(f"Frozen checkpoint size changed: {spec['id']}")
        parent_milestone = next(
            (
                row
                for row in parent_runs[spec["seed"]]["milestones"]
                if int(row["step"]) == spec["step"]
            ),
            None,
        )
        if parent_milestone is None:
            raise RuntimeError(f"Parent milestone missing: {spec['id']}")
        if parent_milestone["source"]["checkpoint_sha256"] != observed_hash:
            raise RuntimeError(f"Parent milestone source mismatch: {spec['id']}")
        audit.append(
            {
                **spec,
                "observed_checkpoint_sha256": observed_hash,
                "observed_checkpoint_size_bytes": observed_size,
                "parent_replay_gate_passed": True,
                "parent_milestone_integrity_passed": bool(
                    parent_milestone["integrity"]["passed"]
                ),
                "source_validation_passed": bool(
                    parent_milestone["integrity"]["passed"]
                    and observed_hash == spec["checkpoint_sha256"]
                    and observed_size == spec["checkpoint_size_bytes"]
                ),
            }
        )
    parent_audit = {
        "result": str(PARENT_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "expected_sha256": PARENT_RESULT_SHA256,
        "observed_sha256": observed_parent_hash,
        "integrity_passed": parent["integrity"]["passed"],
        "classification": parent["diagnosis"]["classification"],
        "all_four_exact_replay_gates_passed": all(
            parent_runs[seed]["replay_gate"]["passed"]
            for seed in [TARGET_SEED, *CONTROL_SEEDS]
        ),
    }
    return audit, parent_audit


def causal_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        chunks=args.chunks,
        chunk_size=args.chunk_size,
        samples=args.samples,
        eval_batch_size=args.eval_batch_size,
        dataset_seed=args.dataset_seed,
    )


def validate_resumed_metrics(
    metrics: dict[str, dict[str, Any]], args: argparse.Namespace
) -> None:
    unexpected = set(metrics) - set(CONDITIONS)
    if unexpected:
        raise RuntimeError(f"Unexpected resumed conditions: {sorted(unexpected)}")
    for name, metric in metrics.items():
        if metric.get("condition") != name:
            raise RuntimeError(f"Resumed condition label mismatch: {name}")
        if metric.get("samples") != args.samples or metric.get("chunks") != args.chunks:
            raise RuntimeError(f"Resumed causal protocol mismatch: {name}")


def selective_profile(
    metrics: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    query = {name: metrics[name]["query"] for name in CONDITIONS}
    minimum_local = min(row["local"] for row in metrics.values())
    whole_max = max(query[name] for name in WHOLE_CONTROLS)
    behavior_formed = bool(
        query["intact"] >= args.formed_threshold
        and minimum_local >= args.local_threshold
    )
    whole_memory_causal = bool(
        behavior_formed and whole_max <= args.disruption_threshold
    )

    def layer_rule(layer: int) -> dict[str, Any]:
        other_layers = [item for item in (1, 2, 3) if item != layer]
        strict = bool(
            whole_memory_causal
            and query[f"zero_l{layer}"] <= args.disruption_threshold
            and query[f"batch_roll_l{layer}"] <= args.disruption_threshold
            and query[f"keep_l{layer}"] >= args.strict_retention_threshold
            and all(
                query[f"zero_l{other}"] >= args.strict_preservation_threshold
                for other in other_layers
            )
            and all(
                query[f"keep_l{other}"] <= args.disruption_threshold
                for other in other_layers
            )
        )
        preservation_floor = max(
            args.weak_intact_threshold,
            query["intact"] - args.weak_preservation_margin,
        )
        best_other_keep = max(query[f"keep_l{other}"] for other in other_layers)
        selectivity_gap = query[f"keep_l{layer}"] - best_other_keep
        checks = {
            "not_strict_core": not strict,
            "intact_at_least_weak_floor": query["intact"] >= args.weak_intact_threshold,
            "minimum_local_at_least_floor": minimum_local >= args.local_threshold,
            "whole_controls_at_most_disruption_floor": whole_max <= args.disruption_threshold,
            "zero_target_at_most_disruption_floor": query[f"zero_l{layer}"] <= args.disruption_threshold,
            "roll_target_at_most_disruption_floor": query[f"batch_roll_l{layer}"] <= args.disruption_threshold,
            "keep_target_at_least_weak_floor": query[f"keep_l{layer}"] >= args.weak_retention_threshold,
            "other_single_keeps_at_most_disruption_floor": all(
                query[f"keep_l{other}"] <= args.disruption_threshold
                for other in other_layers
            ),
            "other_layer_zeros_preserve_intact": all(
                query[f"zero_l{other}"] >= preservation_floor
                for other in other_layers
            ),
            "target_keep_selectivity_gap": selectivity_gap >= args.weak_selectivity_gap,
        }
        return {
            "strict_core": strict,
            "weak_selective_precursor": all(checks.values()),
            "preservation_floor": preservation_floor,
            "best_other_keep": best_other_keep,
            "keep_target_minus_best_other_single": selectivity_gap,
            "checks": checks,
        }

    l2 = layer_rule(2)
    l3 = layer_rule(3)
    return {
        "minimum_local": minimum_local,
        "whole_memory_max_disrupted_query": whole_max,
        "behavior_formed": behavior_formed,
        "whole_memory_causal": whole_memory_causal,
        "strict_l2_core": l2["strict_core"],
        "weak_l2_selective_precursor": l2["weak_selective_precursor"],
        "l2_selectivity": l2,
        "strict_l3_core": l3["strict_core"],
        "weak_l3_selective_precursor": l3["weak_selective_precursor"],
        "l3_selectivity": l3,
    }


def run_milestone(
    source_spec: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    folder = root / f"seed{source_spec['seed']}" / f"step_{source_spec['step']:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        result = read_json(result_path)
        if result["source"]["checkpoint_sha256"] != source_spec["checkpoint_sha256"]:
            raise RuntimeError(f"Cached source mismatch: {source_spec['id']}")
        if result["integrity"].get("shared_dataset_seed") != args.dataset_seed:
            raise RuntimeError(f"Cached dataset mismatch: {source_spec['id']}")
        if not result["integrity"]["passed"]:
            raise RuntimeError(f"Cached milestone integrity failed: {source_spec['id']}")
        return result
    checkpoint = ROOT / source_spec["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_fingerprint = state_dict_fingerprint(state["model"])
    model, probe = build(device, args.chunk_size)
    del probe
    model.load_state_dict(state["model"])
    del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fingerprint_before = model_fingerprint(model)
    if fingerprint_before != expected_fingerprint:
        raise RuntimeError(f"Model fingerprint mismatch: {source_spec['id']}")
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
            f"seed={source_spec['seed']} step={source_spec['step']} "
            f"condition={condition} query={metric['query']:.2%} "
            f"local={metric['local']:.2%}",
            flush=True,
        )
    fingerprint_after = model_fingerprint(model)
    profile = selective_profile(metrics, args)
    integrity = {
        "checkpoint_sha256": source_spec["checkpoint_sha256"],
        "expected_model_fingerprint": expected_fingerprint,
        "model_fingerprint_before": fingerprint_before,
        "model_fingerprint_after": fingerprint_after,
        "model_fingerprint_unchanged": fingerprint_before == fingerprint_after,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "all_conditions_present_in_registered_order": list(metrics) == CONDITIONS,
        "fixed_samples_every_condition": all(
            row["samples"] == args.samples for row in metrics.values()
        ),
        "shared_dataset_seed": args.dataset_seed,
        "training_performed": False,
    }
    integrity["passed"] = bool(
        integrity["model_fingerprint_unchanged"]
        and integrity["all_parameters_frozen"]
        and integrity["all_conditions_present_in_registered_order"]
        and integrity["fixed_samples_every_condition"]
        and not integrity["training_performed"]
    )
    result = {
        "id": source_spec["id"],
        "seed": source_spec["seed"],
        "step": source_spec["step"],
        "source": source_spec,
        "metrics": metrics,
        "profile": profile,
        "integrity": integrity,
    }
    atomic_save(result_path, result)
    del model
    torch.cuda.empty_cache()
    return result


def diagnose(results: list[dict[str, Any]], integrity_passed: bool) -> dict[str, Any]:
    by_id = {row["id"]: row for row in results}
    positive_hits = [
        item
        for item in REGISTERED_L2_POSITIVES
        if by_id[item]["profile"]["weak_l2_selective_precursor"]
    ]
    control_false_positives = [
        item
        for item in REGISTERED_L2_NEGATIVE_CONTROLS
        if by_id[item]["profile"]["weak_l2_selective_precursor"]
    ]
    l3_calibration_hits = [
        item
        for item in REGISTERED_L3_CALIBRATION_POSITIVES
        if by_id[item]["profile"]["weak_l3_selective_precursor"]
        or by_id[item]["profile"]["strict_l3_core"]
    ]
    if not integrity_passed:
        classification = "formal_integrity_failed_causal_interpretation_closed"
    elif len(positive_hits) == len(REGISTERED_L2_POSITIVES) and not control_false_positives:
        classification = "independent_weak_L2_precursor_confirmed"
    elif len(positive_hits) == len(REGISTERED_L2_POSITIVES):
        classification = "weak_L2_signal_not_route_specific"
    elif positive_hits:
        classification = "weak_L2_precursor_partially_replicated"
    else:
        classification = "weak_L2_precursor_not_replicated"
    target_window = [row for row in results if row["seed"] == TARGET_SEED]
    return {
        "classification": classification,
        "registered_L2_positive_hits": positive_hits,
        "registered_L2_positive_count": len(positive_hits),
        "registered_L2_positive_expected": len(REGISTERED_L2_POSITIVES),
        "default_L3_control_L2_false_positives": control_false_positives,
        "default_L3_control_L2_false_positive_count": len(control_false_positives),
        "default_L3_control_count": len(REGISTERED_L2_NEGATIVE_CONTROLS),
        "secondary_L3_calibration_hits": l3_calibration_hits,
        "secondary_L3_calibration_count": len(l3_calibration_hits),
        "secondary_L3_calibration_expected": len(REGISTERED_L3_CALIBRATION_POSITIVES),
        "seed1879_window_L2_positive_steps": [
            row["step"]
            for row in target_window
            if row["profile"]["weak_l2_selective_precursor"]
            or row["profile"]["strict_l2_core"]
        ],
        "registered_stop_boundary": (
            "Report the fixed N=4096 independent-panel result; do not alter the "
            "mirrored rule, add checkpoints, extend samples, or resume training."
        ),
    }


def build_integrity(
    source_audit: list[dict[str, Any]],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    integrity = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "parent_result_sha256": sha256_file(PARENT_RESULT),
        "all_parent_exact_replay_gates_passed": all(
            row["parent_replay_gate_passed"] for row in source_audit
        ),
        "all_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "expected_frozen_milestones": len(SOURCE_SPECS),
        "completed_frozen_milestones": len(results),
        "all_milestone_integrity_passed": all(
            row["integrity"]["passed"] for row in results
        ),
        "all_conditions_exact": all(
            list(row["metrics"]) == CONDITIONS for row in results
        ),
        "fixed_N_completed": all(
            metric["samples"] == args.samples
            for row in results
            for metric in row["metrics"].values()
        ),
        "shared_fresh_dataset_seed": args.dataset_seed,
        "parent_dataset_seed": PARENT_DATASET_SEED,
        "independent_dataset_used": args.dataset_seed != PARENT_DATASET_SEED,
        "training_performed": False,
        "checkpoint_selection_changed_after_run": False,
        "seed909_used": False,
    }
    integrity["passed"] = bool(
        integrity["all_parent_exact_replay_gates_passed"]
        and integrity["all_source_hashes_validated"]
        and integrity["expected_frozen_milestones"]
        == integrity["completed_frozen_milestones"]
        and integrity["all_milestone_integrity_passed"]
        and integrity["all_conditions_exact"]
        and integrity["fixed_N_completed"]
        and integrity["independent_dataset_used"]
        and not integrity["training_performed"]
        and not integrity["checkpoint_selection_changed_after_run"]
        and not integrity["seed909_used"]
    )
    return integrity


def plot_results(results: list[dict[str, Any]], path: Path) -> None:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in results:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for axis, seed in zip(axes.flat, [1879, 2203, 2551, 2909]):
        rows = sorted(by_seed[seed], key=lambda item: item["step"])
        steps = [row["step"] for row in rows]
        for condition, label, color in [
            ("intact", "intact", "#222222"),
            ("keep_l2", "keep L2", "#0077b6"),
            ("keep_l3", "keep L3", "#d1495b"),
            ("zero_l2", "zero L2", "#56b4e9"),
            ("zero_l3", "zero L3", "#e69f00"),
        ]:
            axis.plot(
                steps,
                [100 * row["metrics"][condition]["query"] for row in rows],
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.axhline(20, color="#666666", linestyle=":", linewidth=1.5)
        if seed == TARGET_SEED:
            for step in (1400, 1600):
                axis.axvline(step, color="#0077b6", alpha=0.18, linewidth=8)
            title = "seed1879: independently tested L2 window"
        else:
            title = f"seed{seed}: default-L3 negative controls"
        axis.set_title(title)
        axis.set_xlabel("C2 replay step")
        axis.set_ylabel("Query accuracy (%)")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle("Level 7.5.2 independent weak-L2 precursor confirmation", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_smoke(args: argparse.Namespace) -> int:
    protocol = read_json(STATIC_PREREGISTRATION)
    source_audit, parent_audit = validate_sources(protocol)
    args.chunks = 4
    args.samples = 32
    args.eval_batch_size = 8
    args.dataset_seed = 7529999
    args.force = True
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / "experiments/level7_5_2/smoke"
    row = run_milestone(SOURCE_SPECS[2], args, device, dtype, root)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "all_16_source_hashes_validated": all(
            item["source_validation_passed"] for item in source_audit
        ),
        "parent_audit": parent_audit,
        "conditions_evaluated": len(row["metrics"]),
        "frozen_fingerprint_unchanged": row["integrity"]["model_fingerprint_unchanged"],
        "profile_fields_present": all(
            key in row["profile"]
            for key in (
                "weak_l2_selective_precursor",
                "weak_l3_selective_precursor",
                "l2_selectivity",
                "l3_selectivity",
            )
        ),
    }
    result["passed"] = bool(
        result["all_16_source_hashes_validated"]
        and result["conditions_evaluated"] == len(CONDITIONS)
        and result["frozen_fingerprint_unchanged"]
        and result["profile_fields_present"]
    )
    atomic_save(root / "smoke_result.json", result)
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
        "stage": "independent_frozen_panel",
        "completed_milestones": [],
        "active_milestone": None,
        "expected_milestones": len(SOURCE_SPECS),
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    results = []
    for source_spec in SOURCE_SPECS:
        progress["active_milestone"] = source_spec["id"]
        atomic_save(root / "progress.json", progress)
        row = run_milestone(source_spec, args, device, dtype, root)
        results.append(row)
        progress["completed_milestones"].append(source_spec["id"])
        atomic_save(root / "progress.json", progress)
    integrity = build_integrity(source_audit, results, args)
    diagnosis = diagnose(results, integrity["passed"])
    elapsed_seconds = time.perf_counter() - started
    result = {
        "protocol": protocol,
        "parent_audit": parent_audit,
        "source_audit": source_audit,
        "milestones": results,
        "diagnosis": diagnosis,
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    summary = {
        "diagnosis": diagnosis,
        "integrity": integrity,
        "milestones": [
            {
                "id": row["id"],
                "seed": row["seed"],
                "step": row["step"],
                "registered_role": row["source"]["registered_role"],
                "intact_query": row["metrics"]["intact"]["query"],
                "keep_l2_query": row["metrics"]["keep_l2"]["query"],
                "keep_l3_query": row["metrics"]["keep_l3"]["query"],
                "weak_l2_selective_precursor": row["profile"]["weak_l2_selective_precursor"],
                "weak_l3_selective_precursor": row["profile"]["weak_l3_selective_precursor"],
                "l2_selectivity": row["profile"]["l2_selectivity"],
                "l3_selectivity": row["profile"]["l3_selectivity"],
            }
            for row in results
        ],
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_results(results, root / "independent_weak_L2_confirmation.png")
    atomic_save(
        root / "progress.json",
        {
            "stage": "complete",
            "completed_milestones": [row["id"] for row in SOURCE_SPECS],
            "expected_milestones": len(SOURCE_SPECS),
            "classification": diagnosis["classification"],
            "integrity_passed": integrity["passed"],
            "seed909_locked": True,
        },
    )
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

