"""Level 7.3.1 high-precision replication of seed1879's L2 Memory route."""

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
import numpy as np
import torch

from run_level6_6_local import build
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save, model_fingerprint
from run_level7_3_local import evaluate_condition, read_json, sha256_file


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_3_1"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
SOURCE_CHECKPOINT = Path(
    "experiments/level7_2/formal/seed1879/zero_probe_step0750.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "cbba0c6db219e16be274bd0e77612972b7ad7d24ffa2bddb7b7cb858bf7a74f6"
)
EXPECTED_MODEL_FINGERPRINT = (
    "ad113264a0227bb69770dc56b7f395b0bf999e7b4eb8c487408d07100043f77c"
)
OLD_PANEL_SEEDS = (7218790, 7218791, 7218792, 7300000)
CONDITIONS = [
    "intact",
    "reset_all",
    "zero_all",
    "batch_roll_all",
    "zero_l2",
    "batch_roll_l2",
    "keep_l2",
    "zero_l3",
    "batch_roll_l3",
    "keep_l3",
    "keep_l2_l3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=7310000)
    parser.add_argument("--intact-lower-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-upper-threshold", type=float, default=0.20)
    parser.add_argument("--sufficiency-lower-threshold", type=float, default=0.90)
    parser.add_argument("--l3-retention-lower-threshold", type=float, default=0.80)
    parser.add_argument("--local-lower-threshold", type=float, default=0.90)
    parser.add_argument("--output", default="experiments/level7_3_1/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "chunks": 16,
        "chunk_size": 128,
        "samples": 8192,
        "eval_batch_size": 16,
        "dataset_seed": 7310000,
        "intact_lower_threshold": 0.90,
        "disruption_upper_threshold": 0.20,
        "sufficiency_lower_threshold": 0.90,
        "l3_retention_lower_threshold": 0.80,
        "local_lower_threshold": 0.90,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.3.1 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_3_1/formal":
        raise ValueError("Formal output path is locked")


def validate_source() -> dict[str, Any]:
    parent_result_path = ROOT / "experiments/level7_3/formal/result.json"
    if not parent_result_path.is_file():
        raise FileNotFoundError(parent_result_path)
    parent = read_json(parent_result_path)
    if not parent["integrity"]["passed"]:
        raise RuntimeError("Level 7.3 source integrity did not pass")
    if (
        parent["diagnosis"]["classification"]
        != "cross_initialization_layer_heterogeneity_confirmed"
    ):
        raise RuntimeError("Level 7.3 did not register the required parent result")
    prior_runs = [row for row in parent["runs"] if int(row["seed"]) == 1879]
    if len(prior_runs) != 1:
        raise RuntimeError("Expected exactly one seed1879 Level 7.3 source run")
    prior = prior_runs[0]
    if prior["profile"]["profile_class"] != "layer2_dominant":
        raise RuntimeError("Level 7.3 did not classify seed1879 as L2 dominant")
    if prior["source"]["checkpoint"] != SOURCE_CHECKPOINT.as_posix():
        raise RuntimeError("Level 7.3 seed1879 checkpoint provenance changed")
    checkpoint = ROOT / SOURCE_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen seed1879 checkpoint hash mismatch")
    if prior["integrity"]["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("Level 7.3 checkpoint audit does not match current source")
    if (
        prior["integrity"]["model_fingerprint_after"]
        != EXPECTED_MODEL_FINGERPRINT
    ):
        raise RuntimeError("Level 7.3 model fingerprint does not match preregistration")
    parent_dataset_seed = int(parent["integrity"]["shared_fresh_dataset_seed"])
    if parent_dataset_seed == 7310000:
        raise RuntimeError("Level 7.3.1 must not reuse the Level 7.3 panel")
    parent_script = ROOT / "run_level7_3_local.py"
    parent_preregistration = ROOT / "experiments/level7_3/preregistration.json"
    if sha256_file(parent_script) != parent["integrity"]["script_sha256"]:
        raise RuntimeError("Level 7.3 runner changed after the source result")
    if (
        sha256_file(parent_preregistration)
        != parent["integrity"]["preregistration_sha256"]
    ):
        raise RuntimeError("Level 7.3 preregistration changed after the source result")
    return {
        "seed": 1879,
        "checkpoint": SOURCE_CHECKPOINT.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "expected_model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
        "parent_result": "experiments/level7_3/formal/result.json",
        "parent_result_sha256": sha256_file(parent_result_path),
        "parent_classification": parent["diagnosis"]["classification"],
        "parent_profile_class": prior["profile"]["profile_class"],
        "parent_dataset_seed": parent_dataset_seed,
        "parent_keep_l2": prior["metrics"]["keep_l2"],
        "source_validation_passed": True,
    }


def interval(metrics: dict[str, dict[str, Any]], condition: str, field: str) -> list[float]:
    return metrics[condition][f"{field}_wilson95"]


def diagnose(
    metrics: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    query_lower = {
        name: interval(metrics, name, "query")[0] for name in CONDITIONS
    }
    query_upper = {
        name: interval(metrics, name, "query")[1] for name in CONDITIONS
    }
    minimum_local_lower = min(
        interval(metrics, name, "local")[0] for name in CONDITIONS
    )
    gates = {
        "intact_behavior": (
            query_lower["intact"] >= args.intact_lower_threshold
        ),
        "local_behavior": minimum_local_lower >= args.local_lower_threshold,
        "whole_memory_causality": all(
            query_upper[name] <= args.disruption_upper_threshold
            for name in ("reset_all", "zero_all", "batch_roll_all")
        ),
        "L2_necessity": (
            query_upper["zero_l2"] <= args.disruption_upper_threshold
        ),
        "L2_sample_alignment": (
            query_upper["batch_roll_l2"] <= args.disruption_upper_threshold
        ),
        "L2_single_layer_sufficiency": (
            query_lower["keep_l2"] >= args.sufficiency_lower_threshold
        ),
        "L3_nonnecessity_contrast": all(
            query_lower[name] >= args.l3_retention_lower_threshold
            for name in ("zero_l3", "batch_roll_l3")
        ),
        "L3_insufficiency_contrast": (
            query_upper["keep_l3"] <= args.disruption_upper_threshold
        ),
        "L2_L3_positive_pair_control": (
            query_lower["keep_l2_l3"] >= args.sufficiency_lower_threshold
        ),
    }
    all_passed = all(gates.values())
    non_sufficiency_passed = all(
        passed
        for name, passed in gates.items()
        if name != "L2_single_layer_sufficiency"
    )
    if all_passed:
        classification = "high_precision_l2_route_confirmed"
    elif (
        non_sufficiency_passed
        and not gates["L2_single_layer_sufficiency"]
        and metrics["keep_l2"]["query"] >= args.sufficiency_lower_threshold
    ):
        classification = (
            "l2_route_supported_but_single_layer_sufficiency_inconclusive"
        )
    else:
        classification = "high_precision_l2_route_not_confirmed"
    return {
        "classification": classification,
        "all_registered_gates_passed": all_passed,
        "gates": gates,
        "minimum_local_wilson95_lower": minimum_local_lower,
        "keep_l2_query": metrics["keep_l2"]["query"],
        "keep_l2_query_wilson95": metrics["keep_l2"]["query_wilson95"],
        "fixed_samples": args.samples,
        "registered_stop_boundary": (
            "Report the fixed classification; do not add samples, conditions, "
            "checkpoints, or a repair after observing results."
        ),
    }


def validate_resumed_rows(
    rows: dict[str, dict[str, Any]], args: argparse.Namespace
) -> None:
    unexpected = set(rows) - set(CONDITIONS)
    if unexpected:
        raise RuntimeError(f"Unexpected resumed conditions: {sorted(unexpected)}")
    for name, metric in rows.items():
        if metric.get("condition") != name:
            raise RuntimeError(f"Resumed condition label mismatch: {name}")
        if metric.get("samples") != args.samples or metric.get("chunks") != args.chunks:
            raise RuntimeError(f"Resumed metric protocol mismatch: {name}")


def plot_precision(
    metrics: dict[str, dict[str, Any]], path: Path
) -> None:
    values = np.array([100 * metrics[name]["query"] for name in CONDITIONS])
    lowers = np.array([
        100 * metrics[name]["query_wilson95"][0] for name in CONDITIONS
    ])
    uppers = np.array([
        100 * metrics[name]["query_wilson95"][1] for name in CONDITIONS
    ])
    errors = np.vstack([values - lowers, uppers - values])
    labels = [
        name.replace("batch_roll", "roll").replace("_", " ")
        for name in CONDITIONS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    colors = ["#4c78a8"] * 4 + ["#e45756"] * 3 + ["#72b7b2"] * 3 + ["#54a24b"]
    axes[0].bar(range(len(CONDITIONS)), values, color=colors)
    axes[0].errorbar(
        range(len(CONDITIONS)), values, yerr=errors,
        fmt="none", ecolor="black", capsize=3, linewidth=1,
    )
    axes[0].axhline(20, color="#b23a48", linestyle=":", label="upper <=20%")
    axes[0].axhline(80, color="#8c6d31", linestyle="-.", label="lower >=80%")
    axes[0].axhline(90, color="#333333", linestyle="--", label="lower >=90%")
    axes[0].set_xticks(range(len(CONDITIONS)), labels, rotation=45, ha="right")
    axes[0].set_ylim(0, 103)
    axes[0].set_ylabel("Query accuracy (%)")
    axes[0].set_title("Fixed N=8192 causal conditions (95% Wilson CI)")
    axes[0].legend(loc="center right")

    intervention_names = ["zero", "batch roll", "keep only"]
    l2_names = ["zero_l2", "batch_roll_l2", "keep_l2"]
    l3_names = ["zero_l3", "batch_roll_l3", "keep_l3"]
    x = np.arange(3)
    width = 0.34
    for offset, names, label, color in (
        (-width / 2, l2_names, "L2", "#e45756"),
        (width / 2, l3_names, "L3", "#72b7b2"),
    ):
        layer_values = np.array([100 * metrics[name]["query"] for name in names])
        layer_lowers = np.array([
            100 * metrics[name]["query_wilson95"][0] for name in names
        ])
        layer_uppers = np.array([
            100 * metrics[name]["query_wilson95"][1] for name in names
        ])
        axes[1].bar(x + offset, layer_values, width, label=label, color=color)
        axes[1].errorbar(
            x + offset, layer_values,
            yerr=np.vstack([
                layer_values - layer_lowers,
                layer_uppers - layer_values,
            ]),
            fmt="none", ecolor="black", capsize=3, linewidth=1,
        )
    axes[1].axhline(20, color="#b23a48", linestyle=":")
    axes[1].axhline(80, color="#8c6d31", linestyle="-.")
    axes[1].axhline(90, color="#333333", linestyle="--")
    axes[1].set_xticks(x, intervention_names)
    axes[1].set_ylim(0, 103)
    axes[1].set_ylabel("Query accuracy (%)")
    axes[1].set_title("Preregistered L2 route versus matched L3 contrast")
    axes[1].legend()
    fig.suptitle("IST Level 7.3.1: High-Precision Seed1879 L2 Route", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_smoke(args: argparse.Namespace) -> int:
    args.output = "experiments/level7_3_1/smoke"
    args.chunks = 2
    args.chunk_size = 32
    args.samples = 8
    args.eval_batch_size = 4
    args.dataset_seed = 7310023
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    source_audit = validate_source()
    state = torch.load(ROOT / SOURCE_CHECKPOINT, map_location=device, weights_only=False)
    model, _ = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = model_fingerprint(model)
    metrics = {
        condition: evaluate_condition(model, args, condition, device, dtype)
        for condition in CONDITIONS
    }
    diagnosis = diagnose(metrics, args)
    after = model_fingerprint(model)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "conditions": list(metrics),
        "diagnosis_exercised": diagnosis["classification"],
        "source_checkpoint_sha256": source_audit["checkpoint_sha256"],
        "source_fingerprint_match": before == EXPECTED_MODEL_FINGERPRINT,
        "fingerprint_unchanged": before == after,
        "passed": bool(
            set(metrics) == set(CONDITIONS)
            and before == EXPECTED_MODEL_FINGERPRINT
            and before == after
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
    protocol = read_json(STATIC_PREREGISTRATION)
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
    source_audit = validate_source()
    progress = {
        "stage": "high_precision_causal_panel",
        "seed": 1879,
        "conditions": CONDITIONS,
        "completed_conditions": [],
        "fixed_samples": args.samples,
        "dataset_seed": args.dataset_seed,
        "seed909_locked": True,
    }
    progress_path = root / "progress.json"
    atomic_save(progress_path, progress)

    state = torch.load(ROOT / SOURCE_CHECKPOINT, map_location=device, weights_only=False)
    model, _ = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fingerprint_before = model_fingerprint(model)
    if fingerprint_before != EXPECTED_MODEL_FINGERPRINT:
        raise RuntimeError("Loaded model fingerprint does not match preregistration")

    condition_path = root / "condition_progress.json"
    metrics = (
        read_json(condition_path)
        if condition_path.exists() and not args.force else {}
    )
    validate_resumed_rows(metrics, args)
    for condition in CONDITIONS:
        if condition in metrics:
            continue
        metric = evaluate_condition(model, args, condition, device, dtype)
        metrics[condition] = metric
        atomic_save(condition_path, metrics)
        progress["completed_conditions"] = [
            name for name in CONDITIONS if name in metrics
        ]
        atomic_save(progress_path, progress)
        low, high = metric["query_wilson95"]
        print(
            f"seed=1879 condition={condition} query={metric['query']:.2%} "
            f"wilson95=[{low:.2%},{high:.2%}] local={metric['local']:.2%}",
            flush=True,
        )

    fingerprint_after = model_fingerprint(model)
    diagnosis = diagnose(metrics, args)
    model_integrity = {
        "model_fingerprint_before": fingerprint_before,
        "model_fingerprint_after": fingerprint_after,
        "model_fingerprint_unchanged": fingerprint_before == fingerprint_after,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "all_conditions_present": set(metrics) == set(CONDITIONS),
        "condition_order_exact": list(metrics) == CONDITIONS,
        "fixed_samples_every_condition": all(
            row["samples"] == args.samples for row in metrics.values()
        ),
        "fresh_dataset_seed": args.dataset_seed,
        "old_panel_reused": args.dataset_seed in OLD_PANEL_SEEDS,
    }
    model_integrity["passed"] = bool(
        model_integrity["model_fingerprint_unchanged"]
        and model_integrity["all_parameters_frozen"]
        and model_integrity["all_conditions_present"]
        and model_integrity["condition_order_exact"]
        and model_integrity["fixed_samples_every_condition"]
        and not model_integrity["old_panel_reused"]
    )
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "source_validation_passed": source_audit["source_validation_passed"],
        "checkpoint_sha256_exact": (
            source_audit["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
        ),
        "conditions_exact": set(metrics) == set(CONDITIONS),
        "fixed_N_completed": all(
            row["samples"] == args.samples for row in metrics.values()
        ),
        "no_training_or_selection": True,
        "no_old_panel_reuse": args.dataset_seed not in OLD_PANEL_SEEDS,
        "seed909_used": False,
        "model_integrity": model_integrity,
    }
    integrity["passed"] = bool(
        integrity["source_validation_passed"]
        and integrity["checkpoint_sha256_exact"]
        and integrity["conditions_exact"]
        and integrity["fixed_N_completed"]
        and integrity["no_training_or_selection"]
        and integrity["no_old_panel_reuse"]
        and not integrity["seed909_used"]
        and model_integrity["passed"]
    )
    elapsed_seconds = time.perf_counter() - started
    result = {
        "protocol": protocol,
        "source_audit": source_audit,
        "metrics": metrics,
        "diagnosis": diagnosis,
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    summary = {
        "integrity": integrity,
        "diagnosis": diagnosis,
        "conditions": {
            name: {
                "query": metrics[name]["query"],
                "query_wilson95": metrics[name]["query_wilson95"],
                "local": metrics[name]["local"],
                "local_wilson95": metrics[name]["local_wilson95"],
            }
            for name in CONDITIONS
        },
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_precision(metrics, root / "route_precision.png")
    atomic_save(progress_path, {
        "stage": "complete",
        "seed": 1879,
        "completed_conditions": CONDITIONS,
        "fixed_samples": args.samples,
        "classification": diagnosis["classification"],
        "integrity_passed": integrity["passed"],
        "seed909_locked": True,
    })
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
