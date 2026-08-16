"""Level 7.4 frozen seed1879 L2/L3 Memory-route formation trajectory."""

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
LEVEL_DIR = ROOT / "experiments" / "level7_4"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
OLD_PANEL_SEEDS = (7218790, 7218791, 7218792, 7300000, 7310000)
TIMELINE = [
    {
        "id": "curriculum_2",
        "checkpoint": "experiments/level7_2/formal/seed1879/curriculum_stage1.pt",
        "stage": "curriculum",
        "trained_chunks": 2,
        "expected_sha256": "9939755860050c602798b6cec0320ac68fd5197876f389f802d461669034fd6c",
    },
    {
        "id": "curriculum_4",
        "checkpoint": "experiments/level7_2/formal/seed1879/curriculum_stage2.pt",
        "stage": "curriculum",
        "trained_chunks": 4,
        "expected_sha256": "2703f5ae720e0f5a973244bbca8275b6654c7c25b3c17c95e7dd618dcec4ebbf",
    },
    {
        "id": "curriculum_8",
        "checkpoint": "experiments/level7_2/formal/seed1879/curriculum_stage3.pt",
        "stage": "curriculum",
        "trained_chunks": 8,
        "expected_sha256": "572338a38c001b65da5e9202de1a867ae67aa8c225d4d682b2253e29850d1df7",
    },
    {
        "id": "curriculum_16",
        "checkpoint": "experiments/level7_2/formal/seed1879/curriculum_stage4.pt",
        "stage": "curriculum",
        "trained_chunks": 16,
        "expected_sha256": "7c9699be19b23d98d5b0c352a544c9bdbd7879d6aee202438614c4224e3e12f6",
    },
    {
        "id": "withdrawal_p20",
        "checkpoint": "experiments/level7_2/formal/seed1879/withdrawal_phase1.pt",
        "stage": "withdrawal",
        "probe_weight": 0.2,
        "phase_steps": 300,
        "expected_sha256": "03238536ce675f3c878ccce9bd4774e32ba0b6595f37c4180615d7a51f1e82a7",
    },
    {
        "id": "withdrawal_p10",
        "checkpoint": "experiments/level7_2/formal/seed1879/withdrawal_phase2.pt",
        "stage": "withdrawal",
        "probe_weight": 0.1,
        "phase_steps": 300,
        "expected_sha256": "bf316e33722a7574e3c641fe2c24f814c0b81418a7e50a754cff54d5ad72fb11",
    },
    {
        "id": "zero_0300",
        "checkpoint": "experiments/level7_2/formal/seed1879/zero_probe_step0300.pt",
        "stage": "zero_probe",
        "probe_weight": 0.0,
        "zero_probe_step": 300,
        "expected_sha256": "04d1b7a46c01df3090b2f42f21ccb76b41a19e9554f6c08817a8d61e55e5b897",
    },
    {
        "id": "zero_0450",
        "checkpoint": "experiments/level7_2/formal/seed1879/zero_probe_step0450.pt",
        "stage": "zero_probe",
        "probe_weight": 0.0,
        "zero_probe_step": 450,
        "expected_sha256": "631eb20256ff1c06ee3c3f16c870c682e4a3d4debf5dd1c144dc09776a3599c7",
    },
    {
        "id": "zero_0600",
        "checkpoint": "experiments/level7_2/formal/seed1879/zero_probe_step0600.pt",
        "stage": "zero_probe",
        "probe_weight": 0.0,
        "zero_probe_step": 600,
        "expected_sha256": "e3d42c5cedb17cf46463cdc2aaab7dbd5662d86d06a32fa1cf0fe7f6d1b4e68d",
    },
    {
        "id": "zero_0750",
        "checkpoint": "experiments/level7_2/formal/seed1879/zero_probe_step0750.pt",
        "stage": "zero_probe",
        "probe_weight": 0.0,
        "zero_probe_step": 750,
        "expected_sha256": "cbba0c6db219e16be274bd0e77612972b7ad7d24ffa2bddb7b7cb858bf7a74f6",
    },
]
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
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=7400000)
    parser.add_argument("--formed-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--core-retention-threshold", type=float, default=0.80)
    parser.add_argument("--pair-sufficiency-threshold", type=float, default=0.90)
    parser.add_argument("--pair-gain-threshold", type=float, default=0.03)
    parser.add_argument("--roll-drop-threshold", type=float, default=0.05)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--output", default="experiments/level7_4/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "chunks": 16,
        "chunk_size": 128,
        "samples": 1024,
        "eval_batch_size": 16,
        "dataset_seed": 7400000,
        "formed_threshold": 0.90,
        "disruption_threshold": 0.20,
        "core_retention_threshold": 0.80,
        "pair_sufficiency_threshold": 0.90,
        "pair_gain_threshold": 0.03,
        "roll_drop_threshold": 0.05,
        "local_threshold": 0.90,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.4 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_4/formal":
        raise ValueError("Formal output path is locked")


def validate_sources(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    registered = protocol["frozen_timeline"]
    for expected, actual in zip(TIMELINE, registered, strict=True):
        for key in ("id", "checkpoint", "expected_sha256"):
            if expected[key] != actual[key]:
                raise RuntimeError(f"Static timeline mismatch at {expected['id']}: {key}")

    level7_2_path = ROOT / "experiments/level7_2/formal/result.json"
    level7_2 = read_json(level7_2_path)
    if not level7_2["integrity"]["passed"]:
        raise RuntimeError("Level 7.2 source integrity did not pass")
    training_path = ROOT / "experiments/level7_2/formal/seed1879/training_result.json"
    training = read_json(training_path)
    if int(training["seed"]) != 1879 or not training["reached_candidates"]:
        raise RuntimeError("Seed1879 did not reach the registered timeline")
    stages = training["stages"]
    if [row["chunks"] for row in stages] != [2, 4, 8, 16]:
        raise RuntimeError("Seed1879 curriculum stage order changed")
    if not all(row["passed"] for row in stages):
        raise RuntimeError("Seed1879 curriculum contains a failed source stage")
    if [row["step"] for row in training["candidates"]] != [300, 450, 600, 750]:
        raise RuntimeError("Seed1879 candidate timeline changed")

    level7_3_path = ROOT / "experiments/level7_3/formal/result.json"
    level7_3 = read_json(level7_3_path)
    if not level7_3["integrity"]["passed"]:
        raise RuntimeError("Level 7.3 parent integrity did not pass")
    if (
        level7_3["diagnosis"]["classification"]
        != "cross_initialization_layer_heterogeneity_confirmed"
    ):
        raise RuntimeError("Unexpected Level 7.3 parent classification")

    level7_3_1_path = ROOT / "experiments/level7_3_1/formal/result.json"
    level7_3_1 = read_json(level7_3_1_path)
    if not level7_3_1["integrity"]["passed"]:
        raise RuntimeError("Level 7.3.1 parent integrity did not pass")
    if (
        level7_3_1["diagnosis"]["classification"]
        != "l2_route_supported_but_single_layer_sufficiency_inconclusive"
    ):
        raise RuntimeError("Unexpected Level 7.3.1 parent classification")

    audit = []
    for index, source in enumerate(TIMELINE):
        checkpoint = ROOT / source["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != source["expected_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {source['id']}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" not in state:
            raise RuntimeError(f"Checkpoint lacks model state: {source['id']}")
        if source["stage"] == "curriculum":
            saved_stages = state.get("stages", [])
            if len(saved_stages) != index + 1:
                raise RuntimeError(f"Curriculum metadata mismatch: {source['id']}")
            if saved_stages[-1]["chunks"] != source["trained_chunks"]:
                raise RuntimeError(f"Curriculum chunk metadata mismatch: {source['id']}")
        if source["stage"] == "zero_probe":
            if int(state.get("zero_probe_step", -1)) != source["zero_probe_step"]:
                raise RuntimeError(f"Zero-Probe step mismatch: {source['id']}")
        del state
        audit.append({
            **source,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "source_index": index,
            "source_validation_passed": True,
        })
    parent_audit = {
        "level7_2_result_sha256": sha256_file(level7_2_path),
        "seed1879_training_result_sha256": sha256_file(training_path),
        "level7_3_result_sha256": sha256_file(level7_3_path),
        "level7_3_1_result_sha256": sha256_file(level7_3_1_path),
    }
    for row in audit:
        row["parent_audit"] = parent_audit
    return audit


def classify_checkpoint(
    metrics: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    query = {name: metrics[name]["query"] for name in CONDITIONS}
    minimum_local = min(metrics[name]["local"] for name in CONDITIONS)
    behavior_formed = bool(
        query["intact"] >= args.formed_threshold
        and minimum_local >= args.local_threshold
    )
    max_whole_disruption = max(
        query[name] for name in ("reset_all", "zero_all", "batch_roll_all")
    )
    whole_memory_causal = bool(
        behavior_formed and max_whole_disruption <= args.disruption_threshold
    )
    pair_gain = query["keep_l2_l3"] - query["keep_l2"]
    l3_roll_drop = query["intact"] - query["batch_roll_l3"]
    l2_core = bool(
        whole_memory_causal
        and query["zero_l2"] <= args.disruption_threshold
        and query["batch_roll_l2"] <= args.disruption_threshold
        and query["keep_l2"] >= args.core_retention_threshold
        and query["zero_l3"] >= args.core_retention_threshold
        and query["keep_l3"] <= args.disruption_threshold
        and query["keep_l2_l3"] >= args.pair_sufficiency_threshold
    )
    l3_support = bool(
        pair_gain >= args.pair_gain_threshold
        and l3_roll_drop >= args.roll_drop_threshold
    )
    l3_core = bool(
        whole_memory_causal
        and query["zero_l3"] <= args.disruption_threshold
        and query["batch_roll_l3"] <= args.disruption_threshold
        and query["keep_l3"] >= args.core_retention_threshold
        and query["zero_l2"] >= args.core_retention_threshold
        and query["keep_l2"] <= args.disruption_threshold
    )
    if not behavior_formed:
        route_class = "unformed_behavior"
    elif not whole_memory_causal:
        route_class = "formed_noncausal_memory"
    elif l2_core and l3_support:
        route_class = "l2_core_l3_supported"
    elif l2_core:
        route_class = "l2_core_minimal_l3_support"
    elif l3_core:
        route_class = "l3_core"
    else:
        route_class = "distributed_or_other"
    return {
        "route_class": route_class,
        "behavior_formed": behavior_formed,
        "whole_memory_causal": whole_memory_causal,
        "l2_core": l2_core,
        "l3_support": l3_support,
        "l3_core": l3_core,
        "minimum_local": minimum_local,
        "max_whole_disruption": max_whole_disruption,
        "effects": {
            "l2_zero_drop": query["intact"] - query["zero_l2"],
            "l2_roll_drop": query["intact"] - query["batch_roll_l2"],
            "l3_zero_drop": query["intact"] - query["zero_l3"],
            "l3_roll_drop": l3_roll_drop,
            "l2_single_retention": query["keep_l2"],
            "l3_single_retention": query["keep_l3"],
            "l2_l3_pair_retention": query["keep_l2_l3"],
            "l3_pair_gain_over_l2": pair_gain,
            "pair_minus_intact": query["keep_l2_l3"] - query["intact"],
        },
    }


def diagnose_trajectory(results: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["id"] for row in results]
    expected_ids = [row["id"] for row in TIMELINE]
    if ids != expected_ids:
        raise RuntimeError("Trajectory order is incomplete or changed")
    by_id = {row["id"]: row for row in results}
    target_class = "l2_core_l3_supported"
    post_ids = expected_ids[3:]
    final_tail = ["zero_0450", "zero_0600", "zero_0750"]
    post_classes = [by_id[name]["profile"]["route_class"] for name in post_ids]
    formed_causal_post_classes = sorted({
        by_id[name]["profile"]["route_class"]
        for name in post_ids
        if by_id[name]["profile"]["behavior_formed"]
        and by_id[name]["profile"]["whole_memory_causal"]
    })
    curriculum16_target = (
        by_id["curriculum_16"]["profile"]["route_class"] == target_class
    )
    final_tail_target = all(
        by_id[name]["profile"]["route_class"] == target_class
        for name in final_tail
    )
    if all(route_class == target_class for route_class in post_classes):
        classification = "l2_core_l3_support_established_by_16chunk_and_stable"
    elif not curriculum16_target and final_tail_target:
        classification = "l2_core_l3_support_emerges_during_withdrawal"
    elif curriculum16_target and not final_tail_target:
        classification = "post_curriculum_route_destabilization"
    elif len(formed_causal_post_classes) >= 2:
        classification = "post_curriculum_route_migration_observed"
    else:
        classification = "trajectory_unresolved"

    earliest_formed = next(
        (
            row["id"] for row in results
            if row["profile"]["behavior_formed"]
        ),
        None,
    )
    earliest_target = next(
        (
            row["id"] for row in results
            if row["profile"]["route_class"] == target_class
        ),
        None,
    )
    stable_suffix = None
    for index, row in enumerate(results):
        if all(
            later["profile"]["route_class"] == target_class
            for later in results[index:]
        ):
            stable_suffix = row["id"]
            break
    transitions = []
    for previous, current in zip(results, results[1:]):
        old_class = previous["profile"]["route_class"]
        new_class = current["profile"]["route_class"]
        if old_class != new_class:
            transitions.append({
                "from": previous["id"],
                "to": current["id"],
                "from_class": old_class,
                "to_class": new_class,
            })
    return {
        "classification": classification,
        "checkpoints": len(results),
        "earliest_formed_16chunk_behavior": earliest_formed,
        "earliest_l2_core_l3_supported": earliest_target,
        "earliest_stable_l2_core_l3_supported_suffix": stable_suffix,
        "post_curriculum_route_classes": {
            name: by_id[name]["profile"]["route_class"] for name in post_ids
        },
        "formed_causal_post_classes": formed_causal_post_classes,
        "adjacent_route_transitions": transitions,
        "final_endpoint_reproduces_parent_signature": (
            by_id["zero_0750"]["profile"]["route_class"] == target_class
        ),
        "registered_stop_boundary": (
            "Report the fixed trajectory; do not add checkpoints, samples, "
            "conditions, or retraining after observing results."
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


def run_checkpoint(
    source: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    folder = root / source["id"]
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        result = read_json(result_path)
        if result["source"]["checkpoint_sha256"] != source["checkpoint_sha256"]:
            raise RuntimeError(f"Cached source mismatch: {source['id']}")
        return result
    state = torch.load(ROOT / source["checkpoint"], map_location=device, weights_only=False)
    model, probe = build(device, args.chunk_size)
    del probe
    model.load_state_dict(state["model"])
    del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fingerprint_before = model_fingerprint(model)
    condition_path = folder / "condition_progress.json"
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
        print(
            f"checkpoint={source['id']} condition={condition} "
            f"query={metric['query']:.2%} local={metric['local']:.2%}",
            flush=True,
        )
    fingerprint_after = model_fingerprint(model)
    profile = classify_checkpoint(metrics, args)
    integrity = {
        "checkpoint_sha256": source["checkpoint_sha256"],
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
    }
    integrity["passed"] = bool(
        integrity["model_fingerprint_unchanged"]
        and integrity["all_parameters_frozen"]
        and integrity["all_conditions_present"]
        and integrity["condition_order_exact"]
        and integrity["fixed_samples_every_condition"]
    )
    result = {
        "id": source["id"],
        "source": source,
        "metrics": metrics,
        "profile": profile,
        "integrity": integrity,
    }
    atomic_save(result_path, result)
    del model
    torch.cuda.empty_cache()
    return result


def plot_trajectory(results: list[dict[str, Any]], path: Path) -> None:
    labels = ["C2", "C4", "C8", "C16", "W.2", "W.1", "Z300", "Z450", "Z600", "Z750"]
    x = np.arange(len(results))
    fig, axes = plt.subplots(3, 1, figsize=(17, 15), sharex=True)

    panels = [
        (
            axes[0],
            [("intact", "intact"), ("reset_all", "reset all"),
             ("zero_all", "zero all"), ("batch_roll_all", "roll all")],
            "Behavior and complete-Memory controls",
        ),
        (
            axes[1],
            [("zero_l2", "zero L2"), ("batch_roll_l2", "roll L2"),
             ("zero_l3", "zero L3"), ("batch_roll_l3", "roll L3")],
            "Layer necessity and sample alignment",
        ),
        (
            axes[2],
            [("keep_l2", "keep L2"), ("keep_l3", "keep L3"),
             ("keep_l2_l3", "keep L2+L3")],
            "Single- and paired-layer retention",
        ),
    ]
    for axis, specs, title in panels:
        for condition, label in specs:
            values = np.array([
                100 * row["metrics"][condition]["query"] for row in results
            ])
            lowers = np.array([
                100 * row["metrics"][condition]["query_wilson95"][0]
                for row in results
            ])
            uppers = np.array([
                100 * row["metrics"][condition]["query_wilson95"][1]
                for row in results
            ])
            axis.plot(x, values, marker="o", linewidth=2, label=label)
            axis.fill_between(x, lowers, uppers, alpha=0.10)
        axis.axhline(20, color="#b23a48", linestyle=":", linewidth=1.5)
        axis.axhline(80, color="#8c6d31", linestyle="-.", linewidth=1.2)
        axis.axhline(90, color="#333333", linestyle="--", linewidth=1.2)
        axis.axvline(3.5, color="#777777", linestyle=":")
        axis.axvline(5.5, color="#777777", linestyle=":")
        axis.set_ylim(0, 105)
        axis.set_ylabel("Query accuracy (%)")
        axis.set_title(title)
        axis.legend(ncol=4, fontsize=9, loc="lower right")

    abbreviations = {
        "unformed_behavior": "unformed",
        "formed_noncausal_memory": "noncausal",
        "l2_core_l3_supported": "L2 core + L3 support",
        "l2_core_minimal_l3_support": "L2 core",
        "l3_core": "L3 core",
        "distributed_or_other": "other",
    }
    for index, row in enumerate(results):
        axes[0].text(
            index, 102, abbreviations[row["profile"]["route_class"]],
            ha="center", va="top", fontsize=7, rotation=30,
        )
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel("Frozen training checkpoint")
    fig.suptitle(
        "IST Level 7.4: Seed1879 L2/L3 Causal Route Formation Trajectory",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_smoke(args: argparse.Namespace) -> int:
    args.output = "experiments/level7_4/smoke"
    args.chunks = 2
    args.chunk_size = 32
    args.samples = 8
    args.eval_batch_size = 4
    args.dataset_seed = 7400023
    protocol = read_json(STATIC_PREREGISTRATION)
    source_audit = validate_sources(protocol)
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    results = []
    fingerprints_unchanged = True
    for source in source_audit:
        state = torch.load(ROOT / source["checkpoint"], map_location=device, weights_only=False)
        model, probe = build(device, args.chunk_size)
        del probe
        model.load_state_dict(state["model"])
        del state
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = model_fingerprint(model)
        metrics = {
            condition: evaluate_condition(model, args, condition, device, dtype)
            for condition in CONDITIONS
        }
        after = model_fingerprint(model)
        fingerprints_unchanged &= before == after
        results.append({
            "id": source["id"],
            "profile": classify_checkpoint(metrics, args),
        })
        del model
        torch.cuda.empty_cache()
    diagnosis = diagnose_trajectory(results)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "checkpoints": len(results),
        "conditions_per_checkpoint": len(CONDITIONS),
        "trajectory_classification_exercised": diagnosis["classification"],
        "all_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "fingerprints_unchanged": fingerprints_unchanged,
    }
    result["passed"] = bool(
        result["checkpoints"] == len(TIMELINE)
        and result["conditions_per_checkpoint"] == len(CONDITIONS)
        and result["all_source_hashes_validated"]
        and result["fingerprints_unchanged"]
    )
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "result.json", result)
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
    source_audit = validate_sources(protocol)
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
        "stage": "frozen_route_trajectory",
        "timeline": [row["id"] for row in source_audit],
        "completed_checkpoints": [],
        "samples_per_condition": args.samples,
        "dataset_seed": args.dataset_seed,
        "seed909_locked": True,
    }
    progress_path = root / "progress.json"
    atomic_save(progress_path, progress)
    results = []
    for source in source_audit:
        result = run_checkpoint(source, args, device, dtype, root)
        results.append(result)
        progress["completed_checkpoints"].append(source["id"])
        atomic_save(progress_path, progress)
    diagnosis = diagnose_trajectory(results)
    checkpoint_integrity = all(row["integrity"]["passed"] for row in results)
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "timeline_exact": [row["id"] for row in results]
        == [row["id"] for row in TIMELINE],
        "all_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "all_checkpoint_integrity_passed": checkpoint_integrity,
        "conditions_exact": all(
            set(row["metrics"]) == set(CONDITIONS) for row in results
        ),
        "fixed_N_completed": all(
            metric["samples"] == args.samples
            for row in results for metric in row["metrics"].values()
        ),
        "shared_fresh_dataset_seed": args.dataset_seed,
        "no_parent_panel_reuse": args.dataset_seed not in OLD_PANEL_SEEDS,
        "no_training_or_selection": True,
        "seed909_used": False,
    }
    integrity["passed"] = bool(
        integrity["timeline_exact"]
        and integrity["all_source_hashes_validated"]
        and integrity["all_checkpoint_integrity_passed"]
        and integrity["conditions_exact"]
        and integrity["fixed_N_completed"]
        and integrity["no_parent_panel_reuse"]
        and integrity["no_training_or_selection"]
        and not integrity["seed909_used"]
    )
    elapsed_seconds = time.perf_counter() - started
    result = {
        "protocol": protocol,
        "source_audit": source_audit,
        "runs": results,
        "diagnosis": diagnosis,
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    summary = {
        "integrity": integrity,
        "diagnosis": diagnosis,
        "trajectory": [
            {
                "id": row["id"],
                "stage": row["source"]["stage"],
                "route_class": row["profile"]["route_class"],
                "behavior_formed": row["profile"]["behavior_formed"],
                "whole_memory_causal": row["profile"]["whole_memory_causal"],
                "intact_query": row["metrics"]["intact"]["query"],
                "zero_l2": row["metrics"]["zero_l2"]["query"],
                "batch_roll_l2": row["metrics"]["batch_roll_l2"]["query"],
                "keep_l2": row["metrics"]["keep_l2"]["query"],
                "zero_l3": row["metrics"]["zero_l3"]["query"],
                "batch_roll_l3": row["metrics"]["batch_roll_l3"]["query"],
                "keep_l3": row["metrics"]["keep_l3"]["query"],
                "keep_l2_l3": row["metrics"]["keep_l2_l3"]["query"],
                "effects": row["profile"]["effects"],
            }
            for row in results
        ],
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_trajectory(results, root / "route_trajectory.png")
    atomic_save(progress_path, {
        "stage": "complete",
        "completed_checkpoints": [row["id"] for row in results],
        "conditions_per_checkpoint": len(CONDITIONS),
        "classification": diagnosis["classification"],
        "integrity_passed": integrity["passed"],
        "seed909_locked": True,
    })
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
