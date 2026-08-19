"""Level 7.5.1 endpoint-qualified dense fixed-to-C2 route replay."""

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
from run_level6_6_local import build, random_step, restore
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save, model_fingerprint
from run_level7_3_local import CONDITIONS, evaluate_condition, read_json, sha256_file
from run_level7_4_1_local import (
    atomic_torch_save,
    canonical_fingerprint,
    rng_equal,
    run_mini_branch,
    state_dict_fingerprint,
)
from run_level7_5_local import run_causal_milestone as run_base_causal_milestone


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_5_1"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
FORMAL_SEEDS = [2203, 2551, 2909, 1879]
L3_DEFAULT_SEEDS = [2203, 2551, 2909]
COMPARISON_SEED = 1879
WHOLE_CONTROLS = ("reset_all", "zero_all", "batch_roll_all")

SOURCE_SPECS = [
    {
        "seed": 2203,
        "route_group": "Level7.5_L3_default",
        "start_checkpoint": "experiments/level7_5/formal/seed2203/level6_1.pt",
        "start_checkpoint_sha256": "56f16453bb76276d219c1c41eafe856f5c9cabeb1ecbe096ec6b3ec4a90b37b9",
        "reference_endpoint": "experiments/level7_5/formal/seed2203/curriculum_stage1.pt",
        "reference_endpoint_sha256": "9632d6ed078ed328a2541d680325590fa3c1c92459efcc5837735fbb7b67c141",
        "reference_stop_step": 1000,
    },
    {
        "seed": 2551,
        "route_group": "Level7.5_L3_default",
        "start_checkpoint": "experiments/level7_5/formal/seed2551/level6_1.pt",
        "start_checkpoint_sha256": "0f9fff5aaa37ef952a295212eb14c966ddcc65115e0310dea64f9e941fee0125",
        "reference_endpoint": "experiments/level7_5/formal/seed2551/curriculum_stage1.pt",
        "reference_endpoint_sha256": "6e0b478978e7cbb39e3111a90edf64358b41d44986e044d96fd35317e3617100",
        "reference_stop_step": 800,
    },
    {
        "seed": 2909,
        "route_group": "Level7.5_L3_default",
        "start_checkpoint": "experiments/level7_5/formal/seed2909/level6_1.pt",
        "start_checkpoint_sha256": "e52dc48b363fd4b3a90211db2cfbfe39f21269b2b25bd6e276bfe30b64071579",
        "reference_endpoint": "experiments/level7_5/formal/seed2909/curriculum_stage1.pt",
        "reference_endpoint_sha256": "9bedb9025827207586d15e064e30a0e8b9c6ef6510f0369697572784db28ad2f",
        "reference_stop_step": 800,
    },
    {
        "seed": 1879,
        "route_group": "exceptional_L2_core_L3_support",
        "start_checkpoint": "experiments/level7_2/formal/seed1879/level6_1.pt",
        "start_checkpoint_sha256": "ffd969e6873f373f8a96132ff22387faa0b95dc05a52c6f08bda441acda9141a",
        "reference_endpoint": "experiments/level7_2/formal/seed1879/curriculum_stage1.pt",
        "reference_endpoint_sha256": "9939755860050c602798b6cec0320ac68fd5197876f389f802d461669034fd6c",
        "reference_stop_step": 2300,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--training-chunks", type=int, default=2)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--maximum-steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--training-eval-batches", type=int, default=10)
    parser.add_argument("--training-eval-batch-size", type=int, default=8)
    parser.add_argument("--training-gate", type=float, default=0.95)
    parser.add_argument("--causal-chunks", type=int, default=16)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--causal-eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=7510000)
    parser.add_argument("--formed-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--core-retention-threshold", type=float, default=0.80)
    parser.add_argument("--pair-sufficiency-threshold", type=float, default=0.90)
    parser.add_argument("--pair-gain-threshold", type=float, default=0.03)
    parser.add_argument("--roll-drop-threshold", type=float, default=0.05)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--precursor-intact-threshold", type=float, default=0.75)
    parser.add_argument("--precursor-retention-threshold", type=float, default=0.70)
    parser.add_argument("--weak-l3-intact-threshold", type=float, default=0.20)
    parser.add_argument("--weak-l3-retention-threshold", type=float, default=0.20)
    parser.add_argument("--weak-l3-preservation-margin", type=float, default=0.05)
    parser.add_argument("--weak-l3-selectivity-gap", type=float, default=0.10)
    parser.add_argument("--output", default="experiments/level7_5_1/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "chunk_size": 128,
        "training_chunks": 2,
        "training_batch_size": 8,
        "learning_rate": 1e-3,
        "probe_weight": 0.5,
        "maximum_steps": 2500,
        "eval_every": 100,
        "training_eval_batches": 10,
        "training_eval_batch_size": 8,
        "training_gate": 0.95,
        "causal_chunks": 16,
        "causal_samples": 1024,
        "causal_eval_batch_size": 16,
        "dataset_seed": 7510000,
        "formed_threshold": 0.90,
        "disruption_threshold": 0.20,
        "core_retention_threshold": 0.80,
        "pair_sufficiency_threshold": 0.90,
        "pair_gain_threshold": 0.03,
        "roll_drop_threshold": 0.05,
        "local_threshold": 0.90,
        "precursor_intact_threshold": 0.75,
        "precursor_retention_threshold": 0.70,
        "weak_l3_intact_threshold": 0.20,
        "weak_l3_retention_threshold": 0.20,
        "weak_l3_preservation_margin": 0.05,
        "weak_l3_selectivity_gap": 0.10,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.5.1 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_5_1/formal":
        raise ValueError("Formal output path is locked")


def validate_static_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("frozen_sources") != SOURCE_SPECS:
        raise RuntimeError("Static Level 7.5.1 source registration changed")
    if protocol["fresh_shared_causal_panel"].get("conditions") != CONDITIONS:
        raise RuntimeError("Static Level 7.5.1 condition order changed")
    if protocol["fresh_shared_causal_panel"].get("dataset_seed") != 7510000:
        raise RuntimeError("Static Level 7.5.1 dataset seed changed")


def reference_history(state: dict[str, Any]) -> list[dict[str, Any]]:
    history = state.get("history", [])
    return [row for row in history if int(row.get("stage", 1)) == 1]


def reference_stop(state: dict[str, Any]) -> int:
    stage_result = state.get("stage_result")
    if stage_result is not None:
        return int(stage_result["steps"])
    stages = state.get("stages", [])
    if not stages:
        raise RuntimeError("Reference C2 endpoint lacks stage metadata")
    return int(stages[-1]["steps"])


def behavior_consecutive(history: list[dict[str, Any]], threshold: float) -> int:
    consecutive = 0
    for row in history:
        consecutive = consecutive + 1 if row["query"] >= threshold else 0
    return consecutive


def validate_sources(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_static_protocol(protocol)
    parent_75_path = ROOT / "experiments/level7_5/formal/result.json"
    parent_741_path = ROOT / "experiments/level7_4_1/formal/result.json"
    parent_75 = read_json(parent_75_path)
    parent_741 = read_json(parent_741_path)
    if not parent_75["integrity"]["passed"]:
        raise RuntimeError("Level 7.5 parent integrity did not pass")
    if parent_75["diagnosis"]["classification"] != "alternative_route_formation_observed":
        raise RuntimeError("Unexpected Level 7.5 parent classification")
    if parent_741["diagnosis"]["classification"] != "exact_replay_and_single_stable_formation_transition":
        raise RuntimeError("Unexpected Level 7.4.1 parent classification")
    parent_routes = {
        int(row["seed"]): row["diagnosis"]["endpoint_route"]
        for row in parent_75["runs"]
    }
    if any(parent_routes.get(seed) != "l3_core" for seed in L3_DEFAULT_SEEDS):
        raise RuntimeError("One or more Level 7.5 sources lacks the registered L3 route")
    audits = []
    for spec in SOURCE_SPECS:
        start_path = ROOT / spec["start_checkpoint"]
        endpoint_path = ROOT / spec["reference_endpoint"]
        if not start_path.is_file() or not endpoint_path.is_file():
            raise FileNotFoundError(start_path if not start_path.is_file() else endpoint_path)
        if sha256_file(start_path) != spec["start_checkpoint_sha256"]:
            raise RuntimeError(f"Start checkpoint hash changed: seed={spec['seed']}")
        if sha256_file(endpoint_path) != spec["reference_endpoint_sha256"]:
            raise RuntimeError(f"Reference endpoint hash changed: seed={spec['seed']}")
        endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
        history = reference_history(endpoint)
        stop_step = reference_stop(endpoint)
        consecutive = behavior_consecutive(history, 0.95)
        if stop_step != spec["reference_stop_step"] or history[-1]["step"] != stop_step:
            raise RuntimeError(f"Reference stop mismatch: seed={spec['seed']}")
        if consecutive != 2:
            raise RuntimeError(f"Reference consecutive-pass state changed: seed={spec['seed']}")
        audits.append(
            {
                **spec,
                "start_checkpoint_size_bytes": start_path.stat().st_size,
                "reference_endpoint_size_bytes": endpoint_path.stat().st_size,
                "reference_model_fingerprint": state_dict_fingerprint(endpoint["model"]),
                "reference_probe_fingerprint": state_dict_fingerprint(endpoint["probe"]),
                "reference_optimizer_fingerprint": canonical_fingerprint(endpoint["optimizer"]),
                "reference_CPU_RNG_fingerprint": canonical_fingerprint(endpoint["cpu_rng"]),
                "reference_CUDA_RNG_fingerprint": canonical_fingerprint(endpoint["cuda_rng"]),
                "reference_validation_history": history,
                "reference_consecutive_behavior_passes": consecutive,
                "source_validation_passed": True,
            }
        )
        del endpoint
    parent_audit = {
        "level7_5_result_sha256": sha256_file(parent_75_path),
        "level7_4_1_result_sha256": sha256_file(parent_741_path),
        "level7_5_classification": parent_75["diagnosis"]["classification"],
        "level7_4_1_classification": parent_741["diagnosis"]["classification"],
    }
    return audits, parent_audit


def replay_snapshot_path(seed_root: Path, step: int) -> Path:
    return seed_root / "replay" / f"model_step{step:04d}.pt"


def registered_replay_steps(stop_step: int) -> list[int]:
    return [1, *range(100, stop_step + 1, 100)]


def save_replay_state(
    seed_root: Path,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
    step: int,
    consecutive: int,
    history: list[dict[str, Any]],
) -> None:
    replay_root = seed_root / "replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        replay_snapshot_path(seed_root, step),
        {
            "model": model.state_dict(),
            "model_seed": seed,
            "stage": 1,
            "chunks": 2,
            "replay_step": step,
            "validation": history[-1],
        },
    )
    atomic_torch_save(
        replay_root / "resume.pt",
        {
            "model": model.state_dict(),
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "model_seed": seed,
            "replay_step": step,
            "consecutive_behavior_passes": consecutive,
            "replay_history": history,
        },
    )


def run_exact_replay(
    source: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    seed_root: Path,
) -> dict[str, Any]:
    gate_path = seed_root / "replay_gate.json"
    if gate_path.exists() and not args.force:
        return read_json(gate_path)
    seed = int(source["seed"])
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    resume_path = seed_root / "replay" / "resume.pt"
    if resume_path.exists() and not args.force:
        state = restore(resume_path, model, probe, optimizer, device)
        last_step = int(state["replay_step"])
        consecutive = int(state["consecutive_behavior_passes"])
        history = state["replay_history"]
        print(f"seed={seed} resumed fixed-to-C2 replay step={last_step}", flush=True)
    else:
        restore(ROOT / source["start_checkpoint"], model, probe, optimizer, device)
        set_seed(seed + 20000)
        last_step = 0
        consecutive = 0
        history = []
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    eval_args = argparse.Namespace(
        eval_batches=args.training_eval_batches,
        eval_batch_size=args.training_eval_batch_size,
        chunk_size=args.chunk_size,
    )
    stop_step = last_step if consecutive >= 2 else None
    if consecutive < 2:
        for step in range(last_step + 1, args.maximum_steps + 1):
            model.train()
            probe.train()
            random_step(
                model,
                probe,
                optimizer,
                args,
                args.training_chunks,
                args.training_batch_size,
                args.probe_weight,
                device,
                dtype,
            )
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(
                    model, probe, eval_args, args.training_chunks, device, dtype
                )
                history.append({"stage": 1, "step": step, **metric})
                ok = metric["query"] >= args.training_gate
                consecutive = consecutive + 1 if ok else 0
                save_replay_state(
                    seed_root, model, probe, optimizer, seed, step, consecutive, history
                )
                atomic_save(
                    seed_root / "replay_progress.json",
                    {
                        "seed": seed,
                        "completed_step": step,
                        "consecutive_behavior_passes": consecutive,
                        "saved_milestones": [row["step"] for row in history],
                        "causal_panel_opened": False,
                    },
                )
                print(
                    f"seed={seed} replay C2 step={step} "
                    f"query={metric['query']:.2%} probe={metric['probe_min']:.2%} "
                    f"consecutive={consecutive}",
                    flush=True,
                )
                if consecutive >= 2:
                    stop_step = step
                    break
    if stop_step is None:
        stop_step = args.maximum_steps
    reference = torch.load(
        ROOT / source["reference_endpoint"], map_location="cpu", weights_only=False
    )
    current_model = model_fingerprint(model)
    current_probe = state_dict_fingerprint(probe.state_dict())
    current_optimizer = canonical_fingerprint(optimizer.state_dict())
    current_cpu_rng = torch.get_rng_state().cpu()
    current_cuda_rng = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    expected_steps = registered_replay_steps(source["reference_stop_step"])
    comparisons = {
        "model_state_exact": current_model == source["reference_model_fingerprint"],
        "probe_state_exact": current_probe == source["reference_probe_fingerprint"],
        "optimizer_state_exact": current_optimizer == source["reference_optimizer_fingerprint"],
        "CPU_RNG_exact": torch.equal(current_cpu_rng, reference["cpu_rng"].cpu()),
        "CUDA_RNG_exact": rng_equal(current_cuda_rng, reference["cuda_rng"]),
        "validation_history_exact": history == source["reference_validation_history"],
        "stop_step_exact": stop_step == source["reference_stop_step"],
        "consecutive_pass_state_exact": consecutive == 2,
        "all_registered_milestones_saved": all(
            replay_snapshot_path(seed_root, step).is_file() for step in expected_steps
        ),
    }
    gate = {
        "seed": seed,
        "passed": all(comparisons.values()),
        "comparisons": comparisons,
        "stop_step": stop_step,
        "reference_stop_step": source["reference_stop_step"],
        "consecutive_behavior_passes": consecutive,
        "replay_model_fingerprint": current_model,
        "reference_model_fingerprint": source["reference_model_fingerprint"],
        "replay_probe_fingerprint": current_probe,
        "reference_probe_fingerprint": source["reference_probe_fingerprint"],
        "replay_optimizer_fingerprint": current_optimizer,
        "reference_optimizer_fingerprint": source["reference_optimizer_fingerprint"],
        "replay_CPU_RNG_fingerprint": canonical_fingerprint(current_cpu_rng),
        "reference_CPU_RNG_fingerprint": source["reference_CPU_RNG_fingerprint"],
        "replay_CUDA_RNG_fingerprint": canonical_fingerprint(current_cuda_rng),
        "reference_CUDA_RNG_fingerprint": source["reference_CUDA_RNG_fingerprint"],
        "replay_validation_history": history,
        "reference_validation_history": source["reference_validation_history"],
        "causal_panel_authorized": all(comparisons.values()),
    }
    atomic_save(gate_path, gate)
    del reference, model, probe, optimizer
    torch.cuda.empty_cache()
    return gate


def milestone_sources(source: dict[str, Any], seed_root: Path) -> list[dict[str, Any]]:
    specs = [(0, ROOT / source["start_checkpoint"], "original_fixed_endpoint")]
    specs.extend(
        (step, replay_snapshot_path(seed_root, step), "exact_C2_replay")
        for step in registered_replay_steps(source["reference_stop_step"])
    )
    rows = []
    for step, path, origin in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "id": f"step_{step:04d}",
                "seed": source["seed"],
                "step": step,
                "origin": origin,
                "route_group": source["route_group"],
                "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256_file(path),
            }
        )
    return rows


def add_l3_profile(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metrics = result["metrics"]
    query = {name: metrics[name]["query"] for name in CONDITIONS}
    profile = result["profile"]
    minimum_local = profile["minimum_local"]
    strict_l3_core = bool(
        profile["whole_memory_causal"]
        and query["zero_l3"] <= args.disruption_threshold
        and query["batch_roll_l3"] <= args.disruption_threshold
        and query["keep_l3"] >= args.pair_sufficiency_threshold
        and query["zero_l1"] >= args.core_retention_threshold
        and query["zero_l2"] >= args.core_retention_threshold
        and query["keep_l1"] <= args.disruption_threshold
        and query["keep_l2"] <= args.disruption_threshold
    )
    preservation_floor = max(
        args.weak_l3_intact_threshold,
        query["intact"] - args.weak_l3_preservation_margin,
    )
    selectivity_gap = query["keep_l3"] - max(query["keep_l1"], query["keep_l2"])
    weak_precursor = bool(
        not strict_l3_core
        and query["intact"] >= args.weak_l3_intact_threshold
        and minimum_local >= args.local_threshold
        and max(query[name] for name in WHOLE_CONTROLS) <= args.disruption_threshold
        and query["zero_l3"] <= args.disruption_threshold
        and query["batch_roll_l3"] <= args.disruption_threshold
        and query["keep_l3"] >= args.weak_l3_retention_threshold
        and query["keep_l1"] <= args.disruption_threshold
        and query["keep_l2"] <= args.disruption_threshold
        and query["zero_l1"] >= preservation_floor
        and query["zero_l2"] >= preservation_floor
        and selectivity_gap >= args.weak_l3_selectivity_gap
    )
    profile["strict_l3_core"] = strict_l3_core
    profile["weak_l3_selective_precursor"] = weak_precursor
    profile["l3_route_selected"] = strict_l3_core or weak_precursor
    profile["l3_selectivity"] = {
        "preservation_floor": preservation_floor,
        "keep_l3_minus_best_other_single": selectivity_gap,
        "zero_l1_minus_zero_l3": query["zero_l1"] - query["zero_l3"],
        "zero_l2_minus_zero_l3": query["zero_l2"] - query["zero_l3"],
    }
    return result


def run_causal_milestone(
    milestone: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    seed_root: Path,
    base_model_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result = run_base_causal_milestone(
        milestone, args, device, dtype, seed_root, base_model_state
    )
    result["model_displacement_from_fixed"] = result[
        "model_displacement_from_C2"
    ]
    result = add_l3_profile(result, args)
    atomic_save(seed_root / "causal" / milestone["id"] / "result.json", result)
    return result


def diagnose_seed(
    source: dict[str, Any], gate: dict[str, Any], milestones: list[dict[str, Any]]
) -> dict[str, Any]:
    seed = int(source["seed"])
    if not gate["passed"]:
        return {
            "seed": seed,
            "route_group": source["route_group"],
            "classification": "replay_endpoint_mismatch_causal_panel_closed",
            "exact_replay_passed": False,
            "causal_panel_opened": False,
            "first_weak_l3_precursor_step": None,
            "first_strict_l3_core_step": None,
        }
    selected = [row["profile"]["l3_route_selected"] for row in milestones]
    precursor = [row["profile"]["weak_l3_selective_precursor"] for row in milestones]
    strict = [row["profile"]["strict_l3_core"] for row in milestones]
    first_selected = next(
        (row["step"] for row, flag in zip(milestones, selected) if flag), None
    )
    first_precursor = next(
        (row["step"] for row, flag in zip(milestones, precursor) if flag), None
    )
    first_strict = next(
        (row["step"] for row, flag in zip(milestones, strict) if flag), None
    )
    endpoint_selected = selected[-1]
    if selected[0]:
        classification = "L3_precursor_preexisting_at_fixed"
    elif any(selected) and endpoint_selected:
        classification = "L3_precursor_formed_during_C2"
    elif any(selected):
        classification = "L3_precursor_regressed_before_C2"
    else:
        classification = "L3_precursor_absent_through_C2"
    membership_changes = sum(
        left != right for left, right in zip(selected, selected[1:])
    )
    regression = any(left and not right for left, right in zip(selected, selected[1:]))
    transitions = []
    for previous, current in zip(milestones, milestones[1:]):
        old_value = previous["profile"]["l3_route_selected"]
        new_value = current["profile"]["l3_route_selected"]
        if old_value != new_value:
            transitions.append(
                {
                    "from_step": previous["step"],
                    "to_step": current["step"],
                    "from_selected": old_value,
                    "to_selected": new_value,
                }
            )
    endpoint = milestones[-1]
    return {
        "seed": seed,
        "route_group": source["route_group"],
        "classification": classification,
        "exact_replay_passed": True,
        "causal_panel_opened": True,
        "milestones": len(milestones),
        "first_l3_selected_step": first_selected,
        "first_weak_l3_precursor_step": first_precursor,
        "first_strict_l3_core_step": first_strict,
        "endpoint_l3_selected": endpoint_selected,
        "endpoint_intact_query": endpoint["metrics"]["intact"]["query"],
        "endpoint_route_class": endpoint["profile"]["route_class"],
        "membership_changes": membership_changes,
        "regression_observed": regression,
        "selection_transitions": transitions,
    }


def diagnose_cohort(seed_diagnoses: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed = {row["seed"]: row for row in seed_diagnoses}
    all_exact = all(row["exact_replay_passed"] for row in seed_diagnoses)
    default_outcomes = [by_seed[seed]["classification"] for seed in L3_DEFAULT_SEEDS]
    comparison = by_seed[COMPARISON_SEED]
    if not all_exact:
        classification = "exact_replay_incomplete_causal_interpretation_partial"
    elif (
        all(value == "L3_precursor_formed_during_C2" for value in default_outcomes)
        and comparison["classification"] == "L3_precursor_absent_through_C2"
    ):
        classification = "default_L3_precursor_divergence_confirmed"
    elif (
        all(by_seed[seed].get("endpoint_l3_selected") for seed in L3_DEFAULT_SEEDS)
        and comparison.get("endpoint_l3_selected")
    ):
        classification = "L3_precursor_not_route_specific"
    elif any(
        not by_seed[seed].get("endpoint_l3_selected", False)
        for seed in L3_DEFAULT_SEEDS
    ):
        classification = "L3_precursor_not_universal"
    else:
        classification = "route_bifurcation_unresolved"
    counts: dict[str, int] = {}
    for row in seed_diagnoses:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    return {
        "classification": classification,
        "all_exact_replay_gates_passed": all_exact,
        "qualified_seeds": sum(row["exact_replay_passed"] for row in seed_diagnoses),
        "L3_default_seeds_formed_precursor_during_C2": sum(
            by_seed[seed]["classification"] == "L3_precursor_formed_during_C2"
            for seed in L3_DEFAULT_SEEDS
        ),
        "seed1879_L3_precursor_observed": bool(
            comparison.get("first_l3_selected_step") is not None
        ),
        "per_seed_outcome_counts": counts,
        "registered_stop_boundary": (
            "Report the fixed four-trajectory result; do not insert milestones, "
            "relax precursor thresholds, replace a source, or continue training."
        ),
    }


def plot_trajectories(seed_runs: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(len(seed_runs), 2, figsize=(18, 5 * len(seed_runs)))
    for row_index, seed_run in enumerate(seed_runs):
        seed = seed_run["seed"]
        milestones = seed_run["milestones"]
        if not milestones:
            for axis in axes[row_index]:
                axis.text(0.5, 0.5, "exact replay gate failed", ha="center")
                axis.set_axis_off()
            continue
        steps = [row["step"] for row in milestones]
        left = axes[row_index, 0]
        for condition, label in [
            ("intact", "intact"),
            ("reset_all", "reset all"),
            ("zero_all", "zero all"),
            ("batch_roll_all", "roll all"),
        ]:
            left.plot(
                steps,
                [100 * row["metrics"][condition]["query"] for row in milestones],
                marker="o",
                label=label,
            )
        left.axhline(20, color="#b23a48", linestyle=":")
        left.axhline(90, color="#333333", linestyle="--")
        left.set_ylim(0, 105)
        left.set_title(f"seed {seed}: behavior and whole-Memory controls")
        left.set_ylabel("Query accuracy (%)")
        left.legend(ncol=2, fontsize=8)
        left.grid(alpha=0.2)
        right = axes[row_index, 1]
        for condition, label in [
            ("zero_l3", "zero L3"),
            ("batch_roll_l3", "roll L3"),
            ("keep_l1", "keep L1"),
            ("keep_l2", "keep L2"),
            ("keep_l3", "keep L3"),
        ]:
            right.plot(
                steps,
                [100 * row["metrics"][condition]["query"] for row in milestones],
                marker="o",
                label=label,
            )
        right.axhline(20, color="#b23a48", linestyle=":")
        right.axhline(90, color="#333333", linestyle="--")
        right.set_ylim(0, 105)
        right.set_title(f"seed {seed}: L3 selection trajectory")
        right.legend(ncol=2, fontsize=8)
        right.grid(alpha=0.2)
        first = seed_run["diagnosis"].get("first_l3_selected_step")
        if first is not None:
            for axis in axes[row_index]:
                axis.axvline(first, color="#2a9d8f", linestyle="--", linewidth=1.5)
        for axis in axes[row_index]:
            axis.set_xlabel("C2 replay step (0 = fixed-stage endpoint)")
    fig.suptitle("IST Level 7.5.1: Fixed-to-C2 Route-Bifurcation Replay")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_integrity(
    source_audit: list[dict[str, Any]],
    seed_runs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_seed = {row["seed"]: row for row in seed_runs}
    all_milestones = [
        milestone for row in seed_runs for milestone in row["milestones"]
    ]
    expected_qualified = sum(
        1 + len(registered_replay_steps(source["reference_stop_step"]))
        for source in source_audit
        if by_seed[source["seed"]]["replay_gate"]["passed"]
    )
    panels_closed_on_failure = all(
        row["replay_gate"]["passed"] or len(row["milestones"]) == 0
        for row in seed_runs
    )
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "formal_seeds_exact": [row["seed"] for row in seed_runs] == FORMAL_SEEDS,
        "all_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "causal_panels_closed_on_replay_failure": panels_closed_on_failure,
        "expected_qualified_milestones": expected_qualified,
        "completed_qualified_milestones": len(all_milestones),
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
        "no_training_beyond_C2": True,
        "no_source_replacement_or_extension": True,
        "seed909_used": False,
    }
    integrity["passed"] = bool(
        integrity["formal_seeds_exact"]
        and integrity["all_source_hashes_validated"]
        and integrity["causal_panels_closed_on_replay_failure"]
        and integrity["expected_qualified_milestones"]
        == integrity["completed_qualified_milestones"]
        and integrity["all_milestone_integrity_passed"]
        and integrity["all_conditions_exact"]
        and integrity["fixed_N_completed"]
        and integrity["no_training_beyond_C2"]
        and integrity["no_source_replacement_or_extension"]
        and not integrity["seed909_used"]
    )
    return integrity


def smoke_panel(
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], bool]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, probe = build(device, args.chunk_size)
    del probe
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = model_fingerprint(model)
    panel_args = argparse.Namespace(
        chunks=args.causal_chunks,
        chunk_size=args.chunk_size,
        samples=args.causal_samples,
        eval_batch_size=args.causal_eval_batch_size,
        dataset_seed=args.dataset_seed,
    )
    metrics = {
        condition: evaluate_condition(model, panel_args, condition, device, dtype)
        for condition in CONDITIONS
    }
    after = model_fingerprint(model)
    result = {
        "metrics": metrics,
        "profile": {
            "minimum_local": min(row["local"] for row in metrics.values()),
            "whole_memory_causal": False,
        },
    }
    add_l3_profile(result, args)
    del model
    torch.cuda.empty_cache()
    return result["profile"], before == after


def run_smoke(args: argparse.Namespace) -> int:
    protocol = read_json(STATIC_PREREGISTRATION)
    source_audit, _ = validate_sources(protocol)
    args.output = "experiments/level7_5_1/smoke"
    args.chunk_size = 32
    args.causal_chunks = 4
    args.causal_samples = 32
    args.causal_eval_batch_size = 8
    args.dataset_seed = 7519999
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    smoke_root = ROOT / args.output
    work_root = smoke_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    set_seed(75123)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    start_path = work_root / "mini_start.pt"
    atomic_torch_save(
        start_path,
        {
            "model": model.state_dict(),
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
    )
    del model, probe, optimizer
    torch.cuda.empty_cache()
    reference = run_mini_branch(start_path, args, device, dtype, False, work_root)
    replay = run_mini_branch(start_path, args, device, dtype, True, work_root)
    comparisons = {key: reference[key] == replay[key] for key in reference}
    profile, fingerprint_unchanged = smoke_panel(start_path, args, device, dtype)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "actual_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "miniature_exact_replay_comparisons": comparisons,
        "intermediate_save_preserved_exact_trajectory": all(comparisons.values()),
        "conditions_evaluated": len(CONDITIONS),
        "frozen_fingerprint_unchanged": fingerprint_unchanged,
        "L3_profile_fields_present": all(
            key in profile
            for key in ("strict_l3_core", "weak_l3_selective_precursor", "l3_route_selected")
        ),
    }
    result["passed"] = bool(
        result["actual_source_hashes_validated"]
        and result["intermediate_save_preserved_exact_trajectory"]
        and result["conditions_evaluated"] == 16
        and result["frozen_fingerprint_unchanged"]
        and result["L3_profile_fields_present"]
    )
    atomic_save(smoke_root / "result.json", result)
    if work_root.resolve().parent != smoke_root.resolve():
        raise RuntimeError("Refusing to clean an unexpected smoke directory")
    shutil.rmtree(work_root)
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
    progress: dict[str, Any] = {
        "stage": "exact_fixed_to_C2_replay",
        "formal_seeds": FORMAL_SEEDS,
        "completed_seeds": [],
        "active_seed": None,
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    seed_runs = []
    for source in source_audit:
        seed = int(source["seed"])
        seed_root = root / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        progress["active_seed"] = seed
        atomic_save(root / "progress.json", progress)
        gate = run_exact_replay(source, args, device, dtype, seed_root)
        milestones: list[dict[str, Any]] = []
        if gate["passed"]:
            start_state = torch.load(
                ROOT / source["start_checkpoint"], map_location="cpu", weights_only=False
            )
            base_model_state = start_state["model"]
            del start_state
            for milestone in milestone_sources(source, seed_root):
                progress["active_milestone"] = milestone["id"]
                atomic_save(root / "progress.json", progress)
                milestones.append(
                    run_causal_milestone(
                        milestone, args, device, dtype, seed_root, base_model_state
                    )
                )
            del base_model_state
        diagnosis = diagnose_seed(source, gate, milestones)
        seed_result = {
            "seed": seed,
            "source": source,
            "replay_gate": gate,
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
    integrity = build_integrity(source_audit, seed_runs, args)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "protocol": protocol,
        "source_audit": source_audit,
        "parent_audit": parent_audit,
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
                "route_group": row["source"]["route_group"],
                "replay_gate_passed": row["replay_gate"]["passed"],
                "diagnosis": row["diagnosis"],
                "trajectory": [
                    {
                        "step": milestone["step"],
                        "intact_query": milestone["metrics"]["intact"]["query"],
                        "route_class": milestone["profile"]["route_class"],
                        "weak_l3_selective_precursor": milestone["profile"][
                            "weak_l3_selective_precursor"
                        ],
                        "strict_l3_core": milestone["profile"]["strict_l3_core"],
                        "l3_route_selected": milestone["profile"]["l3_route_selected"],
                        "l3_selectivity": milestone["profile"]["l3_selectivity"],
                        "layer_atlas": milestone["profile"]["layer_atlas"],
                        "model_displacement_from_fixed": milestone[
                            "model_displacement_from_fixed"
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
    plot_trajectories(seed_runs, root / "fixed_to_C2_route_bifurcation.png")
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
