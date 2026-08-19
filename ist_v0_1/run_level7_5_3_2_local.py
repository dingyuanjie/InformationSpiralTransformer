"""Level 7.5.3.2: optimizer-state x data-stream causal bifurcation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from long_context_test import set_seed
from run_level6_2_local import evaluate
from run_level6_6_local import build, random_step, restore
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save
from run_level7_3_local import CONDITIONS, read_json, sha256_file
from run_level7_4_1_local import (
    atomic_torch_save,
    canonical_fingerprint,
    state_dict_fingerprint,
)
from run_level7_5_3_local import save_resume
from run_level7_5_3_1_local import SCREEN_CONDITIONS, run_panel


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_5_3_2"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
PARENT_RESULT = ROOT / "experiments/level7_5_3_1/formal/result.json"
PARENT_RESULT_SHA256 = "2f8c5d677c68a978c27e59a97bb343587a8a950f84bce62bd3a6f55e103d7e42"
PARENT_CLASSIFICATION = "continued_C4_destabilizes_preformed_routes"
SCREEN_STEPS = [0, 300, 600, 1000]
SCREEN_DATASET_SEED = 7532000
CONFIRM_DATASET_SEED = 7532001
RESET_DATA_SEED = 7532002
EXACT_ARM = "preserve_optimizer_preserve_rng"
INTERVENTION_ARMS = [
    "reset_optimizer_preserve_rng",
    "preserve_optimizer_reset_rng",
    "reset_optimizer_reset_rng",
]
ALL_ARMS = [EXACT_ARM, *INTERVENTION_ARMS]
ARM_FACTORS = {
    EXACT_ARM: {"preserve_optimizer": True, "preserve_rng": True},
    "reset_optimizer_preserve_rng": {
        "preserve_optimizer": False,
        "preserve_rng": True,
    },
    "preserve_optimizer_reset_rng": {
        "preserve_optimizer": True,
        "preserve_rng": False,
    },
    "reset_optimizer_reset_rng": {
        "preserve_optimizer": False,
        "preserve_rng": False,
    },
}


def milestone(step: int, checkpoint: str, sha256: str) -> dict[str, Any]:
    return {
        "recovery_step": step,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256,
    }


def source(
    seed: int,
    branch: str,
    selection_role: str,
    expected_route: str,
    baseline_route: str,
    parent_final_route: str,
    parent_final_query: float,
    checkpoint_sha256: str,
    checkpoint_size: int,
    exact_milestones: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"seed{seed}_{branch}",
        "seed": seed,
        "branch": branch,
        "selection_role": selection_role,
        "expected_route": expected_route,
        "baseline_route": baseline_route,
        "parent_final_route": parent_final_route,
        "parent_final_query": parent_final_query,
        "checkpoint": (
            f"experiments/level7_5_3/formal/seed{seed}/{branch}/C4_endpoint.pt"
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_size,
        "exact_reference_milestones": exact_milestones,
    }


SOURCE_SPECS = [
    source(
        1879,
        "intact_replay",
        "persistent_L2_loss",
        "l2_core_l3_supported",
        "l2_core_l3_supported",
        "unformed_behavior",
        0.7919921875,
        "4330acaf0cb3100bb2595cd1852fab24c9dc0c071b1f3007123bd1e0862196ed",
        4325455,
        [
            milestone(300, "experiments/level7_5_3_1/formal/seed1879/intact_replay/recovery/model_step0300.pt", "685d889eaa55887fd8c3dda2a90cbafe4122ba45cbed153d29fbf0999d0c1529"),
            milestone(600, "experiments/level7_5_3_1/formal/seed1879/intact_replay/recovery/model_step0600.pt", "cea26b58797c7504501f9760c6c88a43978acc501985d41c8da22380d14ecf0a"),
            milestone(1000, "experiments/level7_5_3_1/formal/seed1879/intact_replay/recovery/model_step1000.pt", "c62ea19e74530f297851987dc4f179a1f364f3a014030fc1a77961b3ff35e73d"),
        ],
    ),
    source(
        2203,
        "selected_layer_suppression",
        "unformed_L3_recovery",
        "l3_core",
        "unformed_behavior",
        "l3_core",
        0.96630859375,
        "0da36751fac997921584685d5a0c0b4ea08697308de7bdc427c00fd9fb0e812d",
        4321743,
        [
            milestone(300, "experiments/level7_5_3_1/formal/seed2203/selected_layer_suppression/recovery/model_step0300.pt", "2a5aaec9900175914a82b00ee3933dd3ede1345992c56d2c66a6edac724b4a2a"),
            milestone(600, "experiments/level7_5_3_1/formal/seed2203/selected_layer_suppression/recovery/model_step0600.pt", "7534737a98973d9cc1042a9eb10dd619fd568df105a85b9474ef239a675d9a21"),
            milestone(1000, "experiments/level7_5_3_1/formal/seed2203/selected_layer_suppression/recovery/model_step1000.pt", "7240f64766198a81e95ec1b61ce24c0efe72b6169faf1d4aa141bd86e4cf6741"),
        ],
    ),
    source(
        2551,
        "selected_layer_suppression",
        "late_L3_collapse",
        "l3_core",
        "l3_core",
        "unformed_behavior",
        0.24365234375,
        "14c776f78821f968b0caf909312b863352b8fd9d887d970426245f3cf41560a2",
        4320911,
        [
            milestone(300, "experiments/level7_5_3_1/formal/seed2551/selected_layer_suppression/recovery/model_step0300.pt", "93541943f299134dcd8f9e27af978458066b33f9bd5739f9a790644973ecc04c"),
            milestone(600, "experiments/level7_5_3_1/formal/seed2551/selected_layer_suppression/recovery/model_step0600.pt", "b39d6b34aa79f61ae3a40a683d972a8677570638c52f6a94c8c1f4f6346f2e4b"),
            milestone(1000, "experiments/level7_5_3_1/formal/seed2551/selected_layer_suppression/recovery/model_step1000.pt", "2fd04c46007b8939ef1e3b87dc91d8c4184644e78bcaea599d1981a3fb2ead18"),
        ],
    ),
    source(
        2909,
        "intact_replay",
        "transient_L3_collapse_recovery",
        "l3_core",
        "l3_core",
        "l3_core",
        0.94091796875,
        "9862476ac927196b36da89ce25500b4a70de4347b80bf7bedc23b3b0f2358fe8",
        4321167,
        [
            milestone(300, "experiments/level7_5_3_1/formal/seed2909/intact_replay/recovery/model_step0300.pt", "97a08c2044a93ddfbbd323117c4d212ef7d630135bb3aefa67cedbd03bcfe8a0"),
            milestone(600, "experiments/level7_5_3_1/formal/seed2909/intact_replay/recovery/model_step0600.pt", "e92edfc4eec7fc424d39abdcd873bb2d93355fe610361e9ca263de96c6895886"),
            milestone(1000, "experiments/level7_5_3_1/formal/seed2909/intact_replay/recovery/model_step1000.pt", "cc6d570ee364eea938993f2db3108cec4daa9135dfc93bfa3cdb79a803a9334f"),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--training-steps", type=int, default=1000)
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
    parser.add_argument("--reset-data-seed", type=int, default=RESET_DATA_SEED)
    parser.add_argument("--formed-threshold", type=float, default=0.90)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--material-query-delta", type=float, default=0.15)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--core-retention-threshold", type=float, default=0.80)
    parser.add_argument("--pair-sufficiency-threshold", type=float, default=0.90)
    parser.add_argument("--pair-gain-threshold", type=float, default=0.03)
    parser.add_argument("--roll-drop-threshold", type=float, default=0.05)
    parser.add_argument("--precursor-intact-threshold", type=float, default=0.75)
    parser.add_argument("--precursor-retention-threshold", type=float, default=0.70)
    parser.add_argument("--output", default="experiments/level7_5_3_2/formal")
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
        "training_steps": 1000,
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
        "reset_data_seed": RESET_DATA_SEED,
        "formed_threshold": 0.90,
        "local_threshold": 0.90,
        "material_query_delta": 0.15,
        "disruption_threshold": 0.20,
        "core_retention_threshold": 0.80,
        "pair_sufficiency_threshold": 0.90,
        "pair_gain_threshold": 0.03,
        "roll_drop_threshold": 0.05,
        "precursor_intact_threshold": 0.75,
        "precursor_retention_threshold": 0.70,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.5.3.2 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_5_3_2/formal":
        raise ValueError("Formal Level 7.5.3.2 output path is locked")


def validate_static_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("frozen_sources") != SOURCE_SPECS:
        raise RuntimeError("Static Level 7.5.3.2 sources changed")
    if protocol.get("arms") != [
        {"name": name, **ARM_FACTORS[name]} for name in ALL_ARMS
    ]:
        raise RuntimeError("Static Level 7.5.3.2 arms changed")
    if protocol.get("trajectory_screen_panel", {}).get("milestones") != SCREEN_STEPS:
        raise RuntimeError("Static trajectory milestones changed")
    if protocol.get("trajectory_screen_panel", {}).get("conditions") != SCREEN_CONDITIONS:
        raise RuntimeError("Static screen conditions changed")
    if protocol.get("final_confirmation_panel", {}).get("conditions") != CONDITIONS:
        raise RuntimeError("Static confirmation conditions changed")
    if protocol.get("data_stream_reset", {}).get("seed") != RESET_DATA_SEED:
        raise RuntimeError("Static data-stream reset seed changed")


def expected_layer(spec: dict[str, Any]) -> int:
    return 2 if spec["expected_route"].startswith("l2_") else 3


def expected_route_formed(spec: dict[str, Any], route: str) -> bool:
    if expected_layer(spec) == 2:
        return route in {"l2_core_l3_supported", "l2_core_minimal_l3_support"}
    return route == "l3_core"


def validate_sources(
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise RuntimeError("Frozen Level 7.5.3.1 parent result changed")
    parent = read_json(PARENT_RESULT)
    if not parent["integrity"]["passed"]:
        raise RuntimeError("Level 7.5.3.1 parent integrity did not pass")
    if parent["diagnosis"]["classification"] != PARENT_CLASSIFICATION:
        raise RuntimeError("Level 7.5.3.1 parent classification changed")
    parent_training = {row["id"]: row for row in parent["recovery_training"]}
    audit = []
    for spec in SOURCE_SPECS:
        source_path = ROOT / spec["checkpoint"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if sha256_file(source_path) != spec["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen source changed: {spec['id']}")
        if source_path.stat().st_size != spec["checkpoint_size_bytes"]:
            raise RuntimeError(f"Frozen source size changed: {spec['id']}")
        source_state = torch.load(source_path, map_location="cpu", weights_only=False)
        required = {"model", "probe", "optimizer", "cpu_rng", "cuda_rng"}
        if not required.issubset(source_state):
            raise RuntimeError(f"Frozen source is incomplete: {spec['id']}")
        parent_rows = {
            int(row["recovery_step"]): row
            for row in parent_training[spec["id"]]["milestones"]
        }
        exact_audit = []
        for row in spec["exact_reference_milestones"]:
            step = int(row["recovery_step"])
            path = ROOT / row["checkpoint"]
            if not path.is_file() or sha256_file(path) != row["checkpoint_sha256"]:
                raise RuntimeError(
                    f"Exact reference changed: {spec['id']} step={step}"
                )
            if parent_rows[step] != row:
                raise RuntimeError(
                    f"Parent reference mismatch: {spec['id']} step={step}"
                )
            exact_audit.append({**row, "checkpoint_size_bytes": path.stat().st_size})
        audit.append(
            {
                "id": spec["id"],
                "checkpoint": spec["checkpoint"],
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "model_fingerprint": state_dict_fingerprint(source_state["model"]),
                "probe_fingerprint": state_dict_fingerprint(source_state["probe"]),
                "optimizer_fingerprint": canonical_fingerprint(source_state["optimizer"]),
                "CPU_RNG_fingerprint": canonical_fingerprint(source_state["cpu_rng"]),
                "CUDA_RNG_fingerprint": canonical_fingerprint(source_state["cuda_rng"]),
                "exact_reference_milestones": exact_audit,
                "passed": True,
            }
        )
        del source_state
    return audit, {
        "result": str(PARENT_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": PARENT_RESULT_SHA256,
        "classification": parent["diagnosis"]["classification"],
        "integrity_passed": parent["integrity"]["passed"],
    }


def initialize_arm(
    spec: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    factors = ARM_FACTORS[arm]
    source_state = torch.load(
        ROOT / spec["checkpoint"], map_location=device, weights_only=False
    )
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    model.load_state_dict(source_state["model"])
    probe.load_state_dict(source_state["probe"])
    if factors["preserve_optimizer"]:
        optimizer.load_state_dict(source_state["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    if factors["preserve_rng"]:
        torch.set_rng_state(source_state["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([item.cpu() for item in source_state["cuda_rng"]])
    else:
        set_seed(args.reset_data_seed)
    model_exact = state_dict_fingerprint(model.state_dict()) == state_dict_fingerprint(
        source_state["model"]
    )
    probe_exact = state_dict_fingerprint(probe.state_dict()) == state_dict_fingerprint(
        source_state["probe"]
    )
    optimizer_exact = canonical_fingerprint(
        optimizer.state_dict()
    ) == canonical_fingerprint(source_state["optimizer"])
    optimizer_entries = len(optimizer.state_dict()["state"])
    cpu_exact = canonical_fingerprint(torch.get_rng_state().cpu()) == canonical_fingerprint(
        source_state["cpu_rng"]
    )
    cuda_exact = canonical_fingerprint(
        [item.cpu() for item in torch.cuda.get_rng_state_all()]
    ) == canonical_fingerprint(source_state["cuda_rng"])
    optimizer_gate = optimizer_exact if factors["preserve_optimizer"] else (
        optimizer_entries == 0 and not optimizer_exact
    )
    rng_gate = (cpu_exact and cuda_exact) if factors["preserve_rng"] else (
        not cpu_exact and not cuda_exact
    )
    audit = {
        "arm": arm,
        "model_source_exact": model_exact,
        "probe_source_exact": probe_exact,
        "optimizer_source_exact": optimizer_exact,
        "optimizer_state_entries": optimizer_entries,
        "optimizer_reset": not factors["preserve_optimizer"],
        "CPU_RNG_source_exact": cpu_exact,
        "CUDA_RNG_source_exact": cuda_exact,
        "RNG_reset": not factors["preserve_rng"],
        "reset_data_seed": None if factors["preserve_rng"] else args.reset_data_seed,
        "initial_CPU_RNG_fingerprint": canonical_fingerprint(torch.get_rng_state().cpu()),
        "initial_CUDA_RNG_fingerprint": canonical_fingerprint(
            [item.cpu() for item in torch.cuda.get_rng_state_all()]
        ),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "passed": bool(model_exact and probe_exact and optimizer_gate and rng_gate),
    }
    del source_state
    return model, probe, optimizer, audit


def snapshot_path(folder: Path, step: int) -> Path:
    return folder / "training" / f"model_step{step:04d}.pt"


def save_snapshot(
    folder: Path,
    spec: dict[str, Any],
    arm: str,
    step: int,
    model: torch.nn.Module,
    validation: dict[str, Any],
) -> None:
    atomic_torch_save(
        snapshot_path(folder, step),
        {
            "model": model.state_dict(),
            "seed": spec["seed"],
            "source_branch": spec["branch"],
            "arm": arm,
            "training_step": step,
            "validation": validation,
            "memory_mask_active": False,
        },
    )


def run_training_arm(
    spec: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    if arm not in INTERVENTION_ARMS:
        raise ValueError(f"No new training is permitted for arm={arm}")
    folder = root / f"seed{spec['seed']}" / spec["branch"] / arm
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "training_result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    resume_path = folder / "training" / "resume.pt"
    if resume_path.exists() and not args.force:
        state = restore(resume_path, model, probe, optimizer, device)
        if state["arm"] != arm or state["source_id"] != spec["id"]:
            raise RuntimeError(f"Resume metadata mismatch: {spec['id']} {arm}")
        last_step = int(state["training_step"])
        history = state["training_history"]
        initialization_audit = state["initialization_audit"]
        print(
            f"seed={spec['seed']} source={spec['branch']} arm={arm} "
            f"resumed step={last_step}",
            flush=True,
        )
    else:
        del model, probe, optimizer
        model, probe, optimizer, initialization_audit = initialize_arm(
            spec, arm, args, device
        )
        if not initialization_audit["passed"]:
            raise RuntimeError(f"Initialization gate failed: {spec['id']} {arm}")
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
    for step in range(last_step + 1, args.training_steps + 1):
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
            history.append({"training_step": step, **metric})
            if step in SCREEN_STEPS:
                save_snapshot(folder, spec, arm, step, model, metric)
            save_resume(
                resume_path,
                model,
                probe,
                optimizer,
                {
                    "source_id": spec["id"],
                    "arm": arm,
                    "training_step": step,
                    "training_history": history,
                    "initialization_audit": initialization_audit,
                    "memory_mask_active_steps": 0,
                },
            )
            print(
                f"seed={spec['seed']} source={spec['branch']} arm={arm} "
                f"step={step} query={metric['query']:.2%} "
                f"probe={metric['probe_min']:.2%}",
                flush=True,
            )
    milestones = []
    for step in SCREEN_STEPS[1:]:
        path = snapshot_path(folder, step)
        if not path.is_file():
            raise FileNotFoundError(path)
        milestones.append(
            {
                "recovery_step": step,
                "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256_file(path),
            }
        )
    result = {
        "id": f"{spec['id']}__{arm}",
        "source_id": spec["id"],
        "seed": spec["seed"],
        "source_branch": spec["branch"],
        "arm": arm,
        "factors": ARM_FACTORS[arm],
        "initialization_audit": initialization_audit,
        "training_steps_completed": args.training_steps,
        "memory_mask_active_steps": 0,
        "history": history,
        "milestones": milestones,
        "training_complete": True,
        "training_beyond_registered_budget": False,
    }
    atomic_save(result_path, result)
    del model, probe, optimizer
    torch.cuda.empty_cache()
    return result


def panel_spec(spec: dict[str, Any], arm: str) -> dict[str, Any]:
    branch = f"{spec['branch']}/{arm}"
    return {
        "id": f"{spec['id']}__{arm}",
        "seed": spec["seed"],
        "branch": branch,
        "baseline_route": spec["baseline_route"],
        "expected_route": spec["expected_route"],
    }


def source_panel_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return panel_spec(spec, "shared_source")


def source_milestone(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "recovery_step": 0,
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
    }


def exact_milestones(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return spec["exact_reference_milestones"]


def mark_progress(progress: dict[str, Any], key: str, value: str) -> None:
    if value not in progress[key]:
        progress[key].append(value)


def screen_formed(
    spec: dict[str, Any], row: dict[str, Any], args: argparse.Namespace
) -> bool:
    profile = row["screen_profile"]
    return bool(
        profile["intact_query"] >= args.formed_threshold
        and profile["minimum_local"] >= args.local_threshold
        and profile["dominant_retention_layer"] == expected_layer(spec)
    )


def diagnose(
    source_screens: list[dict[str, Any]],
    trajectory_screens: list[dict[str, Any]],
    confirmations: list[dict[str, Any]],
    args: argparse.Namespace,
    integrity_passed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_by_id = {
        row["id"].removesuffix("__shared_source"): row for row in source_screens
    }
    trajectory_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in trajectory_screens:
        trajectory_by_id.setdefault(row["id"], []).append(row)
    confirmation_by_id = {row["id"]: row for row in confirmations}
    outcomes = []
    for spec in SOURCE_SPECS:
        source_row = source_by_id[spec["id"]]
        for arm in ALL_ARMS:
            ident = f"{spec['id']}__{arm}"
            later = sorted(
                trajectory_by_id[ident], key=lambda row: row["recovery_step"]
            )
            trajectory = [source_row, *later]
            points = []
            for row in trajectory:
                profile = row["screen_profile"]
                points.append(
                    {
                        "training_step": row["recovery_step"],
                        "intact_query": profile["intact_query"],
                        "minimum_local": profile["minimum_local"],
                        "dominant_retention_layer": profile[
                            "dominant_retention_layer"
                        ],
                        "formed": screen_formed(spec, row, args),
                    }
                )
            confirmation = confirmation_by_id[ident]
            final_route = confirmation["profile"]["route_class"]
            final_formed = expected_route_formed(spec, final_route)
            post_source = points[1:]
            outcomes.append(
                {
                    "id": ident,
                    "source_id": spec["id"],
                    "seed": spec["seed"],
                    "source_branch": spec["branch"],
                    "selection_role": spec["selection_role"],
                    "arm": arm,
                    "factors": ARM_FACTORS[arm],
                    "trajectory": points,
                    "formed_vector_300_600_1000": [
                        row["formed"] for row in post_source
                    ],
                    "trajectory_query_mean": statistics.fmean(
                        row["intact_query"] for row in post_source
                    ),
                    "trajectory_query_min": min(
                        row["intact_query"] for row in post_source
                    ),
                    "final_route": final_route,
                    "final_expected_route_formed": final_formed,
                    "final_intact_query": confirmation["metrics"]["intact"]["query"],
                    "route_stability_score": sum(
                        row["formed"] for row in post_source
                    )
                    + int(final_formed),
                }
            )
    by_source_arm = {
        (row["source_id"], row["arm"]): row for row in outcomes
    }
    comparisons = []
    for spec in SOURCE_SPECS:
        reference = by_source_arm[(spec["id"], EXACT_ARM)]
        reference_queries = {
            row["training_step"]: row["intact_query"]
            for row in reference["trajectory"]
        }
        for arm in INTERVENTION_ARMS:
            row = by_source_arm[(spec["id"], arm)]
            max_query_delta = max(
                abs(point["intact_query"] - reference_queries[point["training_step"]])
                for point in row["trajectory"][1:]
            )
            vector_changed = (
                row["formed_vector_300_600_1000"]
                != reference["formed_vector_300_600_1000"]
            )
            final_fate_changed = (
                row["final_expected_route_formed"]
                != reference["final_expected_route_formed"]
            )
            score_delta = (
                row["route_stability_score"] - reference["route_stability_score"]
            )
            comparisons.append(
                {
                    "source_id": spec["id"],
                    "selection_role": spec["selection_role"],
                    "arm": arm,
                    "formed_vector_changed": vector_changed,
                    "final_fate_changed": final_fate_changed,
                    "max_absolute_query_delta": max_query_delta,
                    "route_stability_score_delta": score_delta,
                    "material_effect": bool(
                        vector_changed
                        or final_fate_changed
                        or max_query_delta >= args.material_query_delta
                    ),
                    "stabilized": score_delta >= 2,
                    "destabilized": score_delta <= -2,
                }
            )
    arm_summaries = {}
    for arm in INTERVENTION_ARMS:
        rows = [row for row in comparisons if row["arm"] == arm]
        arm_summaries[arm] = {
            "material_effect_sources": sum(row["material_effect"] for row in rows),
            "final_fate_changed_sources": sum(
                row["final_fate_changed"] for row in rows
            ),
            "stabilized_sources": sum(row["stabilized"] for row in rows),
            "destabilized_sources": sum(row["destabilized"] for row in rows),
        }
    optimizer_count = arm_summaries["reset_optimizer_preserve_rng"][
        "material_effect_sources"
    ]
    rng_count = arm_summaries["preserve_optimizer_reset_rng"][
        "material_effect_sources"
    ]
    joint_count = arm_summaries["reset_optimizer_reset_rng"][
        "material_effect_sources"
    ]
    if not integrity_passed:
        classification = "formal_integrity_failed_causal_interpretation_closed"
    elif optimizer_count >= 3 and rng_count <= 1:
        classification = "optimizer_state_primary_driver"
    elif rng_count >= 3 and optimizer_count <= 1:
        classification = "data_stream_primary_driver"
    elif optimizer_count <= 1 and rng_count <= 1 and joint_count >= 3:
        classification = "optimizer_rng_interaction_primary"
    elif optimizer_count >= 2 and rng_count >= 2:
        classification = "optimizer_and_data_stream_both_causal"
    elif optimizer_count == 0 and rng_count == 0 and joint_count == 0:
        classification = "endpoint_weight_basin_dominant"
    else:
        classification = "heterogeneous_causal_control"
    diagnosis = {
        "classification": classification,
        "outcome_stratified_sources": len(SOURCE_SPECS),
        "arm_summaries": arm_summaries,
        "interpretation_scope": (
            "Mechanism diagnosis for four frozen outcome-stratified endpoints; "
            "not a prevalence estimate over all Level 7.5.3.1 branches."
        ),
        "registered_stop_boundary": (
            "Report the fixed 2x2 bifurcation; do not add seeds, endpoints, "
            "training steps, reset seeds, or post-hoc arms."
        ),
    }
    return outcomes, {"comparisons": comparisons, "diagnosis": diagnosis}


def build_integrity(
    source_audit: list[dict[str, Any]],
    training: list[dict[str, Any]],
    source_screens: list[dict[str, Any]],
    trajectory_screens: list[dict[str, Any]],
    confirmations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reset_rng_pairs = {
        (
            row["initialization_audit"]["initial_CPU_RNG_fingerprint"],
            row["initialization_audit"]["initial_CUDA_RNG_fingerprint"],
        )
        for row in training
        if row["initialization_audit"]["RNG_reset"]
    }
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "parent_result_sha256": sha256_file(PARENT_RESULT),
        "all_sources_and_exact_references_validated": all(
            row["passed"] for row in source_audit
        ),
        "expected_new_training_arms": len(SOURCE_SPECS) * len(INTERVENTION_ARMS),
        "completed_new_training_arms": len(training),
        "all_training_steps_exact": all(
            row["training_steps_completed"] == args.training_steps for row in training
        ),
        "all_initialization_gates_passed": all(
            row["initialization_audit"]["passed"] for row in training
        ),
        "all_optimizer_resets_empty": all(
            row["initialization_audit"]["optimizer_state_entries"] == 0
            for row in training
            if row["initialization_audit"]["optimizer_reset"]
        ),
        "all_optimizer_preservations_exact": all(
            row["initialization_audit"]["optimizer_source_exact"]
            for row in training
            if not row["initialization_audit"]["optimizer_reset"]
        ),
        "all_RNG_preservations_exact": all(
            row["initialization_audit"]["CPU_RNG_source_exact"]
            and row["initialization_audit"]["CUDA_RNG_source_exact"]
            for row in training
            if not row["initialization_audit"]["RNG_reset"]
        ),
        "all_RNG_resets_share_one_locked_state": len(reset_rng_pairs) == 1,
        "no_memory_masks": all(row["memory_mask_active_steps"] == 0 for row in training),
        "no_training_beyond_budget": all(
            not row["training_beyond_registered_budget"] for row in training
        ),
        "expected_shared_source_screens": len(SOURCE_SPECS),
        "completed_shared_source_screens": len(source_screens),
        "expected_trajectory_screens": len(SOURCE_SPECS) * len(ALL_ARMS) * 3,
        "completed_trajectory_screens": len(trajectory_screens),
        "all_screen_integrity_passed": all(
            row["integrity"]["passed"]
            for row in [*source_screens, *trajectory_screens]
        ),
        "screen_dataset_seed": args.screen_dataset_seed,
        "expected_confirmation_panels": len(SOURCE_SPECS) * len(ALL_ARMS),
        "completed_confirmation_panels": len(confirmations),
        "all_confirmation_integrity_passed": all(
            row["integrity"]["passed"] for row in confirmations
        ),
        "confirmation_dataset_seed": args.confirm_dataset_seed,
        "seed909_used": False,
    }
    integrity["passed"] = bool(
        integrity["all_sources_and_exact_references_validated"]
        and integrity["completed_new_training_arms"]
        == integrity["expected_new_training_arms"]
        and integrity["all_training_steps_exact"]
        and integrity["all_initialization_gates_passed"]
        and integrity["all_optimizer_resets_empty"]
        and integrity["all_optimizer_preservations_exact"]
        and integrity["all_RNG_preservations_exact"]
        and integrity["all_RNG_resets_share_one_locked_state"]
        and integrity["no_memory_masks"]
        and integrity["no_training_beyond_budget"]
        and integrity["completed_shared_source_screens"]
        == integrity["expected_shared_source_screens"]
        and integrity["completed_trajectory_screens"]
        == integrity["expected_trajectory_screens"]
        and integrity["all_screen_integrity_passed"]
        and integrity["completed_confirmation_panels"]
        == integrity["expected_confirmation_panels"]
        and integrity["all_confirmation_integrity_passed"]
        and not integrity["seed909_used"]
    )
    return integrity


def plot_bifurcations(outcomes: list[dict[str, Any]], path: Path) -> None:
    colors = {
        EXACT_ARM: "#333333",
        "reset_optimizer_preserve_rng": "#d1495b",
        "preserve_optimizer_reset_rng": "#0077b6",
        "reset_optimizer_reset_rng": "#2a9d8f",
    }
    labels = {
        EXACT_ARM: "exact reference",
        "reset_optimizer_preserve_rng": "reset optimizer",
        "preserve_optimizer_reset_rng": "reset data stream",
        "reset_optimizer_reset_rng": "reset both",
    }
    by_source_arm = {
        (row["source_id"], row["arm"]): row for row in outcomes
    }
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    for axis, spec in zip(axes.flat, SOURCE_SPECS):
        for arm in ALL_ARMS:
            row = by_source_arm[(spec["id"], arm)]
            axis.plot(
                [point["training_step"] for point in row["trajectory"]],
                [100.0 * point["intact_query"] for point in row["trajectory"]],
                marker="o",
                linewidth=2,
                color=colors[arm],
                label=labels[arm],
            )
        axis.axhline(90.0, color="#777777", linestyle="--", linewidth=1.5)
        axis.set_title(f"seed{spec['seed']} · {spec['selection_role']}")
        axis.set_xlabel("Additional C4 steps")
        axis.set_ylabel("Fresh 16-chunk query accuracy (%)")
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 105)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=4)
    figure.suptitle("Level 7.5.3.2 optimizer-state × data-stream bifurcation", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_smoke(args: argparse.Namespace) -> int:
    protocol = read_json(STATIC_PREREGISTRATION)
    validate_static_protocol(protocol)
    source_audit, parent_audit = validate_sources(protocol)
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    spec = SOURCE_SPECS[0]
    rows = []
    for arm in INTERVENTION_ARMS:
        model, probe, optimizer, audit = initialize_arm(spec, arm, args, device)
        random_step(model, probe, optimizer, args, 4, 2, args.probe_weight, device, dtype)
        rows.append(
            {
                "arm": arm,
                "initialization_audit": audit,
                "post_step_model_fingerprint": state_dict_fingerprint(
                    model.state_dict()
                ),
            }
        )
        del model, probe, optimizer
        torch.cuda.empty_cache()
    reset_rows = [
        row for row in rows if row["initialization_audit"]["RNG_reset"]
    ]
    reset_rng_equal = len(
        {
            (
                row["initialization_audit"]["initial_CPU_RNG_fingerprint"],
                row["initialization_audit"]["initial_CUDA_RNG_fingerprint"],
            )
            for row in reset_rows
        }
    ) == 1
    result = {
        "smoke_test": True,
        "source_audit": source_audit,
        "parent_audit": parent_audit,
        "arms": rows,
        "all_initialization_gates_passed": all(
            row["initialization_audit"]["passed"] for row in rows
        ),
        "reset_rng_equal": reset_rng_equal,
    }
    result["passed"] = bool(
        result["all_initialization_gates_passed"] and reset_rng_equal
    )
    path = ROOT / "experiments/level7_5_3_2/smoke/result.json"
    atomic_save(path, result)
    print("LEVEL7_5_3_2_SMOKE_PASS" if result["passed"] else "LEVEL7_5_3_2_SMOKE_FAIL")
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
    progress_path = root / "progress.json"
    if progress_path.exists() and not args.force:
        progress = read_json(progress_path)
    else:
        progress = {
            "stage": "causal_training",
            "completed_training_arms": [],
            "completed_source_screens": [],
            "completed_trajectory_screens": [],
            "completed_confirmation_panels": [],
            "active": None,
            "seed909_locked": True,
        }
    atomic_save(progress_path, progress)
    training = []
    for spec in SOURCE_SPECS:
        for arm in INTERVENTION_ARMS:
            ident = f"{spec['id']}__{arm}"
            progress["active"] = ident
            progress["stage"] = "causal_training"
            atomic_save(progress_path, progress)
            row = run_training_arm(spec, arm, args, device, dtype, root)
            training.append(row)
            mark_progress(progress, "completed_training_arms", ident)
            atomic_save(progress_path, progress)
    source_screens = []
    for spec in SOURCE_SPECS:
        ident = f"{spec['id']}__shared_source"
        progress["active"] = ident
        progress["stage"] = "shared_source_screens"
        atomic_save(progress_path, progress)
        row = run_panel(
            source_panel_spec(spec),
            source_milestone(spec),
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
        source_screens.append(row)
        mark_progress(progress, "completed_source_screens", ident)
        atomic_save(progress_path, progress)
    training_by_key = {(row["source_id"], row["arm"]): row for row in training}
    trajectory_screens = []
    for spec in SOURCE_SPECS:
        for arm in ALL_ARMS:
            milestones = (
                exact_milestones(spec)
                if arm == EXACT_ARM
                else training_by_key[(spec["id"], arm)]["milestones"]
            )
            for item in milestones:
                ident = f"{spec['id']}__{arm}/step{item['recovery_step']}"
                progress["active"] = ident
                progress["stage"] = "trajectory_screens"
                atomic_save(progress_path, progress)
                row = run_panel(
                    panel_spec(spec, arm),
                    item,
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
                trajectory_screens.append(row)
                mark_progress(progress, "completed_trajectory_screens", ident)
                atomic_save(progress_path, progress)
    confirmations = []
    for spec in SOURCE_SPECS:
        for arm in ALL_ARMS:
            milestones = (
                exact_milestones(spec)
                if arm == EXACT_ARM
                else training_by_key[(spec["id"], arm)]["milestones"]
            )
            item = milestones[-1]
            ident = f"{spec['id']}__{arm}"
            progress["active"] = ident
            progress["stage"] = "final_confirmation_panels"
            atomic_save(progress_path, progress)
            row = run_panel(
                panel_spec(spec, arm),
                item,
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
            confirmations.append(row)
            mark_progress(progress, "completed_confirmation_panels", ident)
            atomic_save(progress_path, progress)
    integrity = build_integrity(
        source_audit,
        training,
        source_screens,
        trajectory_screens,
        confirmations,
        args,
    )
    outcomes, causal = diagnose(
        source_screens,
        trajectory_screens,
        confirmations,
        args,
        integrity["passed"],
    )
    elapsed = time.perf_counter() - started
    diagnosis = causal["diagnosis"]
    result = {
        "protocol": protocol,
        "parent_audit": parent_audit,
        "source_audit": source_audit,
        "training_arms": training,
        "shared_source_screens": source_screens,
        "trajectory_screens": trajectory_screens,
        "confirmation_panels": confirmations,
        "endpoint_outcomes": outcomes,
        "causal_comparisons": causal["comparisons"],
        "diagnosis": diagnosis,
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed,
    }
    summary = {
        "diagnosis": diagnosis,
        "integrity": integrity,
        "arm_summaries": diagnosis["arm_summaries"],
        "endpoint_outcomes": outcomes,
        "causal_comparisons": causal["comparisons"],
        "elapsed_seconds_this_invocation": elapsed,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_bifurcations(outcomes, root / "optimizer_rng_bifurcation.png")
    atomic_save(
        progress_path,
        {
            "stage": "complete",
            "completed_training_arms": len(training),
            "completed_source_screens": len(source_screens),
            "completed_trajectory_screens": len(trajectory_screens),
            "completed_confirmation_panels": len(confirmations),
            "classification": diagnosis["classification"],
            "integrity_passed": integrity["passed"],
            "seed909_locked": True,
        },
    )
    print(json.dumps(diagnosis, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
