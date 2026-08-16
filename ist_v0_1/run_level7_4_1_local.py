"""Level 7.4.1 deterministic dense replay of seed1879 C2-to-C4 formation."""

from __future__ import annotations

import argparse
import hashlib
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

from long_context_test import set_seed
from run_level6_2_local import evaluate
from run_level6_6_local import build, random_step, restore
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save, model_fingerprint
from run_level7_3_local import evaluate_condition, read_json, sha256_file
from run_level7_4_local import classify_checkpoint


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_4_1"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
START_CHECKPOINT = Path(
    "experiments/level7_2/formal/seed1879/curriculum_stage1.pt"
)
REFERENCE_CHECKPOINT = Path(
    "experiments/level7_2/formal/seed1879/curriculum_stage2.pt"
)
START_SHA256 = "9939755860050c602798b6cec0320ac68fd5197876f389f802d461669034fd6c"
REFERENCE_SHA256 = "2703f5ae720e0f5a973244bbca8275b6654c7c25b3c17c95e7dd618dcec4ebbf"
REPLAY_STEPS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
MILESTONE_STEPS = [0, *REPLAY_STEPS]
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
OLD_PANEL_SEEDS = (7218790, 7218791, 7218792, 7300000, 7310000, 7400000)
TARGET_ROUTE = "l2_core_l3_supported"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--training-chunks", type=int, default=4)
    parser.add_argument("--training-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--maximum-steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--training-eval-batches", type=int, default=10)
    parser.add_argument("--training-eval-batch-size", type=int, default=8)
    parser.add_argument("--causal-chunks", type=int, default=16)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--causal-eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=7410000)
    parser.add_argument("--formed-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--core-retention-threshold", type=float, default=0.80)
    parser.add_argument("--pair-sufficiency-threshold", type=float, default=0.90)
    parser.add_argument("--pair-gain-threshold", type=float, default=0.03)
    parser.add_argument("--roll-drop-threshold", type=float, default=0.05)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--output", default="experiments/level7_4_1/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "chunk_size": 128,
        "training_chunks": 4,
        "training_batch_size": 4,
        "learning_rate": 1e-3,
        "probe_weight": 0.5,
        "maximum_steps": 1500,
        "eval_every": 100,
        "training_eval_batches": 10,
        "training_eval_batch_size": 8,
        "causal_chunks": 16,
        "causal_samples": 1024,
        "causal_eval_batch_size": 16,
        "dataset_seed": 7410000,
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
        raise ValueError(f"Formal Level 7.4.1 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_4_1/formal":
        raise ValueError("Formal output path is locked")


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _digest_value(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        byte_view = tensor.reshape(1).view(torch.uint8) if tensor.ndim == 0 else tensor.view(torch.uint8)
        digest.update(byte_view.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _digest_value(digest, key)
            _digest_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _digest_value(digest, item)
    elif value is None:
        digest.update(b"none")
    elif isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
    else:
        raise TypeError(f"Unsupported fingerprint value: {type(value)!r}")


def canonical_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _digest_value(digest, value)
    return digest.hexdigest()


def state_dict_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Match run_level7_1_local.model_fingerprint without constructing a module."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_equal(left: list[torch.Tensor], right: list[torch.Tensor]) -> bool:
    return len(left) == len(right) and all(
        torch.equal(a.detach().cpu(), b.detach().cpu())
        for a, b in zip(left, right)
    )


def validate_sources(protocol: dict[str, Any]) -> dict[str, Any]:
    registered = protocol["frozen_sources"]
    expected = {
        "start_checkpoint": START_CHECKPOINT.as_posix(),
        "start_checkpoint_sha256": START_SHA256,
        "reference_endpoint": REFERENCE_CHECKPOINT.as_posix(),
        "reference_endpoint_sha256": REFERENCE_SHA256,
        "seed": 1879,
        "parent_result": "experiments/level7_4/formal/result.json",
    }
    if registered != expected:
        raise RuntimeError("Static C2/C4 source registration changed")
    start_path = ROOT / START_CHECKPOINT
    reference_path = ROOT / REFERENCE_CHECKPOINT
    if sha256_file(start_path) != START_SHA256:
        raise RuntimeError("Original C2 checkpoint hash mismatch")
    if sha256_file(reference_path) != REFERENCE_SHA256:
        raise RuntimeError("Original C4 checkpoint hash mismatch")

    parent_path = ROOT / "experiments/level7_4/formal/result.json"
    parent = read_json(parent_path)
    if not parent["integrity"]["passed"]:
        raise RuntimeError("Level 7.4 parent integrity did not pass")
    if (
        parent["diagnosis"]["classification"]
        != "l2_core_l3_support_established_by_16chunk_and_stable"
    ):
        raise RuntimeError("Unexpected Level 7.4 parent classification")
    if (
        sha256_file(ROOT / "run_level7_4_local.py")
        != parent["integrity"]["script_sha256"]
    ):
        raise RuntimeError("Level 7.4 runner changed after the parent result")
    if (
        sha256_file(ROOT / "experiments/level7_4/preregistration.json")
        != parent["integrity"]["preregistration_sha256"]
    ):
        raise RuntimeError("Level 7.4 preregistration changed after the parent result")
    parent_runs = {row["id"]: row for row in parent["runs"]}
    if parent_runs["curriculum_2"]["profile"]["route_class"] != "unformed_behavior":
        raise RuntimeError("Level 7.4 C2 source is not registered as unformed")
    if parent_runs["curriculum_4"]["profile"]["route_class"] != TARGET_ROUTE:
        raise RuntimeError("Level 7.4 C4 source lacks the target route")
    if parent_runs["curriculum_2"]["integrity"]["checkpoint_sha256"] != START_SHA256:
        raise RuntimeError("Level 7.4 C2 checkpoint audit changed")
    if parent_runs["curriculum_4"]["integrity"]["checkpoint_sha256"] != REFERENCE_SHA256:
        raise RuntimeError("Level 7.4 C4 checkpoint audit changed")

    start_state = torch.load(start_path, map_location="cpu", weights_only=False)
    reference_state = torch.load(reference_path, map_location="cpu", weights_only=False)
    if len(start_state.get("stages", [])) != 1:
        raise RuntimeError("Original C2 stage metadata changed")
    if [row["chunks"] for row in reference_state.get("stages", [])] != [2, 4]:
        raise RuntimeError("Original C4 stage metadata changed")
    stage2_history = [
        row for row in reference_state["history"] if int(row["stage"]) == 2
    ]
    if [row["step"] for row in stage2_history] != REPLAY_STEPS:
        raise RuntimeError("Original C4 validation schedule changed")
    if reference_state["stages"][-1]["steps"] != 1000:
        raise RuntimeError("Original C4 stop step changed")

    start_model_fingerprint = state_dict_fingerprint(start_state["model"])
    reference_model_fingerprint = state_dict_fingerprint(reference_state["model"])
    if (
        start_model_fingerprint
        != parent_runs["curriculum_2"]["integrity"]["model_fingerprint_before"]
    ):
        raise RuntimeError("C2 model fingerprint disagrees with Level 7.4")
    if (
        reference_model_fingerprint
        != parent_runs["curriculum_4"]["integrity"]["model_fingerprint_before"]
    ):
        raise RuntimeError("C4 model fingerprint disagrees with Level 7.4")
    audit = {
        "seed": 1879,
        "start_checkpoint": START_CHECKPOINT.as_posix(),
        "start_checkpoint_sha256": START_SHA256,
        "start_model_fingerprint": start_model_fingerprint,
        "reference_checkpoint": REFERENCE_CHECKPOINT.as_posix(),
        "reference_checkpoint_sha256": REFERENCE_SHA256,
        "reference_model_fingerprint": reference_model_fingerprint,
        "reference_probe_fingerprint": state_dict_fingerprint(reference_state["probe"]),
        "reference_optimizer_fingerprint": canonical_fingerprint(reference_state["optimizer"]),
        "reference_cpu_rng_fingerprint": canonical_fingerprint(reference_state["cpu_rng"]),
        "reference_cuda_rng_fingerprint": canonical_fingerprint(reference_state["cuda_rng"]),
        "reference_stop_step": reference_state["stages"][-1]["steps"],
        "reference_stage2_history": stage2_history,
        "parent_result": "experiments/level7_4/formal/result.json",
        "parent_result_sha256": sha256_file(parent_path),
        "parent_classification": parent["diagnosis"]["classification"],
        "source_validation_passed": True,
    }
    del start_state, reference_state
    return audit


def replay_snapshot_path(replay_root: Path, step: int) -> Path:
    return replay_root / f"model_step{step:04d}.pt"


def save_replay_state(
    replay_root: Path,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    consecutive: int,
    history: list[dict[str, Any]],
) -> None:
    snapshot = {
        "model": model.state_dict(),
        "replay_step": step,
        "model_fingerprint": model_fingerprint(model),
    }
    atomic_torch_save(replay_snapshot_path(replay_root, step), snapshot)
    resume = {
        "model": model.state_dict(),
        "probe": probe.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "replay_step": step,
        "consecutive": consecutive,
        "replay_history": history,
    }
    atomic_torch_save(replay_root / "resume.pt", resume)


def run_exact_replay(
    args: argparse.Namespace,
    source_audit: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    gate_path = root / "replay_gate.json"
    if gate_path.exists() and not args.force:
        return read_json(gate_path)
    replay_root = root / "replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    resume_path = replay_root / "resume.pt"
    if resume_path.exists() and not args.force:
        state = restore(resume_path, model, probe, optimizer, device)
        last_step = int(state["replay_step"])
        consecutive = int(state["consecutive"])
        history = state["replay_history"]
        print(f"resumed deterministic replay at step={last_step}", flush=True)
    else:
        restore(ROOT / START_CHECKPOINT, model, probe, optimizer, device)
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
                record = {"stage": 2, "step": step, **metric}
                history.append(record)
                ok = metric["query"] >= 0.95
                consecutive = consecutive + 1 if ok else 0
                save_replay_state(
                    replay_root, model, probe, optimizer,
                    step, consecutive, history,
                )
                atomic_save(root / "replay_progress.json", {
                    "stage": "exact_replay",
                    "completed_step": step,
                    "consecutive_behavior_passes": consecutive,
                    "saved_milestones": [
                        value for value in REPLAY_STEPS
                        if replay_snapshot_path(replay_root, value).is_file()
                    ],
                    "causal_panel_opened": False,
                })
                print(
                    f"replay step={step} query={metric['query']:.2%} "
                    f"probe={metric['probe_min']:.2%} consecutive={consecutive}",
                    flush=True,
                )
                if consecutive >= 2:
                    stop_step = step
                    break
    if stop_step is None:
        stop_step = args.maximum_steps

    reference = torch.load(
        ROOT / REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    reference_history = source_audit["reference_stage2_history"]
    current_model = model_fingerprint(model)
    current_probe = state_dict_fingerprint(probe.state_dict())
    current_optimizer = canonical_fingerprint(optimizer.state_dict())
    current_cpu_rng = torch.get_rng_state().cpu()
    current_cuda_rng = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    comparisons = {
        "model_state_exact": (
            current_model == source_audit["reference_model_fingerprint"]
        ),
        "probe_state_exact": (
            current_probe == source_audit["reference_probe_fingerprint"]
        ),
        "optimizer_state_exact": (
            current_optimizer == source_audit["reference_optimizer_fingerprint"]
        ),
        "CPU_RNG_exact": torch.equal(current_cpu_rng, reference["cpu_rng"].cpu()),
        "CUDA_RNG_exact": rng_equal(current_cuda_rng, reference["cuda_rng"]),
        "validation_history_exact": history == reference_history,
        "stop_step_exact": stop_step == source_audit["reference_stop_step"] == 1000,
        "consecutive_pass_state_exact": consecutive == 2,
        "all_registered_milestones_saved": all(
            replay_snapshot_path(replay_root, step).is_file()
            for step in REPLAY_STEPS
        ),
    }
    gate = {
        "passed": all(comparisons.values()),
        "comparisons": comparisons,
        "stop_step": stop_step,
        "consecutive_behavior_passes": consecutive,
        "replay_model_fingerprint": current_model,
        "reference_model_fingerprint": source_audit["reference_model_fingerprint"],
        "replay_probe_fingerprint": current_probe,
        "reference_probe_fingerprint": source_audit["reference_probe_fingerprint"],
        "replay_optimizer_fingerprint": current_optimizer,
        "reference_optimizer_fingerprint": source_audit["reference_optimizer_fingerprint"],
        "replay_CPU_RNG_fingerprint": canonical_fingerprint(current_cpu_rng),
        "reference_CPU_RNG_fingerprint": source_audit["reference_cpu_rng_fingerprint"],
        "replay_CUDA_RNG_fingerprint": canonical_fingerprint(current_cuda_rng),
        "reference_CUDA_RNG_fingerprint": source_audit["reference_cuda_rng_fingerprint"],
        "replay_validation_history": history,
        "reference_validation_history": reference_history,
        "causal_panel_authorized": all(comparisons.values()),
    }
    atomic_save(gate_path, gate)
    del reference, model, probe, optimizer
    torch.cuda.empty_cache()
    return gate


def model_displacement(
    state: dict[str, torch.Tensor], base: dict[str, torch.Tensor]
) -> dict[str, float]:
    squared_difference = 0.0
    squared_base = 0.0
    maximum_absolute = 0.0
    for name in sorted(base):
        left = state[name].detach().cpu()
        right = base[name].detach().cpu()
        if not (left.is_floating_point() or left.is_complex()):
            continue
        difference = left.to(torch.float64) - right.to(torch.float64)
        squared_difference += difference.square().sum().item()
        squared_base += right.to(torch.float64).square().sum().item()
        maximum_absolute = max(maximum_absolute, difference.abs().max().item())
    l2 = squared_difference ** 0.5
    base_l2 = squared_base ** 0.5
    return {
        "L2_parameter_displacement": l2,
        "relative_L2_parameter_displacement": l2 / base_l2 if base_l2 else 0.0,
        "maximum_absolute_parameter_change": maximum_absolute,
    }


def causal_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        chunks=args.causal_chunks,
        chunk_size=args.chunk_size,
        samples=args.causal_samples,
        eval_batch_size=args.causal_eval_batch_size,
        dataset_seed=args.dataset_seed,
    )


def milestone_sources(root: Path) -> list[dict[str, Any]]:
    replay_root = root / "replay"
    sources = []
    for step in MILESTONE_STEPS:
        path = ROOT / START_CHECKPOINT if step == 0 else replay_snapshot_path(replay_root, step)
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append({
            "id": f"step_{step:04d}",
            "step": step,
            "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(path),
            "origin": "original_C2" if step == 0 else "exact_replay",
        })
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


def run_causal_milestone(
    source: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
    base_model_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    folder = root / "causal" / source["id"]
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        result = read_json(result_path)
        if result["source"]["checkpoint_sha256"] != source["checkpoint_sha256"]:
            raise RuntimeError(f"Cached milestone source mismatch: {source['id']}")
        return result
    state = torch.load(ROOT / source["checkpoint"], map_location="cpu", weights_only=False)
    displacement = model_displacement(state["model"], base_model_state)
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
        raise RuntimeError(f"Milestone model fingerprint mismatch: {source['id']}")
    condition_path = folder / "condition_progress.json"
    metrics = (
        read_json(condition_path)
        if condition_path.exists() and not args.force else {}
    )
    validate_resumed_metrics(metrics, args)
    evaluation_args = causal_args(args)
    for condition in CONDITIONS:
        if condition in metrics:
            continue
        metric = evaluate_condition(model, evaluation_args, condition, device, dtype)
        metrics[condition] = metric
        atomic_save(condition_path, metrics)
        print(
            f"milestone={source['id']} condition={condition} "
            f"query={metric['query']:.2%} local={metric['local']:.2%}",
            flush=True,
        )
    fingerprint_after = model_fingerprint(model)
    profile = classify_checkpoint(metrics, args)
    integrity = {
        "checkpoint_sha256": source["checkpoint_sha256"],
        "expected_model_fingerprint": expected_fingerprint,
        "model_fingerprint_before": fingerprint_before,
        "model_fingerprint_after": fingerprint_after,
        "model_fingerprint_unchanged": fingerprint_before == fingerprint_after,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "all_conditions_present": set(metrics) == set(CONDITIONS),
        "condition_order_exact": list(metrics) == CONDITIONS,
        "fixed_samples_every_condition": all(
            row["samples"] == args.causal_samples for row in metrics.values()
        ),
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


def diagnose_dense_trajectory(
    results: list[dict[str, Any]], replay_gate: dict[str, Any]
) -> dict[str, Any]:
    if not replay_gate["passed"]:
        return {
            "classification": "replay_endpoint_mismatch_causal_panel_closed",
            "endpoint_qualification_passed": False,
            "causal_panel_opened": False,
            "formation_interval": None,
            "registered_stop_boundary": (
                "Do not interpret milestones or relax exact replay requirements."
            ),
        }
    steps = [row["step"] for row in results]
    if steps != MILESTONE_STEPS:
        raise RuntimeError("Dense milestone order changed")
    target_flags = [row["profile"]["route_class"] == TARGET_ROUTE for row in results]
    target_changes = sum(
        left != right for left, right in zip(target_flags, target_flags[1:])
    )
    any_regression = any(
        left and not right for left, right in zip(target_flags, target_flags[1:])
    )
    if any_regression or target_changes > 1:
        classification = "exact_replay_with_route_regression_or_multiple_transitions"
    elif target_flags[-1] and not target_flags[0] and target_changes == 1:
        classification = "exact_replay_and_single_stable_formation_transition"
    else:
        classification = "exact_replay_without_registered_route_formation"
    first_target_index = next(
        (index for index, value in enumerate(target_flags) if value), None
    )
    first_formed_index = next(
        (
            index for index, row in enumerate(results)
            if row["profile"]["behavior_formed"]
        ),
        None,
    )
    if first_target_index is not None and first_target_index > 0:
        formation_interval = {
            "last_non_target_step": results[first_target_index - 1]["step"],
            "first_stable_target_step": results[first_target_index]["step"],
        }
    else:
        formation_interval = None
    route_transitions = []
    for previous, current in zip(results, results[1:]):
        old_class = previous["profile"]["route_class"]
        new_class = current["profile"]["route_class"]
        if old_class != new_class:
            route_transitions.append({
                "from_step": previous["step"],
                "to_step": current["step"],
                "from_class": old_class,
                "to_class": new_class,
            })
    first_target_step = (
        results[first_target_index]["step"] if first_target_index is not None else None
    )
    first_formed_step = (
        results[first_formed_index]["step"] if first_formed_index is not None else None
    )
    return {
        "classification": classification,
        "endpoint_qualification_passed": True,
        "causal_panel_opened": True,
        "milestones": len(results),
        "first_formed_16chunk_step": first_formed_step,
        "first_l2_core_l3_supported_step": first_target_step,
        "behavior_route_synchronous_at_saved_resolution": (
            first_formed_step == first_target_step and first_formed_step is not None
        ),
        "formation_interval": formation_interval,
        "target_membership_changes": target_changes,
        "target_route_regression_observed": any_regression,
        "route_transitions": route_transitions,
        "final_route_class": results[-1]["profile"]["route_class"],
        "registered_stop_boundary": (
            "Report the fixed dense trajectory; do not insert milestones or "
            "change thresholds after observing formation."
        ),
    }


def plot_dense_trajectory(
    results: list[dict[str, Any]], diagnosis: dict[str, Any], path: Path
) -> None:
    labels = [str(row["step"]) for row in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(4, 1, figsize=(18, 19), sharex=True)
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
        axis.axhline(20, color="#b23a48", linestyle=":")
        axis.axhline(80, color="#8c6d31", linestyle="-.")
        axis.axhline(90, color="#333333", linestyle="--")
        axis.set_ylim(0, 105)
        axis.set_ylabel("Query accuracy (%)")
        axis.set_title(title)
        axis.legend(ncol=4, fontsize=9, loc="lower right")

    displacement = np.array([
        100 * row["model_displacement_from_C2"]["relative_L2_parameter_displacement"]
        for row in results
    ])
    axes[3].plot(x, displacement, color="#6f4e7c", marker="o", linewidth=2)
    axes[3].set_ylabel("Relative parameter displacement (%)")
    axes[3].set_title("Model-state displacement from original C2")
    axes[3].grid(alpha=0.25)
    formation = diagnosis.get("formation_interval")
    if formation:
        first_step = formation["first_stable_target_step"]
        formation_index = next(
            index for index, row in enumerate(results) if row["step"] == first_step
        )
        for axis in axes:
            axis.axvline(
                formation_index, color="#2a9d8f", linestyle="--", linewidth=1.5
            )
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
    axes[3].set_xticks(x, labels)
    axes[3].set_xlabel("Exact replay step (0 = original C2)")
    fig.suptitle(
        "IST Level 7.4.1: Deterministic Dense C2-to-C4 Formation Replay",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_mini_branch(
    start_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    save_midpoint: bool,
    smoke_root: Path,
) -> dict[str, Any]:
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=1e-3
    )
    restore(start_path, model, probe, optimizer, device)
    eval_args = argparse.Namespace(
        eval_batches=1, eval_batch_size=4, chunk_size=args.chunk_size
    )
    history = []
    for step in range(1, 4):
        model.train()
        probe.train()
        random_step(model, probe, optimizer, args, 2, 2, 0.5, device, dtype)
        if step in (1, 3):
            metric = evaluate(model, probe, eval_args, 2, device, dtype)
            history.append({"step": step, **metric})
            if save_midpoint and step == 1:
                atomic_torch_save(smoke_root / "mini_midpoint.pt", {
                    "model": model.state_dict(),
                    "probe": probe.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                })
    result = {
        "model": model_fingerprint(model),
        "probe": state_dict_fingerprint(probe.state_dict()),
        "optimizer": canonical_fingerprint(optimizer.state_dict()),
        "cpu_rng": canonical_fingerprint(torch.get_rng_state()),
        "cuda_rng": canonical_fingerprint(torch.cuda.get_rng_state_all()),
        "history": history,
    }
    del model, probe, optimizer
    torch.cuda.empty_cache()
    return result


def run_smoke(args: argparse.Namespace) -> int:
    args.output = "experiments/level7_4_1/smoke"
    args.chunk_size = 32
    protocol = read_json(STATIC_PREREGISTRATION)
    source_audit = validate_sources(protocol)
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    smoke_root = ROOT / args.output
    smoke_root.mkdir(parents=True, exist_ok=True)
    set_seed(74123)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=1e-3
    )
    start_path = smoke_root / "mini_start.pt"
    atomic_torch_save(start_path, {
        "model": model.state_dict(),
        "probe": probe.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    })
    del model, probe, optimizer
    torch.cuda.empty_cache()
    reference = run_mini_branch(
        start_path, args, device, dtype, False, smoke_root
    )
    replay = run_mini_branch(
        start_path, args, device, dtype, True, smoke_root
    )
    comparisons = {key: reference[key] == replay[key] for key in reference}
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "actual_C2_C4_source_hashes_validated": source_audit["source_validation_passed"],
        "miniature_reference_replay_comparisons": comparisons,
        "intermediate_save_preserved_exact_trajectory": all(comparisons.values()),
    }
    result["passed"] = bool(
        result["actual_C2_C4_source_hashes_validated"]
        and result["intermediate_save_preserved_exact_trajectory"]
    )
    atomic_save(smoke_root / "result.json", result)
    start_path.unlink(missing_ok=True)
    (smoke_root / "mini_midpoint.pt").unlink(missing_ok=True)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def build_integrity(
    source_audit: dict[str, Any],
    replay_gate: dict[str, Any],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    causal_opened = bool(replay_gate["passed"])
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "source_validation_passed": source_audit["source_validation_passed"],
        "endpoint_qualification_passed": replay_gate["passed"],
        "causal_panel_opened": causal_opened,
        "milestones_exact": (
            [row["step"] for row in results] == MILESTONE_STEPS
            if causal_opened else len(results) == 0
        ),
        "all_milestone_integrity_passed": (
            all(row["integrity"]["passed"] for row in results)
            if causal_opened else True
        ),
        "fixed_N_completed": (
            all(
                metric["samples"] == args.causal_samples
                for row in results for metric in row["metrics"].values()
            )
            if causal_opened else True
        ),
        "shared_fresh_dataset_seed": args.dataset_seed if causal_opened else None,
        "no_parent_panel_reuse": args.dataset_seed not in OLD_PANEL_SEEDS,
        "no_posthoc_milestone_or_training_change": True,
        "seed909_used": False,
    }
    integrity["passed"] = bool(
        integrity["source_validation_passed"]
        and integrity["milestones_exact"]
        and integrity["all_milestone_integrity_passed"]
        and integrity["fixed_N_completed"]
        and integrity["no_parent_panel_reuse"]
        and integrity["no_posthoc_milestone_or_training_change"]
        and not integrity["seed909_used"]
    )
    return integrity


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
    atomic_save(root / "progress.json", {
        "stage": "exact_replay",
        "endpoint_qualification_passed": None,
        "causal_panel_opened": False,
        "completed_milestones": [],
        "seed909_locked": True,
    })
    replay_gate = run_exact_replay(args, source_audit, device, dtype, root)
    results: list[dict[str, Any]] = []
    if replay_gate["passed"]:
        atomic_save(root / "progress.json", {
            "stage": "dense_frozen_causal_trajectory",
            "endpoint_qualification_passed": True,
            "causal_panel_opened": True,
            "completed_milestones": [],
            "seed909_locked": True,
        })
        start_state = torch.load(
            ROOT / START_CHECKPOINT, map_location="cpu", weights_only=False
        )
        base_model_state = start_state["model"]
        del start_state
        sources = milestone_sources(root)
        completed = []
        for source in sources:
            result = run_causal_milestone(
                source, args, device, dtype, root, base_model_state
            )
            results.append(result)
            completed.append(source["id"])
            atomic_save(root / "progress.json", {
                "stage": "dense_frozen_causal_trajectory",
                "endpoint_qualification_passed": True,
                "causal_panel_opened": True,
                "completed_milestones": completed,
                "seed909_locked": True,
            })
        del base_model_state
    diagnosis = diagnose_dense_trajectory(results, replay_gate)
    integrity = build_integrity(source_audit, replay_gate, results, args)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "protocol": protocol,
        "source_audit": source_audit,
        "replay_gate": replay_gate,
        "runs": results,
        "diagnosis": diagnosis,
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    summary = {
        "integrity": integrity,
        "replay_gate": replay_gate,
        "diagnosis": diagnosis,
        "trajectory": [
            {
                "step": row["step"],
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
                "model_displacement_from_C2": row["model_displacement_from_C2"],
            }
            for row in results
        ],
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    if replay_gate["passed"]:
        plot_dense_trajectory(
            results, diagnosis, root / "dense_formation_trajectory.png"
        )
    atomic_save(root / "progress.json", {
        "stage": "complete",
        "endpoint_qualification_passed": replay_gate["passed"],
        "causal_panel_opened": replay_gate["passed"],
        "completed_milestones": [row["id"] for row in results],
        "classification": diagnosis["classification"],
        "integrity_passed": integrity["passed"],
        "seed909_locked": True,
    })
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
