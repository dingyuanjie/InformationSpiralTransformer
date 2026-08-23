"""Level 7.5.5.6: L3 slot AdamW step/weight-decay decomposition."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

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
from run_level7_5_3_2_local import (
    EXACT_ARM as PARENT_EXACT_ARM,
    SCREEN_STEPS,
    SOURCE_SPECS,
    exact_milestones,
    expected_layer,
    expected_route_formed,
    source_milestone,
)


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_5_5_6"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
PARENT_RESULT = ROOT / "experiments/level7_5_5_5/formal/result.json"
PARENT_RESULT_SHA256 = "2f684d0aed31fd9b4640f08e27aa9d08150bf33ecc2137f417e25432a38677b4"
PARENT_CLASSIFICATION = "minimum_sufficient_recovery_candidates_found"
EXACT_ARM = PARENT_EXACT_ARM
INTERVENTION_ARMS = [
    "all_memory_frozen",
    "restore_l3_read_fusion",
    "read_slot_dose_025_keep", "read_slot_dose_025_reset_m1",
    "read_slot_dose_025_reset_m1_step", "read_slot_dose_025_reset_m1_nodecay",
    "read_slot_dose_025_reset_m1_step_nodecay",
]
ALL_ARMS = [EXACT_ARM, *INTERVENTION_ARMS]
SCREEN_DATASET_SEED = 7556000
CONFIRM_DATASET_SEED = 7556001
ARM_GROUPS = {EXACT_ARM:{"kind":"none","layer":None}}
for _arm in INTERVENTION_ARMS: ARM_GROUPS[_arm] = {"kind":"combination","layer":None}


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
    parser.add_argument("--output", default="experiments/level7_5_5_6/formal")
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
        raise ValueError(f"Formal Level 7.5.5.6 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_5_5_6/formal":
        raise ValueError("Formal Level 7.5.5.6 output path is locked")


def validate_static_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("frozen_source_ids") != [spec["id"] for spec in SOURCE_SPECS]:
        raise RuntimeError("Static source selection changed")
    if protocol.get("arms") not in ([{"name": name, **ARM_GROUPS[name]} for name in ALL_ARMS], ALL_ARMS):
        raise RuntimeError("Static parameter-group arms changed")
    if protocol.get("trajectory_screen_panel", {}).get("milestones") != SCREEN_STEPS:
        raise RuntimeError("Static trajectory milestones changed")
    if protocol.get("trajectory_screen_panel", {}).get("conditions") != SCREEN_CONDITIONS:
        raise RuntimeError("Static screen conditions changed")
    if protocol.get("final_confirmation_panel", {}).get("conditions") != CONDITIONS:
        raise RuntimeError("Static confirmation conditions changed")


def validate_parent(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise RuntimeError("Frozen Level 7.5.3.2 result changed")
    parent = read_json(PARENT_RESULT)
    if not parent["integrity"]["passed"]:
        raise RuntimeError("Level 7.5.3.2 parent integrity did not pass")
    if parent["diagnosis"]["classification"] != PARENT_CLASSIFICATION:
        raise RuntimeError("Level 7.5.3.2 parent classification changed")
    audit = []
    for spec in SOURCE_SPECS:
        path = ROOT / spec["checkpoint"]
        if not path.is_file() or sha256_file(path) != spec["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen source changed: {spec['id']}")
        if path.stat().st_size != spec["checkpoint_size_bytes"]:
            raise RuntimeError(f"Frozen source size changed: {spec['id']}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model", "probe", "optimizer", "cpu_rng", "cuda_rng"}
        if not required.issubset(state):
            raise RuntimeError(f"Frozen source incomplete: {spec['id']}")
        refs = []
        for row in exact_milestones(spec):
            ref = ROOT / row["checkpoint"]
            if not ref.is_file() or sha256_file(ref) != row["checkpoint_sha256"]:
                raise RuntimeError(
                    f"Exact reference changed: {spec['id']} step={row['recovery_step']}"
                )
            refs.append({**row, "checkpoint_size_bytes": ref.stat().st_size})
        audit.append(
            {
                "id": spec["id"],
                "source_sha256": spec["checkpoint_sha256"],
                "model_fingerprint": state_dict_fingerprint(state["model"]),
                "probe_fingerprint": state_dict_fingerprint(state["probe"]),
                "optimizer_fingerprint": canonical_fingerprint(state["optimizer"]),
                "CPU_RNG_fingerprint": canonical_fingerprint(state["cpu_rng"]),
                "CUDA_RNG_fingerprint": canonical_fingerprint(state["cuda_rng"]),
                "exact_reference_milestones": refs,
                "passed": True,
            }
        )
        del state
    return audit, {
        "result": str(PARENT_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": PARENT_RESULT_SHA256,
        "classification": parent["diagnosis"]["classification"],
        "integrity_passed": parent["integrity"]["passed"],
    }


def selected_parameter_names(model: torch.nn.Module, arm: str) -> list[str]:
    if arm == EXACT_ARM:
        return []
    all_groups={(layer,kind) for layer in (2,3) for kind in ('slot_queries','write_core','read_fusion')}
    restored=set()
    if arm.startswith('read_slot_dose_'): restored={(3,'slot_queries'),(3,'read_fusion')}
    elif arm.startswith('restore_l2_'): restored={(2,arm[len('restore_l2_'):])}
    elif arm.startswith('restore_l3_'): restored={(3,arm[len('restore_l3_'):])}
    elif arm == 'restore_all_l2': restored={(2,k) for k in ('slot_queries','write_core','read_fusion')}
    elif arm == 'restore_all_l3': restored={(3,k) for k in ('slot_queries','write_core','read_fusion')}
    elif arm.startswith('restore_cross_'):
        kind=arm[len('restore_cross_'):]; restored={(2,kind),(3,kind)}
    elif arm != 'all_memory_frozen': raise RuntimeError(f'Unknown recovery arm={arm}')
    groups=sorted(all_groups-restored)
    prefixes=[]
    for layer, kind in groups:
        block=layer-1
        if kind=='slot_queries': prefixes.append(f'blocks.{block}.memory.slot_queries')
        elif kind=='write_core': prefixes += [f'blocks.{block}.memory.encoder.',f'blocks.{block}.memory.memory_key.',f'blocks.{block}.memory.memory_attention.']
        elif kind=='read_fusion': prefixes += [f'blocks.{block}.memory_read.',f'blocks.{block}.memory_fusion_gate.']
    names = [name for name, _ in model.named_parameters() if name.startswith(tuple(prefixes))]
    if not names:
        raise RuntimeError(f"No parameters matched intervention arm={arm}")
    return names


def delayed_slot_names(model: torch.nn.Module, arm: str) -> list[str]:
    if not arm.startswith('read_slot_dose_'):
        return []
    return [name for name, _ in model.named_parameters() if name == 'blocks.2.memory.slot_queries']


def delayed_unfreeze_step(arm: str) -> int | None:
    return 300 if arm.startswith('read_slot_dose_') else None


def slot_gradient_dose(arm: str) -> float:
    return int(arm.split('_')[3]) / 100.0 if arm.startswith('read_slot_dose_') else 1.0


def slot_optimizer_reset_mode(arm: str) -> str:
    if arm.endswith('_reset_m12'):
        return 'm12'
    if '_reset_m1' in arm:
        return 'm1'
    return 'keep'


def reset_slot_optimizer_state(optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter], mode: str) -> None:
    if mode == 'keep':
        return
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if 'exp_avg' in state:
            state['exp_avg'].zero_()
        if mode == 'm12':
            for key in ('exp_avg_sq', 'max_exp_avg_sq'):
                if key in state:
                    state[key].zero_()


def reset_slot_step_requested(arm: str) -> bool:
    return '_step' in arm


def no_slot_weight_decay_requested(arm: str) -> bool:
    return arm.endswith('_nodecay')


def reset_slot_optimizer_step(optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        state=optimizer.state.get(parameter,{})
        if 'step' in state:
            if torch.is_tensor(state['step']): state['step'].zero_()
            else: state['step']=0


def install_slot_decay_cancellation(optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]):
    original_step=optimizer.step
    def step_without_slot_decay(*args, **kwargs):
        before=[parameter.detach().clone() for parameter in parameters]
        result=original_step(*args, **kwargs)
        for group in optimizer.param_groups:
            lr=float(group['lr']); decay=float(group.get('weight_decay',0.0))
            members=set(group['params'])
            with torch.no_grad():
                for parameter, old in zip(parameters,before):
                    if parameter in members: parameter.add_(old, alpha=lr*decay)
        return result
    optimizer.step=step_without_slot_decay


def fingerprint_names(model: torch.nn.Module, names: list[str]) -> str:
    state = model.state_dict()
    return state_dict_fingerprint({name: state[name] for name in names})


def initialize_exact(
    spec: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, dict[str, Any]]:
    source = torch.load(ROOT / spec["checkpoint"], map_location=device, weights_only=False)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    model.load_state_dict(source["model"])
    probe.load_state_dict(source["probe"])
    optimizer.load_state_dict(source["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    torch.set_rng_state(source["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in source["cuda_rng"]])
    names = selected_parameter_names(model, arm)
    delayed_names = delayed_slot_names(model, arm)
    source_model_exact = state_dict_fingerprint(model.state_dict()) == state_dict_fingerprint(source["model"])
    source_probe_exact = state_dict_fingerprint(probe.state_dict()) == state_dict_fingerprint(source["probe"])
    source_optimizer_exact = canonical_fingerprint(optimizer.state_dict()) == canonical_fingerprint(source["optimizer"])
    source_cpu_exact = canonical_fingerprint(torch.get_rng_state().cpu()) == canonical_fingerprint(source["cpu_rng"])
    source_cuda_exact = canonical_fingerprint([item.cpu() for item in torch.cuda.get_rng_state_all()]) == canonical_fingerprint(source["cuda_rng"])
    for name, parameter in model.named_parameters():
        if name in names or name in delayed_names:
            parameter.requires_grad_(False)
    audit = {
        "arm": arm,
        "frozen_parameter_names": names,
        "frozen_parameter_count": len(names),
        "delayed_parameter_names": delayed_names,
        "delayed_unfreeze_step": delayed_unfreeze_step(arm),
        "slot_gradient_dose": slot_gradient_dose(arm),
        "slot_optimizer_reset_mode": slot_optimizer_reset_mode(arm),
        "slot_optimizer_step_reset": reset_slot_step_requested(arm),
        "slot_weight_decay_cancelled": no_slot_weight_decay_requested(arm),
        "source_model_exact": source_model_exact,
        "source_probe_exact": source_probe_exact,
        "source_optimizer_exact": source_optimizer_exact,
        "source_CPU_RNG_exact": source_cpu_exact,
        "source_CUDA_RNG_exact": source_cuda_exact,
        "frozen_initial_fingerprint": fingerprint_names(model, names) if names else None,
        "frozen_parameters_require_grad_false": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name in names or name in delayed_names
        ),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "passed": bool(
            source_model_exact
            and source_probe_exact
            and source_optimizer_exact
            and source_cpu_exact
            and source_cuda_exact
            and (not names or all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name in names or name in delayed_names
            ))
        ),
    }
    del source
    return model, probe, optimizer, audit


def snapshot_path(folder: Path, step: int) -> Path:
    return folder / "training" / f"model_step{step:04d}.pt"


def run_training_arm(
    spec: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
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
        if state["source_id"] != spec["id"] or state["arm"] != arm:
            raise RuntimeError(f"Resume metadata mismatch: {spec['id']} {arm}")
        last_step = int(state["training_step"])
        history = state["training_history"]
        audit = state["initialization_audit"]
        names = list(audit["frozen_parameter_names"])
        delayed_names = list(audit.get("delayed_parameter_names", []))
        threshold = audit.get("delayed_unfreeze_step")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name not in names and not (name in delayed_names and last_step < threshold))
        print(f"seed={spec['seed']} arm={arm} resumed step={last_step}", flush=True)
    else:
        del model, probe, optimizer
        model, probe, optimizer, audit = initialize_exact(spec, arm, args, device)
        if not audit["passed"]:
            raise RuntimeError(f"Initialization gate failed: {spec['id']} {arm}")
        names = list(audit["frozen_parameter_names"])
        delayed_names = list(audit.get("delayed_parameter_names", []))
        threshold = audit.get("delayed_unfreeze_step")
        last_step = 0
        history = []
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    hook_handles=[]
    dose=float(audit.get("slot_gradient_dose", 1.0))
    reset_mode=str(audit.get("slot_optimizer_reset_mode", "keep"))
    reset_step=bool(audit.get("slot_optimizer_step_reset", False))
    cancel_decay=bool(audit.get("slot_weight_decay_cancelled", False))
    if threshold is not None and last_step > threshold:
        delayed_parameters=[]
        for name, parameter in model.named_parameters():
            if name in delayed_names:
                parameter.requires_grad_(True)
                delayed_parameters.append(parameter)
                hook_handles.append(parameter.register_hook(lambda grad, scale=dose: grad * scale))
        if cancel_decay: install_slot_decay_cancellation(optimizer, delayed_parameters)
    eval_args = argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    for step in range(last_step + 1, args.training_steps + 1):
        if threshold is not None and step == threshold + 1:
            delayed_parameters=[]
            for name, parameter in model.named_parameters():
                if name in delayed_names:
                    parameter.requires_grad_(True)
                    delayed_parameters.append(parameter)
                    hook_handles.append(parameter.register_hook(lambda grad, scale=dose: grad * scale))
            reset_slot_optimizer_state(optimizer, delayed_parameters, reset_mode)
            if reset_step: reset_slot_optimizer_step(optimizer, delayed_parameters)
            if cancel_decay: install_slot_decay_cancellation(optimizer, delayed_parameters)
            print(f"seed={spec['seed']} arm={arm} opened_slots_after={threshold} dose={dose:.2f} reset={reset_mode} step_reset={reset_step} nodecay={cancel_decay}", flush=True)
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
                atomic_torch_save(
                    snapshot_path(folder, step),
                    {
                        "model": model.state_dict(),
                        "seed": spec["seed"],
                        "source_id": spec["id"],
                        "arm": arm,
                        "training_step": step,
                        "validation": metric,
                        "frozen_parameter_names": names,
                        "frozen_parameter_fingerprint": fingerprint_names(model, names) if names else None,
                        "delayed_parameter_names": delayed_names,
                        "delayed_parameter_fingerprint": fingerprint_names(model, delayed_names) if delayed_names else None,
                        "delayed_unfreeze_step": threshold,
                        "memory_mask_active": False,
                    },
                )
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
                    "initialization_audit": audit,
                    "memory_mask_active_steps": 0,
                },
            )
            print(
                f"seed={spec['seed']} arm={arm} step={step} "
                f"query={metric['query']:.2%} probe={metric['probe_min']:.2%}",
                flush=True,
            )
    milestones = []
    for step in SCREEN_STEPS[1:]:
        path = snapshot_path(folder, step)
        if not path.is_file():
            raise FileNotFoundError(path)
        milestones.append({
            "recovery_step": step,
            "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(path),
        })
    result = {
        "id": f"{spec['id']}__{arm}",
        "source_id": spec["id"],
        "seed": spec["seed"],
        "source_branch": spec["branch"],
        "arm": arm,
        "parameter_group": ARM_GROUPS[arm],
        "initialization_audit": audit,
        "frozen_parameter_names": names,
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
    return {
        "id": f"{spec['id']}__{arm}",
        "seed": spec["seed"],
        "branch": f"{spec['branch']}/{arm}",
        "baseline_route": spec["baseline_route"],
        "expected_route": spec["expected_route"],
    }


def source_panel_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return panel_spec(spec, "shared_source")


def mark_progress(progress: dict[str, Any], key: str, value: str) -> None:
    if value not in progress[key]:
        progress[key].append(value)


def screen_formed(spec: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> bool:
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
    source_by_id = {row["id"].removesuffix("__shared_source"): row for row in source_screens}
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for row in trajectory_screens:
        trajectories.setdefault(row["id"], []).append(row)
    confirms = {row["id"]: row for row in confirmations}
    outcomes = []
    for spec in SOURCE_SPECS:
        for arm in ALL_ARMS:
            ident = f"{spec['id']}__{arm}"
            rows = [source_by_id[spec["id"]], *sorted(trajectories[ident], key=lambda row: row["recovery_step"])]
            points = []
            for row in rows:
                profile = row["screen_profile"]
                points.append({
                    "training_step": row["recovery_step"],
                    "intact_query": profile["intact_query"],
                    "minimum_local": profile["minimum_local"],
                    "dominant_retention_layer": profile["dominant_retention_layer"],
                    "formed": screen_formed(spec, row, args),
                })
            confirmation = confirms[ident]
            final_route = confirmation["profile"]["route_class"]
            final_formed = expected_route_formed(spec, final_route)
            post = points[1:]
            outcomes.append({
                "id": ident,
                "source_id": spec["id"],
                "seed": spec["seed"],
                "selection_role": spec["selection_role"],
                "arm": arm,
                "parameter_group": ARM_GROUPS[arm],
                "trajectory": points,
                "formed_vector_300_600_1000": [row["formed"] for row in post],
                "trajectory_query_mean": statistics.fmean(row["intact_query"] for row in post),
                "trajectory_query_min": min(row["intact_query"] for row in post),
                "final_route": final_route,
                "final_expected_route_formed": final_formed,
                "final_intact_query": confirmation["metrics"]["intact"]["query"],
            })
    keyed = {(row["source_id"], row["arm"]): row for row in outcomes}
    comparisons = []
    for spec in SOURCE_SPECS:
        reference = keyed[(spec["id"], EXACT_ARM)]
        reference_queries = {row["training_step"]: row["intact_query"] for row in reference["trajectory"]}
        for arm in INTERVENTION_ARMS:
            row = keyed[(spec["id"], arm)]
            max_delta = max(abs(point["intact_query"] - reference_queries[point["training_step"]]) for point in row["trajectory"][1:])
            vector_changed = row["formed_vector_300_600_1000"] != reference["formed_vector_300_600_1000"]
            fate_changed = row["final_expected_route_formed"] != reference["final_expected_route_formed"]
            comparisons.append({
                "source_id": spec["id"],
                "selection_role": spec["selection_role"],
                "arm": arm,
                "formed_vector_changed": vector_changed,
                "final_fate_changed": fate_changed,
                "max_absolute_query_delta": max_delta,
                "material_effect": bool(vector_changed or fate_changed or max_delta >= args.material_query_delta),
            })
    by_arm = {arm: [row for row in comparisons if row["arm"] == arm] for arm in INTERVENTION_ARMS}
    effect_counts = {arm: sum(row["material_effect"] for row in rows) for arm, rows in by_arm.items()}
    recovery_gains={}
    recovery_counts={}
    for arm in INTERVENTION_ARMS[1:]:
        gains=[keyed[(spec['id'],arm)]['final_intact_query']-keyed[(spec['id'],'all_memory_frozen')]['final_intact_query'] for spec in SOURCE_SPECS]
        recovery_gains[arm]=gains
        recovery_counts[arm]=sum(gain >= 0.05 for gain in gains)
    sufficient=[arm for arm,count in recovery_counts.items() if count >= 3]
    dose_rows=[]
    for index, spec in enumerate(SOURCE_SPECS):
        read=recovery_gains['restore_l3_read_fusion'][index]
        row={'source_id':spec['id'],'read_only_gain':read}
        for arm in INTERVENTION_ARMS[2:]:
            row[arm]=recovery_gains[arm][index]
            row[f'{arm}_matches_read_only']=recovery_gains[arm][index] >= read
        dose_rows.append(row)
    if not integrity_passed:
        classification = "formal_integrity_failed_parameter_interpretation_closed"
    elif sufficient: classification = "minimum_sufficient_recovery_candidates_found"
    else: classification = "no_single_registered_recovery_sufficient"
    diagnosis = {
        "classification": classification,
        "effect_counts": effect_counts,
        "recovery_gain_over_all_frozen": recovery_gains,
        "recovery_count_at_5pp": recovery_counts,
        "minimum_sufficient_candidates": sufficient,
        "slot_gradient_dose_rows": dose_rows,
        "dose_matches_read_only_counts": {arm:sum(row[f'{arm}_matches_read_only'] for row in dose_rows) for arm in INTERVENTION_ARMS[2:]},
        "outcome_stratified_sources": len(SOURCE_SPECS),
        "interpretation_scope": "Four frozen outcome-stratified endpoints; not a prevalence estimate.",
        "registered_stop_boundary": "Report the fixed parameter-group intervention. Do not add groups, layers, seeds, steps, or post-hoc gates.",
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
    frozen_ok = []
    for row in training:
        initial = row["initialization_audit"]["frozen_initial_fingerprint"]
        for step in row["milestones"]:
            state = torch.load(ROOT / step["checkpoint"], map_location="cpu", weights_only=False)
            names = row["frozen_parameter_names"]
            state_fingerprint = state_dict_fingerprint({name: state["model"][name] for name in names}) if names else None
            frozen_ok.append(state_fingerprint == initial)
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "parent_result_sha256": sha256_file(PARENT_RESULT),
        "all_sources_validated": all(row["passed"] for row in source_audit),
        "expected_new_training_arms": len(SOURCE_SPECS) * len(INTERVENTION_ARMS),
        "completed_new_training_arms": len(training),
        "all_training_steps_exact": all(row["training_steps_completed"] == args.training_steps for row in training),
        "all_initialization_gates_passed": all(row["initialization_audit"]["passed"] for row in training),
        "all_optimizer_states_preserved": all(row["initialization_audit"]["source_optimizer_exact"] for row in training),
        "all_CPU_RNG_preserved": all(row["initialization_audit"]["source_CPU_RNG_exact"] for row in training),
        "all_CUDA_RNG_preserved": all(row["initialization_audit"]["source_CUDA_RNG_exact"] for row in training),
        "all_frozen_parameter_snapshots_unchanged": all(frozen_ok),
        "no_memory_masks": all(row["memory_mask_active_steps"] == 0 for row in training),
        "no_training_beyond_budget": all(not row["training_beyond_registered_budget"] for row in training),
        "expected_shared_source_screens": len(SOURCE_SPECS),
        "completed_shared_source_screens": len(source_screens),
        "expected_trajectory_screens": len(SOURCE_SPECS) * len(ALL_ARMS) * 3,
        "completed_trajectory_screens": len(trajectory_screens),
        "all_screen_integrity_passed": all(row["integrity"]["passed"] for row in [*source_screens, *trajectory_screens]),
        "expected_confirmation_panels": len(SOURCE_SPECS) * len(ALL_ARMS),
        "completed_confirmation_panels": len(confirmations),
        "all_confirmation_integrity_passed": all(row["integrity"]["passed"] for row in confirmations),
        "screen_dataset_seed": args.screen_dataset_seed,
        "confirmation_dataset_seed": args.confirm_dataset_seed,
        "seed909_used": False,
    }
    integrity["passed"] = bool(
        integrity["all_sources_validated"]
        and integrity["completed_new_training_arms"] == integrity["expected_new_training_arms"]
        and integrity["all_training_steps_exact"]
        and integrity["all_initialization_gates_passed"]
        and integrity["all_optimizer_states_preserved"]
        and integrity["all_CPU_RNG_preserved"]
        and integrity["all_CUDA_RNG_preserved"]
        and integrity["all_frozen_parameter_snapshots_unchanged"]
        and integrity["no_memory_masks"]
        and integrity["no_training_beyond_budget"]
        and integrity["completed_shared_source_screens"] == integrity["expected_shared_source_screens"]
        and integrity["completed_trajectory_screens"] == integrity["expected_trajectory_screens"]
        and integrity["all_screen_integrity_passed"]
        and integrity["completed_confirmation_panels"] == integrity["expected_confirmation_panels"]
        and integrity["all_confirmation_integrity_passed"]
        and not integrity["seed909_used"]
    )
    return integrity


def plot_parameter_effects(outcomes: list[dict[str, Any]], path: Path) -> None:
    palette=["#d1495b","#0077b6","#f4a261","#2a9d8f","#6a4c93","#8ac926","#e76f51","#264653","#457b9d","#bc6c25","#606c38","#9b5de5","#00b4d8","#f15bb5","#52b788"]
    colors={EXACT_ARM:"#333333", **{arm:palette[i] for i,arm in enumerate(INTERVENTION_ARMS)}}
    labels={EXACT_ARM:"exact reference", **{arm:arm.replace('_',' ') for arm in INTERVENTION_ARMS}}
    by_key = {(row["source_id"], row["arm"]): row for row in outcomes}
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    for axis, spec in zip(axes.flat, SOURCE_SPECS):
        for arm in ALL_ARMS:
            row = by_key[(spec["id"], arm)]
            axis.plot(
                [point["training_step"] for point in row["trajectory"]],
                [100 * point["intact_query"] for point in row["trajectory"]],
                marker="o",
                linewidth=2,
                color=colors[arm],
                label=labels[arm],
            )
        axis.axhline(90, color="#777777", linestyle="--", linewidth=1.5)
        axis.set_title(f"seed{spec['seed']} · {spec['selection_role']}")
        axis.set_xlabel("Additional C4 steps")
        axis.set_ylabel("Fresh 16-chunk query accuracy (%)")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.25)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=5)
    figure.suptitle("Level 7.5.5.6 L3 slot step/decay decomposition", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_smoke(args: argparse.Namespace) -> int:
    protocol = read_json(STATIC_PREREGISTRATION)
    validate_static_protocol(protocol)
    source_audit, parent_audit = validate_parent(protocol)
    configure_cuda()
    device = torch.device("cuda")
    spec = SOURCE_SPECS[0]
    rows = []
    for arm in INTERVENTION_ARMS:
        model, probe, optimizer, audit = initialize_exact(spec, arm, args, device)
        frozen_before = audit["frozen_initial_fingerprint"]
        random_step(model, probe, optimizer, args, 4, 2, args.probe_weight, device, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        frozen_after = fingerprint_names(model, audit["frozen_parameter_names"]) if audit["frozen_parameter_names"] else None
        rows.append({"arm": arm, "audit": audit, "frozen_unchanged_after_one_step": frozen_before == frozen_after})
        del model, probe, optimizer
        torch.cuda.empty_cache()
    result = {
        "smoke_test": True,
        "source_audit": source_audit,
        "parent_audit": parent_audit,
        "arms": rows,
        "passed": all(row["audit"]["passed"] and row["frozen_unchanged_after_one_step"] for row in rows),
    }
    atomic_save(ROOT / "experiments/level7_5_5_6/smoke/result.json", result)
    print("level7_5_5_6_SMOKE_PASS" if result["passed"] else "level7_5_5_6_SMOKE_FAIL")
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
    source_audit, parent_audit = validate_parent(protocol)
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
            "stage": "parameter_group_training",
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
            progress["stage"] = "parameter_group_training"
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
            milestones = exact_milestones(spec) if arm == EXACT_ARM else training_by_key[(spec["id"], arm)]["milestones"]
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
            milestones = exact_milestones(spec) if arm == EXACT_ARM else training_by_key[(spec["id"], arm)]["milestones"]
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
    integrity = build_integrity(source_audit, training, source_screens, trajectory_screens, confirmations, args)
    outcomes, causal = diagnose(source_screens, trajectory_screens, confirmations, args, integrity["passed"])
    elapsed = time.perf_counter() - started
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
        "diagnosis": causal["diagnosis"],
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed,
    }
    summary = {
        "diagnosis": causal["diagnosis"],
        "integrity": integrity,
        "endpoint_outcomes": outcomes,
        "causal_comparisons": causal["comparisons"],
        "elapsed_seconds_this_invocation": elapsed,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_parameter_effects(outcomes, root / "memory_parameter_group_effects.png")
    atomic_save(progress_path, {
        "stage": "complete",
        "completed_training_arms": len(training),
        "completed_source_screens": len(source_screens),
        "completed_trajectory_screens": len(trajectory_screens),
        "completed_confirmation_panels": len(confirmations),
        "classification": causal["diagnosis"]["classification"],
        "integrity_passed": integrity["passed"],
        "seed909_locked": True,
    })
    print(json.dumps(causal["diagnosis"], indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())



