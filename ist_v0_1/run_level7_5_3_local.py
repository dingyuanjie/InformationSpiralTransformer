"""Level 7.5.3: training-time counterfactual test of route commitment."""

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
import torch.nn.functional as F

from long_context_test import set_seed
from run_level6_2_local import evaluate, make_chunks, vector
from run_level6_6_local import build, random_step, restore
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save, model_fingerprint
from run_level7_3_local import CONDITIONS, evaluate_condition, read_json, sha256_file
from run_level7_4_1_local import (
    atomic_torch_save,
    canonical_fingerprint,
    rng_equal,
    state_dict_fingerprint,
)
from run_level7_5_local import profile_checkpoint


ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments/level7_5_3"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"
PARENT_RESULT = ROOT / "experiments/level7_5_2/formal/result.json"
PARENT_RESULT_SHA256 = "66aa4e6ca8f707fa0683005bb1c9eecb7159845bf269c11717e5b238384e2889"
FORMAL_SEEDS = [1879, 2203, 2551, 2909]
DEFAULT_L3_SEEDS = [2203, 2551, 2909]
FORMAL_DATASET_SEED = 7530000
BRANCHES = [
    {"name": "intact_replay", "suppression_role": "none"},
    {"name": "selected_layer_suppression", "suppression_role": "selected"},
    {"name": "other_layer_suppression", "suppression_role": "other"},
]

SOURCE_SPECS = [
    {
        "seed": 1879,
        "route_group": "exceptional_L2_core_L3_support",
        "expected_route": "l2_core_l3_supported",
        "selected_layer": 2,
        "other_layer": 3,
        "intervention_start_step": 1201,
        "intervention_end_step": 1400,
        "C2_stop_step": 2300,
        "C4_stop_step": 1000,
        "start_checkpoint": "experiments/level7_2/formal/seed1879/level6_1.pt",
        "start_checkpoint_sha256": "ffd969e6873f373f8a96132ff22387faa0b95dc05a52c6f08bda441acda9141a",
        "C2_reference": "experiments/level7_2/formal/seed1879/curriculum_stage1.pt",
        "C2_reference_sha256": "9939755860050c602798b6cec0320ac68fd5197876f389f802d461669034fd6c",
        "C4_reference": "experiments/level7_2/formal/seed1879/curriculum_stage2.pt",
        "C4_reference_sha256": "2703f5ae720e0f5a973244bbca8275b6654c7c25b3c17c95e7dd618dcec4ebbf",
        "C4_reference_history": "combined_curriculum_stage2",
    },
    {
        "seed": 2203,
        "route_group": "default_L3",
        "expected_route": "l3_core",
        "selected_layer": 3,
        "other_layer": 2,
        "intervention_start_step": 801,
        "intervention_end_step": 1000,
        "C2_stop_step": 1000,
        "C4_stop_step": 500,
        "start_checkpoint": "experiments/level7_5/formal/seed2203/level6_1.pt",
        "start_checkpoint_sha256": "56f16453bb76276d219c1c41eafe856f5c9cabeb1ecbe096ec6b3ec4a90b37b9",
        "C2_reference": "experiments/level7_5/formal/seed2203/curriculum_stage1.pt",
        "C2_reference_sha256": "9632d6ed078ed328a2541d680325590fa3c1c92459efcc5837735fbb7b67c141",
        "C4_reference": "experiments/level7_5/formal/seed2203/stage2/resume.pt",
        "C4_reference_sha256": "7d191e860b1eeda3e073cc602cd30a865198e6f18777aaaa479929ac8e9293a6",
        "C4_reference_history": "stage2_only",
    },
    {
        "seed": 2551,
        "route_group": "default_L3",
        "expected_route": "l3_core",
        "selected_layer": 3,
        "other_layer": 2,
        "intervention_start_step": 501,
        "intervention_end_step": 700,
        "C2_stop_step": 800,
        "C4_stop_step": 300,
        "start_checkpoint": "experiments/level7_5/formal/seed2551/level6_1.pt",
        "start_checkpoint_sha256": "0f9fff5aaa37ef952a295212eb14c966ddcc65115e0310dea64f9e941fee0125",
        "C2_reference": "experiments/level7_5/formal/seed2551/curriculum_stage1.pt",
        "C2_reference_sha256": "6e0b478978e7cbb39e3111a90edf64358b41d44986e044d96fd35317e3617100",
        "C4_reference": "experiments/level7_5/formal/seed2551/stage2/resume.pt",
        "C4_reference_sha256": "30f33a062d6e302c8e473598316f37df32b416755f60e8d01230864fedc78dad",
        "C4_reference_history": "stage2_only",
    },
    {
        "seed": 2909,
        "route_group": "default_L3",
        "expected_route": "l3_core",
        "selected_layer": 3,
        "other_layer": 2,
        "intervention_start_step": 501,
        "intervention_end_step": 700,
        "C2_stop_step": 800,
        "C4_stop_step": 400,
        "start_checkpoint": "experiments/level7_5/formal/seed2909/level6_1.pt",
        "start_checkpoint_sha256": "e52dc48b363fd4b3a90211db2cfbfe39f21269b2b25bd6e276bfe30b64071579",
        "C2_reference": "experiments/level7_5/formal/seed2909/curriculum_stage1.pt",
        "C2_reference_sha256": "9bedb9025827207586d15e064e30a0e8b9c6ef6510f0369697572784db28ad2f",
        "C4_reference": "experiments/level7_5/formal/seed2909/stage2/resume.pt",
        "C4_reference_sha256": "80bca482b6f354209f190e30b64c52e76c66e796e748ede34205461dede7b0c2",
        "C4_reference_history": "stage2_only",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--C2-batch-size", type=int, default=8)
    parser.add_argument("--C4-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--causal-chunks", type=int, default=16)
    parser.add_argument("--causal-samples", type=int, default=2048)
    parser.add_argument("--causal-eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=FORMAL_DATASET_SEED)
    parser.add_argument("--formed-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--core-retention-threshold", type=float, default=0.80)
    parser.add_argument("--pair-sufficiency-threshold", type=float, default=0.90)
    parser.add_argument("--pair-gain-threshold", type=float, default=0.03)
    parser.add_argument("--roll-drop-threshold", type=float, default=0.05)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--precursor-intact-threshold", type=float, default=0.75)
    parser.add_argument("--precursor-retention-threshold", type=float, default=0.70)
    parser.add_argument("--output", default="experiments/level7_5_3/formal")
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
        "C2_batch_size": 8,
        "C4_batch_size": 4,
        "learning_rate": 1e-3,
        "probe_weight": 0.5,
        "eval_every": 100,
        "eval_batches": 10,
        "eval_batch_size": 8,
        "causal_chunks": 16,
        "causal_samples": 2048,
        "causal_eval_batch_size": 16,
        "dataset_seed": FORMAL_DATASET_SEED,
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
        raise ValueError(f"Formal Level 7.5.3 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_5_3/formal":
        raise ValueError("Formal Level 7.5.3 output path is locked")


def validate_static_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("frozen_sources") != SOURCE_SPECS:
        raise RuntimeError("Static Level 7.5.3 sources changed")
    if protocol.get("registered_branches") != BRANCHES:
        raise RuntimeError("Static Level 7.5.3 branches changed")
    panel = protocol.get("fresh_endpoint_causal_panel", {})
    if panel.get("conditions") != CONDITIONS:
        raise RuntimeError("Static Level 7.5.3 condition order changed")
    if panel.get("dataset_seed") != FORMAL_DATASET_SEED:
        raise RuntimeError("Static Level 7.5.3 dataset seed changed")


def reference_history(state: dict[str, Any], stage: int) -> list[dict[str, Any]]:
    return [row for row in state.get("history", []) if int(row.get("stage", stage)) == stage]


def validate_sources(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_static_protocol(protocol)
    if not PARENT_RESULT.is_file() or sha256_file(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise RuntimeError("Frozen Level 7.5.2 parent result changed")
    parent = read_json(PARENT_RESULT)
    if not parent["integrity"]["passed"]:
        raise RuntimeError("Level 7.5.2 parent integrity failed")
    if parent["diagnosis"]["classification"] != "weak_L2_precursor_partially_replicated":
        raise RuntimeError("Unexpected Level 7.5.2 parent classification")
    level75 = read_json(ROOT / "experiments/level7_5/formal/result.json")
    default_routes = {
        int(row["seed"]): row["diagnosis"]["endpoint_route"]
        for row in level75["runs"]
    }
    level741 = read_json(ROOT / "experiments/level7_4_1/formal/result.json")
    audit = []
    for spec in SOURCE_SPECS:
        paths_and_hashes = [
            ("start", spec["start_checkpoint"], spec["start_checkpoint_sha256"]),
            ("C2", spec["C2_reference"], spec["C2_reference_sha256"]),
            ("C4", spec["C4_reference"], spec["C4_reference_sha256"]),
        ]
        observed = {}
        for label, relative, expected_hash in paths_and_hashes:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = sha256_file(path)
            if digest != expected_hash:
                raise RuntimeError(f"{label} source hash changed: seed={spec['seed']}")
            observed[label] = digest
        c2 = torch.load(ROOT / spec["C2_reference"], map_location="cpu", weights_only=False)
        c4 = torch.load(ROOT / spec["C4_reference"], map_location="cpu", weights_only=False)
        if spec["seed"] == 1879:
            prior_route = level741["diagnosis"]["final_route_class"]
            c4_history = reference_history(c4, 2)
        else:
            prior_route = default_routes[spec["seed"]]
            c4_history = c4["history"]
        if prior_route != spec["expected_route"]:
            raise RuntimeError(f"Expected route changed: seed={spec['seed']}")
        c2_history = reference_history(c2, 1)
        if c2_history[-1]["step"] != spec["C2_stop_step"]:
            raise RuntimeError(f"C2 stop mismatch: seed={spec['seed']}")
        if c4_history[-1]["step"] != spec["C4_stop_step"]:
            raise RuntimeError(f"C4 stop mismatch: seed={spec['seed']}")
        audit.append(
            {
                **spec,
                "observed_hashes": observed,
                "prior_route": prior_route,
                "source_validation_passed": True,
                "C2_reference_model_fingerprint": state_dict_fingerprint(c2["model"]),
                "C2_reference_probe_fingerprint": state_dict_fingerprint(c2["probe"]),
                "C2_reference_optimizer_fingerprint": canonical_fingerprint(c2["optimizer"]),
                "C2_reference_CPU_RNG_fingerprint": canonical_fingerprint(c2["cpu_rng"]),
                "C2_reference_CUDA_RNG_fingerprint": canonical_fingerprint(c2["cuda_rng"]),
                "C2_reference_history": c2_history,
                "C4_reference_model_fingerprint": state_dict_fingerprint(c4["model"]),
                "C4_reference_probe_fingerprint": state_dict_fingerprint(c4["probe"]),
                "C4_reference_optimizer_fingerprint": canonical_fingerprint(c4["optimizer"]),
                "C4_reference_CPU_RNG_fingerprint": canonical_fingerprint(c4["cpu_rng"]),
                "C4_reference_CUDA_RNG_fingerprint": canonical_fingerprint(c4["cuda_rng"]),
                "C4_reference_history_rows": c4_history,
            }
        )
        del c2, c4
    return audit, {
        "result": str(PARENT_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": PARENT_RESULT_SHA256,
        "integrity_passed": parent["integrity"]["passed"],
        "classification": parent["diagnosis"]["classification"],
    }


def suppression_layer(spec: dict[str, Any], branch: dict[str, str]) -> int | None:
    if branch["suppression_role"] == "selected":
        return int(spec["selected_layer"])
    if branch["suppression_role"] == "other":
        return int(spec["other_layer"])
    return None


def suppress_memory(memory: list[torch.Tensor], layer: int) -> list[torch.Tensor]:
    index = layer - 1
    if index < 0 or index >= len(memory):
        raise ValueError(f"Invalid suppression layer: {layer}")
    return [item * 0.0 if position == index else item for position, item in enumerate(memory)]


def intervened_random_step(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    count: int,
    batch: int,
    weight: float,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None,
) -> None:
    if layer is None:
        random_step(model, probe, optimizer, args, count, batch, weight, device, dtype)
        return
    chunks, target, position = make_chunks(batch, count, args.chunk_size, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        memory = None
        first_logits = None
        probes = []
        for chunk_index in range(chunks.size(1)):
            logits, produced = model(
                chunks[:, chunk_index],
                memory=memory,
                return_memory=True,
                per_layer_memory=True,
            )
            if chunk_index == 0:
                first_logits = logits
            memory = suppress_memory(produced, layer)
            probes.append(probe(vector(memory)))
        rows = torch.arange(batch, device=device)
        query_loss = F.cross_entropy(logits[:, -1, :16], target)
        local_loss = F.cross_entropy(first_logits[rows, position, :16], target)
        probe_loss = torch.stack(
            [F.cross_entropy(item, target) for item in probes]
        ).mean()
        loss = (
            query_loss
            + 0.5 * local_loss
            + weight * probe_loss
            + 0.1 * model.memory_diversity_loss()
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(probe.parameters()), 1.0
    )
    optimizer.step()


def save_resume(
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


def exact_gate(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reference_path: Path,
    history: list[dict[str, Any]],
    reference_history_rows: list[dict[str, Any]],
    observed_step: int,
    expected_step: int,
) -> dict[str, Any]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    current_model = model_fingerprint(model)
    current_probe = state_dict_fingerprint(probe.state_dict())
    current_optimizer = canonical_fingerprint(optimizer.state_dict())
    current_cpu = torch.get_rng_state().cpu()
    current_cuda = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    comparisons = {
        "model_state_exact": current_model == state_dict_fingerprint(reference["model"]),
        "probe_state_exact": current_probe == state_dict_fingerprint(reference["probe"]),
        "optimizer_state_exact": current_optimizer == canonical_fingerprint(reference["optimizer"]),
        "CPU_RNG_exact": torch.equal(current_cpu, reference["cpu_rng"].cpu()),
        "CUDA_RNG_exact": rng_equal(current_cuda, reference["cuda_rng"]),
        "validation_history_exact": history == reference_history_rows,
        "stop_step_exact": observed_step == expected_step,
    }
    result = {
        "passed": all(comparisons.values()),
        "comparisons": comparisons,
        "observed_step": observed_step,
        "expected_step": expected_step,
        "model_fingerprint": current_model,
        "probe_fingerprint": current_probe,
        "optimizer_fingerprint": current_optimizer,
        "CPU_RNG_fingerprint": canonical_fingerprint(current_cpu),
        "CUDA_RNG_fingerprint": canonical_fingerprint(current_cuda),
    }
    del reference
    return result


def branch_root(root: Path, seed: int, branch_name: str) -> Path:
    return root / f"seed{seed}" / branch_name


def run_branch(
    spec: dict[str, Any],
    source_audit: dict[str, Any],
    branch: dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    folder = branch_root(root, spec["seed"], branch["name"])
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "training_result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
    )
    resume_path = folder / "resume.pt"
    if resume_path.exists() and not args.force:
        state = restore(resume_path, model, probe, optimizer, device)
        current_stage = int(state["current_stage"])
        current_step = int(state["current_step"])
        c2_history = state["C2_history"]
        c4_history = state["C4_history"]
        active_steps = int(state["suppression_active_steps"])
        print(
            f"seed={spec['seed']} branch={branch['name']} resumed "
            f"stage={current_stage} step={current_step}",
            flush=True,
        )
    else:
        restore(ROOT / spec["start_checkpoint"], model, probe, optimizer, device)
        set_seed(int(spec["seed"]) + 20000)
        current_stage = 1
        current_step = 0
        c2_history = []
        c4_history = []
        active_steps = 0
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    layer = suppression_layer(spec, branch)
    eval_args = argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    c2_gate_path = folder / "C2_exact_gate.json"
    if current_stage == 1:
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        for step in range(current_step + 1, int(spec["C2_stop_step"]) + 1):
            model.train()
            probe.train()
            active = bool(
                layer is not None
                and int(spec["intervention_start_step"]) <= step
                <= int(spec["intervention_end_step"])
            )
            intervened_random_step(
                model,
                probe,
                optimizer,
                args,
                2,
                args.C2_batch_size,
                args.probe_weight,
                device,
                dtype,
                layer if active else None,
            )
            active_steps += int(active)
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(model, probe, eval_args, 2, device, dtype)
                c2_history.append({"stage": 1, "step": step, **metric})
                save_resume(
                    resume_path,
                    model,
                    probe,
                    optimizer,
                    {
                        "seed": spec["seed"],
                        "branch": branch["name"],
                        "current_stage": 1,
                        "current_step": step,
                        "C2_history": c2_history,
                        "C4_history": c4_history,
                        "suppression_active_steps": active_steps,
                    },
                )
                print(
                    f"seed={spec['seed']} branch={branch['name']} C2 step={step} "
                    f"active={active} query={metric['query']:.2%} "
                    f"probe={metric['probe_min']:.2%}",
                    flush=True,
                )
        if branch["name"] == "intact_replay":
            c2_gate = exact_gate(
                model,
                probe,
                optimizer,
                ROOT / spec["C2_reference"],
                c2_history,
                source_audit["C2_reference_history"],
                int(spec["C2_stop_step"]),
                int(spec["C2_stop_step"]),
            )
        else:
            c2_gate = {"passed": None, "not_applicable_to_counterfactual": True}
        atomic_save(c2_gate_path, c2_gate)
        save_resume(
            folder / "C2_endpoint.pt",
            model,
            probe,
            optimizer,
            {
                "seed": spec["seed"],
                "branch": branch["name"],
                "C2_history": c2_history,
                "suppression_active_steps": active_steps,
            },
        )
        current_stage = 2
        current_step = 0
        save_resume(
            resume_path,
            model,
            probe,
            optimizer,
            {
                "seed": spec["seed"],
                "branch": branch["name"],
                "current_stage": 2,
                "current_step": 0,
                "C2_history": c2_history,
                "C4_history": c4_history,
                "suppression_active_steps": active_steps,
            },
        )
    if current_stage == 2:
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        for step in range(current_step + 1, int(spec["C4_stop_step"]) + 1):
            model.train()
            probe.train()
            intervened_random_step(
                model,
                probe,
                optimizer,
                args,
                4,
                args.C4_batch_size,
                args.probe_weight,
                device,
                dtype,
                None,
            )
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(model, probe, eval_args, 4, device, dtype)
                c4_history.append({"stage": 2, "step": step, **metric})
                save_resume(
                    resume_path,
                    model,
                    probe,
                    optimizer,
                    {
                        "seed": spec["seed"],
                        "branch": branch["name"],
                        "current_stage": 2,
                        "current_step": step,
                        "C2_history": c2_history,
                        "C4_history": c4_history,
                        "suppression_active_steps": active_steps,
                    },
                )
                print(
                    f"seed={spec['seed']} branch={branch['name']} C4 step={step} "
                    f"query={metric['query']:.2%} probe={metric['probe_min']:.2%}",
                    flush=True,
                )
    if branch["name"] == "intact_replay":
        c4_gate = exact_gate(
            model,
            probe,
            optimizer,
            ROOT / spec["C4_reference"],
            c4_history,
            source_audit["C4_reference_history_rows"],
            int(spec["C4_stop_step"]),
            int(spec["C4_stop_step"]),
        )
    else:
        c4_gate = {"passed": None, "not_applicable_to_counterfactual": True}
    atomic_save(folder / "C4_exact_gate.json", c4_gate)
    endpoint_path = folder / "C4_endpoint.pt"
    save_resume(
        endpoint_path,
        model,
        probe,
        optimizer,
        {
            "seed": spec["seed"],
            "branch": branch["name"],
            "C2_history": c2_history,
            "C4_history": c4_history,
            "suppression_active_steps": active_steps,
        },
    )
    expected_active = (
        int(spec["intervention_end_step"])
        - int(spec["intervention_start_step"])
        + 1
        if layer is not None
        else 0
    )
    result = {
        "seed": spec["seed"],
        "branch": branch["name"],
        "suppression_role": branch["suppression_role"],
        "suppressed_layer": layer,
        "intervention_start_step": spec["intervention_start_step"],
        "intervention_end_step": spec["intervention_end_step"],
        "suppression_active_steps": active_steps,
        "expected_suppression_active_steps": expected_active,
        "suppression_count_exact": active_steps == expected_active,
        "C4_suppression_active_steps": 0,
        "C2_stop_step": spec["C2_stop_step"],
        "C4_stop_step": spec["C4_stop_step"],
        "C2_history": c2_history,
        "C4_history": c4_history,
        "C2_exact_gate": read_json(c2_gate_path),
        "C4_exact_gate": c4_gate,
        "endpoint_checkpoint": str(endpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "endpoint_checkpoint_sha256": sha256_file(endpoint_path),
        "training_complete": True,
        "training_beyond_registered_C4": False,
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


def validate_resumed_metrics(
    metrics: dict[str, dict[str, Any]], args: argparse.Namespace
) -> None:
    if set(metrics) - set(CONDITIONS):
        raise RuntimeError("Unexpected resumed endpoint condition")
    for name, metric in metrics.items():
        if metric.get("condition") != name:
            raise RuntimeError(f"Resumed condition mismatch: {name}")
        if metric.get("samples") != args.causal_samples:
            raise RuntimeError(f"Resumed sample mismatch: {name}")


def run_endpoint_panel(
    training: dict[str, Any],
    spec: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    folder = branch_root(root, spec["seed"], training["branch"]) / "causal"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    checkpoint = ROOT / training["endpoint_checkpoint"]
    if sha256_file(checkpoint) != training["endpoint_checkpoint_sha256"]:
        raise RuntimeError("Counterfactual endpoint checkpoint changed")
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
    if before != expected_fingerprint:
        raise RuntimeError("Endpoint model fingerprint mismatch")
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
            f"seed={spec['seed']} branch={training['branch']} "
            f"condition={condition} query={metric['query']:.2%} "
            f"local={metric['local']:.2%}",
            flush=True,
        )
    after = model_fingerprint(model)
    profile = profile_checkpoint(metrics, args)
    integrity = {
        "endpoint_checkpoint_sha256": training["endpoint_checkpoint_sha256"],
        "expected_model_fingerprint": expected_fingerprint,
        "model_fingerprint_before": before,
        "model_fingerprint_after": after,
        "model_fingerprint_unchanged": before == after,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "all_conditions_in_registered_order": list(metrics) == CONDITIONS,
        "fixed_samples_every_condition": all(
            row["samples"] == args.causal_samples for row in metrics.values()
        ),
        "shared_dataset_seed": args.dataset_seed,
    }
    integrity["passed"] = bool(
        integrity["model_fingerprint_unchanged"]
        and integrity["all_parameters_frozen"]
        and integrity["all_conditions_in_registered_order"]
        and integrity["fixed_samples_every_condition"]
    )
    result = {
        "seed": spec["seed"],
        "branch": training["branch"],
        "suppressed_layer": training["suppressed_layer"],
        "expected_route": spec["expected_route"],
        "metrics": metrics,
        "profile": profile,
        "integrity": integrity,
    }
    atomic_save(result_path, result)
    del model
    torch.cuda.empty_cache()
    return result


def diagnose_seed(
    spec: dict[str, Any],
    training_rows: list[dict[str, Any]],
    panels: list[dict[str, Any]],
) -> dict[str, Any]:
    training = {row["branch"]: row for row in training_rows}
    causal = {row["branch"]: row for row in panels}
    control_exact = bool(
        training["intact_replay"]["C2_exact_gate"]["passed"]
        and training["intact_replay"]["C4_exact_gate"]["passed"]
    )
    routes = {name: row["profile"]["route_class"] for name, row in causal.items()}
    intact_query = {
        name: row["metrics"]["intact"]["query"] for name, row in causal.items()
    }
    expected = spec["expected_route"]
    selected_changed = routes["selected_layer_suppression"] != expected
    other_preserved = routes["other_layer_suppression"] == expected
    specific_effect = selected_changed and other_preserved
    return {
        "seed": spec["seed"],
        "route_group": spec["route_group"],
        "expected_route": expected,
        "selected_layer": spec["selected_layer"],
        "other_layer": spec["other_layer"],
        "control_exact_replay_passed": control_exact,
        "routes": routes,
        "intact_query": intact_query,
        "selected_layer_route_changed": selected_changed,
        "other_layer_route_preserved": other_preserved,
        "layer_specific_commitment_effect": specific_effect,
        "selected_layer_query_drop_from_control": (
            intact_query["intact_replay"]
            - intact_query["selected_layer_suppression"]
        ),
        "other_layer_query_drop_from_control": (
            intact_query["intact_replay"]
            - intact_query["other_layer_suppression"]
        ),
    }


def diagnose_cohort(
    seed_diagnoses: list[dict[str, Any]], integrity_passed: bool
) -> dict[str, Any]:
    by_seed = {row["seed"]: row for row in seed_diagnoses}
    exceptional_specific = by_seed[1879]["layer_specific_commitment_effect"]
    default_specific = sum(
        by_seed[seed]["layer_specific_commitment_effect"]
        for seed in DEFAULT_L3_SEEDS
    )
    selected_changes = sum(
        row["selected_layer_route_changed"] for row in seed_diagnoses
    )
    other_changes = sum(
        not row["other_layer_route_preserved"] for row in seed_diagnoses
    )
    if not integrity_passed:
        classification = "formal_integrity_failed_causal_interpretation_closed"
    elif exceptional_specific and default_specific >= 2:
        classification = "layer_specific_route_commitment_confirmed"
    elif exceptional_specific or default_specific >= 1:
        classification = "partial_layer_specific_route_commitment"
    elif selected_changes == 0:
        classification = "routes_recover_after_transient_selected_layer_suppression"
    elif other_changes > 0:
        classification = "transient_suppression_disrupts_routes_nonspecifically"
    else:
        classification = "heterogeneous_route_commitment_effects"
    return {
        "classification": classification,
        "seed1879_layer_specific_effect": exceptional_specific,
        "default_L3_layer_specific_effect_count": default_specific,
        "default_L3_layer_specific_effect_expected_for_confirmation": 2,
        "selected_layer_route_change_count": selected_changes,
        "other_layer_route_change_count": other_changes,
        "registered_stop_boundary": (
            "Report all twelve fixed branches; do not change intervention windows, "
            "layers, compute budgets, endpoint thresholds, or panel size."
        ),
    }


def build_integrity(
    source_audit: list[dict[str, Any]],
    training: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    controls = [row for row in training if row["branch"] == "intact_replay"]
    interventions = [row for row in training if row["branch"] != "intact_replay"]
    integrity = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "parent_result_sha256": sha256_file(PARENT_RESULT),
        "all_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "all_intact_C2_and_C4_exact_gates_passed": all(
            row["C2_exact_gate"]["passed"] and row["C4_exact_gate"]["passed"]
            for row in controls
        ),
        "expected_training_branches": len(SOURCE_SPECS) * len(BRANCHES),
        "completed_training_branches": len(training),
        "all_suppression_counts_exact": all(
            row["suppression_count_exact"] for row in training
        ),
        "all_interventions_exactly_200_C2_steps": all(
            row["suppression_active_steps"] == 200 for row in interventions
        ),
        "no_C4_suppression": all(
            row["C4_suppression_active_steps"] == 0 for row in training
        ),
        "no_training_beyond_registered_C4": all(
            not row["training_beyond_registered_C4"] for row in training
        ),
        "expected_endpoint_panels": len(SOURCE_SPECS) * len(BRANCHES),
        "completed_endpoint_panels": len(panels),
        "all_endpoint_panel_integrity_passed": all(
            row["integrity"]["passed"] for row in panels
        ),
        "all_conditions_exact": all(
            list(row["metrics"]) == CONDITIONS for row in panels
        ),
        "fixed_N_completed": all(
            metric["samples"] == args.causal_samples
            for row in panels
            for metric in row["metrics"].values()
        ),
        "shared_fresh_dataset_seed": args.dataset_seed,
        "seed909_used": False,
    }
    integrity["passed"] = bool(
        integrity["all_source_hashes_validated"]
        and integrity["all_intact_C2_and_C4_exact_gates_passed"]
        and integrity["expected_training_branches"]
        == integrity["completed_training_branches"]
        and integrity["all_suppression_counts_exact"]
        and integrity["all_interventions_exactly_200_C2_steps"]
        and integrity["no_C4_suppression"]
        and integrity["no_training_beyond_registered_C4"]
        and integrity["expected_endpoint_panels"]
        == integrity["completed_endpoint_panels"]
        and integrity["all_endpoint_panel_integrity_passed"]
        and integrity["all_conditions_exact"]
        and integrity["fixed_N_completed"]
        and not integrity["seed909_used"]
    )
    return integrity


def plot_results(seed_diagnoses: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["seed"]) for row in seed_diagnoses]
    branches = [
        ("intact_replay", "intact replay", "#333333"),
        ("selected_layer_suppression", "selected layer suppressed", "#d1495b"),
        ("other_layer_suppression", "other layer suppressed", "#0077b6"),
    ]
    x = list(range(len(labels)))
    width = 0.24
    fig, axis = plt.subplots(figsize=(13, 7))
    for index, (key, label, color) in enumerate(branches):
        values = [100 * row["intact_query"][key] for row in seed_diagnoses]
        positions = [item + (index - 1) * width for item in x]
        bars = axis.bar(positions, values, width, label=label, color=color)
        for bar, row in zip(bars, seed_diagnoses):
            route = row["routes"][key]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                route.replace("_", "\n"),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axis.axhline(90, color="#666666", linestyle="--", linewidth=1.5)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 116)
    axis.set_ylabel("Fresh 16-chunk query accuracy (%)")
    axis.set_xlabel("Model seed")
    axis.set_title("Level 7.5.3 route-commitment counterfactual endpoints")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def smoke_state(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, str]:
    return {
        "model": model_fingerprint(model),
        "probe": state_dict_fingerprint(probe.state_dict()),
        "optimizer": canonical_fingerprint(optimizer.state_dict()),
        "CPU_RNG": canonical_fingerprint(torch.get_rng_state().cpu()),
        "CUDA_RNG": canonical_fingerprint(
            [item.cpu() for item in torch.cuda.get_rng_state_all()]
        ),
    }


def run_smoke(args: argparse.Namespace) -> int:
    protocol = read_json(STATIC_PREREGISTRATION)
    source_audit, parent_audit = validate_sources(protocol)
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    spec = SOURCE_SPECS[0]

    def one_step(layer: int | None, direct_baseline: bool) -> dict[str, str]:
        model, probe = build(device, args.chunk_size)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(probe.parameters()), lr=args.learning_rate
        )
        restore(ROOT / spec["start_checkpoint"], model, probe, optimizer, device)
        set_seed(7539999)
        model.train()
        probe.train()
        if direct_baseline:
            random_step(model, probe, optimizer, args, 2, 4, args.probe_weight, device, dtype)
        else:
            intervened_random_step(
                model, probe, optimizer, args, 2, 4, args.probe_weight, device, dtype, layer
            )
        value = smoke_state(model, probe, optimizer)
        del model, probe, optimizer
        torch.cuda.empty_cache()
        return value

    baseline = one_step(None, True)
    no_op = one_step(None, False)
    suppressed = one_step(2, False)
    tensors = [torch.randn(2, 3, 4, device=device) for _ in range(3)]
    masked = suppress_memory(tensors, 2)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "all_source_hashes_validated": all(
            row["source_validation_passed"] for row in source_audit
        ),
        "parent_audit": parent_audit,
        "no_op_training_step_exact": baseline == no_op,
        "selected_layer_suppression_changes_training": baseline != suppressed,
        "selected_layer_zeroed": bool(torch.count_nonzero(masked[1]).item() == 0),
        "other_layers_unchanged": bool(
            torch.equal(masked[0], tensors[0]) and torch.equal(masked[2], tensors[2])
        ),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "all_source_hashes_validated",
            "no_op_training_step_exact",
            "selected_layer_suppression_changes_training",
            "selected_layer_zeroed",
            "other_layers_unchanged",
        )
    )
    smoke_root = ROOT / "experiments/level7_5_3/smoke"
    atomic_save(smoke_root / "result.json", result)
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
    audit_by_seed = {int(row["seed"]): row for row in source_audit}
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
        "stage": "counterfactual_training",
        "completed_training_branches": [],
        "completed_endpoint_panels": [],
        "active": None,
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    training_rows = []
    panels = []
    seed_diagnoses = []
    for spec in SOURCE_SPECS:
        seed_training = []
        control = run_branch(
            spec, audit_by_seed[spec["seed"]], BRANCHES[0], args, device, dtype, root
        )
        seed_training.append(control)
        training_rows.append(control)
        progress["completed_training_branches"].append(
            f"seed{spec['seed']}/intact_replay"
        )
        atomic_save(root / "progress.json", progress)
        if not (
            control["C2_exact_gate"]["passed"]
            and control["C4_exact_gate"]["passed"]
        ):
            raise RuntimeError(
                f"Exact intact replay failed; counterfactual panel closed: seed={spec['seed']}"
            )
        for branch in BRANCHES[1:]:
            progress["active"] = f"seed{spec['seed']}/{branch['name']}"
            atomic_save(root / "progress.json", progress)
            row = run_branch(
                spec,
                audit_by_seed[spec["seed"]],
                branch,
                args,
                device,
                dtype,
                root,
            )
            seed_training.append(row)
            training_rows.append(row)
            progress["completed_training_branches"].append(progress["active"])
            atomic_save(root / "progress.json", progress)
        seed_panels = []
        for row in seed_training:
            progress["stage"] = "endpoint_causal_panels"
            progress["active"] = f"seed{spec['seed']}/{row['branch']}"
            atomic_save(root / "progress.json", progress)
            panel = run_endpoint_panel(row, spec, args, device, dtype, root)
            seed_panels.append(panel)
            panels.append(panel)
            progress["completed_endpoint_panels"].append(progress["active"])
            atomic_save(root / "progress.json", progress)
        seed_diagnoses.append(diagnose_seed(spec, seed_training, seed_panels))
    integrity = build_integrity(source_audit, training_rows, panels, args)
    diagnosis = diagnose_cohort(seed_diagnoses, integrity["passed"])
    elapsed = time.perf_counter() - started
    result = {
        "protocol": protocol,
        "parent_audit": parent_audit,
        "source_audit": source_audit,
        "training_branches": training_rows,
        "endpoint_panels": panels,
        "seed_diagnoses": seed_diagnoses,
        "diagnosis": diagnosis,
        "integrity": integrity,
        "elapsed_seconds_this_invocation": elapsed,
    }
    summary = {
        "diagnosis": diagnosis,
        "integrity": integrity,
        "seeds": seed_diagnoses,
        "endpoints": [
            {
                "seed": row["seed"],
                "branch": row["branch"],
                "suppressed_layer": row["suppressed_layer"],
                "route_class": row["profile"]["route_class"],
                "intact_query": row["metrics"]["intact"]["query"],
                "minimum_local": row["profile"]["minimum_local"],
                "whole_memory_causal": row["profile"]["whole_memory_causal"],
                "layer_atlas": row["profile"]["layer_atlas"],
            }
            for row in panels
        ],
        "elapsed_seconds_this_invocation": elapsed,
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_results(seed_diagnoses, root / "route_commitment_counterfactual.png")
    atomic_save(
        root / "progress.json",
        {
            "stage": "complete",
            "completed_training_branches": len(training_rows),
            "completed_endpoint_panels": len(panels),
            "classification": diagnosis["classification"],
            "integrity_passed": integrity["passed"],
            "seed909_locked": True,
        },
    )
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
