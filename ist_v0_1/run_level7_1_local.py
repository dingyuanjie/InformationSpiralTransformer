"""Level 7.1 independent-initialization persistent-Memory replication."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from long_context_test import set_seed
from run_level6_2_local import evaluate, make_chunks
from run_level6_6_local import (
    build,
    checkpoint,
    curriculum,
    fixed_stage,
    random_step,
    restore,
)
from run_level6_18_6_local import configure_cuda


FORMAL_SEEDS = [1217, 1429]
SMOKE_SEED = 17
CONDITIONS = [
    "intact",
    "reset_all",
    "zero_all",
    "batch_roll_all",
    "zero_l3",
    "batch_roll_l3",
    "keep_l3",
]
DISRUPTED = [
    "reset_all",
    "zero_all",
    "batch_roll_all",
    "zero_l3",
    "batch_roll_l3",
]
ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_1"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--fixed-steps", type=int, default=2500)
    parser.add_argument("--fixed-batch-size", type=int, default=16)
    parser.add_argument("--fixed-eval-batch-size", type=int, default=16)
    parser.add_argument("--random-stage1-steps", type=int, default=2500)
    parser.add_argument("--later-steps", type=int, default=1500)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)
    parser.add_argument("--stage4-lr", type=float, default=1e-5)
    parser.add_argument("--withdrawal-lr", type=float, default=5e-6)
    parser.add_argument("--maintenance-steps", type=int, default=750)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--formation-eval-samples", type=int, default=800)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--diagnostic-batch-size", type=int, default=16)
    parser.add_argument("--probe-train-samples", type=int, default=2048)
    parser.add_argument("--probe-val-samples", type=int, default=512)
    parser.add_argument("--diagnostic-samples", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-errors", type=int, default=100)
    parser.add_argument("--intact-query-threshold", type=float, default=0.95)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--keep-l3-threshold", type=float, default=0.90)
    parser.add_argument("--memory-error-threshold", type=float, default=0.75)
    parser.add_argument("--read-gap-threshold", type=float, default=0.15)
    parser.add_argument("--eval-seed-base", type=int, default=7100000)
    parser.add_argument("--probe-seed-base", type=int, default=7110000)
    parser.add_argument("--output", default="experiments/level7_1/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.behavior_only_gate = True
    return args


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    # A fixed ``progress.json.tmp`` is easy for antivirus/indexing processes to
    # catch between the write and rename on Windows.  Once that happens,
    # os.replace can fail with WinError 5 even though neither file is read-only.
    # Use a unique sibling and retry the atomic rename for transient readers.
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(serialized, encoding="utf-8")
    replace_error: PermissionError | None = None
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            replace_error = error
            time.sleep(min(0.05 * (attempt + 1), 0.5))

    # Some Windows readers allow writes but do not share delete access, which
    # makes every rename fail.  Progress/result files are still recoverable from
    # the completed per-branch artifacts, so prefer a direct overwrite after the
    # bounded atomic retries instead of aborting a multi-hour experiment.
    for attempt in range(10):
        try:
            path.write_text(serialized, encoding="utf-8")
            temporary.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(min(0.1 * (attempt + 1), 0.5))
    if replace_error is not None:
        raise replace_error
    raise RuntimeError(f"failed to save JSON: {path}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "seeds": FORMAL_SEEDS,
        "chunk_size": 128,
        "fixed_steps": 2500,
        "fixed_batch_size": 16,
        "fixed_eval_batch_size": 16,
        "random_stage1_steps": 2500,
        "later_steps": 1500,
        "probe_weight": 0.5,
        "stage3_lr": 5e-5,
        "stage4_lr": 1e-5,
        "withdrawal_lr": 5e-6,
        "maintenance_steps": 750,
        "eval_every": 100,
        "eval_batch_size": 8,
        "eval_batches": 10,
        "final_eval_batches": 50,
        "formation_eval_samples": 800,
        "causal_samples": 1024,
        "diagnostic_batch_size": 16,
        "probe_train_samples": 2048,
        "probe_val_samples": 512,
        "diagnostic_samples": 4096,
        "probe_batch_size": 128,
        "probe_epochs": 40,
        "patience": 6,
        "probe_lr": 1e-3,
        "probe_weight_decay": 1e-4,
        "minimum_errors": 100,
        "intact_query_threshold": 0.95,
        "local_threshold": 0.90,
        "disruption_threshold": 0.20,
        "keep_l3_threshold": 0.90,
        "memory_error_threshold": 0.75,
        "read_gap_threshold": 0.15,
        "eval_seed_base": 7100000,
        "probe_seed_base": 7110000,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.1 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_1/formal":
        raise ValueError("Formal output path is locked")


def resumable_withdrawal(
    model: nn.Module,
    probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    folder: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_args = argparse.Namespace(
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
    )
    schedule = [(0.2, 300), (0.1, 300), (0.0, args.maintenance_steps)]
    history: list[dict[str, Any]] = []
    completed_phase = 0
    if not args.force:
        for phase in range(len(schedule), 0, -1):
            path = folder / f"withdrawal_phase{phase}.pt"
            if path.exists():
                state = restore(path, model, probe, optimizer, device)
                history = state.get("withdrawal_history", [])
                completed_phase = phase
                print(f"seed={seed} resumed withdrawal phase={phase}", flush=True)
                break

    for group in optimizer.param_groups:
        group["lr"] = args.withdrawal_lr
    for phase_index in range(completed_phase, len(schedule)):
        phase = phase_index + 1
        weight, steps = schedule[phase_index]
        for parameter in probe.parameters():
            parameter.requires_grad_(weight > 0)
        for step in range(1, steps + 1):
            model.train()
            probe.train(weight > 0)
            random_step(
                model, probe, optimizer, args, 16, 2, weight, device, dtype
            )
            if step == 1 or step % args.eval_every == 0:
                metric = evaluate(model, probe, eval_args, 16, device, dtype)
                history.append(
                    {"phase": phase, "weight": weight, "step": step, **metric}
                )
                atomic_save(folder / "withdrawal_progress.json", history)
                print(
                    f"seed={seed} withdraw={weight} step={step} "
                    f"query={metric['query']:.2%}",
                    flush=True,
                )
        checkpoint(
            folder / f"withdrawal_phase{phase}.pt",
            model,
            probe,
            optimizer,
            {"withdrawal_history": history},
        )

    set_seed(args.eval_seed_base + seed * 10 + 1)
    final_batches = max(
        1, args.formation_eval_samples // args.eval_batch_size
    )
    final = evaluate(
        model, probe, eval_args, 16, device, dtype, batches=final_batches
    )
    return history, final


def train_independent_seed(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    folder: Path,
) -> dict[str, Any]:
    result_path = folder / "training_result.json"
    final_checkpoint = folder / "withdrawal_phase3.pt"
    if result_path.exists() and not args.force:
        result = read_json(result_path)
        if result.get("passed") and not final_checkpoint.exists():
            raise FileNotFoundError(final_checkpoint)
        return result

    folder.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=1e-3
    )
    started = time.perf_counter()
    fixed = fixed_stage(model, probe, optimizer, args, device, dtype, folder, seed)
    if not fixed["passed"]:
        result = {
            "seed": seed,
            "passed": False,
            "failed_phase": "fixed",
            "fixed": fixed,
            "seconds": time.perf_counter() - started,
            "budget_extended": False,
        }
        atomic_save(result_path, result)
        return result

    set_seed(seed + 20000)
    stages, history = curriculum(
        model, probe, optimizer, args, device, dtype, folder, seed
    )
    curriculum_passed = len(stages) == 4 and all(
        stage["passed"] for stage in stages
    )
    if not curriculum_passed:
        result = {
            "seed": seed,
            "passed": False,
            "failed_phase": "curriculum",
            "fixed": fixed,
            "stages": stages,
            "curriculum_history": history,
            "seconds": time.perf_counter() - started,
            "budget_extended": False,
        }
        atomic_save(result_path, result)
        return result

    withdrawal_history, final = resumable_withdrawal(
        model, probe, optimizer, args, device, dtype, folder, seed
    )
    passed = final["query"] >= args.intact_query_threshold
    result = {
        "seed": seed,
        "passed": passed,
        "failed_phase": None if passed else "withdrawal",
        "fixed": fixed,
        "stages": stages,
        "curriculum_history": history,
        "withdrawal_history": withdrawal_history,
        "final": final,
        "seconds": time.perf_counter() - started,
        "budget_extended": False,
    }
    atomic_save(result_path, result)
    return result


def intervene_memory(
    memory: list[torch.Tensor], condition: str
) -> list[torch.Tensor] | None:
    if condition == "intact":
        return memory
    if condition == "reset_all":
        return None
    if condition == "zero_all":
        return [torch.zeros_like(item) for item in memory]
    if condition == "batch_roll_all":
        return [item.roll(1, dims=0) for item in memory]
    if condition == "zero_l3":
        return [*memory[:-1], torch.zeros_like(memory[-1])]
    if condition == "batch_roll_l3":
        return [*memory[:-1], memory[-1].roll(1, dims=0)]
    if condition == "keep_l3":
        return [
            *[torch.zeros_like(item) for item in memory[:-1]],
            memory[-1],
        ]
    raise ValueError(condition)


@torch.no_grad()
def evaluate_causal_condition(
    model: nn.Module,
    args: argparse.Namespace,
    chunks_count: int,
    condition: str,
    samples: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    set_seed(seed)
    model.eval()
    query_correct = 0
    local_correct = 0
    total = 0
    while total < samples:
        batch = min(args.diagnostic_batch_size, samples - total)
        chunks, target, position = make_chunks(
            batch, chunks_count, args.chunk_size, device
        )
        memory = None
        first_logits = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for chunk_index in range(chunks_count):
                logits, produced = model(
                    chunks[:, chunk_index],
                    memory=memory,
                    return_memory=True,
                    per_layer_memory=True,
                )
                if chunk_index == 0:
                    first_logits = logits
                memory = intervene_memory(produced, condition)
        rows = torch.arange(batch, device=device)
        query_correct += (
            logits[:, -1, :16].argmax(-1) == target
        ).sum().item()
        local_correct += (
            first_logits[rows, position, :16].argmax(-1) == target
        ).sum().item()
        total += batch
    return {
        "condition": condition,
        "chunks": chunks_count,
        "samples": total,
        "query": query_correct / total,
        "local": local_correct / total,
    }


@torch.no_grad()
def collect_read_features(
    model: nn.Module,
    args: argparse.Namespace,
    samples: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    chunks_count: int = 16,
) -> dict[str, torch.Tensor]:
    set_seed(seed)
    model.eval()
    memory_parts = []
    hidden_parts = []
    label_parts = []
    prediction_parts = []
    captured: dict[str, torch.Tensor] = {}

    def capture_output_input(_module: nn.Module, inputs: tuple[torch.Tensor]) -> None:
        captured["hidden"] = inputs[0]

    handle = model.output.register_forward_pre_hook(capture_output_input)
    total = 0
    try:
        while total < samples:
            batch = min(args.diagnostic_batch_size, samples - total)
            chunks, target, _ = make_chunks(
                batch, chunks_count, args.chunk_size, device
            )
            memory = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(chunks_count):
                    logits, memory = model(
                        chunks[:, chunk_index],
                        memory=memory,
                        return_memory=True,
                        per_layer_memory=True,
                    )
            memory_parts.append(
                memory[-1].reshape(batch, -1).detach().cpu().to(torch.float16)
            )
            hidden_parts.append(
                captured["hidden"][:, -1].detach().cpu().to(torch.float16)
            )
            label_parts.append(target.cpu())
            prediction_parts.append(logits[:, -1, :16].argmax(-1).cpu())
            total += batch
            if total == batch or total % 512 == 0 or total == samples:
                print(f"read diagnostic {total}/{samples}", flush=True)
    finally:
        handle.remove()
    return {
        "memory_l3_concat": torch.cat(memory_parts),
        "query_hidden": torch.cat(hidden_parts),
        "labels": torch.cat(label_parts),
        "source_predictions": torch.cat(prediction_parts),
    }


def batched_probe_logits(
    probe: nn.Module,
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    probe.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(
                device, torch.float32
            )
            outputs.append(probe((batch - mean) / std).cpu())
    return torch.cat(outputs)


def fit_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    mean = train_x.float().mean(dim=0).to(device)
    std = train_x.float().std(dim=0).clamp_min(1e-4).to(device)
    probe = nn.Linear(train_x.shape[-1], 16).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    best_accuracy = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(probe.state_dict())
    patience = 0
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), args.probe_batch_size):
            ids = order[start : start + args.probe_batch_size]
            x = train_x[ids].to(device, torch.float32)
            y = train_y[ids].to(device)
            loss = F.cross_entropy(probe((x - mean) / std), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation_logits = batched_probe_logits(
            probe, val_x, mean, std, args.probe_batch_size, device
        )
        validation_accuracy = (
            validation_logits.argmax(-1) == val_y
        ).float().mean().item()
        if validation_accuracy > best_accuracy + 1e-5:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    probe.load_state_dict(best_state)
    probe.eval()
    return {
        "probe": probe,
        "mean": mean,
        "std": std,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
    }


def frozen_read_diagnostic(
    model: nn.Module,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    base = args.probe_seed_base + seed * 100
    train = collect_read_features(
        model, args, args.probe_train_samples, base + 1, device, dtype
    )
    validation = collect_read_features(
        model, args, args.probe_val_samples, base + 2, device, dtype
    )
    test = collect_read_features(
        model, args, args.diagnostic_samples, base + 3, device, dtype
    )
    fitted = {}
    test_predictions = {}
    for index, name in enumerate(("memory_l3_concat", "query_hidden")):
        fitted[name] = fit_linear_probe(
            train[name],
            train["labels"],
            validation[name],
            validation["labels"],
            args,
            device,
            base + 10 + index,
        )
        logits = batched_probe_logits(
            fitted[name]["probe"],
            test[name],
            fitted[name]["mean"],
            fitted[name]["std"],
            args.probe_batch_size,
            device,
        )
        test_predictions[name] = logits.argmax(-1)

    labels = test["labels"]
    source = test["source_predictions"]
    error_mask = source != labels
    error_count = int(error_mask.sum().item())
    metrics = {}
    for name in ("memory_l3_concat", "query_hidden"):
        predictions = test_predictions[name]
        metrics[name] = {
            "best_epoch": fitted[name]["best_epoch"],
            "validation_accuracy": fitted[name]["best_validation_accuracy"],
            "test_accuracy": (predictions == labels).float().mean().item(),
            "source_error_accuracy": (
                (predictions[error_mask] == labels[error_mask]).float().mean().item()
                if error_count
                else None
            ),
        }
    memory_error = metrics["memory_l3_concat"]["source_error_accuracy"]
    hidden_error = metrics["query_hidden"]["source_error_accuracy"]
    powered = error_count >= args.minimum_errors
    gap = memory_error - hidden_error if error_count else None
    replicated = bool(
        powered
        and memory_error >= args.memory_error_threshold
        and gap >= args.read_gap_threshold
    )
    return {
        "splits": {
            "train": args.probe_train_samples,
            "validation": args.probe_val_samples,
            "test": args.diagnostic_samples,
            "seeds": [base + 1, base + 2, base + 3],
            "unique": len({base + 1, base + 2, base + 3}) == 3,
        },
        "source_accuracy": (source == labels).float().mean().item(),
        "source_errors": error_count,
        "minimum_errors": args.minimum_errors,
        "powered": powered,
        "metrics": metrics,
        "memory_minus_query_hidden_on_errors": gap,
        "read_gap_replicated": replicated,
        "classification": (
            "read_gap_replicated"
            if replicated
            else "read_gap_underpowered"
            if not powered
            else "read_gap_not_generalized"
        ),
        "used_to_modify_or_gate_model": False,
    }


def evaluate_formed_seed(
    seed: int,
    training: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    folder: Path,
) -> dict[str, Any]:
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    if not training["passed"]:
        result = {
            "seed": seed,
            "training": training,
            "formed": False,
            "primary_passed": False,
            "causal": None,
            "read_diagnostic": None,
            "integrity": {
                "budget_extended": training.get("budget_extended", False),
                "seed_replaced": False,
                "seed909_used": False,
            },
        }
        atomic_save(result_path, result)
        return result

    checkpoint_path = folder / "withdrawal_phase3.pt"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, probe = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    probe.load_state_dict(state["probe"])
    for module in (model, probe):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    fingerprint_before = model_fingerprint(model)

    causal_rows = []
    causal_seed = args.eval_seed_base + seed * 10 + 2
    for condition in CONDITIONS:
        metric = evaluate_causal_condition(
            model,
            args,
            16,
            condition,
            args.causal_samples,
            causal_seed,
            device,
            dtype,
        )
        causal_rows.append(metric)
        print(
            f"seed={seed} causal={condition} query={metric['query']:.2%} "
            f"local={metric['local']:.2%}",
            flush=True,
        )
    by_condition = {row["condition"]: row for row in causal_rows}
    max_disrupted = max(by_condition[name]["query"] for name in DISRUPTED)
    min_local = min(row["local"] for row in causal_rows)
    causal_passed = bool(
        by_condition["intact"]["query"] >= args.intact_query_threshold
        and min_local >= args.local_threshold
        and max_disrupted <= args.disruption_threshold
        and by_condition["keep_l3"]["query"] >= args.keep_l3_threshold
    )
    causal = {
        "samples_per_condition": args.causal_samples,
        "paired_dataset_seed": causal_seed,
        "conditions": by_condition,
        "max_disrupted_query": max_disrupted,
        "minimum_local": min_local,
        "passed": causal_passed,
    }

    read_diagnostic = frozen_read_diagnostic(
        model, args, seed, device, dtype
    )
    fingerprint_after = model_fingerprint(model)
    integrity = {
        "model_fingerprint_before": fingerprint_before,
        "model_fingerprint_after": fingerprint_after,
        "model_fingerprint_unchanged": fingerprint_before == fingerprint_after,
        "all_model_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "budget_extended": training.get("budget_extended", False),
        "seed_replaced": False,
        "seed909_used": False,
        "old_checkpoint_used": False,
        "read_probe_used_to_modify_model": False,
    }
    integrity["passed"] = all(
        [
            integrity["model_fingerprint_unchanged"],
            integrity["all_model_parameters_frozen"],
            not integrity["budget_extended"],
            not integrity["seed_replaced"],
            not integrity["seed909_used"],
            not integrity["old_checkpoint_used"],
            not integrity["read_probe_used_to_modify_model"],
        ]
    )
    result = {
        "seed": seed,
        "training": training,
        "formed": True,
        "causal": causal,
        "read_diagnostic": read_diagnostic,
        "integrity": integrity,
        "primary_passed": bool(causal_passed and integrity["passed"]),
    }
    atomic_save(result_path, result)
    return result


def classify(results: list[dict[str, Any]]) -> dict[str, Any]:
    formed = [result for result in results if result["formed"]]
    primary = [result for result in results if result["primary_passed"]]
    formed_causal_failures = [
        result for result in formed if not result["primary_passed"]
    ]
    if formed_causal_failures:
        classification = "causal_replication_failed"
    elif len(primary) == len(FORMAL_SEEDS):
        classification = "strong_independent_replication"
    elif len(primary) == 1 and len(formed) == 1:
        classification = "conditional_independent_replication"
    elif not formed:
        classification = "formation_replication_failed"
    else:
        classification = "causal_replication_failed"
    read_results = [
        result["read_diagnostic"]
        for result in formed
        if result["read_diagnostic"] is not None
    ]
    return {
        "classification": classification,
        "strong_replication_passed": (
            classification == "strong_independent_replication"
        ),
        "conditional_replication_supported": classification
        in {
            "strong_independent_replication",
            "conditional_independent_replication",
        },
        "seeds_total": len(results),
        "seeds_formed": len(formed),
        "seeds_primary_passed": len(primary),
        "read_gap_replicated_seeds": sum(
            item["read_gap_replicated"] for item in read_results
        ),
        "read_gap_powered_seeds": sum(item["powered"] for item in read_results),
        "read_gap_secondary_only": True,
        "registered_stop_boundary": (
            "Record this classification; do not extend, replace a seed, rescue "
            "an output head, or reopen router repair."
        ),
    }


def plot_results(results: list[dict[str, Any]], path: Path) -> None:
    formed = [result for result in results if result["formed"]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    labels = [str(result["seed"]) for result in results]
    final_queries = [
        100 * result["training"].get("final", {}).get("query", 0.0)
        for result in results
    ]
    colors = ["#2e7d32" if result["formed"] else "#b23a48" for result in results]
    axes[0].bar(labels, final_queries, color=colors)
    axes[0].axhline(95, color="#333333", linestyle="--")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Fresh 16-chunk formation")
    axes[0].set_ylabel("Query accuracy (%)")

    if formed:
        x = np.arange(len(CONDITIONS))
        width = 0.8 / len(formed)
        for index, result in enumerate(formed):
            values = [
                100 * result["causal"]["conditions"][name]["query"]
                for name in CONDITIONS
            ]
            axes[1].bar(
                x - 0.4 + width / 2 + index * width,
                values,
                width,
                label=f"seed {result['seed']}",
            )
        axes[1].set_xticks(
            x,
            [
                "intact",
                "reset",
                "zero",
                "roll",
                "zero L3",
                "roll L3",
                "keep L3",
            ],
            rotation=35,
            ha="right",
        )
        axes[1].axhline(20, color="#b23a48", linestyle=":")
        axes[1].axhline(90, color="#333333", linestyle="--")
        axes[1].legend()
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Paired causal interventions")
    axes[1].set_ylabel("Query accuracy (%)")

    if formed:
        x = np.arange(len(formed))
        memory = [
            100 * value if value is not None else np.nan
            for result in formed
            for value in [
                result["read_diagnostic"]["metrics"]["memory_l3_concat"][
                    "source_error_accuracy"
                ]
            ]
        ]
        hidden = [
            100 * value if value is not None else np.nan
            for result in formed
            for value in [
                result["read_diagnostic"]["metrics"]["query_hidden"][
                    "source_error_accuracy"
                ]
            ]
        ]
        axes[2].bar(x - 0.18, memory, 0.36, label="L3 Memory")
        axes[2].bar(x + 0.18, hidden, 0.36, label="Query hidden")
        axes[2].set_xticks(x, [str(result["seed"]) for result in formed])
        axes[2].legend()
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Decoding on source errors")
    axes[2].set_ylabel("Probe accuracy (%)")
    fig.suptitle("IST Level 7.1: Independent-Initialization Replication", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_smoke(args: argparse.Namespace) -> int:
    args.seeds = [SMOKE_SEED]
    args.output = "experiments/level7_1/smoke"
    args.chunk_size = 32
    args.diagnostic_batch_size = 4
    args.probe_train_samples = 16
    args.probe_val_samples = 8
    args.diagnostic_samples = 16
    args.causal_samples = 8
    args.probe_batch_size = 8
    args.probe_epochs = 2
    args.patience = 1
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    set_seed(SMOKE_SEED)
    model, probe = build(device, args.chunk_size)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=1e-3
    )
    random_step(model, probe, optimizer, args, 2, 2, 0.5, device, dtype)
    checkpoint_path = root / "restart_roundtrip.pt"
    checkpoint(
        checkpoint_path,
        model,
        probe,
        optimizer,
        {"smoke_checkpoint": True},
    )
    roundtrip_model, roundtrip_probe = build(device, args.chunk_size)
    roundtrip_optimizer = torch.optim.AdamW(
        list(roundtrip_model.parameters()) + list(roundtrip_probe.parameters()),
        lr=1e-3,
    )
    restore(
        checkpoint_path,
        roundtrip_model,
        roundtrip_probe,
        roundtrip_optimizer,
        device,
    )
    checkpoint_roundtrip = model_fingerprint(model) == model_fingerprint(
        roundtrip_model
    )
    checkpoint_path.unlink(missing_ok=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fingerprint_before = model_fingerprint(model)
    causal = [
        evaluate_causal_condition(
            model, args, 2, condition, args.causal_samples, 7100017,
            device, dtype,
        )
        for condition in CONDITIONS
    ]
    diagnostic = frozen_read_diagnostic(model, args, SMOKE_SEED, device, dtype)
    fingerprint_after = model_fingerprint(model)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "seed": SMOKE_SEED,
        "causal_conditions_exercised": [row["condition"] for row in causal],
        "read_diagnostic_exercised": diagnostic is not None,
        "checkpoint_roundtrip_passed": checkpoint_roundtrip,
        "model_fingerprint_unchanged": fingerprint_before == fingerprint_after,
        "passed": (
            len(causal) == len(CONDITIONS)
            and diagnostic is not None
            and checkpoint_roundtrip
            and fingerprint_before == fingerprint_after
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
        "stage": "training",
        "completed_seeds": [],
        "formal_seeds": FORMAL_SEEDS,
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    results = []
    for seed in FORMAL_SEEDS:
        folder = root / f"seed{seed}"
        training = train_independent_seed(seed, args, device, dtype, folder)
        progress["stage"] = "diagnostics"
        progress["active_seed"] = seed
        atomic_save(root / "progress.json", progress)
        result = evaluate_formed_seed(
            seed, training, args, device, dtype, folder
        )
        results.append(result)
        progress["completed_seeds"].append(seed)
        progress.pop("active_seed", None)
        atomic_save(root / "progress.json", progress)
        torch.cuda.empty_cache()

    diagnosis = classify(results)
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "formal_seeds_exact": [result["seed"] for result in results]
        == FORMAL_SEEDS,
        "seed909_used": False,
        "old_checkpoint_used": False,
        "training_budget_extended": False,
        "router_or_output_head_repair_used": False,
        "all_formed_seed_integrity_passed": all(
            result["integrity"].get("passed", True)
            for result in results
            if result["formed"]
        ),
    }
    integrity["passed"] = all(
        [
            integrity["formal_seeds_exact"],
            not integrity["seed909_used"],
            not integrity["old_checkpoint_used"],
            not integrity["training_budget_extended"],
            not integrity["router_or_output_head_repair_used"],
            integrity["all_formed_seed_integrity_passed"],
        ]
    )
    result = {
        "protocol": protocol,
        "integrity": integrity,
        "runs": results,
        "diagnosis": diagnosis,
    }
    summary = {
        "integrity": integrity,
        "diagnosis": diagnosis,
        "seeds": [
            {
                "seed": item["seed"],
                "formed": item["formed"],
                "primary_passed": item["primary_passed"],
                "final_query": item["training"].get("final", {}).get("query"),
                "causal_passed": item["causal"]["passed"]
                if item["causal"]
                else None,
                "read_gap_classification": item["read_diagnostic"][
                    "classification"
                ]
                if item["read_diagnostic"]
                else None,
                "seconds": item["training"].get("seconds"),
            }
            for item in results
        ],
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_results(results, root / "independent_replication.png")
    progress = {
        "stage": "complete",
        "completed_seeds": FORMAL_SEEDS,
        "classification": diagnosis["classification"],
        "integrity_passed": integrity["passed"],
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
