"""Level 7.3 cross-initialization layerwise persistent-Memory causal atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from long_context_test import set_seed
from run_level6_2_local import make_chunks
from run_level6_6_local import build
from run_level6_18_6_local import configure_cuda
from run_level7_1_local import atomic_save, model_fingerprint
from run_level7_2_local import metric_with_interval


SOURCES = [
    {
        "seed": 606,
        "checkpoint": "experiments/level6_8/formal/seed606/withdrawal_phase3.pt",
        "provenance": "Level 6.8 behavior pass",
    },
    {
        "seed": 808,
        "checkpoint": "experiments/level6_8/formal/seed808/withdrawal_phase3.pt",
        "provenance": "Level 6.8 behavior pass",
    },
    {
        "seed": 1001,
        "checkpoint": "experiments/level6_8/formal/seed1001/withdrawal_phase3.pt",
        "provenance": "Level 6.8 behavior pass",
    },
    {
        "seed": 1879,
        "checkpoint": "experiments/level7_2/formal/seed1879/zero_probe_step0750.pt",
        "provenance": "Level 7.2 selected protected behavior pass",
    },
]
CONDITIONS = [
    "intact",
    "reset_all",
    "zero_all",
    "batch_roll_all",
    "zero_l1",
    "zero_l2",
    "zero_l3",
    "batch_roll_l1",
    "batch_roll_l2",
    "batch_roll_l3",
    "keep_l1",
    "keep_l2",
    "keep_l3",
    "keep_l1_l2",
    "keep_l1_l3",
    "keep_l2_l3",
]
LAYERS = [1, 2, 3]
PAIRS = [(1, 2), (1, 3), (2, 3)]
ROOT = Path(__file__).resolve().parent
LEVEL_DIR = ROOT / "experiments" / "level7_3"
STATIC_PREREGISTRATION = LEVEL_DIR / "preregistration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed", type=int, default=7300000)
    parser.add_argument("--intact-threshold", type=float, default=0.90)
    parser.add_argument("--disruption-threshold", type=float, default=0.20)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.90)
    parser.add_argument("--local-threshold", type=float, default=0.90)
    parser.add_argument("--output", default="experiments/level7_3/formal")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_protocol_check(args: argparse.Namespace) -> None:
    expected = {
        "chunks": 16,
        "chunk_size": 128,
        "samples": 2048,
        "eval_batch_size": 16,
        "dataset_seed": 7300000,
        "intact_threshold": 0.90,
        "disruption_threshold": 0.20,
        "sufficiency_threshold": 0.90,
        "local_threshold": 0.90,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(f"Formal Level 7.3 protocol is locked: {mismatches}")
    if args.output != "experiments/level7_3/formal":
        raise ValueError("Formal output path is locked")


def validate_sources() -> list[dict[str, Any]]:
    level6 = read_json(ROOT / "experiments/level6_8/formal/summary.json")
    old_runs = {int(row["seed"]): row for row in level6["runs"]}
    level7 = read_json(ROOT / "experiments/level7_2/formal/result.json")
    new_runs = {int(row["seed"]): row for row in level7["runs"]}
    audit = []
    for source in SOURCES:
        path = ROOT / source["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        seed = source["seed"]
        if seed in old_runs:
            source_passed = bool(old_runs[seed]["passed"])
            source_behavior = old_runs[seed]["final"]["query"]
        else:
            row = new_runs[seed]
            source_passed = bool(row["protected"]["passed"])
            source_behavior = row["protected"]["query"]
        if not source_passed:
            raise RuntimeError(f"Frozen source seed={seed} lacks behavior pass")
        audit.append({
            **source,
            "checkpoint_sha256": sha256_file(path),
            "checkpoint_size_bytes": path.stat().st_size,
            "source_behavior": source_behavior,
            "source_passed": source_passed,
        })
    return audit


def parse_layers(condition: str) -> list[int]:
    return [int(token[1:]) - 1 for token in condition.split("_") if token.startswith("l")]


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
    selected = set(parse_layers(condition))
    if condition.startswith("zero_"):
        return [
            torch.zeros_like(item) if index in selected else item
            for index, item in enumerate(memory)
        ]
    if condition.startswith("batch_roll_"):
        return [
            item.roll(1, dims=0) if index in selected else item
            for index, item in enumerate(memory)
        ]
    if condition.startswith("keep_"):
        return [
            item if index in selected else torch.zeros_like(item)
            for index, item in enumerate(memory)
        ]
    raise ValueError(condition)


@torch.no_grad()
def evaluate_condition(
    model: torch.nn.Module,
    args: argparse.Namespace,
    condition: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    set_seed(args.dataset_seed)
    model.eval()
    total = 0
    query_correct = 0
    local_correct = 0
    while total < args.samples:
        batch = min(args.eval_batch_size, args.samples - total)
        chunks, target, position = make_chunks(
            batch, args.chunks, args.chunk_size, device
        )
        memory = None
        first_logits = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for chunk_index in range(args.chunks):
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
    return metric_with_interval({
        "condition": condition,
        "chunks": args.chunks,
        "samples": total,
        "query": query_correct / total,
        "local": local_correct / total,
    })


def layer_profile(metrics: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    necessary = [
        layer for layer in LAYERS
        if metrics[f"zero_l{layer}"]["query"] <= args.disruption_threshold
    ]
    sensitive = [
        layer for layer in LAYERS
        if metrics[f"batch_roll_l{layer}"]["query"] <= args.disruption_threshold
    ]
    sufficient = [
        layer for layer in LAYERS
        if metrics[f"keep_l{layer}"]["query"] >= args.sufficiency_threshold
    ]
    sufficient_pairs = [
        [left, right] for left, right in PAIRS
        if metrics[f"keep_l{left}_l{right}"]["query"] >= args.sufficiency_threshold
    ]
    dominant = sorted(set(necessary) & set(sufficient))
    if len(dominant) == 1:
        profile_class = f"layer{dominant[0]}_dominant"
    elif len(sufficient) >= 2:
        profile_class = "redundant_single_layer_sufficiency"
    elif len(sufficient) == 1:
        profile_class = f"layer{sufficient[0]}_sufficient_not_necessary"
    elif sufficient_pairs:
        profile_class = "distributed_pair_sufficiency"
    else:
        profile_class = "fully_distributed_or_unresolved"
    signature = {
        "necessary_layers": necessary,
        "misassignment_sensitive_layers": sensitive,
        "sufficient_layers": sufficient,
        "sufficient_pairs": sufficient_pairs,
    }
    signature_key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    whole_disrupted = max(
        metrics[name]["query"]
        for name in ("reset_all", "zero_all", "batch_roll_all")
    )
    minimum_local = min(row["local"] for row in metrics.values())
    whole_memory_passed = bool(
        metrics["intact"]["query"] >= args.intact_threshold
        and whole_disrupted <= args.disruption_threshold
        and minimum_local >= args.local_threshold
    )
    return {
        **signature,
        "signature_key": signature_key,
        "profile_class": profile_class,
        "whole_memory_max_disrupted_query": whole_disrupted,
        "minimum_local": minimum_local,
        "whole_memory_causality_passed": whole_memory_passed,
    }


def run_model(
    source: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    root: Path,
) -> dict[str, Any]:
    seed = source["seed"]
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return read_json(result_path)
    state = torch.load(ROOT / source["checkpoint"], map_location=device, weights_only=False)
    model, _ = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fingerprint_before = model_fingerprint(model)
    progress_path = folder / "condition_progress.json"
    rows = (
        read_json(progress_path)
        if progress_path.exists() and not args.force else {}
    )
    for condition in CONDITIONS:
        if condition in rows:
            continue
        metric = evaluate_condition(model, args, condition, device, dtype)
        rows[condition] = metric
        atomic_save(progress_path, rows)
        print(
            f"seed={seed} condition={condition} query={metric['query']:.2%} "
            f"local={metric['local']:.2%}", flush=True,
        )
    fingerprint_after = model_fingerprint(model)
    profile = layer_profile(rows, args)
    integrity = {
        "checkpoint_sha256": source["checkpoint_sha256"],
        "model_fingerprint_before": fingerprint_before,
        "model_fingerprint_after": fingerprint_after,
        "model_fingerprint_unchanged": fingerprint_before == fingerprint_after,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "all_conditions_present": set(rows) == set(CONDITIONS),
        "fresh_dataset_seed": args.dataset_seed,
        "passed": bool(
            fingerprint_before == fingerprint_after
            and all(not parameter.requires_grad for parameter in model.parameters())
            and set(rows) == set(CONDITIONS)
        ),
    }
    result = {
        "seed": seed,
        "source": source,
        "metrics": rows,
        "profile": profile,
        "integrity": integrity,
    }
    atomic_save(result_path, result)
    return result


def diagnose(results: list[dict[str, Any]]) -> dict[str, Any]:
    global_passes = sum(
        result["profile"]["whole_memory_causality_passed"] for result in results
    )
    unique_signatures = sorted({
        result["profile"]["signature_key"] for result in results
    })
    if global_passes < len(results):
        classification = "global_memory_causality_not_universal"
    elif len(unique_signatures) >= 2:
        classification = "cross_initialization_layer_heterogeneity_confirmed"
    else:
        classification = "layer_localization_homogeneous"
    return {
        "classification": classification,
        "models": len(results),
        "whole_memory_causal_passes": global_passes,
        "unique_layer_signatures": len(unique_signatures),
        "profile_classes": {
            str(result["seed"]): result["profile"]["profile_class"]
            for result in results
        },
        "heterogeneity_confirmed": classification
        == "cross_initialization_layer_heterogeneity_confirmed",
        "registered_stop_boundary": (
            "Report the fixed atlas; do not select a preferred layer "
            "intervention, alter thresholds, or modify a source model."
        ),
    }


def plot_atlas(results: list[dict[str, Any]], path: Path) -> None:
    seeds = [result["seed"] for result in results]
    values = np.array([
        [100 * result["metrics"][condition]["query"] for condition in CONDITIONS]
        for result in results
    ])
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    image = axes[0].imshow(values, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    axes[0].set_yticks(range(len(seeds)), [str(seed) for seed in seeds])
    axes[0].set_xticks(
        range(len(CONDITIONS)),
        [name.replace("batch_roll", "roll").replace("_", " ") for name in CONDITIONS],
        rotation=45, ha="right",
    )
    axes[0].set_xlabel("Causal condition")
    axes[0].set_ylabel("Model seed")
    axes[0].set_title("Fresh 16-chunk query accuracy (%)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] < 45 else "black"
            axes[0].text(
                column, row, f"{values[row, column]:.0f}",
                ha="center", va="center", fontsize=8, color=color,
            )
    fig.colorbar(image, ax=axes[0], fraction=0.025, pad=0.02)

    x = np.arange(len(seeds))
    width = 0.12
    bar_specs = [
        ("zero_l1", "zero L1"), ("zero_l2", "zero L2"), ("zero_l3", "zero L3"),
        ("keep_l1", "keep L1"), ("keep_l2", "keep L2"), ("keep_l3", "keep L3"),
    ]
    for index, (condition, label) in enumerate(bar_specs):
        axes[1].bar(
            x - 0.36 + width / 2 + index * width,
            [100 * result["metrics"][condition]["query"] for result in results],
            width, label=label,
        )
    axes[1].axhline(20, color="#b23a48", linestyle=":", label="necessity <=20%")
    axes[1].axhline(90, color="#333333", linestyle="--", label="sufficiency >=90%")
    axes[1].set_xticks(x, [str(seed) for seed in seeds])
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Model seed")
    axes[1].set_ylabel("Query accuracy (%)")
    axes[1].set_title("Single-layer necessity and sufficiency")
    axes[1].legend(ncol=2, fontsize=9)
    fig.suptitle("IST Level 7.3: Cross-Initialization Layerwise Causal Atlas", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_smoke(args: argparse.Namespace) -> int:
    args.output = "experiments/level7_3/smoke"
    args.chunks = 2
    args.chunk_size = 32
    args.samples = 8
    args.eval_batch_size = 4
    args.dataset_seed = 7300023
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    set_seed(23)
    model, _ = build(device, args.chunk_size)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = model_fingerprint(model)
    metrics = {
        condition: evaluate_condition(model, args, condition, device, dtype)
        for condition in CONDITIONS
    }
    profile = layer_profile(metrics, args)
    after = model_fingerprint(model)
    result = {
        "smoke_test": True,
        "scientific_evidence": False,
        "conditions": list(metrics),
        "profile_exercised": bool(profile["signature_key"]),
        "fingerprint_unchanged": before == after,
        "passed": bool(
            set(metrics) == set(CONDITIONS)
            and profile["signature_key"] and before == after
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
    source_audit = validate_sources()
    progress = {
        "stage": "causal_atlas",
        "models": [source["seed"] for source in source_audit],
        "completed_models": [],
        "seed909_locked": True,
    }
    atomic_save(root / "progress.json", progress)
    results = []
    for source in source_audit:
        result = run_model(source, args, device, dtype, root)
        results.append(result)
        progress["completed_models"].append(source["seed"])
        atomic_save(root / "progress.json", progress)
        torch.cuda.empty_cache()
    diagnosis = diagnose(results)
    integrity = {
        "script_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(STATIC_PREREGISTRATION),
        "source_seeds_exact": [source["seed"] for source in source_audit]
        == [source["seed"] for source in SOURCES],
        "conditions_exact": all(
            set(result["metrics"]) == set(CONDITIONS) for result in results
        ),
        "shared_fresh_dataset_seed": args.dataset_seed,
        "no_training_or_selection": True,
        "seed909_used": False,
        "all_model_integrity_passed": all(
            result["integrity"]["passed"] for result in results
        ),
    }
    integrity["passed"] = all([
        integrity["source_seeds_exact"],
        integrity["conditions_exact"],
        integrity["no_training_or_selection"],
        not integrity["seed909_used"],
        integrity["all_model_integrity_passed"],
    ])
    result = {
        "protocol": protocol,
        "source_audit": source_audit,
        "integrity": integrity,
        "runs": results,
        "diagnosis": diagnosis,
    }
    summary = {
        "integrity": integrity,
        "diagnosis": diagnosis,
        "models": [
            {
                "seed": row["seed"],
                "intact_query": row["metrics"]["intact"]["query"],
                "whole_memory_causality_passed": row["profile"]["whole_memory_causality_passed"],
                "profile_class": row["profile"]["profile_class"],
                "necessary_layers": row["profile"]["necessary_layers"],
                "sufficient_layers": row["profile"]["sufficient_layers"],
                "sufficient_pairs": row["profile"]["sufficient_pairs"],
            }
            for row in results
        ],
    }
    atomic_save(result_path, result)
    atomic_save(root / "summary.json", summary)
    plot_atlas(results, root / "layer_causal_atlas.png")
    atomic_save(root / "progress.json", {
        "stage": "complete",
        "completed_models": [source["seed"] for source in source_audit],
        "classification": diagnosis["classification"],
        "integrity_passed": integrity["passed"],
        "seed909_locked": True,
    })
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
