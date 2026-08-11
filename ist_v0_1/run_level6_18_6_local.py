import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from long_context_test import set_seed
from run_level6_2_local import make_chunks, vector
from run_level6_6_local import build
from run_level6_9_local import CONDITIONS, intervene
from run_level6_13_1_local import bootstrap_mean_ci, mcnemar
from run_level6_18_2_local import fit_linear


SEED = 707
CHUNK_COUNTS = [8, 12, 16]
MODEL_NAMES = ["source", "updated"]
ROUTE_PREFIXES = (
    "blocks.2.memory_read.",
    "blocks.2.memory_fusion_gate.",
)
FEATURE_STAGES = [
    "memory_l3_concat",
    "pre_fusion_feature",
    "memory_context",
    "fusion_gate",
    "fused_feature",
    "ffn_output",
    "query_hidden",
    "deployed_logits",
]
# The gate and pre-fusion feature are side inputs. These edges are the closest
# useful scalar-decoding approximation to the actual final-block causal path.
PATH_STAGES = [
    "memory_l3_concat",
    "memory_context",
    "fused_feature",
    "ffn_output",
    "query_hidden",
    "deployed_behavior",
]
DISPLAY_NAMES = {
    "memory_l3_concat": "Persistent\nMemory",
    "pre_fusion_feature": "Pre-fusion\nfeature",
    "memory_context": "Read\ncontext",
    "fusion_gate": "Fusion\ngate",
    "fused_feature": "Fused\nfeature",
    "ffn_output": "FFN\noutput",
    "query_hidden": "Query\nhidden",
    "deployed_logits": "Refit on\nlogits",
    "deployed_behavior": "Deployed\nargmax",
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def accuracy(predictions, labels):
    return (predictions == labels).float().mean().item()


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("Level 6.18.6 requires a CUDA GPU")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def checkpoint_audit(source_state, updated_state):
    source_model = source_state["model"]
    updated_model = updated_state["model"]
    if set(source_model) != set(updated_model):
        raise RuntimeError("Source and update-500 model state keys differ")
    changed = []
    unchanged = []
    illegal = []
    for name, source_value in source_model.items():
        updated_value = updated_model[name]
        if torch.equal(source_value, updated_value):
            unchanged.append(name)
            continue
        row = {
            "name": name,
            "shape": list(source_value.shape),
            "parameters": source_value.numel(),
            "max_abs_change": (
                source_value.float() - updated_value.float()
            ).abs().max().item(),
            "allowed": name.startswith(ROUTE_PREFIXES),
        }
        changed.append(row)
        if not row["allowed"]:
            illegal.append(name)
    probe_equal = (
        set(source_state["probe"]) == set(updated_state["probe"])
        and all(
            torch.equal(value, updated_state["probe"][name])
            for name, value in source_state["probe"].items()
        )
    )
    route_names = {
        name for name in source_model if name.startswith(ROUTE_PREFIXES)
    }
    changed_names = {row["name"] for row in changed}
    passed = (
        not illegal
        and changed_names == route_names
        and len(changed) == 6
        and sum(row["parameters"] for row in changed) == 24896
        and probe_equal
    )
    return {
        "passed": passed,
        "changed_tensors": changed,
        "changed_tensor_count": len(changed),
        "changed_parameter_count": sum(row["parameters"] for row in changed),
        "unchanged_tensor_count": len(unchanged),
        "illegal_changes": illegal,
        "original_memory_probe_unchanged": probe_equal,
        "expected_prefixes": list(ROUTE_PREFIXES),
    }


def load_pair(args, device):
    source_state = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    updated_state = torch.load(
        args.updated_checkpoint, map_location="cpu", weights_only=False
    )
    source_meta = source_state.get("level6_18_3", {})
    if not source_meta.get("success", {}).get("passed"):
        raise RuntimeError("The Level 6.18.3 source checkpoint did not pass formally")
    update_meta = updated_state.get("routing_training", {})
    if update_meta.get("update") != 500:
        raise RuntimeError(
            "Expected the Level 6.18.5 update-500 latest checkpoint; "
            f"found update={update_meta.get('update')}"
        )
    audit = checkpoint_audit(source_state, updated_state)
    if not audit["passed"]:
        raise RuntimeError(f"Checkpoint boundary audit failed: {audit}")

    models = {}
    probes = {}
    for name, state in (("source", source_state), ("updated", updated_state)):
        model, probe = build(device, args.chunk_size)
        model.load_state_dict(state["model"])
        probe.load_state_dict(state["probe"])
        model.eval()
        probe.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
        models[name] = model
        probes[name] = probe
    return models, probes, audit, update_meta


@torch.no_grad()
def collect_interfaces(model, original_probe, args, chunks_count, samples,
                       seed, device, dtype):
    set_seed(seed)
    features = {stage: [] for stage in FEATURE_STAGES}
    labels = []
    predictions = []
    original_probe_predictions = []
    local_predictions = []
    captured = {}
    final_block = model.blocks[-1]

    def memory_hook(_module, _inputs, output):
        captured["pre_fusion_feature"] = output[1]

    def memory_read_hook(_module, _inputs, output):
        captured["memory_context"] = output[0]

    def gate_hook(_module, _inputs, output):
        captured["fusion_gate"] = output

    def ffn_pre_hook(_module, inputs):
        captured["fused_feature"] = inputs[0]

    def ffn_hook(_module, _inputs, output):
        captured["ffn_output"] = output

    def norm_hook(_module, _inputs, output):
        captured["query_hidden"] = output

    handles = [
        final_block.memory.register_forward_hook(memory_hook),
        final_block.memory_read.register_forward_hook(memory_read_hook),
        final_block.memory_fusion_gate.register_forward_hook(gate_hook),
        final_block.ffn.register_forward_pre_hook(ffn_pre_hook),
        final_block.ffn.register_forward_hook(ffn_hook),
        final_block.norm2.register_forward_hook(norm_hook),
    ]
    total = 0
    try:
        while total < samples:
            batch = min(args.extract_batch_size, samples - total)
            chunks, target, position = make_chunks(
                batch, chunks_count, args.chunk_size, device
            )
            memory = None
            first_logits = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(chunks_count):
                    logits, memory = model(
                        chunks[:, chunk_index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    if chunk_index == 0:
                        first_logits = logits
                probe_logits = original_probe(vector(memory))
            rows = torch.arange(batch, device=device)
            query_predictions = logits[:, -1, :16].argmax(-1)
            labels.append(target.cpu())
            predictions.append(query_predictions.cpu())
            original_probe_predictions.append(probe_logits.argmax(-1).cpu())
            local_predictions.append(
                first_logits[rows, position, :16].argmax(-1).cpu()
            )
            features["memory_l3_concat"].append(
                memory[-1].reshape(batch, -1).detach().cpu().to(torch.float16)
            )
            for stage in FEATURE_STAGES[1:-1]:
                if stage not in captured:
                    raise RuntimeError(f"Hook did not capture {stage}")
                features[stage].append(
                    captured[stage][:, -1].detach().cpu().to(torch.float16)
                )
            features["deployed_logits"].append(
                logits[:, -1, :16].detach().cpu().to(torch.float16)
            )
            total += batch
    finally:
        for handle in handles:
            handle.remove()

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    original_probe_predictions = torch.cat(original_probe_predictions)
    local_predictions = torch.cat(local_predictions)
    return {
        "features": {name: torch.cat(items) for name, items in features.items()},
        "labels": labels,
        "predictions": predictions,
        "original_probe_predictions": original_probe_predictions,
        "local_predictions": local_predictions,
        "behavior": {
            "query": accuracy(predictions, labels),
            "original_memory_probe": accuracy(original_probe_predictions, labels),
            "local": accuracy(local_predictions, labels),
            "samples": len(labels),
        },
    }


def paired_statistics(source_predictions, updated_predictions, labels,
                      args, seed):
    source_correct = (source_predictions == labels).numpy().astype(np.int8)
    updated_correct = (updated_predictions == labels).numpy().astype(np.int8)
    change = updated_correct.astype(np.float64) - source_correct.astype(np.float64)
    return {
        "source_accuracy": float(source_correct.mean()),
        "updated_accuracy": float(updated_correct.mean()),
        "accuracy_change": bootstrap_mean_ci(
            change, seed, args.bootstrap_iterations
        ),
        "mcnemar": mcnemar(source_correct, updated_correct),
    }


def representation_shift(source, updated):
    source = source.float()
    updated = updated.float()
    delta = updated - source
    source_norm = source.norm(dim=-1).clamp_min(1e-8)
    delta_norm = delta.norm(dim=-1)
    return {
        "exactly_equal": torch.equal(source, updated),
        "max_abs_change": delta.abs().max().item(),
        "mean_l2_change": delta_norm.mean().item(),
        "mean_relative_l2_change": (delta_norm / source_norm).mean().item(),
        "mean_cosine_similarity": F.cosine_similarity(
            source, updated, dim=-1, eps=1e-8
        ).mean().item(),
    }


def path_drops(probes, behavior):
    values = {
        stage: probes[stage]["test_accuracy"]
        for stage in PATH_STAGES[:-1]
    }
    values["deployed_behavior"] = behavior["query"]
    edges = []
    for left, right in zip(PATH_STAGES[:-1], PATH_STAGES[1:]):
        edges.append({
            "from": left,
            "to": right,
            "from_accuracy": values[left],
            "to_accuracy": values[right],
            "drop": values[left] - values[right],
        })
    largest = max(edges, key=lambda row: row["drop"])
    return {"values": values, "edges": edges, "largest_drop": largest}


def process_length(models, original_probes, args, chunks_count, root,
                   device, dtype):
    result_path = root / f"chunks{chunks_count}.json"
    predictions_path = root / f"predictions_chunks{chunks_count}.json"
    if result_path.exists() and predictions_path.exists() and not args.force:
        print(f"chunks={chunks_count} reusing completed tomography", flush=True)
        return (
            json.loads(result_path.read_text(encoding="utf-8")),
            json.loads(predictions_path.read_text(encoding="utf-8")),
        )

    split_sizes = {
        "train": args.train_samples,
        "validation": args.val_samples,
        "test": args.test_samples,
    }
    split_offsets = {"train": 1, "validation": 2, "test": 3}
    base = args.dataset_seed_base + chunks_count * 10000
    datasets = {name: {} for name in MODEL_NAMES}
    for split, samples in split_sizes.items():
        seed = base + split_offsets[split]
        for model_name in MODEL_NAMES:
            datasets[model_name][split] = collect_interfaces(
                models[model_name], original_probes[model_name], args,
                chunks_count, samples, seed, device, dtype,
            )
        if not torch.equal(
            datasets["source"][split]["labels"],
            datasets["updated"][split]["labels"],
        ):
            raise RuntimeError(
                f"Source/update labels diverged at chunks={chunks_count} split={split}"
            )

    invariant_stages = ["memory_l3_concat", "pre_fusion_feature"]
    upstream_invariance = {}
    for stage in invariant_stages:
        rows = {
            split: torch.equal(
                datasets["source"][split]["features"][stage],
                datasets["updated"][split]["features"][stage],
            )
            for split in split_sizes
        }
        upstream_invariance[stage] = {
            "splits": rows,
            "passed": all(rows.values()),
        }
        if not upstream_invariance[stage]["passed"]:
            raise RuntimeError(
                f"Architecturally upstream stage changed: chunks={chunks_count} {stage}"
            )

    probe_metrics = {name: {} for name in MODEL_NAMES}
    probe_predictions = {name: {} for name in MODEL_NAMES}
    for model_name in MODEL_NAMES:
        train = datasets[model_name]["train"]
        validation = datasets[model_name]["validation"]
        test = datasets[model_name]["test"]
        for stage_index, stage in enumerate(FEATURE_STAGES):
            metric, predictions = fit_linear(
                train["features"][stage], train["labels"],
                validation["features"][stage], validation["labels"],
                test["features"][stage], test["labels"],
                args, device,
                # Matched seeds make identical representations produce an
                # identical fitted decoder instead of Probe-optimizer noise.
                base + 1000 + stage_index,
            )
            probe_metrics[model_name][stage] = metric
            probe_predictions[model_name][stage] = predictions
            print(
                f"chunks={chunks_count} model={model_name} stage={stage} "
                f"test={metric['test_accuracy']:.2%}",
                flush=True,
            )

    labels = datasets["source"]["test"]["labels"]
    comparisons = {}
    for stage_index, stage in enumerate(FEATURE_STAGES):
        comparisons[stage] = paired_statistics(
            probe_predictions["source"][stage],
            probe_predictions["updated"][stage], labels, args,
            base + 2000 + stage_index,
        )
    comparisons["deployed_behavior"] = paired_statistics(
        datasets["source"]["test"]["predictions"],
        datasets["updated"]["test"]["predictions"], labels, args,
        base + 2100,
    )

    shifts = {
        stage: representation_shift(
            datasets["source"]["test"]["features"][stage],
            datasets["updated"]["test"]["features"][stage],
        ) for stage in FEATURE_STAGES
    }
    first_changed = next(
        (stage for stage in FEATURE_STAGES if shifts[stage]["max_abs_change"] > 0),
        None,
    )
    result = {
        "chunks": chunks_count,
        "samples": split_sizes,
        "behavior": {
            name: datasets[name]["test"]["behavior"] for name in MODEL_NAMES
        },
        "probes": probe_metrics,
        "source_to_updated": comparisons,
        "representation_shift": shifts,
        "first_numerically_changed_stage": first_changed,
        "upstream_invariance": upstream_invariance,
        "path_decodability": {
            name: path_drops(
                probe_metrics[name], datasets[name]["test"]["behavior"]
            ) for name in MODEL_NAMES
        },
    }
    predictions = {
        "chunks": chunks_count,
        "labels": labels.tolist(),
        "models": {
            name: {
                "deployed": datasets[name]["test"]["predictions"].tolist(),
                "original_memory_probe": datasets[name]["test"][
                    "original_probe_predictions"
                ].tolist(),
                "local": datasets[name]["test"]["local_predictions"].tolist(),
                "refitted_probes": {
                    stage: predictions.tolist()
                    for stage, predictions in probe_predictions[name].items()
                },
            } for name in MODEL_NAMES
        },
    }
    save(result_path, result)
    save(predictions_path, predictions)
    return result, predictions


@torch.no_grad()
def causal_evaluate(model, args, condition, seed, device, dtype):
    set_seed(seed)
    labels = []
    predictions = []
    local_predictions = []
    total = 0
    while total < args.causal_samples:
        batch = min(args.extract_batch_size, args.causal_samples - total)
        chunks, target, position = make_chunks(batch, 16, args.chunk_size, device)
        memory = None
        first_logits = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for chunk_index in range(16):
                logits, produced = model(
                    chunks[:, chunk_index], memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
                if chunk_index == 0:
                    first_logits = logits
                memory = intervene(produced, condition)
        rows = torch.arange(batch, device=device)
        labels.append(target.cpu())
        predictions.append(logits[:, -1, :16].argmax(-1).cpu())
        local_predictions.append(
            first_logits[rows, position, :16].argmax(-1).cpu()
        )
        total += batch
    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    local_predictions = torch.cat(local_predictions)
    return {
        "labels": labels,
        "predictions": predictions,
        "local_predictions": local_predictions,
        "metric": {
            "condition": condition,
            "query": accuracy(predictions, labels),
            "local": accuracy(local_predictions, labels),
            "samples": len(labels),
        },
    }


def run_causal(models, args, device, dtype):
    if args.skip_causal:
        return {"skipped": True}, {"skipped": True}
    rows = {name: {} for name in MODEL_NAMES}
    raw = {name: {} for name in MODEL_NAMES}
    common_labels = None
    for condition_index, condition in enumerate(CONDITIONS):
        collected = {}
        for model_name in MODEL_NAMES:
            collected[model_name] = causal_evaluate(
                models[model_name], args, condition, args.causal_seed,
                device, dtype,
            )
        labels = collected["source"]["labels"]
        if not torch.equal(labels, collected["updated"]["labels"]):
            raise RuntimeError(f"Causal labels diverged for {condition}")
        if common_labels is None:
            common_labels = labels
        elif not torch.equal(common_labels, labels):
            raise RuntimeError(f"Causal condition labels diverged for {condition}")
        for model_name in MODEL_NAMES:
            rows[model_name][condition] = collected[model_name]["metric"]
            raw[model_name][condition] = collected[model_name][
                "predictions"
            ].tolist()
        rows.setdefault("source_to_updated", {})[condition] = paired_statistics(
            collected["source"]["predictions"],
            collected["updated"]["predictions"], labels, args,
            args.causal_seed + 100 + condition_index,
        )
        print(
            f"causal={condition} source={rows['source'][condition]['query']:.2%} "
            f"updated={rows['updated'][condition]['query']:.2%}",
            flush=True,
        )
    return rows, {
        "labels": common_labels.tolist(),
        "models": raw,
    }


def classify(length_results, args):
    primary = length_results["16"]
    changes = {
        stage: primary["source_to_updated"][stage]["accuracy_change"]["estimate"]
        for stage in FEATURE_STAGES
    }
    behavior_change = primary["source_to_updated"]["deployed_behavior"][
        "accuracy_change"
    ]["estimate"]
    context_change = changes["memory_context"]
    query_change = changes["query_hidden"]
    if (
        context_change < args.no_effect_threshold
        and query_change < args.no_effect_threshold
    ):
        classification = "no_generalizable_route_change"
        next_boundary = "Do not broaden training yet; inspect read supervision and sample conditioning."
    elif (
        context_change >= args.material_improvement
        and query_change < args.no_effect_threshold
    ):
        classification = "post_context_erasure"
        next_boundary = "Test a registered fusion + FFN/norm2 intervention boundary."
    elif (
        query_change >= args.material_improvement
        and behavior_change < args.no_effect_threshold
    ):
        classification = "output_mismatch"
        next_boundary = "Keep routing frozen and recalibrate the output head."
    elif behavior_change >= args.material_improvement:
        classification = "route_generalized"
        next_boundary = "Confirm on independent initialization before more intervention."
    else:
        classification = "mixed_or_small_effect"
        next_boundary = "Target the first interface with a significant decodability loss."
    return {
        "classification": classification,
        "primary_chunks": 16,
        "context_probe_change": context_change,
        "query_hidden_probe_change": query_change,
        "deployed_behavior_change": behavior_change,
        "all_stage_probe_changes": changes,
        "first_numerically_changed_stage": primary[
            "first_numerically_changed_stage"
        ],
        "source_largest_drop": primary["path_decodability"]["source"][
            "largest_drop"
        ],
        "updated_largest_drop": primary["path_decodability"]["updated"][
            "largest_drop"
        ],
        "thresholds": {
            "no_effect": args.no_effect_threshold,
            "material_improvement": args.material_improvement,
        },
        "registered_next_boundary": next_boundary,
    }


def plot_result(length_results, path):
    stages = FEATURE_STAGES + ["deployed_behavior"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.3), sharey=True)
    for axis, chunks_count in zip(axes, CHUNK_COUNTS):
        item = length_results[str(chunks_count)]
        for model_name, color, marker in (
            ("source", "#4c78a8", "o"),
            ("updated", "#e45756", "s"),
        ):
            values = [
                item["probes"][model_name][stage]["test_accuracy"]
                if stage != "deployed_behavior"
                else item["behavior"][model_name]["query"]
                for stage in stages
            ]
            axis.plot(
                range(len(stages)), [100 * value for value in values],
                marker=marker, linewidth=2, markersize=5,
                color=color, label=model_name,
            )
        axis.axhline(95, color="#444444", linestyle="--", linewidth=1)
        axis.set_title(f"{chunks_count} chunks", fontsize=15)
        axis.set_xticks(
            range(len(stages)),
            [DISPLAY_NAMES[stage] for stage in stages],
            rotation=35, ha="right", fontsize=9,
        )
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(0, 103)
    axes[0].set_ylabel("Held-out accuracy (%)", fontsize=13)
    axes[-1].legend(loc="lower left", fontsize=11)
    fig.suptitle(
        "IST Level 6.18.6: Frozen Final-Block Routing Tomography",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.18.6",
        "status": "frozen post-failure mechanism tomography; not a rescue test",
        "seed": SEED,
        "models": {
            "source": "formally passed Level 6.18.3 head-rescued checkpoint",
            "updated": "failed Level 6.18.5 update-500 routing_latest checkpoint",
        },
        "checkpoint_boundary": {
            "only_allowed_changed_prefixes": list(ROUTE_PREFIXES),
            "expected_changed_tensors": 6,
            "expected_changed_parameters": 24896,
        },
        "chunks": CHUNK_COUNTS,
        "interfaces": FEATURE_STAGES,
        "side_inputs": ["pre_fusion_feature", "fusion_gate"],
        "approximate_information_path": PATH_STAGES,
        "splits": {
            "train": args.train_samples,
            "validation": args.val_samples,
            "test": args.test_samples,
            "identical_examples_across_models": True,
            "disjoint_seeded_splits": True,
        },
        "comparison": {
            "independently_refitted_standardized_linear_probes": True,
            "matched_probe_seed_across_models_per_interface": True,
            "paired_bootstrap_and_mcnemar": True,
            "representation_shift_on_same_test_examples": True,
            "upstream_stages_required_exactly_invariant": [
                "memory_l3_concat", "pre_fusion_feature"
            ],
        },
        "classification_rule_16_chunks": {
            "context_and_query_improvement_below_1pp": "no_generalizable_route_change",
            "context_improvement_at_least_2pp_but_query_below_1pp": "post_context_erasure",
            "query_improvement_at_least_2pp_but_behavior_below_1pp": "output_mismatch",
            "behavior_improvement_at_least_2pp": "route_generalized",
            "otherwise": "mixed_or_small_effect",
        },
        "causal_conditions_16_chunks": CONDITIONS,
        "causal_status": "diagnostic only; not a success gate",
        "no_model_or_probe_parameter_updates": True,
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED:
        raise ValueError("Level 6.18.6 is fixed to seed707")
    if args.chunks != CHUNK_COUNTS:
        raise ValueError("Level 6.18.6 is fixed to chunks 8, 12, and 16")
    for checkpoint in (args.source_checkpoint, args.updated_checkpoint):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)
    if min(args.train_samples, args.val_samples, args.test_samples) <= 0:
        raise ValueError("Probe split sizes must be positive")
    if args.extract_batch_size < 2:
        raise ValueError("batch-roll requires extract-batch-size >= 2")


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.6 frozen final-block routing tomography"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=int, nargs="+", default=CHUNK_COUNTS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--source-checkpoint",
        default="experiments/level6_18_3/formal/rescued_head_checkpoint.pt",
    )
    parser.add_argument(
        "--updated-checkpoint",
        default="experiments/level6_18_5/formal/routing_latest.pt",
    )
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--dataset-seed-base", type=int, default=6186000)
    parser.add_argument("--causal-samples", type=int, default=512)
    parser.add_argument("--causal-seed", type=int, default=6186900)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--no-effect-threshold", type=float, default=0.01)
    parser.add_argument("--material-improvement", type=float, default=0.02)
    parser.add_argument("--output", default="experiments/level6_18_6/formal")
    parser.add_argument("--skip-causal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    validate(args)

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    protocol = preregistration(args)
    save(root / "preregistration.json", protocol)
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps(result["diagnosis"], indent=2))
        return

    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    models, probes, audit, update_meta = load_pair(args, device)
    print(
        f"checkpoint audit passed: {audit['changed_tensor_count']} tensors, "
        f"{audit['changed_parameter_count']} parameters",
        flush=True,
    )

    length_results = {}
    length_predictions = {}
    for chunks_count in CHUNK_COUNTS:
        item, predictions = process_length(
            models, probes, args, chunks_count, root, device, dtype
        )
        length_results[str(chunks_count)] = item
        length_predictions[str(chunks_count)] = predictions

    causal, causal_predictions = run_causal(models, args, device, dtype)
    diagnosis = classify(length_results, args)
    result = {
        "protocol": protocol,
        "checkpoint_audit": audit,
        "level6_18_5_update": {
            "update": update_meta.get("update"),
            "passed": update_meta.get("passed"),
            "stable_streak": update_meta.get("stable_streak"),
        },
        "lengths": length_results,
        "causal": causal,
        "diagnosis": diagnosis,
    }
    summary = {
        "checkpoint_audit": audit,
        "behavior": {
            count: item["source_to_updated"]["deployed_behavior"]
            for count, item in length_results.items()
        },
        "first_changed_stage": {
            count: item["first_numerically_changed_stage"]
            for count, item in length_results.items()
        },
        "largest_path_drop": {
            count: item["path_decodability"]
            for count, item in length_results.items()
        },
        "diagnosis": diagnosis,
    }
    save(result_path, result)
    save(root / "summary.json", summary)
    save(root / "predictions.json", {
        "lengths": length_predictions,
        "causal": causal_predictions,
    })
    plot_result(length_results, root / "routing_tomography.png")
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
