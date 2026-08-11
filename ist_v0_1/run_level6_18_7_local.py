import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from long_context_test import set_seed
from run_level6_2_local import make_chunks
from run_level6_18_6_local import (
    accuracy,
    configure_cuda,
    load_pair,
    paired_statistics,
    save,
)


SEED = 707
CHUNK_COUNTS = [8, 12, 16]
PATCHES = {
    "context": ["memory_context"],
    "gate_activation": ["fusion_gate"],
    "context_gate": ["memory_context", "fusion_gate"],
    "fused_feature": ["fused_feature"],
    "ffn_output": ["ffn_output"],
    "query_hidden": ["query_hidden"],
}
REPRODUCTION_PATCHES = [
    "context_gate",
    "fused_feature",
    "ffn_output",
    "query_hidden",
]


class ActivationController:
    """Capture or replace only the final query position in the last block."""

    def __init__(self, model):
        self.patch = {}
        self.captured = {}
        block = model.blocks[-1]
        self.handles = [
            block.memory_read.register_forward_hook(self._memory_read_hook),
            block.memory_fusion_gate.register_forward_hook(self._gate_hook),
            block.ffn.register_forward_pre_hook(self._ffn_pre_hook),
            block.ffn.register_forward_hook(self._ffn_hook),
            block.norm2.register_forward_hook(self._norm_hook),
        ]

    def set_patch(self, patch=None):
        self.patch = {} if patch is None else patch
        self.captured = {}

    def snapshot(self):
        missing = set(
            [
                "memory_context",
                "fusion_gate",
                "fused_feature",
                "ffn_output",
                "query_hidden",
            ]
        ) - set(self.captured)
        if missing:
            raise RuntimeError(f"Activation capture incomplete: {sorted(missing)}")
        return {name: value.detach().clone() for name, value in self.captured.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()

    def _process(self, stage, value):
        self.captured[stage] = value.detach().clone()
        donor = self.patch.get(stage)
        if donor is None:
            return value
        if donor.shape != value.shape:
            raise RuntimeError(
                f"Patch shape mismatch for {stage}: {donor.shape} != {value.shape}"
            )
        output = value.clone()
        output[:, -1] = donor[:, -1].to(device=value.device, dtype=value.dtype)
        return output

    def _memory_read_hook(self, _module, _inputs, output):
        context = self._process("memory_context", output[0])
        return (context,) + tuple(output[1:])

    def _gate_hook(self, _module, _inputs, output):
        return self._process("fusion_gate", output)

    def _ffn_pre_hook(self, _module, inputs):
        fused = self._process("fused_feature", inputs[0])
        return (fused,) + tuple(inputs[1:])

    def _ffn_hook(self, _module, _inputs, output):
        return self._process("ffn_output", output)

    def _norm_hook(self, _module, _inputs, output):
        return self._process("query_hidden", output)


def condition_key(direction, patch_name):
    if direction == "forward":
        return f"updated_{patch_name}_to_source"
    return f"source_{patch_name}_to_updated"


def condition_specs():
    output = []
    for direction in ("forward", "reverse"):
        for patch_name, stages in PATCHES.items():
            output.append({
                "key": condition_key(direction, patch_name),
                "direction": direction,
                "patch": patch_name,
                "stages": stages,
                "receiver": "source" if direction == "forward" else "updated",
                "donor": "updated" if direction == "forward" else "source",
            })
    return output


def memories_equal(left, right):
    return len(left) == len(right) and all(
        torch.equal(a, b) for a, b in zip(left, right)
    )


@torch.no_grad()
def collect_transplants(models, args, chunks_count, device, dtype):
    specs = condition_specs()
    keys = ["source", "updated"] + [item["key"] for item in specs]
    prediction_parts = {key: [] for key in keys}
    labels_parts = []
    local_parts = []
    reproduction = {
        item["key"]: {
            "donor": item["donor"],
            "prediction_matches": 0,
            "logit_max_abs_difference": 0.0,
        }
        for item in specs if item["patch"] in REPRODUCTION_PATCHES
    }
    memory_exact = True
    source_controller = ActivationController(models["source"])
    updated_controller = ActivationController(models["updated"])
    controllers = {"source": source_controller, "updated": updated_controller}
    total = 0
    set_seed(args.dataset_seed_base + chunks_count * 1000)
    try:
        while total < args.samples:
            batch = min(args.eval_batch_size, args.samples - total)
            chunks, target, position = make_chunks(
                batch, chunks_count, args.chunk_size, device
            )
            memory = None
            first_logits = None
            source_controller.set_patch()
            with torch.autocast(device_type="cuda", dtype=dtype):
                # Route outputs are not returned as persistent Memory. One
                # shared source prefix is therefore exact for both receivers.
                for chunk_index in range(chunks_count - 1):
                    logits, memory = models["source"](
                        chunks[:, chunk_index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    if chunk_index == 0:
                        first_logits = logits

                final_chunk = chunks[:, -1]
                source_controller.set_patch()
                source_logits, source_memory = models["source"](
                    final_chunk, memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
                source_activations = source_controller.snapshot()

                updated_controller.set_patch()
                updated_logits, updated_memory = models["updated"](
                    final_chunk, memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
                updated_activations = updated_controller.snapshot()

                if not memories_equal(source_memory, updated_memory):
                    memory_exact = False

                logits_by_key = {
                    "source": source_logits,
                    "updated": updated_logits,
                }
                donors = {
                    "source": source_activations,
                    "updated": updated_activations,
                }
                baseline_logits = {
                    "source": source_logits,
                    "updated": updated_logits,
                }
                for item in specs:
                    patch = {
                        stage: donors[item["donor"]][stage]
                        for stage in item["stages"]
                    }
                    controller = controllers[item["receiver"]]
                    controller.set_patch(patch)
                    patched_logits, _ = models[item["receiver"]](
                        final_chunk, memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    logits_by_key[item["key"]] = patched_logits
                    if item["key"] in reproduction:
                        donor_logits = baseline_logits[item["donor"]][:, -1, :16]
                        patch_logits = patched_logits[:, -1, :16]
                        difference = (
                            patch_logits.float() - donor_logits.float()
                        ).abs().max().item()
                        reproduction[item["key"]]["logit_max_abs_difference"] = max(
                            reproduction[item["key"]]["logit_max_abs_difference"],
                            difference,
                        )
                        reproduction[item["key"]]["prediction_matches"] += int(
                            (
                                patch_logits.argmax(-1)
                                == donor_logits.argmax(-1)
                            ).sum().item()
                        )

            labels_parts.append(target.cpu())
            rows = torch.arange(batch, device=device)
            local_parts.append(
                first_logits[rows, position, :16].argmax(-1).cpu()
            )
            for key, logits in logits_by_key.items():
                prediction_parts[key].append(
                    logits[:, -1, :16].argmax(-1).cpu()
                )
            total += batch
    finally:
        source_controller.close()
        updated_controller.close()

    labels = torch.cat(labels_parts)
    predictions = {
        key: torch.cat(parts) for key, parts in prediction_parts.items()
    }
    local_predictions = torch.cat(local_parts)
    for row in reproduction.values():
        row["prediction_match_rate"] = row.pop("prediction_matches") / len(labels)
        row["passed"] = (
            row["prediction_match_rate"] == 1.0
            and row["logit_max_abs_difference"] == 0.0
        )
    return {
        "labels": labels,
        "predictions": predictions,
        "local_predictions": local_predictions,
        "memory_exact": memory_exact,
        "reproduction": reproduction,
    }


def holm_adjust(named_p_values):
    ordered = sorted(named_p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, value * (total - rank))
        running = max(running, candidate)
        adjusted[name] = running
    return {
        name: {
            "raw_p": named_p_values[name],
            "holm_p": adjusted[name],
            "significant_0.05": adjusted[name] < 0.05,
        }
        for name in named_p_values
    }


def analyze_transplants(collected, args, chunks_count):
    labels = collected["labels"]
    predictions = collected["predictions"]
    specs = condition_specs()
    metrics = {
        key: {
            "accuracy": accuracy(value, labels),
            "samples": len(labels),
        } for key, value in predictions.items()
    }
    comparisons = {}
    for index, item in enumerate(specs):
        receiver = item["receiver"]
        comparisons[item["key"]] = paired_statistics(
            predictions[receiver], predictions[item["key"]], labels, args,
            args.bootstrap_seed_base + chunks_count * 100 + index,
        )

    forward_context = condition_key("forward", "context")
    reverse_context = condition_key("reverse", "context")
    decomposition = {
        "forward": {
            "updated_context_through_source_gate": paired_statistics(
                predictions["source"], predictions[forward_context], labels, args,
                args.bootstrap_seed_base + chunks_count * 100 + 50,
            ),
            "updated_gate_after_updated_context": paired_statistics(
                predictions[forward_context], predictions["updated"], labels, args,
                args.bootstrap_seed_base + chunks_count * 100 + 51,
            ),
            "total_source_to_updated": paired_statistics(
                predictions["source"], predictions["updated"], labels, args,
                args.bootstrap_seed_base + chunks_count * 100 + 52,
            ),
        },
        "reverse": {
            "source_context_through_updated_gate": paired_statistics(
                predictions["updated"], predictions[reverse_context], labels, args,
                args.bootstrap_seed_base + chunks_count * 100 + 53,
            ),
            "source_gate_after_source_context": paired_statistics(
                predictions[reverse_context], predictions["source"], labels, args,
                args.bootstrap_seed_base + chunks_count * 100 + 54,
            ),
            "total_updated_to_source": paired_statistics(
                predictions["updated"], predictions["source"], labels, args,
                args.bootstrap_seed_base + chunks_count * 100 + 55,
            ),
        },
    }
    primary_p = {
        "context_effect": decomposition["forward"][
            "updated_context_through_source_gate"
        ]["mcnemar"]["p"],
        "gate_after_context": decomposition["forward"][
            "updated_gate_after_updated_context"
        ]["mcnemar"]["p"],
        "total_route_update": decomposition["forward"][
            "total_source_to_updated"
        ]["mcnemar"]["p"],
    }
    decomposition["primary_holm_family"] = holm_adjust(primary_p)
    integrity = {
        "returned_memory_exactly_invariant": collected["memory_exact"],
        "exact_donor_reproduction": collected["reproduction"],
    }
    integrity["all_reproduction_passed"] = all(
        row["passed"] for row in collected["reproduction"].values()
    )
    integrity["passed"] = (
        integrity["returned_memory_exactly_invariant"]
        and integrity["all_reproduction_passed"]
    )
    return {
        "chunks": chunks_count,
        "metrics": metrics,
        "local_control": {
            "accuracy": accuracy(collected["local_predictions"], labels),
            "samples": len(labels),
        },
        "comparisons_to_receiver": comparisons,
        "decomposition": decomposition,
        "integrity": integrity,
    }


def process_length(models, args, chunks_count, root, device, dtype):
    result_path = root / f"chunks{chunks_count}.json"
    predictions_path = root / f"predictions_chunks{chunks_count}.json"
    if result_path.exists() and predictions_path.exists() and not args.force:
        print(f"chunks={chunks_count} reusing completed transplant panel", flush=True)
        return (
            json.loads(result_path.read_text(encoding="utf-8")),
            json.loads(predictions_path.read_text(encoding="utf-8")),
        )
    collected = collect_transplants(
        models, args, chunks_count, device, dtype
    )
    result = analyze_transplants(collected, args, chunks_count)
    predictions = {
        "chunks": chunks_count,
        "labels": collected["labels"].tolist(),
        "local": collected["local_predictions"].tolist(),
        "conditions": {
            key: value.tolist()
            for key, value in collected["predictions"].items()
        },
    }
    save(result_path, result)
    save(predictions_path, predictions)
    print(
        f"chunks={chunks_count} source={result['metrics']['source']['accuracy']:.2%} "
        f"updated={result['metrics']['updated']['accuracy']:.2%} "
        f"context->source={result['metrics'][condition_key('forward', 'context')]['accuracy']:.2%} "
        f"integrity={result['integrity']['passed']}",
        flush=True,
    )
    return result, predictions


def confirmed_effect(statistic, holm_row, direction):
    estimate = statistic["accuracy_change"]["estimate"]
    return holm_row["significant_0.05"] and (
        estimate > 0 if direction == "positive" else estimate < 0
    )


def classify(length_results):
    primary = length_results["16"]["decomposition"]
    forward = primary["forward"]
    holm = primary["primary_holm_family"]
    context = forward["updated_context_through_source_gate"]
    gate = forward["updated_gate_after_updated_context"]
    total = forward["total_source_to_updated"]
    context_positive = confirmed_effect(
        context, holm["context_effect"], "positive"
    )
    gate_negative = confirmed_effect(
        gate, holm["gate_after_context"], "negative"
    )
    total_positive = confirmed_effect(
        total, holm["total_route_update"], "positive"
    )
    integrity = all(
        item["integrity"]["passed"] for item in length_results.values()
    )
    if not integrity:
        classification = "invalid_transplant_integrity"
        next_boundary = "Repair activation-hook or checkpoint invariance before interpretation."
    elif context_positive and gate_negative:
        classification = "fusion_gate_causally_cancels_context_gain"
        next_boundary = "Register gate/fusion alignment intervention; keep FFN and head frozen."
    elif not context_positive:
        classification = "context_decodability_gain_not_behaviorally_causal"
        next_boundary = "Diagnose task-aligned read subspace before any broader training."
    elif context_positive and not total_positive:
        classification = "post_context_attenuation_not_localized_to_gate"
        next_boundary = "Test residual/FFN component patching with task-aligned context directions."
    elif total_positive:
        classification = "route_update_behaviorally_causal"
        next_boundary = "Confirm the route effect on independent data and initialization."
    else:
        classification = "mixed_or_asymmetric_transplant_effect"
        next_boundary = "Repeat only the registered primary contrasts on a larger panel."
    return {
        "classification": classification,
        "integrity_passed_all_lengths": integrity,
        "primary_chunks": 16,
        "context_effect": context,
        "gate_after_context_effect": gate,
        "total_route_effect": total,
        "primary_holm_family": holm,
        "context_effect_confirmed_positive": context_positive,
        "gate_effect_confirmed_negative": gate_negative,
        "total_effect_confirmed_positive": total_positive,
        "registered_next_boundary": next_boundary,
    }


def plot_result(length_results, path):
    display = [
        ("source", "Source"),
        (condition_key("forward", "context"), "U context\ninto S"),
        (condition_key("forward", "gate_activation"), "U gate\ninto S"),
        (condition_key("forward", "context_gate"), "U context+gate\ninto S"),
        ("updated", "Updated"),
        (condition_key("reverse", "context"), "S context\ninto U"),
        (condition_key("reverse", "gate_activation"), "S gate\ninto U"),
        (condition_key("reverse", "context_gate"), "S context+gate\ninto U"),
    ]
    colors = [
        "#4c78a8", "#72b7b2", "#54a24b", "#2f7f5f",
        "#e45756", "#f2cf5b", "#f58518", "#b44b28",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True)
    for axis, chunks_count in zip(axes, CHUNK_COUNTS):
        metrics = length_results[str(chunks_count)]["metrics"]
        values = [100 * metrics[key]["accuracy"] for key, _ in display]
        bars = axis.bar(range(len(display)), values, color=colors)
        axis.axhline(95, color="#333333", linestyle="--", linewidth=1.2)
        axis.set_title(f"{chunks_count} chunks", fontsize=15)
        axis.set_xticks(
            range(len(display)), [label for _, label in display],
            rotation=32, ha="right", fontsize=9,
        )
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2, value + 0.25,
                f"{value:.1f}", ha="center", va="bottom", fontsize=8,
            )
    axes[0].set_ylabel("Held-out query accuracy (%)", fontsize=13)
    axes[0].set_ylim(80, 101)
    fig.suptitle(
        "IST Level 6.18.7: Bidirectional Final-Query Activation Transplant",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.18.7",
        "status": "frozen bidirectional causal activation transplant",
        "seed": SEED,
        "models": {
            "source": "formally passed Level 6.18.3 checkpoint",
            "updated": "Level 6.18.5 update-500 routing_latest checkpoint",
        },
        "checkpoint_boundary_inherited_from_level6_18_6": {
            "changed_tensors": 6,
            "changed_parameters": 24896,
            "only_memory_read_and_fusion_gate": True,
        },
        "chunks": CHUNK_COUNTS,
        "samples_per_length": args.samples,
        "patch_scope": {
            "chunk": "final chunk only",
            "token": "final query position only",
            "directions": ["updated_to_source", "source_to_updated"],
            "patches": PATCHES,
            "gate_activation_only_is_synthetic": True,
        },
        "primary_16_chunk_contrasts": {
            "context_effect": "source -> updated context through source gate",
            "gate_after_context": "updated context through source gate -> full updated",
            "total_route_update": "source -> full updated",
            "multiplicity": "Holm correction across these three exact McNemar tests",
        },
        "reverse_direction": "corroborative restoration panel",
        "integrity_requirements": {
            "returned_memory_exactly_invariant": True,
            "context_plus_gate_reproduces_donor_logits_exactly": True,
            "fused_feature_reproduces_donor_logits_exactly": True,
            "ffn_output_reproduces_donor_logits_exactly": True,
            "query_hidden_reproduces_donor_logits_exactly": True,
        },
        "decision_rule": {
            "positive_context_and_negative_gate_after_holm": "fusion_gate_causally_cancels_context_gain",
            "context_not_positive_after_holm": "context_decodability_gain_not_behaviorally_causal",
            "positive_context_but_total_not_positive": "post_context_attenuation_not_localized_to_gate",
            "positive_total_after_holm": "route_update_behaviorally_causal",
        },
        "no_parameter_or_probe_updates": True,
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED:
        raise ValueError("Level 6.18.7 is fixed to seed707")
    if args.chunks != CHUNK_COUNTS:
        raise ValueError("Level 6.18.7 is fixed to chunks 8, 12, and 16")
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.eval_batch_size < 2:
        raise ValueError("eval-batch-size must be at least 2")
    for checkpoint in (args.source_checkpoint, args.updated_checkpoint):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.7 bidirectional final-query activation transplant"
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
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dataset-seed-base", type=int, default=6187000)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed-base", type=int, default=6187800)
    parser.add_argument("--output", default="experiments/level6_18_7/formal")
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
    del probes
    print(
        f"checkpoint audit passed: {audit['changed_tensor_count']} tensors, "
        f"{audit['changed_parameter_count']} parameters",
        flush=True,
    )

    length_results = {}
    length_predictions = {}
    for chunks_count in CHUNK_COUNTS:
        item, predictions = process_length(
            models, args, chunks_count, root, device, dtype
        )
        length_results[str(chunks_count)] = item
        length_predictions[str(chunks_count)] = predictions
    diagnosis = classify(length_results)
    result = {
        "protocol": protocol,
        "checkpoint_audit": audit,
        "level6_18_5_update": {
            "update": update_meta.get("update"),
            "passed": update_meta.get("passed"),
            "stable_streak": update_meta.get("stable_streak"),
        },
        "lengths": length_results,
        "diagnosis": diagnosis,
    }
    summary = {
        "baseline": {
            count: {
                "source": item["metrics"]["source"],
                "updated": item["metrics"]["updated"],
            } for count, item in length_results.items()
        },
        "decomposition": {
            count: item["decomposition"]
            for count, item in length_results.items()
        },
        "integrity": {
            count: item["integrity"]
            for count, item in length_results.items()
        },
        "diagnosis": diagnosis,
    }
    save(result_path, result)
    save(root / "summary.json", summary)
    save(root / "predictions.json", {"lengths": length_predictions})
    plot_result(length_results, root / "activation_transplant.png")
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
