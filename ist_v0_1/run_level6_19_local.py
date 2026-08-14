import argparse
import bisect
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from long_context_test import set_seed
from run_level6_2_local import make_chunks
from run_level6_6_local import build
from run_level6_18_6_local import configure_cuda, save
from run_level6_18_8_local import continuous_effect


SEED = 707
CHUNKS = 16
SLOTS = 32
WIDTH = 64
FEATURES = [
    "memory_l3_concat",
    "pre_fusion_feature",
    "memory_context",
    "fusion_delta",
    "fused_feature",
    "ffn_output",
    "pre_norm_residual",
    "query_hidden",
    "deployed_logits",
]
CAUSAL_PATH = [
    "memory_l3_concat",
    "memory_context",
    "fused_feature",
    "pre_norm_residual",
    "query_hidden",
    "deployed_behavior",
]
DISPLAY = {
    "memory_l3_concat": "Persistent\nMemory",
    "pre_fusion_feature": "Pre-fusion\nfeature",
    "memory_context": "Read\ncontext",
    "fusion_delta": "Gate x\ncontext",
    "fused_feature": "Fused\nfeature",
    "ffn_output": "FFN output\n(side branch)",
    "pre_norm_residual": "Pre-norm\nresidual",
    "query_hidden": "Query\nhidden",
    "deployed_logits": "Deployed\nlogits",
    "deployed_behavior": "Deployed\nargmax",
}


class FinalTrace:
    def __init__(self, model):
        self.values = {}
        block = model.blocks[-1]
        self.handles = [
            block.memory.register_forward_hook(self._memory_hook),
            block.memory_read.register_forward_pre_hook(self._read_pre_hook),
            block.memory_read.register_forward_hook(self._read_hook),
            block.memory_fusion_gate.register_forward_hook(self._gate_hook),
            block.ffn.register_forward_pre_hook(self._ffn_pre_hook),
            block.ffn.register_forward_hook(self._ffn_hook),
            block.norm2.register_forward_pre_hook(self._norm_pre_hook),
            block.norm2.register_forward_hook(self._norm_hook),
        ]

    def clear(self):
        self.values = {}

    def close(self):
        for handle in self.handles:
            handle.remove()

    def require(self):
        expected = {
            "pre_fusion_feature", "read_query", "read_memory",
            "memory_context", "fusion_gate", "fused_feature",
            "ffn_output", "pre_norm_residual", "query_hidden",
        }
        missing = expected - set(self.values)
        if missing:
            raise RuntimeError(f"Final trace missing {sorted(missing)}")

    def _memory_hook(self, _module, _inputs, output):
        self.values["pre_fusion_feature"] = output[1]

    def _read_pre_hook(self, _module, inputs):
        self.values["read_query"] = inputs[0]
        self.values["read_memory"] = inputs[1]

    def _read_hook(self, _module, _inputs, output):
        self.values["memory_context"] = output[0]

    def _gate_hook(self, _module, _inputs, output):
        self.values["fusion_gate"] = output

    def _ffn_pre_hook(self, _module, inputs):
        self.values["fused_feature"] = inputs[0]

    def _ffn_hook(self, _module, _inputs, output):
        self.values["ffn_output"] = output

    def _norm_pre_hook(self, _module, inputs):
        self.values["pre_norm_residual"] = inputs[0]

    def _norm_hook(self, _module, _inputs, output):
        self.values["query_hidden"] = output


def load_frozen(args, device):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not state.get("level6_18_3", {}).get("success", {}).get("passed"):
        raise RuntimeError("Level 6.18.3 source checkpoint did not pass formally")
    model, original_probe = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    original_probe.load_state_dict(state["probe"])
    for module in (model, original_probe):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, original_probe, state.get("level6_18_3", {})


def task_competitor(logits, labels):
    masked = logits.float().clone()
    rows = torch.arange(len(labels), device=labels.device)
    masked[rows, labels] = -torch.inf
    return masked.argmax(dim=-1)


def append_query_feature(parts, name, value):
    parts[name].append(value[:, -1].detach().cpu().to(torch.float16))


def gradient_access(model, trace, final_chunk, incoming_memory, labels,
                    competitor, deployed_logits, dtype):
    block = model.blocks[-1]
    query = trace.values["read_query"].detach()
    read_memory = trace.values["read_memory"].detach()
    # Attention is descriptive only. The gradient audit below uses a second
    # complete model forward with the original need_weights=False path.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        _, attention = block.memory_read(
            query, read_memory, read_memory, need_weights=True,
            average_attn_weights=False,
        )
    captured = {}

    def make_memory_leaf(_module, _inputs, output):
        leaf = output[0].detach().clone().requires_grad_(True)
        captured["memory"] = leaf
        return leaf, output[1]

    def retain_context(_module, _inputs, output):
        captured["context"] = output[0]

    def retain_fused(_module, inputs):
        captured["fused"] = inputs[0]

    def retain_residual(_module, inputs):
        captured["residual"] = inputs[0]

    handles = [
        block.memory.register_forward_hook(make_memory_leaf),
        block.memory_read.register_forward_hook(retain_context),
        block.ffn.register_forward_pre_hook(retain_fused),
        block.norm2.register_forward_pre_hook(retain_residual),
    ]
    try:
        with torch.enable_grad(), torch.autocast(
            device_type="cuda", dtype=dtype
        ):
            full_logits, _ = model(
                final_chunk, memory=incoming_memory,
                return_memory=True, per_layer_memory=True,
            )
            logits = full_logits[:, -1, :16].float()
            expected = {"memory", "context", "fused", "residual"}
            missing = expected - set(captured)
            if missing:
                raise RuntimeError(
                    f"Differentiable forward missing {sorted(missing)}"
                )
            memory = captured["memory"]
            context = captured["context"]
            fused = captured["fused"]
            pre_norm = captured["residual"]
            rows = torch.arange(len(labels), device=labels.device)
            margin = logits[rows, labels] - logits[rows, competitor]
            gradients = torch.autograd.grad(
                margin.sum(), (memory, context, fused, pre_norm),
                only_inputs=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    reconstruction_max = (
        logits.detach() - deployed_logits.float()
    ).abs().max().item()
    memory_gradient, context_gradient, fused_gradient, residual_gradient = gradients
    return {
        "memory_gradient": memory_gradient.detach().cpu().to(torch.float16),
        "context_gradient_norm": context_gradient[:, -1].float().norm(
            dim=-1
        ).detach().cpu(),
        "fused_gradient_norm": fused_gradient[:, -1].float().norm(
            dim=-1
        ).detach().cpu(),
        "residual_gradient_norm": residual_gradient[:, -1].float().norm(
            dim=-1
        ).detach().cpu(),
        "attention": attention[:, :, -1].float().mean(
            dim=1
        ).detach().cpu().to(torch.float16),
        "margin": margin.detach().cpu(),
        "reconstruction_max_abs": reconstruction_max,
    }


def collect(model, args, samples, seed, device, dtype, gradients=False):
    set_seed(seed)
    trace = FinalTrace(model)
    feature_parts = {name: [] for name in FEATURES}
    labels_parts = []
    prediction_parts = []
    confidence_parts = []
    competitor_parts = []
    gradient_parts = {
        "memory_gradient": [],
        "context_gradient_norm": [],
        "fused_gradient_norm": [],
        "residual_gradient_norm": [],
        "attention": [],
        "margin": [],
    }
    reconstruction_max = 0.0
    total = 0
    try:
        while total < samples:
            batch = min(args.extract_batch_size, samples - total)
            chunks, target, _ = make_chunks(
                batch, CHUNKS, args.chunk_size, device
            )
            memory = None
            trace.clear()
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=dtype
            ):
                for chunk_index in range(CHUNKS - 1):
                    _, memory = model(
                        chunks[:, chunk_index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                incoming_memory = memory
                logits, memory = model(
                    chunks[:, -1], memory=incoming_memory,
                    return_memory=True, per_layer_memory=True,
                )
            trace.require()
            deployed = logits[:, -1, :16].float()
            predictions = deployed.argmax(dim=-1)
            top2 = deployed.topk(2, dim=-1).values
            confidence = top2[:, 0] - top2[:, 1]
            competitor = task_competitor(deployed, target)

            labels_parts.append(target.cpu())
            prediction_parts.append(predictions.cpu())
            confidence_parts.append(confidence.cpu())
            competitor_parts.append(competitor.cpu())
            feature_parts["memory_l3_concat"].append(
                memory[-1].reshape(batch, -1).detach().cpu().to(torch.float16)
            )
            for name in (
                "pre_fusion_feature", "memory_context", "fused_feature",
                "ffn_output", "pre_norm_residual", "query_hidden",
            ):
                append_query_feature(feature_parts, name, trace.values[name])
            fusion_delta = (
                trace.values["fusion_gate"] * trace.values["memory_context"]
            )
            append_query_feature(feature_parts, "fusion_delta", fusion_delta)
            feature_parts["deployed_logits"].append(
                deployed.detach().cpu().to(torch.float16)
            )
            if gradients:
                access = gradient_access(
                    model, trace, chunks[:, -1], incoming_memory, target,
                    competitor, deployed, dtype
                )
                reconstruction_max = max(
                    reconstruction_max, access.pop("reconstruction_max_abs")
                )
                for name, value in access.items():
                    gradient_parts[name].append(value)
            total += batch
            if gradients and (
                total == batch or total % args.log_every_samples == 0
            ):
                print(f"hard-example extraction {total}/{samples}", flush=True)
    finally:
        trace.close()
    output = {
        "features": {
            name: torch.cat(parts) for name, parts in feature_parts.items()
        },
        "labels": torch.cat(labels_parts),
        "predictions": torch.cat(prediction_parts),
        "confidence": torch.cat(confidence_parts),
        "competitor": torch.cat(competitor_parts),
    }
    if gradients:
        output["gradient"] = {
            name: torch.cat(parts) for name, parts in gradient_parts.items()
        }
        output["gradient_reconstruction_max_abs"] = reconstruction_max
    return output


def batched_probe_predictions(model, x, mean, std, batch_size, device):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].to(device, torch.float32)
            outputs.append(model((batch - mean) / std).cpu())
    return torch.cat(outputs)


def fit_probe(train_x, train_y, val_x, val_y, args, device, seed):
    set_seed(seed)
    mean = train_x.float().mean(dim=0).to(device)
    std = train_x.float().std(dim=0).clamp_min(1e-4).to(device)
    probe = nn.Linear(train_x.shape[-1], 16).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    patience = 0
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), args.probe_batch_size):
            ids = order[start:start + args.probe_batch_size]
            x = train_x[ids].to(device, torch.float32)
            y = train_y[ids].to(device)
            logits = probe((x - mean) / std)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        val_logits = batched_probe_predictions(
            probe, val_x, mean, std, args.probe_batch_size, device
        )
        val_accuracy = (
            val_logits.argmax(-1) == val_y
        ).float().mean().item()
        if val_accuracy > best_accuracy + 1e-5:
            best_accuracy = val_accuracy
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
        "metric": {
            "features": train_x.shape[-1],
            "best_epoch": best_epoch,
            "best_val_accuracy": best_accuracy,
        },
    }


def probe_logits(fitted, features, args, device):
    return batched_probe_predictions(
        fitted["probe"], features, fitted["mean"], fitted["std"],
        args.probe_batch_size, device,
    )


def match_confidence(wrong_ids, correct_ids, confidence):
    available = sorted(
        (float(confidence[index]), int(index)) for index in correct_ids.tolist()
    )
    matched = []
    distances = []
    for wrong in sorted(
        wrong_ids.tolist(), key=lambda index: float(confidence[index])
    ):
        value = float(confidence[wrong])
        location = bisect.bisect_left(available, (value, -1))
        candidates = []
        if location < len(available):
            candidates.append(location)
        if location > 0:
            candidates.append(location - 1)
        chosen = min(
            candidates,
            key=lambda item: (
                abs(available[item][0] - value), available[item][1]
            ),
        )
        selected_value, selected_id = available.pop(chosen)
        matched.append(selected_id)
        distances.append(abs(selected_value - value))
    # Return pairs in the original wrong-id order for paired statistics.
    lookup = {
        wrong: matched_id for wrong, matched_id in zip(
            sorted(wrong_ids.tolist(), key=lambda index: float(confidence[index])),
            matched,
        )
    }
    ordered = torch.tensor(
        [lookup[int(index)] for index in wrong_ids.tolist()], dtype=torch.long
    )
    return ordered, {
        "mean_absolute_confidence_distance": float(np.mean(distances)),
        "max_absolute_confidence_distance": float(np.max(distances)),
    }


def decision_margin(logits, labels):
    rows = torch.arange(len(labels))
    correct = logits[rows, labels]
    competitor = task_competitor(logits, labels)
    return correct - logits[rows, competitor], competitor


def paired_group_effect(error_values, matched_values, args, seed):
    difference = (
        torch.as_tensor(error_values).float()
        - torch.as_tensor(matched_values).float()
    ).numpy()
    return continuous_effect(difference, args, seed)


def group_summary(values, error_ids, matched_ids, args, seed):
    values = torch.as_tensor(values).float()
    return {
        "error_mean": values[error_ids].mean().item(),
        "matched_correct_mean": values[matched_ids].mean().item(),
        "error_median": values[error_ids].median().item(),
        "matched_correct_median": values[matched_ids].median().item(),
        "paired_error_minus_matched": paired_group_effect(
            values[error_ids], values[matched_ids], args, seed
        ),
    }


def probe_direction_and_slots(fitted, dataset, error_ids, matched_ids):
    features = dataset["features"]["memory_l3_concat"].float()
    labels = dataset["labels"]
    deployed_competitor = dataset["competitor"]
    mean = fitted["mean"].detach().cpu()
    std = fitted["std"].detach().cpu()
    weight = fitted["probe"].weight.detach().cpu()
    standardized = (features - mean) / std
    rows = torch.arange(len(labels))
    direction_standardized = (
        weight[labels] - weight[deployed_competitor]
    )
    slot_contributions = (
        standardized.reshape(-1, SLOTS, WIDTH)
        * direction_standardized.reshape(-1, SLOTS, WIDTH)
    ).sum(dim=-1)
    raw_direction = (
        direction_standardized / std[None, :]
    ).reshape(-1, SLOTS, WIDTH)
    memory_gradient = dataset["gradient"]["memory_gradient"].float()
    alignment = F.cosine_similarity(
        memory_gradient.reshape(len(labels), -1),
        raw_direction.reshape(len(labels), -1), dim=-1, eps=1e-8,
    )
    directional_access = (
        memory_gradient * raw_direction
    ).sum(dim=(-1, -2))
    gradient_slot_norm = memory_gradient.norm(dim=-1)
    attention = dataset["gradient"]["attention"].float()
    top_slots = slot_contributions.topk(4, dim=-1).indices
    top_attention = attention.gather(1, top_slots).sum(dim=-1)
    top_gradient_fraction = (
        gradient_slot_norm.gather(1, top_slots).sum(dim=-1)
        / gradient_slot_norm.sum(dim=-1).clamp_min(1e-8)
    )
    return {
        "slot_contributions": slot_contributions,
        "raw_probe_direction": raw_direction,
        "gradient_probe_alignment": alignment,
        "gradient_probe_directional_access": directional_access,
        "gradient_slot_norm": gradient_slot_norm,
        "attention": attention,
        "top4_probe_slot_attention_mass": top_attention,
        "top4_probe_slot_gradient_fraction": top_gradient_fraction,
    }


def aggregate_slots(slot_data, error_ids, matched_ids):
    output = []
    for slot in range(SLOTS):
        output.append({
            "slot": slot,
            "error_probe_contribution": slot_data[
                "slot_contributions"
            ][error_ids, slot].mean().item(),
            "matched_probe_contribution": slot_data[
                "slot_contributions"
            ][matched_ids, slot].mean().item(),
            "error_attention": slot_data[
                "attention"
            ][error_ids, slot].mean().item(),
            "matched_attention": slot_data[
                "attention"
            ][matched_ids, slot].mean().item(),
            "error_gradient_norm": slot_data[
                "gradient_slot_norm"
            ][error_ids, slot].mean().item(),
            "matched_gradient_norm": slot_data[
                "gradient_slot_norm"
            ][matched_ids, slot].mean().item(),
        })
    return output


def classify(stage_rows, args):
    accuracies = {
        stage: stage_rows[stage]["error_accuracy"] for stage in FEATURES
    }
    accuracies["deployed_behavior"] = 0.0
    memory = accuracies["memory_l3_concat"]
    context = accuracies["memory_context"]
    fused = accuracies["fused_feature"]
    residual = accuracies["pre_norm_residual"]
    query = accuracies["query_hidden"]
    if memory < args.hard_information_threshold:
        classification = "hard_example_memory_encoding_failure"
        boundary = "Persistent Memory encoding/composition, not read rescue."
    elif memory - context >= args.material_stage_drop:
        classification = "hard_example_memory_to_read_access_failure"
        boundary = "Read routing/slot selection is the first hard-example obstruction."
    elif context - fused >= args.material_stage_drop:
        classification = "hard_example_fusion_failure"
        boundary = "Fusion/residual input integration is the first obstruction."
    elif fused - residual >= args.material_stage_drop:
        classification = "hard_example_ffn_residual_failure"
        boundary = "Frozen FFN/residual transformation is the first obstruction."
    elif residual - query >= args.material_stage_drop:
        classification = "hard_example_normalization_failure"
        boundary = "Final normalization is the first obstruction."
    elif query >= args.hard_information_threshold:
        classification = "hard_example_output_alignment_failure"
        boundary = "Correct information reaches query hidden but deployed head misaligns."
    else:
        classification = "mixed_hard_example_access_failure"
        boundary = "No single registered interface explains the hard-example deficit."
    drops = []
    for left, right in zip(CAUSAL_PATH[:-1], CAUSAL_PATH[1:]):
        drops.append({
            "from": left,
            "to": right,
            "from_accuracy": accuracies[left],
            "to_accuracy": accuracies[right],
            "drop": accuracies[left] - accuracies[right],
        })
    return {
        "classification": classification,
        "error_group_stage_accuracies": accuracies,
        "causal_path_drops": drops,
        "largest_drop": max(drops, key=lambda row: row["drop"]),
        "thresholds": {
            "hard_information": args.hard_information_threshold,
            "material_stage_drop": args.material_stage_drop,
        },
        "registered_next_boundary": boundary,
    }


def analyze(dataset, fitted, args, device):
    labels = dataset["labels"]
    predictions = dataset["predictions"]
    wrong_ids = torch.where(predictions != labels)[0]
    correct_ids = torch.where(predictions == labels)[0]
    if len(wrong_ids) < args.minimum_errors:
        raise RuntimeError(
            f"Only {len(wrong_ids)} errors; minimum is {args.minimum_errors}"
        )
    matched_ids, matching = match_confidence(
        wrong_ids, correct_ids, dataset["confidence"]
    )
    matching.update({
        "errors": len(wrong_ids),
        "matched_correct": len(matched_ids),
        "error_confidence_mean": dataset["confidence"][wrong_ids].mean().item(),
        "matched_confidence_mean": dataset["confidence"][matched_ids].mean().item(),
    })

    stage_rows = {}
    stage_predictions = {}
    probe_states = {}
    for stage_index, stage in enumerate(FEATURES):
        logits = probe_logits(
            fitted[stage], dataset["features"][stage], args, device
        )
        margin, _ = decision_margin(logits, labels)
        probe_predictions = logits.argmax(dim=-1)
        correct = probe_predictions == labels
        stage_rows[stage] = {
            **fitted[stage]["metric"],
            "overall_accuracy": correct.float().mean().item(),
            "error_accuracy": correct[wrong_ids].float().mean().item(),
            "matched_correct_accuracy": correct[matched_ids].float().mean().item(),
            "error_margin": margin[wrong_ids].mean().item(),
            "matched_correct_margin": margin[matched_ids].mean().item(),
            "paired_accuracy_error_minus_matched": paired_group_effect(
                correct[wrong_ids].float(), correct[matched_ids].float(),
                args, args.analysis_seed + stage_index,
            ),
            "paired_margin_error_minus_matched": paired_group_effect(
                margin[wrong_ids], margin[matched_ids],
                args, args.analysis_seed + 100 + stage_index,
            ),
        }
        stage_predictions[stage] = probe_predictions
        probe_states[stage] = {
            "mean": fitted[stage]["mean"].detach().cpu(),
            "std": fitted[stage]["std"].detach().cpu(),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in fitted[stage]["probe"].state_dict().items()
            },
            "metric": fitted[stage]["metric"],
        }
        print(
            f"stage={stage} overall={stage_rows[stage]['overall_accuracy']:.2%} "
            f"errors={stage_rows[stage]['error_accuracy']:.2%} "
            f"matched={stage_rows[stage]['matched_correct_accuracy']:.2%}",
            flush=True,
        )

    memory_data = probe_direction_and_slots(
        fitted["memory_l3_concat"], dataset, wrong_ids, matched_ids
    )
    gradient_metrics = {}
    gradient_values = {
        "memory_gradient_norm": dataset["gradient"][
            "memory_gradient"
        ].float().norm(dim=(-1, -2)),
        "context_gradient_norm": dataset["gradient"]["context_gradient_norm"],
        "fused_gradient_norm": dataset["gradient"]["fused_gradient_norm"],
        "residual_gradient_norm": dataset["gradient"]["residual_gradient_norm"],
        "gradient_probe_alignment": memory_data["gradient_probe_alignment"],
        "gradient_probe_directional_access": memory_data[
            "gradient_probe_directional_access"
        ],
        "top4_probe_slot_attention_mass": memory_data[
            "top4_probe_slot_attention_mass"
        ],
        "top4_probe_slot_gradient_fraction": memory_data[
            "top4_probe_slot_gradient_fraction"
        ],
        "deployed_correct_rival_margin": dataset["gradient"]["margin"],
    }
    for index, (name, values) in enumerate(gradient_values.items()):
        gradient_metrics[name] = group_summary(
            values, wrong_ids, matched_ids, args,
            args.analysis_seed + 300 + index,
        )

    slot_rows = aggregate_slots(memory_data, wrong_ids, matched_ids)
    diagnosis = classify(stage_rows, args)
    diagnosis.update({
        "source_accuracy": (predictions == labels).float().mean().item(),
        "source_errors": len(wrong_ids),
        "memory_probe_correct_on_source_errors": stage_rows[
            "memory_l3_concat"
        ]["error_accuracy"],
        "oracle_source_or_memory_accuracy": (
            (predictions == labels)
            | (stage_predictions["memory_l3_concat"] == labels)
        ).float().mean().item(),
    })
    raw = {
        "labels": labels.tolist(),
        "deployed_predictions": predictions.tolist(),
        "confidence": dataset["confidence"].tolist(),
        "competitor": dataset["competitor"].tolist(),
        "error_ids": wrong_ids.tolist(),
        "matched_correct_ids": matched_ids.tolist(),
        "probe_predictions": {
            stage: values.tolist() for stage, values in stage_predictions.items()
        },
        "gradient_metrics": {
            name: values.tolist() for name, values in gradient_values.items()
        },
        "slot_probe_contributions": memory_data[
            "slot_contributions"
        ].tolist(),
        "slot_attention": memory_data["attention"].tolist(),
        "slot_gradient_norm": memory_data["gradient_slot_norm"].tolist(),
    }
    return {
        "matching": matching,
        "stages": stage_rows,
        "gradient_accessibility": gradient_metrics,
        "slots": slot_rows,
        "diagnosis": diagnosis,
    }, raw, probe_states


def plot_result(analysis, path):
    stages = FEATURES + ["deployed_behavior"]
    error_values = [
        analysis["stages"][stage]["error_accuracy"]
        if stage != "deployed_behavior" else 0.0
        for stage in stages
    ]
    matched_values = [
        analysis["stages"][stage]["matched_correct_accuracy"]
        if stage != "deployed_behavior" else 1.0
        for stage in stages
    ]
    gradient_names = [
        "memory_gradient_norm", "context_gradient_norm",
        "gradient_probe_alignment", "top4_probe_slot_attention_mass",
        "top4_probe_slot_gradient_fraction",
    ]
    slots = analysis["slots"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    x = np.arange(len(stages))
    axes[0].plot(
        x, 100 * np.asarray(error_values), marker="o", linewidth=2,
        color="#e45756", label="source errors",
    )
    axes[0].plot(
        x, 100 * np.asarray(matched_values), marker="s", linewidth=2,
        color="#4c78a8", label="confidence-matched correct",
    )
    axes[0].set_xticks(
        x, [DISPLAY[stage] for stage in stages], rotation=34, ha="right"
    )
    axes[0].set_ylabel("Independent decoder accuracy (%)")
    axes[0].set_title("Where correct-label information remains")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    gx = np.arange(len(gradient_names))
    error_gradient = [
        analysis["gradient_accessibility"][name]["error_mean"]
        for name in gradient_names
    ]
    matched_gradient = [
        analysis["gradient_accessibility"][name]["matched_correct_mean"]
        for name in gradient_names
    ]
    width = 0.38
    axes[1].bar(gx - width / 2, error_gradient, width, color="#e45756", label="errors")
    axes[1].bar(gx + width / 2, matched_gradient, width, color="#4c78a8", label="matched")
    axes[1].set_xticks(
        gx,
        ["Memory grad\nnorm", "Context grad\nnorm", "Grad-probe\nalignment",
         "Top4 slot\nattention", "Top4 slot\ngradient"],
        rotation=25, ha="right",
    )
    axes[1].set_title("Gradient accessibility and slot targeting")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].scatter(
        [row["error_probe_contribution"] for row in slots],
        [row["error_attention"] for row in slots],
        s=50 + 800 * np.asarray([row["error_gradient_norm"] for row in slots]),
        color="#72b7b2", alpha=0.8, edgecolor="white",
    )
    for row in slots:
        axes[2].text(
            row["error_probe_contribution"], row["error_attention"],
            str(row["slot"]), fontsize=7, ha="center", va="center",
        )
    axes[2].set_xlabel("Error-group Memory-probe contribution")
    axes[2].set_ylabel("Error-group mean read attention")
    axes[2].set_title("Slot code versus deployed read (size = grad norm)")
    axes[2].grid(alpha=0.2)
    fig.suptitle("IST Level 6.19: Frozen Hard-Example and Residual-Path Diagnosis", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.19",
        "status": "frozen hard-example and residual-path diagnosis",
        "model": "formally passed Level 6.18.3 source checkpoint",
        "seed": SEED,
        "chunks": CHUNKS,
        "splits": {
            "probe_train": args.probe_train_samples,
            "probe_validation": args.probe_val_samples,
            "hard_example_diagnostic": args.diagnostic_samples,
            "all_seeds_disjoint": True,
        },
        "groups": {
            "hard": "deployed source errors on diagnostic split",
            "control": "source-correct examples matched 1:1 without replacement on top1-top2 confidence",
            "minimum_errors": args.minimum_errors,
        },
        "independent_linear_decoders": FEATURES,
        "registered_causal_path": CAUSAL_PATH,
        "side_diagnostics": [
            "pre_fusion_feature", "fusion_delta", "ffn_output",
            "slot contributions", "read attention", "slot gradient norms",
        ],
        "gradient_accessibility": {
            "margin": "correct logit minus strongest incorrect logit",
            "targets": [
                "persistent read Memory", "read context", "fused feature",
                "pre-norm residual",
            ],
            "memory_code_direction": "independent all-slot Memory probe correct-vs-deployed-rival gradient",
        },
        "decision_rule": {
            "hard_information_threshold": args.hard_information_threshold,
            "material_stage_drop": args.material_stage_drop,
            "first_registered_path_drop_has_priority": True,
        },
        "no_parameter_updates": True,
        "level6_18_9_candidate_not_used": True,
        "protected_tests_not_used": True,
        "seed909_locked": True,
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError("Level 6.19 is fixed to seed707 at 16 chunks")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    if min(
        args.probe_train_samples, args.probe_val_samples,
        args.diagnostic_samples,
    ) <= 0:
        raise ValueError("all split sizes must be positive")
    if args.minimum_errors <= 0:
        raise ValueError("minimum-errors must be positive")


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.19 frozen hard-example/residual-path diagnosis"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint",
        default="experiments/level6_18_3/formal/rescued_head_checkpoint.pt",
    )
    parser.add_argument("--probe-train-samples", type=int, default=2048)
    parser.add_argument("--probe-val-samples", type=int, default=512)
    parser.add_argument("--diagnostic-samples", type=int, default=4096)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-train-seed", type=int, default=6190000)
    parser.add_argument("--probe-val-seed", type=int, default=6190100)
    parser.add_argument("--diagnostic-seed", type=int, default=6190200)
    parser.add_argument("--probe-fit-seed", type=int, default=6190300)
    parser.add_argument("--analysis-seed", type=int, default=6190400)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--minimum-errors", type=int, default=200)
    parser.add_argument("--hard-information-threshold", type=float, default=0.75)
    parser.add_argument("--material-stage-drop", type=float, default=0.15)
    parser.add_argument("--log-every-samples", type=int, default=512)
    parser.add_argument("--output", default="experiments/level6_19/formal")
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
        print(json.dumps(result["analysis"]["diagnosis"], indent=2))
        return

    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, original_probe, checkpoint_meta = load_frozen(args, device)
    del original_probe
    train = collect(
        model, args, args.probe_train_samples, args.probe_train_seed,
        device, dtype, False,
    )
    validation = collect(
        model, args, args.probe_val_samples, args.probe_val_seed,
        device, dtype, False,
    )
    diagnostic = collect(
        model, args, args.diagnostic_samples, args.diagnostic_seed,
        device, dtype, True,
    )
    if diagnostic["gradient_reconstruction_max_abs"] != 0.0:
        raise RuntimeError(
            "Gradient reconstruction did not exactly reproduce deployed logits: "
            f"{diagnostic['gradient_reconstruction_max_abs']}"
        )

    fitted = {}
    for index, stage in enumerate(FEATURES):
        fitted[stage] = fit_probe(
            train["features"][stage], train["labels"],
            validation["features"][stage], validation["labels"],
            args, device, args.probe_fit_seed + index,
        )
        print(
            f"fit stage={stage} val={fitted[stage]['metric']['best_val_accuracy']:.2%}",
            flush=True,
        )
    analysis, raw, probe_states = analyze(
        diagnostic, fitted, args, device
    )
    integrity = {
        "gradient_reconstruction_max_abs": diagnostic[
            "gradient_reconstruction_max_abs"
        ],
        "gradient_reconstruction_exact": (
            diagnostic["gradient_reconstruction_max_abs"] == 0.0
        ),
        "all_model_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "split_seeds_unique": len({
            args.probe_train_seed, args.probe_val_seed, args.diagnostic_seed
        }) == 3,
    }
    integrity["passed"] = all([
        integrity["gradient_reconstruction_exact"],
        integrity["all_model_parameters_frozen"],
        integrity["split_seeds_unique"],
    ])
    result = {
        "protocol": protocol,
        "checkpoint_meta": checkpoint_meta,
        "integrity": integrity,
        "analysis": analysis,
    }
    save(result_path, result)
    save(root / "summary.json", {
        "integrity": integrity,
        "matching": analysis["matching"],
        "stages": analysis["stages"],
        "gradient_accessibility": analysis["gradient_accessibility"],
        "diagnosis": analysis["diagnosis"],
    })
    save(root / "predictions.json", raw)
    torch.save(probe_states, root / "linear_probes.pt")
    plot_result(analysis, root / "hard_example_diagnosis.png")
    print(json.dumps(analysis["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
