import argparse
import hashlib
import json
import math
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
from run_level6_18_6_local import configure_cuda, paired_statistics, save
from run_level6_18_7_local import holm_adjust, memories_equal
from run_level6_18_8_local import continuous_effect
from run_level6_19_local import match_confidence


SEED = 707
CHUNKS = 16
SLOTS = 32
WIDTH = 64
TOP_K = 4
ODDS = [2.0, 4.0, 8.0]
MAIN_ODDS = 4.0


def odds_key(value):
    return f"top4_odds{value:g}".replace(".", "_")


def condition_keys(args):
    keys = ["source"]
    keys.extend(odds_key(value) for value in ODDS)
    keys.extend(["low4_odds4", "rolled_top4_odds4"])
    keys.extend(f"random4_odds4_{index + 1}" for index in range(args.random_repeats))
    keys.append("context_gradient_positive_control")
    return keys


def tensor_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_frozen(args, device):
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not source.get("level6_18_3", {}).get("success", {}).get("passed"):
        raise RuntimeError("Level 6.18.3 source checkpoint did not pass formally")
    model, original_probe = build(device, args.chunk_size)
    model.load_state_dict(source["model"])
    original_probe.load_state_dict(source["probe"])
    for module in (model, original_probe):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    saved = torch.load(args.probes, map_location="cpu", weights_only=False)
    probes = {}
    for stage in ("memory_l3_concat", "memory_context"):
        if stage not in saved:
            raise RuntimeError(f"Level 6.19 probe checkpoint lacks {stage}")
        row = saved[stage]
        features = int(row["mean"].numel())
        probe = nn.Linear(features, 16).to(device)
        probe.load_state_dict(row["state_dict"])
        probe.eval()
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
        probes[stage] = {
            "probe": probe,
            "mean": row["mean"].float().to(device),
            "std": row["std"].float().to(device),
            "metric": row.get("metric", {}),
        }
    return model, original_probe, probes, source.get("level6_18_3", {})


def probe_logits(fitted, features):
    standardized = (features.float() - fitted["mean"]) / fitted["std"]
    return fitted["probe"](standardized)


def task_competitor(logits, labels):
    masked = logits.float().clone()
    rows = torch.arange(len(labels), device=labels.device)
    masked[rows, labels] = -torch.inf
    return masked.argmax(dim=-1)


def read_mask(query, selected, odds, heads):
    if selected is None or odds == 1.0:
        return None
    batch, length = query.shape[:2]
    slots = selected.shape[1]
    mask = torch.zeros(
        batch * heads, length, SLOTS, device=query.device, dtype=query.dtype
    )
    expanded = selected[:, None, :].expand(batch, heads, slots).reshape(
        batch * heads, slots
    )
    rows = torch.arange(batch * heads, device=query.device)[:, None]
    mask[rows, length - 1, expanded] = math.log(odds)
    return mask


class ReadController:
    """Bias final-query read logits or replace its context in the final block."""

    def __init__(self, model):
        self.module = model.blocks[-1].memory_read
        self.heads = self.module.num_heads
        self.selected = None
        self.odds = 1.0
        self.context_patch = None
        self.gradient_leaf = False
        self.live_context = None
        self.captured = {}
        self.suspended = False
        self.handles = [
            self.module.register_forward_pre_hook(
                self._pre_hook, with_kwargs=True
            ),
            self.module.register_forward_hook(self._output_hook),
        ]

    def configure(self, selected=None, odds=1.0, context_patch=None,
                  gradient_leaf=False):
        self.selected = selected
        self.odds = float(odds)
        self.context_patch = context_patch
        self.gradient_leaf = gradient_leaf
        self.live_context = None
        self.captured = {}

    def close(self):
        for handle in self.handles:
            handle.remove()

    def snapshot(self):
        required = {"read_query", "read_memory", "memory_context"}
        missing = required - set(self.captured)
        if missing:
            raise RuntimeError(f"Read capture incomplete: {sorted(missing)}")
        return {
            name: value.detach().clone()
            for name, value in self.captured.items()
        }

    def _pre_hook(self, _module, inputs, kwargs):
        if self.suspended:
            return inputs, kwargs
        query, key = inputs[:2]
        self.captured["read_query"] = query.detach()
        self.captured["read_memory"] = key.detach()
        mask = read_mask(query, self.selected, self.odds, self.heads)
        if mask is not None:
            existing = kwargs.get("attn_mask")
            if existing is not None:
                mask = mask + existing.to(device=mask.device, dtype=mask.dtype)
            kwargs = dict(kwargs)
            kwargs["attn_mask"] = mask
        return inputs, kwargs

    def _output_hook(self, _module, _inputs, output):
        if self.suspended:
            return output
        context = output[0]
        if self.gradient_leaf:
            context = context.detach().clone().requires_grad_(True)
            self.live_context = context
        if self.context_patch is not None:
            if self.context_patch.shape != context[:, -1].shape:
                raise RuntimeError(
                    "Context patch shape mismatch: "
                    f"{self.context_patch.shape} != {context[:, -1].shape}"
                )
            context = context.clone()
            context[:, -1] = self.context_patch.to(
                device=context.device, dtype=context.dtype
            )
        self.captured["memory_context"] = context.detach()
        return (context,) + tuple(output[1:])

    @torch.no_grad()
    def selected_attention_mass(self, query, memory, selected, odds, dtype):
        mask = read_mask(query, selected, odds, self.heads)
        self.suspended = True
        try:
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, weights = self.module(
                    query, memory, memory, attn_mask=mask,
                    need_weights=True, average_attn_weights=False,
                )
        finally:
            self.suspended = False
        query_weights = weights[:, :, -1].float()
        indices = selected[:, None, :].expand(
            -1, self.heads, -1
        )
        return query_weights.gather(-1, indices).sum(dim=-1).mean(dim=1)


def norm_match(direction, target_norm):
    norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return direction / norm * target_norm[:, None]


def select_slots(memory, labels, competitor, fitted):
    flat = memory.float().reshape(len(labels), -1)
    standardized = (flat - fitted["mean"]) / fitted["std"]
    weight = fitted["probe"].weight.float()
    direction = weight[labels] - weight[competitor]
    contribution = (
        standardized.reshape(-1, SLOTS, WIDTH)
        * direction.reshape(-1, SLOTS, WIDTH)
    ).sum(dim=-1)
    top = contribution.topk(TOP_K, dim=-1).indices
    low = contribution.topk(TOP_K, dim=-1, largest=False).indices
    return contribution, top, low, fitted["probe"](standardized)


def empty_parts(keys):
    fields = [
        "predictions", "fixed_margin", "decision_margin", "cross_entropy",
        "context_predictions", "context_margin", "context_delta_norm",
    ]
    return {key: {field: [] for field in fields} for key in keys}


def append_condition(parts, key, logits, target, competitor, context,
                     context_probe, source_context):
    task_logits = logits[:, -1, :16].float()
    rows = torch.arange(len(target), device=target.device)
    correct = task_logits[rows, target]
    fixed_margin = correct - task_logits[rows, competitor]
    masked = task_logits.clone()
    masked[rows, target] = -torch.inf
    decision_margin = correct - masked.max(dim=-1).values
    query_context = context[:, -1].float()
    context_logits = probe_logits(context_probe, query_context)
    parts[key]["predictions"].append(task_logits.argmax(dim=-1).cpu())
    parts[key]["fixed_margin"].append(fixed_margin.detach().cpu())
    parts[key]["decision_margin"].append(decision_margin.detach().cpu())
    parts[key]["cross_entropy"].append(
        F.cross_entropy(task_logits, target, reduction="none").detach().cpu()
    )
    parts[key]["context_predictions"].append(
        context_logits.argmax(dim=-1).detach().cpu()
    )
    parts[key]["context_margin"].append(
        (context_logits[rows, target] - context_logits[rows, competitor])
        .detach().cpu()
    )
    parts[key]["context_delta_norm"].append(
        (query_context - source_context.float()).norm(dim=-1).detach().cpu()
    )


@torch.no_grad()
def shared_prefix(model, controller, chunks, dtype):
    controller.configure()
    memory = None
    with torch.autocast(device_type="cuda", dtype=dtype):
        for chunk_index in range(CHUNKS - 1):
            _, memory = model(
                chunks[:, chunk_index], memory=memory,
                return_memory=True, per_layer_memory=True,
            )
    return memory


def run_final(model, controller, final_chunk, incoming_memory, dtype,
              selected=None, odds=1.0, context_patch=None,
              gradient_leaf=False):
    controller.configure(
        selected=selected, odds=odds, context_patch=context_patch,
        gradient_leaf=gradient_leaf,
    )
    context = torch.enable_grad() if gradient_leaf else torch.no_grad()
    with context, torch.autocast(device_type="cuda", dtype=dtype):
        logits, memory = model(
            final_chunk, memory=incoming_memory,
            return_memory=True, per_layer_memory=True,
        )
    return logits, memory, controller.snapshot()


def random_slots(batch, repeats, generator, device):
    output = []
    for _ in range(repeats):
        scores = torch.rand(batch, SLOTS, generator=generator)
        output.append(scores.topk(TOP_K, dim=-1).indices.to(device))
    return output


def collect(model, probes, args, device, dtype, root):
    keys = condition_keys(args)
    parts = empty_parts(keys)
    labels_parts = []
    confidence_parts = []
    competitor_parts = []
    memory_probe_prediction_parts = []
    top_slot_parts = []
    low_slot_parts = []
    contribution_parts = []
    attention_parts = {"source_top4_mass": [], "target_top4_mass": []}
    controller = ReadController(model)
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(args.random_slot_seed)
    total = 0
    returned_memory_exact = True
    gradient_reconstruction_max_abs = 0.0
    repeated_source_max_abs = 0.0
    source_correct_running = 0
    set_seed(args.dataset_seed)
    try:
        while total < args.samples:
            batch = min(args.eval_batch_size, args.samples - total)
            chunks, target, _ = make_chunks(
                batch, CHUNKS, args.chunk_size, device
            )
            incoming_memory = shared_prefix(model, controller, chunks, dtype)
            final_chunk = chunks[:, -1]
            source_logits, source_memory, source_trace = run_final(
                model, controller, final_chunk, incoming_memory, dtype
            )
            task_logits = source_logits[:, -1, :16].float()
            competitor = task_competitor(task_logits, target)
            top2 = task_logits.topk(2, dim=-1).values
            confidence = top2[:, 0] - top2[:, 1]
            contribution, top_slots, low_slots, memory_probe_logits = select_slots(
                source_trace["read_memory"], target, competitor,
                probes["memory_l3_concat"],
            )
            rolled_slots = torch.roll(top_slots, shifts=1, dims=0)
            random_selected = random_slots(
                batch, args.random_repeats, random_generator, device
            )

            gradient_logits, gradient_memory, gradient_trace = run_final(
                model, controller, final_chunk, incoming_memory, dtype,
                gradient_leaf=True,
            )
            gradient_task = gradient_logits[:, -1, :16].float()
            rows = torch.arange(batch, device=device)
            gradient_margin = (
                gradient_task[rows, target]
                - gradient_task[rows, competitor]
            )
            context_gradient = torch.autograd.grad(
                gradient_margin.sum(), controller.live_context,
                only_inputs=True,
            )[0][:, -1].detach().float()
            gradient_reconstruction_max_abs = max(
                gradient_reconstruction_max_abs,
                (gradient_task.detach() - task_logits).abs().max().item(),
            )
            returned_memory_exact = (
                returned_memory_exact
                and memories_equal(source_memory, gradient_memory)
            )

            source_context = source_trace["memory_context"][:, -1]
            append_condition(
                parts, "source", source_logits, target, competitor,
                source_trace["memory_context"], probes["memory_context"],
                source_context,
            )

            main_context = None
            for odds in ODDS:
                key = odds_key(odds)
                logits, produced, trace = run_final(
                    model, controller, final_chunk, incoming_memory, dtype,
                    selected=top_slots, odds=odds,
                )
                returned_memory_exact = (
                    returned_memory_exact
                    and memories_equal(source_memory, produced)
                )
                append_condition(
                    parts, key, logits, target, competitor,
                    trace["memory_context"], probes["memory_context"],
                    source_context,
                )
                if odds == MAIN_ODDS:
                    main_context = trace["memory_context"][:, -1].float()

            controls = [
                ("low4_odds4", low_slots),
                ("rolled_top4_odds4", rolled_slots),
            ]
            controls.extend(
                (f"random4_odds4_{index + 1}", selected)
                for index, selected in enumerate(random_selected)
            )
            for key, selected in controls:
                logits, produced, trace = run_final(
                    model, controller, final_chunk, incoming_memory, dtype,
                    selected=selected, odds=MAIN_ODDS,
                )
                returned_memory_exact = (
                    returned_memory_exact
                    and memories_equal(source_memory, produced)
                )
                append_condition(
                    parts, key, logits, target, competitor,
                    trace["memory_context"], probes["memory_context"],
                    source_context,
                )

            main_delta_norm = (
                main_context - source_context.float()
            ).norm(dim=-1)
            positive_query = source_context.float() + norm_match(
                context_gradient, main_delta_norm
            )
            positive_logits, positive_memory, positive_trace = run_final(
                model, controller, final_chunk, incoming_memory, dtype,
                context_patch=positive_query,
            )
            returned_memory_exact = (
                returned_memory_exact
                and memories_equal(source_memory, positive_memory)
            )
            append_condition(
                parts, "context_gradient_positive_control", positive_logits,
                target, competitor, positive_trace["memory_context"],
                probes["memory_context"], source_context,
            )

            source_mass = controller.selected_attention_mass(
                source_trace["read_query"], source_trace["read_memory"],
                top_slots, 1.0, dtype,
            )
            target_mass = controller.selected_attention_mass(
                source_trace["read_query"], source_trace["read_memory"],
                top_slots, MAIN_ODDS, dtype,
            )
            attention_parts["source_top4_mass"].append(source_mass.cpu())
            attention_parts["target_top4_mass"].append(target_mass.cpu())

            repeated_logits, repeated_memory, _ = run_final(
                model, controller, final_chunk, incoming_memory, dtype
            )
            repeated_source_max_abs = max(
                repeated_source_max_abs,
                (repeated_logits[:, -1, :16].float() - task_logits)
                .abs().max().item(),
            )
            returned_memory_exact = (
                returned_memory_exact
                and memories_equal(source_memory, repeated_memory)
            )

            labels_parts.append(target.cpu())
            confidence_parts.append(confidence.detach().cpu())
            competitor_parts.append(competitor.cpu())
            memory_probe_prediction_parts.append(
                memory_probe_logits.argmax(dim=-1).detach().cpu()
            )
            top_slot_parts.append(top_slots.cpu())
            low_slot_parts.append(low_slots.cpu())
            contribution_parts.append(contribution.detach().cpu().to(torch.float16))
            source_correct_running += int(
                (task_logits.argmax(dim=-1) == target).sum().item()
            )
            total += batch
            if total == batch or total % args.log_every_samples == 0:
                print(
                    f"Level 6.19.1 samples={total}/{args.samples} "
                    f"current_source={source_correct_running / total:.2%}",
                    flush=True,
                )
                save(root / "progress.json", {
                    "samples_complete": total,
                    "samples_total": args.samples,
                    "returned_memory_exact_so_far": returned_memory_exact,
                    "gradient_reconstruction_max_abs_so_far": (
                        gradient_reconstruction_max_abs
                    ),
                })
    finally:
        controller.close()

    labels = torch.cat(labels_parts)
    return {
        "labels": labels,
        "confidence": torch.cat(confidence_parts),
        "competitor": torch.cat(competitor_parts),
        "memory_probe_predictions": torch.cat(memory_probe_prediction_parts),
        "top_slots": torch.cat(top_slot_parts),
        "low_slots": torch.cat(low_slot_parts),
        "slot_contributions": torch.cat(contribution_parts),
        "attention": {
            key: torch.cat(value) for key, value in attention_parts.items()
        },
        "conditions": {
            key: {
                field: torch.cat(values)
                for field, values in fields.items()
            }
            for key, fields in parts.items()
        },
        "integrity": {
            "returned_memory_exactly_invariant": returned_memory_exact,
            "gradient_reconstruction_max_abs": gradient_reconstruction_max_abs,
            "gradient_reconstruction_exact": (
                gradient_reconstruction_max_abs == 0.0
            ),
            "repeated_source_logit_max_abs": repeated_source_max_abs,
            "repeated_source_logits_exact": repeated_source_max_abs == 0.0,
        },
    }


def paired_continuous(left, right, args, seed):
    values = right.double().numpy() - left.double().numpy()
    result = continuous_effect(values, args, seed)
    result["left_mean"] = left.double().mean().item()
    result["right_mean"] = right.double().mean().item()
    return result


def condition_metric(item, labels, ids):
    prediction = item["predictions"][ids]
    context_prediction = item["context_predictions"][ids]
    group_labels = labels[ids]
    return {
        "accuracy": (prediction == group_labels).float().mean().item(),
        "fixed_margin_mean": item["fixed_margin"][ids].mean().item(),
        "decision_margin_mean": item["decision_margin"][ids].mean().item(),
        "cross_entropy_mean": item["cross_entropy"][ids].mean().item(),
        "context_decoder_accuracy": (
            context_prediction == group_labels
        ).float().mean().item(),
        "context_margin_mean": item["context_margin"][ids].mean().item(),
        "context_delta_norm_mean": item["context_delta_norm"][ids].mean().item(),
        "samples": int(len(ids)),
    }


def subset_effect(source, updated, labels, ids, args, seed):
    return {
        "accuracy": paired_statistics(
            source["predictions"][ids], updated["predictions"][ids],
            labels[ids], args, seed,
        ),
        "fixed_margin": paired_continuous(
            source["fixed_margin"][ids], updated["fixed_margin"][ids],
            args, seed + 100,
        ),
        "context_margin": paired_continuous(
            source["context_margin"][ids], updated["context_margin"][ids],
            args, seed + 200,
        ),
        "context_decoder_accuracy": paired_statistics(
            source["context_predictions"][ids],
            updated["context_predictions"][ids], labels[ids], args,
            seed + 300,
        ),
    }


def effect_from_values(values, args, seed):
    return continuous_effect(values.double().numpy(), args, seed)


def analyze(collected, args):
    labels = collected["labels"]
    conditions = collected["conditions"]
    source = conditions["source"]
    source_correct = source["predictions"] == labels
    errors = torch.where(~source_correct)[0]
    correct = torch.where(source_correct)[0]
    if len(errors) < args.minimum_errors:
        raise RuntimeError(
            f"Only {len(errors)} source errors; minimum is {args.minimum_errors}"
        )
    matched, matching = match_confidence(
        errors, correct, collected["confidence"]
    )
    memory_correct = collected["memory_probe_predictions"] == labels
    memory_errors = torch.where((~source_correct) & memory_correct)[0]
    if len(memory_errors) < args.minimum_memory_decodable_errors:
        raise RuntimeError(
            f"Only {len(memory_errors)} Memory-decodable source errors; "
            f"minimum is {args.minimum_memory_decodable_errors}"
        )
    groups = {
        "all": torch.arange(len(labels)),
        "source_errors": errors,
        "memory_decodable_errors": memory_errors,
        "confidence_matched_correct": matched,
        "source_correct": correct,
    }
    metrics = {
        group: {
            key: condition_metric(item, labels, ids)
            for key, item in conditions.items()
        }
        for group, ids in groups.items()
    }
    main_key = odds_key(MAIN_ODDS)
    effects = {
        group: {
            key: subset_effect(
                source, item, labels, ids, args,
                args.analysis_seed + group_index * 1000 + condition_index * 10,
            )
            for condition_index, (key, item) in enumerate(conditions.items())
            if key != "source"
        }
        for group_index, (group, ids) in enumerate(groups.items())
    }

    ids = memory_errors
    random_keys = [
        f"random4_odds4_{index + 1}"
        for index in range(args.random_repeats)
    ]
    random_context = torch.stack([
        conditions[key]["context_margin"] for key in random_keys
    ]).mean(dim=0)
    random_correct = torch.stack([
        (conditions[key]["predictions"] == labels).float()
        for key in random_keys
    ]).mean(dim=0)
    main_context = conditions[main_key]["context_margin"]
    main_correct = (conditions[main_key]["predictions"] == labels).float()
    source_correct_float = source_correct.float()
    context_primary_values = {
        "top4_vs_source": main_context[ids] - source["context_margin"][ids],
        "top4_vs_random_mean": main_context[ids] - random_context[ids],
        "top4_vs_low4": (
            main_context[ids] - conditions["low4_odds4"]["context_margin"][ids]
        ),
        "top4_vs_rolled_top4": (
            main_context[ids]
            - conditions["rolled_top4_odds4"]["context_margin"][ids]
        ),
    }
    behavior_primary_values = {
        "top4_vs_source": main_correct[ids] - source_correct_float[ids],
        "top4_vs_random_mean": main_correct[ids] - random_correct[ids],
        "top4_vs_low4": main_correct[ids] - (
            conditions["low4_odds4"]["predictions"][ids] == labels[ids]
        ).float(),
        "top4_vs_rolled_top4": main_correct[ids] - (
            conditions["rolled_top4_odds4"]["predictions"][ids] == labels[ids]
        ).float(),
    }
    context_primary = {
        key: effect_from_values(
            value, args, args.analysis_seed + 10000 + index
        )
        for index, (key, value) in enumerate(context_primary_values.items())
    }
    behavior_primary = {
        key: effect_from_values(
            value, args, args.analysis_seed + 10100 + index
        )
        for index, (key, value) in enumerate(behavior_primary_values.items())
    }
    context_holm = holm_adjust({
        key: value["sign_flip_p_two_sided"]
        for key, value in context_primary.items()
    })
    behavior_holm = holm_adjust({
        key: value["sign_flip_p_two_sided"]
        for key, value in behavior_primary.items()
    })
    for key in context_primary:
        context_primary[key]["multiplicity"] = context_holm[key]
        behavior_primary[key]["multiplicity"] = behavior_holm[key]

    positive = conditions["context_gradient_positive_control"]
    positive_control = {
        "fixed_margin": paired_continuous(
            source["fixed_margin"][ids], positive["fixed_margin"][ids],
            args, args.analysis_seed + 10200,
        ),
        "accuracy": paired_statistics(
            source["predictions"][ids], positive["predictions"][ids],
            labels[ids], args, args.analysis_seed + 10201,
        ),
        "mean_equalized_context_delta_norm": positive[
            "context_delta_norm"
        ][ids].mean().item(),
    }
    attention_change = paired_continuous(
        collected["attention"]["source_top4_mass"][ids],
        collected["attention"]["target_top4_mass"][ids],
        args, args.analysis_seed + 10300,
    )

    context_specific = all(
        row["estimate"] > 0
        and row["multiplicity"]["significant_0.05"]
        for row in context_primary.values()
    )
    behavior_specific = all(
        row["estimate"] > 0
        and row["multiplicity"]["significant_0.05"]
        for row in behavior_primary.values()
    )
    positive_operational = (
        positive_control["fixed_margin"]["estimate"] > 0
        and positive_control["fixed_margin"]["sign_flip_p_two_sided"] < 0.05
    )
    all_retention = effects["all"][main_key]["accuracy"][
        "accuracy_change"
    ]["ci95"][0] >= -args.full_noninferiority
    matched_retention = effects["confidence_matched_correct"][main_key][
        "accuracy"
    ]["accuracy_change"]["ci95"][0] >= -args.matched_noninferiority
    retention = all_retention and matched_retention

    if context_specific and behavior_specific:
        if retention:
            classification = "selective_slot_read_causally_supported"
            boundary = (
                "Replicate the frozen top-k read intervention on a disjoint "
                "panel before considering a learned router."
            )
        else:
            classification = "selective_slot_read_effect_not_safe"
            boundary = (
                "The targeted read has a specific effect but fails registered "
                "retention; do not convert it into a learned router."
            )
    elif context_specific and positive_operational:
        classification = "selective_slot_read_context_causal_but_behavior_insufficient"
        boundary = (
            "The targeted read reaches context but does not selectively rescue "
            "behavior; test the fused-to-residual boundary next."
        )
    elif positive_operational:
        classification = "linear_slot_ranking_not_causally_sufficient"
        boundary = (
            "The downstream path responds to equal-dose context improvement, "
            "but linear-probe slot targeting is not a sufficient read mechanism."
        )
    else:
        classification = "registered_dose_or_downstream_positive_control_failed"
        boundary = (
            "Do not broaden the intervention until context dose and the "
            "downstream positive control are independently calibrated."
        )

    matching.update({
        "errors": int(len(errors)),
        "matched_correct": int(len(matched)),
        "memory_decodable_errors": int(len(memory_errors)),
        "error_confidence_mean": collected["confidence"][errors].mean().item(),
        "matched_confidence_mean": collected["confidence"][matched].mean().item(),
    })
    diagnosis = {
        "classification": classification,
        "source_accuracy": source_correct.float().mean().item(),
        "source_errors": int(len(errors)),
        "memory_probe_correct_on_source_errors": memory_correct[
            errors
        ].float().mean().item(),
        "memory_decodable_errors": int(len(memory_errors)),
        "main_condition": main_key,
        "context_specificity_passed": context_specific,
        "behavior_specificity_passed": behavior_specific,
        "positive_control_operational": positive_operational,
        "full_panel_retention_passed": all_retention,
        "matched_correct_retention_passed": matched_retention,
        "registered_next_boundary": boundary,
    }
    return {
        "matching": matching,
        "metrics": metrics,
        "effects_vs_source": effects,
        "primary": {
            "population": "Memory-decodable source errors",
            "context_margin_specificity": context_primary,
            "deployed_behavior_specificity": behavior_primary,
            "attention_targeting_check": attention_change,
            "context_gradient_positive_control": positive_control,
        },
        "diagnosis": diagnosis,
    }, groups


def plot_result(analysis, path):
    main = odds_key(MAIN_ODDS)
    metrics = analysis["metrics"]["memory_decodable_errors"]
    labels = [
        "Source", "Top4 x2", "Top4 x4", "Top4 x8", "Low4 x4",
        "Rolled x4", "Positive\ncontext",
    ]
    keys = [
        "source", odds_key(2.0), main, odds_key(8.0), "low4_odds4",
        "rolled_top4_odds4", "context_gradient_positive_control",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].bar(
        np.arange(len(keys)),
        [100 * metrics[key]["context_decoder_accuracy"] for key in keys],
        color=["#4C78A8", "#59A14F", "#59A14F", "#59A14F", "#E15759", "#F28E2B", "#B07AA1"],
    )
    axes[0].set_xticks(np.arange(len(keys)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Context decoder accuracy (%)")
    axes[0].set_title("Memory-decodable source errors")

    source_context = metrics["source"]["context_margin_mean"]
    source_deployed = metrics["source"]["fixed_margin_mean"]
    axes[1].bar(
        np.arange(len(keys)),
        [metrics[key]["context_margin_mean"] - source_context for key in keys],
        color="#59A14F",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(keys)), labels, rotation=25, ha="right")
    axes[1].set_ylabel("Context probe margin change")
    axes[1].set_title("Read-context information change")

    axes[2].bar(
        np.arange(len(keys)),
        [metrics[key]["fixed_margin_mean"] - source_deployed for key in keys],
        color="#B07AA1",
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(np.arange(len(keys)), labels, rotation=25, ha="right")
    axes[2].set_ylabel("Deployed correct-rival margin change")
    axes[2].set_title("Frozen downstream response")
    figure.suptitle(
        "IST Level 6.19.1: Selective Slot-Read Causal Intervention",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def preregistration(args):
    return {
        "level": "6.19.1",
        "status": "frozen selective slot-read causal intervention",
        "source": "formally passed Level 6.18.3 seed707 checkpoint",
        "slot_code": "frozen Level 6.19 independent all-slot Memory probe",
        "seed": SEED,
        "chunks": CHUNKS,
        "intervention": {
            "location": "final block MemoryAttention, final query only",
            "mechanism": "add log(odds) to selected slot attention logits",
            "top_k": TOP_K,
            "dose_odds": ODDS,
            "registered_main_dose": MAIN_ODDS,
            "selection": (
                "per-example correct-label versus frozen deployed-rival "
                "contribution under the frozen Level 6.19 Memory probe"
            ),
            "interpretation": "label-aware mechanism intervention, not deployment",
        },
        "equal_dose_controls": {
            "low_contribution": "bottom four slots at odds 4",
            "cross_example_roll": "another batch example's top four slots at odds 4",
            "random": f"{args.random_repeats} independent random four-slot controls at odds 4",
            "no_op": "unmodified frozen source",
        },
        "positive_control": {
            "location": "final-query read context",
            "direction": "deployed correct-versus-rival context gradient",
            "dose": "per-example L2 norm matched to top4 odds-4 context change",
        },
        "primary_population": "source errors whose frozen Memory probe is correct",
        "primary_context_family": [
            "top4 odds4 versus source",
            "top4 odds4 versus random-control mean",
            "top4 odds4 versus low4 odds4",
            "top4 odds4 versus rolled-top4 odds4",
        ],
        "primary_behavior_family": [
            "the same four contrasts using deployed correctness"
        ],
        "multiplicity": "Holm within each four-test family",
        "decision_rule": {
            "context_specificity": (
                "all four context-margin estimates positive with Holm p < 0.05"
            ),
            "behavior_specificity": (
                "all four deployed-correctness estimates positive with Holm p < 0.05"
            ),
            "full_accuracy_noninferiority": args.full_noninferiority,
            "matched_correct_noninferiority": args.matched_noninferiority,
            "positive_control": "positive fixed-margin estimate with p < 0.05",
        },
        "integrity": {
            "all_model_and_probe_parameters_frozen": True,
            "returned_persistent_memory_must_be_exactly_invariant": True,
            "gradient-leaf forward_must_reconstruct_source_logits_exactly": True,
            "repeat_source_forward_must_be_exact": True,
            "failed_Level_6_18_9_candidate_not_used": True,
            "protected_tests_not_used": True,
            "seed909_locked": True,
        },
        "protocol": {
            key: value for key, value in vars(args).items()
            if key not in {"dry_run", "smoke_test", "force"}
        },
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError("Level 6.19.1 is fixed to seed707 at 16 chunks")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    if not Path(args.probes).exists():
        raise FileNotFoundError(args.probes)
    if args.samples <= 0 or args.eval_batch_size <= 0:
        raise ValueError("samples and eval-batch-size must be positive")
    if args.samples % args.eval_batch_size != 0:
        raise ValueError("samples must be divisible by eval-batch-size")
    if args.random_repeats <= 0:
        raise ValueError("random-repeats must be positive")
    if not args.smoke_test and (
        args.samples != 4096
        or args.random_repeats != 4
        or args.dataset_seed != 6192100
    ):
        raise ValueError(
            "Formal Level 6.19.1 fixes samples=4096, random-repeats=4, "
            "and dataset-seed=6192100; use --smoke-test for reduced checks"
        )


def raw_predictions(collected, groups):
    return {
        "labels": collected["labels"].tolist(),
        "confidence": collected["confidence"].tolist(),
        "competitor": collected["competitor"].tolist(),
        "memory_probe_predictions": collected[
            "memory_probe_predictions"
        ].tolist(),
        "groups": {name: value.tolist() for name, value in groups.items()},
        "top_slots": collected["top_slots"].tolist(),
        "low_slots": collected["low_slots"].tolist(),
        "slot_contributions": collected["slot_contributions"].tolist(),
        "attention": {
            key: value.tolist() for key, value in collected["attention"].items()
        },
        "conditions": {
            key: {name: value.tolist() for name, value in row.items()}
            for key, row in collected["conditions"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.19.1 frozen selective slot-read intervention"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint",
        default="experiments/level6_18_3/formal/rescued_head_checkpoint.pt",
    )
    parser.add_argument(
        "--probes", default="experiments/level6_19/formal/linear_probes.pt"
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--random-repeats", type=int, default=4)
    parser.add_argument("--dataset-seed", type=int, default=6192100)
    parser.add_argument("--random-slot-seed", type=int, default=6191200)
    parser.add_argument("--analysis-seed", type=int, default=6191300)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--minimum-errors", type=int, default=200)
    parser.add_argument(
        "--minimum-memory-decodable-errors", type=int, default=150
    )
    parser.add_argument("--full-noninferiority", type=float, default=0.01)
    parser.add_argument("--matched-noninferiority", type=float, default=0.02)
    parser.add_argument("--log-every-samples", type=int, default=256)
    parser.add_argument("--output", default="experiments/level6_19_1/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    # Smoke panels are code checks only and are disjoint from the unopened
    # formal panel even when all seed flags are left at their defaults.
    if args.smoke_test:
        args.dataset_seed += 50_000_000
        args.random_slot_seed += 50_000_000
        args.analysis_seed += 50_000_000
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
    model, original_probe, probes, checkpoint_meta = load_frozen(args, device)
    model_before = tensor_fingerprint(model)
    original_probe_before = tensor_fingerprint(original_probe)
    level6_19_probes_before = {
        key: tensor_fingerprint(value["probe"])
        for key, value in probes.items()
    }
    collected = collect(model, probes, args, device, dtype, root)
    analysis, groups = analyze(collected, args)
    integrity = {
        **collected["integrity"],
        "all_model_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "all_probe_parameters_frozen": all(
            not parameter.requires_grad
            for module in [original_probe] + [
                value["probe"] for value in probes.values()
            ]
            for parameter in module.parameters()
        ),
        "model_state_unchanged": model_before == tensor_fingerprint(model),
        "original_probe_state_unchanged": (
            original_probe_before == tensor_fingerprint(original_probe)
        ),
        "level6_19_probe_states_unchanged": all(
            level6_19_probes_before[key]
            == tensor_fingerprint(value["probe"])
            for key, value in probes.items()
        ),
        "failed_candidate_not_used": True,
        "protected_tests_not_used": True,
        "seed909_locked": True,
    }
    integrity["passed"] = all([
        integrity["returned_memory_exactly_invariant"],
        integrity["gradient_reconstruction_exact"],
        integrity["repeated_source_logits_exact"],
        integrity["all_model_parameters_frozen"],
        integrity["all_probe_parameters_frozen"],
        integrity["model_state_unchanged"],
        integrity["original_probe_state_unchanged"],
        integrity["level6_19_probe_states_unchanged"],
    ])
    analysis["diagnosis"]["integrity_passed"] = integrity["passed"]
    if not integrity["passed"]:
        analysis["diagnosis"]["classification"] = "integrity_failure"
        analysis["diagnosis"]["registered_next_boundary"] = (
            "Stop; repair the intervention implementation before interpretation."
        )
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
        "primary": analysis["primary"],
        "diagnosis": analysis["diagnosis"],
    })
    save(root / "predictions.json", raw_predictions(collected, groups))
    plot_result(analysis, root / "selective_slot_read_intervention.png")
    print(json.dumps(analysis["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
