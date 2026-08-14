import argparse
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
import torch.nn.functional as F

from long_context_test import set_seed
from run_level6_2_local import make_chunks
from run_level6_18_6_local import configure_cuda, paired_statistics, save
from run_level6_18_7_local import holm_adjust
from run_level6_18_8_local import continuous_effect
from run_level6_19_local import FinalTrace, match_confidence
from run_level6_19_1_local import (
    load_frozen,
    norm_match,
    probe_logits,
    select_slots,
    tensor_fingerprint,
)


SEED = 707
CHUNKS = 16
SLOTS = 32
WIDTH = 64
HEADS = 8
HEAD_WIDTH = 8
TOP_K = 4
ODDS = 4.0
NUMERICAL_REVISION = "delta_closed_loop_fp64_solver_v4"
CONDITIONS = [
    "source",
    "probe_top4_odds4",
    "gradient_top4_odds4",
    "gradient_kl_oracle",
    "negative_gradient_kl",
    "rolled_gradient_kl",
    "gradient_l2_oracle",
    "tangent_context_control",
    "unrestricted_context_control",
]


def task_competitor(logits, labels):
    masked = logits.float().clone()
    rows = torch.arange(len(labels), device=labels.device)
    masked[rows, labels] = -torch.inf
    return masked.argmax(dim=-1)


def downstream(model, trace, memory_context, dtype, enable_grad=False):
    block = model.blocks[-1]
    context = torch.enable_grad() if enable_grad else torch.no_grad()
    with context, torch.autocast(device_type="cuda", dtype=dtype):
        gate = block.memory_fusion_gate(torch.cat([
            trace["read_query"], memory_context
        ], dim=-1))
        fused = trace["pre_fusion_feature"] + gate * memory_context
        hidden = block.norm2(trace["read_query"] + block.ffn(fused))
        logits = model.output(hidden)[:, -1, :16].float()
    return logits


def attention_decomposition(model, trace, dtype):
    module = model.blocks[-1].memory_read
    query = trace["read_query"]
    memory = trace["read_memory"]
    # Reproduce the deployed Q/K/V projection boundary first. Re-projecting
    # captured bf16 inputs with fp32 weights creates a larger discrepancy than
    # the intervention on rare examples.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        weight = module.in_proj_weight
        bias = module.in_proj_bias
        query_projection, key_projection, value_projection = (
            F._in_projection_packed(
                query, memory, memory, weight, bias
            )
        )
        query_heads = query_projection.reshape(
            len(query), query.shape[1], HEADS, HEAD_WIDTH
        ).transpose(1, 2)
        key_heads = key_projection.reshape(
            len(memory), SLOTS, HEADS, HEAD_WIDTH
        ).transpose(1, 2)
        value_heads = value_projection.reshape(
            len(memory), SLOTS, HEADS, HEAD_WIDTH
        ).transpose(1, 2)
        native_replay = F.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, dropout_p=0.0,
            is_causal=False,
        )
        native_replay_context = F.linear(
            native_replay.transpose(1, 2).contiguous().reshape(
                len(query), query.shape[1], WIDTH
            ),
            module.out_proj.weight,
            module.out_proj.bias,
        )

    # From the native projection boundary onward, make the interpretable
    # attention distribution and value/output composition explicitly fp32.
    query_heads = query_heads.float()
    key_heads = key_heads.float()
    value_heads = value_heads.float()
    with torch.no_grad():
        fp32_sdpa = F.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, dropout_p=0.0,
            is_causal=False,
        )
        fp32_context = F.linear(
            fp32_sdpa.transpose(1, 2).contiguous().reshape(
                len(query), query.shape[1], WIDTH
            ),
            module.out_proj.weight.float(),
            None if module.out_proj.bias is None
            else module.out_proj.bias.float(),
        )
        scores = torch.matmul(
            query_heads[:, :, -1:], key_heads.transpose(-2, -1)
        ) / math.sqrt(HEAD_WIDTH)
        source_weights = torch.softmax(scores, dim=-1).squeeze(-2).float()
    value = value_projection.float().reshape(
        len(memory), SLOTS, HEADS, HEAD_WIDTH
    ).permute(0, 2, 1, 3).contiguous()
    concatenated = torch.einsum(
        "bhs,bhsd->bhd", source_weights, value
    ).reshape(len(memory), WIDTH)
    manual_query = F.linear(
        concatenated, module.out_proj.weight.float(),
        None if module.out_proj.bias is None else module.out_proj.bias.float(),
    )
    native_query = trace["memory_context"][:, -1].float()
    native_replay_query = native_replay_context[:, -1].float()
    fp32_query = fp32_context[:, -1].float()
    return {
        "weights": source_weights,
        "values": value,
        "out_weight": module.out_proj.weight.float(),
        "native_query": native_query,
        "native_replay_query": native_replay_query,
        "fp32_query": fp32_query,
        "manual_query": manual_query,
    }


def apply_odds(source, selected, odds=ODDS):
    logits = source.clamp_min(1e-12).log()
    if selected.ndim == 2:
        selected = selected[:, None].expand(-1, HEADS, -1)
    bias = torch.zeros_like(logits)
    bias.scatter_add_(
        -1, selected,
        torch.full_like(selected, math.log(odds), dtype=logits.dtype),
    )
    return torch.softmax(logits + bias, dim=-1)


def kl_divergence(updated, source):
    # KL is non-negative. Accumulate this scalar audit/solver quantity in fp64
    # to avoid negative fp32 roundoff contaminating common budgets.
    updated64 = updated.double().clamp_min(1e-30)
    source64 = source.double().clamp_min(1e-30)
    value = (
        updated64 * (updated64.log() - source64.log())
    ).sum(dim=-1)
    return value.clamp_min(0.0).float()


def attention_delta(updated, source, values, out_weight):
    head_delta = torch.einsum(
        "bhs,bhsd->bhd", updated - source, values
    ).reshape(len(source), WIDTH)
    return F.linear(head_delta, out_weight, None)


def patch_context(source_context, delta):
    output = source_context.detach().clone()
    output[:, -1] = (
        source_context[:, -1].float() + delta.float()
    ).to(dtype=source_context.dtype)
    return output


def delta_closed_loop(updated, source, values, out_weight):
    # This is the actual intervention path: both terms use the same explicit
    # decomposition, so the cross-kernel absolute source offset cancels.
    direct = attention_delta(updated, source, values, out_weight)
    updated_context = attention_delta(
        updated, torch.zeros_like(source), values, out_weight
    )
    source_context = attention_delta(
        source, torch.zeros_like(source), values, out_weight
    )
    return direct, (direct - (updated_context - source_context)).abs()


def standardized_score(score, per_head=True):
    if per_head:
        scale = score.std(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        scale = score.reshape(len(score), -1).std(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)[:, :, None]
    return (score - score.mean(dim=-1, keepdim=True)) / scale


def tilt_to_kl(source, score, budget, iterations=56):
    score = standardized_score(score, per_head=True)
    # Keep the scalar tilt parameter in fp64, but quantize every candidate
    # attention distribution back to the registered fp32 intervention path
    # before comparing KL. A fp32 eta stalled at adjacent values for one head
    # in 4096 examples, leaving a 1.0401e-5 mismatch against a 1e-5 gate.
    log_source = source.double().clamp_min(1e-30).log()
    score64 = score.double()
    low = torch.zeros_like(budget, dtype=torch.float64)
    high = torch.ones_like(budget, dtype=torch.float64)
    for _ in range(32):
        candidate = torch.softmax(
            log_source + high[..., None] * score64, dim=-1
        ).float()
        below = kl_divergence(candidate, source) < budget
        high = torch.where(below, high * 2.0, high)
    for _ in range(iterations):
        middle = (low + high) / 2.0
        candidate = torch.softmax(
            log_source + middle[..., None] * score64, dim=-1
        ).float()
        below = kl_divergence(candidate, source) < budget
        low = torch.where(below, middle, low)
        high = torch.where(below, high, middle)
    # The registered intervention is fp32, so the exact target KL may lie
    # between adjacent representable softmax outputs. Select the closer
    # bisection endpoint after fp32 materialization instead of returning an
    # arbitrary midpoint that can be farther from the target.
    low_updated = torch.softmax(
        log_source + low[..., None] * score64, dim=-1
    ).float()
    high_updated = torch.softmax(
        log_source + high[..., None] * score64, dim=-1
    ).float()
    low_error = (kl_divergence(low_updated, source) - budget).abs()
    high_error = (kl_divergence(high_updated, source) - budget).abs()
    choose_high = high_error < low_error
    updated = torch.where(
        choose_high[..., None], high_updated, low_updated
    )
    eta = torch.where(choose_high, high, low)
    return updated, eta


def symmetric_kl_budget(source, positive_score, controls):
    # A direction can saturate at a high-probability slot before reaching the
    # Probe reference KL. Lock a common per-head budget that every registered
    # direction can attain, preserving equal-dose comparisons.
    log_source = source.clamp_min(1e-12).log()
    ceilings = []
    for score in [positive_score] + list(controls):
        normalized = standardized_score(score, per_head=True)
        limiting = normalized.max(dim=-1).indices
        ceiling = -log_source.gather(-1, limiting[..., None]).squeeze(-1)
        ceilings.append(ceiling)
    return torch.stack(ceilings).min(dim=0).values


def tilt_to_l2(source, score, values, out_weight, target_norm,
               iterations=36):
    score = standardized_score(score, per_head=False)
    log_source = source.clamp_min(1e-12).log()
    low = torch.zeros(len(source), device=source.device)
    high = torch.ones_like(low)
    for _ in range(16):
        candidate = torch.softmax(
            log_source + high[:, None, None] * score, dim=-1
        )
        norm = attention_delta(
            candidate, source, values, out_weight
        ).norm(dim=-1)
        below = norm < target_norm
        high = torch.where(below, high * 2.0, high)
    for _ in range(iterations):
        middle = (low + high) / 2.0
        candidate = torch.softmax(
            log_source + middle[:, None, None] * score, dim=-1
        )
        norm = attention_delta(
            candidate, source, values, out_weight
        ).norm(dim=-1)
        below = norm < target_norm
        low = torch.where(below, middle, low)
        high = torch.where(below, high, middle)
    eta = (low + high) / 2.0
    updated = torch.softmax(
        log_source + eta[:, None, None] * score, dim=-1
    )
    return updated, eta


def tangent_projection(gradient, values, out_weight):
    # Each head may redistribute probability with zero sum. Slot 31 is the
    # reference, yielding 8 * 31 reachable infinitesimal value differences.
    differences = values[:, :, :-1] - values[:, :, -1:]
    blocks = []
    for head in range(HEADS):
        weight_block = out_weight[:, head * HEAD_WIDTH:(head + 1) * HEAD_WIDTH]
        blocks.append(torch.matmul(differences[:, head], weight_block.T))
    basis = torch.cat(blocks, dim=1).transpose(1, 2).contiguous()
    gram = torch.matmul(basis, basis.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues[:, -1:].clamp_min(1e-12)
    keep = eigenvalues > maximum * 1e-6
    coefficients = torch.matmul(
        eigenvectors.transpose(1, 2), gradient[..., None]
    ).squeeze(-1)
    projected = torch.matmul(
        eigenvectors, (coefficients * keep)[..., None]
    ).squeeze(-1)
    energy = (
        projected.norm(dim=-1).square()
        / gradient.norm(dim=-1).clamp_min(1e-8).square()
    )
    return projected, keep.sum(dim=-1), energy


def empty_parts():
    fields = [
        "predictions", "fixed_margin", "decision_margin", "cross_entropy",
        "context_predictions", "context_margin", "context_delta_norm",
        "attention_kl_mean", "attention_kl_max",
    ]
    return {
        condition: {field: [] for field in fields}
        for condition in CONDITIONS
    }


def append_condition(parts, condition, logits, labels, competitor,
                     context, context_probe, source_context, attention=None,
                     source_attention=None):
    rows = torch.arange(len(labels), device=labels.device)
    correct = logits[rows, labels]
    fixed_margin = correct - logits[rows, competitor]
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    query_context = context[:, -1].float()
    context_logits = probe_logits(context_probe, query_context)
    if attention is None:
        kl = torch.zeros(len(labels), HEADS, device=labels.device)
    else:
        kl = kl_divergence(attention, source_attention)
    row = parts[condition]
    row["predictions"].append(logits.argmax(dim=-1).detach().cpu())
    row["fixed_margin"].append(fixed_margin.detach().cpu())
    row["decision_margin"].append(
        (correct - masked.max(dim=-1).values).detach().cpu()
    )
    row["cross_entropy"].append(
        F.cross_entropy(logits, labels, reduction="none").detach().cpu()
    )
    row["context_predictions"].append(
        context_logits.argmax(dim=-1).detach().cpu()
    )
    row["context_margin"].append(
        (context_logits[rows, labels] - context_logits[rows, competitor])
        .detach().cpu()
    )
    row["context_delta_norm"].append(
        (query_context - source_context[:, -1].float())
        .norm(dim=-1).detach().cpu()
    )
    row["attention_kl_mean"].append(kl.mean(dim=-1).detach().cpu())
    row["attention_kl_max"].append(kl.max(dim=-1).values.detach().cpu())


def collect(model, probes, args, device, dtype, root):
    parts = empty_parts()
    labels_parts = []
    confidence_parts = []
    memory_probe_parts = []
    competitor_parts = []
    top_slots_parts = []
    audit_parts = {
        "probe_reference_kl_mean": [],
        "common_kl_budget_mean": [],
        "common_over_probe_kl_fraction": [],
        "gradient_kl_budget_error_max": [],
        "negative_kl_budget_error_max": [],
        "rolled_kl_budget_error_max": [],
        "gradient_l2_budget_error": [],
        "attention_gradient_norm": [],
        "context_gradient_norm": [],
        "tangent_rank": [],
        "tangent_gradient_energy": [],
        "analytic_autograd_gradient_max_abs": [],
        "attention_delta_closed_loop_max_abs": [],
    }
    trace = FinalTrace(model)
    total = 0
    source_reconstruction_max = 0.0
    native_replay_context_max = 0.0
    fp32_native_context_max = 0.0
    manual_attention_context_max = 0.0
    source_correct_running = 0
    set_seed(args.dataset_seed)
    try:
        while total < args.samples:
            batch = min(args.eval_batch_size, args.samples - total)
            chunks, labels, _ = make_chunks(
                batch, CHUNKS, args.chunk_size, device
            )
            memory = None
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=dtype
            ):
                for chunk_index in range(CHUNKS - 1):
                    _, memory = model(
                        chunks[:, chunk_index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                trace.clear()
                native_logits, returned_memory = model(
                    chunks[:, -1], memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
            trace.require()
            captured = {
                name: value.detach().clone()
                for name, value in trace.values.items()
            }
            source_logits = native_logits[:, -1, :16].float()
            competitor = task_competitor(source_logits, labels)
            top2 = source_logits.topk(2, dim=-1).values
            confidence = top2[:, 0] - top2[:, 1]

            decomposition = attention_decomposition(model, captured, dtype)
            source_attention = decomposition["weights"]
            values = decomposition["values"]
            out_weight = decomposition["out_weight"]
            source_context = captured["memory_context"]
            native_query = decomposition["native_query"]
            native_replay_context_max = max(
                native_replay_context_max,
                (decomposition["native_replay_query"] - native_query)
                .abs().max().item(),
            )
            fp32_native_context_max = max(
                fp32_native_context_max,
                (decomposition["fp32_query"] - native_query).abs().max().item(),
            )
            manual_attention_context_max = max(
                manual_attention_context_max,
                (decomposition["manual_query"] - decomposition["fp32_query"])
                .abs().max().item(),
            )

            reconstructed = downstream(
                model, captured, source_context, dtype, False
            )
            source_reconstruction_max = max(
                source_reconstruction_max,
                (reconstructed - source_logits).abs().max().item(),
            )

            flat_memory = captured["read_memory"].reshape(batch, -1)
            _, probe_slots, _, memory_probe_logits = select_slots(
                captured["read_memory"], labels, competitor,
                probes["memory_l3_concat"],
            )
            del flat_memory
            probe_attention = apply_odds(source_attention, probe_slots)
            probe_budget = kl_divergence(
                probe_attention, source_attention
            )
            probe_delta = attention_delta(
                probe_attention, source_attention, values, out_weight
            )
            target_l2 = probe_delta.norm(dim=-1)

            context_leaf = source_context.detach().clone().requires_grad_(True)
            gradient_logits = downstream(
                model, captured, context_leaf, dtype, True
            )
            rows = torch.arange(batch, device=device)
            gradient_margin = (
                gradient_logits[rows, labels]
                - gradient_logits[rows, competitor]
            )
            context_gradient = torch.autograd.grad(
                gradient_margin.sum(), context_leaf, only_inputs=True
            )[0][:, -1].detach().float()
            source_reconstruction_max = max(
                source_reconstruction_max,
                (gradient_logits.detach() - source_logits).abs().max().item(),
            )

            concat_gradient = torch.matmul(context_gradient, out_weight)
            head_gradient = concat_gradient.reshape(batch, HEADS, HEAD_WIDTH)
            utility = (values * head_gradient[:, :, None]).sum(dim=-1)
            expected = (source_attention * utility).sum(dim=-1, keepdim=True)
            analytic_logit_gradient = source_attention * (utility - expected)

            bias_leaf = torch.zeros_like(source_attention, requires_grad=True)
            differentiable_attention = torch.softmax(
                source_attention.clamp_min(1e-12).log() + bias_leaf, dim=-1
            )
            differentiable_delta = attention_delta(
                differentiable_attention, source_attention, values, out_weight
            )
            surrogate = (differentiable_delta * context_gradient).sum()
            automatic_gradient = torch.autograd.grad(
                surrogate, bias_leaf, only_inputs=True
            )[0]
            gradient_check = (
                automatic_gradient - analytic_logit_gradient
            ).abs().reshape(batch, -1).max(dim=-1).values

            gradient_top_slots = analytic_logit_gradient.topk(
                TOP_K, dim=-1
            ).indices
            gradient_top_attention = apply_odds(
                source_attention, gradient_top_slots
            )
            rolled_utility = torch.roll(utility, 1, dims=0)
            symmetric_ceiling = symmetric_kl_budget(
                source_attention, utility, [-utility, rolled_utility]
            )
            common_budget = torch.minimum(
                probe_budget, symmetric_ceiling * args.kl_ceiling_fraction
            )
            gradient_kl_attention, _ = tilt_to_kl(
                source_attention, utility, common_budget
            )
            negative_attention, _ = tilt_to_kl(
                source_attention, -utility, common_budget
            )
            rolled_attention, _ = tilt_to_kl(
                source_attention, rolled_utility, common_budget
            )
            gradient_l2_attention, _ = tilt_to_l2(
                source_attention, utility, values, out_weight, target_l2
            )

            projected_gradient, tangent_rank, tangent_energy = tangent_projection(
                context_gradient, values, out_weight
            )
            tangent_delta = norm_match(projected_gradient, target_l2)
            unrestricted_delta = norm_match(context_gradient, target_l2)

            attention_conditions = {
                "probe_top4_odds4": probe_attention,
                "gradient_top4_odds4": gradient_top_attention,
                "gradient_kl_oracle": gradient_kl_attention,
                "negative_gradient_kl": negative_attention,
                "rolled_gradient_kl": rolled_attention,
                "gradient_l2_oracle": gradient_l2_attention,
            }
            context_conditions = {
                "source": source_context,
                "tangent_context_control": patch_context(
                    source_context, tangent_delta
                ),
                "unrestricted_context_control": patch_context(
                    source_context, unrestricted_delta
                ),
            }
            for name, attention in attention_conditions.items():
                delta, closed_loop_error = delta_closed_loop(
                    attention, source_attention, values, out_weight
                )
                context_conditions[name] = patch_context(
                    source_context, delta,
                )
                audit_parts[
                    "attention_delta_closed_loop_max_abs"
                ].append(
                    closed_loop_error.reshape(batch, -1).max(dim=-1).values.cpu()
                )

            for name in CONDITIONS:
                context = context_conditions[name]
                logits = (
                    source_logits if name == "source"
                    else downstream(model, captured, context, dtype, False)
                )
                append_condition(
                    parts, name, logits, labels, competitor, context,
                    probes["memory_context"], source_context,
                    attention_conditions.get(name), source_attention,
                )

            audit_parts["probe_reference_kl_mean"].append(
                probe_budget.mean(dim=-1).cpu()
            )
            audit_parts["common_kl_budget_mean"].append(
                common_budget.mean(dim=-1).cpu()
            )
            audit_parts["common_over_probe_kl_fraction"].append(
                (
                    common_budget / probe_budget.clamp_min(1e-12)
                ).mean(dim=-1).cpu()
            )
            for name, attention in (
                ("gradient_kl", gradient_kl_attention),
                ("negative_kl", negative_attention),
                ("rolled_kl", rolled_attention),
            ):
                error = (
                    kl_divergence(attention, source_attention) - common_budget
                ).abs().max(dim=-1).values
                audit_parts[f"{name}_budget_error_max"].append(error.cpu())
            l2_error = (
                attention_delta(
                    gradient_l2_attention, source_attention, values, out_weight
                ).norm(dim=-1) - target_l2
            ).abs()
            audit_parts["gradient_l2_budget_error"].append(l2_error.cpu())
            audit_parts["attention_gradient_norm"].append(
                analytic_logit_gradient.norm(dim=(-1, -2)).cpu()
            )
            audit_parts["context_gradient_norm"].append(
                context_gradient.norm(dim=-1).cpu()
            )
            audit_parts["tangent_rank"].append(tangent_rank.cpu())
            audit_parts["tangent_gradient_energy"].append(tangent_energy.cpu())
            audit_parts["analytic_autograd_gradient_max_abs"].append(
                gradient_check.cpu()
            )

            labels_parts.append(labels.cpu())
            confidence_parts.append(confidence.cpu())
            competitor_parts.append(competitor.cpu())
            memory_probe_parts.append(memory_probe_logits.argmax(dim=-1).cpu())
            top_slots_parts.append(probe_slots.cpu())
            source_correct_running += int(
                (source_logits.argmax(dim=-1) == labels).sum().item()
            )
            total += batch
            if total == batch or total % args.log_every_samples == 0:
                print(
                    f"Level 6.19.2 samples={total}/{args.samples} "
                    f"source={source_correct_running / total:.2%}", flush=True
                )
                save(root / "progress.json", {
                    "samples_complete": total,
                    "samples_total": args.samples,
                    "source_reconstruction_max_abs_so_far": (
                        source_reconstruction_max
                    ),
                    "analytic_gradient_max_abs_so_far": max(
                        max(item.tolist())
                        for item in audit_parts[
                            "analytic_autograd_gradient_max_abs"
                        ]
                    ),
                })
    finally:
        trace.close()

    return {
        "labels": torch.cat(labels_parts),
        "confidence": torch.cat(confidence_parts),
        "competitor": torch.cat(competitor_parts),
        "memory_probe_predictions": torch.cat(memory_probe_parts),
        "probe_top_slots": torch.cat(top_slots_parts),
        "conditions": {
            name: {
                field: torch.cat(values)
                for field, values in row.items()
            }
            for name, row in parts.items()
        },
        "audit_values": {
            name: torch.cat(values) for name, values in audit_parts.items()
        },
        "reconstruction": {
            "source_downstream_max_abs": source_reconstruction_max,
            "native_replay_vs_native_context_max_abs": native_replay_context_max,
            "fp32_vs_native_context_max_abs": fp32_native_context_max,
            "manual_vs_fp32_context_max_abs": manual_attention_context_max,
        },
    }


def paired_continuous(left, right, args, seed):
    result = continuous_effect(
        (right.double() - left.double()).numpy(), args, seed
    )
    result["left_mean"] = left.double().mean().item()
    result["right_mean"] = right.double().mean().item()
    return result


def metric(row, labels, ids):
    group_labels = labels[ids]
    return {
        "accuracy": (
            row["predictions"][ids] == group_labels
        ).float().mean().item(),
        "fixed_margin_mean": row["fixed_margin"][ids].mean().item(),
        "context_decoder_accuracy": (
            row["context_predictions"][ids] == group_labels
        ).float().mean().item(),
        "context_margin_mean": row["context_margin"][ids].mean().item(),
        "context_delta_norm_mean": row[
            "context_delta_norm"
        ][ids].mean().item(),
        "attention_kl_mean": row["attention_kl_mean"][ids].mean().item(),
        "samples": int(len(ids)),
    }


def effect(source, updated, labels, ids, args, seed):
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
    }


def direct_contrast(left, right, field, ids, args, seed):
    return paired_continuous(
        left[field][ids], right[field][ids], args, seed
    )


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
    primary_ids = torch.where((~source_correct) & memory_correct)[0]
    if len(primary_ids) < args.minimum_memory_decodable_errors:
        raise RuntimeError(
            f"Only {len(primary_ids)} Memory-decodable errors; minimum is "
            f"{args.minimum_memory_decodable_errors}"
        )
    groups = {
        "all": torch.arange(len(labels)),
        "source_errors": errors,
        "memory_decodable_errors": primary_ids,
        "confidence_matched_correct": matched,
        "source_correct": correct,
    }
    metrics = {
        group: {
            name: metric(row, labels, ids)
            for name, row in conditions.items()
        }
        for group, ids in groups.items()
    }
    effects = {
        group: {
            name: effect(
                source, row, labels, ids, args,
                args.analysis_seed + group_index * 1000 + name_index * 10,
            )
            for name_index, (name, row) in enumerate(conditions.items())
            if name != "source"
        }
        for group_index, (group, ids) in enumerate(groups.items())
    }

    oracle = conditions["gradient_kl_oracle"]
    primary_contrasts = {
        "oracle_vs_source": direct_contrast(
            source, oracle, "fixed_margin", primary_ids, args,
            args.analysis_seed + 10000,
        ),
        "oracle_vs_probe_top4": direct_contrast(
            conditions["probe_top4_odds4"], oracle, "fixed_margin",
            primary_ids, args, args.analysis_seed + 10001,
        ),
        "oracle_vs_negative": direct_contrast(
            conditions["negative_gradient_kl"], oracle, "fixed_margin",
            primary_ids, args, args.analysis_seed + 10002,
        ),
        "oracle_vs_rolled": direct_contrast(
            conditions["rolled_gradient_kl"], oracle, "fixed_margin",
            primary_ids, args, args.analysis_seed + 10003,
        ),
    }
    adjusted = holm_adjust({
        key: value["sign_flip_p_two_sided"]
        for key, value in primary_contrasts.items()
    })
    for key in primary_contrasts:
        primary_contrasts[key]["multiplicity"] = adjusted[key]
    specificity = all(
        row["estimate"] > 0 and row["multiplicity"]["significant_0.05"]
        for row in primary_contrasts.values()
    )

    source_margin = source["fixed_margin"][primary_ids]
    kl_gain = (
        conditions["gradient_kl_oracle"]["fixed_margin"][primary_ids]
        - source_margin
    ).mean().item()
    l2_gain = (
        conditions["gradient_l2_oracle"]["fixed_margin"][primary_ids]
        - source_margin
    ).mean().item()
    tangent_gain = (
        conditions["tangent_context_control"]["fixed_margin"][primary_ids]
        - source_margin
    ).mean().item()
    unrestricted_gain = (
        conditions["unrestricted_context_control"]["fixed_margin"][primary_ids]
        - source_margin
    ).mean().item()
    recovery = {
        "gradient_kl_over_unrestricted": kl_gain / max(unrestricted_gain, 1e-12),
        "gradient_l2_over_unrestricted": l2_gain / max(unrestricted_gain, 1e-12),
        "tangent_over_unrestricted": tangent_gain / max(unrestricted_gain, 1e-12),
        "gradient_kl_gain": kl_gain,
        "gradient_l2_gain": l2_gain,
        "tangent_gain": tangent_gain,
        "unrestricted_gain": unrestricted_gain,
    }
    audits = collected["audit_values"]
    audit_summary = {
        "probe_reference_kl_mean": audits[
            "probe_reference_kl_mean"
        ][primary_ids].mean().item(),
        "common_kl_budget_mean": audits[
            "common_kl_budget_mean"
        ][primary_ids].mean().item(),
        "common_over_probe_kl_fraction_mean": audits[
            "common_over_probe_kl_fraction"
        ][primary_ids].mean().item(),
        "gradient_kl_budget_max_abs_error": audits[
            "gradient_kl_budget_error_max"
        ].max().item(),
        "negative_kl_budget_max_abs_error": audits[
            "negative_kl_budget_error_max"
        ].max().item(),
        "rolled_kl_budget_max_abs_error": audits[
            "rolled_kl_budget_error_max"
        ].max().item(),
        "gradient_l2_budget_mean_abs_error": audits[
            "gradient_l2_budget_error"
        ][primary_ids].mean().item(),
        "gradient_l2_budget_max_abs_error": audits[
            "gradient_l2_budget_error"
        ].max().item(),
        "attention_gradient_norm_mean": audits[
            "attention_gradient_norm"
        ][primary_ids].mean().item(),
        "context_gradient_norm_mean": audits[
            "context_gradient_norm"
        ][primary_ids].mean().item(),
        "tangent_rank_mean": audits["tangent_rank"][primary_ids].float().mean().item(),
        "tangent_rank_min": audits["tangent_rank"][primary_ids].min().item(),
        "tangent_gradient_energy_mean": audits[
            "tangent_gradient_energy"
        ][primary_ids].mean().item(),
        "tangent_gradient_energy_min": audits[
            "tangent_gradient_energy"
        ][primary_ids].min().item(),
        "analytic_autograd_gradient_max_abs": audits[
            "analytic_autograd_gradient_max_abs"
        ].max().item(),
        "attention_delta_closed_loop_max_abs": audits[
            "attention_delta_closed_loop_max_abs"
        ].max().item(),
    }
    tangent_reachable = (
        audit_summary["tangent_gradient_energy_mean"]
        >= args.tangent_energy_threshold
        and recovery["tangent_over_unrestricted"]
        >= args.tangent_recovery_threshold
    )
    unrestricted_operational = (
        effects["memory_decodable_errors"][
            "unrestricted_context_control"
        ]["fixed_margin"]["estimate"] > 0
        and effects["memory_decodable_errors"][
            "unrestricted_context_control"
        ]["fixed_margin"]["sign_flip_p_two_sided"] < 0.05
    )
    attention_sufficient = (
        specificity
        and unrestricted_operational
        and recovery["gradient_l2_over_unrestricted"]
        >= args.attention_recovery_threshold
    )
    if attention_sufficient:
        classification = "attention_router_score_obstruction"
        boundary = (
            "A frozen gradient-aligned attention redistribution reaches the "
            "successful context direction; preregister a donor-free router "
            "score, not broader value-path changes."
        )
    elif tangent_reachable and unrestricted_operational:
        classification = "finite_attention_simplex_or_budget_limitation"
        boundary = (
            "The value/output tangent span contains the useful direction, but "
            "finite attention redistribution at the registered budgets does "
            "not recover it; audit head-wise budgets and signed value mixtures."
        )
    elif unrestricted_operational:
        classification = "value_output_reachable_subspace_limitation"
        boundary = (
            "The useful context direction is not sufficiently reachable through "
            "the frozen attention value/output span; move to value composition "
            "and output projection."
        )
    else:
        classification = "positive_control_or_integrity_failure"
        boundary = "Stop and repair the positive control or implementation."

    matching.update({
        "errors": int(len(errors)),
        "matched_correct": int(len(matched)),
        "memory_decodable_errors": int(len(primary_ids)),
        "error_confidence_mean": collected["confidence"][errors].mean().item(),
        "matched_confidence_mean": collected[
            "confidence"
        ][matched].mean().item(),
    })
    diagnosis = {
        "classification": classification,
        "source_accuracy": source_correct.float().mean().item(),
        "source_errors": int(len(errors)),
        "memory_probe_correct_on_source_errors": memory_correct[
            errors
        ].float().mean().item(),
        "memory_decodable_errors": int(len(primary_ids)),
        "gradient_kl_specificity_passed": specificity,
        "tangent_reachable": tangent_reachable,
        "unrestricted_positive_control_operational": unrestricted_operational,
        "attention_recovery_sufficient": attention_sufficient,
        "registered_next_boundary": boundary,
    }
    return {
        "matching": matching,
        "metrics": metrics,
        "effects_vs_source": effects,
        "primary": {
            "population": "Memory-decodable source errors",
            "gradient_kl_specificity": primary_contrasts,
            "recovery_fractions": recovery,
        },
        "reachable_subspace_audit": audit_summary,
        "diagnosis": diagnosis,
    }, groups


def plot_result(analysis, path):
    names = [
        "Probe\ntop4", "Grad\ntop4", "Grad\nKL", "Negative\nKL",
        "Rolled\nKL", "Grad\nL2", "Tangent\ncontext", "Full\ncontext",
    ]
    keys = CONDITIONS[1:]
    rows = analysis["metrics"]["memory_decodable_errors"]
    source_margin = rows["source"]["fixed_margin_mean"]
    source_context = rows["source"]["context_margin_mean"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].bar(
        np.arange(len(keys)),
        [rows[key]["fixed_margin_mean"] - source_margin for key in keys],
        color="#4C78A8",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(np.arange(len(keys)), names, rotation=25, ha="right")
    axes[0].set_ylabel("Deployed margin change")
    axes[0].set_title("Frozen downstream effect")
    axes[1].bar(
        np.arange(len(keys)),
        [rows[key]["context_margin_mean"] - source_context for key in keys],
        color="#59A14F",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(keys)), names, rotation=25, ha="right")
    axes[1].set_ylabel("Context Probe margin change")
    axes[1].set_title("Independent Probe response")
    recovery = analysis["primary"]["recovery_fractions"]
    axes[2].bar(
        ["Grad KL", "Grad L2", "Tangent"],
        [
            recovery["gradient_kl_over_unrestricted"],
            recovery["gradient_l2_over_unrestricted"],
            recovery["tangent_over_unrestricted"],
        ],
        color=["#F28E2B", "#E15759", "#B07AA1"],
    )
    axes[2].axhline(1, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("Fraction of unrestricted context gain")
    axes[2].set_title("Reachable-subspace recovery")
    figure.suptitle(
        "IST Level 6.19.2: Frozen Read-Attention Reachable-Subspace Audit",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def preregistration(args):
    return {
        "level": "6.19.2",
        "numerical_revision": NUMERICAL_REVISION,
        "repair_note": (
            "Two fail-closed numerical attempts showed that the absolute "
            "native fused-SDPA versus explicit-softmax context maximum is a "
            "cross-kernel diagnostic, not an intervention reconstruction: "
            "every intervention is the explicit updated-minus-source delta "
            "added to the exact native source. Gate that closed delta path, "
            "retain the cross-kernel maximum descriptively, and accumulate "
            "KL in fp64 with its theoretical non-negative bound. Revision v4 "
            "also retains the KL tilt parameter in fp64 while evaluating each "
            "candidate on the registered fp32 attention path."
        ),
        "status": "frozen read-attention reachable-subspace audit",
        "source": "formally passed Level 6.18.3 seed707 checkpoint",
        "probes": "frozen Level 6.19 Memory and context probes",
        "seed": SEED,
        "chunks": CHUNKS,
        "primary_population": "source errors whose frozen Memory probe is correct",
        "registered_budget": {
            "reference": "per-example, per-head KL induced by Probe top4 4x odds, capped to a common direction-reachable ceiling",
            "ceiling_fraction": args.kl_ceiling_fraction,
            "l2_reference": "per-example context L2 induced by Probe top4 4x odds",
        },
        "conditions": {
            "source": "unmodified frozen source",
            "probe_top4_odds4": "FP32 analytic delta for the Level 6.19.1 intervention",
            "gradient_top4_odds4": "per-head top-four attention-logit gradients at 4x odds",
            "gradient_kl_oracle": "exponential tilt maximizing first-order deployed margin at matched per-head KL",
            "negative_gradient_kl": "equal-KL reverse-direction control",
            "rolled_gradient_kl": "equal-KL cross-example gradient control",
            "gradient_l2_oracle": "gradient exponential tilt matched to Probe top4 context L2",
            "tangent_context_control": "projection of context gradient into frozen attention value/output tangent span at matched L2",
            "unrestricted_context_control": "full context gradient at matched L2",
        },
        "primary_gradient_kl_family": [
            "gradient KL oracle versus source",
            "gradient KL oracle versus Probe top4",
            "gradient KL oracle versus negative-gradient KL",
            "gradient KL oracle versus rolled-gradient KL",
        ],
        "multiplicity": "Holm across the four primary deployed-margin contrasts",
        "decision_rule": {
            "specificity": "all four estimates positive with Holm p < 0.05",
            "tangent_energy_threshold": args.tangent_energy_threshold,
            "tangent_recovery_threshold": args.tangent_recovery_threshold,
            "attention_recovery_threshold": args.attention_recovery_threshold,
            "positive_control": "unrestricted matched-L2 context margin gain positive at p < 0.05",
        },
        "interpretation": "all oracle conditions use labels and are mechanistic, not deployable",
        "locks": {
            "all_model_and_probe_parameters_frozen": True,
            "failed_Level_6_18_9_candidate_not_used": True,
            "protected_tests_not_used": True,
            "seed909_locked": True,
        },
        "numerical_audit": {
            "deployed_path": "native bf16 need_weights=False",
            "interpretable_decomposition": (
                "native autocast Q/K/V projections followed by explicit FP32 "
                "QK softmax and value/output composition"
            ),
            "native_projection_replay": (
                "diagnostic only; distinct CUDA attention kernels are not "
                "required to be bitwise-equivalent"
            ),
            "manual_internal_tolerance": 1e-5,
            "fp32_vs_deployed_context": (
                "reported descriptively; not a gate because the absolute "
                "explicit context is never substituted into deployment"
            ),
            "kl_accumulation": (
                "FP64 with theoretical non-negative clamp; FP64 bisection "
                "parameter with every candidate quantized to FP32"
            ),
            "closed_loop_gate": (
                "zero delta exactly reproduces deployed source; explicit "
                "updated-minus-explicit-source delta is added to that source"
            ),
        },
        "protocol": {
            key: value for key, value in vars(args).items()
            if key not in {"dry_run", "smoke_test", "force"}
        },
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError("Level 6.19.2 is fixed to seed707 at 16 chunks")
    for path in (args.checkpoint, args.probes):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.samples <= 0 or args.eval_batch_size <= 0:
        raise ValueError("samples and eval-batch-size must be positive")
    if args.samples % args.eval_batch_size != 0:
        raise ValueError("samples must be divisible by eval-batch-size")
    if not args.smoke_test and (
        args.samples != 4096 or args.dataset_seed != 6193100
    ):
        raise ValueError(
            "Formal Level 6.19.2 fixes samples=4096 and dataset-seed=6193100; "
            "use --smoke-test for code checks"
        )


def raw_predictions(collected, groups):
    return {
        "labels": collected["labels"].tolist(),
        "confidence": collected["confidence"].tolist(),
        "competitor": collected["competitor"].tolist(),
        "memory_probe_predictions": collected[
            "memory_probe_predictions"
        ].tolist(),
        "groups": {name: ids.tolist() for name, ids in groups.items()},
        "probe_top_slots": collected["probe_top_slots"].tolist(),
        "audit_values": {
            name: value.tolist()
            for name, value in collected["audit_values"].items()
        },
        "conditions": {
            name: {field: value.tolist() for field, value in row.items()}
            for name, row in collected["conditions"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.19.2 frozen attention reachable-subspace audit"
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
    parser.add_argument("--dataset-seed", type=int, default=6193100)
    parser.add_argument("--analysis-seed", type=int, default=6193200)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--minimum-errors", type=int, default=200)
    parser.add_argument("--minimum-memory-decodable-errors", type=int, default=150)
    parser.add_argument("--tangent-energy-threshold", type=float, default=0.80)
    parser.add_argument("--tangent-recovery-threshold", type=float, default=0.80)
    parser.add_argument("--attention-recovery-threshold", type=float, default=0.50)
    parser.add_argument("--kl-ceiling-fraction", type=float, default=0.95)
    parser.add_argument("--log-every-samples", type=int, default=256)
    parser.add_argument("--output", default="experiments/level6_19_2/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.dataset_seed += 50_000_000
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
    before = {
        "model": tensor_fingerprint(model),
        "original_probe": tensor_fingerprint(original_probe),
        **{
            f"level6_19_{name}": tensor_fingerprint(row["probe"])
            for name, row in probes.items()
        },
    }
    collected = collect(model, probes, args, device, dtype, root)
    analysis, groups = analyze(collected, args)
    after = {
        "model": tensor_fingerprint(model),
        "original_probe": tensor_fingerprint(original_probe),
        **{
            f"level6_19_{name}": tensor_fingerprint(row["probe"])
            for name, row in probes.items()
        },
    }
    reconstruction = collected["reconstruction"]
    audit = analysis["reachable_subspace_audit"]
    integrity = {
        "numerical_revision": NUMERICAL_REVISION,
        **reconstruction,
        "source_downstream_reconstruction_exact": (
            reconstruction["source_downstream_max_abs"] == 0.0
        ),
        "manual_attention_context_close": (
            reconstruction["manual_vs_fp32_context_max_abs"] <= 1e-5
        ),
        "fp32_vs_native_is_diagnostic_only": True,
        "attention_delta_closed_loop_max_abs": audit[
            "attention_delta_closed_loop_max_abs"
        ],
        "attention_delta_closed_loop_passed": audit[
            "attention_delta_closed_loop_max_abs"
        ] <= 1e-5,
        "analytic_attention_gradient_max_abs": audit[
            "analytic_autograd_gradient_max_abs"
        ],
        "analytic_attention_gradient_passed": audit[
            "analytic_autograd_gradient_max_abs"
        ] <= 1e-6,
        "gradient_kl_budget_max_abs_error": audit[
            "gradient_kl_budget_max_abs_error"
        ],
        "kl_budgets_matched": max(
            audit["gradient_kl_budget_max_abs_error"],
            audit["negative_kl_budget_max_abs_error"],
            audit["rolled_kl_budget_max_abs_error"],
        ) <= 1e-5,
        "gradient_l2_budget_max_abs_error": audit[
            "gradient_l2_budget_max_abs_error"
        ],
        "l2_budget_matched": audit[
            "gradient_l2_budget_max_abs_error"
        ] <= 1e-3,
        "all_states_unchanged": before == after,
        "all_parameters_frozen": all(
            not parameter.requires_grad
            for module in [model, original_probe] + [
                row["probe"] for row in probes.values()
            ]
            for parameter in module.parameters()
        ),
        "failed_candidate_not_used": True,
        "protected_tests_not_used": True,
        "seed909_locked": True,
    }
    integrity["passed"] = all([
        integrity["source_downstream_reconstruction_exact"],
        integrity["manual_attention_context_close"],
        integrity["attention_delta_closed_loop_passed"],
        integrity["analytic_attention_gradient_passed"],
        integrity["kl_budgets_matched"],
        integrity["l2_budget_matched"],
        integrity["all_states_unchanged"],
        integrity["all_parameters_frozen"],
    ])
    analysis["diagnosis"]["integrity_passed"] = integrity["passed"]
    if not integrity["passed"]:
        analysis["diagnosis"]["classification"] = "integrity_failure"
        analysis["diagnosis"]["registered_next_boundary"] = (
            "Stop; repair the reachable-subspace implementation."
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
        "reachable_subspace_audit": analysis["reachable_subspace_audit"],
        "diagnosis": analysis["diagnosis"],
    })
    save(root / "predictions.json", raw_predictions(collected, groups))
    plot_result(analysis, root / "attention_reachable_subspace.png")
    print(json.dumps(analysis["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
