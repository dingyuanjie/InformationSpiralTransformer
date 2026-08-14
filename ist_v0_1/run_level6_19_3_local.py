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
from run_level6_19_2_local import (
    HEADS,
    HEAD_WIDTH,
    SLOTS,
    WIDTH,
    attention_decomposition,
    attention_delta,
    delta_closed_loop,
    downstream,
    kl_divergence,
    patch_context,
    standardized_score,
    task_competitor,
    tilt_to_l2,
)


SEED = 707
CHUNKS = 16
NUMERICAL_REVISION = "delta_closed_loop_fp64_solver_v4"
CORE_CONDITIONS = [
    "source",
    "finite_shared_l2",
    "finite_head_budget_l2",
    "signed_affine_l2",
    "negative_signed_l2",
    "rolled_signed_l2",
    "head_permuted_signed_l2",
    "unrestricted_context_l2",
]
HEAD_CONDITIONS = [
    *[f"signed_head_{head}_only_l2" for head in range(HEADS)],
    *[f"signed_without_head_{head}_l2" for head in range(HEADS)],
]
CONDITIONS = CORE_CONDITIONS + HEAD_CONDITIONS


def tangent_basis(values, out_weight):
    differences = values[:, :, :-1] - values[:, :, -1:]
    blocks = []
    for head in range(HEADS):
        weight = out_weight[
            :, head * HEAD_WIDTH:(head + 1) * HEAD_WIDTH
        ]
        blocks.append(torch.matmul(differences[:, head], weight.T))
    return torch.stack(blocks, dim=1).transpose(-2, -1).contiguous()


def subspace_projection(gradient, basis):
    """Project [B,64] onto columns of [B,64,K] with a relative rank gate."""
    gram = torch.matmul(basis, basis.transpose(-2, -1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues[:, -1:].clamp_min(1e-12)
    keep = eigenvalues > maximum * 1e-6
    coefficients = torch.matmul(
        eigenvectors.transpose(-2, -1), gradient[..., None]
    ).squeeze(-1)
    projected = torch.matmul(
        eigenvectors, (coefficients * keep)[..., None]
    ).squeeze(-1)
    energy = (
        projected.norm(dim=-1).square()
        / gradient.norm(dim=-1).clamp_min(1e-8).square()
    )
    return projected, keep.sum(dim=-1), energy


def signed_affine_solution(gradient, basis, target_l2):
    """Minimum-norm zero-sum per-head coefficients for the tangent projection."""
    flat = basis.permute(0, 2, 1, 3).reshape(
        len(basis), WIDTH, HEADS * (SLOTS - 1)
    )
    gram = torch.matmul(flat, flat.transpose(-2, -1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues[:, -1:].clamp_min(1e-12)
    keep = eigenvalues > maximum * 1e-6
    rotated = torch.matmul(
        eigenvectors.transpose(-2, -1), gradient[..., None]
    ).squeeze(-1)
    inverse = torch.where(
        keep, eigenvalues.clamp_min(1e-12).reciprocal(),
        torch.zeros_like(eigenvalues),
    )
    dual = torch.matmul(
        eigenvectors, (rotated * inverse)[..., None]
    ).squeeze(-1)
    coefficients = torch.matmul(
        flat.transpose(-2, -1), dual[..., None]
    ).squeeze(-1).reshape(len(basis), HEADS, SLOTS - 1)
    projected = torch.matmul(
        flat, coefficients.reshape(len(basis), -1)[..., None]
    ).squeeze(-1)
    scale = (
        target_l2 / projected.norm(dim=-1).clamp_min(1e-8)
    )
    coefficients = coefficients * scale[:, None, None]
    head_delta = torch.matmul(
        basis, coefficients[..., None]
    ).squeeze(-1)
    delta = head_delta.sum(dim=1)

    full_coefficients = torch.cat([
        coefficients, -coefficients.sum(dim=-1, keepdim=True)
    ], dim=-1)
    return {
        "delta": delta,
        "head_delta": head_delta,
        "coefficients": full_coefficients,
        "rank": keep.sum(dim=-1),
        "energy": (
            projected.norm(dim=-1).square()
            / gradient.norm(dim=-1).clamp_min(1e-8).square()
        ),
    }


def coefficient_feasibility(source, coefficients):
    positive_mask = coefficients < 0
    negative_mask = coefficients > 0
    positive_scale = torch.where(
        positive_mask,
        source / (-coefficients).clamp_min(1e-30),
        torch.full_like(source, torch.inf),
    ).min(dim=-1).values
    negative_scale = torch.where(
        negative_mask,
        source / coefficients.clamp_min(1e-30),
        torch.full_like(source, torch.inf),
    ).min(dim=-1).values
    proposed = source + coefficients
    return {
        "positive_feasible_scale": positive_scale,
        "negative_feasible_scale": negative_scale,
        "negative_mass": (-proposed.clamp_max(0)).sum(dim=-1),
        "negative_fraction": (proposed < 0).float().mean(dim=-1),
        "maximum_weight": proposed.max(dim=-1).values,
    }


def infinitesimal_head_directions(source, utility, values, out_weight):
    # Use the same globally standardized score as Level 6.19.2. With an
    # all-ones head allocation this is exactly the registered shared-dose
    # softmax family; only the eight non-negative head multipliers are new.
    score = standardized_score(utility, per_head=False)
    expected = (source * score).sum(dim=-1, keepdim=True)
    derivative = source * (score - expected)
    latent = torch.einsum("bhs,bhsd->bhd", derivative, values)
    directions = []
    for head in range(HEADS):
        weight = out_weight[
            :, head * HEAD_WIDTH:(head + 1) * HEAD_WIDTH
        ]
        directions.append(F.linear(latent[:, head], weight, None))
    return score, torch.stack(directions, dim=1)


def nonnegative_head_allocation(directions, gradient, iterations=96):
    norm = directions.norm(dim=-1).clamp_min(1e-8)
    normalized = directions / norm[..., None]
    matrix = normalized.transpose(1, 2).contiguous()
    gram = torch.matmul(matrix.transpose(1, 2), matrix)
    rhs = torch.matmul(
        matrix.transpose(1, 2), gradient[..., None]
    ).squeeze(-1)
    largest = torch.linalg.eigvalsh(gram)[:, -1].clamp_min(1e-6)
    beta = torch.zeros_like(rhs)
    for _ in range(iterations):
        residual = torch.matmul(gram, beta[..., None]).squeeze(-1) - rhs
        beta = (beta - residual / largest[:, None]).clamp_min(0)
    eta_weight = beta / norm
    eta_weight = eta_weight / eta_weight.mean(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    projected = torch.matmul(matrix, beta[..., None]).squeeze(-1)
    energy = (
        projected.norm(dim=-1).square()
        / gradient.norm(dim=-1).clamp_min(1e-8).square()
    )
    return eta_weight, energy


def head_allocated_tilt_to_l2(source, score, allocation, values, out_weight,
                              target_l2, iterations=56):
    log_source = source.double().clamp_min(1e-30).log()
    direction = allocation.double()[..., None] * score.double()
    low = torch.zeros(len(source), device=source.device, dtype=torch.float64)
    high = torch.ones_like(low)
    for _ in range(28):
        candidate = torch.softmax(
            log_source + high[:, None, None] * direction, dim=-1
        ).float()
        norm = attention_delta(
            candidate, source, values, out_weight
        ).norm(dim=-1)
        high = torch.where(norm < target_l2, high * 2.0, high)
    for _ in range(iterations):
        middle = (low + high) / 2.0
        candidate = torch.softmax(
            log_source + middle[:, None, None] * direction, dim=-1
        ).float()
        norm = attention_delta(
            candidate, source, values, out_weight
        ).norm(dim=-1)
        below = norm < target_l2
        low = torch.where(below, middle, low)
        high = torch.where(below, high, middle)
    low_attention = torch.softmax(
        log_source + low[:, None, None] * direction, dim=-1
    ).float()
    high_attention = torch.softmax(
        log_source + high[:, None, None] * direction, dim=-1
    ).float()
    low_error = (
        attention_delta(low_attention, source, values, out_weight)
        .norm(dim=-1) - target_l2
    ).abs()
    high_error = (
        attention_delta(high_attention, source, values, out_weight)
        .norm(dim=-1) - target_l2
    ).abs()
    choose_high = high_error < low_error
    attention = torch.where(
        choose_high[:, None, None], high_attention, low_attention
    )
    eta = torch.where(choose_high, high, low)
    return attention, eta


def inverse_softplus(value):
    value = value.clamp_min(1e-6)
    return value + torch.log(-torch.expm1(-value))


def optimize_finite_head_budget(source, score, initial_allocation, values,
                                out_weight, target_l2, gradient,
                                iterations=80):
    """Optimize all eight non-negative finite-softmax head doses at equal L2."""
    square_root = initial_allocation.clamp_min(1e-4).sqrt()
    square_root = square_root / square_root.mean(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    peaked = torch.full_like(initial_allocation, 0.1)
    peaked.scatter_(
        -1, initial_allocation.argmax(dim=-1, keepdim=True),
        7.3,
    )
    starts = [
        torch.ones_like(initial_allocation),
        initial_allocation.clamp_min(1e-4),
        square_root,
        peaked,
    ]
    candidates = []
    log_source = source.detach().double().clamp_min(1e-30).log()
    score64 = score.detach().double()
    values_constant = values.detach()
    out_constant = out_weight.detach()
    target = target_l2.detach()
    gradient_constant = gradient.detach()
    for allocation in starts:
        _, scalar = head_allocated_tilt_to_l2(
            source, score, allocation, values, out_weight, target_l2
        )
        eta_start = (
            scalar.float()[:, None] * allocation
        ).clamp_min(1e-5)
        raw = inverse_softplus(eta_start).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=0.08)
        with torch.enable_grad():
            for _ in range(iterations):
                eta = F.softplus(raw)
                attention = torch.softmax(
                    log_source
                    + eta.double()[..., None] * score64,
                    dim=-1,
                ).float()
                delta = attention_delta(
                    attention, source, values_constant, out_constant
                )
                norm = delta.norm(dim=-1).clamp_min(1e-8)
                normalized_gain = (
                    (delta * gradient_constant).sum(dim=-1)
                    / (
                        target.clamp_min(1e-8)
                        * gradient_constant.norm(dim=-1).clamp_min(1e-8)
                    )
                )
                dose_error = (
                    (norm - target) / target.clamp_min(1e-8)
                )
                loss = (-normalized_gain + 12.0 * dose_error.square()).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        eta = F.softplus(raw.detach())
        allocation = eta / eta.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        attention, scalar = head_allocated_tilt_to_l2(
            source, score, allocation, values, out_weight, target_l2
        )
        delta = attention_delta(attention, source, values, out_weight)
        gain = (delta * gradient).sum(dim=-1)
        candidates.append((attention, allocation, scalar, gain))

    gains = torch.stack([candidate[3] for candidate in candidates], dim=-1)
    selected = gains.argmax(dim=-1)
    attention_stack = torch.stack(
        [candidate[0] for candidate in candidates], dim=1
    )
    allocation_stack = torch.stack(
        [candidate[1] for candidate in candidates], dim=1
    )
    scalar_stack = torch.stack(
        [candidate[2] for candidate in candidates], dim=1
    )
    rows = torch.arange(len(source), device=source.device)
    attention = attention_stack[rows, selected]
    allocation = allocation_stack[rows, selected]
    scalar = scalar_stack[rows, selected]
    diagnostics = {
        "selected_start": selected,
        "candidate_gain": gains,
        "best_over_uniform_gain": gains.max(dim=-1).values - gains[:, 0],
        "top_two_gain_gap": (
            gains.topk(2, dim=-1).values[:, 0]
            - gains.topk(2, dim=-1).values[:, 1]
        ),
    }
    return attention, allocation, scalar, diagnostics


def empty_parts():
    fields = [
        "predictions", "fixed_margin", "decision_margin", "cross_entropy",
        "context_predictions", "context_margin", "context_delta_norm",
        "attention_kl_mean",
    ]
    return {
        condition: {field: [] for field in fields}
        for condition in CONDITIONS
    }


def append_condition(parts, condition, logits, labels, competitor, context,
                     context_probe, source_context, attention=None,
                     source_attention=None):
    rows = torch.arange(len(labels), device=labels.device)
    correct = logits[rows, labels]
    fixed_margin = correct - logits[rows, competitor]
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    query_context = context[:, -1].float()
    context_logits = probe_logits(context_probe, query_context)
    if attention is None:
        kl = torch.zeros(len(labels), device=labels.device)
    else:
        kl = kl_divergence(attention, source_attention).mean(dim=-1)
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
    row["attention_kl_mean"].append(kl.detach().cpu())


def collect(model, probes, args, device, dtype, root):
    parts = empty_parts()
    labels_parts = []
    confidence_parts = []
    competitor_parts = []
    memory_probe_parts = []
    audit_parts = {
        "target_l2": [],
        "finite_shared_l2_error": [],
        "finite_head_budget_l2_error": [],
        "signed_l2_error": [],
        "delta_closed_loop_error": [],
        "manual_fp32_error": [],
        "analytic_gradient_error": [],
        "full_tangent_rank": [],
        "full_tangent_energy": [],
        "head_allocation_energy": [],
        "source_entropy": [],
        "head_rank": [],
        "head_gradient_energy": [],
        "head_allocation": [],
        "head_budget_selected_start": [],
        "head_budget_candidate_gain": [],
        "head_budget_best_over_uniform_gain": [],
        "head_budget_top_two_gain_gap": [],
        "uniform_family_delta_error": [],
        "finite_shared_head_norm": [],
        "finite_shared_head_gain": [],
        "finite_head_budget_head_norm": [],
        "finite_head_budget_head_gain": [],
        "signed_head_norm": [],
        "signed_head_gain": [],
        "signed_positive_feasible_scale": [],
        "signed_negative_feasible_scale": [],
        "signed_negative_mass": [],
        "signed_negative_fraction": [],
        "signed_maximum_weight": [],
    }
    trace = FinalTrace(model)
    total = 0
    source_reconstruction_max = 0.0
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
                native_logits, _ = model(
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
            manual_error = (
                decomposition["manual_query"] - decomposition["fp32_query"]
            ).abs().reshape(batch, -1).max(dim=-1).values

            reconstructed = downstream(
                model, captured, source_context, dtype, False
            )
            source_reconstruction_max = max(
                source_reconstruction_max,
                (reconstructed - source_logits).abs().max().item(),
            )

            _, probe_slots, _, memory_probe_logits = select_slots(
                captured["read_memory"], labels, competitor,
                probes["memory_l3_concat"],
            )
            probe_attention = source_attention.clone()
            logits = source_attention.clamp_min(1e-12).log()
            bias = torch.zeros_like(logits)
            expanded_slots = probe_slots[:, None].expand(-1, HEADS, -1)
            bias.scatter_add_(
                -1, expanded_slots,
                torch.full_like(
                    expanded_slots, math.log(4.0), dtype=logits.dtype
                ),
            )
            probe_attention = torch.softmax(logits + bias, dim=-1)
            probe_delta = attention_delta(
                probe_attention, source_attention, values, out_weight
            )
            target_l2 = probe_delta.norm(dim=-1)

            context_leaf = source_context.detach().clone().requires_grad_(True)
            gradient_logits = downstream(
                model, captured, context_leaf, dtype, True
            )
            rows = torch.arange(batch, device=device)
            margin = (
                gradient_logits[rows, labels]
                - gradient_logits[rows, competitor]
            )
            context_gradient = torch.autograd.grad(
                margin.sum(), context_leaf, only_inputs=True
            )[0][:, -1].detach().float()
            source_reconstruction_max = max(
                source_reconstruction_max,
                (gradient_logits.detach() - source_logits).abs().max().item(),
            )

            concat_gradient = torch.matmul(context_gradient, out_weight)
            head_gradient = concat_gradient.reshape(batch, HEADS, HEAD_WIDTH)
            utility = (values * head_gradient[:, :, None]).sum(dim=-1)
            expected = (
                source_attention * utility
            ).sum(dim=-1, keepdim=True)
            analytic_gradient = source_attention * (utility - expected)
            bias_leaf = torch.zeros_like(source_attention, requires_grad=True)
            differentiable_attention = torch.softmax(
                source_attention.clamp_min(1e-12).log() + bias_leaf, dim=-1
            )
            differentiable_delta = attention_delta(
                differentiable_attention, source_attention, values, out_weight
            )
            automatic_gradient = torch.autograd.grad(
                (differentiable_delta * context_gradient).sum(), bias_leaf,
                only_inputs=True,
            )[0]
            gradient_error = (
                analytic_gradient - automatic_gradient
            ).abs().reshape(batch, -1).max(dim=-1).values

            shared_attention, _ = tilt_to_l2(
                source_attention, utility, values, out_weight, target_l2
            )
            shared_delta, shared_closed = delta_closed_loop(
                shared_attention, source_attention, values, out_weight
            )
            score, infinitesimal = infinitesimal_head_directions(
                source_attention, utility, values, out_weight
            )
            linearized_allocation, allocation_energy = nonnegative_head_allocation(
                infinitesimal, context_gradient
            )
            uniform_attention, _ = head_allocated_tilt_to_l2(
                source_attention, score, torch.ones_like(linearized_allocation),
                values, out_weight, target_l2,
            )
            uniform_delta = attention_delta(
                uniform_attention, source_attention, values, out_weight
            )
            head_budget_attention, allocation, _, allocation_diagnostics = (
                optimize_finite_head_budget(
                source_attention, score, linearized_allocation, values,
                out_weight, target_l2, context_gradient,
                )
            )
            head_budget_delta, head_budget_closed = delta_closed_loop(
                head_budget_attention, source_attention, values, out_weight
            )

            basis = tangent_basis(values, out_weight)
            signed = signed_affine_solution(
                context_gradient, basis, target_l2
            )
            signed_delta = signed["delta"]
            feasibility = coefficient_feasibility(
                source_attention, signed["coefficients"]
            )
            negative_signed = -signed_delta
            rolled_signed = signed_affine_solution(
                torch.roll(context_gradient, 1, dims=0), basis, target_l2
            )["delta"]
            permuted_coefficients = torch.roll(
                signed["coefficients"], 1, dims=1
            )[..., :-1]
            permuted_delta = torch.matmul(
                basis, permuted_coefficients[..., None]
            ).squeeze(-1).sum(dim=1)
            permuted_delta = norm_match(permuted_delta, target_l2)
            unrestricted_delta = norm_match(context_gradient, target_l2)

            head_only = []
            without_head = []
            head_rank = []
            head_energy = []
            for head in range(HEADS):
                projected, rank, energy = subspace_projection(
                    context_gradient, basis[:, head]
                )
                head_only.append(norm_match(projected, target_l2))
                head_rank.append(rank)
                head_energy.append(energy)
                remaining = torch.cat([
                    basis[:, :head], basis[:, head + 1:]
                ], dim=1).permute(0, 2, 1, 3).reshape(batch, WIDTH, -1)
                projected_without, _, _ = subspace_projection(
                    context_gradient, remaining
                )
                without_head.append(
                    norm_match(projected_without, target_l2)
                )

            attention_conditions = {
                "finite_shared_l2": shared_attention,
                "finite_head_budget_l2": head_budget_attention,
            }
            delta_conditions = {
                "signed_affine_l2": signed_delta,
                "negative_signed_l2": negative_signed,
                "rolled_signed_l2": rolled_signed,
                "head_permuted_signed_l2": permuted_delta,
                "unrestricted_context_l2": unrestricted_delta,
                **{
                    f"signed_head_{head}_only_l2": head_only[head]
                    for head in range(HEADS)
                },
                **{
                    f"signed_without_head_{head}_l2": without_head[head]
                    for head in range(HEADS)
                },
            }
            contexts = {"source": source_context}
            for name, attention in attention_conditions.items():
                delta = (
                    shared_delta if name == "finite_shared_l2"
                    else head_budget_delta
                )
                contexts[name] = patch_context(source_context, delta)
            for name, delta in delta_conditions.items():
                contexts[name] = patch_context(source_context, delta)

            for name in CONDITIONS:
                context = contexts[name]
                updated_logits = (
                    source_logits if name == "source"
                    else downstream(model, captured, context, dtype, False)
                )
                append_condition(
                    parts, name, updated_logits, labels, competitor, context,
                    probes["memory_context"], source_context,
                    attention_conditions.get(name), source_attention,
                )

            def head_context_delta(attention):
                latent = torch.einsum(
                    "bhs,bhsd->bhd", attention - source_attention, values
                )
                output = []
                for head in range(HEADS):
                    weight = out_weight[
                        :, head * HEAD_WIDTH:(head + 1) * HEAD_WIDTH
                    ]
                    output.append(F.linear(latent[:, head], weight, None))
                return torch.stack(output, dim=1)

            shared_head_delta = head_context_delta(shared_attention)
            budget_head_delta = head_context_delta(head_budget_attention)
            entropy = -(
                source_attention
                * source_attention.clamp_min(1e-30).log()
            ).sum(dim=-1)
            audit_parts["target_l2"].append(target_l2.cpu())
            audit_parts["finite_shared_l2_error"].append(
                (shared_delta.norm(dim=-1) - target_l2).abs().cpu()
            )
            audit_parts["finite_head_budget_l2_error"].append(
                (head_budget_delta.norm(dim=-1) - target_l2).abs().cpu()
            )
            audit_parts["signed_l2_error"].append(
                (signed_delta.norm(dim=-1) - target_l2).abs().cpu()
            )
            audit_parts["delta_closed_loop_error"].append(
                torch.maximum(
                    shared_closed.reshape(batch, -1).max(dim=-1).values,
                    head_budget_closed.reshape(batch, -1).max(dim=-1).values,
                ).cpu()
            )
            audit_parts["manual_fp32_error"].append(manual_error.cpu())
            audit_parts["analytic_gradient_error"].append(
                gradient_error.cpu()
            )
            audit_parts["full_tangent_rank"].append(signed["rank"].cpu())
            audit_parts["full_tangent_energy"].append(signed["energy"].cpu())
            audit_parts["head_allocation_energy"].append(
                allocation_energy.cpu()
            )
            audit_parts["source_entropy"].append(entropy.cpu())
            audit_parts["head_rank"].append(
                torch.stack(head_rank, dim=-1).cpu()
            )
            audit_parts["head_gradient_energy"].append(
                torch.stack(head_energy, dim=-1).cpu()
            )
            audit_parts["head_allocation"].append(allocation.cpu())
            audit_parts["head_budget_selected_start"].append(
                allocation_diagnostics["selected_start"].cpu()
            )
            audit_parts["head_budget_candidate_gain"].append(
                allocation_diagnostics["candidate_gain"].cpu()
            )
            audit_parts["head_budget_best_over_uniform_gain"].append(
                allocation_diagnostics["best_over_uniform_gain"].cpu()
            )
            audit_parts["head_budget_top_two_gain_gap"].append(
                allocation_diagnostics["top_two_gain_gap"].cpu()
            )
            audit_parts["uniform_family_delta_error"].append(
                (uniform_delta - shared_delta).abs().reshape(
                    batch, -1
                ).max(dim=-1).values.cpu()
            )
            for prefix, head_delta in (
                ("finite_shared", shared_head_delta),
                ("finite_head_budget", budget_head_delta),
                ("signed", signed["head_delta"]),
            ):
                audit_parts[f"{prefix}_head_norm"].append(
                    head_delta.norm(dim=-1).cpu()
                )
                audit_parts[f"{prefix}_head_gain"].append(
                    (head_delta * context_gradient[:, None]).sum(dim=-1).cpu()
                )
            for name, value in feasibility.items():
                audit_parts[f"signed_{name}"].append(value.cpu())

            labels_parts.append(labels.cpu())
            confidence_parts.append(confidence.cpu())
            competitor_parts.append(competitor.cpu())
            memory_probe_parts.append(memory_probe_logits.argmax(dim=-1).cpu())
            source_correct_running += int(
                (source_logits.argmax(dim=-1) == labels).sum().item()
            )
            total += batch
            if total == batch or total % args.log_every_samples == 0:
                print(
                    f"Level 6.19.3 samples={total}/{args.samples} "
                    f"source={source_correct_running / total:.2%}",
                    flush=True,
                )
                save(root / "progress.json", {
                    "samples_complete": total,
                    "samples_total": args.samples,
                    "source_reconstruction_max_abs_so_far": (
                        source_reconstruction_max
                    ),
                    "closed_loop_max_abs_so_far": max(
                        value.max().item()
                        for value in audit_parts["delta_closed_loop_error"]
                    ),
                })
    finally:
        trace.close()

    return {
        "labels": torch.cat(labels_parts),
        "confidence": torch.cat(confidence_parts),
        "competitor": torch.cat(competitor_parts),
        "memory_probe_predictions": torch.cat(memory_probe_parts),
        "conditions": {
            name: {
                field: torch.cat(values)
                for field, values in row.items()
            }
            for name, row in parts.items()
        },
        "audit_values": {
            name: torch.cat(values)
            for name, values in audit_parts.items()
        },
        "source_reconstruction_max_abs": source_reconstruction_max,
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


def effect(source, updated, labels, ids, args, seed, full_statistics=True):
    if not full_statistics:
        source_correct = source["predictions"][ids] == labels[ids]
        updated_correct = updated["predictions"][ids] == labels[ids]
        margin_change = (
            updated["fixed_margin"][ids] - source["fixed_margin"][ids]
        )
        return {
            "descriptive_only": True,
            "accuracy_change": (
                updated_correct.float().mean()
                - source_correct.float().mean()
            ).item(),
            "corrections": int((~source_correct & updated_correct).sum().item()),
            "regressions": int((source_correct & ~updated_correct).sum().item()),
            "fixed_margin_change_mean": margin_change.mean().item(),
        }
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


def direct_contrast(left, right, ids, args, seed):
    return paired_continuous(
        left["fixed_margin"][ids], right["fixed_margin"][ids], args, seed
    )


def summarize_heads(audits, ids):
    rows = []
    for head in range(HEADS):
        rows.append({
            "head": head,
            "source_entropy_mean": audits["source_entropy"][ids, head]
            .mean().item(),
            "rank_mean": audits["head_rank"][ids, head].float().mean().item(),
            "gradient_energy_mean": audits[
                "head_gradient_energy"
            ][ids, head].mean().item(),
            "allocation_mean": audits["head_allocation"][ids, head]
            .mean().item(),
            "finite_shared_norm_mean": audits[
                "finite_shared_head_norm"
            ][ids, head].mean().item(),
            "finite_shared_first_order_gain_mean": audits[
                "finite_shared_head_gain"
            ][ids, head].mean().item(),
            "finite_head_budget_norm_mean": audits[
                "finite_head_budget_head_norm"
            ][ids, head].mean().item(),
            "finite_head_budget_first_order_gain_mean": audits[
                "finite_head_budget_head_gain"
            ][ids, head].mean().item(),
            "signed_norm_mean": audits[
                "signed_head_norm"
            ][ids, head].mean().item(),
            "signed_first_order_gain_mean": audits[
                "signed_head_gain"
            ][ids, head].mean().item(),
            "positive_feasible_scale_mean": audits[
                "signed_positive_feasible_scale"
            ][ids, head].mean().item(),
            "negative_mass_mean": audits[
                "signed_negative_mass"
            ][ids, head].mean().item(),
            "negative_fraction_mean": audits[
                "signed_negative_fraction"
            ][ids, head].mean().item(),
        })
    return rows


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
    primary = torch.where((~source_correct) & memory_correct)[0]
    if len(primary) < args.minimum_memory_decodable_errors:
        raise RuntimeError(
            f"Only {len(primary)} Memory-decodable errors; minimum is "
            f"{args.minimum_memory_decodable_errors}"
        )
    groups = {
        "all": torch.arange(len(labels)),
        "source_errors": errors,
        "memory_decodable_errors": primary,
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
                full_statistics=name in CORE_CONDITIONS,
            )
            for name_index, (name, row) in enumerate(conditions.items())
            if name != "source"
        }
        for group_index, (group, ids) in enumerate(groups.items())
    }

    signed = conditions["signed_affine_l2"]
    family = {
        "signed_vs_source": direct_contrast(
            source, signed, primary, args, args.analysis_seed + 20_000
        ),
        "signed_vs_finite_shared": direct_contrast(
            conditions["finite_shared_l2"], signed, primary, args,
            args.analysis_seed + 20_001,
        ),
        "signed_vs_negative": direct_contrast(
            conditions["negative_signed_l2"], signed, primary, args,
            args.analysis_seed + 20_002,
        ),
        "signed_vs_rolled": direct_contrast(
            conditions["rolled_signed_l2"], signed, primary, args,
            args.analysis_seed + 20_003,
        ),
        "signed_vs_head_permuted": direct_contrast(
            conditions["head_permuted_signed_l2"], signed, primary, args,
            args.analysis_seed + 20_004,
        ),
    }
    adjusted = holm_adjust({
        name: row["sign_flip_p_two_sided"] for name, row in family.items()
    })
    for name in family:
        family[name]["multiplicity"] = adjusted[name]
    specificity = all(
        row["estimate"] > 0
        and row["multiplicity"]["significant_0.05"]
        for row in family.values()
    )

    source_margin = source["fixed_margin"][primary]
    gains = {
        name: (
            conditions[name]["fixed_margin"][primary] - source_margin
        ).mean().item()
        for name in CORE_CONDITIONS if name != "source"
    }
    unrestricted_gain = gains["unrestricted_context_l2"]
    recovery = {
        name: gain / max(unrestricted_gain, 1e-12)
        for name, gain in gains.items()
    }
    head_budget_contrast = direct_contrast(
        conditions["finite_shared_l2"],
        conditions["finite_head_budget_l2"], primary, args,
        args.analysis_seed + 21_000,
    )
    audits = collected["audit_values"]
    geometry = {
        "target_l2_mean": audits["target_l2"][primary].mean().item(),
        "finite_shared_l2_max_abs_error": audits[
            "finite_shared_l2_error"
        ].max().item(),
        "finite_head_budget_l2_max_abs_error": audits[
            "finite_head_budget_l2_error"
        ].max().item(),
        "signed_l2_max_abs_error": audits[
            "signed_l2_error"
        ].max().item(),
        "delta_closed_loop_max_abs_error": audits[
            "delta_closed_loop_error"
        ].max().item(),
        "manual_fp32_max_abs_error": audits[
            "manual_fp32_error"
        ].max().item(),
        "analytic_gradient_max_abs_error": audits[
            "analytic_gradient_error"
        ].max().item(),
        "full_tangent_rank_mean": audits[
            "full_tangent_rank"
        ][primary].float().mean().item(),
        "full_tangent_rank_min": audits[
            "full_tangent_rank"
        ][primary].min().item(),
        "full_tangent_energy_mean": audits[
            "full_tangent_energy"
        ][primary].mean().item(),
        "head_allocation_energy_mean": audits[
            "head_allocation_energy"
        ][primary].mean().item(),
        "uniform_family_delta_max_abs_error": audits[
            "uniform_family_delta_error"
        ].max().item(),
        "head_budget_selected_start_counts": {
            str(index): int(
                (audits["head_budget_selected_start"][primary] == index)
                .sum().item()
            )
            for index in range(4)
        },
        "head_budget_best_over_uniform_gain_mean": audits[
            "head_budget_best_over_uniform_gain"
        ][primary].mean().item(),
        "head_budget_top_two_gain_gap_mean": audits[
            "head_budget_top_two_gain_gap"
        ][primary].mean().item(),
        "signed_negative_mass_mean": audits[
            "signed_negative_mass"
        ][primary].mean().item(),
        "signed_negative_fraction_mean": audits[
            "signed_negative_fraction"
        ][primary].mean().item(),
        "signed_infeasible_head_fraction": (
            audits["signed_positive_feasible_scale"][primary] < 1.0
        ).float().mean().item(),
    }
    head_rows = summarize_heads(audits, primary)
    for head, row in enumerate(head_rows):
        row["only_margin_gain"] = (
            metrics["memory_decodable_errors"][
                f"signed_head_{head}_only_l2"
            ]["fixed_margin_mean"]
            - metrics["memory_decodable_errors"]["source"][
                "fixed_margin_mean"
            ]
        )
        row["leave_one_out_margin_gain"] = (
            metrics["memory_decodable_errors"][
                f"signed_without_head_{head}_l2"
            ]["fixed_margin_mean"]
            - metrics["memory_decodable_errors"]["source"][
                "fixed_margin_mean"
            ]
        )
        row["unique_margin_loss"] = (
            gains["signed_affine_l2"] - row["leave_one_out_margin_gain"]
        )

    unrestricted_operational = (
        effects["memory_decodable_errors"][
            "unrestricted_context_l2"
        ]["fixed_margin"]["estimate"] > 0
        and effects["memory_decodable_errors"][
            "unrestricted_context_l2"
        ]["fixed_margin"]["sign_flip_p_two_sided"] < 0.05
    )
    head_budget_closes = (
        recovery["finite_head_budget_l2"]
        >= args.recovery_threshold
        and head_budget_contrast["estimate"] > 0
        and head_budget_contrast["sign_flip_p_two_sided"] < 0.05
    )
    signed_closes = (
        recovery["signed_affine_l2"] >= args.recovery_threshold
        and specificity
    )
    if not unrestricted_operational:
        classification = "positive_control_failure"
        boundary = "Stop; repair the unrestricted context positive control."
    elif head_budget_closes:
        classification = "head_budget_allocation_obstruction"
        boundary = (
            "Finite attention closes the gap after head-wise dose "
            "reallocation; preregister a donor-free head-budget router rule."
        )
    elif signed_closes:
        classification = "signed_affine_simplex_obstruction"
        boundary = (
            "Head-budget reallocation is insufficient but signed affine "
            "value mixing closes the tangent gap; isolate the minimal heads "
            "and test a gated residual/signed-value read."
        )
    else:
        classification = "head_subspace_interaction_unresolved"
        boundary = (
            "Neither finite head budgets nor the registered signed mixture "
            "closes the gap; audit nonlinear downstream curvature and "
            "cross-head output overlap before modifying the read."
        )

    matching.update({
        "errors": int(len(errors)),
        "matched_correct": int(len(matched)),
        "memory_decodable_errors": int(len(primary)),
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
        "memory_decodable_errors": int(len(primary)),
        "signed_specificity_passed": specificity,
        "unrestricted_positive_control_operational": unrestricted_operational,
        "head_budget_closes_gap": head_budget_closes,
        "signed_affine_closes_gap": signed_closes,
        "registered_next_boundary": boundary,
    }
    return {
        "matching": matching,
        "metrics": metrics,
        "effects_vs_source": effects,
        "primary": {
            "population": "Memory-decodable source errors",
            "signed_specificity": family,
            "head_budget_vs_shared": head_budget_contrast,
            "margin_gains": gains,
            "recovery_fractions": recovery,
        },
        "geometry": geometry,
        "head_tomography": head_rows,
        "diagnosis": diagnosis,
    }, groups


def plot_result(analysis, path):
    primary = analysis["metrics"]["memory_decodable_errors"]
    source_margin = primary["source"]["fixed_margin_mean"]
    core = [
        "finite_shared_l2", "finite_head_budget_l2", "signed_affine_l2",
        "negative_signed_l2", "rolled_signed_l2",
        "head_permuted_signed_l2", "unrestricted_context_l2",
    ]
    labels = [
        "Finite\nshared", "Finite\nhead budget", "Signed\naffine",
        "Negative\nsigned", "Rolled\nsigned", "Head\npermuted",
        "Full\ncontext",
    ]
    heads = analysis["head_tomography"]
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    axes[0].bar(
        np.arange(len(core)),
        [primary[name]["fixed_margin_mean"] - source_margin for name in core],
        color=["#E15759", "#F28E2B", "#B07AA1", "#9C755F",
               "#BAB0AC", "#76B7B2", "#4E79A7"],
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(np.arange(len(core)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Deployed margin change")
    axes[0].set_title("Equal-L2 mechanism panel")

    x = np.arange(HEADS)
    axes[1].bar(
        x - 0.2, [row["only_margin_gain"] for row in heads], width=0.4,
        label="Head only", color="#59A14F",
    )
    axes[1].bar(
        x + 0.2, [row["unique_margin_loss"] for row in heads], width=0.4,
        label="Loss when removed", color="#EDC948",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, [str(head) for head in range(HEADS)])
    axes[1].set_xlabel("Read head")
    axes[1].set_ylabel("Primary deployed-margin effect")
    axes[1].set_title("Head-only and leave-one-out tomography")
    axes[1].legend()

    axes[2].bar(
        x - 0.2,
        [row["allocation_mean"] for row in heads], width=0.4,
        label="Finite allocation", color="#F28E2B",
    )
    axes[2].bar(
        x + 0.2,
        [row["negative_fraction_mean"] for row in heads], width=0.4,
        label="Signed negative fraction", color="#B07AA1",
    )
    axes[2].set_xticks(x, [str(head) for head in range(HEADS)])
    axes[2].set_xlabel("Read head")
    axes[2].set_title("Dose allocation and simplex violation")
    axes[2].legend()
    figure.suptitle(
        "IST Level 6.19.3: Head-Budget and Signed-Value Tomography",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def preregistration(args):
    return {
        "level": "6.19.3",
        "status": "frozen head-budget and signed-value tomography",
        "source": "formally passed Level 6.18.3 seed707 checkpoint",
        "parent_result": (
            "Level 6.19.2 finite_attention_simplex_or_budget_limitation"
        ),
        "seed": SEED,
        "chunks": CHUNKS,
        "primary_population": (
            "source errors whose frozen Level 6.19 Memory Probe is correct"
        ),
        "common_dose": (
            "per-example context L2 induced by Probe top-four 4x odds"
        ),
        "conditions": {
            "finite_shared_l2": (
                "Level 6.19.2 gradient attention tilt with one shared dose"
            ),
            "finite_head_budget_l2": (
                "eight-dimensional nonnegative finite-softmax Oracle, "
                "initialized from uniform, linearized NNLS, square-root NNLS, "
                "and a peaked best-head allocation, "
                "optimized on exact finite context and rematched to total L2"
            ),
            "signed_affine_l2": (
                "minimum-coefficient zero-sum signed per-head value mixture "
                "projecting the deployed context gradient into the full "
                "frozen value/output tangent span"
            ),
            "controls": [
                "negative signed direction",
                "cross-example rolled signed direction",
                "head-permuted signed coefficients",
                "unrestricted context gradient",
            ],
            "head_tomography": (
                "eight equal-L2 one-head projections and eight equal-L2 "
                "leave-one-head-out projections"
            ),
        },
        "primary_signed_family": [
            "signed versus source",
            "signed versus finite shared-dose attention",
            "signed versus negative signed",
            "signed versus rolled signed",
            "signed versus head-permuted signed",
        ],
        "multiplicity": "Holm across five signed-direction contrasts",
        "decision_rule": {
            "recovery_threshold": args.recovery_threshold,
            "head_budget_closure": (
                "finite head-budget recovery >= threshold and positive "
                "head-budget versus shared-dose margin contrast at p < 0.05"
            ),
            "signed_closure": (
                "signed recovery >= threshold and every signed-direction "
                "contrast positive with Holm p < 0.05"
            ),
            "positive_control": (
                "unrestricted equal-L2 context margin gain positive at p < 0.05"
            ),
        },
        "interpretation": (
            "all optimized directions use labels and are mechanistic, not deployable"
        ),
        "locks": {
            "all_model_and_probe_parameters_frozen": True,
            "failed_Level_6_18_9_candidate_not_used": True,
            "protected_tests_not_used": True,
            "seed909_locked": True,
            "optimizer_search_closed": True,
        },
        "numerical_audit": {
            "revision": NUMERICAL_REVISION,
            "intervention": (
                "explicit updated-minus-source FP32 attention delta added "
                "to exact native source context"
            ),
            "manual_internal_tolerance": 1e-5,
            "closed_loop_tolerance": 1e-5,
            "attention_l2_tolerance": 1e-3,
            "signed_l2_tolerance": 1e-5,
            "analytic_gradient_tolerance": 1e-6,
        },
        "protocol": {
            key: value for key, value in vars(args).items()
            if key not in {"dry_run", "smoke_test", "force"}
        },
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError("Level 6.19.3 is fixed to seed707 at 16 chunks")
    for path in (args.checkpoint, args.probes):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.samples <= 0 or args.eval_batch_size <= 0:
        raise ValueError("samples and eval-batch-size must be positive")
    if args.samples % args.eval_batch_size != 0:
        raise ValueError("samples must be divisible by eval-batch-size")
    if not args.smoke_test and (
        args.samples != 4096 or args.dataset_seed != 6193300
    ):
        raise ValueError(
            "Formal Level 6.19.3 fixes samples=4096 and dataset-seed=6193300; "
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
        description="Level 6.19.3 head-budget and signed-value tomography"
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
    parser.add_argument("--dataset-seed", type=int, default=6193300)
    parser.add_argument("--analysis-seed", type=int, default=6193400)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--minimum-errors", type=int, default=200)
    parser.add_argument("--minimum-memory-decodable-errors", type=int, default=150)
    parser.add_argument("--recovery-threshold", type=float, default=0.80)
    parser.add_argument("--log-every-samples", type=int, default=256)
    parser.add_argument("--output", default="experiments/level6_19_3/formal")
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
    geometry = analysis["geometry"]
    integrity = {
        "source_downstream_max_abs": collected[
            "source_reconstruction_max_abs"
        ],
        "source_downstream_reconstruction_exact": (
            collected["source_reconstruction_max_abs"] == 0.0
        ),
        "manual_fp32_max_abs_error": geometry["manual_fp32_max_abs_error"],
        "manual_fp32_passed": geometry["manual_fp32_max_abs_error"] <= 1e-5,
        "delta_closed_loop_max_abs_error": geometry[
            "delta_closed_loop_max_abs_error"
        ],
        "delta_closed_loop_passed": geometry[
            "delta_closed_loop_max_abs_error"
        ] <= 1e-5,
        "analytic_gradient_max_abs_error": geometry[
            "analytic_gradient_max_abs_error"
        ],
        "analytic_gradient_passed": geometry[
            "analytic_gradient_max_abs_error"
        ] <= 1e-6,
        "uniform_family_delta_max_abs_error": geometry[
            "uniform_family_delta_max_abs_error"
        ],
        "uniform_family_reproduced": geometry[
            "uniform_family_delta_max_abs_error"
        ] <= 1e-5,
        "finite_shared_l2_max_abs_error": geometry[
            "finite_shared_l2_max_abs_error"
        ],
        "finite_head_budget_l2_max_abs_error": geometry[
            "finite_head_budget_l2_max_abs_error"
        ],
        "attention_l2_passed": max(
            geometry["finite_shared_l2_max_abs_error"],
            geometry["finite_head_budget_l2_max_abs_error"],
        ) <= 1e-3,
        "signed_l2_max_abs_error": geometry["signed_l2_max_abs_error"],
        "signed_l2_passed": geometry["signed_l2_max_abs_error"] <= 1e-5,
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
        "optimizer_search_closed": True,
    }
    integrity["passed"] = all([
        integrity["source_downstream_reconstruction_exact"],
        integrity["manual_fp32_passed"],
        integrity["delta_closed_loop_passed"],
        integrity["analytic_gradient_passed"],
        integrity["uniform_family_reproduced"],
        integrity["attention_l2_passed"],
        integrity["signed_l2_passed"],
        integrity["all_states_unchanged"],
        integrity["all_parameters_frozen"],
    ])
    analysis["diagnosis"]["integrity_passed"] = integrity["passed"]
    if not integrity["passed"]:
        analysis["diagnosis"]["classification"] = "integrity_failure"
        analysis["diagnosis"]["registered_next_boundary"] = (
            "Stop; repair the Level 6.19.3 implementation."
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
        "geometry": analysis["geometry"],
        "head_tomography": analysis["head_tomography"],
        "diagnosis": analysis["diagnosis"],
    })
    save(root / "predictions.json", raw_predictions(collected, groups))
    plot_result(analysis, root / "head_signed_tomography.png")
    print(json.dumps(analysis["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
