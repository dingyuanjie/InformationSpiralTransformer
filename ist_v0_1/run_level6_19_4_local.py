import argparse
import copy
import hashlib
import itertools
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
    SLOTS,
    WIDTH,
    attention_decomposition,
    attention_delta,
    downstream,
    patch_context,
    task_competitor,
)
from run_level6_19_3_local import signed_affine_solution, tangent_basis


SEED = 707
CHUNKS = 16
LEVEL = "6.19.4"
PARENT_DATASET_SEED = 6193300
SUBSET_SEED = PARENT_DATASET_SEED
TRAIN_SEED = 6194100
VALIDATION_SEED = 6194200
DIAGNOSTIC_SEED = 6194300
ANALYSIS_SEED = 6194400
ROUTER_SEED = 6194500
DOSE_CAP = 0.8642004728317261
NUMERICAL_REVISION = (
    "query_delta_training_exact_full_diagnostic_"
    "simplex_8192_recovery_v2"
)

ROUTER_KINDS = ["signed", "nonnegative", "residual"]
CONDITIONS = [
    "source",
    "signed_router",
    "nonnegative_router",
    "matched_residual_router",
    "signed_shuffled_memory",
    "signed_rolled_delta",
    "signed_head_permuted",
    "label_oracle_selected_subset",
    "label_oracle_full_signed",
]


def subset_masks(device="cpu"):
    rows = []
    labels = []
    for size in range(1, HEADS + 1):
        for heads in itertools.combinations(range(HEADS), size):
            mask = torch.zeros(HEADS, dtype=torch.float32)
            mask[list(heads)] = 1.0
            rows.append(mask)
            labels.append(list(heads))
    return torch.stack(rows).to(device), labels


def parameter_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def projected_atoms(values, out_weight):
    rows = []
    head_width = WIDTH // HEADS
    for head in range(HEADS):
        weight = out_weight[:, head * head_width:(head + 1) * head_width]
        rows.append(F.linear(values[:, head], weight, None))
    return torch.stack(rows, dim=1)


def target_dose(read_memory, labels, competitor, probes, source_attention,
                values, out_weight):
    _, slots, _, memory_logits = select_slots(
        read_memory, labels, competitor, probes["memory_l3_concat"]
    )
    logits = source_attention.clamp_min(1e-12).log()
    bias = torch.zeros_like(logits)
    expanded = slots[:, None].expand(-1, HEADS, -1)
    bias.scatter_add_(
        -1, expanded,
        torch.full_like(expanded, math.log(4.0), dtype=logits.dtype),
    )
    updated = torch.softmax(logits + bias, dim=-1)
    delta = attention_delta(updated, source_attention, values, out_weight)
    return delta.norm(dim=-1), memory_logits.argmax(dim=-1)


def context_margin_gradient(model, captured, source_context, labels,
                            competitor, dtype):
    leaf = source_context.detach().clone().requires_grad_(True)
    logits = downstream(model, captured, leaf, dtype, True)
    rows = torch.arange(len(labels), device=labels.device)
    margin = logits[rows, labels] - logits[rows, competitor]
    gradient = torch.autograd.grad(
        margin.sum(), leaf, only_inputs=True
    )[0][:, -1].detach().float()
    return gradient, logits.detach()


class GatedReadRouter(nn.Module):
    """Equal-parameter input-conditioned read used under three output rules."""

    def __init__(self, kind, hidden, head_mask, dose_cap, basis_seed):
        super().__init__()
        if kind not in ROUTER_KINDS:
            raise ValueError(kind)
        self.kind = kind
        self.hidden = hidden
        self.global_net = nn.Sequential(
            nn.Linear(WIDTH * 3, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.atom_net = nn.Linear(WIDTH, hidden, bias=False)
        self.attention_net = nn.Linear(2, hidden, bias=False)
        self.head_embedding = nn.Parameter(torch.zeros(HEADS, hidden))
        self.slot_bias = nn.Parameter(torch.zeros(HEADS, SLOTS))
        self.score = nn.Linear(hidden, 1)
        self.gate = nn.Linear(hidden, 1)
        self.register_buffer("head_mask", head_mask.float().reshape(1, HEADS, 1))
        self.register_buffer("dose_cap", torch.tensor(float(dose_cap)))
        generator = torch.Generator().manual_seed(basis_seed)
        residual = torch.randn(HEADS, SLOTS, WIDTH, generator=generator)
        residual = residual - residual.mean(dim=1, keepdim=True)
        residual = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("residual_basis", residual)
        nn.init.normal_(self.head_embedding, std=0.02)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def slot_scores(self, query, pre_fusion, source_context, source_attention,
                    atoms):
        global_hidden = self.global_net(torch.cat([
            query, pre_fusion, source_context
        ], dim=-1))
        atom_unit = atoms / atoms.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        attention_features = torch.stack([
            source_attention,
            source_attention.clamp_min(1e-8).log() / 8.0,
        ], dim=-1)
        hidden = (
            self.atom_net(atom_unit)
            + self.attention_net(attention_features)
            + global_hidden[:, None, None]
            + self.head_embedding[None, :, None]
        )
        scores = self.score(F.gelu(hidden)).squeeze(-1) + self.slot_bias
        gate = torch.sigmoid(self.gate(global_hidden)).squeeze(-1)
        return scores, gate

    def signed_delta(self, scores, gate, atoms, coefficients=None):
        if coefficients is None:
            coefficients = scores - scores.mean(dim=-1, keepdim=True)
            coefficients = coefficients * self.head_mask
        direction = torch.einsum("bhs,bhsd->bd", coefficients, atoms)
        delta = norm_match(direction, self.dose_cap.expand(len(direction)))
        delta = delta * gate[:, None]
        return delta, coefficients

    def forward(self, query, pre_fusion, source_context, source_attention,
                atoms):
        scores, gate = self.slot_scores(
            query, pre_fusion, source_context, source_attention, atoms
        )
        if self.kind == "signed":
            delta, coefficients = self.signed_delta(scores, gate, atoms)
            return {"delta": delta, "gate": gate, "coefficients": coefficients}
        if self.kind == "nonnegative":
            proposal = torch.softmax(
                source_attention.clamp_min(1e-8).log() + scores, dim=-1
            )
            mask = self.head_mask.bool()
            updated = torch.where(mask, proposal, source_attention)
            direction = torch.einsum(
                "bhs,bhsd->bd", updated - source_attention, atoms
            )
            scale = (
                self.dose_cap
                / direction.norm(dim=-1).clamp_min(1e-8)
            ).clamp_max(1.0)
            delta = direction * (scale * gate)[:, None]
            return {"delta": delta, "gate": gate, "coefficients": updated}
        coefficients = scores - scores.mean(dim=-1, keepdim=True)
        coefficients = coefficients * self.head_mask
        direction = torch.einsum(
            "bhs,hsd->bd", coefficients, self.residual_basis
        )
        delta = norm_match(direction, self.dose_cap.expand(len(direction)))
        delta = delta * gate[:, None]
        return {"delta": delta, "gate": gate, "coefficients": coefficients}


def query_downstream(model, query, pre_fusion, memory_context, dtype):
    block = model.blocks[-1]
    with torch.autocast(device_type="cuda", dtype=dtype):
        gate = block.memory_fusion_gate(
            torch.cat([query, memory_context], dim=-1)
        )
        fused = pre_fusion + gate * memory_context
        hidden = block.norm2(query + block.ffn(fused))
        logits = model.output(hidden)[:, :16].float()
    return logits


def cache_batch(cache, ids, device):
    return {
        key: value[ids].to(
            device=device,
            dtype=torch.long if key == "labels" else torch.float32,
        )
        for key, value in cache.items()
    }


def collect_router_cache(model, probes, selected_mask, args, samples, seed,
                         device, dtype, root, split):
    parts = {
        name: [] for name in [
            "query", "pre_fusion", "source_context", "source_attention",
            "atoms", "source_logits", "labels",
            "teacher_delta",
        ]
    }
    trace = FinalTrace(model)
    total = 0
    replay_error = 0.0
    set_seed(seed)
    try:
        while total < samples:
            batch = min(args.eval_batch_size, samples - total)
            chunks, labels, _ = make_chunks(
                batch, CHUNKS, args.chunk_size, device
            )
            memory = None
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=dtype
            ):
                for index in range(CHUNKS - 1):
                    _, memory = model(
                        chunks[:, index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                trace.clear()
                native_logits, _ = model(
                    chunks[:, -1], memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
            trace.require()
            captured = trace.values
            decomposition = attention_decomposition(model, captured, dtype)
            source_logits = native_logits[:, -1, :16].float()
            query = captured["read_query"][:, -1].float()
            pre_fusion = captured["pre_fusion_feature"][:, -1].float()
            context = captured["memory_context"][:, -1].float()
            with torch.no_grad():
                replay = query_downstream(
                    model, query, pre_fusion, context, dtype
                )
            replay_error = max(
                replay_error, (replay - source_logits).abs().max().item()
            )
            values = decomposition["values"]
            atoms = projected_atoms(values, decomposition["out_weight"])
            competitor = task_competitor(source_logits, labels)
            target_l2, _ = target_dose(
                captured["read_memory"], labels, competitor, probes,
                decomposition["weights"], decomposition["values"],
                decomposition["out_weight"],
            )
            gradient, _ = context_margin_gradient(
                model, captured, captured["memory_context"], labels,
                competitor, dtype,
            )
            signed = signed_affine_solution(
                gradient,
                tangent_basis(
                    decomposition["values"], decomposition["out_weight"]
                ),
                target_l2,
            )
            teacher_delta = (
                signed["head_delta"] * selected_mask[None, :, None]
            ).sum(dim=1)
            teacher_delta = norm_match(teacher_delta, target_l2)
            rows = {
                "query": query,
                "pre_fusion": pre_fusion,
                "source_context": context,
                "source_attention": decomposition["weights"],
                "atoms": atoms,
                "source_logits": source_logits,
                "labels": labels,
                "teacher_delta": teacher_delta,
            }
            for name, value in rows.items():
                target_dtype = torch.long if name == "labels" else torch.float32
                parts[name].append(value.detach().cpu().to(target_dtype))
            total += batch
            if total == batch or total % args.log_every_samples == 0:
                print(
                    f"Level {LEVEL} cache {split}={total}/{samples}",
                    flush=True,
                )
                save(root / "progress.json", {
                    "stage": f"cache_{split}",
                    "samples_complete": total,
                    "samples_total": samples,
                    "query_replay_max_abs_so_far": replay_error,
                })
    finally:
        trace.close()
    return {name: torch.cat(values) for name, values in parts.items()}, replay_error


def offline_router_logits(model, batch, delta, dtype):
    source_replay = query_downstream(
        model, batch["query"], batch["pre_fusion"], batch["source_context"],
        dtype,
    )
    updated_replay = query_downstream(
        model, batch["query"], batch["pre_fusion"],
        batch["source_context"] + delta,
        dtype,
    )
    return batch["source_logits"] + updated_replay - source_replay


def evaluate_router_cache(model, router, cache, args, device, dtype):
    router.eval()
    losses = []
    predictions = []
    gates = []
    teacher_losses = []
    with torch.no_grad():
        for start in range(0, len(cache["labels"]), args.router_batch_size):
            ids = slice(start, start + args.router_batch_size)
            batch = cache_batch(cache, ids, device)
            output = router(
                batch["query"], batch["pre_fusion"],
                batch["source_context"], batch["source_attention"],
                batch["atoms"],
            )
            logits = offline_router_logits(
                model, batch, output["delta"], dtype
            )
            losses.append(F.cross_entropy(
                logits, batch["labels"], reduction="none"
            ).cpu())
            teacher_unit = batch["teacher_delta"] / batch[
                "teacher_delta"
            ].norm(dim=-1, keepdim=True).clamp_min(1e-8)
            delta_unit = output["delta"] / output["delta"].norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            cosine_loss = 1.0 - (teacher_unit * delta_unit).sum(dim=-1)
            dose_loss = (
                output["delta"].norm(dim=-1)
                / batch["teacher_delta"].norm(dim=-1).clamp_min(1e-8)
                - 1.0
            ).square()
            teacher_losses.append((cosine_loss + 0.25 * dose_loss).cpu())
            predictions.append(logits.argmax(dim=-1).cpu())
            gates.append(output["gate"].cpu())
    loss = torch.cat(losses)
    prediction = torch.cat(predictions)
    labels = cache["labels"]
    teacher_loss = torch.cat(teacher_losses).mean().item()
    return {
        "loss": loss.mean().item(),
        "teacher_loss": teacher_loss,
        "selection_objective": loss.mean().item() + (
            args.distillation_weight * teacher_loss
        ),
        "accuracy": (prediction == labels).float().mean().item(),
        "gate_mean": torch.cat(gates).mean().item(),
    }


def train_router(model, kind, head_mask, train_cache, val_cache, args, device,
                 dtype, seed):
    set_seed(seed)
    router = GatedReadRouter(
        kind, args.router_hidden, head_mask.to(device), DOSE_CAP,
        args.residual_basis_seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.router_lr,
        weight_decay=args.router_weight_decay,
    )
    best = None
    best_loss = math.inf
    patience = 0
    history = []
    generator = torch.Generator().manual_seed(seed + 1)
    for epoch in range(1, args.router_epochs + 1):
        router.train()
        order = torch.randperm(len(train_cache["labels"]), generator=generator)
        running = []
        for start in range(0, len(order), args.router_batch_size):
            ids = order[start:start + args.router_batch_size]
            batch = cache_batch(train_cache, ids, device)
            output = router(
                batch["query"], batch["pre_fusion"],
                batch["source_context"], batch["source_attention"],
                batch["atoms"],
            )
            logits = offline_router_logits(
                model, batch, output["delta"], dtype
            )
            teacher_unit = batch["teacher_delta"] / batch[
                "teacher_delta"
            ].norm(dim=-1, keepdim=True).clamp_min(1e-8)
            delta_unit = output["delta"] / output["delta"].norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            cosine_loss = 1.0 - (teacher_unit * delta_unit).sum(dim=-1)
            dose_loss = (
                output["delta"].norm(dim=-1)
                / batch["teacher_delta"].norm(dim=-1).clamp_min(1e-8)
                - 1.0
            ).square()
            teacher_loss = cosine_loss.mean() + 0.25 * dose_loss.mean()
            loss = (
                F.cross_entropy(logits, batch["labels"])
                + args.distillation_weight * teacher_loss
            )
            loss = loss + args.gate_penalty * output["gate"].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            running.append(loss.detach().item())
        validation = evaluate_router_cache(
            model, router, val_cache, args, device, dtype
        )
        row = {
            "epoch": epoch,
            "train_objective": float(np.mean(running)),
            "validation": validation,
        }
        history.append(row)
        print(
            f"Level {LEVEL} router={kind} epoch={epoch} "
            f"train={row['train_objective']:.5f} "
            f"val={validation['loss']:.5f} "
            f"acc={validation['accuracy']:.2%} "
            f"gate={validation['gate_mean']:.3f}",
            flush=True,
        )
        if (
            validation["selection_objective"]
            < best_loss - args.minimum_loss_improvement
        ):
            best_loss = validation["selection_objective"]
            best = copy.deepcopy(router.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best is None:
        raise RuntimeError(f"{kind} router did not produce a checkpoint")
    router.load_state_dict(best)
    final = evaluate_router_cache(
        model, router, val_cache, args, device, dtype
    )
    return router, {
        "kind": kind,
        "parameters": sum(parameter.numel() for parameter in router.parameters()),
        "best_epoch": best_epoch,
        "best_validation": final,
        "epochs_run": len(history),
        "history": history,
    }


def calibrate_subsets(model, probes, args, device, dtype, root):
    masks, labels_for_masks = subset_masks(device)
    deployed_gains = []
    first_order_gains = []
    full_deployed_gains = []
    full_first_order_gains = []
    source_errors = 0
    primary_count = 0
    source_correct = 0
    total = 0
    trace = FinalTrace(model)
    set_seed(args.subset_seed)
    try:
        while total < args.subset_samples:
            batch = min(args.eval_batch_size, args.subset_samples - total)
            chunks, labels, _ = make_chunks(
                batch, CHUNKS, args.chunk_size, device
            )
            memory = None
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=dtype
            ):
                for index in range(CHUNKS - 1):
                    _, memory = model(
                        chunks[:, index], memory=memory,
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
            predictions = source_logits.argmax(dim=-1)
            competitor = task_competitor(source_logits, labels)
            decomposition = attention_decomposition(model, captured, dtype)
            target_l2, memory_predictions = target_dose(
                captured["read_memory"], labels, competitor, probes,
                decomposition["weights"], decomposition["values"],
                decomposition["out_weight"],
            )
            gradient, _ = context_margin_gradient(
                model, captured, captured["memory_context"], labels,
                competitor, dtype,
            )
            signed = signed_affine_solution(
                gradient,
                tangent_basis(
                    decomposition["values"], decomposition["out_weight"]
                ),
                target_l2,
            )
            primary = (predictions != labels) & (memory_predictions == labels)
            source_correct += int((predictions == labels).sum().item())
            source_errors += int((predictions != labels).sum().item())
            if primary.any():
                head_delta = signed["head_delta"][primary]
                subset_delta = torch.einsum("kh,bhd->bkd", masks, head_delta)
                subset_delta = subset_delta / subset_delta.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8) * target_l2[primary, None, None]
                subset_first_order = torch.einsum(
                    "bkd,bd->bk", subset_delta, gradient[primary]
                )
                full_first_order = (
                    signed["delta"][primary] * gradient[primary]
                ).sum(dim=-1)
                query = captured["read_query"][:, -1].float()[primary]
                pre = captured["pre_fusion_feature"][:, -1].float()[primary]
                context = captured["memory_context"][:, -1].float()[primary]
                primary_logits = source_logits[primary]
                primary_labels = labels[primary]
                primary_competitor = competitor[primary]
                rows = torch.arange(len(primary_labels), device=device)
                source_fixed = (
                    primary_logits[rows, primary_labels]
                    - primary_logits[rows, primary_competitor]
                )
                with torch.no_grad():
                    source_replay = query_downstream(
                        model, query, pre, context, dtype
                    )
                    repeat = len(labels_for_masks)
                    updated_replay = query_downstream(
                        model,
                        query[:, None].expand(-1, repeat, -1).reshape(-1, WIDTH),
                        pre[:, None].expand(-1, repeat, -1).reshape(-1, WIDTH),
                        (context[:, None] + subset_delta).reshape(-1, WIDTH),
                        dtype,
                    ).reshape(len(query), repeat, -1)
                    updated_logits = (
                        primary_logits[:, None]
                        + updated_replay
                        - source_replay[:, None]
                    )
                    subset_fixed = (
                        updated_logits[rows, :, primary_labels]
                        - updated_logits[rows, :, primary_competitor]
                    )
                    full_replay = query_downstream(
                        model, query, pre,
                        context + signed["delta"][primary], dtype,
                    )
                    full_logits = primary_logits + full_replay - source_replay
                    full_fixed = (
                        full_logits[rows, primary_labels]
                        - full_logits[rows, primary_competitor]
                    )
                deployed_gains.append((subset_fixed - source_fixed[:, None]).cpu())
                first_order_gains.append(subset_first_order.detach().cpu())
                full_deployed_gains.append((full_fixed - source_fixed).cpu())
                full_first_order_gains.append(full_first_order.detach().cpu())
                primary_count += int(primary.sum().item())
            total += batch
            if total == batch or total % args.log_every_samples == 0:
                print(
                    f"Level {LEVEL} subset={total}/{args.subset_samples} "
                    f"source={source_correct / total:.2%} primary={primary_count}",
                    flush=True,
                )
                save(root / "progress.json", {
                    "stage": "subset_calibration",
                    "samples_complete": total,
                    "samples_total": args.subset_samples,
                    "primary_examples": primary_count,
                })
    finally:
        trace.close()
    if primary_count < args.minimum_subset_primary:
        raise RuntimeError(
            f"subset calibration has {primary_count} primary examples; "
            f"requires {args.minimum_subset_primary}"
        )
    deployed_gain = torch.cat(deployed_gains)
    first_order_gain = torch.cat(first_order_gains)
    full_deployed_gain = torch.cat(full_deployed_gains)
    full_first_order_gain = torch.cat(full_first_order_gains)
    mean_deployed_gain = deployed_gain.mean(dim=0)
    mean_first_order_gain = first_order_gain.mean(dim=0)
    recovery = (
        mean_deployed_gain / full_deployed_gain.mean().clamp_min(1e-12)
    )
    rows = []
    for index, heads in enumerate(labels_for_masks):
        rows.append({
            "index": index,
            "heads": heads,
            "size": len(heads),
            "mean_deployed_margin_gain": mean_deployed_gain[index].item(),
            "mean_first_order_gain": mean_first_order_gain[index].item(),
            "recovery_fraction": recovery[index].item(),
            "positive_fraction": (
                deployed_gain[:, index] > 0
            ).float().mean().item(),
        })

    def choose(threshold):
        candidates = [row for row in rows if row["recovery_fraction"] >= threshold]
        if not candidates:
            return max(rows, key=lambda row: row["recovery_fraction"])
        minimum_size = min(row["size"] for row in candidates)
        same_size = [row for row in candidates if row["size"] == minimum_size]
        return max(same_size, key=lambda row: (
            row["recovery_fraction"], [-head for head in row["heads"]]
        ))

    selected80 = choose(args.subset_recovery_80)
    selected90 = choose(args.subset_recovery_90)
    result = {
        "samples": args.subset_samples,
        "dataset_seed": args.subset_seed,
        "source_accuracy": source_correct / total,
        "source_errors": source_errors,
        "primary_examples": primary_count,
        "full_signed_deployed_margin_gain_mean": full_deployed_gain.mean().item(),
        "full_signed_first_order_gain_mean": full_first_order_gain.mean().item(),
        "selected_80": selected80,
        "selected_90": selected90,
        "subsets": rows,
    }
    return result


def project_product_simplex(values):
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    sorted_values, _ = flat.sort(dim=-1, descending=True)
    cumulative = sorted_values.cumsum(dim=-1) - 1.0
    indices = torch.arange(
        1, shape[-1] + 1, device=values.device, dtype=values.dtype
    )
    support = sorted_values - cumulative / indices > 0
    rho = support.sum(dim=-1).clamp_min(1)
    theta = cumulative.gather(1, (rho - 1)[:, None]).squeeze(1) / rho
    return (flat - theta[:, None]).clamp_min(0).reshape(shape)


def simplex_target_projection(source, atoms, target, iterations):
    centered = atoms - atoms.mean(dim=2, keepdim=True)
    flat_atoms = centered.reshape(len(centered), HEADS * SLOTS, WIDTH)
    context_gram = torch.matmul(flat_atoms.transpose(1, 2), flat_atoms)
    lipschitz = torch.linalg.eigvalsh(context_gram)[:, -1].clamp_min(1e-6)
    starts = torch.stack([source, torch.full_like(source, 1.0 / SLOTS)], dim=1)
    atom = centered[:, None]
    target_expanded = target[:, None]
    current = starts.clone()
    accelerated = current.clone()
    momentum = torch.ones(
        len(source), 2, device=source.device, dtype=source.dtype
    )
    for _ in range(iterations):
        delta = torch.einsum(
            "brhs,brhsd->brd", accelerated - source[:, None], atom
        )
        residual = delta - target_expanded
        gradient = torch.einsum("brd,brhsd->brhs", residual, atom)
        updated = project_product_simplex(
            accelerated - gradient / lipschitz[:, None, None, None]
        )
        next_momentum = (1.0 + torch.sqrt(1.0 + 4.0 * momentum.square())) / 2.0
        factor = ((momentum - 1.0) / next_momentum)[..., None, None]
        accelerated = updated + factor * (updated - current)
        current = updated
        momentum = next_momentum
    delta = torch.einsum("brhs,brhsd->brd", current - source[:, None], atom)
    error = (delta - target_expanded).norm(dim=-1)
    best = error.argmin(dim=1)
    rows = torch.arange(len(source), device=source.device)
    selected = current[rows, best]
    selected_delta = delta[rows, best]
    selected_error = error[rows, best]
    restart_delta_gap = (
        (delta[:, 0] - delta[:, 1]).norm(dim=-1)
        / target.norm(dim=-1).clamp_min(1e-8)
    )
    gradient = torch.einsum(
        "bd,bhsd->bhs", selected_delta - target, centered
    )
    projected = project_product_simplex(
        selected - gradient / lipschitz[:, None, None]
    )
    mapping = (selected - projected).norm(dim=(1, 2))
    return {
        "attention": selected,
        "delta": selected_delta,
        "absolute_error": selected_error,
        "relative_error": selected_error / target.norm(dim=-1).clamp_min(1e-8),
        "restart_delta_gap": restart_delta_gap,
        "projected_gradient_mapping": mapping,
    }


def empty_condition_parts():
    return {
        condition: {
            field: [] for field in [
                "predictions", "fixed_margin", "decision_margin",
                "cross_entropy", "context_delta_norm", "gate",
            ]
        }
        for condition in CONDITIONS
    }


def append_diagnostic(parts, name, logits, labels, competitor, delta, gate=None):
    rows = torch.arange(len(labels), device=labels.device)
    correct = logits[rows, labels]
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    row = parts[name]
    row["predictions"].append(logits.argmax(dim=-1).detach().cpu())
    row["fixed_margin"].append(
        (correct - logits[rows, competitor]).detach().cpu()
    )
    row["decision_margin"].append(
        (correct - masked.max(dim=-1).values).detach().cpu()
    )
    row["cross_entropy"].append(
        F.cross_entropy(logits, labels, reduction="none").detach().cpu()
    )
    row["context_delta_norm"].append(delta.norm(dim=-1).detach().cpu())
    if gate is None:
        gate = torch.zeros(len(labels), device=labels.device)
    row["gate"].append(gate.detach().cpu())


def diagnostic_evaluation(model, probes, routers, selected_mask, args, device,
                          dtype, root):
    parts = empty_condition_parts()
    labels_parts = []
    confidence_parts = []
    competitor_parts = []
    memory_prediction_parts = []
    simplex_parts = {name: [] for name in [
        "relative_error", "absolute_error", "restart_delta_gap",
        "projected_gradient_mapping", "target_l2",
    ]}
    simplex_source_parts = []
    simplex_atom_parts = []
    simplex_target_parts = []
    simplex_batch_ids = []
    source_reconstruction_error = 0.0
    selected_oracle_l2_error = 0.0
    full_oracle_l2_error = 0.0
    total = 0
    source_correct = 0
    trace = FinalTrace(model)
    set_seed(args.diagnostic_seed)
    try:
        while total < args.diagnostic_samples:
            batch = min(args.eval_batch_size, args.diagnostic_samples - total)
            chunks, labels, _ = make_chunks(
                batch, CHUNKS, args.chunk_size, device
            )
            memory = None
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=dtype
            ):
                for index in range(CHUNKS - 1):
                    _, memory = model(
                        chunks[:, index], memory=memory,
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
            confidence = source_logits.topk(2, dim=-1).values
            confidence = confidence[:, 0] - confidence[:, 1]
            decomposition = attention_decomposition(model, captured, dtype)
            source_attention = decomposition["weights"]
            values = decomposition["values"]
            out_weight = decomposition["out_weight"]
            atoms = projected_atoms(values, out_weight)
            source_context = captured["memory_context"]
            target_l2, memory_predictions = target_dose(
                captured["read_memory"], labels, competitor, probes,
                source_attention, values, out_weight,
            )
            gradient, reconstructed = context_margin_gradient(
                model, captured, source_context, labels, competitor, dtype
            )
            source_reconstruction_error = max(
                source_reconstruction_error,
                (reconstructed - source_logits).abs().max().item(),
            )
            signed = signed_affine_solution(
                gradient, tangent_basis(values, out_weight), target_l2
            )
            full_delta = signed["delta"]
            selected_delta = (
                signed["head_delta"] * selected_mask[None, :, None]
            ).sum(dim=1)
            selected_delta = norm_match(selected_delta, target_l2)
            full_oracle_l2_error = max(
                full_oracle_l2_error,
                (full_delta.norm(dim=-1) - target_l2).abs().max().item(),
            )
            selected_oracle_l2_error = max(
                selected_oracle_l2_error,
                (selected_delta.norm(dim=-1) - target_l2).abs().max().item(),
            )
            query = captured["read_query"][:, -1].float()
            pre = captured["pre_fusion_feature"][:, -1].float()
            context_query = source_context[:, -1].float()
            router_input = (query, pre, context_query, source_attention, atoms)
            with torch.no_grad():
                signed_output = routers["signed"](*router_input)
                nonnegative_output = routers["nonnegative"](*router_input)
                residual_output = routers["residual"](*router_input)
                shuffled_output = routers["signed"](
                    query, pre, context_query.roll(1, 0),
                    source_attention.roll(1, 0), atoms.roll(1, 0),
                )
                rolled_delta = signed_output["delta"].roll(1, 0)
                permuted_coefficients = signed_output["coefficients"].roll(1, 1)
                permuted_delta, _ = routers["signed"].signed_delta(
                    torch.zeros_like(permuted_coefficients),
                    signed_output["gate"], atoms, permuted_coefficients,
                )
            primary = (
                (source_logits.argmax(dim=-1) != labels)
                & (memory_predictions == labels)
            )
            if primary.any():
                simplex_source_parts.append(source_attention[primary].cpu())
                simplex_atom_parts.append(atoms[primary].cpu())
                simplex_target_parts.append(full_delta[primary].cpu())
                simplex_batch_ids.extend(
                    (torch.where(primary)[0] + total).tolist()
                )
            deltas = {
                "source": torch.zeros_like(full_delta),
                "signed_router": signed_output["delta"],
                "nonnegative_router": nonnegative_output["delta"],
                "matched_residual_router": residual_output["delta"],
                "signed_shuffled_memory": shuffled_output["delta"],
                "signed_rolled_delta": rolled_delta,
                "signed_head_permuted": permuted_delta,
                "label_oracle_selected_subset": selected_delta,
                "label_oracle_full_signed": full_delta,
            }
            gates = {
                "signed_router": signed_output["gate"],
                "nonnegative_router": nonnegative_output["gate"],
                "matched_residual_router": residual_output["gate"],
                "signed_shuffled_memory": shuffled_output["gate"],
                "signed_rolled_delta": signed_output["gate"].roll(1, 0),
                "signed_head_permuted": signed_output["gate"],
            }
            for name in CONDITIONS:
                if name == "source":
                    logits = source_logits
                else:
                    logits = downstream(
                        model, captured,
                        patch_context(source_context, deltas[name]), dtype, False,
                    )
                append_diagnostic(
                    parts, name, logits, labels, competitor, deltas[name],
                    gates.get(name),
                )
            labels_parts.append(labels.cpu())
            confidence_parts.append(confidence.cpu())
            competitor_parts.append(competitor.cpu())
            memory_prediction_parts.append(memory_predictions.cpu())
            source_correct += int(
                (source_logits.argmax(dim=-1) == labels).sum().item()
            )
            total += batch
            if total == batch or total % args.log_every_samples == 0:
                print(
                    f"Level {LEVEL} diagnostic={total}/{args.diagnostic_samples} "
                    f"source={source_correct / total:.2%}",
                    flush=True,
                )
                save(root / "progress.json", {
                    "stage": "formal_diagnostic",
                    "samples_complete": total,
                    "samples_total": args.diagnostic_samples,
                    "source_accuracy_so_far": source_correct / total,
                })
    finally:
        trace.close()
    if not simplex_source_parts:
        raise RuntimeError("formal diagnostic produced no primary simplex examples")
    simplex_source = torch.cat(simplex_source_parts)
    simplex_atoms = torch.cat(simplex_atom_parts)
    simplex_target = torch.cat(simplex_target_parts)
    simplex_outputs = {
        name: [] for name in simplex_parts if name != "target_l2"
    }
    for start in range(0, len(simplex_target), args.simplex_batch_size):
        end = start + args.simplex_batch_size
        projection = simplex_target_projection(
            simplex_source[start:end].to(device),
            simplex_atoms[start:end].to(device),
            simplex_target[start:end].to(device),
            args.simplex_iterations,
        )
        for name in simplex_outputs:
            simplex_outputs[name].append(projection[name].cpu())
        print(
            f"Level {LEVEL} simplex={min(end, len(simplex_target))}/"
            f"{len(simplex_target)}",
            flush=True,
        )
        save(root / "progress.json", {
            "stage": "simplex_projection",
            "samples_complete": min(end, len(simplex_target)),
            "samples_total": len(simplex_target),
        })
    for name, values in simplex_outputs.items():
        output = torch.full((args.diagnostic_samples,), torch.nan)
        output[simplex_batch_ids] = torch.cat(values)
        simplex_parts[name].append(output)
    target_output = torch.full((args.diagnostic_samples,), torch.nan)
    target_output[simplex_batch_ids] = simplex_target.norm(dim=-1)
    simplex_parts["target_l2"].append(target_output)
    return {
        "labels": torch.cat(labels_parts),
        "confidence": torch.cat(confidence_parts),
        "competitor": torch.cat(competitor_parts),
        "memory_predictions": torch.cat(memory_prediction_parts),
        "conditions": {
            name: {field: torch.cat(values) for field, values in row.items()}
            for name, row in parts.items()
        },
        "simplex": {name: torch.cat(values) for name, values in simplex_parts.items()},
        "source_reconstruction_max_abs": source_reconstruction_error,
        "selected_oracle_l2_max_abs_error": selected_oracle_l2_error,
        "full_oracle_l2_max_abs_error": full_oracle_l2_error,
    }


def paired_continuous(left, right, args, seed):
    result = continuous_effect(
        (right.double() - left.double()).numpy(), args, seed
    )
    result["left_mean"] = left.double().mean().item()
    result["right_mean"] = right.double().mean().item()
    return result


def condition_metric(row, labels, ids):
    return {
        "samples": int(len(ids)),
        "accuracy": (
            row["predictions"][ids] == labels[ids]
        ).float().mean().item(),
        "fixed_margin_mean": row["fixed_margin"][ids].mean().item(),
        "decision_margin_mean": row["decision_margin"][ids].mean().item(),
        "cross_entropy_mean": row["cross_entropy"][ids].mean().item(),
        "context_delta_norm_mean": row["context_delta_norm"][ids].mean().item(),
        "gate_mean": row["gate"][ids].mean().item(),
    }


def analyze_diagnostic(collected, subset_result, training, args):
    labels = collected["labels"]
    conditions = collected["conditions"]
    source_prediction = conditions["source"]["predictions"]
    wrong = torch.where(source_prediction != labels)[0]
    correct = torch.where(source_prediction == labels)[0]
    primary = torch.where(
        (source_prediction != labels)
        & (collected["memory_predictions"] == labels)
    )[0]
    matched, matching = match_confidence(
        wrong, correct, collected["confidence"]
    )
    groups = {
        "all": torch.arange(len(labels)),
        "source_errors": wrong,
        "memory_decodable_errors": primary,
        "confidence_matched_correct": matched,
        "source_correct": correct,
    }
    if len(wrong) < args.minimum_errors:
        raise RuntimeError(
            f"diagnostic has {len(wrong)} source errors; requires {args.minimum_errors}"
        )
    if len(primary) < args.minimum_memory_decodable_errors:
        raise RuntimeError(
            f"diagnostic has {len(primary)} primary examples; requires "
            f"{args.minimum_memory_decodable_errors}"
        )
    metrics = {
        population: {
            name: condition_metric(row, labels, ids)
            for name, row in conditions.items()
        }
        for population, ids in groups.items()
    }
    effects = {}
    for population, ids in groups.items():
        source = conditions["source"]
        effects[population] = {}
        for offset, (name, row) in enumerate(conditions.items()):
            if name == "source":
                continue
            effects[population][name] = {
                "accuracy": paired_statistics(
                    source["predictions"][ids], row["predictions"][ids],
                    labels[ids], args,
                    args.analysis_seed + 1000 * offset + len(ids),
                ),
                "fixed_margin": paired_continuous(
                    source["fixed_margin"][ids], row["fixed_margin"][ids],
                    args, args.analysis_seed + 2000 * offset + len(ids),
                ),
            }
    signed_family_names = [
        "source", "nonnegative_router", "matched_residual_router",
        "signed_shuffled_memory", "signed_rolled_delta",
        "signed_head_permuted",
    ]
    signed = conditions["signed_router"]["fixed_margin"][primary]
    specificity = {}
    p_values = {}
    for offset, name in enumerate(signed_family_names):
        left = (
            conditions["source"]["fixed_margin"][primary]
            if name == "source"
            else conditions[name]["fixed_margin"][primary]
        )
        result = paired_continuous(
            left, signed, args, args.analysis_seed + 30000 + offset * 100
        )
        key = f"signed_vs_{name}"
        specificity[key] = result
        p_values[key] = result["sign_flip_p_two_sided"]
    adjusted = holm_adjust(p_values)
    for name in specificity:
        specificity[name]["multiplicity"] = adjusted[name]
    source_gain = conditions["source"]["fixed_margin"][primary]
    oracle_full_gain = (
        conditions["label_oracle_full_signed"]["fixed_margin"][primary]
        - source_gain
    ).mean().item()
    router_gain = (signed - source_gain).mean().item()
    router_recovery = router_gain / max(oracle_full_gain, 1e-12)
    simplex = collected["simplex"]
    finite_simplex = torch.isfinite(simplex["relative_error"])
    simplex_primary = finite_simplex & torch.isin(
        torch.arange(len(labels)), primary
    )
    relative = simplex["relative_error"][simplex_primary]
    feasible = relative <= args.simplex_feasibility_tolerance
    simplex_converged = (
        simplex["restart_delta_gap"][simplex_primary].max().item()
        <= args.simplex_restart_tolerance
        and simplex["projected_gradient_mapping"][simplex_primary].max().item()
        <= args.simplex_mapping_tolerance
    )
    full_accuracy = effects["all"]["signed_router"]["accuracy"]
    specificity_passed = all(
        row["estimate"] > 0
        and row["multiplicity"]["significant_0.05"]
        for row in specificity.values()
    )
    router_passed = (
        router_recovery >= args.router_recovery_threshold
        and specificity_passed
        and full_accuracy["accuracy_change"]["ci95"][0]
        >= -args.full_accuracy_noninferiority
    )
    if router_passed:
        classification = "label_free_signed_read_supported"
        next_boundary = (
            "Repeat the frozen signed router across independent initializations "
            "before opening seed909 or any protected test."
        )
    elif subset_result["selected_90"]["recovery_fraction"] >= args.subset_recovery_90:
        classification = "oracle_not_compiled_into_label_free_router"
        next_boundary = (
            "Keep the trunk frozen; diagnose router observability and supervision "
            "rather than reopening optimizer or model search."
        )
    else:
        classification = "distributed_head_mechanism_unresolved"
        next_boundary = (
            "Stop router training and refine the minimal-head causal basis."
        )
    return {
        "matching": {
            "source_errors": len(wrong),
            "memory_decodable_errors": len(primary),
            "matched_correct": len(matched),
            **matching,
        },
        "metrics": metrics,
        "effects_vs_source": effects,
        "primary": {
            "population": "Memory-decodable source errors",
            "signed_specificity": specificity,
            "full_oracle_margin_gain": oracle_full_gain,
            "signed_router_margin_gain": router_gain,
            "signed_router_oracle_recovery": router_recovery,
            "specificity_passed": specificity_passed,
        },
        "simplex_audit": {
            "samples": int(len(relative)),
            "feasibility_tolerance": args.simplex_feasibility_tolerance,
            "feasible_fraction": feasible.float().mean().item(),
            "converged": simplex_converged,
            "restart_tolerance": args.simplex_restart_tolerance,
            "mapping_tolerance": args.simplex_mapping_tolerance,
            "relative_error_mean": relative.mean().item(),
            "relative_error_median": relative.median().item(),
            "relative_error_max": relative.max().item(),
            "restart_delta_gap_max": simplex["restart_delta_gap"][
                simplex_primary
            ].max().item(),
            "projected_gradient_mapping_max": simplex[
                "projected_gradient_mapping"
            ][simplex_primary].max().item(),
        },
        "training": training,
        "diagnosis": {
            "classification": classification,
            "source_accuracy": metrics["all"]["source"]["accuracy"],
            "source_errors": len(wrong),
            "memory_decodable_errors": len(primary),
            "minimal_80_heads": subset_result["selected_80"]["heads"],
            "minimal_90_heads": subset_result["selected_90"]["heads"],
            "signed_router_passed": router_passed,
            "simplex_audit_converged": simplex_converged,
            "registered_next_boundary": next_boundary,
        },
    }, groups


def plot_result(analysis, subset_result, path):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    by_size = {}
    for row in subset_result["subsets"]:
        by_size.setdefault(row["size"], []).append(row["recovery_fraction"])
    sizes = sorted(by_size)
    axes[0].boxplot([by_size[size] for size in sizes], tick_labels=sizes)
    axes[0].axhline(0.8, color="#E15759", linestyle="--", label="80%")
    axes[0].axhline(0.9, color="#F28E2B", linestyle="--", label="90%")
    axes[0].set_xlabel("Number of read heads")
    axes[0].set_ylabel("Deployed signed-margin recovery")
    axes[0].set_title("All 255 head subsets")
    axes[0].legend()

    names = [
        "signed_router", "nonnegative_router", "matched_residual_router",
        "signed_shuffled_memory", "signed_rolled_delta",
        "signed_head_permuted", "label_oracle_selected_subset",
        "label_oracle_full_signed",
    ]
    source = analysis["metrics"]["memory_decodable_errors"]["source"][
        "fixed_margin_mean"
    ]
    gains = [
        analysis["metrics"]["memory_decodable_errors"][name][
            "fixed_margin_mean"
        ] - source for name in names
    ]
    axes[1].bar(np.arange(len(names)), gains, color="#59A14F")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(names)), [
        "Signed", "Nonneg.", "Residual", "Shuffled", "Rolled",
        "Head perm.", "Subset Oracle", "Full Oracle",
    ], rotation=30, ha="right")
    axes[1].set_ylabel("Deployed margin change")
    axes[1].set_title("Fresh diagnostic mechanism panel")

    simplex = analysis["simplex_audit"]
    axes[2].bar(
        ["Feasible", "Infeasible"],
        [simplex["feasible_fraction"], 1 - simplex["feasible_fraction"]],
        color=["#4E79A7", "#E15759"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Fraction of primary examples")
    axes[2].set_title(
        f"Exact simplex target projection\n"
        f"median residual={simplex['relative_error_median']:.3f}"
    )
    figure.suptitle(
        "IST Level 6.19.4: Minimal Heads and Gated Signed Read",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def preregistration(args):
    return {
        "level": LEVEL,
        "status": "minimal-head and label-free gated signed-read test",
        "source": "formally passed Level 6.18.3 seed707 checkpoint",
        "parent_result": "Level 6.19.3 signed_affine_simplex_obstruction",
        "frozen_parent_boundary": (
            "Head-budget reallocation is insufficient but signed affine value "
            "mixing closes the tangent gap; isolate the minimal heads and test "
            "a gated residual/signed-value read."
        ),
        "splits": {
            "subset_calibration": {
                "samples": args.subset_samples,
                "seed": args.subset_seed,
                "role": "replay of Level 6.19.3 data; all 255 subsets",
            },
            "router_train": {
                "samples": args.train_samples,
                "seed": args.train_seed,
            },
            "router_validation": {
                "samples": args.validation_samples,
                "seed": args.validation_seed,
            },
            "formal_diagnostic": {
                "samples": args.diagnostic_samples,
                "seed": args.diagnostic_seed,
                "opened_once_after_router_selection": True,
            },
        },
        "subset_rule": {
            "population": "Memory-decodable source errors",
            "quantity": (
                "equal-L2 actual deployed-margin gain after frozen nonlinear "
                "downstream replay of masked per-head components; first-order "
                "gain is secondary"
            ),
            "enumeration": "all 255 non-empty subsets of eight read heads",
            "thresholds": [args.subset_recovery_80, args.subset_recovery_90],
            "selection": (
                "smallest subset meeting threshold; then maximum recovery; "
                "then lexicographic head order"
            ),
            "router_mask": "the selected 90% subset",
        },
        "simplex_audit": {
            "target": "the full equal-L2 signed-affine context delta",
            "solver": (
                "two-start projected gradient for the convex least-squares "
                "projection onto the product of eight 32-slot simplices"
            ),
            "iterations": args.simplex_iterations,
            "feasible_relative_residual": args.simplex_feasibility_tolerance,
            "convergence_gates": {
                "maximum_relative_two_start_delta_gap": (
                    args.simplex_restart_tolerance
                ),
                "maximum_projected_gradient_mapping": (
                    args.simplex_mapping_tolerance
                ),
            },
            "interpretation": (
                "tests exact target representability, not every possible "
                "equal-dose simplex direction"
            ),
        },
        "routers": {
            "common": (
                "same input-conditioned slot scorer, head mask, parameter "
                "count, training split, optimizer, and global dose cap"
            ),
            "signed": "zero-mean per-head signed projected-value coefficients",
            "nonnegative": "valid softmax attention followed by source interpolation",
            "residual": "matched-parameter fixed-basis unrestricted residual",
            "inference_uses_true_label": False,
            "training_uses_task_labels": True,
            "training_objective": (
                "task cross-entropy plus fixed-weight distillation toward the "
                "selected-subset signed Oracle context delta"
            ),
            "distillation_weight": args.distillation_weight,
            "dose_cap": DOSE_CAP,
        },
        "registered_signed_family": [
            "signed versus source",
            "signed versus nonnegative",
            "signed versus matched residual",
            "signed versus shuffled memory",
            "signed versus rolled delta",
            "signed versus head-permuted coefficients",
        ],
        "decision_rule": {
            "router_recovery_threshold": args.router_recovery_threshold,
            "specificity": "all six positive after Holm correction at 0.05",
            "full_accuracy_noninferiority": args.full_accuracy_noninferiority,
            "success": (
                "signed router reaches recovery threshold, passes specificity, "
                "and the lower full-accuracy CI exceeds the noninferiority bound"
            ),
        },
        "locks": {
            "seed707_trunk_and_existing_probes_frozen": True,
            "failed_Level_6_18_9_candidate_not_used": True,
            "protected_tests_not_used": True,
            "seed909_locked": True,
            "optimizer_search_closed": True,
            "one_fixed_router_initialization": True,
        },
        "numerical_revision": NUMERICAL_REVISION,
        "protocol": {
            key: value for key, value in vars(args).items()
            if key not in {"dry_run", "smoke_test", "force"}
        },
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError(f"Level {LEVEL} is fixed to seed707 at 16 chunks")
    for path in (args.checkpoint, args.probes, args.parent_result):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if min(
        args.subset_samples, args.train_samples, args.validation_samples,
        args.diagnostic_samples, args.eval_batch_size, args.router_batch_size,
    ) <= 0:
        raise ValueError("all sample and batch sizes must be positive")
    for value in (
        args.subset_samples, args.train_samples, args.validation_samples,
        args.diagnostic_samples,
    ):
        if value % args.eval_batch_size != 0:
            raise ValueError("every split size must be divisible by eval-batch-size")
    if not args.smoke_test and (
        args.subset_samples != 4096
        or args.train_samples != 2048
        or args.validation_samples != 512
        or args.diagnostic_samples != 4096
        or args.subset_seed != SUBSET_SEED
        or args.train_seed != TRAIN_SEED
        or args.validation_seed != VALIDATION_SEED
        or args.diagnostic_seed != DIAGNOSTIC_SEED
    ):
        raise ValueError(
            f"Formal Level {LEVEL} split sizes and seeds are fixed; use "
            "--smoke-test for implementation checks"
        )
    parent = json.loads(Path(args.parent_result).read_text(encoding="utf-8"))
    if (
        not parent.get("integrity", {}).get("passed")
        or parent.get("analysis", {}).get("diagnosis", {}).get("classification")
        != "signed_affine_simplex_obstruction"
    ):
        raise RuntimeError("Level 6.19.3 parent result is not formally passed")


def serializable_predictions(collected, groups):
    return {
        "labels": collected["labels"].tolist(),
        "confidence": collected["confidence"].tolist(),
        "competitor": collected["competitor"].tolist(),
        "memory_predictions": collected["memory_predictions"].tolist(),
        "groups": {name: value.tolist() for name, value in groups.items()},
        "conditions": {
            name: {field: value.tolist() for field, value in row.items()}
            for name, row in collected["conditions"].items()
        },
        "simplex": {
            name: value.tolist() for name, value in collected["simplex"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.19.4 minimal heads and gated signed read"
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
    parser.add_argument(
        "--parent-result", default="experiments/level6_19_3/formal/result.json"
    )
    parser.add_argument("--subset-samples", type=int, default=4096)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--diagnostic-samples", type=int, default=4096)
    parser.add_argument("--subset-seed", type=int, default=SUBSET_SEED)
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--validation-seed", type=int, default=VALIDATION_SEED)
    parser.add_argument("--diagnostic-seed", type=int, default=DIAGNOSTIC_SEED)
    parser.add_argument("--analysis-seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--router-seed", type=int, default=ROUTER_SEED)
    parser.add_argument("--residual-basis-seed", type=int, default=6194600)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--router-batch-size", type=int, default=64)
    parser.add_argument("--router-hidden", type=int, default=32)
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--router-weight-decay", type=float, default=1e-4)
    parser.add_argument("--distillation-weight", type=float, default=0.25)
    parser.add_argument("--gate-penalty", type=float, default=1e-4)
    parser.add_argument("--minimum-loss-improvement", type=float, default=1e-5)
    parser.add_argument("--subset-recovery-80", type=float, default=0.80)
    parser.add_argument("--subset-recovery-90", type=float, default=0.90)
    parser.add_argument("--router-recovery-threshold", type=float, default=0.25)
    parser.add_argument("--full-accuracy-noninferiority", type=float, default=0.0025)
    # The first formal run showed that 2,048 accelerated iterations left two
    # long-tail examples just outside the pre-registered numerical audit.  The
    # scientific target and tolerance remain unchanged; only the deterministic
    # convex solver budget is increased for the explicitly marked recovery run.
    parser.add_argument("--simplex-iterations", type=int, default=8192)
    parser.add_argument("--simplex-batch-size", type=int, default=64)
    parser.add_argument("--simplex-feasibility-tolerance", type=float, default=0.01)
    parser.add_argument("--simplex-restart-tolerance", type=float, default=0.01)
    parser.add_argument("--simplex-mapping-tolerance", type=float, default=1e-5)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--minimum-errors", type=int, default=200)
    parser.add_argument("--minimum-memory-decodable-errors", type=int, default=150)
    parser.add_argument("--minimum-subset-primary", type=int, default=150)
    parser.add_argument("--log-every-samples", type=int, default=256)
    parser.add_argument(
        "--output", default="experiments/level6_19_4/formal_recovery"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.subset_samples = min(args.subset_samples, 64)
        args.train_samples = min(args.train_samples, 64)
        args.validation_samples = min(args.validation_samples, 32)
        args.diagnostic_samples = min(args.diagnostic_samples, 64)
        args.router_epochs = min(args.router_epochs, 2)
        args.patience = min(args.patience, 2)
        args.bootstrap_iterations = min(args.bootstrap_iterations, 100)
        args.sign_flip_iterations = min(args.sign_flip_iterations, 100)
        args.minimum_errors = 1
        args.minimum_memory_decodable_errors = 1
        args.minimum_subset_primary = 1
        args.subset_seed += 50_000_000
        args.train_seed += 50_000_000
        args.validation_seed += 50_000_000
        args.diagnostic_seed += 50_000_000
        args.analysis_seed += 50_000_000
        args.router_seed += 50_000_000
        if args.output == "experiments/level6_19_4/formal_recovery":
            args.output = "experiments/level6_19_4/smoke"
    validate(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    protocol = preregistration(args)
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps(result["analysis"]["diagnosis"], indent=2))
        return
    save(root / "preregistration.json", protocol)

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

    subset_result = calibrate_subsets(
        model, probes, args, device, dtype, root
    )
    save(root / "subset_calibration.json", subset_result)
    selected_heads = subset_result["selected_90"]["heads"]
    selected_mask = torch.zeros(HEADS, device=device)
    selected_mask[selected_heads] = 1.0

    train_cache, train_replay = collect_router_cache(
        model, probes, selected_mask, args, args.train_samples, args.train_seed,
        device, dtype, root, "train",
    )
    validation_cache, validation_replay = collect_router_cache(
        model, probes, selected_mask, args, args.validation_samples,
        args.validation_seed,
        device, dtype, root, "validation",
    )
    routers = {}
    training = {}
    for offset, kind in enumerate(ROUTER_KINDS):
        router, row = train_router(
            model, kind, selected_mask, train_cache, validation_cache,
            args, device, dtype, args.router_seed + offset * 1000,
        )
        routers[kind] = router
        training[kind] = row
        save(root / "progress.json", {
            "stage": "router_training",
            "completed_kinds": list(routers),
        })
    parameter_counts = {row["parameters"] for row in training.values()}
    training["audit"] = {
        "equal_parameter_counts": len(parameter_counts) == 1,
        "parameter_count": next(iter(parameter_counts)),
        "train_query_replay_max_abs": train_replay,
        "validation_query_replay_max_abs": validation_replay,
        "selected_heads": selected_heads,
        "inference_uses_true_label": False,
    }
    save(root / "router_training.json", training)
    torch.save({
        "level": LEVEL,
        "selected_heads": selected_heads,
        "dose_cap": DOSE_CAP,
        "router_hidden": args.router_hidden,
        "residual_basis_seed": args.residual_basis_seed,
        "states": {kind: router.state_dict() for kind, router in routers.items()},
        "training": training,
    }, root / "router_checkpoint.pt")

    collected = diagnostic_evaluation(
        model, probes, routers, selected_mask, args, device, dtype, root
    )
    analysis, groups = analyze_diagnostic(
        collected, subset_result, training, args
    )
    after = {
        "model": tensor_fingerprint(model),
        "original_probe": tensor_fingerprint(original_probe),
        **{
            f"level6_19_{name}": tensor_fingerprint(row["probe"])
            for name, row in probes.items()
        },
    }
    router_parameters_equal = training["audit"]["equal_parameter_counts"]
    split_seeds = [
        args.subset_seed, args.train_seed, args.validation_seed,
        args.diagnostic_seed,
    ]
    integrity = {
        "source_downstream_max_abs": collected[
            "source_reconstruction_max_abs"
        ],
        "source_downstream_reconstruction_exact": (
            collected["source_reconstruction_max_abs"] == 0.0
        ),
        "selected_oracle_l2_max_abs_error": collected[
            "selected_oracle_l2_max_abs_error"
        ],
        "full_oracle_l2_max_abs_error": collected[
            "full_oracle_l2_max_abs_error"
        ],
        "oracle_l2_passed": max(
            collected["selected_oracle_l2_max_abs_error"],
            collected["full_oracle_l2_max_abs_error"],
        ) <= 1e-5,
        "frozen_states_unchanged": before == after,
        "frozen_parameters_remain_frozen": all(
            not parameter.requires_grad
            for module in [model, original_probe] + [
                row["probe"] for row in probes.values()
            ]
            for parameter in module.parameters()
        ),
        "router_parameter_counts_equal": router_parameters_equal,
        "simplex_audit_converged": analysis["simplex_audit"]["converged"],
        "split_seeds_unique_except_registered_parent_replay": (
            len(set(split_seeds)) == len(split_seeds)
        ),
        "diagnostic_not_used_for_training_or_selection": True,
        "failed_candidate_not_used": True,
        "protected_tests_not_used": True,
        "seed909_locked": True,
        "optimizer_search_closed": True,
    }
    integrity["passed"] = all([
        integrity["source_downstream_reconstruction_exact"],
        integrity["oracle_l2_passed"],
        integrity["frozen_states_unchanged"],
        integrity["frozen_parameters_remain_frozen"],
        integrity["router_parameter_counts_equal"],
        integrity["simplex_audit_converged"],
        integrity["split_seeds_unique_except_registered_parent_replay"],
    ])
    analysis["diagnosis"]["integrity_passed"] = integrity["passed"]
    if not integrity["passed"]:
        analysis["diagnosis"]["classification"] = "integrity_failure"
        analysis["diagnosis"]["registered_next_boundary"] = (
            f"Stop; repair the Level {LEVEL} implementation."
        )
    result = {
        "protocol": protocol,
        "checkpoint_meta": checkpoint_meta,
        "integrity": integrity,
        "subset_calibration": subset_result,
        "analysis": analysis,
    }
    save(result_path, result)
    save(root / "summary.json", {
        "integrity": integrity,
        "subset_selection": {
            "selected_80": subset_result["selected_80"],
            "selected_90": subset_result["selected_90"],
        },
        "matching": analysis["matching"],
        "primary": analysis["primary"],
        "simplex_audit": analysis["simplex_audit"],
        "training": training,
        "diagnosis": analysis["diagnosis"],
    })
    save(root / "predictions.json", serializable_predictions(collected, groups))
    plot_result(analysis, subset_result, root / "minimal_heads_gated_read.png")
    save(root / "progress.json", {
        "stage": "complete",
        "integrity_passed": integrity["passed"],
        "classification": analysis["diagnosis"]["classification"],
        "signed_router_passed": analysis["diagnosis"]["signed_router_passed"],
    })
    print(json.dumps(analysis["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
