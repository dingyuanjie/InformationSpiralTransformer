import argparse
import copy
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
from run_level6_18_6_local import configure_cuda, paired_statistics, save
from run_level6_19_local import FinalTrace
from run_level6_19_1_local import load_frozen, norm_match, tensor_fingerprint
from run_level6_19_2_local import (
    HEADS,
    SLOTS,
    WIDTH,
    attention_decomposition,
    task_competitor,
)
from run_level6_19_3_local import signed_affine_solution, tangent_basis
from run_level6_19_4_local import (
    GatedReadRouter,
    context_margin_gradient,
    parameter_fingerprint,
    projected_atoms,
    query_downstream,
    target_dose,
)


LEVEL = "6.19.5"
SEED = 707
CHUNKS = 16
TRAIN_SEED = 6195100
VALIDATION_SEED = 6195200
DIAGNOSTIC_SEED = 6195300
ANALYSIS_SEED = 6195400
PROBE_SEED = 6195500
RESIDUAL_BASIS_SEED = 6195600
RECOVERY_THRESHOLD = 0.25
DOSE_PREDICTION_CAP = 8.0

CONDITIONS = [
    "source",
    "frozen_learned_dose_learned_direction",
    "oracle_dose_learned_direction",
    "learned_dose_oracle_direction",
    "oracle_dose_oracle_direction",
    "probed_dose_oracle_direction",
    "oracle_dose_signed_distilled_direction",
    "oracle_dose_residual_distilled_direction",
]


def trainable_fingerprint(module):
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class ObservableScalarProbe(nn.Module):
    """Fixed small probe over exactly the Level 6.19.4 router observables."""

    def __init__(self, hidden):
        super().__init__()
        self.global_net = nn.Sequential(
            nn.Linear(WIDTH * 3, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.atom_net = nn.Linear(WIDTH, hidden, bias=False)
        self.attention_net = nn.Linear(2, hidden, bias=False)
        self.head_embedding = nn.Parameter(torch.zeros(HEADS, hidden))
        self.readout = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.head_embedding, std=0.02)

    def forward(self, query, pre_fusion, source_context, source_attention,
                atoms):
        global_hidden = self.global_net(torch.cat([
            query, pre_fusion, source_context
        ], dim=-1))
        atom_unit = atoms / atoms.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        attention_features = torch.stack([
            source_attention,
            source_attention.clamp_min(1e-8).log() / 8.0,
        ], dim=-1)
        slot_hidden = F.gelu(
            self.atom_net(atom_unit)
            + self.attention_net(attention_features)
            + global_hidden[:, None, None]
            + self.head_embedding[None, :, None]
        )
        pooled_mean = slot_hidden.mean(dim=(1, 2))
        pooled_max = slot_hidden.amax(dim=(1, 2))
        return self.readout(torch.cat([
            global_hidden, pooled_mean, pooled_max
        ], dim=-1)).squeeze(-1)


class DirectionDistiller(nn.Module):
    """Equal-parameter signed-value or fixed-residual direction probe."""

    def __init__(self, kind, hidden, head_mask, basis_seed):
        super().__init__()
        if kind not in {"signed", "residual"}:
            raise ValueError(kind)
        self.kind = kind
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
        self.register_buffer(
            "head_mask", head_mask.float().reshape(1, HEADS, 1)
        )
        generator = torch.Generator().manual_seed(basis_seed)
        residual = torch.randn(HEADS, SLOTS, WIDTH, generator=generator)
        residual = residual - residual.mean(dim=1, keepdim=True)
        residual = residual / residual.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        self.register_buffer("residual_basis", residual)
        nn.init.normal_(self.head_embedding, std=0.02)
        nn.init.normal_(self.slot_bias, std=0.01)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.zeros_(self.score.bias)

    def forward(self, query, pre_fusion, source_context, source_attention,
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
        coefficients = scores - scores.mean(dim=-1, keepdim=True)
        coefficients = coefficients * self.head_mask
        if self.kind == "signed":
            direction = torch.einsum("bhs,bhsd->bd", coefficients, atoms)
        else:
            direction = torch.einsum(
                "bhs,hsd->bd", coefficients, self.residual_basis
            )
        unit = direction / direction.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return {"unit": unit, "coefficients": coefficients}


def router_inputs(batch):
    return (
        batch["query"],
        batch["pre_fusion"],
        batch["source_context"],
        batch["source_attention"],
        batch["atoms"],
    )


def cache_batch(cache, ids, device):
    integer = {"labels", "competitor", "memory_predictions"}
    return {
        key: value[ids].to(
            device=device,
            dtype=torch.long if key in integer else torch.float32,
        )
        for key, value in cache.items()
    }


def load_parent_routers(args, device):
    result = json.loads(Path(args.parent_result).read_text(encoding="utf-8"))
    if (
        not result.get("integrity", {}).get("passed")
        or result.get("analysis", {}).get("diagnosis", {}).get(
            "classification"
        ) != "oracle_not_compiled_into_label_free_router"
    ):
        raise RuntimeError("Level 6.19.4 canonical parent did not pass")
    saved = torch.load(
        args.parent_routers, map_location="cpu", weights_only=False
    )
    selected_heads = saved["selected_heads"]
    expected_heads = result["subset_calibration"]["selected_90"]["heads"]
    if selected_heads != expected_heads:
        raise RuntimeError("parent selected-head mask does not match result")
    mask = torch.zeros(HEADS, device=device)
    mask[selected_heads] = 1.0
    routers = {}
    for kind in ("signed", "nonnegative", "residual"):
        router = GatedReadRouter(
            kind,
            saved["router_hidden"],
            mask,
            saved["dose_cap"],
            saved["residual_basis_seed"],
        ).to(device)
        router.load_state_dict(saved["states"][kind])
        router.eval()
        for parameter in router.parameters():
            parameter.requires_grad_(False)
        routers[kind] = router
    return routers, mask, saved, result


def collect_cache(model, probes, args, samples, seed, device, dtype, root,
                  split):
    fields = [
        "query", "pre_fusion", "source_context", "source_attention", "atoms",
        "source_logits", "labels", "competitor", "memory_predictions",
        "oracle_delta", "oracle_dose", "gradient", "primary",
    ]
    parts = {field: [] for field in fields}
    trace = FinalTrace(model)
    total = 0
    replay_error = 0.0
    oracle_l2_error = 0.0
    primary_count = 0
    set_seed(seed)
    try:
        while total < samples:
            size = min(args.eval_batch_size, samples - total)
            chunks, labels, _ = make_chunks(
                size, CHUNKS, args.chunk_size, device
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
            decomposition = attention_decomposition(model, captured, dtype)
            atoms = projected_atoms(
                decomposition["values"], decomposition["out_weight"]
            )
            dose, memory_predictions = target_dose(
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
                dose,
            )
            oracle_delta = signed["delta"]
            oracle_l2_error = max(
                oracle_l2_error,
                (oracle_delta.norm(dim=-1) - dose).abs().max().item(),
            )
            query = captured["read_query"][:, -1].float()
            pre_fusion = captured["pre_fusion_feature"][:, -1].float()
            source_context = captured["memory_context"][:, -1].float()
            with torch.no_grad():
                replay = query_downstream(
                    model, query, pre_fusion, source_context, dtype
                )
            replay_error = max(
                replay_error, (replay - source_logits).abs().max().item()
            )
            primary = (
                (source_logits.argmax(dim=-1) != labels)
                & (memory_predictions == labels)
            )
            primary_count += int(primary.sum().item())
            rows = {
                "query": query,
                "pre_fusion": pre_fusion,
                "source_context": source_context,
                "source_attention": decomposition["weights"],
                "atoms": atoms,
                "source_logits": source_logits,
                "labels": labels,
                "competitor": competitor,
                "memory_predictions": memory_predictions,
                "oracle_delta": oracle_delta,
                "oracle_dose": dose,
                "gradient": gradient,
                "primary": primary.float(),
            }
            for name, value in rows.items():
                if name in {"labels", "competitor", "memory_predictions"}:
                    value = value.long()
                else:
                    value = value.float()
                parts[name].append(value.detach().cpu())
            total += size
            if total == size or total % args.log_every_samples == 0:
                print(
                    f"Level {LEVEL} cache {split}={total}/{samples} "
                    f"primary={primary_count}",
                    flush=True,
                )
                save(root / "progress.json", {
                    "stage": f"cache_{split}",
                    "samples_complete": total,
                    "samples_total": samples,
                    "primary_examples": primary_count,
                })
    finally:
        trace.close()
    return {
        "cache": {name: torch.cat(values) for name, values in parts.items()},
        "source_replay_max_abs": replay_error,
        "oracle_l2_max_abs_error": oracle_l2_error,
        "primary_examples": primary_count,
    }


def scalar_prediction(probe, cache, args, device):
    probe.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(cache["labels"]), args.probe_batch_size):
            batch = cache_batch(
                cache, slice(start, start + args.probe_batch_size), device
            )
            output.append(probe(*router_inputs(batch)).cpu())
    return torch.cat(output)


def train_scalar_probe(kind, train_cache, validation_cache, args, device,
                       seed, dose_stats=None):
    set_seed(seed)
    probe = ObservableScalarProbe(args.probe_hidden).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    order_generator = torch.Generator().manual_seed(seed + 1)
    history = []
    target = train_cache["primary"] if kind == "classifier" else (
        torch.log1p(train_cache["oracle_dose"]) - dose_stats["mean"]
    ) / dose_stats["std"]
    positives = train_cache["primary"].sum().item()
    negatives = len(target) - positives
    positive_weight = max(negatives / max(positives, 1.0), 1.0)
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        order = torch.randperm(len(target), generator=order_generator)
        losses = []
        for start in range(0, len(order), args.probe_batch_size):
            ids = order[start:start + args.probe_batch_size]
            batch = cache_batch(train_cache, ids, device)
            prediction = probe(*router_inputs(batch))
            batch_target = target[ids].to(device)
            if kind == "classifier":
                loss = F.binary_cross_entropy_with_logits(
                    prediction, batch_target,
                    pos_weight=torch.tensor(positive_weight, device=device),
                )
            else:
                loss = F.mse_loss(prediction, batch_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach().item())
        validation_raw = scalar_prediction(
            probe, validation_cache, args, device
        )
        if kind == "classifier":
            validation_loss = F.binary_cross_entropy_with_logits(
                validation_raw,
                validation_cache["primary"],
                pos_weight=torch.tensor(positive_weight),
            ).item()
        else:
            validation_target = (
                torch.log1p(validation_cache["oracle_dose"])
                - dose_stats["mean"]
            ) / dose_stats["std"]
            validation_loss = F.mse_loss(
                validation_raw, validation_target
            ).item()
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
        })
        print(
            f"Level {LEVEL} probe={kind} epoch={epoch} "
            f"train={history[-1]['train_loss']:.5f} "
            f"val={validation_loss:.5f}",
            flush=True,
        )
    return probe, {
        "kind": kind,
        "parameters": sum(p.numel() for p in probe.parameters()),
        "epochs": args.probe_epochs,
        "fixed_final_epoch_used": True,
        "positive_weight": positive_weight if kind == "classifier" else None,
        "history": history,
    }


def direction_prediction(probe, cache, args, device):
    probe.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(cache["labels"]), args.probe_batch_size):
            batch = cache_batch(
                cache, slice(start, start + args.probe_batch_size), device
            )
            output.append(probe(*router_inputs(batch))["unit"].cpu())
    return torch.cat(output)


def train_direction_probe(kind, head_mask, train_cache, validation_cache,
                          args, device, seed):
    set_seed(seed)
    probe = DirectionDistiller(
        kind, args.probe_hidden, head_mask, args.residual_basis_seed
    ).to(device)
    initial_trainable_fingerprint = trainable_fingerprint(probe)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    order_generator = torch.Generator().manual_seed(seed + 1)
    target_unit = train_cache["oracle_delta"] / train_cache[
        "oracle_delta"
    ].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    history = []
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        order = torch.randperm(len(target_unit), generator=order_generator)
        losses = []
        for start in range(0, len(order), args.probe_batch_size):
            ids = order[start:start + args.probe_batch_size]
            batch = cache_batch(train_cache, ids, device)
            prediction = probe(*router_inputs(batch))["unit"]
            loss = 1.0 - (
                prediction * target_unit[ids].to(device)
            ).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach().item())
        validation_unit = direction_prediction(
            probe, validation_cache, args, device
        )
        validation_target = validation_cache["oracle_delta"]
        validation_target = validation_target / validation_target.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        validation_loss = 1.0 - (
            validation_unit * validation_target
        ).sum(dim=-1).mean().item()
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
        })
        print(
            f"Level {LEVEL} probe={kind}_direction epoch={epoch} "
            f"train={history[-1]['train_loss']:.5f} "
            f"val={validation_loss:.5f}",
            flush=True,
        )
    return probe, {
        "kind": f"{kind}_direction",
        "parameters": sum(p.numel() for p in probe.parameters()),
        "epochs": args.probe_epochs,
        "fixed_final_epoch_used": True,
        "initial_trainable_fingerprint": initial_trainable_fingerprint,
        "history": history,
    }


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def binary_observability_metrics(logits, targets):
    scores = logits.detach().double().cpu().numpy()
    labels = targets.detach().bool().cpu().numpy()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("classifier metric requires both classes")
    ranks = average_ranks(scores)
    auroc = (
        ranks[labels].sum() - positives * (positives + 1) / 2
    ) / (positives * negatives)
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order].astype(np.float64)
    precision = np.cumsum(ordered) / np.arange(1, len(labels) + 1)
    average_precision = float((precision * ordered).sum() / positives)
    top = labels[order[:positives]].mean()
    prevalence = positives / len(labels)
    return {
        "samples": len(labels),
        "positives": positives,
        "prevalence": prevalence,
        "auroc": float(auroc),
        "average_precision": average_precision,
        "fixed_prevalence_top_k": positives,
        "fixed_prevalence_precision": float(top),
        "fixed_prevalence_lift": float(top / prevalence),
    }


def dose_observability_metrics(prediction, target):
    prediction = prediction.detach().double().cpu().numpy()
    target = target.detach().double().cpu().numpy()
    residual = ((target - prediction) ** 2).sum()
    total = ((target - target.mean()) ** 2).sum()
    r2 = 1.0 - residual / max(total, 1e-12)
    pred_rank = average_ranks(prediction)
    target_rank = average_ranks(target)
    spearman = np.corrcoef(pred_rank, target_rank)[0, 1]
    design = np.stack([np.ones_like(prediction), prediction], axis=1)
    intercept, slope = np.linalg.lstsq(design, target, rcond=None)[0]
    return {
        "samples": len(target),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
        "mae": float(np.abs(target - prediction).mean()),
        "rmse": float(np.sqrt(((target - prediction) ** 2).mean())),
        "r2": float(r2),
        "spearman": float(spearman),
        "calibration_intercept_target_on_prediction": float(intercept),
        "calibration_slope_target_on_prediction": float(slope),
    }


def condition_row(logits, labels, competitor, delta, primary):
    rows = torch.arange(len(labels), device=labels.device)
    predictions = logits.argmax(dim=-1)
    fixed_margin = logits[rows, labels] - logits[rows, competitor]
    primary_ids = torch.where(primary)[0]
    return {
        "predictions": predictions.detach().cpu(),
        "fixed_margin": fixed_margin.detach().cpu(),
        "context_delta_norm": delta.norm(dim=-1).detach().cpu(),
        "primary_fixed_margin": fixed_margin[primary_ids].detach().cpu(),
        "primary_predictions": predictions[primary_ids].detach().cpu(),
    }


def evaluate_formal(model, routers, probes, diagnostic_cache, dose_stats,
                    args, device, dtype, root):
    cache = diagnostic_cache
    condition_parts = {
        name: {
            field: [] for field in [
                "predictions", "fixed_margin", "context_delta_norm",
                "primary_fixed_margin", "primary_predictions",
            ]
        }
        for name in CONDITIONS
    }
    classifier_logits_parts = []
    dose_prediction_parts = []
    signed_unit_parts = []
    residual_unit_parts = []
    learned_dose_parts = []
    labels_parts = []
    competitor_parts = []
    primary_parts = []
    oracle_dose_parts = []
    oracle_unit_parts = []
    gradient_parts = []
    for start in range(0, len(cache["labels"]), args.probe_batch_size):
        end = min(start + args.probe_batch_size, len(cache["labels"]))
        batch = cache_batch(cache, slice(start, end), device)
        labels = batch["labels"]
        competitor = batch["competitor"]
        primary = batch["primary"].bool()
        oracle_delta = batch["oracle_delta"]
        oracle_dose = oracle_delta.norm(dim=-1)
        oracle_unit = oracle_delta / oracle_dose[:, None].clamp_min(1e-8)
        with torch.no_grad():
            frozen = routers["signed"](*router_inputs(batch))
            learned_delta = frozen["delta"]
            learned_dose = learned_delta.norm(dim=-1)
            learned_unit = learned_delta / learned_dose[
                :, None
            ].clamp_min(1e-8)
            classifier_logits = probes["classifier"](*router_inputs(batch))
            dose_raw = probes["dose"](*router_inputs(batch))
            dose_prediction = torch.expm1(
                dose_raw * dose_stats["std"] + dose_stats["mean"]
            ).clamp(min=0.0, max=args.dose_prediction_cap)
            signed_unit = probes["signed_direction"](
                *router_inputs(batch)
            )["unit"]
            residual_unit = probes["residual_direction"](
                *router_inputs(batch)
            )["unit"]
        deltas = {
            "source": torch.zeros_like(oracle_delta),
            "frozen_learned_dose_learned_direction": learned_delta,
            "oracle_dose_learned_direction": (
                learned_unit * oracle_dose[:, None]
            ),
            "learned_dose_oracle_direction": (
                oracle_unit * learned_dose[:, None]
            ),
            "oracle_dose_oracle_direction": oracle_delta,
            "probed_dose_oracle_direction": (
                oracle_unit * dose_prediction[:, None]
            ),
            "oracle_dose_signed_distilled_direction": (
                signed_unit * oracle_dose[:, None]
            ),
            "oracle_dose_residual_distilled_direction": (
                residual_unit * oracle_dose[:, None]
            ),
        }
        source_replay = query_downstream(
            model, batch["query"], batch["pre_fusion"],
            batch["source_context"], dtype,
        )
        for name, delta in deltas.items():
            if name == "source":
                logits = batch["source_logits"]
            else:
                updated = query_downstream(
                    model, batch["query"], batch["pre_fusion"],
                    batch["source_context"] + delta, dtype,
                )
                logits = batch["source_logits"] + updated - source_replay
            row = condition_row(
                logits, labels, competitor, delta, primary
            )
            for field, value in row.items():
                condition_parts[name][field].append(value)
        classifier_logits_parts.append(classifier_logits.cpu())
        dose_prediction_parts.append(dose_prediction.cpu())
        signed_unit_parts.append(signed_unit.cpu())
        residual_unit_parts.append(residual_unit.cpu())
        learned_dose_parts.append(learned_dose.cpu())
        labels_parts.append(labels.cpu())
        competitor_parts.append(competitor.cpu())
        primary_parts.append(primary.cpu())
        oracle_dose_parts.append(oracle_dose.cpu())
        oracle_unit_parts.append(oracle_unit.cpu())
        gradient_parts.append(batch["gradient"].cpu())
        if end == args.probe_batch_size or end % args.log_every_samples == 0:
            print(
                f"Level {LEVEL} formal replay={end}/{len(cache['labels'])}",
                flush=True,
            )
            save(root / "progress.json", {
                "stage": "formal_replay",
                "samples_complete": end,
                "samples_total": len(cache["labels"]),
            })
    return {
        "labels": torch.cat(labels_parts),
        "competitor": torch.cat(competitor_parts),
        "primary": torch.cat(primary_parts),
        "oracle_dose": torch.cat(oracle_dose_parts),
        "oracle_unit": torch.cat(oracle_unit_parts),
        "gradient": torch.cat(gradient_parts),
        "classifier_logits": torch.cat(classifier_logits_parts),
        "dose_prediction": torch.cat(dose_prediction_parts),
        "learned_dose": torch.cat(learned_dose_parts),
        "signed_distilled_unit": torch.cat(signed_unit_parts),
        "residual_distilled_unit": torch.cat(residual_unit_parts),
        "conditions": {
            name: {
                field: torch.cat(values) for field, values in row.items()
            }
            for name, row in condition_parts.items()
        },
    }


def summarize_condition(row, labels, primary):
    ids = torch.where(primary)[0]
    primary_labels = labels[ids]
    return {
        "full_accuracy": (
            row["predictions"] == labels
        ).float().mean().item(),
        "primary_accuracy": (
            row["primary_predictions"] == primary_labels
        ).float().mean().item(),
        "full_fixed_margin_mean": row["fixed_margin"].mean().item(),
        "primary_fixed_margin_mean": row[
            "primary_fixed_margin"
        ].mean().item(),
        "full_context_l2_mean": row["context_delta_norm"].mean().item(),
        "primary_context_l2_mean": row[
            "context_delta_norm"
        ][ids].mean().item(),
    }


def direction_metrics(unit, oracle_unit, gradient, primary):
    ids = torch.where(primary)[0]
    cosine = (unit * oracle_unit).sum(dim=-1)
    alignment = (unit * gradient).sum(dim=-1)
    oracle_alignment = (oracle_unit * gradient).sum(dim=-1)
    ratio = alignment / oracle_alignment.clamp_min(1e-8)
    return {
        "all_cosine_mean": cosine.mean().item(),
        "primary_cosine_mean": cosine[ids].mean().item(),
        "primary_first_order_alignment_mean": alignment[ids].mean().item(),
        "primary_oracle_alignment_mean": oracle_alignment[ids].mean().item(),
        "primary_first_order_alignment_ratio_mean": ratio[ids].mean().item(),
        "primary_positive_alignment_fraction": (
            alignment[ids] > 0
        ).float().mean().item(),
    }


def analyze_formal(collected, args):
    labels = collected["labels"]
    primary = collected["primary"]
    source = collected["conditions"]["source"]
    source_wrong = source["predictions"] != labels
    metrics = {
        name: summarize_condition(row, labels, primary)
        for name, row in collected["conditions"].items()
    }
    source_primary_margin = metrics["source"]["primary_fixed_margin_mean"]
    full_gain = (
        metrics["oracle_dose_oracle_direction"]["primary_fixed_margin_mean"]
        - source_primary_margin
    )
    effects = {}
    for offset, name in enumerate(CONDITIONS):
        if name == "source":
            continue
        row = collected["conditions"][name]
        gain = metrics[name]["primary_fixed_margin_mean"] - source_primary_margin
        effects[name] = {
            "primary_margin_gain": gain,
            "full_oracle_recovery": gain / max(full_gain, 1e-12),
            "reaches_25_percent": gain / max(full_gain, 1e-12) >= (
                args.recovery_threshold
            ),
            "full_accuracy": paired_statistics(
                source["predictions"], row["predictions"], labels, args,
                args.analysis_seed + offset * 100,
            ),
        }
    classifier = binary_observability_metrics(
        collected["classifier_logits"], primary
    )
    dose = dose_observability_metrics(
        collected["dose_prediction"], collected["oracle_dose"]
    )
    signed_direction = direction_metrics(
        collected["signed_distilled_unit"], collected["oracle_unit"],
        collected["gradient"], primary,
    )
    residual_direction = direction_metrics(
        collected["residual_distilled_unit"], collected["oracle_unit"],
        collected["gradient"], primary,
    )
    oracle_dose_learned = effects[
        "oracle_dose_learned_direction"
    ]["reaches_25_percent"]
    learned_dose_oracle = effects[
        "learned_dose_oracle_direction"
    ]["reaches_25_percent"]
    if oracle_dose_learned and not learned_dose_oracle:
        classification = "dose_gating_dominant_bottleneck"
    elif learned_dose_oracle and not oracle_dose_learned:
        classification = "signed_direction_dominant_bottleneck"
    elif oracle_dose_learned and learned_dose_oracle:
        classification = "joint_calibration_coupling_bottleneck"
    else:
        classification = "both_frozen_components_limiting"
    signed_recovery = effects[
        "oracle_dose_signed_distilled_direction"
    ]["full_oracle_recovery"]
    residual_recovery = effects[
        "oracle_dose_residual_distilled_direction"
    ]["full_oracle_recovery"]
    basis_specificity = (
        "unsupported_residual_matches_or_exceeds_signed"
        if residual_recovery >= signed_recovery
        else "signed_distiller_exceeds_residual_control"
    )
    return {
        "population": {
            "samples": len(labels),
            "source_accuracy": (~source_wrong).float().mean().item(),
            "source_errors": int(source_wrong.sum().item()),
            "memory_decodable_source_errors": int(primary.sum().item()),
        },
        "metrics": metrics,
        "effects": effects,
        "observability": {
            "primary_classifier": classifier,
            "oracle_dose_regression": dose,
            "signed_direction_distiller": signed_direction,
            "residual_direction_control": residual_direction,
        },
        "diagnosis": {
            "classification": classification,
            "frozen_router_recovery": effects[
                "frozen_learned_dose_learned_direction"
            ]["full_oracle_recovery"],
            "oracle_dose_learned_direction_recovery": effects[
                "oracle_dose_learned_direction"
            ]["full_oracle_recovery"],
            "learned_dose_oracle_direction_recovery": effects[
                "learned_dose_oracle_direction"
            ]["full_oracle_recovery"],
            "probed_dose_oracle_direction_recovery": effects[
                "probed_dose_oracle_direction"
            ]["full_oracle_recovery"],
            "signed_distilled_direction_recovery": signed_recovery,
            "residual_distilled_direction_recovery": residual_recovery,
            "basis_specificity": basis_specificity,
            "registered_next_boundary": (
                "Keep the trunk and seed909 locked; use this diagnosis to "
                "choose at most one pre-registered router-supervision repair."
            ),
        },
    }


def plot_analysis(analysis, path):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    effect_names = [
        "frozen_learned_dose_learned_direction",
        "oracle_dose_learned_direction",
        "learned_dose_oracle_direction",
        "probed_dose_oracle_direction",
        "oracle_dose_signed_distilled_direction",
        "oracle_dose_residual_distilled_direction",
        "oracle_dose_oracle_direction",
    ]
    labels = [
        "Learned/Learned", "Oracle dose", "Oracle direction", "Probe dose",
        "Signed distill", "Residual distill", "Full Oracle",
    ]
    values = [
        analysis["effects"][name]["full_oracle_recovery"]
        for name in effect_names
    ]
    axes[0].bar(np.arange(len(values)), values, color="#4E79A7")
    axes[0].axhline(RECOVERY_THRESHOLD, color="#E15759", linestyle="--")
    axes[0].set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
    axes[0].set_ylabel("Full-Oracle margin recovery")
    axes[0].set_title("Dose-direction hybrid ceilings")

    classifier = analysis["observability"]["primary_classifier"]
    axes[1].bar(
        ["AUROC", "AUPRC", "Top-k precision"],
        [classifier["auroc"], classifier["average_precision"],
         classifier["fixed_prevalence_precision"]],
        color=["#59A14F", "#F28E2B", "#B07AA1"],
    )
    axes[1].axhline(
        classifier["prevalence"], color="#777777", linestyle="--",
        label="prevalence",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Error-state observability")
    axes[1].legend()

    signed = analysis["observability"]["signed_direction_distiller"]
    residual = analysis["observability"]["residual_direction_control"]
    axes[2].bar(
        ["Signed\ncosine", "Residual\ncosine", "Dose\nSpearman"],
        [signed["primary_cosine_mean"], residual["primary_cosine_mean"],
         analysis["observability"]["oracle_dose_regression"]["spearman"]],
        color=["#4E79A7", "#E15759", "#76B7B2"],
    )
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set_title("Held-out observable prediction")
    figure.suptitle(
        "IST Level 6.19.5: Router Observability-Supervision Diagnosis",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def preregistration(args):
    return {
        "level": LEVEL,
        "status": "frozen router observability-supervision diagnosis",
        "parent": "Level 6.19.4 canonical formal_recovery result",
        "question": (
            "separate dose/gating, signed-direction, joint-coupling, and "
            "basis-specificity limits of label-free Oracle compilation"
        ),
        "splits": {
            "probe_train": {"samples": args.train_samples, "seed": args.train_seed},
            "probe_validation": {
                "samples": args.validation_samples,
                "seed": args.validation_seed,
                "role": "reporting only; no checkpoint or architecture selection",
            },
            "formal_diagnostic": {
                "samples": args.diagnostic_samples,
                "seed": args.diagnostic_seed,
                "opened_once_after_fixed final-epoch probes are frozen": True,
            },
        },
        "frozen_hybrids": {
            "learned_dose_learned_direction": "frozen Level 6.19.4 signed router",
            "oracle_dose_learned_direction": "label-aware diagnostic",
            "learned_dose_oracle_direction": "label-aware diagnostic",
            "oracle_dose_oracle_direction": "registered full signed Oracle",
        },
        "probes": {
            "inputs": "exactly the Level 6.19.4 router observables",
            "architecture": {
                "hidden": args.probe_hidden,
                "epochs": args.probe_epochs,
                "fixed_final_epoch": True,
                "no_model_or_checkpoint_selection": True,
            },
            "primary_classifier": "weighted BCE; AUROC, AUPRC, prevalence lift",
            "oracle_dose": "standardized log1p MSE; R2, Spearman, calibration",
            "signed_direction": "cosine distillation in projected value basis",
            "residual_direction": (
                "equal-parameter cosine distillation in fixed residual basis"
            ),
        },
        "interpretation": {
            "threshold": args.recovery_threshold,
            "oracle_dose_only_passes": "dose/gating dominant bottleneck",
            "oracle_direction_only_passes": "signed direction dominant bottleneck",
            "both_hybrids_pass": "joint calibration/coupling bottleneck",
            "neither_hybrid_passes": "both frozen components limiting",
            "residual_matches_or_exceeds_signed": (
                "value-basis-specific deployment claim unsupported"
            ),
        },
        "locks": {
            "seed707_trunk_and_all_existing_probes_frozen": True,
            "all_Level_6_19_4_routers_frozen": True,
            "failed_Level_6_18_9_candidate_not_used": True,
            "seed909_locked": True,
            "protected_tests_not_used": True,
            "optimizer_and_model_search_closed": True,
            "diagnostic_only": True,
        },
        "protocol": {
            key: value for key, value in vars(args).items()
            if key not in {"dry_run", "smoke_test", "force"}
        },
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError(f"Level {LEVEL} is fixed to seed707 at 16 chunks")
    for path in (
        args.checkpoint, args.probes, args.parent_result, args.parent_routers
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if min(
        args.train_samples, args.validation_samples, args.diagnostic_samples,
        args.eval_batch_size, args.probe_batch_size, args.probe_epochs,
    ) <= 0:
        raise ValueError("sample, batch, and epoch counts must be positive")
    for value in (
        args.train_samples, args.validation_samples, args.diagnostic_samples
    ):
        if value % args.eval_batch_size:
            raise ValueError("each split must be divisible by eval-batch-size")
    if not args.smoke_test and (
        args.train_samples != 4096
        or args.validation_samples != 1024
        or args.diagnostic_samples != 4096
        or args.train_seed != TRAIN_SEED
        or args.validation_seed != VALIDATION_SEED
        or args.diagnostic_seed != DIAGNOSTIC_SEED
        or args.probe_seed != PROBE_SEED
    ):
        raise ValueError(
            f"Formal Level {LEVEL} split sizes and seeds are fixed; use "
            "--smoke-test for implementation checks"
        )


def serializable_predictions(collected):
    return {
        "labels": collected["labels"].tolist(),
        "competitor": collected["competitor"].tolist(),
        "primary": collected["primary"].tolist(),
        "oracle_dose": collected["oracle_dose"].tolist(),
        "learned_dose": collected["learned_dose"].tolist(),
        "classifier_logits": collected["classifier_logits"].tolist(),
        "dose_prediction": collected["dose_prediction"].tolist(),
        "signed_oracle_cosine": (
            collected["signed_distilled_unit"] * collected["oracle_unit"]
        ).sum(dim=-1).tolist(),
        "residual_oracle_cosine": (
            collected["residual_distilled_unit"] * collected["oracle_unit"]
        ).sum(dim=-1).tolist(),
        "conditions": {
            name: {field: value.tolist() for field, value in row.items()}
            for name, row in collected["conditions"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.19.5 frozen router observability diagnosis"
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
        "--parent-result",
        default="experiments/level6_19_4/formal_recovery/result.json",
    )
    parser.add_argument(
        "--parent-routers",
        default="experiments/level6_19_4/formal_recovery/router_checkpoint.pt",
    )
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--diagnostic-samples", type=int, default=4096)
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--validation-seed", type=int, default=VALIDATION_SEED)
    parser.add_argument("--diagnostic-seed", type=int, default=DIAGNOSTIC_SEED)
    parser.add_argument("--analysis-seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--probe-seed", type=int, default=PROBE_SEED)
    parser.add_argument(
        "--residual-basis-seed", type=int, default=RESIDUAL_BASIS_SEED
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-hidden", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--dose-prediction-cap", type=float, default=DOSE_PREDICTION_CAP
    )
    parser.add_argument(
        "--recovery-threshold", type=float, default=RECOVERY_THRESHOLD
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--minimum-errors", type=int, default=200)
    parser.add_argument("--minimum-primary", type=int, default=150)
    parser.add_argument("--log-every-samples", type=int, default=256)
    parser.add_argument("--output", default="experiments/level6_19_5/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.train_samples = min(args.train_samples, 128)
        args.validation_samples = min(args.validation_samples, 64)
        args.diagnostic_samples = min(args.diagnostic_samples, 128)
        args.probe_epochs = min(args.probe_epochs, 2)
        args.bootstrap_iterations = min(args.bootstrap_iterations, 100)
        args.minimum_errors = 1
        args.minimum_primary = 1
        args.train_seed += 50_000_000
        args.validation_seed += 50_000_000
        args.diagnostic_seed += 50_000_000
        args.analysis_seed += 50_000_000
        args.probe_seed += 50_000_000
        if args.output == "experiments/level6_19_5/formal":
            args.output = "experiments/level6_19_5/smoke"
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
    model, original_probe, existing_probes, checkpoint_meta = load_frozen(
        args, device
    )
    routers, head_mask, parent_router_meta, parent_result = load_parent_routers(
        args, device
    )
    frozen_modules = {
        "model": model,
        "original_probe": original_probe,
        **{
            f"existing_probe_{name}": row["probe"]
            for name, row in existing_probes.items()
        },
        **{f"parent_router_{name}": router for name, router in routers.items()},
    }
    before = {
        name: tensor_fingerprint(module)
        for name, module in frozen_modules.items()
    }
    parent_router_before = {
        name: parameter_fingerprint(router)
        for name, router in routers.items()
    }

    train = collect_cache(
        model, existing_probes, args, args.train_samples, args.train_seed,
        device, dtype, root, "train",
    )
    validation = collect_cache(
        model, existing_probes, args, args.validation_samples,
        args.validation_seed, device, dtype, root, "validation",
    )
    train_log_dose = torch.log1p(train["cache"]["oracle_dose"])
    dose_stats = {
        "mean": train_log_dose.mean().item(),
        "std": train_log_dose.std().clamp_min(1e-6).item(),
    }
    classifier, classifier_training = train_scalar_probe(
        "classifier", train["cache"], validation["cache"], args, device,
        args.probe_seed,
    )
    dose_probe, dose_training = train_scalar_probe(
        "dose", train["cache"], validation["cache"], args, device,
        args.probe_seed + 1000, dose_stats,
    )
    signed_probe, signed_training = train_direction_probe(
        "signed", head_mask, train["cache"], validation["cache"], args,
        device, args.probe_seed + 2000,
    )
    residual_probe, residual_training = train_direction_probe(
        "residual", head_mask, train["cache"], validation["cache"], args,
        # Match every trainable initialization to the signed distiller.  The
        # fixed output basis is the only registered difference between them.
        device, args.probe_seed + 2000,
    )
    diagnostic_probes = {
        "classifier": classifier,
        "dose": dose_probe,
        "signed_direction": signed_probe,
        "residual_direction": residual_probe,
    }
    for probe in diagnostic_probes.values():
        probe.eval()
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
    training = {
        "classifier": classifier_training,
        "dose": dose_training,
        "signed_direction": signed_training,
        "residual_direction": residual_training,
        "dose_stats": dose_stats,
        "audit": {
            "direction_probe_parameter_counts_equal": (
                signed_training["parameters"] == residual_training["parameters"]
            ),
            "direction_probe_trainable_initialization_matched": (
                signed_training["initial_trainable_fingerprint"]
                == residual_training["initial_trainable_fingerprint"]
            ),
            "fixed_final_epoch_used": True,
            "validation_used_for_selection": False,
            "diagnostic_seen_before_freeze": False,
        },
    }
    save(root / "probe_training.json", training)
    torch.save({
        "level": LEVEL,
        "head_mask": head_mask.cpu(),
        "dose_stats": dose_stats,
        "states": {
            name: probe.state_dict()
            for name, probe in diagnostic_probes.items()
        },
        "training": training,
    }, root / "diagnostic_probes.pt")

    diagnostic = collect_cache(
        model, existing_probes, args, args.diagnostic_samples,
        args.diagnostic_seed, device, dtype, root, "formal_diagnostic",
    )
    collected = evaluate_formal(
        model, routers, diagnostic_probes, diagnostic["cache"], dose_stats,
        args, device, dtype, root,
    )
    analysis = analyze_formal(collected, args)
    after = {
        name: tensor_fingerprint(module)
        for name, module in frozen_modules.items()
    }
    parent_router_after = {
        name: parameter_fingerprint(router)
        for name, router in routers.items()
    }
    split_seeds = [
        args.train_seed, args.validation_seed, args.diagnostic_seed
    ]
    integrity = {
        "frozen_module_fingerprints_unchanged": before == after,
        "frozen_parameters_remain_frozen": all(
            not parameter.requires_grad
            for module in frozen_modules.values()
            for parameter in module.parameters()
        ),
        "parent_router_fingerprints_unchanged": (
            parent_router_before == parent_router_after
        ),
        "direction_probe_parameter_counts_equal": training["audit"][
            "direction_probe_parameter_counts_equal"
        ],
        "direction_probe_trainable_initialization_matched": training[
            "audit"
        ]["direction_probe_trainable_initialization_matched"],
        "train_source_replay_max_abs": train["source_replay_max_abs"],
        "validation_source_replay_max_abs": validation[
            "source_replay_max_abs"
        ],
        "diagnostic_source_replay_max_abs": diagnostic[
            "source_replay_max_abs"
        ],
        "oracle_l2_max_abs_error": max(
            train["oracle_l2_max_abs_error"],
            validation["oracle_l2_max_abs_error"],
            diagnostic["oracle_l2_max_abs_error"],
        ),
        "oracle_l2_passed": max(
            train["oracle_l2_max_abs_error"],
            validation["oracle_l2_max_abs_error"],
            diagnostic["oracle_l2_max_abs_error"],
        ) <= 1e-5,
        "split_seeds_unique": len(set(split_seeds)) == len(split_seeds),
        "diagnostic_opened_after_fixed_probe_freeze": True,
        "validation_not_used_for_selection": True,
        "failed_candidate_not_used": True,
        "seed909_locked": True,
        "protected_tests_not_used": True,
        "optimizer_search_closed": True,
    }
    integrity["passed"] = all([
        integrity["frozen_module_fingerprints_unchanged"],
        integrity["frozen_parameters_remain_frozen"],
        integrity["parent_router_fingerprints_unchanged"],
        integrity["direction_probe_parameter_counts_equal"],
        integrity["direction_probe_trainable_initialization_matched"],
        integrity["oracle_l2_passed"],
        integrity["split_seeds_unique"],
        integrity["diagnostic_opened_after_fixed_probe_freeze"],
        integrity["validation_not_used_for_selection"],
    ])
    if (
        analysis["population"]["source_errors"] < args.minimum_errors
        or analysis["population"]["memory_decodable_source_errors"]
        < args.minimum_primary
    ):
        raise RuntimeError("formal population is below its minimum size gate")
    analysis["diagnosis"]["integrity_passed"] = integrity["passed"]
    if not integrity["passed"]:
        analysis["diagnosis"]["classification"] = "integrity_failure"
        analysis["diagnosis"]["registered_next_boundary"] = (
            f"Stop; repair the Level {LEVEL} implementation."
        )
    result = {
        "protocol": protocol,
        "checkpoint_meta": checkpoint_meta,
        "parent_level6_19_4_diagnosis": parent_result["analysis"]["diagnosis"],
        "parent_router_meta": {
            "selected_heads": parent_router_meta["selected_heads"],
            "dose_cap": parent_router_meta["dose_cap"],
        },
        "integrity": integrity,
        "training": training,
        "analysis": analysis,
    }
    save(root / "result.json", result)
    save(root / "summary.json", {
        "integrity": integrity,
        "population": analysis["population"],
        "observability": analysis["observability"],
        "diagnosis": analysis["diagnosis"],
    })
    save(root / "predictions.json", serializable_predictions(collected))
    plot_analysis(analysis, root / "router_observability_diagnosis.png")
    save(root / "progress.json", {
        "stage": "complete",
        "integrity_passed": integrity["passed"],
        "classification": analysis["diagnosis"]["classification"],
    })
    print(json.dumps(analysis["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
