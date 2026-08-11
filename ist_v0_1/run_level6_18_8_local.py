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
from run_level6_2_local import make_chunks
from run_level6_13_1_local import bootstrap_mean_ci
from run_level6_18_6_local import (
    accuracy,
    configure_cuda,
    load_pair,
    paired_statistics,
    save,
)
from run_level6_18_7_local import (
    ActivationController,
    holm_adjust,
    memories_equal,
)


SEED = 707
CHUNKS = 16
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def alpha_key(alpha):
    return f"dose_{alpha:g}".replace(".", "_")


def condition_keys(args):
    keys = ["source", "updated"]
    keys.extend(alpha_key(alpha) for alpha in ALPHAS[1:])
    keys.extend(["gradient_parallel", "gradient_orthogonal"])
    keys.extend(f"batch_roll_{index + 1}" for index in range(args.control_repeats))
    keys.extend(f"random_{index + 1}" for index in range(args.control_repeats))
    return keys


def empty_parts(keys):
    fields = [
        "predictions",
        "fixed_margin",
        "decision_margin",
        "cross_entropy",
    ]
    return {key: {field: [] for field in fields} for key in keys}


def append_condition(parts, key, logits, target, reference_competitor):
    task_logits = logits[:, -1, :16].float()
    rows = torch.arange(len(target), device=target.device)
    correct_logits = task_logits[rows, target]
    fixed_margin = correct_logits - task_logits[rows, reference_competitor]
    masked = task_logits.clone()
    masked[rows, target] = -torch.inf
    decision_margin = correct_logits - masked.max(dim=-1).values
    parts[key]["predictions"].append(task_logits.argmax(-1).cpu())
    parts[key]["fixed_margin"].append(fixed_margin.detach().cpu())
    parts[key]["decision_margin"].append(decision_margin.detach().cpu())
    parts[key]["cross_entropy"].append(
        F.cross_entropy(task_logits, target, reduction="none").detach().cpu()
    )


def norm_match(direction, target_norm):
    norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return direction / norm * target_norm


def context_tensor(base_context, query_value):
    output = base_context.detach().clone()
    output[:, -1] = query_value.to(
        device=base_context.device, dtype=base_context.dtype
    )
    return output


def source_competitor(logits, target):
    task_logits = logits[:, -1, :16].float().clone()
    rows = torch.arange(len(target), device=target.device)
    task_logits[rows, target] = -torch.inf
    return task_logits.argmax(dim=-1)


@torch.no_grad()
def shared_prefix(model, controller, chunks, chunks_count, dtype):
    controller.set_patch()
    memory = None
    with torch.autocast(device_type="cuda", dtype=dtype):
        for chunk_index in range(chunks_count - 1):
            _, memory = model(
                chunks[:, chunk_index], memory=memory,
                return_memory=True, per_layer_memory=True,
            )
    return memory


def collect(models, args, device, dtype):
    keys = condition_keys(args)
    parts = empty_parts(keys)
    labels_parts = []
    competitor_parts = []
    gradient_parts = {
        "true_directional_derivative": [],
        "batch_roll_directional_derivative": [],
        "random_directional_derivative": [],
        "gradient_norm": [],
        "delta_norm": [],
        "gradient_delta_cosine": [],
    }
    source_controller = ActivationController(models["source"])
    updated_controller = ActivationController(models["updated"])
    memory_exact = True
    context_reconstruction_max_abs = 0.0
    gradient_baseline_logit_max_abs = 0.0
    gradients_finite = True
    total = 0
    batch_index = 0
    set_seed(args.dataset_seed)

    def run_source_context(final_chunk, memory, base_context, query):
        donor = context_tensor(base_context, query)
        source_controller.set_patch({"memory_context": donor})
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            logits, _ = models["source"](
                final_chunk, memory=memory,
                return_memory=True, per_layer_memory=True,
            )
        return logits

    try:
        while total < args.samples:
            batch = min(args.eval_batch_size, args.samples - total)
            chunks, target, _ = make_chunks(
                batch, CHUNKS, args.chunk_size, device
            )
            memory = shared_prefix(
                models["source"], source_controller, chunks, CHUNKS, dtype
            )
            final_chunk = chunks[:, -1]
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
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
            memory_exact = memory_exact and memories_equal(
                source_memory, updated_memory
            )
            competitor = source_competitor(source_logits, target)
            source_context = source_activations["memory_context"]
            source_query = source_context[:, -1].float()
            updated_query = updated_activations["memory_context"][:, -1].float()
            delta = updated_query - source_query
            reconstructed = (source_query + delta).to(dtype=source_context.dtype)
            context_reconstruction_max_abs = max(
                context_reconstruction_max_abs,
                (reconstructed - updated_activations["memory_context"][:, -1])
                .abs().max().item(),
            )

            labels_parts.append(target.cpu())
            competitor_parts.append(competitor.cpu())
            append_condition(parts, "source", source_logits, target, competitor)
            append_condition(parts, "updated", updated_logits, target, competitor)

            for alpha in ALPHAS[1:]:
                logits = run_source_context(
                    final_chunk, memory, source_context,
                    source_query + alpha * delta,
                )
                append_condition(
                    parts, alpha_key(alpha), logits, target, competitor
                )

            # Compute the fixed-rival margin gradient at the unmodified source
            # context. Labels are used only in this diagnostic attribution.
            query_variable = (
                source_context[:, -1].detach().clone().requires_grad_(True)
            )
            differentiable_context = torch.cat(
                [source_context[:, :-1].detach(), query_variable[:, None]], dim=1
            )
            source_controller.set_patch(
                {"memory_context": differentiable_context}
            )
            with torch.enable_grad(), torch.autocast(
                device_type="cuda", dtype=dtype
            ):
                gradient_logits, _ = models["source"](
                    final_chunk, memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
                task_logits = gradient_logits[:, -1, :16].float()
                rows = torch.arange(batch, device=device)
                fixed_margin = (
                    task_logits[rows, target]
                    - task_logits[rows, competitor]
                )
                gradient = torch.autograd.grad(
                    fixed_margin.sum(), query_variable, only_inputs=True
                )[0].float()
            gradient_baseline_logit_max_abs = max(
                gradient_baseline_logit_max_abs,
                (
                    gradient_logits[:, -1, :16].float()
                    - source_logits[:, -1, :16].float()
                ).abs().max().item(),
            )
            gradients_finite = gradients_finite and bool(
                torch.isfinite(gradient).all().item()
            )

            gradient_norm = gradient.norm(dim=-1)
            delta_norm = delta.norm(dim=-1)
            directional = (gradient * delta).sum(dim=-1)
            cosine = directional / (
                gradient_norm * delta_norm
            ).clamp_min(1e-8)
            denominator = gradient.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
            parallel = (
                (delta * gradient).sum(dim=-1, keepdim=True) / denominator
            ) * gradient
            orthogonal = delta - parallel

            parallel_logits = run_source_context(
                final_chunk, memory, source_context, source_query + parallel
            )
            orthogonal_logits = run_source_context(
                final_chunk, memory, source_context, source_query + orthogonal
            )
            append_condition(
                parts, "gradient_parallel", parallel_logits, target, competitor
            )
            append_condition(
                parts, "gradient_orthogonal", orthogonal_logits, target, competitor
            )

            batch_directionals = []
            random_directionals = []
            target_norm = delta_norm[:, None]
            for repeat in range(args.control_repeats):
                shift = repeat + 1
                rolled = norm_match(delta.roll(shift, dims=0), target_norm)
                rolled_logits = run_source_context(
                    final_chunk, memory, source_context, source_query + rolled
                )
                append_condition(
                    parts, f"batch_roll_{repeat + 1}",
                    rolled_logits, target, competitor,
                )
                batch_directionals.append((gradient * rolled).sum(dim=-1))

                generator = torch.Generator(device=device)
                generator.manual_seed(
                    args.control_seed + batch_index * 100 + repeat
                )
                random_direction = torch.randn(
                    delta.shape, generator=generator, device=device,
                    dtype=torch.float32,
                )
                random_direction = norm_match(random_direction, target_norm)
                random_logits = run_source_context(
                    final_chunk, memory, source_context,
                    source_query + random_direction,
                )
                append_condition(
                    parts, f"random_{repeat + 1}",
                    random_logits, target, competitor,
                )
                random_directionals.append(
                    (gradient * random_direction).sum(dim=-1)
                )

            gradient_parts["true_directional_derivative"].append(
                directional.detach().cpu()
            )
            gradient_parts["batch_roll_directional_derivative"].append(
                torch.stack(batch_directionals).mean(dim=0).detach().cpu()
            )
            gradient_parts["random_directional_derivative"].append(
                torch.stack(random_directionals).mean(dim=0).detach().cpu()
            )
            gradient_parts["gradient_norm"].append(gradient_norm.detach().cpu())
            gradient_parts["delta_norm"].append(delta_norm.detach().cpu())
            gradient_parts["gradient_delta_cosine"].append(cosine.detach().cpu())

            total += batch
            batch_index += 1
            if batch_index == 1 or batch_index % args.log_every_batches == 0:
                print(
                    f"samples={total}/{args.samples} "
                    f"source={(parts['source']['predictions'][-1] == target.cpu()).float().mean().item():.2%} "
                    f"mean_g_dot_delta={directional.mean().item():.5f}",
                    flush=True,
                )
    finally:
        source_controller.close()
        updated_controller.close()

    dataset = {
        key: {field: torch.cat(items) for field, items in fields.items()}
        for key, fields in parts.items()
    }
    gradients = {
        key: torch.cat(items) for key, items in gradient_parts.items()
    }
    return {
        "labels": torch.cat(labels_parts),
        "reference_competitor": torch.cat(competitor_parts),
        "conditions": dataset,
        "gradients": gradients,
        "integrity": {
            "returned_memory_exactly_invariant": memory_exact,
            "context_alpha1_reconstruction_max_abs": context_reconstruction_max_abs,
            "context_alpha1_reconstruction_exact": context_reconstruction_max_abs == 0.0,
            "gradients_finite": gradients_finite,
            "gradient_baseline_logit_max_abs": gradient_baseline_logit_max_abs,
            "gradient_baseline_logits_exact": gradient_baseline_logit_max_abs == 0.0,
            "gradient_nonzero_fraction": (
                gradients["gradient_norm"] > 0
            ).float().mean().item(),
        },
    }


def sign_flip_p(values, seed, iterations):
    values = np.asarray(values, dtype=np.float64)
    observed = abs(values.mean())
    generator = np.random.default_rng(seed)
    extreme = 1
    completed = 0
    block = 256
    while completed < iterations:
        count = min(block, iterations - completed)
        signs = generator.integers(
            0, 2, size=(count, len(values)), dtype=np.int8
        ) * 2 - 1
        means = (signs * values[None, :]).mean(axis=1)
        extreme += int((np.abs(means) >= observed - 1e-15).sum())
        completed += count
    return extreme / (iterations + 1)


def continuous_effect(values, args, seed):
    values = np.asarray(values, dtype=np.float64)
    return {
        "estimate": float(values.mean()),
        "ci95": bootstrap_mean_ci(
            values, seed, args.bootstrap_iterations
        )["ci95"],
        "sign_flip_p_two_sided": sign_flip_p(
            values, seed + 1, args.sign_flip_iterations
        ),
        "positive_fraction": float((values > 0).mean()),
        "negative_fraction": float((values < 0).mean()),
        "zero_fraction": float((values == 0).mean()),
        "samples": len(values),
    }


def paired_continuous(left, right, args, seed):
    left = left.numpy().astype(np.float64)
    right = right.numpy().astype(np.float64)
    output = continuous_effect(right - left, args, seed)
    output["left_mean"] = float(left.mean())
    output["right_mean"] = float(right.mean())
    return output


def condition_metric(item, labels):
    return {
        "accuracy": accuracy(item["predictions"], labels),
        "fixed_margin_mean": item["fixed_margin"].mean().item(),
        "decision_margin_mean": item["decision_margin"].mean().item(),
        "cross_entropy_mean": item["cross_entropy"].mean().item(),
        "samples": len(labels),
    }


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def analyze(collected, args):
    labels = collected["labels"]
    conditions = collected["conditions"]
    source = conditions["source"]
    metrics = {
        key: condition_metric(item, labels)
        for key, item in conditions.items()
    }
    effects = {}
    effect_keys = [
        alpha_key(alpha) for alpha in ALPHAS[1:]
    ] + ["updated", "gradient_parallel", "gradient_orthogonal"]
    for index, key in enumerate(effect_keys):
        effects[key] = {
            "accuracy": paired_statistics(
                source["predictions"], conditions[key]["predictions"],
                labels, args, args.bootstrap_seed + index,
            ),
            "fixed_margin": paired_continuous(
                source["fixed_margin"], conditions[key]["fixed_margin"],
                args, args.bootstrap_seed + 100 + index,
            ),
            "decision_margin": paired_continuous(
                source["decision_margin"], conditions[key]["decision_margin"],
                args, args.bootstrap_seed + 200 + index,
            ),
            "cross_entropy": paired_continuous(
                source["cross_entropy"], conditions[key]["cross_entropy"],
                args, args.bootstrap_seed + 300 + index,
            ),
        }

    batch_keys = [
        f"batch_roll_{index + 1}" for index in range(args.control_repeats)
    ]
    random_keys = [
        f"random_{index + 1}" for index in range(args.control_repeats)
    ]
    batch_fixed = torch.stack(
        [conditions[key]["fixed_margin"] for key in batch_keys]
    ).mean(dim=0)
    random_fixed = torch.stack(
        [conditions[key]["fixed_margin"] for key in random_keys]
    ).mean(dim=0)
    true_fixed = conditions[alpha_key(1.0)]["fixed_margin"]
    true_change = (true_fixed - source["fixed_margin"]).numpy()
    batch_change = (batch_fixed - source["fixed_margin"]).numpy()
    random_change = (random_fixed - source["fixed_margin"]).numpy()
    primary_values = {
        "true_margin_change": true_change,
        "true_minus_batch_roll": true_change - batch_change,
        "true_minus_random": true_change - random_change,
    }
    primary = {
        name: continuous_effect(
            values, args, args.primary_seed + index * 10
        )
        for index, (name, values) in enumerate(primary_values.items())
    }
    primary_holm = holm_adjust({
        name: row["sign_flip_p_two_sided"] for name, row in primary.items()
    })

    control_panel = {
        "batch_roll": {
            "repeat_metrics": {
                key: metrics[key] for key in batch_keys
            },
            "average_fixed_margin_effect": continuous_effect(
                batch_change, args, args.bootstrap_seed + 500
            ),
        },
        "random": {
            "repeat_metrics": {
                key: metrics[key] for key in random_keys
            },
            "average_fixed_margin_effect": continuous_effect(
                random_change, args, args.bootstrap_seed + 510
            ),
        },
    }

    alpha_array = np.asarray(ALPHAS, dtype=np.float64)
    dose_fixed = torch.stack(
        [
            source["fixed_margin"] if alpha == 0
            else conditions[alpha_key(alpha)]["fixed_margin"]
            for alpha in ALPHAS
        ], dim=1,
    ).numpy().astype(np.float64)
    centered_alpha = alpha_array - alpha_array.mean()
    slopes = (
        dose_fixed * centered_alpha[None, :]
    ).sum(axis=1) / (centered_alpha ** 2).sum()
    dose_curve = {
        "alphas": ALPHAS,
        "metrics": {
            "0": metrics["source"],
            **{
                f"{alpha:g}": metrics[alpha_key(alpha)]
                for alpha in ALPHAS[1:]
            },
        },
        "per_sample_fixed_margin_slope": continuous_effect(
            slopes, args, args.bootstrap_seed + 600
        ),
        "mean_fixed_margin_monotonic_non_decreasing": all(
            dose_fixed[:, index + 1].mean()
            >= dose_fixed[:, index].mean() - 1e-8
            for index in range(len(ALPHAS) - 1)
        ),
    }

    gradients = collected["gradients"]
    directional = gradients["true_directional_derivative"].numpy()
    batch_directional = gradients[
        "batch_roll_directional_derivative"
    ].numpy()
    random_directional = gradients[
        "random_directional_derivative"
    ].numpy()
    gradient_analysis = {
        "true_directional_derivative": continuous_effect(
            directional, args, args.bootstrap_seed + 700
        ),
        "true_minus_batch_roll_derivative": continuous_effect(
            directional - batch_directional,
            args, args.bootstrap_seed + 710,
        ),
        "true_minus_random_derivative": continuous_effect(
            directional - random_directional,
            args, args.bootstrap_seed + 720,
        ),
        "mean_gradient_norm": gradients["gradient_norm"].mean().item(),
        "mean_delta_norm": gradients["delta_norm"].mean().item(),
        "mean_gradient_delta_cosine": gradients[
            "gradient_delta_cosine"
        ].mean().item(),
        "positive_cosine_fraction": (
            gradients["gradient_delta_cosine"] > 0
        ).float().mean().item(),
        "directional_vs_actual_margin_change_correlation": safe_correlation(
            directional, true_change
        ),
    }

    integrity = collected["integrity"]
    integrity["passed"] = (
        integrity["returned_memory_exactly_invariant"]
        and integrity["context_alpha1_reconstruction_exact"]
        and integrity["gradients_finite"]
        and integrity["gradient_baseline_logits_exact"]
        and integrity["gradient_nonzero_fraction"] == 1.0
    )
    primary_pass = {
        name: (
            row["estimate"] > 0
            and primary_holm[name]["significant_0.05"]
        ) for name, row in primary.items()
    }
    if not integrity["passed"]:
        classification = "invalid_margin_or_gradient_integrity"
        next_boundary = "Repair gradient/context reconstruction before interpretation."
    elif all(primary_pass.values()):
        classification = "task_aligned_context_subspace_confirmed"
        next_boundary = "Register task-aligned read supervision with controls; keep other modules frozen."
    elif primary_pass["true_margin_change"]:
        classification = "margin_effect_not_specific_to_true_context_delta"
        next_boundary = "Identify norm or generic perturbation mechanism before training."
    elif not primary_pass["true_margin_change"]:
        classification = "no_confirmed_continuous_margin_alignment"
        next_boundary = "Stop route optimization and retain the result as a weak, unconfirmed effect."
    else:
        classification = "mixed_task_alignment_evidence"
        next_boundary = "Repeat only the frozen primary margin panel on more held-out examples."
    diagnosis = {
        "classification": classification,
        "integrity_passed": integrity["passed"],
        "primary": primary,
        "primary_holm_family": primary_holm,
        "primary_pass": primary_pass,
        "registered_next_boundary": next_boundary,
    }
    return {
        "metrics": metrics,
        "effects_vs_source": effects,
        "dose_curve": dose_curve,
        "control_panel": control_panel,
        "gradient_analysis": gradient_analysis,
        "integrity": integrity,
        "diagnosis": diagnosis,
    }


def plot_result(analysis, collected, path):
    dose = analysis["dose_curve"]
    alphas = np.asarray(dose["alphas"])
    margin_change = np.asarray([
        0.0 if alpha == 0 else analysis["effects_vs_source"][
            alpha_key(alpha)
        ]["fixed_margin"]["estimate"]
        for alpha in alphas
    ])
    accuracies = np.asarray([
        dose["metrics"][f"{alpha:g}"]["accuracy"] for alpha in alphas
    ])
    primary = analysis["diagnosis"]["primary"]
    components = [
        ("True delta", primary["true_margin_change"]),
        (
            "Batch-roll avg",
            analysis["control_panel"]["batch_roll"][
                "average_fixed_margin_effect"
            ],
        ),
        (
            "Random avg",
            analysis["control_panel"]["random"][
                "average_fixed_margin_effect"
            ],
        ),
        (
            "Gradient parallel",
            analysis["effects_vs_source"]["gradient_parallel"]["fixed_margin"],
        ),
        (
            "Gradient orthogonal",
            analysis["effects_vs_source"]["gradient_orthogonal"]["fixed_margin"],
        ),
    ]
    directional = collected["gradients"][
        "true_directional_derivative"
    ].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    axis = axes[0]
    axis.plot(alphas, margin_change, marker="o", color="#4c78a8", linewidth=2)
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xlabel("Context interpolation alpha")
    axis.set_ylabel("Mean fixed-rival margin change", color="#4c78a8")
    twin = axis.twinx()
    twin.plot(alphas, 100 * accuracies, marker="s", color="#e45756", linewidth=2)
    twin.set_ylabel("Query accuracy (%)", color="#e45756")
    axis.set_title("True context-delta dose curve")
    axis.grid(alpha=0.2)

    axis = axes[1]
    estimates = [row[1]["estimate"] for row in components]
    lows = [row[1]["ci95"][0] for row in components]
    highs = [row[1]["ci95"][1] for row in components]
    errors = np.asarray([
        np.asarray(estimates) - np.asarray(lows),
        np.asarray(highs) - np.asarray(estimates),
    ])
    axis.bar(
        range(len(components)), estimates,
        color=["#4c78a8", "#f2cf5b", "#b8b8b8", "#54a24b", "#e45756"],
        yerr=errors, capsize=4,
    )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(
        range(len(components)), [row[0] for row in components],
        rotation=25, ha="right",
    )
    axis.set_ylabel("Mean fixed-rival margin change")
    axis.set_title("True delta, nulls, and gradient components")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[2]
    low, high = np.percentile(directional, [1, 99])
    axis.hist(
        np.clip(directional, low, high), bins=45,
        color="#72b7b2", edgecolor="white",
    )
    axis.axvline(0, color="#333333", linewidth=1.2)
    axis.axvline(directional.mean(), color="#e45756", linestyle="--", linewidth=2)
    axis.set_xlabel("g(context) dot true delta")
    axis.set_ylabel("Examples")
    axis.set_title("First-order task alignment")

    fig.suptitle(
        "IST Level 6.18.8: 16-Chunk Task-Aligned Read Subspace",
        fontsize=17,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def serializable_raw(collected):
    return {
        "labels": collected["labels"].tolist(),
        "reference_competitor": collected["reference_competitor"].tolist(),
        "conditions": {
            key: {
                field: tensor.tolist() for field, tensor in item.items()
            } for key, item in collected["conditions"].items()
        },
        "gradients": {
            key: tensor.tolist() for key, tensor in collected["gradients"].items()
        },
    }


def preregistration(args):
    return {
        "level": "6.18.8",
        "status": "frozen 16-chunk task-aligned read-subspace diagnosis",
        "seed": SEED,
        "source": "Level 6.18.3 formally passed checkpoint",
        "updated": "Level 6.18.5 update-500 checkpoint",
        "chunks": CHUNKS,
        "samples": args.samples,
        "target_activation": "final block memory_context at final query token",
        "dose_curve": ALPHAS,
        "primary_margin": (
            "correct-class logit minus the source model's strongest incorrect logit"
        ),
        "primary_family": {
            "true_margin_change": "source context -> updated context through source gate",
            "true_minus_batch_roll": "true delta effect minus mean of norm-matched batch rolls",
            "true_minus_random": "true delta effect minus mean of norm-matched random directions",
            "test": "two-sided paired sign-flip",
            "multiplicity": "Holm correction across three tests",
            "confirmation": "positive estimate and Holm p < 0.05 for all three",
        },
        "null_controls": {
            "batch_roll_repeats": args.control_repeats,
            "random_repeats": args.control_repeats,
            "norm_matched_per_example": True,
        },
        "gradient_diagnostics": {
            "fixed_source_rival": True,
            "directional_derivative": "margin gradient dot true context delta",
            "components": ["gradient_parallel", "gradient_orthogonal"],
            "uses_test_labels": True,
            "status": "mechanism-only; not deployable and not a primary confirmation",
        },
        "secondary_metrics": [
            "decision margin", "cross entropy", "argmax accuracy"
        ],
        "integrity": {
            "returned_memory_exactly_invariant": True,
            "alpha1_context_reconstruction_exact": True,
            "differentiable_source_context_reproduces_source_logits_exactly": True,
            "finite_nonzero_per-example_gradients": True,
        },
        "no_parameter_or_probe_updates": True,
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError("Level 6.18.8 is fixed to seed707 at 16 chunks")
    if args.alphas != ALPHAS:
        raise ValueError(f"Level 6.18.8 dose curve is fixed to {ALPHAS}")
    if args.samples <= 0 or args.samples % args.eval_batch_size != 0:
        raise ValueError("samples must be positive and divisible by eval-batch-size")
    if not 1 <= args.control_repeats < args.eval_batch_size:
        raise ValueError("control-repeats must be in [1, eval-batch-size)")
    for checkpoint in (args.source_checkpoint, args.updated_checkpoint):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.8 frozen task-aligned read-subspace diagnosis"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
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
    parser.add_argument("--control-repeats", type=int, default=8)
    parser.add_argument("--dataset-seed", type=int, default=6188000)
    parser.add_argument("--control-seed", type=int, default=6188100)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=6188200)
    parser.add_argument("--primary-seed", type=int, default=6188300)
    parser.add_argument("--log-every-batches", type=int, default=8)
    parser.add_argument("--output", default="experiments/level6_18_8/formal")
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
    models, probes, audit, update_meta = load_pair(args, device)
    del probes
    print(
        f"checkpoint audit passed: {audit['changed_tensor_count']} tensors, "
        f"{audit['changed_parameter_count']} parameters",
        flush=True,
    )
    collected = collect(models, args, device, dtype)
    analysis = analyze(collected, args)
    result = {
        "protocol": protocol,
        "checkpoint_audit": audit,
        "level6_18_5_update": {
            "update": update_meta.get("update"),
            "passed": update_meta.get("passed"),
            "stable_streak": update_meta.get("stable_streak"),
        },
        "analysis": analysis,
    }
    save(result_path, result)
    save(root / "summary.json", {
        "dose_curve": analysis["dose_curve"],
        "control_panel": analysis["control_panel"],
        "gradient_analysis": analysis["gradient_analysis"],
        "integrity": analysis["integrity"],
        "diagnosis": analysis["diagnosis"],
    })
    save(root / "predictions.json", serializable_raw(collected))
    plot_result(analysis, collected, root / "task_aligned_subspace.png")
    print(json.dumps(analysis["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
