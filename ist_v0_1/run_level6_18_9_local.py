import argparse
import copy
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from long_context_test import set_seed
from run_level6_2_local import make_chunks
from run_level6_6_local import build, restore
from run_level6_9_local import CONDITIONS, intervene
from run_level6_18_5_local import atomic_checkpoint
from run_level6_18_6_local import (
    configure_cuda,
    paired_statistics,
    save,
)
from run_level6_18_8_local import continuous_effect


SEED = 707
TRAIN_CHUNKS = [8, 12, 16]
READ_PREFIX = "blocks.2.memory_read."


def read_parameters(model):
    selected = [
        (name, parameter) for name, parameter in model.named_parameters()
        if name.startswith(READ_PREFIX)
    ]
    if len(selected) != 4 or sum(parameter.numel() for _, parameter in selected) != 16640:
        raise RuntimeError(
            "Unexpected final memory_read boundary: "
            f"{[(name, tuple(parameter.shape)) for name, parameter in selected]}"
        )
    return selected


def load_source(args, device):
    state = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    if not state.get("level6_18_3", {}).get("success", {}).get("passed"):
        raise RuntimeError("Level 6.18.3 source checkpoint did not pass formally")
    model, probe = build(device, args.chunk_size)
    baseline, baseline_probe = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    probe.load_state_dict(state["probe"])
    baseline.load_state_dict(state["model"])
    baseline_probe.load_state_dict(state["probe"])
    for module in (model, probe, baseline, baseline_probe):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    selected = read_parameters(model)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    source_read = copy.deepcopy(baseline.blocks[-1].memory_read).to(device)
    source_read.eval()
    for parameter in source_read.parameters():
        parameter.requires_grad_(False)
    del baseline_probe
    return model, probe, baseline, source_read, state, selected


class FinalReadCapture:
    def __init__(self, model):
        self.values = {}
        block = model.blocks[-1]
        self.handles = [
            block.memory.register_forward_hook(self._memory_hook),
            block.memory_read.register_forward_pre_hook(self._read_pre_hook),
            block.memory_read.register_forward_hook(self._read_hook),
        ]

    def clear(self):
        self.values = {}

    def close(self):
        for handle in self.handles:
            handle.remove()

    def require(self):
        expected = {"pre_fusion", "read_query", "read_memory", "context"}
        missing = expected - set(self.values)
        if missing:
            raise RuntimeError(f"Final read capture missing {sorted(missing)}")

    def _memory_hook(self, _module, _inputs, output):
        self.values["pre_fusion"] = output[1]

    def _read_pre_hook(self, _module, inputs):
        self.values["read_query"] = inputs[0]
        self.values["read_memory"] = inputs[1]

    def _read_hook(self, _module, _inputs, output):
        self.values["context"] = output[0]


def task_margin(task_logits, target):
    rows = torch.arange(len(target), device=target.device)
    correct = task_logits[rows, target]
    masked = task_logits.clone()
    masked[rows, target] = -torch.inf
    return correct - masked.max(dim=-1).values


def query_logits_from_context(model, query_x, pre_fusion, context):
    block = model.blocks[-1]
    gate = block.memory_fusion_gate(
        torch.cat([query_x, context], dim=-1)
    )
    fused = pre_fusion + gate * context
    hidden = block.norm2(query_x + block.ffn(fused))
    return model.output(hidden)[:, 0, :16]


def train_update(model, source_read, capture, optimizer, selected_parameters,
                 args, device, dtype):
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for chunks_count in TRAIN_CHUNKS:
        chunks, target, _ = make_chunks(
            args.train_batch_size, chunks_count, args.chunk_size, device
        )
        memory = None
        capture.clear()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            for chunk_index in range(chunks_count - 1):
                _, memory = model(
                    chunks[:, chunk_index], memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits, _ = model(
                chunks[:, -1], memory=memory,
                return_memory=True, per_layer_memory=True,
            )
            capture.require()
            task_logits = logits[:, -1, :16].float()
            intact_margin = task_margin(task_logits, target)

            query_x = capture.values["read_query"][:, -1:]
            read_memory = capture.values["read_memory"]
            current_context_full = capture.values["context"]
            current_context = current_context_full[:, -1:]
            pre_fusion = capture.values["pre_fusion"][:, -1:]
            with torch.no_grad():
                source_context, _ = source_read(
                    query_x, read_memory, read_memory, need_weights=False
                )

            rolled_memory = read_memory.roll(1, dims=0)
            rolled_context, _ = model.blocks[-1].memory_read(
                query_x, rolled_memory, rolled_memory, need_weights=False
            )
            rolled_logits = query_logits_from_context(
                model, query_x, pre_fusion, rolled_context
            ).float()
            rolled_margin = task_margin(rolled_logits, target)

            context_gradient_full = torch.autograd.grad(
                intact_margin.sum(), current_context_full,
                retain_graph=True, only_inputs=True,
            )[0]
            context_gradient = context_gradient_full[:, -1:].detach().float()
            delta = current_context.float() - source_context.float()
            denominator = context_gradient.square().sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            parallel = (
                (delta * context_gradient).sum(dim=-1, keepdim=True)
                / denominator
            ) * context_gradient
            orthogonal = delta - parallel

            margin_loss = F.softplus(
                args.margin_target - intact_margin
            ).mean()
            contrast_loss = F.softplus(
                args.contrast_gap - (intact_margin - rolled_margin)
            ).mean()
            orthogonal_loss = orthogonal.square().mean()
            drift_loss = delta.square().mean()
            total_loss = (
                margin_loss
                + args.contrast_weight * contrast_loss
                + args.orthogonal_weight * orthogonal_loss
                + args.drift_weight * drift_loss
            )
            (total_loss / len(TRAIN_CHUNKS)).backward()
        rows.append({
            "chunks": chunks_count,
            "loss": total_loss.detach().float().item(),
            "margin_loss": margin_loss.detach().float().item(),
            "contrast_loss": contrast_loss.detach().float().item(),
            "orthogonal_loss": orthogonal_loss.detach().float().item(),
            "drift_loss": drift_loss.detach().float().item(),
            "query": (task_logits.argmax(-1) == target).float().mean().item(),
            "margin": intact_margin.detach().mean().item(),
            "rolled_margin": rolled_margin.detach().mean().item(),
        })
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        selected_parameters, args.gradient_clip
    )
    optimizer.step()
    return {
        "by_chunks": rows,
        "loss": sum(row["loss"] for row in rows) / len(rows),
        "query": sum(row["query"] for row in rows) / len(rows),
        "margin": sum(row["margin"] for row in rows) / len(rows),
        "rolled_margin": sum(row["rolled_margin"] for row in rows) / len(rows),
        "gradient_norm": float(gradient_norm),
    }


@torch.no_grad()
def evaluate_model(model, args, chunks_count, samples, seed, condition,
                   device, dtype):
    set_seed(seed)
    labels = []
    predictions = []
    margins = []
    losses = []
    local_predictions = []
    total = 0
    model.eval()
    while total < samples:
        batch = min(args.eval_batch_size, samples - total)
        chunks, target, position = make_chunks(
            batch, chunks_count, args.chunk_size, device
        )
        memory = None
        first_logits = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for chunk_index in range(chunks_count):
                logits, produced = model(
                    chunks[:, chunk_index], memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
                if chunk_index == 0:
                    first_logits = logits
                memory = intervene(produced, condition)
        task_logits = logits[:, -1, :16].float()
        rows = torch.arange(batch, device=device)
        labels.append(target.cpu())
        predictions.append(task_logits.argmax(-1).cpu())
        margins.append(task_margin(task_logits, target).cpu())
        losses.append(F.cross_entropy(
            task_logits, target, reduction="none"
        ).cpu())
        local_predictions.append(
            first_logits[rows, position, :16].argmax(-1).cpu()
        )
        total += batch
    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    margins = torch.cat(margins)
    losses = torch.cat(losses)
    local_predictions = torch.cat(local_predictions)
    return {
        "labels": labels,
        "predictions": predictions,
        "margins": margins,
        "cross_entropy": losses,
        "local_predictions": local_predictions,
        "metric": {
            "condition": condition,
            "chunks": chunks_count,
            "samples": len(labels),
            "query": (predictions == labels).float().mean().item(),
            "margin": margins.mean().item(),
            "cross_entropy": losses.mean().item(),
            "local": (local_predictions == labels).float().mean().item(),
        },
    }


def preserved_evaluate(model, args, chunks_count, samples, seed, condition,
                       device, dtype):
    python_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    try:
        return evaluate_model(
            model, args, chunks_count, samples, seed, condition, device, dtype
        )
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)


def validation_seed(args, chunks_count, confirmation):
    return (
        args.validation_seed_base
        + chunks_count * 10
        + (1 if confirmation else 0)
    )


def baseline_validation(baseline, args, device, dtype):
    panels = {"screen": {}, "confirmation": {}}
    for confirmation, name, samples in (
        (False, "screen", args.screen_samples),
        (True, "confirmation", args.confirm_samples),
    ):
        for count in TRAIN_CHUNKS:
            panels[name][str(count)] = evaluate_model(
                baseline, args, count, samples,
                validation_seed(args, count, confirmation),
                "intact", device, dtype,
            )
    return panels


def current_validation(model, baseline_panels, args, confirmation,
                       device, dtype):
    name = "confirmation" if confirmation else "screen"
    samples = args.confirm_samples if confirmation else args.screen_samples
    output = {}
    for count in TRAIN_CHUNKS:
        item = preserved_evaluate(
            model, args, count, samples,
            validation_seed(args, count, confirmation),
            "intact", device, dtype,
        )
        baseline_item = baseline_panels[name][str(count)]
        if not torch.equal(item["labels"], baseline_item["labels"]):
            raise RuntimeError(f"Validation labels diverged at chunks={count}")
        output[str(count)] = {
            "baseline": baseline_item["metric"],
            "rescued": item["metric"],
            "margin_change": (
                item["metric"]["margin"] - baseline_item["metric"]["margin"]
            ),
        }
    return output


def screen_candidate(panel, args):
    return (
        panel["8"]["rescued"]["query"] >= args.screen_retention_threshold
        and panel["12"]["rescued"]["query"] >= args.screen_retention_threshold
        and panel["16"]["rescued"]["query"] >= args.screen_rescue_threshold
        and panel["16"]["margin_change"] >= args.screen_margin_improvement
    )


def confirmed_candidate(panel, args):
    return (
        all(
            panel[str(count)]["rescued"]["query"] >= args.confirm_threshold
            for count in TRAIN_CHUNKS
        )
        and panel["16"]["margin_change"] >= args.confirm_margin_improvement
        and panel["8"]["margin_change"] >= -args.margin_retention_tolerance
        and panel["12"]["margin_change"] >= -args.margin_retention_tolerance
    )


def compact_baseline(panels):
    return {
        panel: {
            count: item["metric"] for count, item in rows.items()
        } for panel, rows in panels.items()
    }


def train_read(model, probe, baseline_panels, source_read, capture,
               optimizer, selected_parameters, args, device, dtype, root):
    latest_path = root / "read_supervision_latest.pt"
    best_path = root / "read_supervision_best.pt"
    stable_path = root / "read_supervision_stable.pt"
    if args.force:
        for path in (latest_path, best_path, stable_path):
            path.unlink(missing_ok=True)
    history = []
    start_update = 0
    stable_streak = 0
    best_score = None
    if stable_path.exists() and not args.force:
        state = restore(stable_path, model, probe, optimizer, device)
        return state["read_training"]
    if latest_path.exists() and not args.force:
        state = restore(latest_path, model, probe, optimizer, device)
        meta = state["read_training"]
        history = meta["history"]
        start_update = meta["update"]
        stable_streak = meta["stable_streak"]
        best_score = meta.get("best_score")
        if start_update >= args.read_updates:
            return meta
    else:
        set_seed(args.training_seed)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    for parameter in selected_parameters:
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = args.read_lr

    last_train = None
    for update in range(start_update + 1, args.read_updates + 1):
        last_train = train_update(
            model, source_read, capture, optimizer, selected_parameters,
            args, device, dtype,
        )
        should_evaluate = (
            update == 1
            or update % args.eval_every_updates == 0
            or update == args.read_updates
        )
        if not should_evaluate:
            continue
        screen = current_validation(
            model, baseline_panels, args, False, device, dtype
        )
        confirmation = None
        if screen_candidate(screen, args):
            confirmation = current_validation(
                model, baseline_panels, args, True, device, dtype
            )
        confirmed = bool(
            confirmation and confirmed_candidate(confirmation, args)
        )
        stable_streak = stable_streak + 1 if confirmed else 0
        row = {
            "update": update,
            "train": last_train,
            "screen": screen,
            "confirmation": confirmation,
            "confirmed": confirmed,
            "stable_streak": stable_streak,
        }
        history.append(row)
        save(root / "read_supervision_progress.json", history)

        score = None
        if confirmation:
            score = [
                confirmation["16"]["rescued"]["query"],
                confirmation["16"]["margin_change"],
                min(
                    confirmation["8"]["rescued"]["query"],
                    confirmation["12"]["rescued"]["query"],
                ),
            ]
        is_best = score is not None and (
            best_score is None or tuple(score) > tuple(best_score)
        )
        if is_best:
            best_score = score
        meta = {
            "update": update,
            "stable_streak": stable_streak,
            "passed": stable_streak >= args.stable_confirmations,
            "best_score": best_score,
            "last_train": last_train,
            "screen": screen,
            "confirmation": confirmation,
            "history": history,
        }
        atomic_checkpoint(
            latest_path, model, probe, optimizer, {"read_training": meta}
        )
        if is_best:
            atomic_checkpoint(
                best_path, model, probe, optimizer, {"read_training": meta}
            )
        if meta["passed"]:
            atomic_checkpoint(
                stable_path, model, probe, optimizer, {"read_training": meta}
            )
            return meta
        print(
            f"update={update} train_margin={last_train['margin']:.3f} "
            f"screen8={screen['8']['rescued']['query']:.2%} "
            f"screen12={screen['12']['rescued']['query']:.2%} "
            f"screen16={screen['16']['rescued']['query']:.2%} "
            f"margin16={screen['16']['margin_change']:+.4f} "
            f"stable={stable_streak}/{args.stable_confirmations}",
            flush=True,
        )
    return {
        "update": args.read_updates,
        "stable_streak": stable_streak,
        "passed": False,
        "best_score": best_score,
        "last_train": last_train,
        "screen": history[-1]["screen"] if history else None,
        "confirmation": history[-1]["confirmation"] if history else None,
        "history": history,
    }


def parameter_audit(source_state, model, probe):
    current = model.state_dict()
    changed = []
    illegal = []
    for name, source_value in source_state["model"].items():
        current_value = current[name].detach().cpu()
        if torch.equal(source_value, current_value):
            continue
        row = {
            "name": name,
            "parameters": source_value.numel(),
            "max_abs_change": (
                source_value.float() - current_value.float()
            ).abs().max().item(),
            "allowed": name.startswith(READ_PREFIX),
        }
        changed.append(row)
        if not row["allowed"]:
            illegal.append(name)
    expected = {
        name for name in source_state["model"] if name.startswith(READ_PREFIX)
    }
    probe_equal = all(
        torch.equal(value, probe.state_dict()[name].detach().cpu())
        for name, value in source_state["probe"].items()
    )
    changed_names = {row["name"] for row in changed}
    passed = (
        not illegal
        and changed_names == expected
        and len(changed) == 4
        and sum(row["parameters"] for row in changed) == 16640
        and probe_equal
    )
    return {
        "passed": passed,
        "changed_tensors": changed,
        "changed_tensor_count": len(changed),
        "changed_parameter_count": sum(row["parameters"] for row in changed),
        "illegal_changes": illegal,
        "probe_unchanged": probe_equal,
        "allowed_prefix": READ_PREFIX,
    }


@torch.no_grad()
def memory_invariance(baseline, rescued, args, device, dtype):
    rows = []
    overall_max = 0.0
    for count in TRAIN_CHUNKS:
        set_seed(args.invariance_seed_base + count)
        chunks, _, _ = make_chunks(
            args.invariance_samples, count, args.chunk_size, device
        )
        baseline_memory = None
        rescued_memory = None
        for chunk_index in range(count):
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, baseline_memory = baseline(
                    chunks[:, chunk_index], memory=baseline_memory,
                    return_memory=True, per_layer_memory=True,
                )
                _, rescued_memory = rescued(
                    chunks[:, chunk_index], memory=rescued_memory,
                    return_memory=True, per_layer_memory=True,
                )
            for layer, (left, right) in enumerate(
                zip(baseline_memory, rescued_memory)
            ):
                maximum = (left.float() - right.float()).abs().max().item()
                overall_max = max(overall_max, maximum)
                rows.append({
                    "chunks": count,
                    "chunk_index": chunk_index,
                    "layer": layer,
                    "max_abs_difference": maximum,
                    "exact": torch.equal(left, right),
                })
    return {
        "passed": all(row["exact"] for row in rows),
        "overall_max_abs_difference": overall_max,
        "rows": rows,
    }


def paired_test(baseline_item, rescued_item, args, count):
    if not torch.equal(baseline_item["labels"], rescued_item["labels"]):
        raise RuntimeError(f"Protected labels diverged at chunks={count}")
    labels = baseline_item["labels"]
    accuracy_stats = paired_statistics(
        baseline_item["predictions"], rescued_item["predictions"],
        labels, args, args.bootstrap_seed_base + count,
    )
    margin_values = (
        rescued_item["margins"] - baseline_item["margins"]
    ).numpy()
    margin_stats = continuous_effect(
        margin_values, args, args.bootstrap_seed_base + 100 + count
    )
    return {
        "baseline": baseline_item["metric"],
        "rescued": rescued_item["metric"],
        "accuracy": accuracy_stats,
        "margin_change": margin_stats,
    }


def causal_summary(rows, args):
    intact = rows["intact"]
    passed = (
        intact["query"] >= args.test_threshold
        and intact["local"] >= args.causal_local_threshold
        and all(
            rows[condition]["query"] <= args.causal_intervention_threshold
            for condition in ("reset", "zero", "batch_roll")
        )
    )
    return {"passed": passed, "conditions": rows}


def plot_training(training, protected, path):
    history = training.get("history", [])
    updates = [row["update"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    if history:
        for count, color in zip(TRAIN_CHUNKS, ("#4c78a8", "#f2cf5b", "#e45756")):
            axes[0].plot(
                updates,
                [100 * row["screen"][str(count)]["rescued"]["query"] for row in history],
                marker="o", markersize=3, color=color, label=f"{count} chunks",
            )
        axes[0].axhline(95, color="#333333", linestyle="--")
        axes[0].set_xlabel("Optimizer update")
        axes[0].set_ylabel("Fixed screen query accuracy (%)")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        axes[1].plot(
            updates,
            [row["screen"]["16"]["margin_change"] for row in history],
            marker="s", markersize=3, color="#54a24b",
        )
        axes[1].axhline(0, color="#333333", linewidth=1)
        axes[1].set_xlabel("Optimizer update")
        axes[1].set_ylabel("16-chunk margin change vs source")
        axes[1].grid(alpha=0.2)
    if protected:
        axes[0].scatter(
            [updates[-1]] * 3,
            [100 * protected[str(count)]["rescued"]["query"] for count in TRAIN_CHUNKS],
            marker="*", s=130, color="#000000", label="protected test",
        )
    fig.suptitle("IST Level 6.18.9: Task-Aligned Memory-Read Supervision")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.18.9",
        "status": "single-protocol task-aligned read-supervision rescue",
        "seed": SEED,
        "source": "formally passed Level 6.18.3 checkpoint",
        "trainable_boundary": {
            "prefix": READ_PREFIX,
            "tensors": 4,
            "parameters": 16640,
            "fusion_gate_frozen": True,
            "all_other_parameters_frozen": True,
        },
        "balanced_training_lengths": TRAIN_CHUNKS,
        "objective": {
            "deployed_margin": "softplus(target_margin - intact_margin)",
            "memory_contrast": "intact margin must beat batch-rolled-Memory margin",
            "orthogonal_drift": "penalize context delta orthogonal to frozen deployed margin gradient",
            "total_drift": "small trust-region penalty from source context",
            "weights": {
                "contrast": args.contrast_weight,
                "orthogonal": args.orthogonal_weight,
                "drift": args.drift_weight,
            },
        },
        "optimizer": {
            "name": "AdamW",
            "lr": args.read_lr,
            "weight_decay": args.read_weight_decay,
            "maximum_updates": args.read_updates,
            "no_search": True,
        },
        "stable_gate": {
            "lengths": TRAIN_CHUNKS,
            "query_threshold": args.confirm_threshold,
            "16_margin_improvement": args.confirm_margin_improvement,
            "successive_confirmations": args.stable_confirmations,
        },
        "fail_closed": (
            "protected tests and causal panel open only after stable validation"
        ),
        "formal_success": {
            "protected_accuracy_all_lengths": args.test_threshold,
            "positive_16_accuracy_ci": True,
            "positive_16_margin_ci_and_sign_flip": True,
            "only_memory_read_changed": True,
            "persistent_memory_exactly_invariant": True,
            "causal_gate": True,
        },
        "seed909_locked": True,
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED:
        raise ValueError("Level 6.18.9 is fixed to seed707")
    if not Path(args.source_checkpoint).exists():
        raise FileNotFoundError(args.source_checkpoint)
    if args.train_batch_size < 2 or args.eval_batch_size < 2:
        raise ValueError("batch-roll requires train/eval batch sizes >= 2")
    if args.invariance_samples > args.eval_batch_size:
        raise ValueError("invariance-samples must fit one evaluation batch")


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.9 task-aligned final memory_read supervision"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--source-checkpoint",
        default="experiments/level6_18_3/formal/rescued_head_checkpoint.pt",
    )
    parser.add_argument("--read-updates", type=int, default=500)
    parser.add_argument("--read-lr", type=float, default=3e-5)
    parser.add_argument("--read-weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--margin-target", type=float, default=1.0)
    parser.add_argument("--contrast-gap", type=float, default=1.0)
    parser.add_argument("--contrast-weight", type=float, default=0.5)
    parser.add_argument("--orthogonal-weight", type=float, default=0.1)
    parser.add_argument("--drift-weight", type=float, default=0.01)
    parser.add_argument("--training-seed", type=int, default=6189000)
    parser.add_argument("--eval-every-updates", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--screen-samples", type=int, default=128)
    parser.add_argument("--confirm-samples", type=int, default=512)
    parser.add_argument("--validation-seed-base", type=int, default=6189100)
    parser.add_argument("--screen-retention-threshold", type=float, default=0.94)
    parser.add_argument("--screen-rescue-threshold", type=float, default=0.93)
    parser.add_argument("--screen-margin-improvement", type=float, default=0.02)
    parser.add_argument("--confirm-threshold", type=float, default=0.95)
    parser.add_argument("--confirm-margin-improvement", type=float, default=0.03)
    parser.add_argument("--margin-retention-tolerance", type=float, default=0.02)
    parser.add_argument("--stable-confirmations", type=int, default=2)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--test-seed-base", type=int, default=6189200)
    parser.add_argument("--test-threshold", type=float, default=0.95)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed-base", type=int, default=6189300)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--causal-seed", type=int, default=6189400)
    parser.add_argument("--causal-intervention-threshold", type=float, default=0.20)
    parser.add_argument("--causal-local-threshold", type=float, default=0.90)
    parser.add_argument("--invariance-samples", type=int, default=8)
    parser.add_argument("--invariance-seed-base", type=int, default=6189500)
    parser.add_argument("--output", default="experiments/level6_18_9/formal")
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
        print(json.dumps(result["success"], indent=2))
        return

    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, probe, baseline, source_read, source_state, selected_named = load_source(
        args, device
    )
    selected_parameters = [parameter for _, parameter in selected_named]
    optimizer = torch.optim.AdamW(
        selected_parameters, lr=args.read_lr,
        weight_decay=args.read_weight_decay,
    )
    baseline_panels = baseline_validation(baseline, args, device, dtype)
    save(root / "baseline_validation.json", compact_baseline(baseline_panels))
    capture = FinalReadCapture(model)
    try:
        training = train_read(
            model, probe, baseline_panels, source_read, capture,
            optimizer, selected_parameters, args, device, dtype, root,
        )
    finally:
        capture.close()

    audit = parameter_audit(source_state, model, probe)
    invariance = memory_invariance(baseline, model, args, device, dtype)
    if not training["passed"]:
        success = {
            "passed": False,
            "reason": "stable_task_aligned_read_gate_failed",
            "protected_tests_opened": False,
        }
        result = {
            "protocol": protocol,
            "baseline_validation": compact_baseline(baseline_panels),
            "read_training": training,
            "parameter_audit": audit,
            "memory_invariance": invariance,
            "success": success,
        }
        save(result_path, result)
        save(root / "summary.json", {
            "last_validation": training.get("screen"),
            "best_score": training.get("best_score"),
            "parameter_audit": audit,
            "memory_invariance": {
                "passed": invariance["passed"],
                "overall_max_abs_difference": invariance[
                    "overall_max_abs_difference"
                ],
            },
            "success": success,
        })
        plot_training(training, None, root / "read_supervision.png")
        print("Stable task-aligned read gate failed; protected tests were not opened.")
        return

    protected = {}
    protected_predictions = {}
    for count in TRAIN_CHUNKS:
        seed = args.test_seed_base + count
        baseline_item = evaluate_model(
            baseline, args, count, args.test_samples, seed,
            "intact", device, dtype,
        )
        rescued_item = evaluate_model(
            model, args, count, args.test_samples, seed,
            "intact", device, dtype,
        )
        protected[str(count)] = paired_test(
            baseline_item, rescued_item, args, count
        )
        protected_predictions[str(count)] = {
            "labels": baseline_item["labels"].tolist(),
            "baseline": baseline_item["predictions"].tolist(),
            "rescued": rescued_item["predictions"].tolist(),
            "baseline_margin": baseline_item["margins"].tolist(),
            "rescued_margin": rescued_item["margins"].tolist(),
        }
        print(
            f"test chunks={count} baseline={baseline_item['metric']['query']:.2%} "
            f"rescued={rescued_item['metric']['query']:.2%} "
            f"margin_change={protected[str(count)]['margin_change']['estimate']:+.4f}",
            flush=True,
        )

    causal_rows = {"baseline": {}, "rescued": {}}
    causal_predictions = {"baseline": {}, "rescued": {}}
    causal_labels = None
    for condition in CONDITIONS:
        baseline_item = evaluate_model(
            baseline, args, 16, args.causal_samples, args.causal_seed,
            condition, device, dtype,
        )
        rescued_item = evaluate_model(
            model, args, 16, args.causal_samples, args.causal_seed,
            condition, device, dtype,
        )
        if not torch.equal(baseline_item["labels"], rescued_item["labels"]):
            raise RuntimeError(f"Causal labels diverged for {condition}")
        if causal_labels is None:
            causal_labels = baseline_item["labels"]
        elif not torch.equal(causal_labels, baseline_item["labels"]):
            raise RuntimeError(f"Causal condition labels diverged for {condition}")
        causal_rows["baseline"][condition] = baseline_item["metric"]
        causal_rows["rescued"][condition] = rescued_item["metric"]
        causal_predictions["baseline"][condition] = baseline_item[
            "predictions"
        ].tolist()
        causal_predictions["rescued"][condition] = rescued_item[
            "predictions"
        ].tolist()
    causal = {
        "baseline": causal_summary(causal_rows["baseline"], args),
        "rescued": causal_summary(causal_rows["rescued"], args),
    }
    primary = protected["16"]
    success = {
        "stable_training_gate": training["passed"],
        "only_memory_read_changed": audit["passed"],
        "persistent_memory_exactly_invariant": invariance["passed"],
        "protected_8_accuracy": protected["8"]["rescued"]["query"] >= args.test_threshold,
        "protected_12_accuracy": protected["12"]["rescued"]["query"] >= args.test_threshold,
        "protected_16_accuracy": protected["16"]["rescued"]["query"] >= args.test_threshold,
        "positive_16_accuracy_ci": primary["accuracy"]["accuracy_change"]["ci95"][0] > 0,
        "positive_16_margin_ci": primary["margin_change"]["ci95"][0] > 0,
        "significant_16_margin": primary["margin_change"]["sign_flip_p_two_sided"] < 0.05,
        "rescued_causal_gate": causal["rescued"]["passed"],
        "protected_tests_opened": True,
    }
    success["passed"] = all(success.values())
    result = {
        "protocol": protocol,
        "baseline_validation": compact_baseline(baseline_panels),
        "read_training": training,
        "parameter_audit": audit,
        "memory_invariance": invariance,
        "protected_tests": protected,
        "causal": causal,
        "success": success,
    }
    save(result_path, result)
    save(root / "summary.json", {
        "training": {
            "update": training["update"],
            "best_score": training["best_score"],
            "confirmation": training["confirmation"],
        },
        "parameter_audit": audit,
        "memory_invariance": {
            "passed": invariance["passed"],
            "overall_max_abs_difference": invariance[
                "overall_max_abs_difference"
            ],
        },
        "protected_tests": protected,
        "causal": causal,
        "success": success,
    })
    save(root / "predictions.json", {
        "protected": protected_predictions,
        "causal_labels": causal_labels.tolist(),
        "causal": causal_predictions,
    })
    torch.save({
        "model": model.state_dict(),
        "probe": probe.state_dict(),
        "level6_18_9": {
            "read_training": training,
            "parameter_audit": audit,
            "memory_invariance": invariance,
            "success": success,
        },
    }, root / "task_aligned_read_checkpoint.pt")
    plot_training(training, protected, root / "read_supervision.png")
    print(json.dumps(success, indent=2))


if __name__ == "__main__":
    main()
