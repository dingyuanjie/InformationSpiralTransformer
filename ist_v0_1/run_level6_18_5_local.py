import argparse
import json
import os
import random
import time
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
from run_level6_6_local import build, checkpoint, restore
from run_level6_9_local import CONDITIONS, intervene
from run_level6_13_1_local import bootstrap_mean_ci, mcnemar


SEED = 707
TRAIN_CHUNKS = 16
PROTECTED_CHUNKS = [8, 12, 16]
ROUTE_PREFIXES = (
    "blocks.2.memory_read.",
    "blocks.2.memory_fusion_gate.",
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, model, probe, optimizer, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    checkpoint(temporary, model, probe, optimizer, payload)
    last_error = None
    for _ in range(10):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.1)
    raise last_error


def route_parameters(model):
    selected = [
        (name, parameter) for name, parameter in model.named_parameters()
        if name.startswith(ROUTE_PREFIXES)
    ]
    if len(selected) != 6 or sum(item.numel() for _, item in selected) != 24896:
        raise RuntimeError(
            f"Unexpected routing boundary: {[(name, tuple(p.shape)) for name, p in selected]}"
        )
    return selected


def load_source(args, device):
    state = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    meta = state.get("level6_18_3", {})
    if not meta.get("success", {}).get("passed"):
        raise RuntimeError("Source Level 6.18.3 checkpoint did not pass formally")
    model, probe = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    probe.load_state_dict(state["probe"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    selected = route_parameters(model)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    return model, probe, state, selected


@torch.no_grad()
def evaluate_predictions(model, args, chunks_count, samples, seed,
                         condition, device, dtype):
    set_seed(seed)
    labels = []
    predictions = []
    local_predictions = []
    total = 0
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
            "chunks": chunks_count,
            "query": (predictions == labels).float().mean().item(),
            "local": (local_predictions == labels).float().mean().item(),
            "samples": len(labels),
        },
    }


def preserved_evaluate(model, args, chunks_count, samples, seed, device, dtype):
    python_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    try:
        return evaluate_predictions(
            model, args, chunks_count, samples, seed, "intact", device, dtype
        )["metric"]
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_state)


def validation(model, args, seed, device, dtype, confirmation=False):
    samples = args.confirm_samples if confirmation else args.screen_samples
    seed_base = args.validation_seed_base + seed * 1000
    return {
        str(count): preserved_evaluate(
            model, args, count, samples,
            seed_base + count * 10 + (1 if confirmation else 0),
            device, dtype,
        ) for count in PROTECTED_CHUNKS
    }


def screen_candidate(metrics, args):
    return (
        metrics["8"]["query"] >= args.screen_retention_threshold
        and metrics["12"]["query"] >= args.screen_retention_threshold
        and metrics["16"]["query"] >= args.screen_rescue_threshold
    )


def confirmed_candidate(metrics, args):
    return all(
        metrics[str(count)]["query"] >= args.confirm_threshold
        for count in PROTECTED_CHUNKS
    )


def train_update(model, optimizer, selected_parameters, args, device, dtype):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    accuracy_total = 0.0
    for _ in range(args.gradient_accumulation):
        chunks, target, _ = make_chunks(
            args.train_batch_size, TRAIN_CHUNKS, args.chunk_size, device
        )
        memory = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for chunk_index in range(TRAIN_CHUNKS):
                logits, memory = model(
                    chunks[:, chunk_index], memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
            loss = F.cross_entropy(logits[:, -1, :16], target)
            scaled = loss / args.gradient_accumulation
        scaled.backward()
        loss_total += loss.detach().float().item()
        accuracy_total += (
            logits[:, -1, :16].argmax(-1) == target
        ).float().mean().item()
    torch.nn.utils.clip_grad_norm_(selected_parameters, args.gradient_clip)
    optimizer.step()
    return {
        "loss": loss_total / args.gradient_accumulation,
        "query": accuracy_total / args.gradient_accumulation,
    }


def train_routing(model, probe, optimizer, selected_parameters, args,
                  device, dtype, root):
    latest_path = root / "routing_latest.pt"
    best_path = root / "routing_best.pt"
    stable_path = root / "routing_stable.pt"
    history = []
    start_update = 0
    stable_streak = 0
    best_score = None

    if stable_path.exists() and not args.force:
        state = restore(stable_path, model, probe, optimizer, device)
        return state["routing_training"]
    if latest_path.exists() and not args.force:
        state = restore(latest_path, model, probe, optimizer, device)
        meta = state["routing_training"]
        history = meta["history"]
        start_update = meta["update"]
        stable_streak = meta["stable_streak"]
        best_score = meta.get("best_score")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    for parameter in selected_parameters:
        parameter.requires_grad_(True)
    for group in optimizer.param_groups:
        group["lr"] = args.routing_lr

    last_train = None
    for update in range(start_update + 1, args.routing_updates + 1):
        last_train = train_update(
            model, optimizer, selected_parameters, args, device, dtype
        )
        should_evaluate = (
            update == 1
            or update % args.eval_every_updates == 0
            or update == args.routing_updates
        )
        if not should_evaluate:
            continue
        screen = validation(model, args, args.seed, device, dtype, False)
        confirmation = None
        if screen_candidate(screen, args):
            confirmation = validation(model, args, args.seed, device, dtype, True)
        confirmed = bool(confirmation and confirmed_candidate(confirmation, args))
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
        save(root / "routing_progress.json", history)

        score = None
        if confirmation:
            score = [
                confirmation["16"]["query"],
                min(confirmation["8"]["query"], confirmation["12"]["query"]),
                sum(confirmation[str(count)]["query"] for count in PROTECTED_CHUNKS) / 3,
            ]
        is_best = confirmed and (
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
            latest_path, model, probe, optimizer, {"routing_training": meta}
        )
        if is_best:
            atomic_checkpoint(
                best_path, model, probe, optimizer, {"routing_training": meta}
            )
        print(
            f"update={update} train={last_train['query']:.2%} "
            f"screen8={screen['8']['query']:.2%} "
            f"screen12={screen['12']['query']:.2%} "
            f"screen16={screen['16']['query']:.2%} "
            f"stable={stable_streak}/{args.stable_confirmations}",
            flush=True,
        )
        if stable_streak >= args.stable_confirmations:
            atomic_checkpoint(
                stable_path, model, probe, optimizer, {"routing_training": meta}
            )
            return meta

    return {
        "update": args.routing_updates,
        "stable_streak": stable_streak,
        "passed": False,
        "best_score": best_score,
        "last_train": last_train,
        "screen": history[-1]["screen"] if history else None,
        "confirmation": history[-1]["confirmation"] if history else None,
        "history": history,
    }


def parameter_audit(source_state, model):
    allowed = {
        name for name in source_state["model"]
        if name.startswith(ROUTE_PREFIXES)
    }
    changed = []
    max_disallowed_change = 0.0
    for name, current in model.state_dict().items():
        difference = (
            current.detach().cpu() - source_state["model"][name].detach().cpu()
        ).abs().max().item()
        if difference > 0:
            changed.append({"name": name, "max_abs_change": difference})
        if name not in allowed:
            max_disallowed_change = max(max_disallowed_change, difference)
    changed_names = {item["name"] for item in changed}
    return {
        "allowed_tensors": sorted(allowed),
        "changed_tensors": changed,
        "max_disallowed_change": max_disallowed_change,
        "passed": max_disallowed_change == 0.0 and bool(changed_names)
                  and changed_names <= allowed,
    }


@torch.no_grad()
def memory_invariance(baseline, rescued, args, device, dtype):
    results = {}
    overall = 0.0
    for count in PROTECTED_CHUNKS:
        set_seed(args.invariance_seed_base + count)
        chunks, _, _ = make_chunks(
            args.invariance_samples, count, args.chunk_size, device
        )
        baseline_memory = None
        rescued_memory = None
        maximum = 0.0
        by_chunk = []
        with torch.autocast(device_type="cuda", dtype=dtype):
            for chunk_index in range(count):
                _, baseline_memory = baseline(
                    chunks[:, chunk_index], memory=baseline_memory,
                    return_memory=True, per_layer_memory=True,
                )
                _, rescued_memory = rescued(
                    chunks[:, chunk_index], memory=rescued_memory,
                    return_memory=True, per_layer_memory=True,
                )
                layer_differences = [
                    (left - right).abs().max().float().item()
                    for left, right in zip(baseline_memory, rescued_memory)
                ]
                chunk_maximum = max(layer_differences)
                by_chunk.append({
                    "chunk": chunk_index + 1,
                    "layer_max_abs": layer_differences,
                    "max_abs": chunk_maximum,
                })
                maximum = max(maximum, chunk_maximum)
        results[str(count)] = {
            "samples": args.invariance_samples,
            "max_abs_difference": maximum,
            "by_chunk": by_chunk,
        }
        overall = max(overall, maximum)
    return {
        "by_length": results,
        "overall_max_abs_difference": overall,
        "passed": overall == 0.0,
    }


def paired_result(baseline, rescued, labels, args, count):
    baseline_correct = (baseline == labels).numpy().astype(np.int8)
    rescued_correct = (rescued == labels).numpy().astype(np.int8)
    change = rescued_correct - baseline_correct
    return {
        "chunks": count,
        "samples": len(labels),
        "baseline_accuracy": float(baseline_correct.mean()),
        "rescued_accuracy": float(rescued_correct.mean()),
        "accuracy_change": bootstrap_mean_ci(
            change, args.bootstrap_seed + count, args.bootstrap_iterations
        ),
        "mcnemar_rescued_vs_baseline": mcnemar(
            rescued_correct, baseline_correct
        ),
        "corrected": int(np.sum(change == 1)),
        "harmed": int(np.sum(change == -1)),
    }


def causal_summary(conditions, args):
    disrupted = max(conditions[name]["query"] for name in CONDITIONS[1:])
    return {
        "intact_query": conditions["intact"]["query"],
        "strongest_disrupted_query": disrupted,
        "query_drop": conditions["intact"]["query"] - disrupted,
        "minimum_local": min(conditions[name]["local"] for name in CONDITIONS),
        "passed": (
            conditions["intact"]["query"] >= args.test_threshold
            and disrupted <= args.causal_intervention_threshold
            and min(conditions[name]["local"] for name in CONDITIONS)
            >= args.causal_local_threshold
        ),
    }


def plot_result(training, paired, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    counts = PROTECTED_CHUNKS
    positions = np.arange(len(counts))
    width = 0.36
    baseline = [100 * paired[str(count)]["baseline_accuracy"] for count in counts]
    rescued = [100 * paired[str(count)]["rescued_accuracy"] for count in counts]
    axes[0].bar(positions - width / 2, baseline, width,
                label="Level 6.18.3 head", color="#d1495b")
    axes[0].bar(positions + width / 2, rescued, width,
                label="Routing rescue", color="#2e6fbb")
    axes[0].axhline(95, color="#333333", linestyle="--")
    axes[0].set_xticks(positions, [f"{count} chunks" for count in counts])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Protected query accuracy (%)")
    axes[0].set_title("Frozen-memory protected tests")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    history = training["history"]
    updates = [row["update"] for row in history]
    for count, color in zip(counts, ["#59a14f", "#edae49", "#5b3a91"]):
        axes[1].plot(
            updates,
            [100 * row["screen"][str(count)]["query"] for row in history],
            marker="o", markersize=3, label=f"{count}-chunk screen",
            color=color,
        )
    axes[1].axhline(95, color="#333333", linestyle="--")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Routing optimizer update")
    axes[1].set_ylabel("Fixed-screen accuracy (%)")
    axes[1].set_title("Routing-only formation trajectory")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("IST Level 6.18.5: Surgical Memory-Read Routing Rescue", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.18.5",
        "hypothesis": (
            "The residual 16-chunk deficit is caused by the final block's "
            "Memory-to-token read path and can be rescued without changing Memory states."
        ),
        "source": "formally passed Level 6.18.3 head-rescued seed707 checkpoint",
        "trainable_boundary": {
            "prefixes": ROUTE_PREFIXES,
            "parameter_tensors": 6,
            "parameters": 24896,
        },
        "training": {
            "length": TRAIN_CHUNKS,
            "loss": "query cross-entropy only",
            "optimizer_updates": args.routing_updates,
            "gradient_accumulation": args.gradient_accumulation,
        },
        "stable_gate": {
            "protected_lengths": PROTECTED_CHUNKS,
            "confirmation_query": args.confirm_threshold,
            "successive_confirmations": args.stable_confirmations,
        },
        "primary_success": {
            "stable_training_gate": True,
            "8_12_16_test_query": f">= {args.test_threshold}",
            "16_improvement_ci95_lower": "> 0",
            "only_route_parameters_changed": True,
            "returned_memory_exactly_invariant": True,
            "16_chunk_causal_gate": True,
        },
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED:
        raise ValueError("Level 6.18.5 is fixed to seed707")
    if not Path(args.source_checkpoint).exists():
        raise FileNotFoundError(args.source_checkpoint)
    if args.eval_batch_size < 2 or args.invariance_samples < 2:
        raise ValueError("batch-roll and invariance checks require batch size >= 2")
    if args.invariance_samples > args.eval_batch_size:
        raise ValueError("invariance-samples must fit in one evaluation batch")


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.5 surgical final-block Memory-read routing rescue"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--source-checkpoint",
        default="experiments/level6_18_3/formal/rescued_head_checkpoint.pt",
    )
    parser.add_argument("--routing-updates", type=int, default=500)
    parser.add_argument("--routing-lr", type=float, default=5e-5)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-every-updates", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--screen-samples", type=int, default=80)
    parser.add_argument("--confirm-samples", type=int, default=256)
    parser.add_argument("--screen-retention-threshold", type=float, default=0.94)
    parser.add_argument("--screen-rescue-threshold", type=float, default=0.92)
    parser.add_argument("--confirm-threshold", type=float, default=0.95)
    parser.add_argument("--stable-confirmations", type=int, default=2)
    parser.add_argument("--training-seed", type=int, default=6185000)
    parser.add_argument("--validation-seed-base", type=int, default=6185100)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--test-seed-base", type=int, default=6185200)
    parser.add_argument("--test-threshold", type=float, default=0.95)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--causal-seed", type=int, default=6185300)
    parser.add_argument("--causal-intervention-threshold", type=float, default=0.20)
    parser.add_argument("--causal-local-threshold", type=float, default=0.90)
    parser.add_argument("--invariance-samples", type=int, default=8)
    parser.add_argument("--invariance-seed-base", type=int, default=6185400)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=6185500)
    parser.add_argument("--output", default="experiments/level6_18_5/formal")
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model, probe, source_state, selected_named = load_source(args, device)
    selected_parameters = [parameter for _, parameter in selected_named]
    optimizer = torch.optim.AdamW(selected_parameters, lr=args.routing_lr)
    set_seed(args.training_seed)
    training = train_routing(
        model, probe, optimizer, selected_parameters, args, device, dtype, root
    )
    if not training["passed"]:
        result = {
            "protocol": protocol,
            "routing_training": training,
            "success": {"passed": False, "reason": "stable_routing_gate_failed"},
        }
        save(result_path, result)
        print("Stable routing gate failed; protected tests were not opened.")
        return

    baseline, baseline_probe, _, _ = load_source(args, device)
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    del baseline_probe
    model.eval()
    baseline.eval()
    audit = parameter_audit(source_state, model)
    invariance = memory_invariance(baseline, model, args, device, dtype)

    paired = {}
    protected_predictions = {}
    test_rows = {}
    for count in PROTECTED_CHUNKS:
        seed = args.test_seed_base + count
        baseline_item = evaluate_predictions(
            baseline, args, count, args.test_samples, seed,
            "intact", device, dtype,
        )
        rescued_item = evaluate_predictions(
            model, args, count, args.test_samples, seed,
            "intact", device, dtype,
        )
        if not torch.equal(baseline_item["labels"], rescued_item["labels"]):
            raise RuntimeError(f"Protected labels diverged at chunks={count}")
        paired[str(count)] = paired_result(
            baseline_item["predictions"], rescued_item["predictions"],
            rescued_item["labels"], args, count,
        )
        paired[str(count)]["baseline_local"] = baseline_item["metric"]["local"]
        paired[str(count)]["rescued_local"] = rescued_item["metric"]["local"]
        test_rows[str(count)] = {
            "baseline": baseline_item["metric"],
            "rescued": rescued_item["metric"],
        }
        protected_predictions[str(count)] = {
            "labels": rescued_item["labels"].tolist(),
            "baseline": baseline_item["predictions"].tolist(),
            "rescued": rescued_item["predictions"].tolist(),
        }
        print(
            f"test chunks={count} baseline={paired[str(count)]['baseline_accuracy']:.2%} "
            f"rescued={paired[str(count)]['rescued_accuracy']:.2%}",
            flush=True,
        )

    causal_rows = {"baseline": {}, "rescued": {}}
    causal_predictions = {"baseline": {}, "rescued": {}}
    causal_labels = None
    for condition in CONDITIONS:
        baseline_item = evaluate_predictions(
            baseline, args, 16, args.causal_samples, args.causal_seed,
            condition, device, dtype,
        )
        rescued_item = evaluate_predictions(
            model, args, 16, args.causal_samples, args.causal_seed,
            condition, device, dtype,
        )
        if not torch.equal(baseline_item["labels"], rescued_item["labels"]):
            raise RuntimeError(f"Causal model labels diverged for {condition}")
        if causal_labels is None:
            causal_labels = rescued_item["labels"]
        elif not torch.equal(causal_labels, rescued_item["labels"]):
            raise RuntimeError(f"Causal condition labels diverged for {condition}")
        causal_rows["baseline"][condition] = baseline_item["metric"]
        causal_rows["rescued"][condition] = rescued_item["metric"]
        causal_predictions["baseline"][condition] = baseline_item["predictions"].tolist()
        causal_predictions["rescued"][condition] = rescued_item["predictions"].tolist()
        print(
            f"causal={condition} baseline={baseline_item['metric']['query']:.2%} "
            f"rescued={rescued_item['metric']['query']:.2%}",
            flush=True,
        )
    causal = {
        "conditions": causal_rows,
        "baseline": causal_summary(causal_rows["baseline"], args),
        "rescued": causal_summary(causal_rows["rescued"], args),
    }

    primary = paired["16"]
    success = {
        "stable_training_gate": training["passed"],
        "only_route_parameters_changed": audit["passed"],
        "memory_states_exactly_invariant": invariance["passed"],
        "retained_8_chunks": paired["8"]["rescued_accuracy"] >= args.test_threshold,
        "retained_12_chunks": paired["12"]["rescued_accuracy"] >= args.test_threshold,
        "rescued_16_chunks": paired["16"]["rescued_accuracy"] >= args.test_threshold,
        "positive_16_chunk_ci": primary["accuracy_change"]["ci95"][0] > 0,
        "causal_gate": causal["rescued"]["passed"],
    }
    success["passed"] = all(success.values())
    result = {
        "protocol": protocol,
        "routing_training": training,
        "parameter_audit": audit,
        "memory_invariance": invariance,
        "protected_tests": test_rows,
        "paired_tests": paired,
        "causal": causal,
        "success": success,
    }
    save(root / "result.json", result)
    save(root / "summary.json", {
        "training": {
            "update": training["update"],
            "best_score": training["best_score"],
            "confirmation": training["confirmation"],
        },
        "parameter_audit": audit,
        "memory_invariance": {
            "overall_max_abs_difference": invariance["overall_max_abs_difference"],
            "passed": invariance["passed"],
        },
        "paired_tests": paired,
        "causal": causal,
        "success": success,
    })
    save(root / "predictions.json", {
        "protected_tests": protected_predictions,
        "causal_labels": causal_labels.tolist(),
        "causal": causal_predictions,
    })
    torch.save({
        "model": model.state_dict(),
        "probe": probe.state_dict(),
        "level6_18_5": {
            "routing_training": training,
            "parameter_audit": audit,
            "memory_invariance": {
                "overall_max_abs_difference": invariance["overall_max_abs_difference"],
                "passed": invariance["passed"],
            },
            "success": success,
        },
    }, root / "routing_rescued_checkpoint.pt")
    plot_result(training, paired, root / "routing_rescue.png")
    print(json.dumps(success, indent=2))


if __name__ == "__main__":
    main()
