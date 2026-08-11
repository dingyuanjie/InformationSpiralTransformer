import argparse
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
from run_level6_9_local import CONDITIONS, intervene
from run_level6_13_1_local import bootstrap_mean_ci, mcnemar


SEED = 707
TRAIN_CHUNKS = 12
TEST_CHUNKS = [8, 12, 16]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_frozen(args, device):
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = state.get("phase_result", {})
    if meta.get("phase") != "bridge" or meta.get("eval_chunks") != TRAIN_CHUNKS:
        raise RuntimeError(
            "Expected the failed Level 6.18.1 8->12 bridge checkpoint; "
            f"found phase={meta.get('phase')} eval_chunks={meta.get('eval_chunks')}"
        )
    model, original_probe = build(device, args.chunk_size)
    model.load_state_dict(state["model"])
    original_probe.load_state_dict(state["probe"])
    model.eval()
    original_probe.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in original_probe.parameters():
        parameter.requires_grad_(False)
    return model, original_probe, meta


@torch.no_grad()
def collect_hidden(model, args, chunks_count, samples, seed, device, dtype):
    set_seed(seed)
    hidden = []
    labels = []
    predictions = []
    local_predictions = []
    captured = {}

    def capture_output_input(_module, inputs):
        captured["hidden"] = inputs[0]

    handle = model.output.register_forward_pre_hook(capture_output_input)
    total = 0
    try:
        while total < samples:
            batch = min(args.extract_batch_size, samples - total)
            chunks, target, position = make_chunks(
                batch, chunks_count, args.chunk_size, device
            )
            memory = None
            first_logits = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(chunks_count):
                    logits, memory = model(
                        chunks[:, chunk_index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    if chunk_index == 0:
                        first_logits = logits
            rows = torch.arange(batch, device=device)
            hidden.append(captured["hidden"][:, -1].detach().float().cpu())
            labels.append(target.cpu())
            predictions.append(logits[:, -1, :16].argmax(-1).cpu())
            local_predictions.append(
                first_logits[rows, position, :16].argmax(-1).cpu()
            )
            total += batch
    finally:
        handle.remove()

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    local_predictions = torch.cat(local_predictions)
    return {
        "hidden": torch.cat(hidden),
        "labels": labels,
        "predictions": predictions,
        "local_predictions": local_predictions,
        "metric": {
            "query": (predictions == labels).float().mean().item(),
            "local": (local_predictions == labels).float().mean().item(),
            "samples": len(labels),
        },
    }


def head_predictions(hidden, weight, bias, batch_size):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(hidden), batch_size):
            x = hidden[start:start + batch_size].float()
            predictions.append(F.linear(x, weight, bias).argmax(-1))
    return torch.cat(predictions)


def accuracy(predictions, labels):
    return (predictions == labels).float().mean().item()


def fit_rescue_head(train, validation_12, original_weight, original_bias,
                    args, device):
    set_seed(args.head_seed)
    train_x = train["hidden"]
    train_y = train["labels"]
    mean = train_x.mean(dim=0).to(device)
    std = train_x.std(dim=0).clamp_min(1e-4).to(device)
    head = nn.Linear(train_x.shape[-1], 16).to(device)

    # Initialize the standardized head to exactly reproduce the deployed raw head.
    with torch.no_grad():
        original_weight_device = original_weight.to(device)
        original_bias_device = original_bias.to(device)
        head.weight.copy_(original_weight_device * std[None, :])
        head.bias.copy_(original_bias_device + original_weight_device @ mean)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.head_lr,
        weight_decay=args.head_weight_decay,
    )
    validation_x = validation_12["hidden"]
    validation_y = validation_12["labels"]
    head.eval()
    initial_predictions = []
    with torch.no_grad():
        for start in range(0, len(validation_y), args.head_batch_size):
            x = validation_x[start:start + args.head_batch_size].to(device)
            x = (x - mean) / std
            initial_predictions.append(head(x).argmax(-1).cpu())
    best_accuracy = accuracy(torch.cat(initial_predictions), validation_y)
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())
    history = [{"epoch": 0, "validation_12": best_accuracy}]
    patience = 0

    for epoch in range(1, args.head_epochs + 1):
        head.train()
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), args.head_batch_size):
            ids = order[start:start + args.head_batch_size]
            x = train_x[ids].to(device)
            x = (x - mean) / std
            y = train_y[ids].to(device)
            loss = F.cross_entropy(head(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(validation_y), args.head_batch_size):
                x = validation_x[start:start + args.head_batch_size].to(device)
                x = (x - mean) / std
                predictions.append(head(x).argmax(-1).cpu())
        validation_accuracy = accuracy(torch.cat(predictions), validation_y)
        history.append({"epoch": epoch, "validation_12": validation_accuracy})
        if validation_accuracy > best_accuracy + 1e-5:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    head.load_state_dict(best_state)
    standardized_weight = head.weight.detach()
    standardized_bias = head.bias.detach()
    raw_weight = (standardized_weight / std[None, :]).cpu()
    raw_bias = (standardized_bias - raw_weight.to(device) @ mean).cpu()
    return {
        "weight": raw_weight,
        "bias": raw_bias,
        "best_epoch": best_epoch,
        "best_validation_12": best_accuracy,
        "epochs_run": history[-1]["epoch"],
        "history": history,
    }


def interpolate_head(original_weight, original_bias, fitted, alpha):
    return (
        original_weight.lerp(fitted["weight"], alpha),
        original_bias.lerp(fitted["bias"], alpha),
    )


def select_alpha(validation_8, validation_12, original_weight, original_bias,
                 fitted, args):
    candidates = []
    for alpha in args.alphas:
        weight, bias = interpolate_head(
            original_weight, original_bias, fitted, alpha
        )
        predictions_8 = head_predictions(
            validation_8["hidden"], weight, bias, args.head_batch_size
        )
        predictions_12 = head_predictions(
            validation_12["hidden"], weight, bias, args.head_batch_size
        )
        accuracy_8 = accuracy(predictions_8, validation_8["labels"])
        accuracy_12 = accuracy(predictions_12, validation_12["labels"])
        candidates.append({
            "alpha": alpha,
            "validation_8": accuracy_8,
            "validation_12": accuracy_12,
            "eligible": (
                accuracy_8 >= args.validation_retention_threshold
                and accuracy_12 >= args.validation_rescue_threshold
            ),
        })
    eligible = [item for item in candidates if item["eligible"]]
    if eligible:
        # Maximize 12-chunk rescue, then 8-chunk retention, then prefer less change.
        selected = max(
            eligible,
            key=lambda item: (
                item["validation_12"], item["validation_8"], -item["alpha"]
            ),
        )
    else:
        # Diagnostic fallback only; it cannot pass the formal selection gate.
        selected = max(
            candidates,
            key=lambda item: (
                min(item["validation_8"], item["validation_12"]),
                item["validation_12"], item["validation_8"], -item["alpha"],
            ),
        )
    return {
        "passed": bool(eligible),
        "selected": selected,
        "candidates": candidates,
    }


@torch.no_grad()
def evaluate_model_condition(model, args, chunks_count, condition, samples,
                             seed, device, dtype):
    set_seed(seed)
    labels = []
    predictions = []
    local_predictions = []
    total = 0
    while total < samples:
        batch = min(args.extract_batch_size, samples - total)
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
            "query": accuracy(predictions, labels),
            "local": accuracy(local_predictions, labels),
            "samples": len(labels),
        },
    }


def paired_result(baseline, rescued, labels, args, chunks_count):
    baseline_correct = (baseline == labels).numpy().astype(np.int8)
    rescued_correct = (rescued == labels).numpy().astype(np.int8)
    improvement = rescued_correct - baseline_correct
    return {
        "chunks": chunks_count,
        "samples": len(labels),
        "baseline_accuracy": float(baseline_correct.mean()),
        "rescued_accuracy": float(rescued_correct.mean()),
        "accuracy_change": bootstrap_mean_ci(
            improvement,
            args.bootstrap_seed + chunks_count,
            args.bootstrap_iterations,
        ),
        "mcnemar_rescued_vs_baseline": mcnemar(
            rescued_correct, baseline_correct
        ),
        "corrected": int(np.sum(improvement == 1)),
        "harmed": int(np.sum(improvement == -1)),
        "unchanged": int(np.sum(improvement == 0)),
    }


def verify_only_output_changed(before, model):
    changed = []
    max_non_output_change = 0.0
    for name, value in model.state_dict().items():
        difference = (value.detach().cpu() - before[name]).abs().max().item()
        if difference > 0:
            changed.append({"name": name, "max_abs_change": difference})
        if not name.startswith("output."):
            max_non_output_change = max(max_non_output_change, difference)
    tail_weight_change = (
        model.output.weight.detach().cpu()[16:] - before["output.weight"][16:]
    ).abs().max().item()
    tail_bias_change = (
        model.output.bias.detach().cpu()[16:] - before["output.bias"][16:]
    ).abs().max().item()
    return {
        "changed_tensors": changed,
        "max_non_output_change": max_non_output_change,
        "unused_output_rows_max_change": max(tail_weight_change, tail_bias_change),
        "passed": (
            max_non_output_change == 0.0
            and tail_weight_change == 0.0
            and tail_bias_change == 0.0
            and {item["name"] for item in changed}
            <= {"output.weight", "output.bias"}
        ),
    }


def plot_results(validation_selection, paired, causal, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    chunks = [8, 12, 16]
    baseline = [100 * paired[str(count)]["baseline_accuracy"] for count in chunks]
    rescued = [100 * paired[str(count)]["rescued_accuracy"] for count in chunks]
    positions = np.arange(len(chunks))
    width = 0.36
    axes[0].bar(positions - width / 2, baseline, width, label="Untouched",
                color="#d1495b")
    axes[0].bar(positions + width / 2, rescued, width, label="Head-only rescue",
                color="#2e6fbb")
    axes[0].axhline(95, color="#333333", linestyle="--", linewidth=1.4)
    axes[0].set_xticks(positions, [f"{count} chunks" for count in chunks])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Held-out query accuracy (%)")
    axes[0].set_title("Protected cross-length tests")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    candidates = validation_selection["candidates"]
    alphas = [item["alpha"] for item in candidates]
    axes[1].plot(alphas, [100 * item["validation_8"] for item in candidates],
                 marker="o", label="8-chunk retention")
    axes[1].plot(alphas, [100 * item["validation_12"] for item in candidates],
                 marker="o", label="12-chunk rescue")
    selected = validation_selection["selected"]
    axes[1].axvline(selected["alpha"], color="#5b3a91", linestyle="--",
                    label=f"selected α={selected['alpha']:.1f}")
    axes[1].axhline(95, color="#333333", linestyle=":", linewidth=1.4)
    axes[1].set_xlabel("Interpolation dose α")
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title(
        f"Validation dose curve; causal drop={100 * causal['query_drop']:.1f} pp"
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("IST Level 6.18.3: Surgical Output-Head Rescue", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.18.3",
        "hypothesis": (
            "A head-only intervention can expose the linearly decodable "
            "12-chunk query representation while preserving the 8-chunk circuit."
        ),
        "source": "seed707 Level 6.18.1 failed 8->12 diagnostic-best checkpoint",
        "trainable_parameters": "first 16 rows of model.output only",
        "training_length": TRAIN_CHUNKS,
        "validation_selection": {
            "alpha_grid": args.alphas,
            "eligible": "8-chunk >= 0.95 and 12-chunk >= 0.95",
            "ranking": "max 12-chunk, then 8-chunk, then smaller alpha",
            "16_chunk_is_never_used": True,
        },
        "protected_tests": TEST_CHUNKS,
        "primary_success": {
            "selection_gate": True,
            "8_chunk_rescued_query": ">= 0.95",
            "12_chunk_rescued_query": ">= 0.95",
            "12_chunk_improvement_ci95_lower": "> 0",
            "only_output_changed": True,
            "rescued_causal_gate": True,
        },
        "causal_gate": {
            "intact_12": ">= 0.95",
            "max_reset_zero_batch_roll": "<= 0.20",
            "min_local": ">= 0.90",
        },
        "sixteen_chunk_result": "protected exploratory transfer; not a success gate",
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED:
        raise ValueError("Level 6.18.3 is fixed to seed707")
    if args.test_chunks != TEST_CHUNKS:
        raise ValueError("Protected test lengths are fixed as 8, 12, 16 chunks")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    if sorted(set(args.alphas)) != ALPHAS:
        raise ValueError("Formal interpolation doses are fixed from 0.0 to 1.0 by 0.1")
    if args.extract_batch_size < 2:
        raise ValueError("batch-roll requires extract-batch-size >= 2")


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.3 surgical output-head rescue"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-chunks", nargs="+", type=int, default=TEST_CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/level6_18_1/formal/seed707/"
            "transition_8_to_16_bridge_best.pt"
        ),
    )
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-seed", type=int, default=6183100)
    parser.add_argument("--dataset-seed-base", type=int, default=6183000)
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--validation-retention-threshold", type=float, default=0.95)
    parser.add_argument("--validation-rescue-threshold", type=float, default=0.95)
    parser.add_argument("--test-retention-threshold", type=float, default=0.95)
    parser.add_argument("--test-rescue-threshold", type=float, default=0.95)
    parser.add_argument("--causal-intervention-threshold", type=float, default=0.20)
    parser.add_argument("--causal-local-threshold", type=float, default=0.90)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=6183200)
    parser.add_argument("--output", default="experiments/level6_18_3/formal")
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

    model, original_probe, source_meta = load_frozen(args, device)
    original_probe_state = {
        name: value.detach().cpu().clone()
        for name, value in original_probe.state_dict().items()
    }
    del original_probe
    before = {name: value.detach().cpu().clone()
              for name, value in model.state_dict().items()}
    original_weight = model.output.weight.detach().cpu()[:16].clone()
    original_bias = model.output.bias.detach().cpu()[:16].clone()

    base = args.dataset_seed_base + args.seed * 100
    train = collect_hidden(
        model, args, TRAIN_CHUNKS, args.train_samples,
        base + 1, device, dtype,
    )
    validation = {
        count: collect_hidden(
            model, args, count, args.validation_samples,
            base + count * 10 + 2, device, dtype,
        ) for count in [8, 12]
    }
    tests = {
        count: collect_hidden(
            model, args, count, args.test_samples,
            base + count * 10 + 3, device, dtype,
        ) for count in args.test_chunks
    }

    fitted = fit_rescue_head(
        train, validation[12], original_weight, original_bias, args, device
    )
    selection = select_alpha(
        validation[8], validation[12], original_weight, original_bias,
        fitted, args,
    )
    alpha = selection["selected"]["alpha"]
    selected_weight, selected_bias = interpolate_head(
        original_weight, original_bias, fitted, alpha
    )
    with torch.no_grad():
        model.output.weight[:16].copy_(selected_weight.to(device))
        model.output.bias[:16].copy_(selected_bias.to(device))
    mutation = verify_only_output_changed(before, model)

    paired = {}
    test_predictions = {}
    rescued_test = {}
    for count in args.test_chunks:
        rescued = evaluate_model_condition(
            model, args, count, "intact", args.test_samples,
            base + count * 10 + 3, device, dtype,
        )
        if not torch.equal(rescued["labels"], tests[count]["labels"]):
            raise RuntimeError(f"paired labels diverged at chunks={count}")
        paired[str(count)] = paired_result(
            tests[count]["predictions"], rescued["predictions"],
            rescued["labels"], args, count,
        )
        paired[str(count)]["baseline_local"] = tests[count]["metric"]["local"]
        paired[str(count)]["rescued_local"] = rescued["metric"]["local"]
        rescued_test[count] = rescued
        test_predictions[str(count)] = {
            "labels": rescued["labels"].tolist(),
            "baseline": tests[count]["predictions"].tolist(),
            "rescued": rescued["predictions"].tolist(),
        }
        print(
            f"chunks={count} baseline={paired[str(count)]['baseline_accuracy']:.2%} "
            f"rescued={paired[str(count)]['rescued_accuracy']:.2%}",
            flush=True,
        )

    causal_conditions = {"intact": rescued_test[12]["metric"]}
    causal_predictions = {"intact": rescued_test[12]["predictions"].tolist()}
    causal_seed = base + 12 * 10 + 30
    # Use a separate protected causal dataset; intact is evaluated afresh below.
    intact_causal = evaluate_model_condition(
        model, args, 12, "intact", args.causal_samples,
        causal_seed, device, dtype,
    )
    causal_conditions["intact"] = intact_causal["metric"]
    causal_predictions["intact"] = intact_causal["predictions"].tolist()
    causal_labels = intact_causal["labels"]
    for condition in CONDITIONS[1:]:
        item = evaluate_model_condition(
            model, args, 12, condition, args.causal_samples,
            causal_seed, device, dtype,
        )
        if not torch.equal(item["labels"], causal_labels):
            raise RuntimeError(f"causal labels diverged for condition={condition}")
        causal_conditions[condition] = item["metric"]
        causal_predictions[condition] = item["predictions"].tolist()
        print(
            f"causal={condition} query={item['metric']['query']:.2%} "
            f"local={item['metric']['local']:.2%}",
            flush=True,
        )
    strongest_disrupted = max(
        causal_conditions[name]["query"] for name in CONDITIONS[1:]
    )
    causal = {
        "conditions": causal_conditions,
        "strongest_disrupted_query": strongest_disrupted,
        "query_drop": causal_conditions["intact"]["query"] - strongest_disrupted,
        "passed": (
            causal_conditions["intact"]["query"] >= args.test_rescue_threshold
            and strongest_disrupted <= args.causal_intervention_threshold
            and min(causal_conditions[name]["local"] for name in CONDITIONS)
            >= args.causal_local_threshold
        ),
    }

    primary_12 = paired["12"]
    success = {
        "selection_gate": selection["passed"],
        "only_output_changed": mutation["passed"],
        "retained_8_chunks": (
            paired["8"]["rescued_accuracy"] >= args.test_retention_threshold
        ),
        "rescued_12_chunks": (
            primary_12["rescued_accuracy"] >= args.test_rescue_threshold
        ),
        "positive_12_chunk_ci": (
            primary_12["accuracy_change"]["ci95"][0] > 0
        ),
        "causal_gate": causal["passed"],
    }
    success["passed"] = all(success.values())

    fitted_serializable = {
        key: value for key, value in fitted.items()
        if key not in {"weight", "bias"}
    }
    result = {
        "protocol": protocol,
        "source_checkpoint": {
            "path": args.checkpoint,
            "phase": source_meta.get("phase"),
            "step": source_meta.get("step"),
            "passed_level6_18_1_gate": source_meta.get("passed"),
        },
        "head_fit": fitted_serializable,
        "validation_selection": selection,
        "mutation_audit": mutation,
        "paired_tests": paired,
        "causal": causal,
        "success": success,
    }
    save(root / "result.json", result)
    save(root / "summary.json", {
        "selected_alpha": alpha,
        "paired_tests": paired,
        "causal": causal,
        "success": success,
    })
    save(root / "predictions.json", {
        "protected_tests": test_predictions,
        "causal_labels": causal_labels.tolist(),
        "causal_rescued": causal_predictions,
    })
    torch.save({
        "model": model.state_dict(),
        "probe": original_probe_state,
        "level6_18_3": {
            "selected_alpha": alpha,
            "validation_selection": selection,
            "success": success,
        },
    }, root / "rescued_head_checkpoint.pt")
    plot_results(selection, paired, causal, root / "head_only_rescue.png")
    print(json.dumps(success, indent=2))


if __name__ == "__main__":
    main()
