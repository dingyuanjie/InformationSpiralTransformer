import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from long_context_test import set_seed
from run_level6_2_local import evaluate, make_chunks, vector
from run_level6_6_local import build
from run_level6_9_local import CONDITIONS, intervene


SEED = 707
CHUNKS = 12
FEATURE_KINDS = [
    "memory_l3_mean",
    "memory_l3_concat",
    "memory_all_mean_concat",
    "memory_all_concat",
    "query_hidden",
    "memory_l3_query",
    "memory_all_query",
]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_frozen(args, device):
    checkpoint_path = Path(args.checkpoint)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = state.get("phase_result", {})
    if meta.get("phase") != "bridge" or meta.get("eval_chunks") != args.chunks:
        raise RuntimeError(
            "Expected the Level 6.18.1 8->12 bridge checkpoint; "
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
def collect_intact(model, original_probe, args, samples, seed, device, dtype):
    set_seed(seed)
    states = []
    query_hidden = []
    labels = []
    query_predictions = []
    original_predictions = []
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
                batch, args.chunks, args.chunk_size, device
            )
            memory = None
            first_logits = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(args.chunks):
                    logits, memory = model(
                        chunks[:, chunk_index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    if chunk_index == 0:
                        first_logits = logits
                original_logits = original_probe(vector(memory))
            rows = torch.arange(batch, device=device)
            states.append(
                torch.stack(memory, dim=1).detach().cpu().to(torch.float16)
            )
            query_hidden.append(
                captured["hidden"][:, -1].detach().cpu().to(torch.float16)
            )
            labels.append(target.cpu())
            query_predictions.append(logits[:, -1, :16].argmax(-1).cpu())
            original_predictions.append(original_logits.argmax(-1).cpu())
            local_predictions.append(
                first_logits[rows, position, :16].argmax(-1).cpu()
            )
            total += batch
    finally:
        handle.remove()

    labels = torch.cat(labels)
    query_predictions = torch.cat(query_predictions)
    original_predictions = torch.cat(original_predictions)
    local_predictions = torch.cat(local_predictions)
    return {
        "states": torch.cat(states),
        "query_hidden": torch.cat(query_hidden),
        "labels": labels,
        "query_predictions": query_predictions,
        "original_predictions": original_predictions,
        "local_predictions": local_predictions,
        "behavior": {
            "query": (query_predictions == labels).float().mean().item(),
            "original_probe": (
                original_predictions == labels
            ).float().mean().item(),
            "local": (local_predictions == labels).float().mean().item(),
            "samples": len(labels),
        },
    }


def feature_views(dataset):
    states = dataset["states"]
    query = dataset["query_hidden"]
    samples, layers, slots, width = states.shape
    layer3 = states[:, -1]
    all_concat = states.reshape(samples, layers * slots * width)
    layer3_concat = layer3.reshape(samples, slots * width)
    return {
        "memory_l3_mean": layer3.mean(dim=1),
        "memory_l3_concat": layer3_concat,
        "memory_all_mean_concat": states.mean(dim=2).reshape(samples, layers * width),
        "memory_all_concat": all_concat,
        "query_hidden": query,
        "memory_l3_query": torch.cat([layer3_concat, query], dim=-1),
        "memory_all_query": torch.cat([all_concat, query], dim=-1),
    }


def batched_predictions(model, x, mean, std, batch_size, device):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].to(device, torch.float32)
            batch = (batch - mean) / std
            predictions.append(model(batch).argmax(-1).cpu())
    return torch.cat(predictions)


def fit_linear(train_x, train_y, val_x, val_y, test_x, test_y,
               args, device, seed):
    set_seed(seed)
    mean = train_x.float().mean(dim=0).to(device)
    std = train_x.float().std(dim=0).clamp_min(1e-4).to(device)
    model = nn.Linear(train_x.shape[-1], 16).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    patience = 0

    for epoch in range(1, args.probe_epochs + 1):
        model.train()
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), args.probe_batch_size):
            ids = order[start:start + args.probe_batch_size]
            x = train_x[ids].to(device, torch.float32)
            x = (x - mean) / std
            y = train_y[ids].to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        val_predictions = batched_predictions(
            model, val_x, mean, std, args.probe_batch_size, device
        )
        accuracy = (val_predictions == val_y).float().mean().item()
        if accuracy > best_accuracy + 1e-5:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    predictions = batched_predictions(
        model, test_x, mean, std, args.probe_batch_size, device
    )
    return {
        "features": train_x.shape[-1],
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "test_accuracy": (predictions == test_y).float().mean().item(),
    }, predictions


def overlap(query_predictions, probe_predictions, labels):
    query_correct = query_predictions == labels
    probe_correct = probe_predictions == labels
    query_wrong = ~query_correct
    return {
        "both_correct": int((query_correct & probe_correct).sum()),
        "probe_only_correct": int((query_wrong & probe_correct).sum()),
        "query_only_correct": int((query_correct & ~probe_correct).sum()),
        "both_wrong": int((query_wrong & ~probe_correct).sum()),
        "probe_accuracy_on_query_errors": (
            probe_correct[query_wrong].float().mean().item()
            if query_wrong.any() else None
        ),
        "oracle_union_accuracy": (
            query_correct | probe_correct
        ).float().mean().item(),
        "samples": len(labels),
    }


@torch.no_grad()
def collect_condition(model, original_probe, args, condition, samples,
                      seed, device, dtype):
    set_seed(seed)
    labels = []
    query_predictions = []
    local_predictions = []
    original_predictions = []
    total = 0
    while total < samples:
        batch = min(args.extract_batch_size, samples - total)
        chunks, target, position = make_chunks(
            batch, args.chunks, args.chunk_size, device
        )
        memory = None
        first_logits = None
        final_probe = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for chunk_index in range(args.chunks):
                logits, produced = model(
                    chunks[:, chunk_index], memory=memory,
                    return_memory=True, per_layer_memory=True,
                )
                if chunk_index == 0:
                    first_logits = logits
                final_probe = original_probe(vector(produced))
                memory = intervene(produced, condition)
        rows = torch.arange(batch, device=device)
        labels.append(target.cpu())
        query_predictions.append(logits[:, -1, :16].argmax(-1).cpu())
        local_predictions.append(
            first_logits[rows, position, :16].argmax(-1).cpu()
        )
        original_predictions.append(final_probe.argmax(-1).cpu())
        total += batch

    labels = torch.cat(labels)
    query_predictions = torch.cat(query_predictions)
    local_predictions = torch.cat(local_predictions)
    original_predictions = torch.cat(original_predictions)
    return {
        "labels": labels,
        "query_predictions": query_predictions,
        "metric": {
            "condition": condition,
            "query": (query_predictions == labels).float().mean().item(),
            "local": (local_predictions == labels).float().mean().item(),
            "original_probe": (
                original_predictions == labels
            ).float().mean().item(),
            "samples": len(labels),
        },
    }


def classify_bottleneck(behavior, probes, args):
    l3 = probes["memory_l3_concat"]["test_accuracy"]
    all_memory = probes["memory_all_concat"]["test_accuracy"]
    memory = max(l3, all_memory)
    query_hidden = probes["query_hidden"]["test_accuracy"]
    gaps = {
        "memory_minus_behavior": memory - behavior,
        "query_hidden_minus_behavior": query_hidden - behavior,
        "memory_minus_query_hidden": memory - query_hidden,
        "all_memory_minus_l3": all_memory - l3,
    }
    if behavior >= args.high_information_threshold:
        classification = "no_behavioral_deficit_on_test"
    elif memory < args.low_information_threshold:
        classification = "memory_propagation_or_encoding_bottleneck"
    elif (
        memory >= args.high_information_threshold
        and query_hidden >= args.high_information_threshold
        and gaps["query_hidden_minus_behavior"] >= args.material_gap
    ):
        classification = "output_head_alignment_bottleneck"
    elif (
        memory >= args.high_information_threshold
        and gaps["memory_minus_query_hidden"] >= args.material_gap
    ):
        classification = "memory_to_query_token_routing_bottleneck"
    else:
        classification = "mixed_or_ambiguous_bottleneck"
    return {
        "classification": classification,
        "best_memory_accuracy": memory,
        "gaps": gaps,
        "thresholds": {
            "high_information": args.high_information_threshold,
            "low_information": args.low_information_threshold,
            "material_gap": args.material_gap,
        },
    }


def plot_accuracies(result, path):
    names = [
        "Task head",
        "Old probe",
        "L3 mean",
        "L3 all slots",
        "All memory",
        "Query hidden",
        "L3 + query",
        "All + query",
    ]
    values = [
        result["test_behavior"]["query"],
        result["test_behavior"]["original_probe"],
        result["probes"]["memory_l3_mean"]["test_accuracy"],
        result["probes"]["memory_l3_concat"]["test_accuracy"],
        result["probes"]["memory_all_concat"]["test_accuracy"],
        result["probes"]["query_hidden"]["test_accuracy"],
        result["probes"]["memory_l3_query"]["test_accuracy"],
        result["probes"]["memory_all_query"]["test_accuracy"],
    ]
    colors = ["#d1495b", "#edae49", "#4c78a8", "#2e6fbb",
              "#173f73", "#59a14f", "#8f63b8", "#5b3a91"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(range(len(values)), [100 * value for value in values], color=colors)
    ax.axhline(95, color="#333333", linestyle="--", linewidth=1.5,
               label="95% high-information threshold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Held-out accuracy (%)", fontsize=13)
    ax.set_title("IST Level 6.18.2: 12-chunk Memory–Query Decoupling", fontsize=16)
    ax.set_xticks(range(len(names)), names, rotation=24, ha="right", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=10, loc="lower right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, 100 * value + 1,
                f"{100 * value:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.18.2",
        "status": "post-hoc mechanism diagnosis; not a recovery success test",
        "model": "frozen seed707 Level 6.18.1 8->12 diagnostic-best checkpoint",
        "chunks": args.chunks,
        "feature_probes": FEATURE_KINDS,
        "splits": {
            "train": args.train_samples,
            "validation": args.val_samples,
            "test": args.test_samples,
            "disjoint_seeded_splits": True,
        },
        "causal_conditions": CONDITIONS,
        "decision_rule": {
            "memory_test_accuracy_below_0.90": "memory propagation/encoding bottleneck",
            "memory_and_query_hidden_at_least_0.95_but_task_head_low": "output-head alignment bottleneck",
            "memory_at_least_0.95_and_memory_minus_query_hidden_at_least_0.05": "memory-to-query-token routing bottleneck",
            "otherwise": "mixed/ambiguous",
        },
        "no_ist_parameter_updates": True,
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError("Level 6.18.2 is fixed to seed707 at 12 chunks")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if min(args.train_samples, args.val_samples, args.test_samples) <= 0:
        raise ValueError("dataset sizes must be positive")
    if args.extract_batch_size < 2:
        raise ValueError("batch-roll requires extract-batch-size >= 2")


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.2 frozen 12-chunk Memory-Query decoupling diagnosis"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/level6_18_1/formal/seed707/"
            "transition_8_to_16_bridge_best.pt"
        ),
    )
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--dataset-seed-base", type=int, default=6182000)
    parser.add_argument("--control-eval-batches", type=int, default=50)
    parser.add_argument("--high-information-threshold", type=float, default=0.95)
    parser.add_argument("--low-information-threshold", type=float, default=0.90)
    parser.add_argument("--material-gap", type=float, default=0.05)
    parser.add_argument("--causal-intact-threshold", type=float, default=0.80)
    parser.add_argument("--causal-intervention-threshold", type=float, default=0.20)
    parser.add_argument("--causal-local-threshold", type=float, default=0.90)
    parser.add_argument("--output", default="experiments/level6_18_2/formal")
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
        print(json.dumps(result["diagnosis"], indent=2))
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

    model, original_probe, checkpoint_meta = load_frozen(args, device)
    base = args.dataset_seed_base + args.seed * 100
    train = collect_intact(
        model, original_probe, args, args.train_samples, base + 1, device, dtype
    )
    validation = collect_intact(
        model, original_probe, args, args.val_samples, base + 2, device, dtype
    )
    test = collect_intact(
        model, original_probe, args, args.test_samples, base + 3, device, dtype
    )
    train_features = feature_views(train)
    validation_features = feature_views(validation)
    test_features = feature_views(test)

    probes = {}
    probe_predictions = {}
    overlaps = {}
    for index, kind in enumerate(FEATURE_KINDS):
        metric, predictions = fit_linear(
            train_features[kind], train["labels"],
            validation_features[kind], validation["labels"],
            test_features[kind], test["labels"],
            args, device, base + 100 + index,
        )
        probes[kind] = metric
        probe_predictions[kind] = predictions
        overlaps[kind] = overlap(
            test["query_predictions"], predictions, test["labels"]
        )
        print(
            f"probe={kind} val={metric['best_val_accuracy']:.2%} "
            f"test={metric['test_accuracy']:.2%}",
            flush=True,
        )

    control_args = argparse.Namespace(
        eval_batches=args.control_eval_batches,
        eval_batch_size=args.extract_batch_size,
        chunk_size=args.chunk_size,
    )
    set_seed(base + 50)
    control_8_chunks = evaluate(
        model, original_probe, control_args, 8, device, dtype,
        args.control_eval_batches,
    )

    causal = {}
    causal_predictions = {"intact": test["query_predictions"].tolist()}
    causal["intact"] = {
        "condition": "intact",
        **test["behavior"],
    }
    for condition in CONDITIONS[1:]:
        item = collect_condition(
            model, original_probe, args, condition, args.test_samples,
            base + 3, device, dtype,
        )
        if not torch.equal(item["labels"], test["labels"]):
            raise RuntimeError(f"causal labels diverged for condition={condition}")
        causal[condition] = item["metric"]
        causal_predictions[condition] = item["query_predictions"].tolist()
        print(
            f"causal={condition} query={item['metric']['query']:.2%} "
            f"local={item['metric']['local']:.2%}",
            flush=True,
        )
    strongest_disrupted = max(causal[name]["query"] for name in CONDITIONS[1:])
    causal_summary = {
        "conditions": causal,
        "strongest_disrupted_query": strongest_disrupted,
        "query_drop": causal["intact"]["query"] - strongest_disrupted,
        "passed": (
            causal["intact"]["query"] >= args.causal_intact_threshold
            and strongest_disrupted <= args.causal_intervention_threshold
            and min(causal[name]["local"] for name in CONDITIONS)
            >= args.causal_local_threshold
        ),
    }

    diagnosis = classify_bottleneck(test["behavior"]["query"], probes, args)
    result = {
        "protocol": protocol,
        "checkpoint": {
            "path": str(Path(args.checkpoint)),
            "phase": checkpoint_meta.get("phase"),
            "step": checkpoint_meta.get("step"),
            "best_score": checkpoint_meta.get("best_score"),
            "passed_level6_18_1_gate": checkpoint_meta.get("passed"),
        },
        "state_shape": list(train["states"].shape[1:]),
        "control_8_chunks": control_8_chunks,
        "train_behavior": train["behavior"],
        "validation_behavior": validation["behavior"],
        "test_behavior": test["behavior"],
        "probes": probes,
        "overlaps": overlaps,
        "causal": causal_summary,
        "diagnosis": diagnosis,
    }
    predictions = {
        "labels": test["labels"].tolist(),
        "task_head": test["query_predictions"].tolist(),
        "original_probe": test["original_predictions"].tolist(),
        "refitted_probes": {
            key: value.tolist() for key, value in probe_predictions.items()
        },
        "causal_task_head": causal_predictions,
    }
    save(root / "predictions.json", predictions)
    save(result_path, result)
    save(root / "summary.json", {
        "test_behavior": result["test_behavior"],
        "probe_test_accuracies": {
            key: value["test_accuracy"] for key, value in probes.items()
        },
        "causal": causal_summary,
        "diagnosis": diagnosis,
    })
    plot_accuracies(result, root / "memory_query_decoupling.png")
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
