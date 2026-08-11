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
from run_level6_2_local import make_chunks, vector
from run_level6_6_local import build
from run_level6_9_local import CONDITIONS, intervene
from run_level6_13_1_local import bootstrap_mean_ci, mcnemar
from run_level6_18_2_local import FEATURE_KINDS, feature_views, fit_linear


SEED = 707
CHUNKS = 16


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint_audit(original_state, rescued_state):
    changed = []
    max_non_output_change = 0.0
    for name, original in original_state["model"].items():
        rescued = rescued_state["model"][name].detach().cpu()
        original = original.detach().cpu()
        difference = (rescued - original).abs().max().item()
        if difference > 0:
            changed.append({"name": name, "max_abs_change": difference})
        if not name.startswith("output."):
            max_non_output_change = max(max_non_output_change, difference)
    tail_weight_change = (
        rescued_state["model"]["output.weight"].detach().cpu()[16:]
        - original_state["model"]["output.weight"].detach().cpu()[16:]
    ).abs().max().item()
    tail_bias_change = (
        rescued_state["model"]["output.bias"].detach().cpu()[16:]
        - original_state["model"]["output.bias"].detach().cpu()[16:]
    ).abs().max().item()
    probe_change = max(
        (
            rescued_state["probe"][name].detach().cpu()
            - original_state["probe"][name].detach().cpu()
        ).abs().max().item()
        for name in original_state["probe"]
    )
    return {
        "changed_tensors": changed,
        "max_non_output_change": max_non_output_change,
        "unused_output_rows_max_change": max(tail_weight_change, tail_bias_change),
        "probe_max_change": probe_change,
        "passed": (
            max_non_output_change == 0.0
            and tail_weight_change == 0.0
            and tail_bias_change == 0.0
            and probe_change == 0.0
            and {item["name"] for item in changed}
            <= {"output.weight", "output.bias"}
        ),
    }


def load_models(args, device):
    original_state = torch.load(
        args.original_checkpoint, map_location=device, weights_only=False
    )
    rescued_state = torch.load(
        args.rescued_checkpoint, map_location=device, weights_only=False
    )
    original_meta = original_state.get("phase_result", {})
    rescued_meta = rescued_state.get("level6_18_3", {})
    if original_meta.get("phase") != "bridge" or original_meta.get("eval_chunks") != 12:
        raise RuntimeError("Original checkpoint is not the failed 8->12 bridge")
    if not rescued_meta.get("success", {}).get("passed"):
        raise RuntimeError("Level 6.18.3 rescued checkpoint did not pass formally")
    audit = checkpoint_audit(original_state, rescued_state)
    if not audit["passed"]:
        raise RuntimeError(f"Original/rescued checkpoint audit failed: {audit}")

    model, original_probe = build(device, args.chunk_size)
    model.load_state_dict(original_state["model"])
    original_probe.load_state_dict(original_state["probe"])
    model.eval()
    original_probe.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in original_probe.parameters():
        parameter.requires_grad_(False)
    transferred_weight = rescued_state["model"]["output.weight"][:16].detach().to(device)
    transferred_bias = rescued_state["model"]["output.bias"][:16].detach().to(device)
    return model, original_probe, transferred_weight, transferred_bias, {
        "checkpoint_audit": audit,
        "original": {
            "phase": original_meta.get("phase"),
            "step": original_meta.get("step"),
            "passed_level6_18_1_gate": original_meta.get("passed"),
        },
        "rescued": rescued_meta,
    }


@torch.no_grad()
def collect_intact(model, original_probe, transferred_weight, transferred_bias,
                   args, samples, seed, device, dtype):
    set_seed(seed)
    states = []
    query_hidden = []
    labels = []
    original_predictions = []
    transferred_predictions = []
    probe_predictions = []
    original_local = []
    transferred_local = []
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
            first_transferred_logits = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(args.chunks):
                    logits, memory = model(
                        chunks[:, chunk_index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    transferred_logits = F.linear(
                        captured["hidden"], transferred_weight, transferred_bias
                    )
                    if chunk_index == 0:
                        first_logits = logits
                        first_transferred_logits = transferred_logits
                original_probe_logits = original_probe(vector(memory))
            rows = torch.arange(batch, device=device)
            states.append(
                torch.stack(memory, dim=1).detach().cpu().to(torch.float16)
            )
            query_hidden.append(
                captured["hidden"][:, -1].detach().cpu().to(torch.float16)
            )
            labels.append(target.cpu())
            original_predictions.append(logits[:, -1, :16].argmax(-1).cpu())
            transferred_predictions.append(
                transferred_logits[:, -1].argmax(-1).cpu()
            )
            probe_predictions.append(original_probe_logits.argmax(-1).cpu())
            original_local.append(
                first_logits[rows, position, :16].argmax(-1).cpu()
            )
            transferred_local.append(
                first_transferred_logits[rows, position].argmax(-1).cpu()
            )
            total += batch
    finally:
        handle.remove()

    labels = torch.cat(labels)
    predictions = {
        "original_head": torch.cat(original_predictions),
        "transferred_head": torch.cat(transferred_predictions),
        "original_probe": torch.cat(probe_predictions),
        "original_local": torch.cat(original_local),
        "transferred_local": torch.cat(transferred_local),
    }
    return {
        "states": torch.cat(states),
        "query_hidden": torch.cat(query_hidden),
        "labels": labels,
        "predictions": predictions,
        "behavior": {
            key: (value == labels).float().mean().item()
            for key, value in predictions.items()
        } | {"samples": len(labels)},
    }


def pairwise_overlap(left, right, labels):
    left_correct = left == labels
    right_correct = right == labels
    left_wrong = ~left_correct
    return {
        "left_accuracy": left_correct.float().mean().item(),
        "right_accuracy": right_correct.float().mean().item(),
        "both_correct": int((left_correct & right_correct).sum()),
        "right_only_correct": int((left_wrong & right_correct).sum()),
        "left_only_correct": int((left_correct & ~right_correct).sum()),
        "both_wrong": int((left_wrong & ~right_correct).sum()),
        "right_accuracy_on_left_errors": (
            right_correct[left_wrong].float().mean().item()
            if left_wrong.any() else None
        ),
        "oracle_union_accuracy": (
            left_correct | right_correct
        ).float().mean().item(),
        "samples": len(labels),
    }


def paired_change(left, right, labels, args, offset):
    left_correct = (left == labels).numpy().astype(np.int8)
    right_correct = (right == labels).numpy().astype(np.int8)
    change = right_correct - left_correct
    return {
        "estimate": float(change.mean()),
        "bootstrap": bootstrap_mean_ci(
            change, args.bootstrap_seed + offset, args.bootstrap_iterations
        ),
        "mcnemar_right_vs_left": mcnemar(right_correct, left_correct),
        "corrected": int(np.sum(change == 1)),
        "harmed": int(np.sum(change == -1)),
    }


@torch.no_grad()
def collect_condition(model, transferred_weight, transferred_bias, args,
                      condition, samples, seed, device, dtype):
    set_seed(seed)
    labels = []
    original_predictions = []
    transferred_predictions = []
    original_local = []
    transferred_local = []
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
            first_transferred_logits = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(args.chunks):
                    logits, produced = model(
                        chunks[:, chunk_index], memory=memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    transferred_logits = F.linear(
                        captured["hidden"], transferred_weight, transferred_bias
                    )
                    if chunk_index == 0:
                        first_logits = logits
                        first_transferred_logits = transferred_logits
                    memory = intervene(produced, condition)
            rows = torch.arange(batch, device=device)
            labels.append(target.cpu())
            original_predictions.append(logits[:, -1, :16].argmax(-1).cpu())
            transferred_predictions.append(
                transferred_logits[:, -1].argmax(-1).cpu()
            )
            original_local.append(
                first_logits[rows, position, :16].argmax(-1).cpu()
            )
            transferred_local.append(
                first_transferred_logits[rows, position].argmax(-1).cpu()
            )
            total += batch
    finally:
        handle.remove()

    labels = torch.cat(labels)
    predictions = {
        "original_head": torch.cat(original_predictions),
        "transferred_head": torch.cat(transferred_predictions),
        "original_local": torch.cat(original_local),
        "transferred_local": torch.cat(transferred_local),
    }
    return {
        "labels": labels,
        "predictions": predictions,
        "metric": {
            "condition": condition,
            **{
                key: (value == labels).float().mean().item()
                for key, value in predictions.items()
            },
            "samples": len(labels),
        },
    }


def causal_summary(conditions, head, args):
    disrupted = max(conditions[name][head] for name in CONDITIONS[1:])
    local_key = "original_local" if head == "original_head" else "transferred_local"
    return {
        "intact_query": conditions["intact"][head],
        "strongest_disrupted_query": disrupted,
        "query_drop": conditions["intact"][head] - disrupted,
        "minimum_local": min(conditions[name][local_key] for name in CONDITIONS),
        "passed": (
            conditions["intact"][head] >= args.causal_intact_threshold
            and disrupted <= args.causal_intervention_threshold
            and min(conditions[name][local_key] for name in CONDITIONS)
            >= args.causal_local_threshold
        ),
    }


def classify(behavior, probes, args):
    memory = max(
        probes["memory_l3_concat"]["test_accuracy"],
        probes["memory_all_concat"]["test_accuracy"],
    )
    query = probes["query_hidden"]["test_accuracy"]
    transferred = behavior["transferred_head"]
    gaps = {
        "memory_minus_original_head": memory - behavior["original_head"],
        "memory_minus_transferred_head": memory - transferred,
        "query_hidden_minus_transferred_head": query - transferred,
        "memory_minus_query_hidden": memory - query,
    }
    if memory < args.low_information_threshold:
        diagnosis = "memory_propagation_or_encoding_degradation"
    elif (
        memory >= args.high_information_threshold
        and query >= args.high_information_threshold
        and transferred < args.high_information_threshold
    ):
        diagnosis = "residual_length_specific_output_alignment"
    elif (
        memory >= args.high_information_threshold
        and gaps["memory_minus_query_hidden"] >= args.material_gap
    ):
        diagnosis = "memory_to_query_token_routing_degradation"
    elif transferred >= args.high_information_threshold:
        diagnosis = "no_residual_16_chunk_deficit"
    else:
        diagnosis = "mixed_or_ambiguous_residual"
    return {
        "classification": diagnosis,
        "best_memory_accuracy": memory,
        "gaps": gaps,
        "thresholds": {
            "high_information": args.high_information_threshold,
            "low_information": args.low_information_threshold,
            "material_gap": args.material_gap,
        },
    }


def plot_result(result, path):
    names = [
        "Original head", "12-chunk\nrescue head", "Old probe",
        "L3 mean", "L3 all slots", "All memory", "Query hidden",
        "All memory\n+ query",
    ]
    values = [
        result["test_behavior"]["original_head"],
        result["test_behavior"]["transferred_head"],
        result["test_behavior"]["original_probe"],
        result["probes"]["memory_l3_mean"]["test_accuracy"],
        result["probes"]["memory_l3_concat"]["test_accuracy"],
        result["probes"]["memory_all_concat"]["test_accuracy"],
        result["probes"]["query_hidden"]["test_accuracy"],
        result["probes"]["memory_all_query"]["test_accuracy"],
    ]
    colors = ["#d1495b", "#2e6fbb", "#edae49", "#6b91bd",
              "#3979bd", "#173f73", "#59a14f", "#5b3a91"]
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    bars = ax.bar(range(len(values)), [100 * item for item in values], color=colors)
    ax.axhline(95, color="#333333", linestyle="--", linewidth=1.5,
               label="95% formation target")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Held-out accuracy (%)", fontsize=13)
    ax.set_title("IST Level 6.18.4: Residual 16-chunk Decoupling", fontsize=16)
    ax.set_xticks(range(len(names)), names, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, 100 * value + 1,
                f"{100 * value:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.18.4",
        "status": "frozen residual 16-chunk mechanism diagnosis",
        "models": {
            "shared_backbone": "untouched seed707 Level 6.18.1 diagnostic-best checkpoint",
            "heads": ["original", "Level 6.18.3 12-chunk rescue"],
            "checkpoint_compatibility_audited": True,
        },
        "chunks": args.chunks,
        "feature_probes": FEATURE_KINDS,
        "splits": {
            "train": args.train_samples,
            "validation": args.val_samples,
            "test": args.test_samples,
        },
        "decision_rule": {
            "memory_below_0.90": "Memory propagation/encoding degradation",
            "memory_and_query_at_least_0.95_but_transferred_head_below_0.95": "residual length-specific output alignment",
            "memory_at_least_0.95_and_memory_minus_query_at_least_0.05": "Memory-to-query routing degradation",
            "transferred_head_at_least_0.95": "no residual deficit",
            "otherwise": "mixed/ambiguous",
        },
        "causal_conditions": CONDITIONS,
        "no_ist_or_head_updates": True,
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError("Level 6.18.4 is fixed to seed707 at 16 chunks")
    for path in [args.original_checkpoint, args.rescued_checkpoint]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.extract_batch_size < 2:
        raise ValueError("batch-roll requires extract-batch-size >= 2")


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.4 frozen residual 16-chunk diagnosis"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--original-checkpoint",
        default=(
            "experiments/level6_18_1/formal/seed707/"
            "transition_8_to_16_bridge_best.pt"
        ),
    )
    parser.add_argument(
        "--rescued-checkpoint",
        default="experiments/level6_18_3/formal/rescued_head_checkpoint.pt",
    )
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--causal-samples", type=int, default=1024)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--dataset-seed-base", type=int, default=6184000)
    parser.add_argument("--high-information-threshold", type=float, default=0.95)
    parser.add_argument("--low-information-threshold", type=float, default=0.90)
    parser.add_argument("--material-gap", type=float, default=0.05)
    parser.add_argument("--causal-intact-threshold", type=float, default=0.80)
    parser.add_argument("--causal-intervention-threshold", type=float, default=0.20)
    parser.add_argument("--causal-local-threshold", type=float, default=0.90)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=6184100)
    parser.add_argument("--output", default="experiments/level6_18_4/formal")
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

    model, original_probe, transferred_weight, transferred_bias, source = load_models(
        args, device
    )
    base = args.dataset_seed_base + args.seed * 100
    train = collect_intact(
        model, original_probe, transferred_weight, transferred_bias,
        args, args.train_samples, base + 1, device, dtype,
    )
    validation = collect_intact(
        model, original_probe, transferred_weight, transferred_bias,
        args, args.val_samples, base + 2, device, dtype,
    )
    test = collect_intact(
        model, original_probe, transferred_weight, transferred_bias,
        args, args.test_samples, base + 3, device, dtype,
    )
    train_features = feature_views(train)
    validation_features = feature_views(validation)
    test_features = feature_views(test)

    probes = {}
    refitted_predictions = {}
    for index, kind in enumerate(FEATURE_KINDS):
        metric, predictions = fit_linear(
            train_features[kind], train["labels"],
            validation_features[kind], validation["labels"],
            test_features[kind], test["labels"],
            args, device, base + 100 + index,
        )
        probes[kind] = metric
        refitted_predictions[kind] = predictions
        print(
            f"probe={kind} val={metric['best_val_accuracy']:.2%} "
            f"test={metric['test_accuracy']:.2%}",
            flush=True,
        )

    comparisons = {
        "original_to_transferred": pairwise_overlap(
            test["predictions"]["original_head"],
            test["predictions"]["transferred_head"], test["labels"]
        ),
        "transferred_to_query_probe": pairwise_overlap(
            test["predictions"]["transferred_head"],
            refitted_predictions["query_hidden"], test["labels"]
        ),
        "transferred_to_memory_probe": pairwise_overlap(
            test["predictions"]["transferred_head"],
            refitted_predictions["memory_all_concat"], test["labels"]
        ),
        "query_probe_to_memory_probe": pairwise_overlap(
            refitted_predictions["query_hidden"],
            refitted_predictions["memory_all_concat"], test["labels"]
        ),
    }
    paired = {
        "original_to_transferred": paired_change(
            test["predictions"]["original_head"],
            test["predictions"]["transferred_head"],
            test["labels"], args, 1,
        ),
        "transferred_to_query_probe": paired_change(
            test["predictions"]["transferred_head"],
            refitted_predictions["query_hidden"],
            test["labels"], args, 2,
        ),
        "transferred_to_memory_probe": paired_change(
            test["predictions"]["transferred_head"],
            refitted_predictions["memory_all_concat"],
            test["labels"], args, 3,
        ),
    }

    causal_rows = {}
    causal_predictions = {}
    causal_labels = None
    causal_seed = base + 50
    for condition in CONDITIONS:
        item = collect_condition(
            model, transferred_weight, transferred_bias, args,
            condition, args.causal_samples, causal_seed, device, dtype,
        )
        if causal_labels is None:
            causal_labels = item["labels"]
        elif not torch.equal(causal_labels, item["labels"]):
            raise RuntimeError(f"causal labels diverged for condition={condition}")
        causal_rows[condition] = item["metric"]
        causal_predictions[condition] = {
            key: value.tolist() for key, value in item["predictions"].items()
        }
        print(
            f"causal={condition} original={item['metric']['original_head']:.2%} "
            f"transferred={item['metric']['transferred_head']:.2%}",
            flush=True,
        )
    causal = {
        "conditions": causal_rows,
        "original_head": causal_summary(causal_rows, "original_head", args),
        "transferred_head": causal_summary(causal_rows, "transferred_head", args),
    }
    diagnosis = classify(test["behavior"], probes, args)
    result = {
        "protocol": protocol,
        "source": source,
        "state_shape": list(train["states"].shape[1:]),
        "train_behavior": train["behavior"],
        "validation_behavior": validation["behavior"],
        "test_behavior": test["behavior"],
        "probes": probes,
        "comparisons": comparisons,
        "paired_changes": paired,
        "causal": causal,
        "diagnosis": diagnosis,
    }
    save(root / "result.json", result)
    save(root / "summary.json", {
        "test_behavior": test["behavior"],
        "probe_test_accuracies": {
            key: value["test_accuracy"] for key, value in probes.items()
        },
        "paired_changes": paired,
        "causal": causal,
        "diagnosis": diagnosis,
    })
    save(root / "predictions.json", {
        "labels": test["labels"].tolist(),
        "deployed_heads": {
            "original": test["predictions"]["original_head"].tolist(),
            "transferred": test["predictions"]["transferred_head"].tolist(),
        },
        "refitted_probes": {
            key: value.tolist() for key, value in refitted_predictions.items()
        },
        "causal_labels": causal_labels.tolist(),
        "causal": causal_predictions,
    })
    plot_result(result, root / "residual_16_decoupling.png")
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
