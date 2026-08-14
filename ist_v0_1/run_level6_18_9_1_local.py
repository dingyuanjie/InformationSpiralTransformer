import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_level6_6_local import build
from run_level6_13_1_local import mcnemar
from run_level6_18_6_local import configure_cuda, save
from run_level6_18_8_local import continuous_effect
from run_level6_18_9_local import (
    READ_PREFIX,
    evaluate_model,
    memory_invariance,
)


SEED = 707
CHUNKS = [8, 12, 16]
PANELS = ["panel_a", "panel_b"]


def checkpoint_audit(source_state, latest_state):
    if latest_state.get("read_training", {}).get("update") != 500:
        raise RuntimeError(
            "Expected the Level 6.18.9 update-500 latest checkpoint"
        )
    source_model = source_state["model"]
    latest_model = latest_state["model"]
    if set(source_model) != set(latest_model):
        raise RuntimeError("Source/latest model keys differ")
    changed = []
    illegal = []
    for name, source_value in source_model.items():
        latest_value = latest_model[name]
        if torch.equal(source_value, latest_value):
            continue
        row = {
            "name": name,
            "parameters": source_value.numel(),
            "max_abs_change": (
                source_value.float() - latest_value.float()
            ).abs().max().item(),
            "allowed": name.startswith(READ_PREFIX),
        }
        changed.append(row)
        if not row["allowed"]:
            illegal.append(name)
    expected = {
        name for name in source_model if name.startswith(READ_PREFIX)
    }
    probe_equal = (
        set(source_state["probe"]) == set(latest_state["probe"])
        and all(
            torch.equal(value, latest_state["probe"][name])
            for name, value in source_state["probe"].items()
        )
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


def load_frozen(args, device):
    source_state = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    latest_state = torch.load(
        args.latest_checkpoint, map_location="cpu", weights_only=False
    )
    if not source_state.get("level6_18_3", {}).get("success", {}).get("passed"):
        raise RuntimeError("Level 6.18.3 source checkpoint did not pass")
    audit = checkpoint_audit(source_state, latest_state)
    if not audit["passed"]:
        raise RuntimeError(f"Checkpoint audit failed: {audit}")
    source, source_probe = build(device, args.chunk_size)
    latest, latest_probe = build(device, args.chunk_size)
    source.load_state_dict(source_state["model"])
    source_probe.load_state_dict(source_state["probe"])
    latest.load_state_dict(latest_state["model"])
    latest_probe.load_state_dict(latest_state["probe"])
    for module in (source, source_probe, latest, latest_probe):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    del source_probe, latest_probe
    return source, latest, audit, latest_state["read_training"]


def accuracy_change(source_predictions, latest_predictions, labels,
                    args, seed):
    source_correct = (source_predictions == labels).numpy().astype("int8")
    latest_correct = (latest_predictions == labels).numpy().astype("int8")
    values = latest_correct.astype("float64") - source_correct.astype("float64")
    result = continuous_effect(values, args, seed)
    result["source_accuracy"] = float(source_correct.mean())
    result["latest_accuracy"] = float(latest_correct.mean())
    result["mcnemar"] = mcnemar(source_correct, latest_correct)
    return result


def compare(source_item, latest_item, args, seed):
    if not torch.equal(source_item["labels"], latest_item["labels"]):
        raise RuntimeError("Source/latest panel labels differ")
    labels = source_item["labels"]
    return {
        "source": source_item["metric"],
        "latest": latest_item["metric"],
        "accuracy_change": accuracy_change(
            source_item["predictions"], latest_item["predictions"],
            labels, args, seed,
        ),
        "margin_change": continuous_effect(
            (latest_item["margins"] - source_item["margins"]).numpy(),
            args, seed + 100,
        ),
        "cross_entropy_change": continuous_effect(
            (
                latest_item["cross_entropy"]
                - source_item["cross_entropy"]
            ).numpy(),
            args, seed + 200,
        ),
    }


def panel_decision(rows, args):
    decisions = {}
    for count in CHUNKS:
        row = rows[str(count)]
        accuracy_lower = row["accuracy_change"]["ci95"][0]
        ce_upper = row["cross_entropy_change"]["ci95"][1]
        margin_lower = row["margin_change"]["ci95"][0]
        if count in (8, 12):
            checks = {
                "accuracy_noninferior": (
                    accuracy_lower >= -args.accuracy_noninferiority_margin
                ),
                "margin_noninferior": (
                    margin_lower >= -args.margin_noninferiority_margin
                ),
                "cross_entropy_noninferior": (
                    ce_upper <= args.cross_entropy_noninferiority_margin
                ),
            }
        else:
            checks = {
                "absolute_accuracy": (
                    row["latest"]["query"] >= args.chunks16_accuracy_threshold
                ),
                "accuracy_noninferior": (
                    accuracy_lower >= -args.accuracy_noninferiority_margin
                ),
                "margin_superior": margin_lower > 0,
                "margin_significant": (
                    row["margin_change"]["sign_flip_p_two_sided"] < args.alpha
                ),
                "cross_entropy_noninferior": (
                    ce_upper <= args.cross_entropy_noninferiority_margin
                ),
            }
        decisions[str(count)] = {
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "by_chunks": decisions,
        "passed": all(item["passed"] for item in decisions.values()),
    }


def aggregate_decision(panel_results, args):
    panels_passed = all(
        panel_results[name]["decision"]["passed"] for name in PANELS
    )
    agreement = {
        str(count): all(
            panel_results[name]["decision"]["by_chunks"][str(count)]["passed"]
            for name in PANELS
        ) for count in CHUNKS
    }
    if panels_passed:
        classification = "frozen_validation_calibration_passed"
        next_boundary = (
            "Register a separate protected-test opening decision; do not train further."
        )
    else:
        classification = "frozen_validation_calibration_failed"
        next_boundary = (
            "Reject the Level 6.18.9 checkpoint as a stable rescue and stop read optimization."
        )
    return {
        "classification": classification,
        "both_panels_passed": panels_passed,
        "agreement_by_chunks": agreement,
        "protected_tests_opened": False,
        "seed909_opened": False,
        "registered_next_boundary": next_boundary,
        "thresholds": {
            "accuracy_noninferiority_margin": args.accuracy_noninferiority_margin,
            "margin_noninferiority_margin": args.margin_noninferiority_margin,
            "cross_entropy_noninferiority_margin": args.cross_entropy_noninferiority_margin,
            "chunks16_absolute_accuracy": args.chunks16_accuracy_threshold,
            "alpha": args.alpha,
        },
    }


def plot_result(panel_results, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    positions = list(range(len(CHUNKS)))
    width = 0.18
    for panel_index, name in enumerate(PANELS):
        rows = panel_results[name]["rows"]
        source_values = [100 * rows[str(count)]["source"]["query"] for count in CHUNKS]
        latest_values = [100 * rows[str(count)]["latest"]["query"] for count in CHUNKS]
        offset = (panel_index * 2 - 1.5) * width
        axes[0].bar(
            [x + offset for x in positions], source_values, width,
            color="#4c78a8", alpha=0.65 + panel_index * 0.2,
            label=f"{name} source",
        )
        axes[0].bar(
            [x + offset + width for x in positions], latest_values, width,
            color="#e45756", alpha=0.65 + panel_index * 0.2,
            label=f"{name} latest",
        )
        margin_estimates = [
            rows[str(count)]["margin_change"]["estimate"] for count in CHUNKS
        ]
        margin_lows = [
            rows[str(count)]["margin_change"]["ci95"][0] for count in CHUNKS
        ]
        margin_highs = [
            rows[str(count)]["margin_change"]["ci95"][1] for count in CHUNKS
        ]
        axes[1].errorbar(
            [x + (panel_index - 0.5) * 0.12 for x in positions],
            margin_estimates,
            yerr=[
                [estimate - low for estimate, low in zip(margin_estimates, margin_lows)],
                [high - estimate for estimate, high in zip(margin_estimates, margin_highs)],
            ],
            marker="o", capsize=4, linewidth=2,
            color=("#54a24b" if panel_index == 0 else "#f58518"),
            label=name,
        )
    axes[0].axhline(95, color="#333333", linestyle="--")
    axes[0].set_ylim(88, 101)
    axes[0].set_xticks(positions, [str(count) for count in CHUNKS])
    axes[0].set_xlabel("Chunks")
    axes[0].set_ylabel("Frozen validation accuracy (%)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(positions, [str(count) for count in CHUNKS])
    axes[1].set_xlabel("Chunks")
    axes[1].set_ylabel("Latest - source mean margin")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("IST Level 6.18.9.1: Frozen Validation Calibration Audit")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preregistration(args):
    return {
        "level": "6.18.9.1",
        "status": "frozen validation-calibration audit",
        "seed": SEED,
        "source": "Level 6.18.3 formally passed checkpoint",
        "candidate": "Level 6.18.9 update-500 latest checkpoint",
        "panels": {
            "names": PANELS,
            "samples_per_length_per_panel": args.samples_per_panel,
            "new_disjoint_seeds": [args.panel_seed_base, args.panel_seed_base + 100000],
            "lengths": CHUNKS,
            "both_panels_must_pass": True,
        },
        "eight_twelve_rule": {
            "paired_accuracy_lower_ci": f">= -{args.accuracy_noninferiority_margin}",
            "paired_margin_lower_ci": f">= -{args.margin_noninferiority_margin}",
            "paired_cross_entropy_upper_ci": f"<= {args.cross_entropy_noninferiority_margin}",
        },
        "sixteen_rule": {
            "candidate_accuracy": f">= {args.chunks16_accuracy_threshold}",
            "paired_accuracy_lower_ci": f">= -{args.accuracy_noninferiority_margin}",
            "paired_margin_lower_ci": "> 0",
            "paired_margin_sign_flip_p": f"< {args.alpha}",
            "paired_cross_entropy_upper_ci": f"<= {args.cross_entropy_noninferiority_margin}",
        },
        "statistics": {
            "paired_bootstrap_iterations": args.bootstrap_iterations,
            "paired_sign_flip_iterations": args.sign_flip_iterations,
            "exact_mcnemar_reported": True,
        },
        "checkpoint_integrity": {
            "changed_tensors": 4,
            "changed_parameters": 16640,
            "only_final_memory_read": True,
            "persistent_memory_exactly_invariant": True,
        },
        "no_parameter_updates": True,
        "protected_tests_locked": True,
        "seed909_locked": True,
        "protocol": vars(args),
    }


def validate(args):
    if args.seed != SEED:
        raise ValueError("Level 6.18.9.1 is fixed to seed707")
    if args.chunks != CHUNKS:
        raise ValueError("Level 6.18.9.1 is fixed to chunks 8, 12, and 16")
    for path in (args.source_checkpoint, args.latest_checkpoint):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.samples_per_panel <= 0:
        raise ValueError("samples-per-panel must be positive")
    if args.eval_batch_size < 2:
        raise ValueError("eval-batch-size must be >= 2")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if min(
        args.accuracy_noninferiority_margin,
        args.margin_noninferiority_margin,
        args.cross_entropy_noninferiority_margin,
    ) < 0:
        raise ValueError("noninferiority margins must be non-negative")


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.18.9.1 frozen validation-calibration audit"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=int, nargs="+", default=CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--source-checkpoint",
        default="experiments/level6_18_3/formal/rescued_head_checkpoint.pt",
    )
    parser.add_argument(
        "--latest-checkpoint",
        default="experiments/level6_18_9/formal/read_supervision_latest.pt",
    )
    parser.add_argument("--samples-per-panel", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--panel-seed-base", type=int, default=61899100)
    parser.add_argument("--accuracy-noninferiority-margin", type=float, default=0.01)
    parser.add_argument("--margin-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--cross-entropy-noninferiority-margin", type=float, default=0.01)
    parser.add_argument("--chunks16-accuracy-threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed-base", type=int, default=61899300)
    parser.add_argument("--invariance-samples", type=int, default=8)
    parser.add_argument("--invariance-seed-base", type=int, default=61899500)
    parser.add_argument("--output", default="experiments/level6_18_9_1/formal")
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
        print(json.dumps(result["decision"], indent=2))
        return

    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    source, latest, audit, training_meta = load_frozen(args, device)
    invariance = memory_invariance(source, latest, args, device, dtype)
    if not invariance["passed"]:
        raise RuntimeError("Persistent-Memory invariance failed")
    panel_results = {}
    predictions = {}
    for panel_index, name in enumerate(PANELS):
        panel_seed = args.panel_seed_base + panel_index * 100000
        rows = {}
        panel_predictions = {}
        for count in CHUNKS:
            seed = panel_seed + count
            source_item = evaluate_model(
                source, args, count, args.samples_per_panel,
                seed, "intact", device, dtype,
            )
            latest_item = evaluate_model(
                latest, args, count, args.samples_per_panel,
                seed, "intact", device, dtype,
            )
            rows[str(count)] = compare(
                source_item, latest_item, args,
                args.bootstrap_seed_base + panel_index * 1000 + count,
            )
            panel_predictions[str(count)] = {
                "labels": source_item["labels"].tolist(),
                "source": source_item["predictions"].tolist(),
                "latest": latest_item["predictions"].tolist(),
                "source_margin": source_item["margins"].tolist(),
                "latest_margin": latest_item["margins"].tolist(),
            }
            print(
                f"{name} chunks={count} "
                f"source={rows[str(count)]['source']['query']:.2%} "
                f"latest={rows[str(count)]['latest']['query']:.2%} "
                f"margin={rows[str(count)]['margin_change']['estimate']:+.4f}",
                flush=True,
            )
        panel_results[name] = {
            "seed_base": panel_seed,
            "rows": rows,
            "decision": panel_decision(rows, args),
        }
        predictions[name] = panel_predictions
        save(root / f"{name}.json", panel_results[name])
        save(root / f"predictions_{name}.json", panel_predictions)
    decision = aggregate_decision(panel_results, args)
    result = {
        "protocol": protocol,
        "checkpoint_audit": audit,
        "level6_18_9_training": {
            "update": training_meta.get("update"),
            "passed": training_meta.get("passed"),
            "stable_streak": training_meta.get("stable_streak"),
        },
        "memory_invariance": invariance,
        "panels": panel_results,
        "decision": decision,
    }
    save(result_path, result)
    save(root / "summary.json", {
        "checkpoint_audit": audit,
        "memory_invariance": {
            "passed": invariance["passed"],
            "overall_max_abs_difference": invariance["overall_max_abs_difference"],
        },
        "panel_decisions": {
            name: panel_results[name]["decision"] for name in PANELS
        },
        "decision": decision,
    })
    save(root / "predictions.json", predictions)
    plot_result(panel_results, root / "validation_calibration.png")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
