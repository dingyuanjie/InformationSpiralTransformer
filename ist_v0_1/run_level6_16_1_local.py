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

from long_context_test import set_seed
from run_level6_2_local import make_chunks
from run_level6_13_1_local import bootstrap_mean_ci, save
from run_level6_14_1_local import choose_candidates, swap_slots
from run_level6_16_local import (
    FEATURE_NAMES, evaluate_probe, fit_logistic, load_model, observable_features,
    probe_scores, reset_control, select_threshold,
)


SEEDS = [606, 808, 1001]
SWAP_TIMES = [4, 8]
ACTIONS = [0.20, 0.25]


def group_time(group):
    return int(group.rsplit("@", 1)[1])


def filter_split(split, swap_after):
    indices = [i for i, group in enumerate(split["group"])
               if group_time(group) == swap_after]
    return {"x": [split["x"][i] for i in indices],
            "y": [split["y"][i] for i in indices],
            "group": [split["group"][i] for i in indices]}


def train_time_detectors(detector_data, args):
    output = {}
    for seed, data in detector_data.items():
        output[seed] = {}
        for swap_after in args.swap_times:
            train = filter_split(data["splits"]["train"], swap_after)
            validation = filter_split(data["splits"]["validation"], swap_after)
            test = filter_split(data["splits"]["test"], swap_after)
            probe = fit_logistic(np.asarray(train["x"], dtype=np.float32),
                                 np.asarray(train["y"], dtype=np.float32), args)
            vx = np.asarray(validation["x"], dtype=np.float32)
            vy = np.asarray(validation["y"], dtype=np.int8)
            selection = select_threshold(vy, probe_scores(probe, vx), args.max_clean_fpr)
            output[seed][swap_after] = {
                "probe": probe, "threshold": selection["threshold"],
                "selection": selection,
                "validation": evaluate_probe(probe, selection["threshold"], validation),
                "test": evaluate_probe(probe, selection["threshold"], test),
            }
    return output


def set_memory_scale(model, scale):
    reset_control(model)
    model.blocks[2].memory.propagation_scale = scale


@torch.no_grad()
def utility_batch(model, chunks, slots, swap_after, actions, dtype):
    memories = {"baseline": None, **{str(action): None for action in actions}}
    feature = None
    logits = {}
    with torch.autocast(device_type="cuda", dtype=dtype):
        for chunk_index in range(chunks.shape[1]):
            chunk_number = chunk_index + 1
            reset_control(model)
            logits["baseline"], memories["baseline"] = model(
                chunks[:, chunk_index], memory=memories["baseline"],
                return_memory=True, per_layer_memory=True)
            if chunk_number == swap_after + 1:
                feature = observable_features(model).float().cpu()
            for action in actions:
                key = str(action)
                if chunk_number == swap_after + 1:
                    set_memory_scale(model, action)
                else:
                    reset_control(model)
                logits[key], memories[key] = model(
                    chunks[:, chunk_index], memory=memories[key],
                    return_memory=True, per_layer_memory=True)
            if chunk_number == swap_after:
                memories = {key: swap_slots(value, slots)
                            for key, value in memories.items()}
    reset_control(model)
    return feature, {key: value[:, -1, :16].argmax(-1).cpu()
                     for key, value in logits.items()}


@torch.no_grad()
def collect_utility_data(model, seed, candidates, args, device, dtype):
    splits = {name: {"x": [], "baseline_correct": [],
                     **{f"utility_{action}": [] for action in args.actions},
                     "group": []}
              for name in ["train", "validation", "test"]}
    seed_groups = {
        "train": [args.utility_seed_base + seed * 100 + i
                  for i in range(args.utility_train_seeds)],
        "validation": [args.utility_seed_base + seed * 100 + 20 + i
                       for i in range(args.utility_validation_seeds)],
        "test": [args.utility_seed_base + seed * 100 + 40 + i
                 for i in range(args.utility_test_seeds)],
    }
    batches = args.utility_samples_per_seed // args.eval_batch_size
    for split, eval_seeds in seed_groups.items():
        for candidate in candidates:
            for swap_after in args.swap_times:
                group = f"{candidate['key']}@{swap_after}"
                for eval_seed in eval_seeds:
                    set_seed(eval_seed)
                    for _ in range(batches):
                        chunks, target, _ = make_chunks(
                            args.eval_batch_size, args.chunks, args.chunk_size, device)
                        feature, predictions = utility_batch(
                            model, chunks, candidate["slots"], swap_after,
                            args.actions, dtype)
                        baseline = (predictions["baseline"] == target.cpu()).numpy().astype(np.int8)
                        splits[split]["x"].extend(feature.tolist())
                        splits[split]["baseline_correct"].extend(baseline.tolist())
                        splits[split]["group"].extend([group] * args.eval_batch_size)
                        for action in args.actions:
                            correct = (predictions[str(action)] == target.cpu()).numpy().astype(np.int8)
                            splits[split][f"utility_{action}"].extend(
                                (correct - baseline).tolist())
    return {"seed": seed, "feature_names": FEATURE_NAMES,
            "seed_groups": seed_groups, "splits": splits}


def fit_utility_classifier(x, utility, args):
    changed = utility != 0
    if changed.sum() < 2:
        mean, std = x.mean(axis=0), x.std(axis=0)
        std[std < 1e-6] = 1.0
        return {"mean": mean, "std": std,
                "weight": np.zeros(x.shape[1], dtype=np.float32), "bias": 0.0}
    label = (utility[changed] > 0).astype(np.float32)
    return fit_logistic(x[changed], label, args)


def detector_risk(detector, x):
    return probe_scores(detector["probe"], x)


def action_probabilities(utility_probes, x, actions):
    return np.stack([probe_scores(utility_probes[action], x) for action in actions], axis=1)


def decision_metrics(split, detector, utility_probes, actions, utility_threshold):
    x = np.asarray(split["x"], dtype=np.float32)
    risk = detector_risk(detector, x)
    probabilities = action_probabilities(utility_probes, x, actions)
    chosen_index = probabilities.argmax(axis=1)
    chosen_probability = probabilities[np.arange(len(x)), chosen_index]
    triggered = ((risk >= detector["threshold"])
                 & (chosen_probability >= utility_threshold))
    utilities = np.stack([np.asarray(split[f"utility_{action}"], dtype=np.int8)
                          for action in actions], axis=1)
    selected_utility = np.where(
        triggered, utilities[np.arange(len(x)), chosen_index], 0).astype(np.int8)
    counts = {str(action): int((triggered & (chosen_index == i)).sum())
              for i, action in enumerate(actions)}
    return {"samples": int(len(x)), "trigger_rate": float(triggered.mean()),
            "accuracy_gain": float(selected_utility.mean()),
            "corrected": int((selected_utility == 1).sum()),
            "harmed": int((selected_utility == -1).sum()),
            "unchanged": int((selected_utility == 0).sum()),
            "action_counts": counts, "selected_utility": selected_utility}


def clean_trigger_rate(detector_data, detector, utility_probes, actions,
                       utility_threshold, swap_after, split_name):
    split = filter_split(detector_data["splits"][split_name], swap_after)
    x = np.asarray(split["x"], dtype=np.float32)
    y = np.asarray(split["y"], dtype=np.int8)
    x = x[y == 0]
    risk = detector_risk(detector, x)
    probabilities = action_probabilities(utility_probes, x, actions)
    triggered = ((risk >= detector["threshold"])
                 & (probabilities.max(axis=1) >= utility_threshold))
    return float(triggered.mean())


def select_utility_threshold(validation, detector_data, detector, utility_probes,
                             actions, swap_after, args):
    best = None
    for threshold in np.linspace(0.50, 0.95, 46):
        metric = decision_metrics(validation, detector, utility_probes, actions, threshold)
        clean_fpr = clean_trigger_rate(detector_data, detector, utility_probes,
                                       actions, threshold, swap_after, "validation")
        candidate = {"threshold": float(threshold), "validation_gain": metric["accuracy_gain"],
                     "validation_trigger_rate": metric["trigger_rate"],
                     "validation_clean_fpr": clean_fpr}
        if clean_fpr <= args.max_clean_fpr + 1e-12:
            if best is None or (candidate["validation_gain"], threshold) > (
                    best["validation_gain"], best["threshold"]):
                best = candidate
    return best if best is not None else {"threshold": 1.0,
                                          "validation_gain": 0.0,
                                          "validation_trigger_rate": 0.0,
                                          "validation_clean_fpr": 0.0}


def train_and_evaluate(detector_data, utility_data, detectors, args):
    results = []
    for seed in args.seeds:
        for swap_after in args.swap_times:
            train = utility_data[seed]["splits"]["train"]
            train_indices = [i for i, group in enumerate(train["group"])
                             if group_time(group) == swap_after]
            train_x = np.asarray([train["x"][i] for i in train_indices], dtype=np.float32)
            utility_probes = {}
            for action in args.actions:
                utility = np.asarray([train[f"utility_{action}"][i]
                                      for i in train_indices], dtype=np.int8)
                utility_probes[action] = fit_utility_classifier(train_x, utility, args)
            validation = filter_utility_split(
                utility_data[seed]["splits"]["validation"], swap_after)
            test = filter_utility_split(utility_data[seed]["splits"]["test"], swap_after)
            detector = detectors[seed][swap_after]
            selection = select_utility_threshold(
                validation, detector_data[seed], detector, utility_probes,
                args.actions, swap_after, args)
            validation_metric = decision_metrics(
                validation, detector, utility_probes, args.actions, selection["threshold"])
            test_metric = decision_metrics(
                test, detector, utility_probes, args.actions, selection["threshold"])
            test_clean_fpr = clean_trigger_rate(
                detector_data[seed], detector, utility_probes, args.actions,
                selection["threshold"], swap_after, "test")
            test_utility = test_metric.pop("selected_utility")
            validation_metric.pop("selected_utility")
            test_metric["accuracy_gain_ci"] = bootstrap_mean_ci(
                test_utility, args.bootstrap_seed + seed * 10 + swap_after,
                args.bootstrap_iterations)
            results.append({"seed": seed, "swap_after": swap_after,
                            "detector_validation": detector["validation"],
                            "detector_test": detector["test"],
                            "selection": selection,
                            "validation_decision": validation_metric,
                            "test_decision": test_metric,
                            "test_clean_fpr": test_clean_fpr,
                            "utility_probes": {
                                str(action): serialize_linear(probe)
                                for action, probe in utility_probes.items()}})
    return results


def filter_utility_split(split, swap_after):
    indices = [i for i, group in enumerate(split["group"])
               if group_time(group) == swap_after]
    output = {"x": [split["x"][i] for i in indices],
              "baseline_correct": [split["baseline_correct"][i] for i in indices],
              "group": [split["group"][i] for i in indices]}
    for key in split:
        if key.startswith("utility_"):
            output[key] = [split[key][i] for i in indices]
    return output


def serialize_linear(probe):
    return {"mean": probe["mean"].tolist(), "std": probe["std"].tolist(),
            "weight": probe["weight"].tolist(), "bias": probe["bias"]}


def serialize_detectors(detectors):
    output = {}
    for seed, by_time in detectors.items():
        output[str(seed)] = {}
        for swap_after, item in by_time.items():
            output[str(seed)][str(swap_after)] = {
                "probe": serialize_linear(item["probe"]),
                "threshold": item["threshold"], "selection": item["selection"],
                "validation": item["validation"], "test": item["test"]}
    return output


def plot_results(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"{x['seed']}@{x['swap_after']}" for x in results]
    gains = [x["test_decision"]["accuracy_gain"] * 100 for x in results]
    fprs = [x["test_clean_fpr"] * 100 for x in results]
    axes[0].bar(labels, gains); axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Risk-utility intervention", ylabel="Test gain (pp)")
    axes[1].bar(labels, fprs); axes[1].axhline(5, color="red", linestyle="--")
    axes[1].set(title="Matched-clean trigger rate", ylabel="Rate (%)")
    for axis in axes: axis.tick_params(axis="x", rotation=45); axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Level 6.16.1 risk-utility gating")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--actions", nargs="+", type=float, default=ACTIONS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--level6-16-root", default="experiments/level6_16/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--utility-samples-per-seed", type=int, default=200)
    parser.add_argument("--utility-train-seeds", type=int, default=2)
    parser.add_argument("--utility-validation-seeds", type=int, default=1)
    parser.add_argument("--utility-test-seeds", type=int, default=1)
    parser.add_argument("--utility-seed-base", type=int, default=8150000)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--probe-lr", type=float, default=0.03)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-clean-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=616100)
    parser.add_argument("--output", default="experiments/level6_16_1/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.utility_samples_per_seed % args.eval_batch_size:
        raise ValueError("utility-samples-per-seed must be divisible by eval-batch-size")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    save(root / "preregistration.json", {
        "joint_success": {"seed808_test_gain": ">= 0",
                          "each_time_clean_fpr": "<= 0.05",
                          "overall_test_gain": "> 0"},
        "detector": "model-and-time calibrated",
        "utility": "predict corrected vs harmed among changed samples",
        "actions": args.actions, "features": FEATURE_NAMES,
        "forbidden_features": ["donor", "answer", "clean reference"],
        "protocol": vars(args)})
    detector_raw = json.loads((Path(args.level6_16_root)
                               / "detector_datasets.json").read_text(encoding="utf-8"))
    detector_data = {int(k): v for k, v in detector_raw.items() if int(k) in args.seeds}
    detectors = train_time_detectors(detector_data, args)
    save(root / "time_detectors.json", serialize_detectors(detectors))
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    utility_path = root / "utility_datasets.json"
    if utility_path.exists() and not args.force:
        utility_data = {int(k): v for k, v in
                        json.loads(utility_path.read_text(encoding="utf-8")).items()}
    else:
        utility_data = {}
        for seed in args.seeds:
            model = load_model(seed, args, device)
            utility_data[seed] = collect_utility_data(
                model, seed, choose_candidates(registration, seed), args, device, dtype)
            save(utility_path, utility_data); torch.cuda.empty_cache()
    results = train_and_evaluate(detector_data, utility_data, detectors, args)
    overall_gain = float(np.mean([x["test_decision"]["accuracy_gain"] for x in results]))
    seed808 = float(np.mean([x["test_decision"]["accuracy_gain"] for x in results
                            if x["seed"] == 808]))
    max_fpr = max(x["test_clean_fpr"] for x in results)
    summary = {"protocol": vars(args), "results": results,
               "joint_success": {"overall_gain": overall_gain,
                                 "seed808_gain": seed808,
                                 "max_time_clean_fpr": max_fpr,
                                 "passed": (overall_gain > 0 and seed808 >= 0
                                            and max_fpr <= args.max_clean_fpr)}}
    save(root / "summary.json", summary)
    plot_results(results, root / "risk_utility_results.png")
    print(json.dumps(summary["joint_success"], indent=2))


if __name__ == "__main__": main()
