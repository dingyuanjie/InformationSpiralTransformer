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
from run_level6_13_1_local import bootstrap_mean_ci, mcnemar, save
from run_level6_14_1_local import choose_candidates, swap_slots
from run_level6_16_local import (
    evaluate_probe, fit_logistic, load_model, observable_features,
    probe_scores, reset_control, select_threshold,
)
from run_level6_16_1_local import fit_utility_classifier, serialize_linear
from run_level6_16_2_local import decide, load_frozen_policy
from run_level6_17_local import selected_scale


SEEDS = [606, 808, 1001]
ACTIONS = [0.20, 0.25]
SWAP_AFTER = 4


@torch.no_grad()
def step2_counterfactual_batch(model, chunks, slots, first_policy, args, dtype):
    names = ["reference", *[str(action) for action in args.actions]]
    memories = {name: None for name in names}
    logits = {}
    step2_feature = None
    first_trigger = np.zeros(chunks.shape[0], dtype=bool)
    with torch.autocast(device_type="cuda", dtype=dtype):
        for chunk_index in range(chunks.shape[1]):
            chunk_number = chunk_index + 1
            relative_step = chunk_number - args.swap_after
            if relative_step == 1:
                reset_control(model)
                _, _ = model(chunks[:, chunk_index], memory=memories["reference"],
                             return_memory=True, per_layer_memory=True)
                feature = observable_features(model).float().cpu()
                trigger, choice, _, _ = decide(first_policy, feature, args.actions)
                first_trigger = trigger
                scale = selected_scale(trigger, choice, args.actions, chunks.device)
                for name in names:
                    reset_control(model)
                    model.blocks[2].memory.propagation_scale = scale
                    logits[name], memories[name] = model(
                        chunks[:, chunk_index], memory=memories[name],
                        return_memory=True, per_layer_memory=True)
            elif relative_step == 2:
                reset_control(model)
                logits["reference"], memories["reference"] = model(
                    chunks[:, chunk_index], memory=memories["reference"],
                    return_memory=True, per_layer_memory=True)
                step2_feature = observable_features(model).float().cpu()
                for action in args.actions:
                    name = str(action)
                    reset_control(model)
                    model.blocks[2].memory.propagation_scale = action
                    logits[name], memories[name] = model(
                        chunks[:, chunk_index], memory=memories[name],
                        return_memory=True, per_layer_memory=True)
            else:
                for name in names:
                    reset_control(model)
                    logits[name], memories[name] = model(
                        chunks[:, chunk_index], memory=memories[name],
                        return_memory=True, per_layer_memory=True)
            if chunk_number == args.swap_after and slots:
                memories = {name: swap_slots(memory, slots)
                            for name, memory in memories.items()}
    reset_control(model)
    predictions = {name: value[:, -1, :16].argmax(-1).cpu().numpy()
                   for name, value in logits.items()}
    return step2_feature, predictions, first_trigger


def empty_split(args):
    output = {"x_polluted": [], "x_clean": [],
              "reference_correct_polluted": [], "reference_correct_clean": [],
              "first_trigger_polluted": [], "first_trigger_clean": [], "group": []}
    for action in args.actions:
        output[f"utility_polluted_{action}"] = []
        output[f"utility_clean_{action}"] = []
    return output


@torch.no_grad()
def collect_data(model, seed, candidates, first_policy, args, device, dtype):
    splits = {name: empty_split(args) for name in ["train", "validation", "test"]}
    seed_groups = {
        "train": [args.data_seed_base + seed * 100 + i for i in range(args.train_seeds)],
        "validation": [args.data_seed_base + seed * 100 + 20 + i
                       for i in range(args.validation_seeds)],
        "test": [args.data_seed_base + seed * 100 + 40 + i
                 for i in range(args.test_seeds)],
    }
    batches = args.samples_per_seed // args.eval_batch_size
    for split_name, eval_seeds in seed_groups.items():
        split = splits[split_name]
        for candidate in candidates:
            group = candidate["key"]
            for eval_seed in eval_seeds:
                set_seed(eval_seed)
                for _ in range(batches):
                    chunks, target, _ = make_chunks(
                        args.eval_batch_size, args.chunks, args.chunk_size, device)
                    polluted_x, polluted_predictions, polluted_first = (
                        step2_counterfactual_batch(
                            model, chunks, candidate["slots"], first_policy,
                            args, dtype))
                    clean_x, clean_predictions, clean_first = step2_counterfactual_batch(
                        model, chunks, [], first_policy, args, dtype)
                    target_np = target.cpu().numpy()
                    polluted_ref = (polluted_predictions["reference"] == target_np).astype(np.int8)
                    clean_ref = (clean_predictions["reference"] == target_np).astype(np.int8)
                    split["x_polluted"].extend(polluted_x.tolist())
                    split["x_clean"].extend(clean_x.tolist())
                    split["reference_correct_polluted"].extend(polluted_ref.tolist())
                    split["reference_correct_clean"].extend(clean_ref.tolist())
                    split["first_trigger_polluted"].extend(polluted_first.tolist())
                    split["first_trigger_clean"].extend(clean_first.tolist())
                    split["group"].extend([group] * args.eval_batch_size)
                    for action in args.actions:
                        polluted_correct = (polluted_predictions[str(action)]
                                            == target_np).astype(np.int8)
                        clean_correct = (clean_predictions[str(action)]
                                         == target_np).astype(np.int8)
                        split[f"utility_polluted_{action}"].extend(
                            (polluted_correct - polluted_ref).tolist())
                        split[f"utility_clean_{action}"].extend(
                            (clean_correct - clean_ref).tolist())
    return {"seed": seed, "seed_groups": seed_groups, "splits": splits}


def detector_split(split):
    return {"x": [*split["x_clean"], *split["x_polluted"]],
            "y": [0] * len(split["x_clean"]) + [1] * len(split["x_polluted"])}


def train_step2_detector(data, seed, args):
    train = detector_split(data["splits"]["train"])
    validation = detector_split(data["splits"]["validation"])
    test = detector_split(data["splits"]["test"])
    probe = fit_logistic(np.asarray(train["x"], dtype=np.float32),
                         np.asarray(train["y"], dtype=np.float32), args)
    vx = np.asarray(validation["x"], dtype=np.float32)
    vy = np.asarray(validation["y"], dtype=np.int8)
    fpr_limit = args.seed1001_step2_fpr if seed == 1001 else args.max_step2_fpr
    selection = select_threshold(vy, probe_scores(probe, vx), fpr_limit)
    return {"probe": probe, "threshold": selection["threshold"],
            "fpr_limit": fpr_limit, "selection": selection,
            "validation": evaluate_probe(probe, selection["threshold"], validation),
            "test": evaluate_probe(probe, selection["threshold"], test)}


def train_utility_probes(data, args):
    split = data["splits"]["train"]
    x = np.asarray(split["x_polluted"], dtype=np.float32)
    return {action: fit_utility_classifier(
        x, np.asarray(split[f"utility_polluted_{action}"], dtype=np.int8), args)
        for action in args.actions}


def decision(split, detector, utilities, threshold, args):
    polluted_x = np.asarray(split["x_polluted"], dtype=np.float32)
    clean_x = np.asarray(split["x_clean"], dtype=np.float32)
    polluted_risk = probe_scores(detector["probe"], polluted_x)
    clean_risk = probe_scores(detector["probe"], clean_x)
    polluted_probability = np.stack(
        [probe_scores(utilities[action], polluted_x) for action in args.actions], axis=1)
    clean_probability = np.stack(
        [probe_scores(utilities[action], clean_x) for action in args.actions], axis=1)
    polluted_choice = polluted_probability.argmax(axis=1)
    clean_choice = clean_probability.argmax(axis=1)
    first_polluted = np.asarray(split["first_trigger_polluted"], dtype=bool)
    first_clean = np.asarray(split["first_trigger_clean"], dtype=bool)
    # A second action is reserved for samples not already acted on at step 1.
    second_polluted = ((~first_polluted)
                       & (polluted_risk >= detector["threshold"])
                       & (polluted_probability.max(axis=1) >= threshold))
    second_clean = ((~first_clean)
                    & (clean_risk >= detector["threshold"])
                    & (clean_probability.max(axis=1) >= threshold))
    polluted_utilities = np.stack([
        np.asarray(split[f"utility_polluted_{action}"], dtype=np.int8)
        for action in args.actions], axis=1)
    clean_utilities = np.stack([
        np.asarray(split[f"utility_clean_{action}"], dtype=np.int8)
        for action in args.actions], axis=1)
    selected_polluted = np.where(
        second_polluted,
        polluted_utilities[np.arange(len(polluted_x)), polluted_choice], 0).astype(np.int8)
    selected_clean = np.where(
        second_clean, clean_utilities[np.arange(len(clean_x)), clean_choice], 0).astype(np.int8)
    union_clean = first_clean | second_clean
    return {"samples": int(len(polluted_x)),
            "incremental_gain": float(selected_polluted.mean()),
            "corrected": int((selected_polluted == 1).sum()),
            "harmed": int((selected_polluted == -1).sum()),
            "second_polluted_trigger_rate": float(second_polluted.mean()),
            "second_clean_trigger_rate": float(second_clean.mean()),
            "first_clean_trigger_rate": float(first_clean.mean()),
            "union_clean_trigger_rate": float(union_clean.mean()),
            "clean_accuracy_delta": float(selected_clean.mean()),
            "action_counts": {str(action): int((second_polluted
                                                  & (polluted_choice == i)).sum())
                              for i, action in enumerate(args.actions)},
            "selected_polluted_utility": selected_polluted,
            "selected_clean_utility": selected_clean}


def select_utility_threshold(data, detector, utilities, seed, args):
    validation = data["splits"]["validation"]
    best = None
    for threshold in np.linspace(0.50, 0.95, 46):
        metric = decision(validation, detector, utilities, threshold, args)
        max_union = args.max_union_clean_fpr
        candidate = {"threshold": float(threshold),
                     "validation_incremental_gain": metric["incremental_gain"],
                     "validation_union_clean_fpr": metric["union_clean_trigger_rate"],
                     "validation_second_trigger_rate": metric["second_polluted_trigger_rate"]}
        if metric["union_clean_trigger_rate"] <= max_union + 1e-12:
            if best is None or (candidate["validation_incremental_gain"], threshold) > (
                    best["validation_incremental_gain"], best["threshold"]):
                best = candidate
    return best if best is not None else {"threshold": 1.0,
                                          "validation_incremental_gain": 0.0,
                                          "validation_union_clean_fpr": float(np.mean(
                                              validation["first_trigger_clean"])),
                                          "validation_second_trigger_rate": 0.0}


def evaluate_all(datasets, args):
    results = []
    for seed in args.seeds:
        data = datasets[seed]
        detector = train_step2_detector(data, seed, args)
        utilities = train_utility_probes(data, args)
        selection = select_utility_threshold(data, detector, utilities, seed, args)
        validation = decision(data["splits"]["validation"], detector, utilities,
                              selection["threshold"], args)
        test = decision(data["splits"]["test"], detector, utilities,
                        selection["threshold"], args)
        test_utility = test.pop("selected_polluted_utility")
        test_clean_utility = test.pop("selected_clean_utility")
        validation.pop("selected_polluted_utility"); validation.pop("selected_clean_utility")
        reference = np.asarray(data["splits"]["test"]["reference_correct_polluted"],
                               dtype=np.int8)
        two_step = reference + test_utility
        test["incremental_gain_ci"] = bootstrap_mean_ci(
            test_utility, args.bootstrap_seed + seed, args.bootstrap_iterations)
        test["clean_accuracy_delta_ci"] = bootstrap_mean_ci(
            test_clean_utility, args.bootstrap_seed + 10000 + seed,
            args.bootstrap_iterations)
        test["mcnemar_incremental"] = mcnemar(two_step, reference)
        results.append({"seed": seed,
                        "detector": {"threshold": detector["threshold"],
                                     "fpr_limit": detector["fpr_limit"],
                                     "selection": detector["selection"],
                                     "validation": detector["validation"],
                                     "test": detector["test"],
                                     "probe": serialize_linear(detector["probe"])},
                        "utility_probes": {str(action): serialize_linear(probe)
                                           for action, probe in utilities.items()},
                        "selection": selection, "validation": validation,
                        "test": test,
                        "test_selected_utility": test_utility.tolist()})
    return results


def aggregate(results, args):
    utility = np.concatenate([np.asarray(x["test_selected_utility"], dtype=np.int8)
                              for x in results])
    return {"samples": int(len(utility)),
            "incremental_gain": bootstrap_mean_ci(
                utility, args.bootstrap_seed + 50000, args.bootstrap_iterations),
            "corrected": int((utility == 1).sum()),
            "harmed": int((utility == -1).sum()),
            "mcnemar": mcnemar(utility.clip(min=0), (-utility).clip(min=0)),
            "max_union_clean_fpr": max(x["test"]["union_clean_trigger_rate"]
                                       for x in results),
            "mean_second_polluted_trigger_rate": float(np.mean(
                [x["test"]["second_polluted_trigger_rate"] for x in results]))}


def strip_test_arrays(results):
    return [{key: value for key, value in item.items()
             if key != "test_selected_utility"} for item in results]


def plot_results(results, aggregate_result, path):
    seeds = [str(x["seed"]) for x in results]
    gains = [x["test"]["incremental_gain"] * 100 for x in results]
    fprs = [x["test"]["union_clean_trigger_rate"] * 100 for x in results]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(seeds, gains); axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Step-2 incremental utility", ylabel="Gain vs frozen step-1 (pp)")
    axes[1].bar(seeds, fprs); axes[1].axhline(5, color="red", linestyle="--")
    axes[1].set(title="First+second clean union triggers", ylabel="Rate (%)")
    for axis in axes: axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Overall incremental gain: {aggregate_result['incremental_gain']['estimate']:.2%}")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Level 6.17.1 dedicated step-2 policy")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--actions", nargs="+", type=float, default=ACTIONS)
    parser.add_argument("--swap-after", type=int, default=SWAP_AFTER)
    parser.add_argument("--swap-times", nargs="+", type=int, default=[SWAP_AFTER])
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--level6-16-1-root", default="experiments/level6_16_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-seed", type=int, default=200)
    parser.add_argument("--train-seeds", type=int, default=2)
    parser.add_argument("--validation-seeds", type=int, default=1)
    parser.add_argument("--test-seeds", type=int, default=1)
    parser.add_argument("--data-seed-base", type=int, default=8750000)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--probe-lr", type=float, default=0.03)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-step2-fpr", type=float, default=0.05)
    parser.add_argument("--seed1001-step2-fpr", type=float, default=0.03)
    parser.add_argument("--max-union-clean-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=617100)
    parser.add_argument("--output", default="experiments/level6_17_1/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_seed % args.eval_batch_size:
        raise ValueError("samples-per-seed must be divisible by eval-batch-size")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    first_policies, fingerprints = load_frozen_policy(args)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "primary_endpoint": "step2 incremental gain vs frozen dynamic one-step",
        "success": {"overall_incremental_ci_lower": "> 0",
                    "seed1001_incremental_gain": ">= 0",
                    "every_model_union_clean_fpr": "<= 0.05"},
        "step2_only_for_step1_misses": True,
        "seed1001_step2_clean_fpr_limit": args.seed1001_step2_fpr,
        "frozen_first_policy_fingerprints": fingerprints,
        "protocol": vars(args)}
    save(root / "preregistration.json", preregistration)
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    dataset_path = root / "step2_datasets.json"
    if dataset_path.exists() and not args.force:
        datasets = {int(k): v for k, v in
                    json.loads(dataset_path.read_text(encoding="utf-8")).items()}
    else:
        datasets = {}
        for seed in args.seeds:
            model = load_model(seed, args, device)
            datasets[seed] = collect_data(
                model, seed, choose_candidates(registration, seed),
                first_policies[(seed, args.swap_after)], args, device, dtype)
            save(dataset_path, datasets); torch.cuda.empty_cache()
    results = evaluate_all(datasets, args)
    aggregate_result = aggregate(results, args)
    seed1001 = next(x for x in results if x["seed"] == 1001)
    passed = (aggregate_result["incremental_gain"]["ci95"][0] > 0
              and seed1001["test"]["incremental_gain"] >= 0
              and aggregate_result["max_union_clean_fpr"] <= args.max_union_clean_fpr)
    summary = {"preregistration": preregistration,
               "results": strip_test_arrays(results),
               "aggregate": aggregate_result, "success": {"passed": passed}}
    save(root / "summary.json", summary)
    plot_results(results, aggregate_result, root / "step2_incremental.png")
    print(json.dumps({"aggregate": aggregate_result,
                      "seed1001_incremental": seed1001["test"]["incremental_gain"],
                      "passed": passed}, indent=2))


if __name__ == "__main__": main()
