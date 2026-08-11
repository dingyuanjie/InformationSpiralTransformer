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
from run_level6_16_local import load_model, observable_features, reset_control
from run_level6_16_2_local import decide, load_frozen_policy


SEEDS = [606, 808, 1001]
ACTIONS = [0.20, 0.25]
SWAP_AFTER = 4
STRATEGIES = [
    "baseline", "fixed_0p20_one", "fixed_0p25_one",
    "fixed_0p20_two", "fixed_0p25_two", "dynamic_one", "dynamic_two",
]


def fixed_scale(strategy, relative_step):
    if strategy == "fixed_0p20_one" and relative_step == 1:
        return 0.20
    if strategy == "fixed_0p25_one" and relative_step == 1:
        return 0.25
    if strategy == "fixed_0p20_two" and relative_step in (1, 2):
        return 0.20
    if strategy == "fixed_0p25_two" and relative_step in (1, 2):
        return 0.25
    return 1.0


def selected_scale(trigger, choice, actions, device):
    scale = torch.ones(len(trigger), device=device, dtype=torch.float32)
    trigger_tensor = torch.as_tensor(trigger, device=device, dtype=torch.bool)
    choice_tensor = torch.as_tensor(choice, device=device, dtype=torch.long)
    for index, action in enumerate(actions):
        mask = trigger_tensor & (choice_tensor == index)
        scale[mask] = action
    return scale[:, None, None]


@torch.no_grad()
def trajectory_batch(model, chunks, slots, policy, args, dtype):
    memories = {strategy: None for strategy in STRATEGIES}
    logits = {}
    trigger_any = {"dynamic_one": np.zeros(chunks.shape[0], dtype=bool),
                   "dynamic_two": np.zeros(chunks.shape[0], dtype=bool)}
    step_triggers = {name: {"1": np.zeros(chunks.shape[0], dtype=bool),
                            "2": np.zeros(chunks.shape[0], dtype=bool)}
                     for name in ["dynamic_one", "dynamic_two"]}
    step_choices = {name: {"1": np.zeros(chunks.shape[0], dtype=np.int8),
                           "2": np.zeros(chunks.shape[0], dtype=np.int8)}
                    for name in ["dynamic_one", "dynamic_two"]}
    with torch.autocast(device_type="cuda", dtype=dtype):
        for chunk_index in range(chunks.shape[1]):
            chunk_number = chunk_index + 1
            relative_step = chunk_number - args.swap_after
            for strategy in STRATEGIES:
                if strategy.startswith("dynamic") and (
                        relative_step == 1 or
                        (strategy == "dynamic_two" and relative_step == 2)):
                    reset_control(model)
                    _, _ = model(chunks[:, chunk_index], memory=memories[strategy],
                                 return_memory=True, per_layer_memory=True)
                    feature = observable_features(model).float().cpu()
                    trigger, choice, _, _ = decide(policy, feature, args.actions)
                    step = str(relative_step)
                    step_triggers[strategy][step] = trigger
                    step_choices[strategy][step] = choice.astype(np.int8)
                    trigger_any[strategy] |= trigger
                    reset_control(model)
                    model.blocks[2].memory.propagation_scale = selected_scale(
                        trigger, choice, args.actions, chunks.device)
                else:
                    reset_control(model)
                    if strategy.startswith("fixed"):
                        model.blocks[2].memory.propagation_scale = fixed_scale(
                            strategy, relative_step)
                logits[strategy], memories[strategy] = model(
                    chunks[:, chunk_index], memory=memories[strategy],
                    return_memory=True, per_layer_memory=True)
            if chunk_number == args.swap_after and slots:
                memories = {name: swap_slots(memory, slots)
                            for name, memory in memories.items()}
    reset_control(model)
    predictions = {name: value[:, -1, :16].argmax(-1).cpu().numpy()
                   for name, value in logits.items()}
    return predictions, trigger_any, step_triggers, step_choices


def extend_nested(storage, source):
    for strategy in source:
        for step in source[strategy]:
            storage[strategy][step].extend(source[strategy][step].tolist())


@torch.no_grad()
def evaluate_condition(model, policy, seed, candidate, args, device, dtype):
    predictions = {mode: {strategy: [] for strategy in STRATEGIES}
                   for mode in ["polluted", "clean"]}
    triggers = {mode: {name: [] for name in ["dynamic_one", "dynamic_two"]}
                for mode in ["polluted", "clean"]}
    step_triggers = {mode: {name: {"1": [], "2": []}
                            for name in ["dynamic_one", "dynamic_two"]}
                     for mode in ["polluted", "clean"]}
    step_choices = {mode: {name: {"1": [], "2": []}
                           for name in ["dynamic_one", "dynamic_two"]}
                    for mode in ["polluted", "clean"]}
    targets, donors = [], []
    eval_seeds = [args.eval_seed_base + seed * 100 + i
                  for i in range(args.eval_seed_count)]
    batches = args.samples_per_eval_seed // args.eval_batch_size
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device)
            for mode, slots in [("polluted", candidate["slots"]), ("clean", [])]:
                batch_predictions, batch_triggers, batch_step_triggers, batch_choices = (
                    trajectory_batch(model, chunks, slots, policy, args, dtype))
                for strategy in STRATEGIES:
                    predictions[mode][strategy].extend(batch_predictions[strategy].tolist())
                for name in ["dynamic_one", "dynamic_two"]:
                    triggers[mode][name].extend(batch_triggers[name].tolist())
                extend_nested(step_triggers[mode], batch_step_triggers)
                extend_nested(step_choices[mode], batch_choices)
            targets.extend(target.cpu().tolist())
            donors.extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
    target = np.asarray(targets)
    donor = np.asarray(donors)
    donor_mask = donor != target
    correct = {mode: {strategy: (np.asarray(values) == target).astype(np.int8)
                      for strategy, values in predictions[mode].items()}
               for mode in predictions}
    baseline = correct["polluted"]["baseline"]
    metrics = {}
    for index, strategy in enumerate(STRATEGIES[1:]):
        strategy_correct = correct["polluted"][strategy]
        clean_base = correct["clean"]["baseline"]
        clean_strategy = correct["clean"][strategy]
        base_donor = ((np.asarray(predictions["polluted"]["baseline"]) == donor)
                      & donor_mask).astype(np.int8)[donor_mask]
        strategy_donor = ((np.asarray(predictions["polluted"][strategy]) == donor)
                          & donor_mask).astype(np.int8)[donor_mask]
        metric = {
            "accuracy": float(strategy_correct.mean()),
            "accuracy_gain": bootstrap_mean_ci(
                strategy_correct - baseline,
                args.bootstrap_seed + seed * 100 + index, args.bootstrap_iterations),
            "clean_accuracy": float(clean_strategy.mean()),
            "clean_accuracy_delta": bootstrap_mean_ci(
                clean_strategy - clean_base,
                args.bootstrap_seed + 100000 + seed * 100 + index,
                args.bootstrap_iterations),
            "donor_reduction": bootstrap_mean_ci(
                base_donor - strategy_donor,
                args.bootstrap_seed + 200000 + seed * 100 + index,
                args.bootstrap_iterations),
            "corrected_samples": int(((strategy_correct == 1) & (baseline == 0)).sum()),
            "harmed_samples": int(((strategy_correct == 0) & (baseline == 1)).sum()),
            "mcnemar": mcnemar(strategy_correct, baseline),
            "correct": strategy_correct.tolist(),
            "clean_correct": clean_strategy.tolist(),
        }
        if strategy.startswith("dynamic"):
            metric["polluted_trigger_rate"] = float(np.mean(triggers["polluted"][strategy]))
            metric["clean_trigger_rate"] = float(np.mean(triggers["clean"][strategy]))
            metric["step_trigger_rates"] = {
                step: {"polluted": float(np.mean(step_triggers["polluted"][strategy][step])),
                       "clean": float(np.mean(step_triggers["clean"][strategy][step]))}
                for step in ["1", "2"]}
            metric["action_counts"] = {
                step: {str(action): int(np.sum(
                    np.asarray(step_triggers["polluted"][strategy][step], dtype=bool)
                    & (np.asarray(step_choices["polluted"][strategy][step]) == action_index)))
                    for action_index, action in enumerate(args.actions)}
                for step in ["1", "2"]}
        metrics[strategy] = metric
    return {"seed": seed, "pair": candidate["key"], "slots": candidate["slots"],
            "category": candidate["category"], "swap_after": args.swap_after,
            "eval_seeds": eval_seeds, "samples": int(len(target)),
            "baseline_accuracy": float(baseline.mean()),
            "clean_baseline_accuracy": float(correct["clean"]["baseline"].mean()),
            "baseline_correct": baseline.tolist(),
            "clean_baseline_correct": correct["clean"]["baseline"].tolist(),
            "strategies": metrics}


def aggregate(rows, args):
    strategies = STRATEGIES[1:]
    output = {}
    baseline = np.concatenate([np.asarray(row["baseline_correct"], dtype=np.int8)
                               for row in rows])
    clean_baseline = np.concatenate([
        np.asarray(row["clean_baseline_correct"], dtype=np.int8) for row in rows])
    for index, strategy in enumerate(strategies):
        correct = np.concatenate([np.asarray(row["strategies"][strategy]["correct"], dtype=np.int8)
                                  for row in rows])
        clean = np.concatenate([np.asarray(row["strategies"][strategy]["clean_correct"], dtype=np.int8)
                                for row in rows])
        metric = {"samples": int(len(correct)),
                  "accuracy_gain": bootstrap_mean_ci(
                      correct - baseline, args.bootstrap_seed + 300000 + index,
                      args.bootstrap_iterations),
                  "clean_accuracy_delta": bootstrap_mean_ci(
                      clean - clean_baseline, args.bootstrap_seed + 400000 + index,
                      args.bootstrap_iterations),
                  "corrected_samples": int(((correct == 1) & (baseline == 0)).sum()),
                  "harmed_samples": int(((correct == 0) & (baseline == 1)).sum()),
                  "mcnemar": mcnemar(correct, baseline)}
        if strategy.startswith("dynamic"):
            metric["max_clean_trigger_rate"] = max(
                row["strategies"][strategy]["clean_trigger_rate"] for row in rows)
            metric["mean_polluted_trigger_rate"] = float(np.mean(
                [row["strategies"][strategy]["polluted_trigger_rate"] for row in rows]))
        output[strategy] = metric
    dynamic_one = np.concatenate([
        np.asarray(row["strategies"]["dynamic_one"]["correct"], dtype=np.int8)
        for row in rows])
    dynamic_two = np.concatenate([
        np.asarray(row["strategies"]["dynamic_two"]["correct"], dtype=np.int8)
        for row in rows])
    output["dynamic_two_vs_one"] = {
        "gain": bootstrap_mean_ci(dynamic_two - dynamic_one,
                                  args.bootstrap_seed + 500000,
                                  args.bootstrap_iterations),
        "corrected_samples": int(((dynamic_two == 1) & (dynamic_one == 0)).sum()),
        "harmed_samples": int(((dynamic_two == 0) & (dynamic_one == 1)).sum()),
        "mcnemar": mcnemar(dynamic_two, dynamic_one)}
    return output


def strip_arrays(rows):
    output = []
    for row in rows:
        item = {key: value for key, value in row.items()
                if key not in ["baseline_correct", "clean_baseline_correct", "strategies"]}
        item["strategies"] = {}
        for strategy, metric in row["strategies"].items():
            item["strategies"][strategy] = {
                key: value for key, value in metric.items()
                if key not in ["correct", "clean_correct"]}
        output.append(item)
    return output


def plot_summary(aggregate_result, path):
    names = STRATEGIES[1:]
    gains = [aggregate_result[name]["accuracy_gain"]["estimate"] * 100 for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(range(len(names)), gains)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=8)
    axes[0].set(title="Chunk-4 counterfactual comparison", ylabel="Accuracy gain (pp)")
    dynamic = ["dynamic_one", "dynamic_two"]
    axes[1].bar(dynamic, [aggregate_result[x]["max_clean_trigger_rate"] * 100
                          for x in dynamic])
    axes[1].axhline(5, color="red", linestyle="--")
    axes[1].set(title="Maximum matched-clean trigger rate", ylabel="Rate (%)")
    for axis in axes: axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Level 6.17 early-pollution multistep defense")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--actions", nargs="+", type=float, default=ACTIONS)
    parser.add_argument("--swap-after", type=int, default=SWAP_AFTER)
    parser.add_argument("--swap-times", nargs="+", type=int, default=[SWAP_AFTER],
                        help="Compatibility field used to load the frozen policy")
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--level6-16-1-root", default="experiments/level6_16_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=200)
    parser.add_argument("--eval-seed-count", type=int, default=4)
    parser.add_argument("--eval-seed-base", type=int, default=8550000)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=617000)
    parser.add_argument("--max-clean-fpr", type=float, default=0.05)
    parser.add_argument("--output", default="experiments/level6_17/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.swap_times != [args.swap_after]:
        raise ValueError("Level 6.17 is preregistered for chunk 4 only")
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    policies, fingerprints = load_frozen_policy(args)
    checkpoint_seeds = sorted(int(path.parent.name.replace("seed", "")) for path in
                              Path(args.level6_8_root).glob("seed*/withdrawal_phase3.pt"))
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "primary": "dynamic_two accuracy gain vs dynamic_one > 0",
        "joint_success": {"dynamic_two_vs_baseline_ci_lower": "> 0",
                          "dynamic_two_vs_one_gain": "> 0",
                          "dynamic_two_max_clean_fpr": "<= 0.05",
                          "seed808_dynamic_two_gain": ">= -0.0025"},
        "frozen_policy_fingerprints": fingerprints,
        "counterfactuals": STRATEGIES,
        "available_checkpoint_seeds": checkpoint_seeds,
        "missing_fourth_checkpoint": len(checkpoint_seeds) < 4,
        "protocol": vars(args)}
    save(root / "preregistration.json", preregistration)
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    partial_path = root / "counterfactuals.partial.json"
    rows = [] if args.force or not partial_path.exists() else json.loads(
        partial_path.read_text(encoding="utf-8"))
    done = {(row["seed"], row["pair"]) for row in rows}
    for seed in args.seeds:
        model = load_model(seed, args, device)
        for candidate in choose_candidates(registration, seed):
            if (seed, candidate["key"]) in done: continue
            row = evaluate_condition(model, policies[(seed, args.swap_after)],
                                     seed, candidate, args, device, dtype)
            rows.append(row); save(partial_path, rows)
            print(f"seed={seed} pair={candidate['key']} "
                  f"one={row['strategies']['dynamic_one']['accuracy_gain']['estimate']:+.2%} "
                  f"two={row['strategies']['dynamic_two']['accuracy_gain']['estimate']:+.2%}",
                  flush=True)
        torch.cuda.empty_cache()
    aggregate_result = aggregate(rows, args)
    seed808_rows = [row for row in rows if row["seed"] == 808]
    seed808_aggregate = aggregate(seed808_rows, args)["dynamic_two"]
    two = aggregate_result["dynamic_two"]
    comparison = aggregate_result["dynamic_two_vs_one"]
    passed = (two["accuracy_gain"]["ci95"][0] > 0
              and comparison["gain"]["estimate"] > 0
              and two["max_clean_trigger_rate"] <= args.max_clean_fpr
              and seed808_aggregate["accuracy_gain"]["estimate"] >= -0.0025)
    summary = {"preregistration": preregistration,
               "conditions": strip_arrays(rows), "aggregate": aggregate_result,
               "seed808_dynamic_two": seed808_aggregate,
               "success": {"passed": passed}}
    save(root / "summary.json", summary)
    plot_summary(aggregate_result, root / "early_pollution_defense.png")
    print(json.dumps({"dynamic_one": aggregate_result["dynamic_one"],
                      "dynamic_two": two, "two_vs_one": comparison,
                      "seed808": seed808_aggregate["accuracy_gain"],
                      "passed": passed}, indent=2))


if __name__ == "__main__": main()
