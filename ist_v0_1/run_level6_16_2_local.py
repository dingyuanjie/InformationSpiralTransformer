import argparse
import hashlib
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
from run_level6_14_1_local import choose_candidates
from run_level6_16_local import load_model, probe_scores
from run_level6_16_1_local import utility_batch


SEEDS = [606, 808, 1001]
SWAP_TIMES = [4, 8]
ACTIONS = [0.20, 0.25]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deserialize_probe(raw):
    return {"mean": np.asarray(raw["mean"], dtype=np.float32),
            "std": np.asarray(raw["std"], dtype=np.float32),
            "weight": np.asarray(raw["weight"], dtype=np.float32),
            "bias": float(raw["bias"])}


def load_frozen_policy(args):
    root = Path(args.level6_16_1_root)
    detector_path = root / "time_detectors.json"
    summary_path = root / "summary.json"
    detectors_raw = json.loads(detector_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policies = {}
    for result in summary["results"]:
        seed = int(result["seed"])
        swap_after = int(result["swap_after"])
        detector_raw = detectors_raw[str(seed)][str(swap_after)]
        policies[(seed, swap_after)] = {
            "detector": deserialize_probe(detector_raw["probe"]),
            "detector_threshold": float(detector_raw["threshold"]),
            "utility_threshold": float(result["selection"]["threshold"]),
            "utility_probes": {
                float(action): deserialize_probe(probe)
                for action, probe in result["utility_probes"].items()},
        }
    expected = {(seed, swap_after) for seed in args.seeds for swap_after in args.swap_times}
    if not expected.issubset(policies):
        raise RuntimeError(f"Frozen policy missing: found {set(policies)}, expected {expected}")
    if summary["protocol"]["actions"] != args.actions:
        raise RuntimeError("Actions differ from the frozen Level 6.16.1 protocol")
    policies = {key: policies[key] for key in expected}
    return policies, {"time_detectors_sha256": file_sha256(detector_path),
                      "level6_16_1_summary_sha256": file_sha256(summary_path)}


def decide(policy, features, actions):
    x = features.numpy().astype(np.float32)
    risk = probe_scores(policy["detector"], x)
    utility = np.stack([probe_scores(policy["utility_probes"][action], x)
                        for action in actions], axis=1)
    choice = utility.argmax(axis=1)
    trigger = ((risk >= policy["detector_threshold"])
               & (utility[np.arange(len(x)), choice] >= policy["utility_threshold"]))
    return trigger, choice, risk, utility


def selected_prediction(predictions, trigger, choice, actions):
    output = predictions["baseline"].numpy().copy()
    for index, action in enumerate(actions):
        mask = trigger & (choice == index)
        output[mask] = predictions[str(action)].numpy()[mask]
    return output


@torch.no_grad()
def evaluate_condition(model, policy, seed, candidate, swap_after,
                       args, device, dtype):
    arrays = {key: [] for key in [
        "target", "donor", "polluted_baseline_prediction", "polluted_policy_prediction",
        "clean_baseline_prediction", "clean_policy_prediction",
        "polluted_trigger", "clean_trigger", "polluted_choice", "clean_choice",
    ]}
    eval_seeds = [args.eval_seed_base + seed * 100 + i
                  for i in range(args.eval_seed_count)]
    batches = args.samples_per_eval_seed // args.eval_batch_size
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device)
            polluted_feature, polluted_predictions = utility_batch(
                model, chunks, candidate["slots"], swap_after, args.actions, dtype)
            clean_feature, clean_predictions = utility_batch(
                model, chunks, [], swap_after, args.actions, dtype)
            polluted_trigger, polluted_choice, _, _ = decide(
                policy, polluted_feature, args.actions)
            clean_trigger, clean_choice, _, _ = decide(
                policy, clean_feature, args.actions)
            arrays["target"].extend(target.cpu().tolist())
            arrays["donor"].extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
            arrays["polluted_baseline_prediction"].extend(
                polluted_predictions["baseline"].tolist())
            arrays["polluted_policy_prediction"].extend(
                selected_prediction(polluted_predictions, polluted_trigger,
                                    polluted_choice, args.actions).tolist())
            arrays["clean_baseline_prediction"].extend(clean_predictions["baseline"].tolist())
            arrays["clean_policy_prediction"].extend(
                selected_prediction(clean_predictions, clean_trigger,
                                    clean_choice, args.actions).tolist())
            arrays["polluted_trigger"].extend(polluted_trigger.tolist())
            arrays["clean_trigger"].extend(clean_trigger.tolist())
            arrays["polluted_choice"].extend(polluted_choice.tolist())
            arrays["clean_choice"].extend(clean_choice.tolist())
    values = {key: np.asarray(value) for key, value in arrays.items()}
    target = values["target"]
    baseline = (values["polluted_baseline_prediction"] == target).astype(np.int8)
    policy_correct = (values["polluted_policy_prediction"] == target).astype(np.int8)
    clean_baseline = (values["clean_baseline_prediction"] == target).astype(np.int8)
    clean_policy = (values["clean_policy_prediction"] == target).astype(np.int8)
    donor_mask = values["donor"] != target
    base_donor = ((values["polluted_baseline_prediction"] == values["donor"])
                  & donor_mask).astype(np.int8)[donor_mask]
    policy_donor = ((values["polluted_policy_prediction"] == values["donor"])
                    & donor_mask).astype(np.int8)[donor_mask]
    accuracy_test = mcnemar(policy_correct, baseline)
    clean_test = mcnemar(clean_policy, clean_baseline)
    return {
        "seed": seed, "pair": candidate["key"], "slots": candidate["slots"],
        "category": candidate["category"], "swap_after": swap_after,
        "eval_seeds": eval_seeds, "samples": int(len(target)),
        "baseline_accuracy": float(baseline.mean()),
        "policy_accuracy": float(policy_correct.mean()),
        "clean_baseline_accuracy": float(clean_baseline.mean()),
        "clean_policy_accuracy": float(clean_policy.mean()),
        "polluted_trigger_rate": float(values["polluted_trigger"].mean()),
        "clean_trigger_rate": float(values["clean_trigger"].mean()),
        "action_counts_polluted": {
            str(action): int((values["polluted_trigger"]
                              & (values["polluted_choice"] == index)).sum())
            for index, action in enumerate(args.actions)},
        "action_counts_clean": {
            str(action): int((values["clean_trigger"]
                              & (values["clean_choice"] == index)).sum())
            for index, action in enumerate(args.actions)},
        "accuracy_gain": bootstrap_mean_ci(
            policy_correct - baseline,
            args.bootstrap_seed + seed * 100 + swap_after, args.bootstrap_iterations),
        "clean_accuracy_delta": bootstrap_mean_ci(
            clean_policy - clean_baseline,
            args.bootstrap_seed + 100000 + seed * 100 + swap_after,
            args.bootstrap_iterations),
        "donor_reduction": bootstrap_mean_ci(
            base_donor - policy_donor,
            args.bootstrap_seed + 200000 + seed * 100 + swap_after,
            args.bootstrap_iterations),
        "corrected_samples": int(((policy_correct == 1) & (baseline == 0)).sum()),
        "harmed_samples": int(((policy_correct == 0) & (baseline == 1)).sum()),
        "mcnemar_accuracy": accuracy_test, "mcnemar_clean": clean_test,
        "baseline_correct": baseline.tolist(), "policy_correct": policy_correct.tolist(),
        "clean_baseline_correct": clean_baseline.tolist(),
        "clean_policy_correct": clean_policy.tolist(),
    }


def aggregate(rows, args):
    def stats_for(subset, seed_offset):
        baseline = np.concatenate([np.asarray(x["baseline_correct"], dtype=np.int8)
                                   for x in subset])
        policy = np.concatenate([np.asarray(x["policy_correct"], dtype=np.int8)
                                 for x in subset])
        clean_base = np.concatenate([np.asarray(x["clean_baseline_correct"], dtype=np.int8)
                                     for x in subset])
        clean_policy = np.concatenate([np.asarray(x["clean_policy_correct"], dtype=np.int8)
                                       for x in subset])
        return {"samples": int(len(baseline)),
                "baseline_accuracy": float(baseline.mean()),
                "policy_accuracy": float(policy.mean()),
                "accuracy_gain": bootstrap_mean_ci(
                    policy - baseline, args.bootstrap_seed + seed_offset,
                    args.bootstrap_iterations),
                "clean_accuracy_delta": bootstrap_mean_ci(
                    clean_policy - clean_base, args.bootstrap_seed + 300000 + seed_offset,
                    args.bootstrap_iterations),
                "corrected_samples": int(((policy == 1) & (baseline == 0)).sum()),
                "harmed_samples": int(((policy == 0) & (baseline == 1)).sum()),
                "mcnemar_accuracy": mcnemar(policy, baseline),
                "polluted_trigger_rate": float(np.mean(
                    [x["polluted_trigger_rate"] for x in subset])),
                "max_clean_trigger_rate": float(max(
                    x["clean_trigger_rate"] for x in subset))}
    by_seed = {str(seed): stats_for([x for x in rows if x["seed"] == seed], seed)
               for seed in args.seeds}
    by_time = {str(time): stats_for([x for x in rows if x["swap_after"] == time],
                                    10000 + time) for time in args.swap_times}
    overall = stats_for(rows, 50000)
    return {"by_seed": by_seed, "by_time": by_time, "overall": overall}


def strip_predictions(row):
    omitted = {"baseline_correct", "policy_correct",
               "clean_baseline_correct", "clean_policy_correct"}
    return {key: value for key, value in row.items() if key not in omitted}


def plot_summary(summary, path):
    seeds = list(summary["aggregate"]["by_seed"])
    gains = [summary["aggregate"]["by_seed"][seed]["accuracy_gain"]["estimate"] * 100
             for seed in seeds]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(seeds, gains); axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Frozen-policy confirmation", xlabel="Model seed",
                ylabel="Accuracy gain (pp)")
    labels, values = [], []
    for row in summary["conditions"]:
        labels.append(f"{row['seed']}@{row['swap_after']}:{row['pair']}")
        values.append(row["clean_trigger_rate"] * 100)
    axes[1].bar(range(len(labels)), values)
    axes[1].axhline(5, color="red", linestyle="--")
    axes[1].set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=6)
    axes[1].set(title="Matched-clean trigger rate", ylabel="Rate (%)")
    for axis in axes: axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Level 6.16.2 frozen confirmation")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--actions", nargs="+", type=float, default=ACTIONS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--level6-16-1-root", default="experiments/level6_16_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=200)
    parser.add_argument("--eval-seed-count", type=int, default=4)
    parser.add_argument("--eval-seed-base", type=int, default=8350000)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=616200)
    parser.add_argument("--seed808-floor", type=float, default=-0.0025)
    parser.add_argument("--max-clean-fpr", type=float, default=0.05)
    parser.add_argument("--output", default="experiments/level6_16_2/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
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
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "frozen_policy_fingerprints": fingerprints,
        "success": {"overall_ci95_lower": "> 0", "seed808_gain": ">= -0.0025",
                    "every_condition_clean_fpr": "<= 0.05",
                    "pooled_corrected_gt_harmed_mcnemar_p": "< 0.05"},
        "no_training_or_threshold_selection": True,
        "samples_per_model_time": (args.samples_per_eval_seed
                                   * args.eval_seed_count * 3),
        "protocol": vars(args)}
    save(root / "preregistration.json", preregistration)
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    partial_path = root / "confirmation.partial.json"
    rows = [] if args.force or not partial_path.exists() else json.loads(
        partial_path.read_text(encoding="utf-8"))
    done = {(x["seed"], x["pair"], x["swap_after"]) for x in rows}
    for seed in args.seeds:
        model = load_model(seed, args, device)
        for candidate in choose_candidates(registration, seed):
            for swap_after in args.swap_times:
                key = (seed, candidate["key"], swap_after)
                if key in done: continue
                row = evaluate_condition(model, policies[(seed, swap_after)], seed,
                                         candidate, swap_after, args, device, dtype)
                rows.append(row); save(partial_path, rows)
                print(f"seed={seed} {candidate['key']}@{swap_after} "
                      f"gain={row['accuracy_gain']['estimate']:+.2%} "
                      f"clean_fpr={row['clean_trigger_rate']:.2%}", flush=True)
        torch.cuda.empty_cache()
    aggregate_result = aggregate(rows, args)
    overall = aggregate_result["overall"]
    seed808 = aggregate_result["by_seed"]["808"]
    max_clean = max(row["clean_trigger_rate"] for row in rows)
    passed = (overall["accuracy_gain"]["ci95"][0] > 0
              and seed808["accuracy_gain"]["estimate"] >= args.seed808_floor
              and max_clean <= args.max_clean_fpr
              and overall["corrected_samples"] > overall["harmed_samples"]
              and overall["mcnemar_accuracy"]["p"] < 0.05)
    summary = {"preregistration": preregistration,
               "conditions": [strip_predictions(row) for row in rows],
               "aggregate": aggregate_result,
               "success": {"max_clean_fpr": max_clean, "passed": passed}}
    save(root / "summary.json", summary)
    plot_summary(summary, root / "frozen_confirmation.png")
    print(json.dumps({"overall": overall["accuracy_gain"],
                      "seed808": seed808["accuracy_gain"],
                      "max_clean_fpr": max_clean, "passed": passed}, indent=2))


if __name__ == "__main__": main()
