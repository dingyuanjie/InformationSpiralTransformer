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
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks
from run_level6_13_1_local import bootstrap_mean_ci, holm, mcnemar, save
from run_level6_14_1_local import choose_candidates, swap_slots


SEEDS = [606, 808, 1001]
FIXED_SCALES = [1.0, 0.75, 0.5, 0.25, 0.0]
ONE_STEP_SCALES = [0.5, 0.25, 0.0]
ADAPTIVE_THRESHOLDS = [-0.25, 0.0, 0.25, 0.5]
SWAP_TIMES = [4, 8]


def defense_name(kind, value, swap_after=None):
    suffix = str(value).replace("-", "m").replace(".", "p")
    if kind == "fixed":
        return f"fixed_{suffix}"
    if kind == "adaptive":
        return f"adaptive_{suffix}"
    return f"one_step_after{swap_after}_{suffix}"


def set_defense(model, condition, chunk_number):
    block = model.blocks[2]
    memory = model.blocks[2].memory
    memory.propagation_consistency_threshold = None
    memory.propagation_consistency_temperature = condition["temperature"]
    block.historical_consistency_threshold = None
    block.historical_consistency_temperature = condition["temperature"]
    if condition["defense_kind"] == "fixed":
        memory.propagation_scale = condition["defense_value"]
        block.historical_read_scale = condition["defense_value"]
    elif condition["defense_kind"] == "adaptive":
        memory.propagation_scale = 1.0
        memory.propagation_consistency_threshold = condition["defense_value"]
        block.historical_read_scale = 1.0
        block.historical_consistency_threshold = condition["defense_value"]
    elif condition["defense_kind"] == "one_step":
        scale = (
            condition["defense_value"]
            if chunk_number == condition["defense_after"] else 1.0
        )
        memory.propagation_scale = scale
        block.historical_read_scale = scale
    else:
        raise ValueError(condition["defense_kind"])


@torch.no_grad()
def evaluate(model, args, condition, device, dtype, eval_seeds):
    predictions = []
    targets = []
    donor_targets = []
    multiplier_sum = 0.0
    multiplier_steps = 0
    batches = args.samples_per_eval_seed // args.eval_batch_size
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device
            )
            memory = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(args.chunks):
                    chunk_number = chunk_index + 1
                    set_defense(model, condition, chunk_number)
                    logits, produced = model(
                        chunks[:, chunk_index], memory=memory, return_memory=True,
                        per_layer_memory=True,
                    )
                    memory = produced
                    multiplier_sum += float(
                        model.blocks[2].memory.last_diagnostics["propagation_multiplier_mean"]
                    )
                    multiplier_steps += 1
                    if (condition["mode"] == "pollution"
                            and chunk_number == condition["swap_after"]):
                        memory = swap_slots(memory, condition["source_slots"])
            prediction = logits[:, -1, :16].argmax(-1)
            predictions.extend(prediction.cpu().tolist())
            targets.extend(target.cpu().tolist())
            donor_targets.extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
    model.blocks[2].memory.propagation_scale = 1.0
    model.blocks[2].memory.propagation_consistency_threshold = None
    model.blocks[2].historical_read_scale = 1.0
    model.blocks[2].historical_consistency_threshold = None
    prediction = np.asarray(predictions)
    target = np.asarray(targets)
    donor = np.asarray(donor_targets)
    correct = (prediction == target).astype(np.int8)
    mismatch = donor != target
    donor_hit = ((prediction == donor) & mismatch).astype(np.int8)
    return {
        "samples": int(len(target)), "accuracy": float(correct.mean()),
        "donor_attraction": float(donor_hit.sum() / mismatch.sum()),
        "propagation_multiplier_mean": multiplier_sum / multiplier_steps,
        "correct": correct.tolist(), "donor_hit": donor_hit.tolist(),
        "donor_mismatch": mismatch.astype(np.int8).tolist(),
    }


def make_defenses(args):
    defenses = []
    for scale in args.fixed_scales:
        defenses.append({"defense_kind": "fixed", "defense_value": scale,
                         "defense_after": None, "temperature": args.adaptive_temperature,
                         "defense": defense_name("fixed", scale)})
    for threshold in args.adaptive_thresholds:
        defenses.append({"defense_kind": "adaptive", "defense_value": threshold,
                         "defense_after": None, "temperature": args.adaptive_temperature,
                         "defense": defense_name("adaptive", threshold)})
    return defenses


def make_conditions(candidates, args):
    conditions = {}
    global_defenses = make_defenses(args)
    for defense in global_defenses:
        name = f"clean__{defense['defense']}"
        conditions[name] = {"name": name, "mode": "clean", "pair": None,
                            "source_slots": None, "swap_after": None, **defense}
    for swap_after in args.swap_times:
        for scale in args.one_step_scales:
            defense = {"defense_kind": "one_step", "defense_value": scale,
                       "defense_after": swap_after + 1,
                       "temperature": args.adaptive_temperature,
                       "defense": defense_name("one_step", scale, swap_after)}
            name = f"clean__{defense['defense']}"
            conditions[name] = {"name": name, "mode": "clean", "pair": None,
                                "source_slots": None, "swap_after": None, **defense}
    for candidate in candidates:
        for swap_after in args.swap_times:
            defenses = list(global_defenses)
            for scale in args.one_step_scales:
                defenses.append({
                    "defense_kind": "one_step", "defense_value": scale,
                    "defense_after": swap_after + 1,
                    "temperature": args.adaptive_temperature,
                    "defense": defense_name("one_step", scale, swap_after),
                })
            for defense in defenses:
                name = f"{candidate['key']}__swap{swap_after}__{defense['defense']}"
                conditions[name] = {
                    "name": name, "mode": "pollution", "pair": candidate["key"],
                    "source_slots": candidate["slots"], "category": candidate["category"],
                    "swap_after": swap_after, **defense,
                }
    return list(conditions.values())


def masked_donor(row):
    mask = np.asarray(row["donor_mismatch"], dtype=bool)
    return np.asarray(row["donor_hit"], dtype=np.int8)[mask]


def analyze(seed, candidates, rows, args):
    table = {row["name"]: row for row in rows}
    clean = {}
    for name, row in table.items():
        if name.startswith("clean__"):
            clean[row["defense"]] = {
                "accuracy": row["accuracy"], "donor_attraction": row["donor_attraction"],
                "propagation_multiplier_mean": row["propagation_multiplier_mean"],
            }
    baseline_clean = clean[defense_name("fixed", 1.0)]["accuracy"]
    clean_fixed = []
    for scale in args.fixed_scales:
        name = defense_name("fixed", scale)
        clean_fixed.append({"scale": scale, **clean[name],
                            "clean_drop": baseline_clean - clean[name]["accuracy"]})
    clean_adaptive = []
    for threshold in args.adaptive_thresholds:
        name = defense_name("adaptive", threshold)
        clean_adaptive.append({"threshold": threshold, **clean[name],
                               "clean_drop": baseline_clean - clean[name]["accuracy"]})

    experiments = []
    for candidate in candidates:
        for swap_after in args.swap_times:
            prefix = f"{candidate['key']}__swap{swap_after}"
            baseline = table[f"{prefix}__{defense_name('fixed', 1.0)}"]
            baseline_correct = np.asarray(baseline["correct"], dtype=np.int8)
            baseline_donor = masked_donor(baseline)
            entries = []
            tests_accuracy = []
            tests_donor = []
            defense_names = [defense_name("fixed", x) for x in args.fixed_scales]
            defense_names += [defense_name("adaptive", x) for x in args.adaptive_thresholds]
            defense_names += [defense_name("one_step", x, swap_after)
                              for x in args.one_step_scales]
            for index, name in enumerate(defense_names):
                row = table[f"{prefix}__{name}"]
                repaired = np.asarray(row["correct"], dtype=np.int8)
                repaired_donor = masked_donor(row)
                accuracy_test = mcnemar(repaired, baseline_correct)
                donor_test = mcnemar(repaired_donor, baseline_donor)
                tests_accuracy.append(accuracy_test)
                tests_donor.append(donor_test)
                clean_metric = clean[name]
                recoverable = clean_metric["accuracy"] - baseline["accuracy"]
                entries.append({
                    "defense": name, "defense_kind": row["defense_kind"],
                    "defense_value": row["defense_value"],
                    "accuracy": row["accuracy"], "donor_attraction": row["donor_attraction"],
                    "clean_accuracy": clean_metric["accuracy"],
                    "clean_drop": baseline_clean - clean_metric["accuracy"],
                    "pollution_gap": clean_metric["accuracy"] - row["accuracy"],
                    "recovery_vs_baseline": bootstrap_mean_ci(
                        repaired - baseline_correct,
                        args.bootstrap_seed + seed * 10000 + swap_after * 100 + index,
                        args.bootstrap_iterations),
                    "recovery_fraction": ((row["accuracy"] - baseline["accuracy"]) / recoverable
                                          if recoverable > 1e-12 else None),
                    "donor_reduction": bootstrap_mean_ci(
                        baseline_donor - repaired_donor,
                        args.bootstrap_seed + seed * 10000 + 5000 + swap_after * 100 + index,
                        args.bootstrap_iterations),
                    "mcnemar_accuracy": accuracy_test, "mcnemar_donor": donor_test,
                    "propagation_multiplier_mean": row["propagation_multiplier_mean"],
                })
            holm(tests_accuracy)
            holm(tests_donor)
            for entry, accuracy_test, donor_test in zip(entries, tests_accuracy, tests_donor):
                entry["mcnemar_accuracy"] = accuracy_test
                entry["mcnemar_donor"] = donor_test
            pareto = [entry for entry in entries if entry["clean_drop"] <= 0.01]
            pareto.sort(key=lambda entry: (entry["pollution_gap"], -entry["accuracy"]))
            experiments.append({
                **candidate, "swap_after": swap_after,
                "baseline_polluted_accuracy": baseline["accuracy"],
                "baseline_donor_attraction": baseline["donor_attraction"],
                "entries": entries, "best_clean_within_1pp": pareto[:5],
            })
    return {"seed": seed, "baseline_clean_accuracy": baseline_clean,
            "clean_fixed": clean_fixed, "clean_adaptive": clean_adaptive,
            "experiments": experiments}


def plot_seed(result, path, args):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fixed = result["clean_fixed"]
    axes[0, 0].plot([x["scale"] for x in fixed], [x["accuracy"] * 100 for x in fixed],
                    marker="o")
    axes[0, 0].set_title("Clean accuracy: fixed propagation scale")
    axes[0, 0].set_xlabel("Scale")
    axes[0, 0].set_ylabel("Accuracy (%)")
    adaptive = result["clean_adaptive"]
    axes[0, 1].plot([x["threshold"] for x in adaptive],
                    [x["accuracy"] * 100 for x in adaptive], marker="o")
    axes[0, 1].set_title("Clean accuracy: adaptive consistency gate")
    axes[0, 1].set_xlabel("Cosine threshold")
    axes[0, 1].set_ylabel("Accuracy (%)")
    for experiment in result["experiments"]:
        fixed_entries = [x for x in experiment["entries"] if x["defense_kind"] == "fixed"]
        label = f"{experiment['key']}@{experiment['swap_after']}"
        axes[1, 0].plot([x["defense_value"] for x in fixed_entries],
                        [x["accuracy"] * 100 for x in fixed_entries], marker="o", label=label)
        axes[1, 1].plot([x["defense_value"] for x in fixed_entries],
                        [x["donor_attraction"] * 100 for x in fixed_entries],
                        marker="o", label=label)
    axes[1, 0].set_title("Polluted accuracy: fixed scale")
    axes[1, 1].set_title("Donor attraction: fixed scale")
    for axis in axes[1]:
        axis.set_xlabel("Scale")
        axis.legend(fontsize=6)
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylabel("Donor-target prediction (%)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(f"IST seed {result['seed']}: inference-time pollution defenses")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_seed(seed, candidates, args, device, dtype, root):
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    source = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(source, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    model.load_state_dict(state["model"])
    eval_seeds = [args.eval_seed_base + seed * 100 + index
                  for index in range(args.eval_seed_count)]
    specs = make_conditions(candidates, args)
    progress_path = folder / "predictions.json"
    rows = [] if args.force or not progress_path.exists() else json.loads(
        progress_path.read_text(encoding="utf-8")
    )
    done = {row["name"] for row in rows}
    for index, condition in enumerate(specs, start=1):
        if condition["name"] in done:
            continue
        metric = evaluate(model, args, condition, device, dtype, eval_seeds)
        rows.append({**condition, **metric})
        save(progress_path, rows)
        print(f"seed={seed} [{index}/{len(specs)}] {condition['name']} "
              f"acc={metric['accuracy']:.2%} donor={metric['donor_attraction']:.2%}", flush=True)
    result = analyze(seed, candidates, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    plot_seed(result, folder / "defense_tradeoff.png", args)
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.15 inference-time pollution defenses")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--fixed-scales", nargs="+", type=float, default=FIXED_SCALES)
    parser.add_argument("--one-step-scales", nargs="+", type=float, default=ONE_STEP_SCALES)
    parser.add_argument("--adaptive-thresholds", nargs="+", type=float,
                        default=ADAPTIVE_THRESHOLDS)
    parser.add_argument("--adaptive-temperature", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=400)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--eval-seed-base", type=int, default=7150000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=615000)
    parser.add_argument("--output", default="experiments/level6_15/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
    if 1.0 not in args.fixed_scales:
        raise ValueError("fixed-scales must include the 1.0 baseline")
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
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        candidates = choose_candidates(registration, seed)
        results.append(run_seed(seed, candidates, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    save(root / "summary.json", {"protocol": vars(args), "runs": results})
    print(json.dumps({str(result["seed"]): {
        "clean": result["baseline_clean_accuracy"],
        "experiments": len(result["experiments"]),
    } for result in results}, indent=2))


if __name__ == "__main__":
    main()
