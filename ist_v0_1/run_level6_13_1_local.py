import argparse
import json
import math
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


SEEDS = [606, 808, 1001]
SLOTS = 32


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def pair_key(slots):
    return "-".join(str(x) for x in sorted(slots))


def mask_slots(memory, keep):
    mask = torch.zeros(SLOTS, device=memory[2].device, dtype=memory[2].dtype)
    mask[keep] = 1
    output = list(memory)
    output[2] = memory[2] * mask[None, :, None]
    return output


@torch.no_grad()
def predict(model, args, keep, device, dtype, eval_seeds):
    predictions = []
    targets = []
    batches = args.samples_per_eval_seed // args.eval_batch_size
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device
            )
            memory = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for index in range(args.chunks):
                    logits, produced = model(
                        chunks[:, index], memory=memory, return_memory=True,
                        per_layer_memory=True,
                    )
                    memory = produced if keep is None else mask_slots(produced, keep)
            prediction = logits[:, -1, :16].argmax(-1)
            predictions.extend(prediction.cpu().tolist())
            targets.extend(target.cpu().tolist())
    correct = (np.asarray(predictions) == np.asarray(targets)).astype(np.int8)
    return {
        "samples": int(len(correct)),
        "accuracy": float(correct.mean()),
        "correct": correct.tolist(),
    }


def mcnemar(left, right):
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    b = int(np.sum((left == 1) & (right == 0)))
    c = int(np.sum((left == 0) & (right == 1)))
    discordant = b + c
    if discordant == 0:
        p = 1.0
    elif discordant <= 500:
        tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
        p = min(1.0, 2.0 * tail / (2 ** discordant))
    else:
        z = max(0.0, abs(b - c) - 1.0) / math.sqrt(discordant)
        p = math.erfc(z / math.sqrt(2.0))
    return {"left_only_correct": b, "right_only_correct": c,
            "discordant": discordant, "p": p}


def bootstrap_mean_ci(values, seed, iterations):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = generator.integers(0, len(values), size=len(values))
        means[index] = values[sample].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return {"estimate": float(values.mean()), "ci95": [float(low), float(high)],
            "iterations": iterations}


def holm(items):
    ordered = sorted(enumerate(items), key=lambda item: item[1]["p"])
    adjusted = [1.0] * len(items)
    running = 0.0
    total = len(items)
    for rank, (original, item) in enumerate(ordered):
        value = min(1.0, item["p"] * (total - rank))
        running = max(running, value)
        adjusted[original] = running
    for item, value in zip(items, adjusted):
        item["p_holm"] = value
        item["significant_holm_0.05"] = value < 0.05


def choose_candidates(level13, path, force=False):
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    aggregate = level13["aggregate"]
    robust = [item["slots"] for item in aggregate["best_robust_pairs"][:2]]
    registration = {"selection_source": "Level 6.13 formal discovery set",
                    "confirmatory_data": "new eval seeds only", "seeds": {}}
    for run in level13["runs"]:
        selected = []
        entries = []

        def add(slots, category):
            key = pair_key(slots)
            if key not in selected:
                selected.append(key)
                best = slots[int(run["single_keep"][slots[1]] > run["single_keep"][slots[0]])]
                entries.append({"slots": sorted(slots), "key": key, "category": category,
                                "discovery_best_single": best})

        add(run["largest_gain_pairs"][0]["slots"], "max_keep_gain")
        add(run["largest_deletion_interactions"][0]["slots"], "max_deletion_interaction")
        for slots in robust:
            add(slots, "cross_seed_robust")
        excluded = set(selected)
        candidates = []
        for key, gain in run["gain_over_best_single"].items():
            if key in excluded:
                continue
            left, right = (int(x) for x in key.split("-"))
            if max(run["single_keep"][left], run["single_keep"][right]) < 0.80:
                continue
            score = abs(gain) + abs(run["deletion_interaction"][key])
            candidates.append((score, [left, right]))
        add(min(candidates)[1], "matched_null_control")
        registration["seeds"][str(run["seed"])] = entries
    save(path, registration)
    return registration


def conditions(candidates):
    output = {"intact": None}
    for candidate in candidates:
        left, right = candidate["slots"]
        output[f"keep_one_{left}"] = [left]
        output[f"keep_one_{right}"] = [right]
        output[f"keep_pair_{left}_{right}"] = [left, right]
        output[f"leave_one_{left}"] = [x for x in range(SLOTS) if x != left]
        output[f"leave_one_{right}"] = [x for x in range(SLOTS) if x != right]
        output[f"leave_pair_{left}_{right}"] = [x for x in range(SLOTS)
                                                  if x not in (left, right)]
    return output


def analyze_seed(seed, candidates, rows, args):
    table = {row["name"]: row for row in rows}
    tests = []
    gain_family = []
    necessity_family = []
    for candidate_index, candidate in enumerate(candidates):
        left, right = candidate["slots"]
        best = candidate["discovery_best_single"]
        pair = np.asarray(table[f"keep_pair_{left}_{right}"]["correct"])
        single = np.asarray(table[f"keep_one_{best}"]["correct"])
        intact = np.asarray(table["intact"]["correct"])
        leave_left = np.asarray(table[f"leave_one_{left}"]["correct"])
        leave_right = np.asarray(table[f"leave_one_{right}"]["correct"])
        leave_pair = np.asarray(table[f"leave_pair_{left}_{right}"]["correct"])
        gain_vector = pair - single
        interaction_vector = leave_left + leave_right - leave_pair - intact
        gain_test = mcnemar(pair, single)
        necessity_test = mcnemar(intact, leave_pair)
        gain_test.update({"candidate": candidate["key"], "family": "keep_gain"})
        necessity_test.update({"candidate": candidate["key"], "family": "pair_necessity"})
        gain_family.append(gain_test)
        necessity_family.append(necessity_test)
        tests.append({
            **candidate,
            "accuracies": {
                "intact": table["intact"]["accuracy"],
                "best_single": table[f"keep_one_{best}"]["accuracy"],
                "keep_pair": table[f"keep_pair_{left}_{right}"]["accuracy"],
                "leave_one_left": table[f"leave_one_{left}"]["accuracy"],
                "leave_one_right": table[f"leave_one_{right}"]["accuracy"],
                "leave_pair": table[f"leave_pair_{left}_{right}"]["accuracy"],
            },
            "keep_gain": bootstrap_mean_ci(
                gain_vector, args.bootstrap_seed + seed * 100 + candidate_index,
                args.bootstrap_iterations,
            ),
            "deletion_interaction": bootstrap_mean_ci(
                interaction_vector, args.bootstrap_seed + seed * 100 + 50 + candidate_index,
                args.bootstrap_iterations,
            ),
            "mcnemar_keep_gain": gain_test,
            "mcnemar_pair_necessity": necessity_test,
        })
    holm(gain_family)
    holm(necessity_family)
    for test, gain_test, necessity_test in zip(tests, gain_family, necessity_family):
        test["mcnemar_keep_gain"] = gain_test
        test["mcnemar_pair_necessity"] = necessity_test
    return {"seed": seed, "samples_per_condition": table["intact"]["samples"],
            "tests": tests}


def plot_seed(result, path):
    labels = [f"{x['key']}\n{x['category']}" for x in result["tests"]]
    gains = np.array([x["keep_gain"]["estimate"] for x in result["tests"]]) * 100
    gain_low = np.array([x["keep_gain"]["ci95"][0] for x in result["tests"]]) * 100
    gain_high = np.array([x["keep_gain"]["ci95"][1] for x in result["tests"]]) * 100
    interactions = np.array([x["deletion_interaction"]["estimate"]
                             for x in result["tests"]]) * 100
    int_low = np.array([x["deletion_interaction"]["ci95"][0]
                        for x in result["tests"]]) * 100
    int_high = np.array([x["deletion_interaction"]["ci95"][1]
                         for x in result["tests"]]) * 100
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].errorbar(x, gains, yerr=[gains - gain_low, gain_high - gains], fmt="o", capsize=4)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Keep-pair gain (pp)")
    axes[1].errorbar(x, interactions, yerr=[interactions - int_low, int_high - interactions],
                     fmt="o", capsize=4, color="tab:red")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Deletion interaction (pp)")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    fig.suptitle(f"IST seed {result['seed']}: preregistered pair confirmation")
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
    progress_path = folder / "predictions.json"
    rows = [] if args.force or not progress_path.exists() else json.loads(
        progress_path.read_text(encoding="utf-8")
    )
    done = {row["name"] for row in rows}
    specs = conditions(candidates)
    for index, (name, keep) in enumerate(specs.items(), start=1):
        if name in done:
            continue
        metric = predict(model, args, keep, device, dtype, eval_seeds)
        rows.append({"name": name, "keep": keep, **metric})
        save(progress_path, rows)
        print(f"seed={seed} [{index}/{len(specs)}] {name} "
              f"accuracy={metric['accuracy']:.2%}", flush=True)
    result = analyze_seed(seed, candidates, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    plot_seed(result, folder / "confirmation_effects.png")
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.13.1 targeted pair confirmation")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-root", default="experiments/level6_13/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=400)
    parser.add_argument("--eval-seed-count", type=int, default=5)
    parser.add_argument("--eval-seed-base", type=int, default=7131000)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=613100)
    parser.add_argument("--output", default="experiments/level6_13_1/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
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

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    level13 = json.loads((Path(args.level6_13_root) / "summary.json").read_text(
        encoding="utf-8"
    ))
    registration = choose_candidates(level13, root / "preregistered_candidates.json",
                                     force=args.force)
    results = []
    for seed in args.seeds:
        candidates = registration["seeds"][str(seed)]
        results.append(run_seed(seed, candidates, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    summary = {"protocol": vars(args), "preregistration": registration, "runs": results}
    save(root / "summary.json", summary)
    print(json.dumps({str(run["seed"]): [
        {"pair": test["key"], "category": test["category"],
         "gain_pp": 100 * test["keep_gain"]["estimate"],
         "gain_ci_pp": [100 * x for x in test["keep_gain"]["ci95"]],
         "gain_p_holm": test["mcnemar_keep_gain"]["p_holm"],
         "deletion_interaction_pp": 100 * test["deletion_interaction"]["estimate"]}
        for test in run["tests"]] for run in results}, indent=2))


if __name__ == "__main__":
    main()
