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


SEEDS = [606, 808, 1001]
SLOTS = 32
TIMES = [1, 2, 4, 8, 12, 15]
MODES = ["zero_once", "zero_persistent", "swap_once", "keep_pair_persistent"]


def pair_key(slots):
    return "-".join(str(x) for x in sorted(slots))


def apply(memory, slots, mode):
    output = list(memory)
    selected = output[2]
    if mode.startswith("keep_pair"):
        mask = torch.zeros(SLOTS, device=selected.device, dtype=selected.dtype)
        mask[slots] = 1
        output[2] = selected * mask[None, :, None]
    elif mode.startswith("zero"):
        output[2] = selected.clone()
        output[2][:, slots, :] = 0
    elif mode.startswith("swap"):
        output[2] = selected.clone()
        output[2][:, slots, :] = torch.roll(selected[:, slots, :], shifts=1, dims=0)
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")
    return output


@torch.no_grad()
def predict(model, args, condition, device, dtype, eval_seeds):
    correct_rows = []
    prediction_rows = []
    target_rows = []
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
                    logits, produced = model(
                        chunks[:, chunk_index], memory=memory, return_memory=True,
                        per_layer_memory=True,
                    )
                    memory = produced
                    if condition["mode"] != "intact":
                        chunk_number = chunk_index + 1
                        start = condition["after_chunk"]
                        persistent = condition["mode"].endswith("persistent")
                        if chunk_number == start or (persistent and chunk_number >= start):
                            memory = apply(memory, condition["slots"], condition["mode"])
            prediction = logits[:, -1, :16].argmax(-1)
            correct_rows.extend((prediction == target).to(torch.int8).cpu().tolist())
            prediction_rows.extend(prediction.cpu().tolist())
            target_rows.extend(target.cpu().tolist())
    return {
        "samples": len(correct_rows),
        "accuracy": float(np.mean(correct_rows)),
        "correct": correct_rows,
        "prediction": prediction_rows,
        "target": target_rows,
    }


def choose_candidates(registration, seed):
    allowed = {"max_keep_gain", "max_deletion_interaction", "cross_seed_robust"}
    output = []
    seen = set()
    for candidate in registration["seeds"][str(seed)]:
        if candidate["category"] not in allowed or candidate["key"] in seen:
            continue
        seen.add(candidate["key"])
        output.append(candidate)
    return output


def make_conditions(candidates, times):
    output = [{"name": "intact", "mode": "intact", "slots": None,
               "pair": None, "category": "baseline", "after_chunk": None}]
    for candidate in candidates:
        for mode in MODES:
            for after_chunk in times:
                output.append({
                    "name": f"{candidate['key']}__{mode}__after{after_chunk}",
                    "mode": mode,
                    "slots": candidate["slots"],
                    "pair": candidate["key"],
                    "category": candidate["category"],
                    "after_chunk": after_chunk,
                })
    return output


def analyze(seed, candidates, rows, args):
    table = {row["name"]: row for row in rows}
    intact = np.asarray(table["intact"]["correct"], dtype=np.int8)
    trajectories = []
    families = {}
    for candidate_index, candidate in enumerate(candidates):
        for mode_index, mode in enumerate(MODES):
            points = []
            family = []
            for time_index, after_chunk in enumerate(args.times):
                name = f"{candidate['key']}__{mode}__after{after_chunk}"
                row = table[name]
                intervention = np.asarray(row["correct"], dtype=np.int8)
                effect_vector = intact - intervention
                paired = mcnemar(intact, intervention)
                paired.update({"after_chunk": after_chunk})
                family.append(paired)
                ci = bootstrap_mean_ci(
                    effect_vector,
                    args.bootstrap_seed + seed * 1000 + candidate_index * 100
                    + mode_index * 10 + time_index,
                    args.bootstrap_iterations,
                )
                points.append({
                    "after_chunk": after_chunk,
                    "accuracy": row["accuracy"],
                    "effect": ci,
                    "mcnemar": paired,
                })
            holm(family)
            for point, paired in zip(points, family):
                point["mcnemar"] = paired
            key = f"{candidate['key']}__{mode}"
            families[key] = family
            trajectories.append({**candidate, "mode": mode, "points": points})
    return {
        "seed": seed,
        "samples_per_condition": table["intact"]["samples"],
        "intact_accuracy": table["intact"]["accuracy"],
        "candidates": candidates,
        "trajectories": trajectories,
    }


def plot_pair(result, candidate, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    mode_titles = {
        "zero_once": "Zero once (recovery)",
        "zero_persistent": "Zero persistently (maintenance)",
        "swap_once": "Batch-swap once (identity)",
        "keep_pair_persistent": "Keep pair persistently (sufficiency)",
    }
    for axis, mode in zip(axes.flat, MODES):
        trajectory = next(x for x in result["trajectories"]
                          if x["key"] == candidate["key"] and x["mode"] == mode)
        times = [x["after_chunk"] for x in trajectory["points"]]
        accuracy = np.array([x["accuracy"] for x in trajectory["points"]]) * 100
        low = np.array([result["intact_accuracy"] - x["effect"]["ci95"][1]
                        for x in trajectory["points"]]) * 100
        high = np.array([result["intact_accuracy"] - x["effect"]["ci95"][0]
                         for x in trajectory["points"]]) * 100
        axis.plot(times, accuracy, marker="o")
        axis.fill_between(times, low, high, alpha=0.2)
        axis.axhline(result["intact_accuracy"] * 100, color="black", linestyle="--",
                     linewidth=1, label="intact")
        axis.set_title(mode_titles[mode])
        axis.set_xticks(times)
        axis.set_ylim(0, 101)
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Query accuracy (%)")
    axes[1, 0].set_ylabel("Query accuracy (%)")
    axes[1, 0].set_xlabel("Intervene after chunk")
    axes[1, 1].set_xlabel("Intervene after chunk")
    axes[0, 0].legend()
    fig.suptitle(f"IST seed {result['seed']} pair {candidate['key']}: dynamic causal trajectory")
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
    specs = make_conditions(candidates, args.times)
    progress_path = folder / "predictions.json"
    rows = [] if args.force or not progress_path.exists() else json.loads(
        progress_path.read_text(encoding="utf-8")
    )
    done = {row["name"] for row in rows}
    for index, condition in enumerate(specs, start=1):
        if condition["name"] in done:
            continue
        metric = predict(model, args, condition, device, dtype, eval_seeds)
        rows.append({**condition, **metric})
        save(progress_path, rows)
        print(f"seed={seed} [{index}/{len(specs)}] {condition['name']} "
              f"accuracy={metric['accuracy']:.2%}", flush=True)
    result = analyze(seed, candidates, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    for candidate in candidates:
        plot_pair(result, candidate, folder / f"pair_{candidate['key']}_trajectory.png")
    return result


def aggregate_common(results):
    common = set(x["key"] for x in results[0]["candidates"])
    for result in results[1:]:
        common &= set(x["key"] for x in result["candidates"])
    output = {}
    for key in sorted(common):
        output[key] = {}
        for mode in MODES:
            curves = []
            for result in results:
                trajectory = next(x for x in result["trajectories"]
                                  if x["key"] == key and x["mode"] == mode)
                curves.append([point["accuracy"] for point in trajectory["points"]])
            output[key][mode] = {
                "mean_accuracy": np.mean(curves, axis=0).tolist(),
                "minimum_accuracy": np.min(curves, axis=0).tolist(),
            }
    return output


def main():
    parser = argparse.ArgumentParser(description="Level 6.14 dynamic causal trajectories")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--times", nargs="+", type=int, default=TIMES)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=400)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--eval-seed-base", type=int, default=7140000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=614000)
    parser.add_argument("--output", default="experiments/level6_14/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
    if any(time < 1 or time >= args.chunks for time in args.times):
        raise ValueError("Every intervention time must be between 1 and chunks-1")
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
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    results = []
    for seed in args.seeds:
        candidates = choose_candidates(registration, seed)
        results.append(run_seed(seed, candidates, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    summary = {"protocol": vars(args), "runs": results,
               "common_pair_aggregate": aggregate_common(results)}
    save(root / "summary.json", summary)
    print(json.dumps({str(result["seed"]): {
        "intact": result["intact_accuracy"],
        "pairs": [candidate["key"] for candidate in result["candidates"]],
    } for result in results}, indent=2))


if __name__ == "__main__":
    main()
