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
SWAP_TIMES = [1, 4, 8, 12]
DELAYS = [0, 1, 2, 4]


def swap_slots(memory, slots):
    output = list(memory)
    output[2] = memory[2].clone()
    output[2][:, slots, :] = torch.roll(memory[2][:, slots, :], shifts=1, dims=0)
    return output


def zero_slots(memory, slots):
    output = list(memory)
    output[2] = memory[2].clone()
    output[2][:, slots, :] = 0
    return output


def restore_slots(memory, clean_memory, slots):
    output = list(memory)
    output[2] = memory[2].clone()
    output[2][:, slots, :] = clean_memory[2][:, slots, :]
    return output


@torch.no_grad()
def predict(model, args, condition, device, dtype, eval_seeds):
    predictions = []
    targets = []
    donor_targets = []
    batches = args.samples_per_eval_seed // args.eval_batch_size
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device
            )
            memory = None
            clean_memory = None
            needs_clean = condition["remedy"] == "restore_clean"
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(args.chunks):
                    logits, produced = model(
                        chunks[:, chunk_index], memory=memory, return_memory=True,
                        per_layer_memory=True,
                    )
                    memory = produced
                    clean_produced = None
                    if needs_clean:
                        _, clean_produced = model(
                            chunks[:, chunk_index], memory=clean_memory, return_memory=True,
                            per_layer_memory=True,
                        )
                        clean_memory = clean_produced
                    if condition["mode"] == "pollution":
                        chunk_number = chunk_index + 1
                        if chunk_number == condition["swap_after"]:
                            memory = swap_slots(memory, condition["slots"])
                        remediation_chunk = condition["swap_after"] + condition["delay"]
                        if chunk_number == remediation_chunk:
                            if condition["remedy"] == "zero":
                                memory = zero_slots(memory, condition["slots"])
                            elif condition["remedy"] == "restore_clean":
                                memory = restore_slots(memory, clean_produced, condition["slots"])
            prediction = logits[:, -1, :16].argmax(-1)
            predictions.extend(prediction.cpu().tolist())
            targets.extend(target.cpu().tolist())
            donor_targets.extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
    prediction = np.asarray(predictions)
    target = np.asarray(targets)
    donor = np.asarray(donor_targets)
    correct = (prediction == target).astype(np.int8)
    mismatch = donor != target
    donor_hit = ((prediction == donor) & mismatch).astype(np.int8)
    return {
        "samples": int(len(target)),
        "accuracy": float(correct.mean()),
        "donor_mismatch_samples": int(mismatch.sum()),
        "donor_attraction": float(donor_hit.sum() / mismatch.sum()),
        "correct": correct.tolist(),
        "donor_hit": donor_hit.tolist(),
        "donor_mismatch": mismatch.astype(np.int8).tolist(),
        "prediction": predictions,
        "target": targets,
        "donor_target": donor_targets,
    }


def choose_candidates(registration, seed):
    wanted = ["max_keep_gain", "max_deletion_interaction", "cross_seed_robust"]
    output = []
    seen = set()
    for category in wanted:
        for candidate in registration["seeds"][str(seed)]:
            if candidate["category"] == category and candidate["key"] not in seen:
                output.append(candidate)
                seen.add(candidate["key"])
                break
    return output


def make_conditions(candidates, swap_times, delays):
    output = [{"name": "intact", "mode": "intact", "pair": None, "slots": None,
               "category": "baseline", "swap_after": None, "remedy": "none", "delay": None}]
    for candidate in candidates:
        for swap_after in swap_times:
            output.append({
                "name": f"{candidate['key']}__swap{swap_after}__hold",
                "mode": "pollution", "pair": candidate["key"],
                "slots": candidate["slots"], "category": candidate["category"],
                "swap_after": swap_after, "remedy": "hold", "delay": -1,
            })
            for remedy in ["zero", "restore_clean"]:
                for delay in delays:
                    output.append({
                        "name": f"{candidate['key']}__swap{swap_after}__{remedy}__delay{delay}",
                        "mode": "pollution", "pair": candidate["key"],
                        "slots": candidate["slots"], "category": candidate["category"],
                        "swap_after": swap_after, "remedy": remedy, "delay": delay,
                    })
    return output


def paired_donor_test(left_row, right_row):
    mask = np.asarray(left_row["donor_mismatch"], dtype=bool)
    left = np.asarray(left_row["donor_hit"], dtype=np.int8)[mask]
    right = np.asarray(right_row["donor_hit"], dtype=np.int8)[mask]
    return mcnemar(left, right), left, right


def analyze(seed, candidates, rows, args):
    table = {row["name"]: row for row in rows}
    intact = table["intact"]
    groups = []
    for candidate_index, candidate in enumerate(candidates):
        for time_index, swap_after in enumerate(args.swap_times):
            hold = table[f"{candidate['key']}__swap{swap_after}__hold"]
            intact_correct = np.asarray(intact["correct"], dtype=np.int8)
            hold_correct = np.asarray(hold["correct"], dtype=np.int8)
            hold_loss = bootstrap_mean_ci(
                intact_correct - hold_correct,
                args.bootstrap_seed + seed * 1000 + candidate_index * 100 + time_index,
                args.bootstrap_iterations,
            )
            hold_accuracy_test = mcnemar(intact_correct, hold_correct)
            hold_donor_test, intact_donor, hold_donor = paired_donor_test(intact, hold)
            donor_excess = bootstrap_mean_ci(
                hold_donor - intact_donor,
                args.bootstrap_seed + seed * 1000 + candidate_index * 100 + 20 + time_index,
                args.bootstrap_iterations,
            )
            remedies = []
            accuracy_family = []
            donor_family = []
            for remedy_index, remedy in enumerate(["zero", "restore_clean"]):
                for delay_index, delay in enumerate(args.delays):
                    row = table[f"{candidate['key']}__swap{swap_after}__{remedy}__delay{delay}"]
                    repaired = np.asarray(row["correct"], dtype=np.int8)
                    accuracy_gain = bootstrap_mean_ci(
                        repaired - hold_correct,
                        args.bootstrap_seed + seed * 1000 + candidate_index * 100
                        + 40 + remedy_index * 10 + delay_index,
                        args.bootstrap_iterations,
                    )
                    accuracy_test = mcnemar(repaired, hold_correct)
                    accuracy_family.append(accuracy_test)
                    donor_test, repaired_donor, hold_donor_masked = paired_donor_test(row, hold)
                    donor_reduction = bootstrap_mean_ci(
                        hold_donor_masked - repaired_donor,
                        args.bootstrap_seed + seed * 1000 + candidate_index * 100
                        + 70 + remedy_index * 10 + delay_index,
                        args.bootstrap_iterations,
                    )
                    donor_family.append(donor_test)
                    remedies.append({
                        "remedy": remedy,
                        "delay": delay,
                        "remediation_chunk": swap_after + delay,
                        "before_final_query": swap_after + delay < args.chunks,
                        "accuracy": row["accuracy"],
                        "donor_attraction": row["donor_attraction"],
                        "accuracy_gain_vs_hold": accuracy_gain,
                        "donor_reduction_vs_hold": donor_reduction,
                        "mcnemar_accuracy_vs_hold": accuracy_test,
                        "mcnemar_donor_vs_hold": donor_test,
                    })
            holm(accuracy_family)
            holm(donor_family)
            for item, accuracy_test, donor_test in zip(remedies, accuracy_family, donor_family):
                item["mcnemar_accuracy_vs_hold"] = accuracy_test
                item["mcnemar_donor_vs_hold"] = donor_test
            groups.append({
                **candidate,
                "swap_after": swap_after,
                "hold_accuracy": hold["accuracy"],
                "hold_donor_attraction": hold["donor_attraction"],
                "hold_accuracy_loss": hold_loss,
                "hold_donor_excess": donor_excess,
                "mcnemar_hold_accuracy_vs_intact": hold_accuracy_test,
                "mcnemar_hold_donor_vs_intact": hold_donor_test,
                "remedies": remedies,
            })
    return {
        "seed": seed,
        "samples_per_condition": intact["samples"],
        "intact_accuracy": intact["accuracy"],
        "intact_donor_attraction": intact["donor_attraction"],
        "candidates": candidates,
        "groups": groups,
    }


def plot_pair(result, candidate, path, swap_times, delays):
    fig, axes = plt.subplots(len(swap_times), 2, figsize=(12, 3 * len(swap_times) + 1),
                             sharex="col", squeeze=False)
    for row_index, swap_after in enumerate(swap_times):
        group = next(x for x in result["groups"]
                     if x["key"] == candidate["key"] and x["swap_after"] == swap_after)
        for remedy, color in [("zero", "tab:blue"), ("restore_clean", "tab:green")]:
            points = [x for x in group["remedies"] if x["remedy"] == remedy]
            axes[row_index, 0].plot(delays, [x["accuracy"] * 100 for x in points],
                                    marker="o", color=color, label=remedy)
            axes[row_index, 1].plot(delays, [x["donor_attraction"] * 100 for x in points],
                                    marker="o", color=color, label=remedy)
        axes[row_index, 0].axhline(group["hold_accuracy"] * 100, color="tab:red",
                                   linestyle="--", label="pollution held")
        axes[row_index, 0].axhline(result["intact_accuracy"] * 100, color="black",
                                   linestyle=":", label="intact")
        axes[row_index, 1].axhline(group["hold_donor_attraction"] * 100, color="tab:red",
                                   linestyle="--", label="pollution held")
        axes[row_index, 1].axhline(result["intact_donor_attraction"] * 100, color="black",
                                   linestyle=":", label="intact")
        axes[row_index, 0].set_ylabel(f"Swap after {swap_after}\nAccuracy (%)")
        axes[row_index, 1].set_ylabel("Donor-target prediction (%)")
        axes[row_index, 0].grid(alpha=0.2)
        axes[row_index, 1].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Remediation delay (chunks)")
    axes[-1, 1].set_xlabel("Remediation delay (chunks)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"IST seed {result['seed']} pair {candidate['key']}: pollution recovery")
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
    specs = make_conditions(candidates, args.swap_times, args.delays)
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
              f"acc={metric['accuracy']:.2%} donor={metric['donor_attraction']:.2%}", flush=True)
    result = analyze(seed, candidates, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    for candidate in candidates:
        plot_pair(result, candidate, folder / f"pair_{candidate['key']}_pollution_recovery.png",
                  args.swap_times, args.delays)
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.14.1 pollution recovery dynamics")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--delays", nargs="+", type=int, default=DELAYS)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=400)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--eval-seed-base", type=int, default=7141000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=614100)
    parser.add_argument("--output", default="experiments/level6_14_1/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
    if any(time < 1 or time >= args.chunks for time in args.swap_times):
        raise ValueError("swap times must be between 1 and chunks-1")
    if args.eval_batch_size < 2:
        raise ValueError("eval-batch-size must be at least 2 for batch swap")
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
    save(root / "summary.json", {"protocol": vars(args), "runs": results})
    print(json.dumps({str(result["seed"]): {
        "intact_accuracy": result["intact_accuracy"],
        "intact_donor_attraction": result["intact_donor_attraction"],
        "pairs": [candidate["key"] for candidate in result["candidates"]],
    } for result in results}, indent=2))


if __name__ == "__main__":
    main()
