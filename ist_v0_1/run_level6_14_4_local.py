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
from run_level6_14_1_local import swap_slots


SEEDS = [606, 808, 1001]
SLOTS = 32
KEEP_COUNTS = [1, 2, 4, 8, 16, 30]
STRATEGIES = ["donor_projection", "source_routing", "single_causal"]


def restore(memory, clean_memory, slots):
    output = [item.clone() for item in memory]
    output[2][:, slots, :] = clean_memory[2][:, slots, :]
    return output


def condition_id(slots):
    return "slots_" + "_".join(str(slot) for slot in sorted(slots))


def build_plan(level14_3, path, args):
    if path.exists() and not args.force:
        return json.loads(path.read_text(encoding="utf-8"))
    plan = {"source": str(Path(args.level6_14_3_root) / "summary.json"),
            "keep_counts": args.keep_counts, "random_repeats": args.random_repeats,
            "seeds": {}}
    for run in level14_3["runs"]:
        graphs = []
        for graph_index, graph in enumerate(run["graphs"]):
            edges = graph["edges"]
            destinations = [edge["destination"] for edge in edges]
            rankings = {
                "donor_projection": [edge["destination"] for edge in sorted(
                    edges, key=lambda edge: edge["donor_projection"], reverse=True)],
                "source_routing": [edge["destination"] for edge in sorted(
                    edges, key=lambda edge: edge["source_routing_sum"], reverse=True)],
                "single_causal": [edge["destination"] for edge in sorted(
                    edges, key=lambda edge: edge["accuracy_gain"]["estimate"], reverse=True)],
            }
            entries = []
            for strategy in STRATEGIES:
                for count in args.keep_counts:
                    slots = sorted(rankings[strategy][:count])
                    entries.append({"strategy": strategy, "count": count, "repeat": None,
                                    "slots": slots, "condition_id": condition_id(slots)})
            random_rankings = []
            for repeat in range(args.random_repeats):
                generator = np.random.default_rng(
                    args.mask_seed_base + run["seed"] * 1000 + graph_index * 20 + repeat
                )
                ranking = generator.permutation(destinations).tolist()
                random_rankings.append(ranking)
                for count in args.keep_counts:
                    slots = sorted(ranking[:count])
                    entries.append({"strategy": "random", "count": count, "repeat": repeat,
                                    "slots": slots, "condition_id": condition_id(slots)})
            graphs.append({
                "pair": graph["key"], "source_slots": graph["slots"],
                "category": graph["category"], "swap_after": graph["swap_after"],
                "rankings": rankings, "random_rankings": random_rankings,
                "entries": entries,
                "single_discovery_gain": {
                    str(edge["destination"]): edge["accuracy_gain"]["estimate"] for edge in edges
                },
            })
        plan["seeds"][str(run["seed"])] = graphs
    save(path, plan)
    return plan


def make_conditions(graphs):
    conditions = {"intact": {"name": "intact", "mode": "intact", "pair": None,
                               "source_slots": None, "swap_after": None,
                               "restore_slots": None}}
    for graph in graphs:
        prefix = f"{graph['pair']}__swap{graph['swap_after']}"
        base = {"mode": "pollution", "pair": graph["pair"],
                "source_slots": graph["source_slots"], "swap_after": graph["swap_after"]}
        controls = {
            "hold": None,
            "restore_source": graph["source_slots"],
            "restore_all_targets": [slot for slot in range(SLOTS)
                                    if slot not in graph["source_slots"]],
            "restore_all_final": list(range(SLOTS)),
        }
        for label, slots in controls.items():
            name = f"{prefix}__{label}"
            conditions[name] = {"name": name, **base, "restore_slots": slots}
        for entry in graph["entries"]:
            name = f"{prefix}__{entry['condition_id']}"
            conditions[name] = {"name": name, **base,
                                "restore_slots": sorted(graph["source_slots"] + entry["slots"])}
    return list(conditions.values())


@torch.no_grad()
def evaluate(model, args, condition, device, dtype, eval_seeds):
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
            clean_memory = None
            polluted_memory = None
            for chunk_index in range(args.chunks):
                chunk_number = chunk_index + 1
                needs_clean = condition["mode"] == "intact" or condition["restore_slots"] is not None
                with torch.autocast(device_type="cuda", dtype=dtype):
                    clean_logits = None
                    clean_produced = None
                    if needs_clean:
                        clean_logits, clean_produced = model(
                            chunks[:, chunk_index], memory=clean_memory, return_memory=True,
                            per_layer_memory=True,
                        )
                        clean_memory = clean_produced
                    if condition["mode"] == "intact":
                        polluted_logits, polluted_produced = clean_logits, clean_produced
                    else:
                        polluted_logits, polluted_produced = model(
                            chunks[:, chunk_index], memory=polluted_memory, return_memory=True,
                            per_layer_memory=True,
                        )
                    polluted_memory = polluted_produced
                    if condition["mode"] == "pollution":
                        if chunk_number == condition["swap_after"]:
                            polluted_memory = swap_slots(polluted_memory, condition["source_slots"])
                        if (condition["restore_slots"] is not None
                                and chunk_number == condition["swap_after"] + 1):
                            polluted_memory = restore(
                                polluted_memory, clean_memory, condition["restore_slots"]
                            )
            logits = clean_logits if condition["mode"] == "intact" else polluted_logits
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
        "samples": int(len(target)), "accuracy": float(correct.mean()),
        "donor_attraction": float(donor_hit.sum() / mismatch.sum()),
        "correct": correct.tolist(), "donor_hit": donor_hit.tolist(),
        "donor_mismatch": mismatch.astype(np.int8).tolist(),
    }


def masked_donor(row):
    mask = np.asarray(row["donor_mismatch"], dtype=bool)
    return np.asarray(row["donor_hit"], dtype=np.int8)[mask]


def analyze(seed, graphs, rows, args):
    table = {row["name"]: row for row in rows}
    intact = table["intact"]
    results = []
    for graph_index, graph in enumerate(graphs):
        prefix = f"{graph['pair']}__swap{graph['swap_after']}"
        hold = table[f"{prefix}__hold"]
        source_row = table[f"{prefix}__restore_source"]
        hold_correct = np.asarray(hold["correct"], dtype=np.int8)
        hold_donor = masked_donor(hold)
        source_correct = np.asarray(source_row["correct"], dtype=np.int8)
        source_donor = masked_donor(source_row)
        entries = []
        accuracy_family = []
        donor_family = []
        for entry_index, entry in enumerate(graph["entries"]):
            row = table[f"{prefix}__{entry['condition_id']}"]
            repaired = np.asarray(row["correct"], dtype=np.int8)
            repaired_donor = masked_donor(row)
            accuracy_test = mcnemar(repaired, source_correct)
            donor_test = mcnemar(repaired_donor, source_donor)
            accuracy_family.append(accuracy_test)
            donor_family.append(donor_test)
            gain = repaired - source_correct
            donor_reduction = source_donor - repaired_donor
            recoverable_gap = intact["accuracy"] - source_row["accuracy"]
            entries.append({
                **entry, "accuracy": row["accuracy"],
                "donor_attraction": row["donor_attraction"],
                "accuracy_gain": bootstrap_mean_ci(
                    gain, args.bootstrap_seed + seed * 10000 + graph_index * 100
                    + entry_index, args.bootstrap_iterations),
                "donor_reduction": bootstrap_mean_ci(
                    donor_reduction, args.bootstrap_seed + seed * 10000 + 5000
                    + graph_index * 100 + entry_index, args.bootstrap_iterations),
                "recovery_fraction": ((row["accuracy"] - source_row["accuracy"]) / recoverable_gap
                                      if recoverable_gap > 1e-12 else None),
                "discovery_additive_prediction": sum(
                    graph["single_discovery_gain"][str(slot)] for slot in entry["slots"]
                ),
                "mcnemar_accuracy": accuracy_test,
                "mcnemar_donor": donor_test,
            })
        holm(accuracy_family)
        holm(donor_family)
        for entry, accuracy_test, donor_test in zip(entries, accuracy_family, donor_family):
            entry["mcnemar_accuracy"] = accuracy_test
            entry["mcnemar_donor"] = donor_test
        controls = {}
        for label in ["restore_source", "restore_all_targets", "restore_all_final"]:
            row = table[f"{prefix}__{label}"]
            controls[label] = {"accuracy": row["accuracy"],
                               "donor_attraction": row["donor_attraction"]}
        curves = {}
        for strategy in STRATEGIES:
            points = [entry for entry in entries if entry["strategy"] == strategy]
            curves[strategy] = sorted(points, key=lambda point: point["count"])
        random_curve = []
        for count in args.keep_counts:
            points = [entry for entry in entries
                      if entry["strategy"] == "random" and entry["count"] == count]
            random_curve.append({
                "count": count,
                "accuracy_mean": float(np.mean([point["accuracy"] for point in points])),
                "accuracy_min": min(point["accuracy"] for point in points),
                "accuracy_max": max(point["accuracy"] for point in points),
                "donor_attraction_mean": float(np.mean(
                    [point["donor_attraction"] for point in points])),
                "recovery_fraction_mean": float(np.mean(
                    [point["recovery_fraction"] for point in points])),
                "points": points,
            })
        thresholds = {}
        for strategy, points in {**curves, "random_mean": random_curve}.items():
            if strategy == "random_mean":
                qualifying = [point["count"] for point in points
                              if point["recovery_fraction_mean"] >= 0.90]
            else:
                qualifying = [point["count"] for point in points
                              if point["recovery_fraction"] >= 0.90]
            thresholds[strategy] = min(qualifying) if qualifying else None
        results.append({
            "pair": graph["pair"], "source_slots": graph["source_slots"],
            "category": graph["category"], "swap_after": graph["swap_after"],
            "intact_accuracy": intact["accuracy"], "hold_accuracy": hold["accuracy"],
            "hold_donor_attraction": hold["donor_attraction"],
            "source_accuracy": source_row["accuracy"],
            "source_donor_attraction": source_row["donor_attraction"], "controls": controls,
            "curves": curves, "random_curve": random_curve,
            "threshold_90_recovery": thresholds,
        })
    return {"seed": seed, "samples_per_condition": intact["samples"],
            "intact_accuracy": intact["accuracy"], "dose_curves": results}


def plot_pair(result, pair, path, keep_counts):
    curves = [curve for curve in result["dose_curves"] if curve["pair"] == pair]
    fig, axes = plt.subplots(len(curves), 2, figsize=(12, 4 * len(curves)), squeeze=False)
    colors = {"donor_projection": "tab:blue", "source_routing": "tab:orange",
              "single_causal": "tab:green"}
    for row_index, curve in enumerate(sorted(curves, key=lambda item: item["swap_after"])):
        for strategy in STRATEGIES:
            points = curve["curves"][strategy]
            axes[row_index, 0].plot(keep_counts, [p["accuracy"] * 100 for p in points],
                                    marker="o", color=colors[strategy], label=strategy)
            axes[row_index, 1].plot(keep_counts,
                                    [p["donor_attraction"] * 100 for p in points],
                                    marker="o", color=colors[strategy], label=strategy)
        random = curve["random_curve"]
        mean_accuracy = np.array([p["accuracy_mean"] for p in random]) * 100
        axes[row_index, 0].plot(keep_counts, mean_accuracy, marker="o", color="gray",
                                label="random mean")
        axes[row_index, 0].fill_between(
            keep_counts, np.array([p["accuracy_min"] for p in random]) * 100,
            np.array([p["accuracy_max"] for p in random]) * 100, color="gray", alpha=0.2)
        axes[row_index, 1].plot(
            keep_counts, np.array([p["donor_attraction_mean"] for p in random]) * 100,
            marker="o", color="gray", label="random mean")
        axes[row_index, 0].axhline(curve["intact_accuracy"] * 100, color="black",
                                   linestyle=":", label="intact")
        axes[row_index, 0].axhline(curve["hold_accuracy"] * 100, color="tab:red",
                                   linestyle="--", label="polluted")
        axes[row_index, 0].axhline(curve["source_accuracy"] * 100, color="tab:purple",
                                   linestyle="-.", label="source restored")
        axes[row_index, 1].axhline(curve["hold_donor_attraction"] * 100,
                                   color="tab:red", linestyle="--", label="polluted")
        axes[row_index, 1].axhline(curve["source_donor_attraction"] * 100,
                                   color="tab:purple", linestyle="-.", label="source restored")
        axes[row_index, 0].set_ylabel(f"Swap {curve['swap_after']}\nAccuracy (%)")
        axes[row_index, 1].set_ylabel("Donor-target prediction (%)")
        for axis in axes[row_index]:
            axis.set_xscale("symlog", linthresh=1)
            axis.set_xticks(keep_counts, keep_counts)
            axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Restored destination slots K")
    axes[-1, 1].set_xlabel("Restored destination slots K")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    fig.suptitle(f"IST seed {result['seed']} pair {pair}: group restoration dose curve")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_seed(seed, graphs, args, device, dtype, root):
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
    specs = make_conditions(graphs)
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
    result = analyze(seed, graphs, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    for pair in sorted(set(graph["pair"] for graph in graphs)):
        plot_pair(result, pair, folder / f"pair_{pair}_dose_curve.png", args.keep_counts)
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.14.4 group restoration dose curves")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-14-3-root", default="experiments/level6_14_3/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--keep-counts", nargs="+", type=int, default=KEEP_COUNTS)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--mask-seed-base", type=int, default=6144000)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=400)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--eval-seed-base", type=int, default=7144000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=614400)
    parser.add_argument("--output", default="experiments/level6_14_4/formal")
    parser.add_argument("--graph-limit", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
    if sorted(args.keep_counts) != args.keep_counts or any(
            count < 1 or count > 30 for count in args.keep_counts):
        raise ValueError("keep-counts must be sorted and between 1 and 30")
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
    level14_3 = json.loads((Path(args.level6_14_3_root) / "summary.json").read_text(
        encoding="utf-8"))
    plan = build_plan(level14_3, root / "selection_plan.json", args)
    results = []
    for seed in args.seeds:
        graphs = plan["seeds"][str(seed)]
        if args.graph_limit is not None:
            graphs = graphs[:args.graph_limit]
        results.append(run_seed(seed, graphs, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    save(root / "summary.json", {"protocol": vars(args), "selection_plan": plan,
                                  "runs": results})
    print(json.dumps({str(result["seed"]): {
        "intact": result["intact_accuracy"], "curves": len(result["dose_curves"])
    } for result in results}, indent=2))


if __name__ == "__main__":
    main()
