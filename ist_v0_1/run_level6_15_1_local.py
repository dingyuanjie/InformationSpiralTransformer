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
SWAP_TIMES = [4, 8]
DOSES = [0.10, 0.20, 0.25, 0.30, 0.40]


def scale_tag(value):
    return str(value).replace(".", "p")


def intervention_specs(args):
    specs = [{"defense": "baseline", "scale": 1.0, "offset": None,
              "duration": 0, "pathway": "none", "family": "baseline"}]
    for scale in args.doses:
        specs.append({"defense": f"dose_{scale_tag(scale)}", "scale": scale,
                      "offset": 1, "duration": 1, "pathway": "both",
                      "family": "dose"})
    for offset in [2, 3]:
        specs.append({"defense": f"window_offset{offset}_0p25", "scale": 0.25,
                      "offset": offset, "duration": 1, "pathway": "both",
                      "family": "window"})
    specs.append({"defense": "duration2_0p25", "scale": 0.25, "offset": 1,
                  "duration": 2, "pathway": "both", "family": "duration"})
    for pathway in ["historical", "memory"]:
        specs.append({"defense": f"path_{pathway}_0p25", "scale": 0.25,
                      "offset": 1, "duration": 1, "pathway": pathway,
                      "family": "pathway"})
    return specs


def active_scale(spec, chunk_number, swap_after):
    if spec["offset"] is None:
        return 1.0
    first = swap_after + spec["offset"]
    return spec["scale"] if first <= chunk_number < first + spec["duration"] else 1.0


def set_intervention(model, spec, chunk_number, swap_after):
    scale = active_scale(spec, chunk_number, swap_after)
    block = model.blocks[2]
    memory = block.memory
    block.historical_consistency_threshold = None
    memory.propagation_consistency_threshold = None
    block.historical_read_scale = scale if spec["pathway"] in ("both", "historical") else 1.0
    memory.propagation_scale = scale if spec["pathway"] in ("both", "memory") else 1.0


def reset_intervention(model):
    block = model.blocks[2]
    block.historical_read_scale = 1.0
    block.historical_consistency_threshold = None
    block.memory.propagation_scale = 1.0
    block.memory.propagation_consistency_threshold = None


def make_conditions(candidates, args):
    specs = intervention_specs(args)
    conditions = [{"name": "clean__baseline", "mode": "clean", "swap_after": None,
                   "slots": None, "pair": None, **specs[0]}]
    for swap_after in args.swap_times:
        for spec in specs[1:]:
            conditions.append({
                "name": f"clean__swap{swap_after}__{spec['defense']}",
                "mode": "clean", "swap_after": swap_after, "slots": None,
                "pair": None, **spec,
            })
    for candidate in candidates:
        for swap_after in args.swap_times:
            for spec in specs:
                conditions.append({
                    "name": f"{candidate['key']}__swap{swap_after}__{spec['defense']}",
                    "mode": "pollution", "swap_after": swap_after,
                    "slots": candidate["slots"], "pair": candidate["key"],
                    "category": candidate["category"], **spec,
                })
    return conditions


@torch.no_grad()
def evaluate(model, args, condition, device, dtype, eval_seeds):
    predictions, targets, donor_targets = [], [], []
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
                    virtual_swap = condition["swap_after"]
                    set_intervention(model, condition, chunk_number,
                                     virtual_swap if virtual_swap is not None else -100)
                    logits, memory = model(
                        chunks[:, chunk_index], memory=memory, return_memory=True,
                        per_layer_memory=True,
                    )
                    if (condition["mode"] == "pollution"
                            and chunk_number == condition["swap_after"]):
                        memory = swap_slots(memory, condition["slots"])
            prediction = logits[:, -1, :16].argmax(-1)
            predictions.extend(prediction.cpu().tolist())
            targets.extend(target.cpu().tolist())
            donor_targets.extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
    reset_intervention(model)
    prediction = np.asarray(predictions)
    target = np.asarray(targets)
    donor = np.asarray(donor_targets)
    mismatch = donor != target
    correct = (prediction == target).astype(np.int8)
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


def analyze(seed, candidates, rows, args):
    table = {row["name"]: row for row in rows}
    baseline_clean = table["clean__baseline"]
    experiments = []
    for candidate in candidates:
        for swap_after in args.swap_times:
            prefix = f"{candidate['key']}__swap{swap_after}"
            baseline = table[f"{prefix}__baseline"]
            base_correct = np.asarray(baseline["correct"], dtype=np.int8)
            base_donor = masked_donor(baseline)
            entries, accuracy_tests, donor_tests = [], [], []
            for index, spec in enumerate(intervention_specs(args)[1:]):
                row = table[f"{prefix}__{spec['defense']}"]
                clean = table[f"clean__swap{swap_after}__{spec['defense']}"]
                correct = np.asarray(row["correct"], dtype=np.int8)
                donor = masked_donor(row)
                accuracy_test = mcnemar(correct, base_correct)
                donor_test = mcnemar(donor, base_donor)
                accuracy_tests.append(accuracy_test)
                donor_tests.append(donor_test)
                entries.append({
                    **spec, "accuracy": row["accuracy"],
                    "donor_attraction": row["donor_attraction"],
                    "clean_accuracy": clean["accuracy"],
                    "clean_delta": clean["accuracy"] - baseline_clean["accuracy"],
                    "accuracy_gain": bootstrap_mean_ci(
                        correct - base_correct,
                        args.bootstrap_seed + seed * 10000 + swap_after * 100 + index,
                        args.bootstrap_iterations),
                    "donor_reduction": bootstrap_mean_ci(
                        base_donor - donor,
                        args.bootstrap_seed + seed * 10000 + 5000 + swap_after * 100 + index,
                        args.bootstrap_iterations),
                    "mcnemar_accuracy": accuracy_test,
                    "mcnemar_donor": donor_test,
                })
            holm(accuracy_tests)
            holm(donor_tests)
            for entry, atest, dtest in zip(entries, accuracy_tests, donor_tests):
                entry["mcnemar_accuracy"] = atest
                entry["mcnemar_donor"] = dtest
            experiments.append({
                **candidate, "swap_after": swap_after,
                "baseline_clean_accuracy": baseline_clean["accuracy"],
                "baseline_polluted_accuracy": baseline["accuracy"],
                "baseline_donor_attraction": baseline["donor_attraction"],
                "entries": entries,
            })
    return {"seed": seed, "eval_seeds": None,
            "baseline_clean_accuracy": baseline_clean["accuracy"],
            "experiments": experiments}


def plot_seed(result, path, args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for experiment in result["experiments"]:
        label = f"{experiment['key']}@{experiment['swap_after']}"
        doses = [x for x in experiment["entries"] if x["family"] == "dose"]
        axes[0].plot([x["scale"] for x in doses],
                     [x["accuracy_gain"]["estimate"] * 100 for x in doses],
                     marker="o", alpha=0.7, label=label)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="First-successor dose", xlabel="Propagation scale",
                ylabel="Accuracy gain (pp)")
    families = ["dose_0p25", "window_offset2_0p25", "window_offset3_0p25",
                "duration2_0p25"]
    labels = ["step +1", "step +2", "step +3", "steps +1,+2"]
    means = []
    for defense in families:
        values = [e["accuracy_gain"]["estimate"] * 100
                  for x in result["experiments"] for e in x["entries"]
                  if e["defense"] == defense]
        means.append(float(np.mean(values)))
    axes[1].bar(labels, means)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Window localization", ylabel="Mean accuracy gain (pp)")
    paths = ["dose_0p25", "path_historical_0p25", "path_memory_0p25"]
    labels = ["both", "SpiralAttention", "MemoryAttention"]
    means = []
    for defense in paths:
        values = [e["accuracy_gain"]["estimate"] * 100
                  for x in result["experiments"] for e in x["entries"]
                  if e["defense"] == defense]
        means.append(float(np.mean(values)))
    axes[2].bar(labels, means)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].set(title="Pathway decomposition", ylabel="Mean accuracy gain (pp)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Level 6.15.1 — IST seed {result['seed']}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_seed(seed, candidates, args, device, dtype, root):
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    model.load_state_dict(state["model"])
    eval_seeds = [args.eval_seed_base + seed * 100 + i
                  for i in range(args.eval_seed_count)]
    conditions = make_conditions(candidates, args)
    prediction_path = folder / "predictions.json"
    rows = [] if args.force or not prediction_path.exists() else json.loads(
        prediction_path.read_text(encoding="utf-8")
    )
    done = {row["name"] for row in rows}
    for index, condition in enumerate(conditions, 1):
        if condition["name"] in done:
            continue
        metric = evaluate(model, args, condition, device, dtype, eval_seeds)
        rows.append({**condition, **metric})
        save(prediction_path, rows)
        print(f"seed={seed} [{index}/{len(conditions)}] {condition['name']} "
              f"acc={metric['accuracy']:.2%} donor={metric['donor_attraction']:.2%}",
              flush=True)
    result = analyze(seed, candidates, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    plot_seed(result, folder / "fine_intervention.png", args)
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.15.1 fine causal intervention")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--doses", nargs="+", type=float, default=DOSES)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=400)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--eval-seed-base", type=int, default=7350000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=615100)
    parser.add_argument("--output", default="experiments/level6_15_1/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
    if 0.25 not in args.doses:
        raise ValueError("doses must include preregistered scale 0.25")
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
    preregistration = {
        "primary_confirmatory_condition": {
            "offset": 1, "duration": 1, "scale": 0.25, "pathway": "both",
            "expected_direction": "accuracy_gain > 0 across all three model seeds",
        },
        "independent_eval_seed_base": args.eval_seed_base,
        "families": ["dose", "window", "duration", "pathway"],
        "protocol": vars(args),
    }
    save(root / "preregistration.json", preregistration)
    results = []
    for seed in args.seeds:
        candidates = choose_candidates(registration, seed)
        results.append(run_seed(seed, candidates, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    save(root / "summary.json", {"preregistration": preregistration, "runs": results})
    print(json.dumps({str(x["seed"]): {
        "clean": x["baseline_clean_accuracy"],
        "experiments": len(x["experiments"]),
    } for x in results}, indent=2))


if __name__ == "__main__":
    main()
