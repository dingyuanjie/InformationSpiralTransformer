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
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks
from run_level6_13_1_local import bootstrap_mean_ci, holm, mcnemar, save
from run_level6_14_1_local import choose_candidates, swap_slots


SEEDS = [606, 808, 1001]
SWAP_TIMES = [4, 8]
FIXED_SCALES = [0.10, 0.15, 0.20, 0.25]
RELATIVE_CAPS = [0.10, 0.20]
TRACE_STEPS = [0, 1, 2, 3]
TRACE_KEYS = [
    "relative_l2", "donor_projection", "clean_cosine", "update_gate",
    "old_new_relative", "old_new_cosine", "encoded_norm",
    "attended_memory_norm", "propagation_ratio", "propagation_multiplier",
]


def tag(value):
    return str(value).replace(".", "p")


def defenses(args):
    output = [{"defense": "baseline", "kind": "baseline", "value": 1.0,
               "offset": None, "duration": 0}]
    for scale in args.fixed_scales:
        output.append({"defense": f"fixed_{tag(scale)}_step1", "kind": "fixed",
                       "value": scale, "offset": 1, "duration": 1})
    output.append({"defense": "fixed_0p20_step2", "kind": "fixed",
                   "value": 0.20, "offset": 2, "duration": 1})
    output.append({"defense": "fixed_0p20_steps12", "kind": "fixed",
                   "value": 0.20, "offset": 1, "duration": 2})
    for cap in args.relative_caps:
        output.append({"defense": f"relative_cap_{tag(cap)}_step1",
                       "kind": "relative_cap", "value": cap,
                       "offset": 1, "duration": 1})
    return output


def intervention_active(spec, chunk_number, swap_after):
    if spec["offset"] is None:
        return False
    first = swap_after + spec["offset"]
    return first <= chunk_number < first + spec["duration"]


def configure(model, spec, chunk_number, swap_after):
    block = model.blocks[2]
    memory = block.memory
    block.historical_read_scale = 1.0
    block.historical_consistency_threshold = None
    memory.propagation_scale = 1.0
    memory.propagation_relative_cap = None
    memory.propagation_consistency_threshold = None
    if intervention_active(spec, chunk_number, swap_after):
        if spec["kind"] == "fixed":
            memory.propagation_scale = spec["value"]
        elif spec["kind"] == "relative_cap":
            memory.propagation_relative_cap = spec["value"]


def snapshot(model):
    diag = model.blocks[2].memory.last_diagnostics
    old = diag["old_memory"].float()
    new = diag["new_memory"].float()
    return {
        "update_gate": diag["update_gate"].float().mean(dim=(1, 2)),
        "old_new_relative": (new - old).norm(dim=-1).mean(dim=-1)
                            / old.norm(dim=-1).mean(dim=-1).clamp_min(1e-8),
        "old_new_cosine": F.cosine_similarity(old, new, dim=-1).mean(dim=-1),
        "encoded_norm": diag["encoded_norm"].float(),
        "attended_memory_norm": diag["attended_memory_norm"].float(),
        "propagation_ratio": diag["propagation_ratio"].float(),
        "propagation_multiplier": torch.full_like(
            diag["encoded_norm"].float(),
            float(diag["propagation_multiplier_mean"]),
        ),
    }


def state_trace(clean, polluted):
    clean = clean[2].float()
    polluted = polluted[2].float()
    delta = polluted - clean
    donor_delta = torch.roll(clean, shifts=1, dims=0) - clean
    return {
        "relative_l2": delta.norm(dim=-1).mean(dim=-1)
                       / clean.norm(dim=-1).mean(dim=-1).clamp_min(1e-8),
        "donor_projection": ((delta * donor_delta).sum(dim=-1)
                             / donor_delta.square().sum(dim=-1).clamp_min(1e-8)).mean(dim=-1),
        "clean_cosine": F.cosine_similarity(clean, polluted, dim=-1).mean(dim=-1),
    }


def append_trace(storage, step, sample_offset, state_values, process_values):
    batch_size = len(state_values["relative_l2"])
    storage["step"].extend([step] * len(state_values["relative_l2"]))
    storage["sample_index"].extend(range(sample_offset, sample_offset + batch_size))
    for key in ["relative_l2", "donor_projection", "clean_cosine"]:
        storage[key].extend(state_values[key].cpu().tolist())
    for key in TRACE_KEYS[3:]:
        storage[key].extend(process_values[key].cpu().tolist())


@torch.no_grad()
def evaluate(model, args, condition, device, dtype, eval_seeds):
    predictions, clean_predictions, targets, donor_targets = [], [], [], []
    trace = {key: [] for key in ["step", "sample_index", *TRACE_KEYS]}
    sample_offset = 0
    batches = args.samples_per_eval_seed // args.eval_batch_size
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device
            )
            reference_memory = None
            defended_clean_memory = None
            polluted_memory = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(args.chunks):
                    chunk_number = chunk_index + 1
                    configure(model, defenses(args)[0], chunk_number, condition["swap_after"])
                    reference_logits, reference_memory = model(
                        chunks[:, chunk_index], memory=reference_memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    configure(model, condition, chunk_number, condition["swap_after"])
                    clean_logits, defended_clean_memory = model(
                        chunks[:, chunk_index], memory=defended_clean_memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    configure(model, condition, chunk_number, condition["swap_after"])
                    logits, polluted_memory = model(
                        chunks[:, chunk_index], memory=polluted_memory,
                        return_memory=True, per_layer_memory=True,
                    )
                    process = snapshot(model)
                    if chunk_number == condition["swap_after"]:
                        polluted_memory = swap_slots(polluted_memory, condition["slots"])
                    relative_step = chunk_number - condition["swap_after"]
                    if relative_step in args.trace_steps:
                        append_trace(trace, relative_step, sample_offset,
                                     state_trace(reference_memory, polluted_memory), process)
            predictions.extend(logits[:, -1, :16].argmax(-1).cpu().tolist())
            clean_predictions.extend(clean_logits[:, -1, :16].argmax(-1).cpu().tolist())
            targets.extend(target.cpu().tolist())
            donor_targets.extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
            sample_offset += args.eval_batch_size
    configure(model, defenses(args)[0], 0, 0)
    prediction = np.asarray(predictions)
    clean_prediction = np.asarray(clean_predictions)
    target = np.asarray(targets)
    donor = np.asarray(donor_targets)
    mismatch = donor != target
    correct = (prediction == target).astype(np.int8)
    clean_correct = (clean_prediction == target).astype(np.int8)
    donor_hit = ((prediction == donor) & mismatch).astype(np.int8)
    return {
        "samples": int(len(target)), "accuracy": float(correct.mean()),
        "matched_clean_accuracy": float(clean_correct.mean()),
        "donor_attraction": float(donor_hit.sum() / mismatch.sum()),
        "correct": correct.tolist(), "matched_clean_correct": clean_correct.tolist(),
        "donor_hit": donor_hit.tolist(),
        "donor_mismatch": mismatch.astype(np.int8).tolist(), "trace": trace,
    }


def make_conditions(candidates, args):
    return [{
        "name": f"{candidate['key']}__swap{swap_after}__{spec['defense']}",
        "pair": candidate["key"], "slots": candidate["slots"],
        "category": candidate["category"], "swap_after": swap_after, **spec,
    } for candidate in candidates for swap_after in args.swap_times
      for spec in defenses(args)]


def masked_donor(row):
    mask = np.asarray(row["donor_mismatch"], dtype=bool)
    return np.asarray(row["donor_hit"], dtype=np.int8)[mask]


def trace_group_means(row, mask):
    sample_count = row["samples"]
    output = {}
    for step in sorted(set(row["trace"]["step"])):
        trace_step = np.asarray(row["trace"]["step"]) == step
        sample_index = np.asarray(row["trace"]["sample_index"], dtype=int)
        selected = trace_step & mask[sample_index]
        output[str(step)] = {
            key: (float(np.mean(np.asarray(row["trace"][key])[selected]))
                  if selected.any() else None) for key in TRACE_KEYS
        }
    return output


def analyze(seed, candidates, rows, args):
    table = {row["name"]: row for row in rows}
    groups = []
    for candidate_index, candidate in enumerate(candidates):
        for time_index, swap_after in enumerate(args.swap_times):
            prefix = f"{candidate['key']}__swap{swap_after}"
            baseline = table[f"{prefix}__baseline"]
            base_correct = np.asarray(baseline["correct"], dtype=np.int8)
            base_donor = masked_donor(baseline)
            entries, accuracy_tests, donor_tests = [], [], []
            for defense_index, spec in enumerate(defenses(args)[1:]):
                row = table[f"{prefix}__{spec['defense']}"]
                correct = np.asarray(row["correct"], dtype=np.int8)
                donor = masked_donor(row)
                accuracy_test = mcnemar(correct, base_correct)
                donor_test = mcnemar(donor, base_donor)
                accuracy_tests.append(accuracy_test)
                donor_tests.append(donor_test)
                corrected = (correct == 1) & (base_correct == 0)
                harmed = (correct == 0) & (base_correct == 1)
                entries.append({
                    **spec, "accuracy": row["accuracy"],
                    "matched_clean_accuracy": row["matched_clean_accuracy"],
                    "accuracy_gain": bootstrap_mean_ci(
                        correct - base_correct,
                        args.bootstrap_seed + seed * 10000 + candidate_index * 1000
                        + time_index * 100 + defense_index, args.bootstrap_iterations),
                    "donor_attraction": row["donor_attraction"],
                    "donor_reduction": bootstrap_mean_ci(
                        base_donor - donor,
                        args.bootstrap_seed + seed * 10000 + 5000 + candidate_index * 1000
                        + time_index * 100 + defense_index, args.bootstrap_iterations),
                    "corrected_samples": int(corrected.sum()),
                    "harmed_samples": int(harmed.sum()),
                    "corrected_trace": trace_group_means(row, corrected),
                    "harmed_trace": trace_group_means(row, harmed),
                    "all_trace": trace_group_means(row, np.ones_like(correct, dtype=bool)),
                    "mcnemar_accuracy": accuracy_test, "mcnemar_donor": donor_test,
                })
            holm(accuracy_tests)
            holm(donor_tests)
            for entry, atest, dtest in zip(entries, accuracy_tests, donor_tests):
                entry["mcnemar_accuracy"] = atest
                entry["mcnemar_donor"] = dtest
            groups.append({
                **candidate, "swap_after": swap_after,
                "baseline_accuracy": baseline["accuracy"],
                "baseline_matched_clean_accuracy": baseline["matched_clean_accuracy"],
                "baseline_donor_attraction": baseline["donor_attraction"],
                "baseline_trace": trace_group_means(
                    baseline, np.ones_like(base_correct, dtype=bool)),
                "entries": entries,
            })
    return {"seed": seed, "eval_seeds": None, "groups": groups}


def plot_seed(result, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = sorted({e["defense"] for g in result["groups"] for e in g["entries"]})
    gains = {name: np.mean([e["accuracy_gain"]["estimate"] * 100
                            for g in result["groups"] for e in g["entries"]
                            if e["defense"] == name]) for name in names}
    axes[0].bar(range(len(names)), [gains[x] for x in names])
    axes[0].set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Accuracy gain (pp)")
    axes[0].set_title("Intervention effects")
    for group in result["groups"]:
        base = group["baseline_trace"]
        steps = sorted(int(x) for x in base)
        axes[1].plot(steps, [base[str(x)]["relative_l2"] for x in steps], alpha=0.6)
        axes[2].plot(steps, [base[str(x)]["donor_projection"] for x in steps], alpha=0.6)
    axes[1].set(title="Baseline pollution trajectory", xlabel="Chunks after swap",
                ylabel="Relative L2")
    axes[2].set(title="Baseline donor projection", xlabel="Chunks after swap",
                ylabel="Projection")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f"Level 6.15.2 — seed {result['seed']}")
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
    eval_seeds = [args.eval_seed_base + seed * 100 + i for i in range(args.eval_seed_count)]
    conditions = make_conditions(candidates, args)
    predictions_path = folder / "predictions.json"
    rows = [] if args.force or not predictions_path.exists() else json.loads(
        predictions_path.read_text(encoding="utf-8"))
    done = {row["name"] for row in rows}
    for index, condition in enumerate(conditions, 1):
        if condition["name"] in done:
            continue
        metric = evaluate(model, args, condition, device, dtype, eval_seeds)
        rows.append({**condition, **metric})
        save(predictions_path, rows)
        print(f"seed={seed} [{index}/{len(conditions)}] {condition['name']} "
              f"acc={metric['accuracy']:.2%} clean={metric['matched_clean_accuracy']:.2%}",
              flush=True)
    result = analyze(seed, candidates, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    plot_seed(result, folder / "reverse_mechanism.png")
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.15.2 reverse mechanism diagnosis")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--fixed-scales", nargs="+", type=float, default=FIXED_SCALES)
    parser.add_argument("--relative-caps", nargs="+", type=float, default=RELATIVE_CAPS)
    parser.add_argument("--trace-steps", nargs="+", type=int, default=TRACE_STEPS)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=400)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--eval-seed-base", type=int, default=7550000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=615200)
    parser.add_argument("--output", default="experiments/level6_15_2/formal")
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
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "hard_model": 808,
        "primary_question": "why fixed first-successor attenuation reverses on seed808",
        "primary_pathway": "MemoryAttention-to-new-Memory",
        "fixed_scales": args.fixed_scales,
        "normalized_caps": args.relative_caps,
        "independent_eval_seed_base": args.eval_seed_base,
        "protocol": vars(args),
    }
    save(root / "preregistration.json", preregistration)
    results = []
    for seed in args.seeds:
        results.append(run_seed(seed, choose_candidates(registration, seed),
                                args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    save(root / "summary.json", {"preregistration": preregistration, "runs": results})
    print(json.dumps({str(x["seed"]): len(x["groups"]) for x in results}, indent=2))


if __name__ == "__main__":
    main()
