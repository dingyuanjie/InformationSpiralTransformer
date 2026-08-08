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
LAYERS = 3
SLOTS = 32
SWAP_TIMES = [1, 4, 8, 12]
DELAYS = [1, 2]
SCOPES = ["source_pair", "final_layer_all", "all_layers"]


def restore(memory, clean_memory, slots, scope):
    output = [item.clone() for item in memory]
    if scope == "source_pair":
        output[2][:, slots, :] = clean_memory[2][:, slots, :]
    elif scope == "final_layer_all":
        output[2] = clean_memory[2].clone()
    elif scope == "all_layers":
        output = [item.clone() for item in clean_memory]
    else:
        raise ValueError(f"Unknown restoration scope: {scope}")
    return output


def diagnostic(clean_memory, polluted_memory):
    relative = []
    projection = []
    cosine = []
    for clean, polluted in zip(clean_memory, polluted_memory):
        clean = clean.float()
        polluted = polluted.float()
        donor = torch.roll(clean, shifts=1, dims=0)
        delta = polluted - clean
        donor_delta = donor - clean
        delta_norm = delta.norm(dim=-1)
        clean_norm = clean.norm(dim=-1).clamp_min(1e-8)
        donor_norm_sq = donor_delta.square().sum(dim=-1).clamp_min(1e-8)
        dot = (delta * donor_delta).sum(dim=-1)
        relative.append((delta_norm / clean_norm).mean(dim=0))
        projection.append((dot / donor_norm_sq).mean(dim=0))
        cosine.append((dot / (delta_norm * donor_delta.norm(dim=-1)).clamp_min(1e-8)).mean(dim=0))
    return torch.stack(relative), torch.stack(projection), torch.stack(cosine)


@torch.no_grad()
def evaluate(model, args, condition, device, dtype, eval_seeds, tomography=False):
    predictions = []
    targets = []
    donor_targets = []
    relative_sum = torch.zeros(args.chunks, LAYERS, SLOTS)
    projection_sum = torch.zeros_like(relative_sum)
    cosine_sum = torch.zeros_like(relative_sum)
    diagnostic_batches = 0
    batches = args.samples_per_eval_seed // args.eval_batch_size
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device
            )
            clean_memory = None
            polluted_memory = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(args.chunks):
                    clean_logits, clean_produced = model(
                        chunks[:, chunk_index], memory=clean_memory, return_memory=True,
                        per_layer_memory=True,
                    )
                    polluted_logits, polluted_produced = model(
                        chunks[:, chunk_index], memory=polluted_memory, return_memory=True,
                        per_layer_memory=True,
                    )
                    clean_memory = clean_produced
                    polluted_memory = polluted_produced
                    if condition["mode"] != "intact":
                        chunk_number = chunk_index + 1
                        if chunk_number == condition["swap_after"]:
                            polluted_memory = swap_slots(polluted_memory, condition["slots"])
                        if condition["scope"] in SCOPES:
                            remediation_chunk = condition["swap_after"] + condition["delay"]
                            if chunk_number == remediation_chunk:
                                polluted_memory = restore(
                                    polluted_memory, clean_memory, condition["slots"],
                                    condition["scope"],
                                )
                    if tomography:
                        relative, projection, cosine = diagnostic(clean_memory, polluted_memory)
                        relative_sum[chunk_index] += relative.cpu()
                        projection_sum[chunk_index] += projection.cpu()
                        cosine_sum[chunk_index] += cosine.cpu()
            logits = clean_logits if condition["mode"] == "intact" else polluted_logits
            prediction = logits[:, -1, :16].argmax(-1)
            predictions.extend(prediction.cpu().tolist())
            targets.extend(target.cpu().tolist())
            donor_targets.extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
            if tomography:
                diagnostic_batches += 1
    prediction = np.asarray(predictions)
    target = np.asarray(targets)
    donor = np.asarray(donor_targets)
    correct = (prediction == target).astype(np.int8)
    mismatch = donor != target
    donor_hit = ((prediction == donor) & mismatch).astype(np.int8)
    result = {
        "samples": int(len(target)),
        "accuracy": float(correct.mean()),
        "donor_attraction": float(donor_hit.sum() / mismatch.sum()),
        "correct": correct.tolist(),
        "donor_hit": donor_hit.tolist(),
        "donor_mismatch": mismatch.astype(np.int8).tolist(),
    }
    if tomography:
        result["tomography"] = {
            "relative_l2": (relative_sum / diagnostic_batches).tolist(),
            "donor_projection": (projection_sum / diagnostic_batches).tolist(),
            "donor_cosine": (cosine_sum / diagnostic_batches).tolist(),
        }
    return result


def make_conditions(candidates, swap_times, delays):
    output = [{"name": "intact", "mode": "intact", "pair": None, "slots": None,
               "category": "baseline", "swap_after": None, "scope": "none", "delay": None}]
    for candidate in candidates:
        for swap_after in swap_times:
            output.append({
                "name": f"{candidate['key']}__swap{swap_after}__hold",
                "mode": "pollution", "pair": candidate["key"], "slots": candidate["slots"],
                "category": candidate["category"], "swap_after": swap_after,
                "scope": "hold", "delay": -1,
            })
            for scope in SCOPES:
                for delay in delays:
                    output.append({
                        "name": f"{candidate['key']}__swap{swap_after}__{scope}__delay{delay}",
                        "mode": "pollution", "pair": candidate["key"],
                        "slots": candidate["slots"], "category": candidate["category"],
                        "swap_after": swap_after, "scope": scope, "delay": delay,
                    })
    return output


def masked_donor(row):
    mask = np.asarray(row["donor_mismatch"], dtype=bool)
    return np.asarray(row["donor_hit"], dtype=np.int8)[mask]


def analyze(seed, candidates, rows, args):
    table = {row["name"]: row for row in rows}
    intact = table["intact"]
    groups = []
    for candidate_index, candidate in enumerate(candidates):
        for time_index, swap_after in enumerate(args.swap_times):
            hold = table[f"{candidate['key']}__swap{swap_after}__hold"]
            hold_correct = np.asarray(hold["correct"], dtype=np.int8)
            hold_donor = masked_donor(hold)
            restorations = []
            accuracy_family = []
            donor_family = []
            for scope_index, scope in enumerate(SCOPES):
                for delay_index, delay in enumerate(args.delays):
                    row = table[f"{candidate['key']}__swap{swap_after}__{scope}__delay{delay}"]
                    corrected = np.asarray(row["correct"], dtype=np.int8)
                    corrected_donor = masked_donor(row)
                    accuracy_test = mcnemar(corrected, hold_correct)
                    donor_test = mcnemar(corrected_donor, hold_donor)
                    accuracy_family.append(accuracy_test)
                    donor_family.append(donor_test)
                    restorations.append({
                        "scope": scope,
                        "delay": delay,
                        "accuracy": row["accuracy"],
                        "donor_attraction": row["donor_attraction"],
                        "accuracy_gain_vs_hold": bootstrap_mean_ci(
                            corrected - hold_correct,
                            args.bootstrap_seed + seed * 1000 + candidate_index * 100
                            + time_index * 20 + scope_index * 5 + delay_index,
                            args.bootstrap_iterations,
                        ),
                        "donor_reduction_vs_hold": bootstrap_mean_ci(
                            hold_donor - corrected_donor,
                            args.bootstrap_seed + seed * 1000 + candidate_index * 100
                            + 50 + time_index * 20 + scope_index * 5 + delay_index,
                            args.bootstrap_iterations,
                        ),
                        "mcnemar_accuracy_vs_hold": accuracy_test,
                        "mcnemar_donor_vs_hold": donor_test,
                    })
            holm(accuracy_family)
            holm(donor_family)
            for item, accuracy_test, donor_test in zip(restorations, accuracy_family, donor_family):
                item["mcnemar_accuracy_vs_hold"] = accuracy_test
                item["mcnemar_donor_vs_hold"] = donor_test
            tomography = hold["tomography"]
            relative = np.asarray(tomography["relative_l2"])
            source = candidate["slots"]
            outside = [slot for slot in range(SLOTS) if slot not in source]
            layer_mass = relative.mean(axis=2)
            source_mass = relative[:, 2, source].mean(axis=1)
            outside_mass = relative[:, 2, outside].mean(axis=1)
            groups.append({
                **candidate,
                "swap_after": swap_after,
                "hold_accuracy": hold["accuracy"],
                "hold_donor_attraction": hold["donor_attraction"],
                "tomography": tomography,
                "layer_relative_l2": layer_mass.tolist(),
                "source_relative_l2": source_mass.tolist(),
                "outside_relative_l2": outside_mass.tolist(),
                "restorations": restorations,
            })
    return {
        "seed": seed,
        "samples_per_condition": intact["samples"],
        "intact_accuracy": intact["accuracy"],
        "intact_donor_attraction": intact["donor_attraction"],
        "candidates": candidates,
        "groups": groups,
    }


def plot_pair(result, candidate, path, swap_times):
    fig, axes = plt.subplots(len(swap_times), 3, figsize=(15, 3 * len(swap_times) + 1),
                             squeeze=False)
    for row_index, swap_after in enumerate(swap_times):
        group = next(x for x in result["groups"]
                     if x["key"] == candidate["key"] and x["swap_after"] == swap_after)
        relative = np.asarray(group["tomography"]["relative_l2"])
        projection = np.asarray(group["tomography"]["donor_projection"])
        image0 = axes[row_index, 0].imshow(relative[:, 2, :].T, aspect="auto", origin="lower",
                                           cmap="magma", vmin=0)
        image1 = axes[row_index, 1].imshow(projection[:, 2, :].T, aspect="auto", origin="lower",
                                           cmap="coolwarm", vmin=-1, vmax=1)
        chunks = np.arange(1, relative.shape[0] + 1)
        for layer in range(LAYERS):
            axes[row_index, 2].plot(chunks, relative[:, layer, :].mean(axis=1),
                                    label=f"layer {layer}")
        axes[row_index, 0].set_ylabel(f"Swap {swap_after}\nSlot")
        axes[row_index, 0].set_title("Final-layer relative L2")
        axes[row_index, 1].set_title("Final-layer donor projection")
        axes[row_index, 2].set_title("Mean contamination by layer")
        axes[row_index, 2].axvline(swap_after, color="black", linestyle="--", linewidth=1)
        axes[row_index, 2].grid(alpha=0.2)
        fig.colorbar(image0, ax=axes[row_index, 0], fraction=0.04, pad=0.02)
        fig.colorbar(image1, ax=axes[row_index, 1], fraction=0.04, pad=0.02)
    axes[-1, 0].set_xlabel("Chunk")
    axes[-1, 1].set_xlabel("Chunk")
    axes[-1, 2].set_xlabel("Chunk")
    axes[0, 2].legend()
    fig.suptitle(f"IST seed {result['seed']} pair {candidate['key']}: pollution tomography")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_restoration(result, candidate, path, swap_times, delays):
    fig, axes = plt.subplots(1, len(swap_times), figsize=(4 * len(swap_times), 4),
                             sharey=True, squeeze=False)
    for index, swap_after in enumerate(swap_times):
        axis = axes[0, index]
        group = next(x for x in result["groups"]
                     if x["key"] == candidate["key"] and x["swap_after"] == swap_after)
        for scope in SCOPES:
            points = [x for x in group["restorations"] if x["scope"] == scope]
            axis.plot(delays, [x["accuracy"] * 100 for x in points], marker="o", label=scope)
        axis.axhline(group["hold_accuracy"] * 100, color="tab:red", linestyle="--",
                     label="hold")
        axis.axhline(result["intact_accuracy"] * 100, color="black", linestyle=":",
                     label="intact")
        axis.set_title(f"Swap after {swap_after}")
        axis.set_xlabel("Restore delay")
        axis.set_xticks(delays)
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"IST seed {result['seed']} pair {candidate['key']}: restoration scope")
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
        tomography = condition["scope"] == "hold"
        metric = evaluate(model, args, condition, device, dtype, eval_seeds,
                          tomography=tomography)
        rows.append({**condition, **metric})
        save(progress_path, rows)
        print(f"seed={seed} [{index}/{len(specs)}] {condition['name']} "
              f"acc={metric['accuracy']:.2%} donor={metric['donor_attraction']:.2%}", flush=True)
    result = analyze(seed, candidates, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    for candidate in candidates:
        plot_pair(result, candidate, folder / f"pair_{candidate['key']}_tomography.png",
                  args.swap_times)
        plot_restoration(result, candidate,
                         folder / f"pair_{candidate['key']}_restoration_scope.png",
                         args.swap_times, args.delays)
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.14.2 pollution propagation tomography")
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
    parser.add_argument("--eval-seed-base", type=int, default=7142000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=614200)
    parser.add_argument("--output", default="experiments/level6_14_2/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
    if any(time < 1 or time >= args.chunks for time in args.swap_times):
        raise ValueError("swap times must be between 1 and chunks-1")
    if any(delay < 1 for delay in args.delays):
        raise ValueError("restoration delays must be positive")
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
        "intact": result["intact_accuracy"],
        "pairs": [candidate["key"] for candidate in result["candidates"]],
    } for result in results}, indent=2))


if __name__ == "__main__":
    main()
