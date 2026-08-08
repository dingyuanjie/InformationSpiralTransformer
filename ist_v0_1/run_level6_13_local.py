import argparse
import itertools
import json
import os
import statistics
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
CHANCE = 1.0 / 16.0


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def mask_slots(memory, keep):
    mask = torch.zeros(SLOTS, device=memory[2].device, dtype=memory[2].dtype)
    mask[keep] = 1
    output = list(memory)
    output[2] = memory[2] * mask[None, :, None]
    return output


@torch.no_grad()
def evaluate(model, args, keep, device, dtype, seed):
    set_seed(seed)
    model.eval()
    query = local = total = 0
    for _ in range(args.eval_batches):
        chunks, target, pos = make_chunks(
            args.eval_batch_size, args.chunks, args.chunk_size, device
        )
        memory = None
        first = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for index in range(args.chunks):
                logits, produced = model(
                    chunks[:, index], memory=memory, return_memory=True,
                    per_layer_memory=True,
                )
                if index == 0:
                    first = logits
                memory = produced if keep is None else mask_slots(produced, keep)
        rows = torch.arange(len(target), device=device)
        query += (logits[:, -1, :16].argmax(-1) == target).sum().item()
        local += (first[rows, pos, :16].argmax(-1) == target).sum().item()
        total += len(target)
    return {"query": query / total, "local": local / total, "samples": total}


def matrix(values, diagonal):
    output = np.full((SLOTS, SLOTS), np.nan)
    np.fill_diagonal(output, diagonal)
    for key, value in values.items():
        left, right = (int(item) for item in key.split("-"))
        output[left, right] = output[right, left] = value
    return output


def top_pairs(values, count=12, reverse=True):
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)[:count]
    return [{"slots": [int(x) for x in key.split("-")], "value": value}
            for key, value in ordered]


def analyze(seed, source, intact, singles, pair_rows):
    keep_pair = {}
    leave_pair = {}
    for row in pair_rows:
        key = f"{row['slots'][0]}-{row['slots'][1]}"
        target = keep_pair if row["kind"] == "keep_pair" else leave_pair
        target[key] = row["query"]

    single_keep = singles["keep_one"]
    single_drop = singles["necessity_drop"]
    gain_over_best = {}
    additive_keep_interaction = {}
    pair_drop = {}
    deletion_interaction = {}
    for key, score in keep_pair.items():
        left, right = (int(x) for x in key.split("-"))
        gain_over_best[key] = score - max(single_keep[left], single_keep[right])
        additive_keep_interaction[key] = (
            score - single_keep[left] - single_keep[right] + CHANCE
        )
    for key, score in leave_pair.items():
        left, right = (int(x) for x in key.split("-"))
        pair_drop[key] = intact["query"] - score
        deletion_interaction[key] = pair_drop[key] - single_drop[left] - single_drop[right]

    sufficient = [
        {"slots": [int(x) for x in key.split("-")], "query": score}
        for key, score in keep_pair.items() if score >= 0.90
    ]
    weak_to_strong = [
        {"slots": [left, right], "query": score,
         "single_query": [single_keep[left], single_keep[right]]}
        for key, score in keep_pair.items()
        for left, right in [[int(x) for x in key.split("-")]]
        if single_keep[left] < 0.20 and single_keep[right] < 0.20 and score >= 0.80
    ]
    return {
        "seed": seed,
        "source": str(source),
        "intact": intact,
        "single_keep": single_keep,
        "single_drop": single_drop,
        "keep_pair": keep_pair,
        "leave_pair": leave_pair,
        "pair_drop": pair_drop,
        "gain_over_best_single": gain_over_best,
        "additive_keep_interaction": additive_keep_interaction,
        "deletion_interaction": deletion_interaction,
        "sufficient_pair_count": len(sufficient),
        "weak_to_strong_pairs": weak_to_strong,
        "best_keep_pairs": top_pairs(keep_pair),
        "largest_gain_pairs": top_pairs(gain_over_best),
        "most_necessary_pairs": top_pairs(pair_drop),
        "largest_deletion_interactions": top_pairs(deletion_interaction),
    }


def plot_result(result, path, title):
    keep = matrix(result["keep_pair"], result["single_keep"]) * 100
    gain = matrix(result["gain_over_best_single"], np.zeros(SLOTS)) * 100
    drop = matrix(result["pair_drop"], result["single_drop"]) * 100
    interaction = matrix(result["deletion_interaction"], np.zeros(SLOTS)) * 100
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        (keep, "Keep-pair accuracy (%)", "viridis", 0, 100),
        (gain, "Gain over best single (pp)", "coolwarm", None, None),
        (drop, "Leave-pair-out drop (pp)", "coolwarm", None, None),
        (interaction, "Joint-deletion interaction (pp)", "coolwarm", None, None),
    ]
    for axis, (data, label, cmap, vmin, vmax) in zip(axes.flat, panels):
        if vmin is None:
            bound = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.25)
            vmin, vmax = -bound, bound
        image = axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axis.set_title(label)
        axis.set_xlabel("Slot B")
        axis.set_ylabel("Slot A")
        axis.set_xticks(range(0, SLOTS, 4))
        axis.set_yticks(range(0, SLOTS, 4))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_seed(seed, args, device, dtype, root):
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    level12_path = Path(args.level6_12_root) / f"seed{seed}" / "result.json"
    singles = json.loads(level12_path.read_text(encoding="utf-8"))
    source = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(source, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    model.load_state_dict(state["model"])

    progress_path = folder / "progress.json"
    rows = [] if args.force or not progress_path.exists() else json.loads(
        progress_path.read_text(encoding="utf-8")
    )
    done = {row["name"] for row in rows}
    pairs = list(itertools.combinations(range(SLOTS), 2))
    if args.pair_limit is not None:
        pairs = pairs[:args.pair_limit]
    conditions = []
    for left, right in pairs:
        conditions.append({"name": f"keep_pair_{left}_{right}", "kind": "keep_pair",
                           "slots": [left, right], "keep": [left, right]})
    for left, right in pairs:
        keep = [slot for slot in range(SLOTS) if slot not in (left, right)]
        conditions.append({"name": f"leave_pair_{left}_{right}", "kind": "leave_pair",
                           "slots": [left, right], "keep": keep})

    eval_seed = args.eval_seed_base + seed
    for index, condition in enumerate(conditions, start=1):
        if condition["name"] in done:
            continue
        metric = evaluate(model, args, condition["keep"], device, dtype, eval_seed)
        row = {key: value for key, value in condition.items() if key != "keep"}
        rows.append({**row, **metric})
        save(progress_path, rows)
        print(f"seed={seed} [{index}/{len(conditions)}] {condition['name']} "
              f"query={metric['query']:.2%}", flush=True)

    result = analyze(seed, source, singles["intact"], singles, rows)
    save(result_path, result)
    plot_result(result, folder / "pair_causal_map.png",
                f"IST seed {seed}: final-layer pair causal map")
    return result


def aggregate_results(results):
    keys = list(results[0]["keep_pair"])
    mean_dict = lambda field: {
        key: statistics.mean(result[field][key] for result in results) for key in keys
    }
    minimum_keep = {
        key: min(result["keep_pair"][key] for result in results) for key in keys
    }
    aggregate = {
        "intact": {"query": statistics.mean(r["intact"]["query"] for r in results)},
        "single_keep": np.mean([r["single_keep"] for r in results], axis=0).tolist(),
        "single_drop": np.mean([r["single_drop"] for r in results], axis=0).tolist(),
        "keep_pair": mean_dict("keep_pair"),
        "leave_pair": mean_dict("leave_pair"),
        "pair_drop": mean_dict("pair_drop"),
        "gain_over_best_single": mean_dict("gain_over_best_single"),
        "additive_keep_interaction": mean_dict("additive_keep_interaction"),
        "deletion_interaction": mean_dict("deletion_interaction"),
    }
    robust = {key: value for key, value in minimum_keep.items() if value >= 0.90}
    aggregate["minimum_keep_across_seeds"] = minimum_keep
    aggregate["robust_90_pair_count"] = len(robust)
    aggregate["best_robust_pairs"] = top_pairs(minimum_keep)
    aggregate["best_keep_pairs"] = top_pairs(aggregate["keep_pair"])
    aggregate["largest_gain_pairs"] = top_pairs(aggregate["gain_over_best_single"])
    aggregate["most_necessary_pairs"] = top_pairs(aggregate["pair_drop"])
    aggregate["largest_deletion_interactions"] = top_pairs(
        aggregate["deletion_interaction"]
    )
    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.13 exhaustive final-layer slot-pair causal analysis"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-12-root", default="experiments/level6_12/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=712500)
    parser.add_argument("--pair-limit", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", default="experiments/level6_13/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

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
    results = []
    for seed in args.seeds:
        results.append(run_seed(seed, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    aggregate = aggregate_results(results)
    summary = {"protocol": vars(args), "runs": results, "aggregate": aggregate}
    save(root / "summary.json", summary)
    plot_result(aggregate, root / "aggregate_pair_causal_map.png",
                "IST aggregate final-layer slot-pair causal map")
    print(json.dumps({
        "robust_90_pair_count": aggregate["robust_90_pair_count"],
        "best_robust_pairs": aggregate["best_robust_pairs"],
        "largest_gain_pairs": aggregate["largest_gain_pairs"][:5],
        "most_necessary_pairs": aggregate["most_necessary_pairs"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
