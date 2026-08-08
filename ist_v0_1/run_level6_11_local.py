import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks, vector


SEEDS = [606, 808, 1001]
KEEP_COUNTS = [1, 2, 4, 8, 16, 32]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def apply_intervention(memory, condition):
    kind = condition["kind"]
    if kind == "intact":
        return memory
    if kind == "zero_layer":
        return [torch.zeros_like(item) if index == condition["layer"] else item
                for index, item in enumerate(memory)]
    if kind == "only_layer":
        return [item if index == condition["layer"] else torch.zeros_like(item)
                for index, item in enumerate(memory)]
    if kind == "keep_slots":
        layer = condition["layer"]
        mask = torch.zeros(memory[layer].shape[1], device=memory[layer].device,
                           dtype=memory[layer].dtype)
        mask[condition["slots"]] = 1
        output = list(memory)
        output[layer] = memory[layer] * mask[None, :, None]
        return output
    raise ValueError(kind)


@torch.no_grad()
def evaluate(model, probe, args, condition, device, dtype, seed):
    set_seed(seed); model.eval(); probe.eval()
    query = local = probe_final = total = 0
    for _ in range(args.eval_batches):
        chunks, target, pos = make_chunks(args.eval_batch_size, args.chunks,
                                          args.chunk_size, device)
        memory = None; first_logits = None; final_probe = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for index in range(args.chunks):
                logits, produced = model(chunks[:, index], memory=memory,
                                         return_memory=True, per_layer_memory=True)
                if index == 0: first_logits = logits
                final_probe = probe(vector(produced))
                memory = apply_intervention(produced, condition)
        rows = torch.arange(len(target), device=device)
        query += (logits[:, -1, :16].argmax(-1) == target).sum().item()
        local += (first_logits[rows, pos, :16].argmax(-1) == target).sum().item()
        probe_final += (final_probe.argmax(-1) == target).sum().item(); total += len(target)
    return {"query": query / total, "local": local / total,
            "probe_final": probe_final / total, "samples": total}


def conditions(seed, args):
    output = [{"name": "intact", "kind": "intact"}]
    for layer in range(3):
        output.append({"name": f"zero_layer{layer}", "kind": "zero_layer", "layer": layer})
    for layer in range(3):
        output.append({"name": f"only_layer{layer}", "kind": "only_layer", "layer": layer})
    for count in KEEP_COUNTS:
        output.append({"name": f"fixed_keep{count}", "kind": "keep_slots", "layer": 2,
                       "slots": list(range(count)), "selection": "fixed", "keep": count})
        for repeat in range(args.random_repeats):
            generator = torch.Generator().manual_seed(args.mask_seed_base + seed * 1000
                                                       + count * 10 + repeat)
            slots = torch.randperm(32, generator=generator)[:count].sort().values.tolist()
            output.append({"name": f"random_keep{count}_r{repeat}", "kind": "keep_slots",
                           "layer": 2, "slots": slots, "selection": "random",
                           "keep": count, "repeat": repeat})
    return output


def run_seed(seed, args, device, dtype, root):
    folder = root / f"seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    source = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(source, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    probe = nn.Linear(192, 16).to(device)
    model.load_state_dict(state["model"]); probe.load_state_dict(state["probe"])
    rows = []; eval_seed = args.eval_seed_base + seed
    for condition in conditions(seed, args):
        metric = evaluate(model, probe, args, condition, device, dtype, eval_seed)
        row = {**condition, **metric}; rows.append(row); save(folder / "progress.json", rows)
        print(f"seed={seed} {condition['name']} query={metric['query']:.2%} ", flush=True)
    table = {row["name"]: row for row in rows}; intact = table["intact"]["query"]
    layer_summary = {
        "intact": intact,
        "zero": [table[f"zero_layer{i}"]["query"] for i in range(3)],
        "only": [table[f"only_layer{i}"]["query"] for i in range(3)],
    }
    layer_summary["localized_to_layer2"] = (
        intact >= args.intact_threshold
        and layer_summary["zero"][2] <= args.destructive_threshold
        and layer_summary["only"][2] >= args.intact_threshold
    )
    slot_summary = []
    for count in KEEP_COUNTS:
        fixed = table[f"fixed_keep{count}"]["query"]
        random_values = [table[f"random_keep{count}_r{repeat}"]["query"]
                         for repeat in range(args.random_repeats)]
        slot_summary.append({"keep": count, "fixed_query": fixed,
                             "random_mean_query": statistics.mean(random_values),
                             "random_min_query": min(random_values),
                             "random_max_query": max(random_values),
                             "random_values": random_values})
    fixed_minimum = next((row["keep"] for row in slot_summary
                          if row["fixed_query"] >= args.intact_threshold), None)
    random_minimum = next((row["keep"] for row in slot_summary
                           if row["random_min_query"] >= args.intact_threshold), None)
    result = {"seed": seed, "source": str(source), "layer_summary": layer_summary,
              "slot_summary": slot_summary, "minimum_fixed_slots": fixed_minimum,
              "minimum_random_slots_all_repeats": random_minimum, "metrics": rows}
    save(result_path, result); return result


def main():
    p = argparse.ArgumentParser(description="Level 6.11 selective causal memory intervention")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--random-repeats", type=int, default=3)
    p.add_argument("--eval-seed-base", type=int, default=711000)
    p.add_argument("--mask-seed-base", type=int, default=712000)
    p.add_argument("--intact-threshold", type=float, default=0.90)
    p.add_argument("--destructive-threshold", type=float, default=0.20)
    p.add_argument("--output", default="experiments/level6_11/formal")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.eval_batch_size < 2: raise ValueError("eval-batch-size must be >=2")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda"); dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True); results = []
    for seed in args.seeds:
        results.append(run_seed(seed, args, device, dtype, root)); save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    summary = {"runs": len(results),
               "layer2_localization_passes": sum(r["layer_summary"]["localized_to_layer2"] for r in results),
               "minimum_fixed_slots": {str(r["seed"]): r["minimum_fixed_slots"] for r in results},
               "minimum_random_slots_all_repeats": {
                   str(r["seed"]): r["minimum_random_slots_all_repeats"] for r in results}}
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
