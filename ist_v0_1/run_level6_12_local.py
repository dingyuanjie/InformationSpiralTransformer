import argparse
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
import torch.nn as nn

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks


SEEDS = [606, 808, 1001]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def slot_mask(memory, keep):
    mask = torch.zeros(memory[2].shape[1], device=memory[2].device, dtype=memory[2].dtype)
    mask[keep] = 1
    output = list(memory); output[2] = memory[2] * mask[None, :, None]
    return output


@torch.no_grad()
def evaluate(model, args, keep, device, dtype, seed, diagnostics=False):
    set_seed(seed); model.eval(); query = local = total = 0
    attention = torch.zeros(32); norms = torch.zeros(32); gates = torch.zeros(32); batches = 0
    for _ in range(args.eval_batches):
        chunks, target, pos = make_chunks(args.eval_batch_size, args.chunks, args.chunk_size, device)
        memory = None; first = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for index in range(args.chunks):
                logits, produced = model(chunks[:, index], memory=memory, return_memory=True,
                                         per_layer_memory=True)
                if index == 0: first = logits
                memory = produced if keep is None else slot_mask(produced, keep)
        rows = torch.arange(len(target), device=device)
        query += (logits[:, -1, :16].argmax(-1) == target).sum().item()
        local += (first[rows, pos, :16].argmax(-1) == target).sum().item(); total += len(target)
        if diagnostics:
            block = model.blocks[2]
            weights = block.last_memory_read_weights.float()  # [B, heads, tokens, slots]
            attention += weights.mean(dim=(0, 1, 2)).cpu()
            norms += produced[2].float().norm(dim=-1).mean(dim=0).cpu()
            gates += block.memory.last_diagnostics["update_gate"].float().mean(dim=(0, 2)).cpu()
            batches += 1
    result = {"query": query / total, "local": local / total, "samples": total}
    if diagnostics:
        result["attention"] = (attention / batches).tolist()
        result["slot_norm"] = (norms / batches).tolist()
        result["update_gate"] = (gates / batches).tolist()
        result["fusion_gate_mean"] = model.blocks[2].last_fusion_gate.float().mean().item()
    return result


def correlation(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return None
    return float(np.corrcoef(a, b)[0, 1])


def ranks(values):
    values = np.asarray(values); order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float); result[order] = np.arange(len(values))
    return result


def plot_map(result, path, title):
    slots = np.arange(32); intact = result["intact"]["query"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(slots, np.array(result["keep_one"]) * 100)
    axes[0].axhline(intact * 100, color="black", linestyle="--", linewidth=1, label="intact")
    axes[0].set_ylabel("Keep-one query (%)"); axes[0].legend()
    axes[1].bar(slots, np.array(result["necessity_drop"]) * 100)
    axes[1].axhline(0, color="black", linewidth=0.8); axes[1].set_ylabel("Leave-one-out drop (pp)")
    axes[2].bar(slots, np.array(result["attention"]) * 100)
    axes[2].set_ylabel("Read attention (%)"); axes[2].set_xlabel("Final-layer memory slot")
    axes[2].set_xticks(np.arange(0, 32, 2)); fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def run_seed(seed, args, device, dtype, root):
    folder = root / f"seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    source = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(source, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    model.load_state_dict(state["model"]); model.blocks[2].capture_memory_read_weights = True
    eval_seed = args.eval_seed_base + seed
    progress_path = folder / "progress.json"
    rows = [] if args.force or not progress_path.exists() else json.loads(progress_path.read_text(encoding="utf-8"))
    done = {row["name"] for row in rows}
    condition_specs = [("intact", None)]
    condition_specs += [(f"keep_one_{slot}", [slot]) for slot in range(32)]
    condition_specs += [(f"leave_one_{slot}", [item for item in range(32) if item != slot])
                        for slot in range(32)]
    for name, keep in condition_specs:
        if name in done: continue
        metric = evaluate(model, args, keep, device, dtype, eval_seed, diagnostics=name == "intact")
        rows.append({"name": name, "keep": keep, **metric}); save(progress_path, rows)
        print(f"seed={seed} {name} query={metric['query']:.2%}", flush=True)
    table = {row["name"]: row for row in rows}; intact = table["intact"]
    keep_one = [table[f"keep_one_{slot}"]["query"] for slot in range(32)]
    leave_one = [table[f"leave_one_{slot}"]["query"] for slot in range(32)]
    necessity = [intact["query"] - value for value in leave_one]
    attention = intact["attention"]; norms = intact["slot_norm"]; gates = intact["update_gate"]
    result = {"seed": seed, "source": str(source), "intact": intact,
              "keep_one": keep_one, "leave_one": leave_one, "necessity_drop": necessity,
              "attention": attention, "slot_norm": norms, "update_gate": gates,
              "correlations": {
                  "attention_keep_pearson": correlation(attention, keep_one),
                  "attention_keep_spearman": correlation(ranks(attention), ranks(keep_one)),
                  "attention_necessity_pearson": correlation(attention, necessity),
                  "attention_necessity_spearman": correlation(ranks(attention), ranks(necessity)),
                  "norm_keep_pearson": correlation(norms, keep_one),
                  "gate_keep_pearson": correlation(gates, keep_one),
              },
              "best_keep_slots": np.argsort(keep_one)[::-1][:8].tolist(),
              "most_necessary_slots": np.argsort(necessity)[::-1][:8].tolist()}
    save(result_path, result)
    plot_map(result, folder / "slot_causal_map.png", f"IST seed {seed}: final-layer slot causal map")
    return result


def main():
    p = argparse.ArgumentParser(description="Level 6.12 exhaustive final-layer causal slot map")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--eval-seed-base", type=int, default=712500)
    p.add_argument("--output", default="experiments/level6_12/formal")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
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
    aggregate = {
        "intact": {"query": statistics.mean(r["intact"]["query"] for r in results)},
        "keep_one": np.mean([r["keep_one"] for r in results], axis=0).tolist(),
        "leave_one": np.mean([r["leave_one"] for r in results], axis=0).tolist(),
        "necessity_drop": np.mean([r["necessity_drop"] for r in results], axis=0).tolist(),
        "attention": np.mean([r["attention"] for r in results], axis=0).tolist(),
    }
    aggregate["correlations"] = {
        "attention_keep_pearson": correlation(aggregate["attention"], aggregate["keep_one"]),
        "attention_keep_spearman": correlation(ranks(aggregate["attention"]), ranks(aggregate["keep_one"])),
        "attention_necessity_pearson": correlation(aggregate["attention"], aggregate["necessity_drop"]),
    }
    summary = {"runs": len(results), "aggregate": aggregate,
               "per_seed_correlations": {str(r["seed"]): r["correlations"] for r in results}}
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    plot_map(aggregate, root / "aggregate_slot_causal_map.png", "IST aggregate final-layer slot causal map")
    print(json.dumps(summary["per_seed_correlations"], indent=2))


if __name__ == "__main__":
    main()
