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


DEFAULT_SEEDS = [606, 808, 1001]
CONDITIONS = ["intact", "reset", "zero", "batch_roll"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def intervene(memory, condition):
    if condition == "intact":
        return memory
    if condition == "reset":
        return None
    if condition == "zero":
        return [torch.zeros_like(item) for item in memory]
    if condition == "batch_roll":
        # Deterministic derangement for batch size > 1: every sample receives
        # another sample's memory while token inputs remain unchanged.
        return [item.roll(1, dims=0) for item in memory]
    raise ValueError(condition)


@torch.no_grad()
def evaluate_condition(model, probe, args, chunks_count, condition, device, dtype):
    model.eval(); probe.eval()
    query = local = probe_final = total = 0
    for _ in range(args.eval_batches):
        chunks, target, pos = make_chunks(args.eval_batch_size, chunks_count, args.chunk_size, device)
        memory = None
        first_logits = None
        final_probe = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for index in range(chunks_count):
                logits, produced = model(
                    chunks[:, index], memory=memory, return_memory=True, per_layer_memory=True
                )
                if index == 0:
                    first_logits = logits
                final_probe = probe(vector(produced))
                memory = intervene(produced, condition)
        rows = torch.arange(len(target), device=device)
        query += (logits[:, -1, :16].argmax(-1) == target).sum().item()
        local += (first_logits[rows, pos, :16].argmax(-1) == target).sum().item()
        probe_final += (final_probe.argmax(-1) == target).sum().item()
        total += len(target)
    return {
        "condition": condition,
        "chunks": chunks_count,
        "total_tokens": chunks_count * args.chunk_size,
        "query": query / total,
        "local": local / total,
        "probe_final": probe_final / total,
        "samples": total,
    }


def run_seed(seed, args, device, dtype, root):
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    probe = nn.Linear(192, 16).to(device)
    model.load_state_dict(state["model"]); probe.load_state_dict(state["probe"])
    rows = []
    for chunks_count in args.chunks:
        eval_seed = args.eval_seed_base + seed * 100 + chunks_count
        for condition in CONDITIONS:
            set_seed(eval_seed)
            metric = evaluate_condition(model, probe, args, chunks_count, condition, device, dtype)
            rows.append(metric); save(folder / "progress.json", rows)
            print(
                f"seed={seed} chunks={chunks_count} condition={condition} "
                f"query={metric['query']:.2%} local={metric['local']:.2%}", flush=True
            )
    by_chunks = {}
    causal_passed = True
    for count in args.chunks:
        selected = {row["condition"]: row for row in rows if row["chunks"] == count}
        causal = {
            "intact_query": selected["intact"]["query"],
            "max_intervened_query": max(selected[name]["query"] for name in CONDITIONS[1:]),
            "min_local": min(selected[name]["local"] for name in CONDITIONS),
            "query_drop": selected["intact"]["query"]
            - max(selected[name]["query"] for name in CONDITIONS[1:]),
        }
        causal["passed"] = (
            selected["intact"]["query"] >= args.intact_threshold
            and causal["max_intervened_query"] <= args.intervention_threshold
            and causal["min_local"] >= args.local_threshold
        )
        causal_passed = causal_passed and causal["passed"]
        by_chunks[str(count)] = causal
    result = {"seed": seed, "checkpoint": str(checkpoint_path), "causal_passed": causal_passed,
              "by_chunks": by_chunks, "metrics": rows}
    save(result_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description="Level 6.9 causal persistent-memory interventions")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--chunks", nargs="+", type=int, default=[2, 4, 8, 16])
    p.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--eval-seed-base", type=int, default=690000)
    p.add_argument("--intact-threshold", type=float, default=0.90)
    p.add_argument("--intervention-threshold", type=float, default=0.20)
    p.add_argument("--local-threshold", type=float, default=0.90)
    p.add_argument("--output", default="experiments/level6_9/formal")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.eval_batch_size < 2: raise ValueError("batch-roll requires eval-batch-size >= 2")
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
    all_drops = [value["query_drop"] for result in results for value in result["by_chunks"].values()]
    summary = {"runs": len(results), "causal_passes": sum(r["causal_passed"] for r in results),
               "causal_pass_rate": statistics.mean(r["causal_passed"] for r in results),
               "mean_query_drop": statistics.mean(all_drops), "minimum_query_drop": min(all_drops)}
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
