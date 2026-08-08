import argparse
import copy
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import evaluate, forward_chunks, make_chunks


DEFAULT_DATA_SEEDS = [71313, 42042, 22026, 70007, 51234]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build(device, chunk_size):
    model = InformationSpiralTransformer(19, 64, 3, chunk_size, "rope", True).to(device)
    probe = nn.Linear(192, 16).to(device)
    return model, probe


def train_step(model, probe, optimizer, args, device, dtype):
    chunks, target, pos = make_chunks(args.batch_size, 16, args.chunk_size, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        last, first, probes, _ = forward_chunks(model, probe, chunks)
        rows = torch.arange(args.batch_size, device=device)
        query_loss = F.cross_entropy(last[:, -1, :16], target)
        local_loss = F.cross_entropy(first[rows, pos, :16], target)
        probe_loss = torch.stack([F.cross_entropy(item, target) for item in probes]).mean()
        loss = query_loss + 0.5 * local_loss + 0.1 * model.memory_diversity_loss()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": loss.detach().float().item(),
        "query_loss": query_loss.detach().float().item(),
        "local_loss": local_loss.detach().float().item(),
        "probe_loss_diagnostic": probe_loss.detach().float().item(),
    }


def independent_evaluate(model, probe, eval_args, device, dtype, seed):
    set_seed(seed)
    return evaluate(
        model,
        probe,
        eval_args,
        16,
        device,
        dtype,
        eval_args.final_eval_batches,
    )


def run(data_seed, args, device, dtype, root):
    folder = root / f"stream_seed{data_seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    set_seed(args.model_seed)
    model, probe = build(device, args.chunk_size)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    probe.load_state_dict(checkpoint["probe"])
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()), lr=args.initial_lr
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = args.initial_lr

    eval_args = argparse.Namespace(
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        final_eval_batches=args.final_eval_batches,
        chunk_size=args.chunk_size,
    )
    set_seed(data_seed)
    baseline = evaluate(model, probe, eval_args, 16, device, dtype)
    history = [{"phase": "baseline", "step": 0, "lr": args.initial_lr, **baseline}]
    best_query = baseline["query"]
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    switched = False
    switch_step = None
    switch_reason = None
    current_lr = args.initial_lr
    consecutive = 0
    first_gate_step = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    total_steps = args.train_steps + args.maintenance_steps
    for step in range(1, total_steps + 1):
        phase = "train" if step <= args.train_steps else "maintenance"
        lr_used = current_lr
        model.train()
        losses = train_step(model, probe, optimizer, args, device, dtype)
        if step % args.eval_every != 0:
            continue
        metric = evaluate(model, probe, eval_args, 16, device, dtype)
        if metric["query"] > best_query:
            best_query = metric["query"]
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
        consecutive = consecutive + 1 if metric["query"] >= 0.95 else 0
        if consecutive >= 2 and first_gate_step is None:
            first_gate_step = step

        if not switched:
            if metric["query"] >= args.gate:
                switch_reason = "gate_crossed"
            elif (
                step >= args.min_deterioration_step
                and best_query >= args.deterioration_floor
                and best_query - metric["query"] >= args.deterioration_drop
            ):
                switch_reason = "validation_deterioration"
            elif step >= args.latest_switch_step:
                switch_reason = "latest_switch"
            if switch_reason is not None:
                switched = True
                switch_step = step
                current_lr = args.stable_lr
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

        row = {
            "phase": phase,
            "step": step,
            "lr_used": lr_used,
            "next_lr": current_lr,
            "switched": switched,
            "switch_reason": switch_reason,
            **losses,
            **metric,
        }
        history.append(row)
        save(folder / "progress.json", history)
        print(
            f"seed={data_seed} step={step} lr={current_lr:.1e} "
            f"query={metric['query']:.2%} best={best_query:.2%}",
            flush=True,
        )

    last_state = copy.deepcopy(model.state_dict())
    heldout_seed = data_seed + args.heldout_seed_offset
    last_final = independent_evaluate(model, probe, eval_args, device, dtype, heldout_seed)
    model.load_state_dict(best_state)
    best_final = independent_evaluate(model, probe, eval_args, device, dtype, heldout_seed)
    model.load_state_dict(last_state)
    tail_rows = [row for row in history if row.get("phase") == "maintenance"][-5:]
    result = {
        "data_seed": data_seed,
        "baseline": baseline,
        "switch_step": switch_step,
        "switch_reason": switch_reason,
        "first_gate_step": first_gate_step,
        "best_validation_query": best_query,
        "best_step": best_step,
        "last_final": last_final,
        "best_checkpoint_final": best_final,
        "last_passed": last_final["query"] >= 0.95,
        "selected_passed": best_final["query"] >= 0.95,
        "maintenance_tail_min_query": min(row["query"] for row in tail_rows),
        "history": history,
        "seconds": time.perf_counter() - started,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576,
    }
    save(result_path, result)
    torch.save(
        {"model": best_state, "probe": probe.state_dict(), "best_step": best_step},
        folder / "best.pt",
    )
    torch.save(
        {"model": last_state, "probe": probe.state_dict(), "step": total_steps},
        folder / "last.pt",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.5.3 adaptive 16-chunk stabilization")
    parser.add_argument("--data-seeds", nargs="+", type=int, default=DEFAULT_DATA_SEEDS)
    parser.add_argument("--checkpoint", default="experiments/level6_5/deterministic/hard400_seed313/stage3.pt")
    parser.add_argument("--model-seed", type=int, default=313)
    parser.add_argument("--initial-lr", type=float, default=5e-5)
    parser.add_argument("--stable-lr", type=float, default=1e-5)
    parser.add_argument("--gate", type=float, default=0.95)
    parser.add_argument("--min-deterioration-step", type=int, default=300)
    parser.add_argument("--deterioration-floor", type=float, default=0.85)
    parser.add_argument("--deterioration-drop", type=float, default=0.10)
    parser.add_argument("--latest-switch-step", type=int, default=600)
    parser.add_argument("--heldout-seed-offset", type=int, default=900000)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--maintenance-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--output", default="experiments/level6_5_3/formal")
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
    for data_seed in args.data_seeds:
        results.append(run(data_seed, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()

    summary = {
        "runs": len(results),
        "last_successes": sum(result["last_passed"] for result in results),
        "selected_successes": sum(result["selected_passed"] for result in results),
        "mean_last_query": statistics.mean(result["last_final"]["query"] for result in results),
        "worst_last_query": min(result["last_final"]["query"] for result in results),
        "mean_selected_query": statistics.mean(
            result["best_checkpoint_final"]["query"] for result in results
        ),
        "worst_selected_query": min(
            result["best_checkpoint_final"]["query"] for result in results
        ),
        "mean_tail_min_query": statistics.mean(
            result["maintenance_tail_min_query"] for result in results
        ),
    }
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
