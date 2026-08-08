import argparse
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


DEFAULT_LRS = [1e-4, 5e-5, 2.5e-5, 1e-5]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def lr_name(lr):
    return f"lr_{lr:.1e}".replace(".", "p").replace("-", "m")


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


def run(lr, args, device, dtype, root):
    folder = root / lr_name(lr)
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

    # Keep the original model+probe parameter-group layout so Adam moments from
    # the stage-3 checkpoint can be restored exactly. Frozen probe parameters
    # receive no gradients and therefore do not update.
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()), lr=lr)
    # Restore Adam moments from the common stage-3 state while deliberately
    # overriding its learning rate for this controlled comparison.
    old_optimizer = checkpoint.get("optimizer")
    optimizer_state_restored = False
    if old_optimizer is not None:
        try:
            optimizer.load_state_dict(old_optimizer)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer_state_restored = True
        except ValueError:
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(probe.parameters()), lr=lr
            )

    eval_args = argparse.Namespace(
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        chunk_size=args.chunk_size,
    )
    set_seed(args.data_seed)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    baseline = evaluate(model, probe, eval_args, 16, device, dtype)
    history = [{"phase": "baseline", "step": 0, **baseline}]
    consecutive = 0
    first_gate_step = None
    best_query = baseline["query"]
    best_step = 0

    for step in range(1, args.train_steps + args.maintenance_steps + 1):
        phase = "train" if step <= args.train_steps else "maintenance"
        model.train()
        losses = train_step(model, probe, optimizer, args, device, dtype)
        if step % args.eval_every == 0:
            metric = evaluate(model, probe, eval_args, 16, device, dtype)
            row = {"phase": phase, "step": step, **losses, **metric}
            history.append(row)
            save(folder / "progress.json", history)
            if metric["query"] > best_query:
                best_query = metric["query"]
                best_step = step
            consecutive = consecutive + 1 if metric["query"] >= 0.95 else 0
            if consecutive >= 2 and first_gate_step is None:
                first_gate_step = step
            print(
                f"lr={lr:.1e} phase={phase} step={step} "
                f"query={metric['query']:.2%} local={metric['local']:.2%}",
                flush=True,
            )

    train_rows = [row for row in history if row.get("phase") == "train"]
    maintenance_rows = [row for row in history if row.get("phase") == "maintenance"]
    final_train = train_rows[-1]
    final_maintenance = evaluate(
        model, probe, eval_args, 16, device, dtype, args.final_eval_batches
    )
    tail = maintenance_rows[-min(5, len(maintenance_rows)) :]
    tail_min_query = min(row["query"] for row in tail)
    passed = (
        first_gate_step is not None
        and final_train["query"] >= 0.95
        and final_maintenance["query"] >= 0.95
        and tail_min_query >= 0.90
    )
    result = {
        "lr": lr,
        "data_seed": args.data_seed,
        "model_seed": args.model_seed,
        "optimizer_state_restored": optimizer_state_restored,
        "baseline": baseline,
        "best_query": best_query,
        "best_step": best_step,
        "first_gate_step": first_gate_step,
        "final_train": final_train,
        "final_maintenance": final_maintenance,
        "maintenance_tail_min_query": tail_min_query,
        "passed": passed,
        "history": history,
        "seconds": time.perf_counter() - started,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576,
    }
    save(result_path, result)
    torch.save({"model": model.state_dict(), "probe": probe.state_dict()}, folder / "final.pt")
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.5.1 16-chunk stability sweep")
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    parser.add_argument("--checkpoint", default="experiments/level6_5/deterministic/hard400_seed313/stage3.pt")
    parser.add_argument("--model-seed", type=int, default=313)
    parser.add_argument("--data-seed", type=int, default=71313)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--maintenance-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--output", default="experiments/level6_5_1/formal")
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
    for lr in args.lrs:
        results.append(run(lr, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()

    summary = [
        {
            "lr": result["lr"],
            "passed": result["passed"],
            "best_query": result["best_query"],
            "final_train_query": result["final_train"]["query"],
            "final_maintenance_query": result["final_maintenance"]["query"],
            "tail_min_query": result["maintenance_tail_min_query"],
            "seconds": result["seconds"],
        }
        for result in results
    ]
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
