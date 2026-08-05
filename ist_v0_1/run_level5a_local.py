import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from baseline_transformer import StandardTransformer
from long_context_test import set_seed
from marked_retrieval_level2 import make_batch
from model import InformationSpiralTransformer


VARIANTS = {
    "transformer": {"kind": "baseline", "diversity": 0.0, "fusion": False},
    "ist-a": {"kind": "ist", "diversity": 0.0, "fusion": False},
    "ist-b": {"kind": "ist", "diversity": 0.1, "fusion": False},
    "ist-c": {"kind": "ist", "diversity": 0.1, "fusion": True},
}


def build_model(config):
    if config["kind"] == "baseline":
        return StandardTransformer(19, 64, 3, 8, 512, 0.0, "rope")
    return InformationSpiralTransformer(19, 64, 3, 512, "rope", config["fusion"])


@torch.no_grad()
def evaluate(model, length, needle_range, batches, batch_size, device, amp_dtype):
    model.eval(); query_correct = local_correct = total = 0
    losses = []
    for _ in range(batches):
        tokens, targets, positions = make_batch(batch_size, length, needle_range, 16, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(tokens)[..., :16]
            query_loss = F.cross_entropy(logits[:, -1], targets)
        rows = torch.arange(batch_size, device=device)
        query_correct += (logits[:, -1].argmax(-1) == targets).sum().item()
        local_correct += (logits[rows, positions].argmax(-1) == targets).sum().item()
        total += batch_size; losses.append(query_loss.item())
    return {"query_accuracy": query_correct / total,
            "local_accuracy": local_correct / total,
            "query_loss": statistics.mean(losses), "samples": total}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(name, config, seed, args, device, amp_dtype, root):
    run_dir = root / f"{name}_seed{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "result.json"
    if final_path.exists() and not args.force:
        print(f"skip completed {run_dir}", flush=True)
        return json.loads(final_path.read_text(encoding="utf-8"))
    set_seed(seed); model = build_model(config).to(device)
    set_seed(seed + 10000); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    parameters = sum(p.numel() for p in model.parameters()); history = []; stages = []
    global_step = 0; started = time.perf_counter(); start_stage = 0
    stage_specs = [(128, 64, args.stage1_max_steps),
                   (256, 128, args.later_max_steps),
                   (512, 509, args.later_max_steps)]
    checkpoints = sorted(run_dir.glob("stage*.pt"))
    if checkpoints and not args.force:
        checkpoint = torch.load(checkpoints[-1], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        stages = checkpoint["stages"]; start_stage = len(stages)
        global_step = sum(stage["steps"] for stage in stages)
        progress_path = run_dir / "progress.json"
        if progress_path.exists(): history = json.loads(progress_path.read_text(encoding="utf-8"))
        print(f"resume {run_dir} after stage {start_stage}", flush=True)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for stage_index, (length, needle_range, max_steps) in enumerate(stage_specs[start_stage:], start_stage + 1):
        passed = False; metric = None; consecutive_passes = 0
        for stage_step in range(1, max_steps + 1):
            global_step += 1; model.train()
            tokens, targets, positions = make_batch(args.batch_size, length, needle_range, 16, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(tokens)[..., :16]; rows = torch.arange(args.batch_size, device=device)
                query_loss = F.cross_entropy(logits[:, -1], targets)
                local_loss = F.cross_entropy(logits[rows, positions], targets)
                loss = query_loss + 0.5 * local_loss
                if config["kind"] == "ist":
                    loss = loss + config["diversity"] * model.memory_diversity_loss()
            loss.backward(); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if stage_step == 1 or stage_step % args.eval_every == 0:
                metric = evaluate(model, length, needle_range, args.eval_batches,
                                  args.eval_batch_size, device, amp_dtype)
                record = {"global_step": global_step, "stage": stage_index,
                          "stage_step": stage_step, "length": length,
                          "query_accuracy": metric["query_accuracy"],
                          "local_accuracy": metric["local_accuracy"],
                          "query_loss": metric["query_loss"], "gradient_norm": float(grad_norm)}
                history.append(record); save_json(run_dir / "progress.json", history)
                print(f"{name} seed={seed} stage={stage_index} step={stage_step} "
                      f"query={metric['query_accuracy']:.2%} local={metric['local_accuracy']:.2%}", flush=True)
                if stage_step >= args.minimum_stage_steps and metric["query_accuracy"] >= args.gate_accuracy:
                    consecutive_passes += 1
                else:
                    consecutive_passes = 0
                if consecutive_passes >= args.consecutive_passes:
                    passed = True; break
        stages.append({"stage": stage_index, "length": length, "steps": stage_step,
                       "passed": passed, "validation": metric})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "variant": name, "seed": seed, "stages": stages}, run_dir / f"stage{stage_index}.pt")
        if not passed: break
    result = {"variant": name, "seed": seed, "parameters": parameters,
              "completed_all_stages": len(stages) == 3 and all(s["passed"] for s in stages),
              "seconds": time.perf_counter() - started,
              "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1048576 if device.type == "cuda" else None,
              "stages": stages, "history": history, "config": vars(args)}
    save_json(final_path, result); return result


def main():
    parser = argparse.ArgumentParser(description="RTX 5060 local Level 5A gated ablation")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[313, 42])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--minimum-stage-steps", type=int, default=500)
    parser.add_argument("--stage1-max-steps", type=int, default=1500)
    parser.add_argument("--later-max-steps", type=int, default=500)
    parser.add_argument("--gate-accuracy", type=float, default=0.90)
    parser.add_argument("--consecutive-passes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="experiments/level5a/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Install a CUDA-enabled PyTorch build.")
    device = torch.device("cuda"); amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU={torch.cuda.get_device_name(0)} AMP={amp_dtype} output={root}", flush=True)
    results = [run_one(name, VARIANTS[name], seed, args, device, amp_dtype, root)
               for name in args.variants for seed in args.seeds]
    summary = []
    for name in args.variants:
        selected = [r for r in results if r["variant"] == name]
        summary.append({"variant": name, "parameters": selected[0]["parameters"],
                        "success_rate": sum(r["completed_all_stages"] for r in selected) / len(selected),
                        "mean_seconds": statistics.mean(r["seconds"] for r in selected),
                        "mean_peak_memory_mb": statistics.mean(r["peak_memory_mb"] for r in selected)})
    save_json(root / "summary.json", {"summary": summary, "runs": results})
    print(json.dumps(summary, indent=2)); print(f"DONE: {root/'summary.json'}")


if __name__ == "__main__": main()
