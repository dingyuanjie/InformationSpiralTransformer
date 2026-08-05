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


DEFAULT_PROFILES = [
    "zero",
    "hard50",
    "hard100",
    "hard200",
    "hard400",
    "hard800",
    "anneal200",
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build(device, chunk_size):
    model = InformationSpiralTransformer(19, 64, 3, chunk_size, "rope", True).to(device)
    probe = nn.Linear(192, 16).to(device)
    return model, probe


def profile_weight(profile, step):
    if profile == "zero":
        return 0.0
    if profile.startswith("hard"):
        duration = int(profile[4:])
        return 0.5 if step <= duration else 0.0
    if profile.startswith("anneal"):
        duration = int(profile[6:])
        if step > duration:
            return 0.0
        # Same integrated probe weight as hard{duration // 2}, but withdrawn smoothly.
        return 0.5 * (duration - step + 0.5) / duration
    raise ValueError(f"unknown profile: {profile}")


def scaffold_end(profile):
    if profile == "zero":
        return 0
    return int(profile[4:] if profile.startswith("hard") else profile[6:])


def train_step(model, probe, optimizer, batch, count, size, device, dtype, weight):
    chunks, target, pos = make_chunks(batch, count, size, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        last, first, probes, _ = forward_chunks(model, probe, chunks)
        rows = torch.arange(batch, device=device)
        query_loss = F.cross_entropy(last[:, -1, :16], target)
        local_loss = F.cross_entropy(first[rows, pos, :16], target)
        probe_loss = torch.stack([F.cross_entropy(item, target) for item in probes]).mean()
        loss = query_loss + 0.5 * local_loss + weight * probe_loss + 0.1 * model.memory_diversity_loss()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), 1.0)
    optimizer.step()
    return {
        "loss": loss.detach().float().item(),
        "query_loss": query_loss.detach().float().item(),
        "local_loss": local_loss.detach().float().item(),
        "probe_loss": probe_loss.detach().float().item(),
    }


def behavior_passed(metric):
    # The target appears only in chunk 1 and the query only in the final chunk,
    # so query accuracy is the direct task-level test of persistent memory.
    return metric["query"] >= 0.95


def probe_diagnostic_passed(metric):
    # A briefly trained and then frozen linear probe can underfit a useful state.
    # Keep this as a representation diagnostic, not a formation gate.
    return metric["probe_min"] >= 0.90


def run(profile, seed, args, device, dtype, root):
    folder = root / f"{profile}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    final_path = folder / "result.json"
    if final_path.exists() and not args.force:
        return json.loads(final_path.read_text(encoding="utf-8"))

    set_seed(seed)
    model, probe = build(device, args.chunk_size)
    set_seed(seed + 60000)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()), lr=1e-3)
    end = scaffold_end(profile)
    history = []
    stages = []
    weighted_probe_steps = 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    specs = [
        (2, args.stage1_steps, 1e-3, 8),
        (4, args.later_steps, 1e-3, 4),
        (8, args.later_steps, 2.5e-4, 4),
        (16, args.later_steps, 1e-4, 2),
    ]
    for stage, (count, steps, lr, batch) in enumerate(specs, 1):
        for group in optimizer.param_groups:
            group["lr"] = lr
        consecutive = 0
        metric = None
        for step in range(1, steps + 1):
            weight = profile_weight(profile, step) if stage == 1 else 0.0
            weighted_probe_steps += weight
            for parameter in probe.parameters():
                parameter.requires_grad_(weight > 0)
            model.train()
            probe.train(weight > 0)
            losses = train_step(model, probe, optimizer, batch, count, args.chunk_size, device, dtype, weight)

            should_eval = step == 1 or step % args.eval_every == 0 or (stage == 1 and step == end)
            if should_eval:
                metric = evaluate(model, probe, args, count, device, dtype)
                row = {
                    "stage": stage,
                    "chunks": count,
                    "step": step,
                    "probe_weight": weight,
                    **losses,
                    **metric,
                }
                history.append(row)
                save(folder / "progress.json", history)
                print(
                    f"profile={profile} seed={seed} chunks={count} step={step} "
                    f"weight={weight:.4f} query={metric['query']:.2%} probe_min={metric['probe_min']:.2%}",
                    flush=True,
                )
                # Formation is counted only after the scaffold is fully removed.
                scaffold_removed = stage > 1 or step > end
                consecutive = consecutive + 1 if scaffold_removed and behavior_passed(metric) else 0
                if consecutive >= 2:
                    break

        stage_result = {
            "chunks": count,
            "steps": step,
            "passed": consecutive >= 2,
            "validation": metric,
        }
        stages.append(stage_result)
        torch.save(
            {
                "model": model.state_dict(),
                "probe": probe.state_dict(),
                "optimizer": optimizer.state_dict(),
                "profile": profile,
                "seed": seed,
                "stages": stages,
                "history": history,
            },
            folder / f"stage{stage}.pt",
        )
        if consecutive < 2:
            break

    maintenance = None
    curriculum_passed = len(stages) == 4 and all(item["passed"] for item in stages)
    if curriculum_passed:
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
        for group in optimizer.param_groups:
            group["lr"] = 1e-4
        for step in range(1, args.maintenance_steps + 1):
            model.train()
            train_step(model, probe, optimizer, 2, 16, args.chunk_size, device, dtype, 0.0)
            if step % args.eval_every == 0:
                metric = evaluate(model, probe, args, 16, device, dtype)
                history.append({"stage": 5, "phase": "maintenance", "step": step, **metric})
                save(folder / "progress.json", history)
        maintenance = evaluate(model, probe, args, 16, device, dtype, args.final_eval_batches)

    result = {
        "profile": profile,
        "seed": seed,
        "scaffold_steps": end,
        "weighted_probe_steps": weighted_probe_steps,
        "curriculum_passed": curriculum_passed,
        "maintenance": maintenance,
        "passed": curriculum_passed and behavior_passed(maintenance),
        "probe_diagnostic_passed": (
            curriculum_passed and probe_diagnostic_passed(maintenance)
        ),
        "stages": stages,
        "history": history,
        "seconds": time.perf_counter() - started,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576,
    }
    save(final_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.5 minimum memory-scaffold search")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[313])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--stage1-steps", type=int, default=3000)
    parser.add_argument("--later-steps", type=int, default=1000)
    parser.add_argument("--maintenance-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--final-eval-batches", type=int, default=50)
    parser.add_argument("--output", default="experiments/level6_5/deterministic")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for profile in args.profiles:
        profile_weight(profile, 1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not args.allow_nondeterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for profile in args.profiles:
        for seed in args.seeds:
            results.append(run(profile, seed, args, device, dtype, root))
            save(root / "runs.partial.json", results)
            torch.cuda.empty_cache()

    summaries = []
    for profile in args.profiles:
        selected = [item for item in results if item["profile"] == profile]
        summaries.append(
            {
                "profile": profile,
                "successes": sum(item["passed"] for item in selected),
                "runs": len(selected),
                "success_rate": statistics.mean(item["passed"] for item in selected),
                "mean_seconds": statistics.mean(item["seconds"] for item in selected),
                "weighted_probe_steps": selected[0]["weighted_probe_steps"],
            }
        )
    save(root / "summary.json", {"protocol": vars(args), "summary": summaries, "runs": results})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
