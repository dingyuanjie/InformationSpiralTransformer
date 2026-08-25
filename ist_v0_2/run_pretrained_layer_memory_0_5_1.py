"""Frozen Memory 0.5.1: unique-stream generalization with binding controls."""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from experiment_utils import ROOT, atomic_json, atomic_torch, parameter_count, run_metadata
from pretrained_layer_memory_adapter import FrozenLayerInjectedIST
from pretrained_memory_adapter import load_qwen
from run_pretrained_base_smoke import MODEL_ID, candidate_ids
from run_pretrained_frozen_memory_0_4 import CHUNK, SEEDS, make_batch, paired_exact, wilson


CONDITIONS = ("full_context_base", "normal", "zero_fast", "reset_memory", "swap_fast")


@torch.no_grad()
def evaluate(backbone, adapter, tokenizer, labels, seeds, split, device, batch):
    correctness = {condition: [] for condition in CONDITIONS}
    targets = []
    for start in range(0, len(seeds), batch):
        batch_seeds = seeds[start:start + batch]
        ids, target = make_batch(tokenizer, batch_seeds, split, device)
        targets.extend(target.cpu().tolist())
        full_logits = backbone(ids, use_cache=False).logits[:, -1, labels.to(device)]
        correctness["full_context_base"].extend(
            (full_logits.argmax(-1) == target).int().cpu().tolist()
        )
        first, second = ids.split(CHUNK, dim=1)
        _, state = adapter(first, None, detach_state=True)
        for condition in ("normal", "zero_fast", "reset_memory", "swap_fast"):
            historical = None if condition == "reset_memory" else state
            logits, _ = adapter(
                second, historical, intervention=condition, detach_state=True
            )
            prediction = logits[:, -1, labels.to(device)].argmax(-1)
            correctness[condition].extend((prediction == target).int().cpu().tolist())
    adapter.clear_intervention()
    return {
        "split": split,
        "samples": len(seeds),
        "correctness": correctness,
        "accuracy": {
            condition: sum(values) / len(values)
            for condition, values in correctness.items()
        },
        "targets": targets,
    }


def causal_score(accuracy):
    controls = (accuracy["zero_fast"], accuracy["reset_memory"], accuracy["swap_fast"])
    return accuracy["normal"] - max(controls)


def save_payload(adapter, optimizer, history, step, best):
    return {
        "adapter": adapter.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "step": step,
        "best": best,
    }


def train_seed(backbone, tokenizer, labels, seed, args, root, device, dtype):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    adapter = FrozenLayerInjectedIST(
        backbone, args.injection_layer, args.heads
    ).to(device=device, dtype=dtype)
    adapter.injection_scale.data = adapter.injection_scale.data.float()
    parameters = adapter.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    resume = folder / "resume.pt"
    history = []
    start = 0
    best = {"score": -1e9, "normal": -1.0, "step": 0}
    if resume.exists() and not args.force:
        saved = torch.load(resume, map_location=device, weights_only=False)
        adapter.load_trainable_state_dict(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        start = int(saved["step"])
        best = saved["best"]
        print(f"resume seed={seed} step={start}", flush=True)

    validation_seeds = [310000000 + seed * 10000 + i for i in range(args.validation_samples)]
    for step in range(start + 1, args.steps + 1):
        example_seeds = [
            320000000 + seed * 10000000 + step * args.batch + i
            for i in range(args.batch)
        ]
        ids, target = make_batch(tokenizer, example_seeds, "train", device)
        first, second = ids.split(CHUNK, dim=1)
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, state = adapter(first, None)
            normal_logits, _ = adapter(second, state, intervention="normal")
            swapped_logits, _ = adapter(second, state, intervention="swap_fast")
            normal_candidates = normal_logits[:, -1, labels.to(device)]
            swapped_candidates = swapped_logits[:, -1, labels.to(device)]
            task_loss = F.cross_entropy(normal_candidates, target)
            normal_logp = F.log_softmax(normal_candidates.float(), dim=-1)
            swapped_logp = F.log_softmax(swapped_candidates.float(), dim=-1)
            target_column = target[:, None]
            normal_target = normal_logp.gather(1, target_column).squeeze(1)
            swapped_target = swapped_logp.gather(1, target_column).squeeze(1)
            # Same-label swaps cannot be distinguished by this four-way task.
            informative = target != torch.roll(target, 1, dims=0)
            if informative.any():
                binding_loss = F.relu(
                    args.binding_margin - (normal_target - swapped_target)
                )[informative].mean()
            else:
                binding_loss = normal_target.sum() * 0.0
            loss = task_loss + args.binding_weight * binding_loss
        loss.backward()
        raw_grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "binding_loss": float(binding_loss.detach()),
                "batch_accuracy": float((normal_candidates.argmax(-1) == target).float().mean()),
                "raw_trainable_grad": raw_grad,
                "injection_scale": float(torch.tanh(adapter.injection_scale.detach())),
                "injection_norm": adapter.last_injection_norm,
                "informative_swap_rate": float(informative.float().mean()),
            }
            if step % args.validate_every == 0 or step == args.steps:
                adapter.eval()
                validation = evaluate(
                    backbone, adapter, tokenizer, labels, validation_seeds,
                    "validation", device, args.eval_batch,
                )
                score = causal_score(validation["accuracy"])
                row["validation"] = validation["accuracy"]
                row["validation_causal_score"] = score
                if score > best["score"] or (
                    score == best["score"]
                    and validation["accuracy"]["normal"] > best["normal"]
                ):
                    best = {
                        "score": score,
                        "normal": validation["accuracy"]["normal"],
                        "step": step,
                    }
                    atomic_torch(folder / "best.pt", {
                        "adapter": adapter.trainable_state_dict(),
                        "step": step,
                        "validation": validation,
                    })
            history.append(row)
            print(f"seed={seed} " + json.dumps(row), flush=True)
            atomic_torch(resume, save_payload(adapter, optimizer, history, step, best))

    selected = torch.load(folder / "best.pt", map_location=device, weights_only=False)
    adapter.load_trainable_state_dict(selected["adapter"])
    heldout_seeds = [330000000 + seed * 10000 + i for i in range(args.heldout_samples)]
    adapter.eval()
    heldout = evaluate(
        backbone, adapter, tokenizer, labels, heldout_seeds,
        "held_out", device, args.eval_batch,
    )
    return adapter, {"seed": seed, "best": best, "history": history, "heldout": heldout}


def summarize(correctness):
    return {
        condition: {
            "accuracy": sum(values) / len(values),
            "correct": sum(values),
            "samples": len(values),
            "wilson95": wilson(sum(values), len(values)),
        }
        for condition, values in correctness.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--heldout-samples", type=int, default=128)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--binding-weight", type=float, default=0.5)
    parser.add_argument("--binding-margin", type=float, default=1.0)
    parser.add_argument("--output", default="experiments/pretrained_base/layer_memory_0_5_1/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.seeds = [2026]
        args.steps = 2
        args.batch = 2
        args.validation_samples = 4
        args.heldout_samples = 4
        args.eval_batch = 2
        args.validate_every = 1
        if args.output.endswith("formal"):
            args.output = args.output[:-6] + "smoke"
    protocol = {
        "stage": "Frozen Memory 0.5.1",
        "task": "unique-stream in-layer Fast-Memory generalization",
        "model_id": args.model_id,
        "distance": 1024,
        "chunk_size": CHUNK,
        "seeds": args.seeds,
        "fresh_adapter_per_seed": True,
        "freeze_backbone": True,
        "injection_layer_requested": args.injection_layer,
        "steps_per_seed": args.steps,
        "batch": args.batch,
        "unique_training_examples_per_seed": args.steps * args.batch,
        "lr": args.lr,
        "binding_weight": args.binding_weight,
        "binding_margin": args.binding_margin,
        "validation_samples": args.validation_samples,
        "heldout_samples_per_seed": args.heldout_samples,
        "checkpoint_selection": "max normal-minus-max(zero_fast, reset_memory, swap_fast)",
        "primary_controls": ["zero_fast", "reset_memory", "swap_fast"],
        "chance": .25,
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    tokenizer, backbone = load_qwen(args.model_id, dtype, device, args.local_files_only)
    labels = candidate_ids(tokenizer)
    runs = []
    resolved_layer = None
    for seed in args.seeds:
        adapter, run = train_seed(
            backbone, tokenizer, labels, seed, args, root, device, dtype
        )
        resolved_layer = adapter.injection_layer
        runs.append(run)
        atomic_json(root / "runs.partial.json", runs)
        del adapter
        torch.cuda.empty_cache()

    aggregate = {
        condition: [
            value
            for run in runs
            for value in run["heldout"]["correctness"][condition]
        ]
        for condition in CONDITIONS
    }
    summary = summarize(aggregate)
    paired = {
        condition: paired_exact(aggregate["normal"], aggregate[condition])
        for condition in ("zero_fast", "reset_memory", "swap_fast")
    }
    passed = (
        summary["normal"]["wilson95"][0] > .25
        and all(test["difference"] > 0 and test["mcnemar_exact_p"] < .05 for test in paired.values())
    )
    protocol["resolved_injection_layer"] = resolved_layer
    protocol["total_backbone_layers"] = len(backbone.model.layers)
    protocol["resolved_revision"] = getattr(backbone.config, "_commit_hash", None)
    result = {
        "status": "complete",
        "generalization_gate_passed": passed,
        "summary": summary,
        "paired_normal_vs_controls": paired,
        "runs": runs,
        "protocol": protocol,
        "backbone_parameters": parameter_count(backbone),
    }
    atomic_json(root / "config.json", protocol)
    atomic_json(root / "run_metadata.json", run_metadata(device, args.seeds))
    atomic_json(root / "raw_results.json", result)
    atomic_json(root / "result.json", result)
    (root / "ANALYSIS.md").write_text(
        "# Frozen Memory 0.5.1\n\nUnique-stream layer-injected Fast Memory with paired "
        "zero, reset, and cross-example swap controls.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

