"""Frozen Memory 0.5: fixed-32 learnability gate for in-layer Fast Memory."""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn.functional as F

from experiment_utils import ROOT, atomic_json, atomic_torch, parameter_count, run_metadata
from pretrained_layer_memory_adapter import FrozenLayerInjectedIST
from pretrained_memory_adapter import load_qwen
from run_pretrained_base_smoke import MODEL_ID, candidate_ids
from run_pretrained_base_smoke_0_2 import FIXED, batch_of, fixed_data, grad_norm


CHUNK = 512


@torch.no_grad()
def score(backbone, adapter, labels, data, condition, batch=4):
    correctness = []
    hook_calls_before = adapter.hook_calls
    for start in range(0, len(data), batch):
        indices = list(range(start, min(start + batch, len(data))))
        ids, target = batch_of(data, indices)
        if condition == "full_context_base":
            logits = backbone(ids, use_cache=False).logits
        else:
            first, second = ids.split(CHUNK, dim=1)
            _, state = adapter(first, None, detach_state=True)
            if condition == "reset_memory":
                state = None
            intervention = condition if condition in {
                "zero_fast", "zero_memory", "roll_fast", "swap_fast", "reset_memory"
            } else "normal"
            logits, _ = adapter(second, state, intervention=intervention, detach_state=True)
        prediction = logits[:, -1, labels.to(ids.device)].argmax(-1)
        correctness.extend((prediction == target).int().cpu().tolist())
    return {
        "accuracy": sum(correctness) / len(correctness),
        "correctness": correctness,
        "hook_calls": adapter.hook_calls - hook_calls_before,
    }


@torch.no_grad()
def identity_check(backbone, adapter, data):
    ids = data[0][0][None, :CHUNK]
    base = backbone(ids, use_cache=False).logits[:, -1]
    adapted, _ = adapter(ids, None, detach_state=True)
    return float((base - adapted[:, -1]).abs().max())


def train(adapter, labels, data, args, root, device, dtype):
    parameters = adapter.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    history = []
    start = 0
    stable = 0
    resume = root / "training_resume.pt"
    if resume.exists() and not args.force:
        saved = torch.load(resume, map_location=device, weights_only=False)
        adapter.load_trainable_state_dict(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        start = int(saved["step"])
        stable = int(saved.get("stable", 0))
        print(f"resume step={start} stable={stable}", flush=True)

    for step in range(start + 1, args.steps + 1):
        rng = random.Random(205000000 + step)
        indices = [rng.randrange(FIXED) for _ in range(args.batch)]
        ids, target = batch_of(data, indices)
        first, second = ids.split(CHUNK, dim=1)
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, state = adapter(first, None)
            logits, _ = adapter(second, state)
            candidates = logits[:, -1, labels.to(device)]
            loss = F.cross_entropy(candidates, target)
        loss.backward()
        raw_grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        gate_grad = (
            float(adapter.injection_scale.grad.detach().abs())
            if adapter.injection_scale.grad is not None else 0.0
        )
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "batch_accuracy": float((candidates.argmax(-1) == target).float().mean()),
                "raw_trainable_grad": raw_grad,
                "raw_gate_grad": gate_grad,
                "injection_scale": float(torch.tanh(adapter.injection_scale.detach())),
                "injection_norm": adapter.last_injection_norm,
                "hook_calls": adapter.hook_calls,
                "fast_writer_grad": grad_norm(adapter.memory.fast_writer),
                "layer_read_grad": grad_norm(adapter.layer_read),
            }
            if step % args.evaluate_every == 0 or step == args.steps:
                adapter.eval()
                normal = score(adapter.backbone, adapter, labels, data, "normal", args.eval_batch)
                zero = score(adapter.backbone, adapter, labels, data, "zero_fast", args.eval_batch)
                reset = score(adapter.backbone, adapter, labels, data, "reset_memory", args.eval_batch)
                swap = score(adapter.backbone, adapter, labels, data, "swap_fast", args.eval_batch)
                gap = normal["accuracy"] - max(zero["accuracy"], reset["accuracy"])
                stable = stable + 1 if normal["accuracy"] >= .95 and gap >= .5 else 0
                row.update({
                    "fixed_normal": normal["accuracy"],
                    "fixed_zero_fast": zero["accuracy"],
                    "fixed_reset_memory": reset["accuracy"],
                    "fixed_swap_fast": swap["accuracy"],
                    "causal_gap": gap,
                    "stable_checks": stable,
                })
            history.append(row)
            print(json.dumps(row), flush=True)
            atomic_torch(resume, {
                "adapter": adapter.trainable_state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "step": step,
                "stable": stable,
            })
            if stable >= 2:
                print("LAYER_MEMORY_OVERFIT_GATE_PASS", flush=True)
                break
    return history, optimizer, history[-1]["step"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", default="experiments/pretrained_base/layer_memory_0_5/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.steps = 2
        args.batch = 1
        args.eval_batch = 4
        args.evaluate_every = 1
        if args.output.endswith("formal"):
            args.output = args.output[:-6] + "smoke"
    protocol = {
        "stage": "Frozen Memory 0.5",
        "task": "fixed-32 in-layer Fast-Memory learnability gate",
        "model_id": args.model_id,
        "distance": 1024,
        "chunk_size": CHUNK,
        "fixed_examples": FIXED,
        "freeze_backbone": True,
        "injection_layer_requested": args.injection_layer,
        "layers_after_injection_if_qwen_24": 4 if args.injection_layer == -4 else None,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "conditions": ["normal", "zero_fast", "reset_memory", "roll_fast", "swap_fast"],
        "gate": {"normal": .95, "causal_gap": .5, "consecutive_checks": 2},
        "not_a_generalization_result": True,
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
    torch.manual_seed(204000001)
    torch.cuda.manual_seed_all(204000001)
    tokenizer, backbone = load_qwen(args.model_id, dtype, device, args.local_files_only)
    labels = candidate_ids(tokenizer)
    adapter = FrozenLayerInjectedIST(backbone, args.injection_layer, args.heads).to(device=device, dtype=dtype)
    adapter.injection_scale.data = adapter.injection_scale.data.float()
    data = fixed_data(tokenizer, device, 203000000, "train")
    identity_delta = identity_check(backbone, adapter, data)
    history, optimizer, completed = train(adapter, labels, data, args, root, device, dtype)
    adapter.eval()
    conditions = ("full_context_base", "normal", "zero_fast", "reset_memory", "roll_fast", "swap_fast")
    fixed = {condition: score(backbone, adapter, labels, data, condition, args.eval_batch) for condition in conditions}
    heldout_data = fixed_data(tokenizer, device, 206000000, "held_out")
    heldout = {condition: score(backbone, adapter, labels, heldout_data, condition, args.eval_batch) for condition in conditions}
    gap = fixed["normal"]["accuracy"] - max(
        fixed["zero_fast"]["accuracy"], fixed["reset_memory"]["accuracy"]
    )
    passed = fixed["normal"]["accuracy"] >= .95 and gap >= .5
    protocol["resolved_injection_layer"] = adapter.injection_layer
    protocol["total_backbone_layers"] = adapter.total_layers
    protocol["resolved_revision"] = getattr(backbone.config, "_commit_hash", None)
    result = {
        "status": "complete",
        "overfit_causal_gate_passed": passed,
        "identity_no_history_max_logit_delta": identity_delta,
        "completed_steps": completed,
        "fixed_train": fixed,
        "heldout_diagnostic": heldout,
        "causal_gap": gap,
        "history": history,
        "protocol": protocol,
        "backbone_parameters": parameter_count(backbone),
        "trainable_parameters": sum(parameter.numel() for parameter in adapter.trainable_parameters()),
    }
    atomic_json(root / "config.json", protocol)
    atomic_json(root / "run_metadata.json", run_metadata(device, 204000001))
    atomic_torch(root / "memory_checkpoint.pt", {
        "adapter": adapter.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "revision": protocol["resolved_revision"],
    })
    atomic_json(root / "raw_results.json", result)
    atomic_json(root / "result.json", result)
    (root / "ANALYSIS.md").write_text(
        "# Frozen Memory 0.5\n\nSingle-upper-layer injection, fixed-set causal learnability gate. "
        "This stage is not a held-out generalization claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
