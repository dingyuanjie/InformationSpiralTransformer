"""Frozen Memory 0.5.2: layer-aligned full-context teacher distillation."""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from experiment_utils import ROOT, atomic_json, atomic_torch, parameter_count, run_metadata
from pretrained_layer_memory_adapter import FrozenLayerInjectedIST
from pretrained_memory_adapter import load_qwen
from run_pretrained_base_smoke import MODEL_ID, candidate_ids
from run_pretrained_frozen_memory_0_4 import CHUNK, SEEDS, make_batch, paired_exact
from run_pretrained_layer_memory_0_5_1 import (
    CONDITIONS, causal_score, evaluate, summarize,
)


@torch.no_grad()
def full_context_layer_target(backbone, input_ids, layer_index):
    """Capture the full-context input to the same decoder layer used by IST."""
    captured = {}

    def hook(_module, args, kwargs):
        captured["query"] = args[0][:, -1].detach().clone()

    handle = backbone.model.layers[layer_index].register_forward_pre_hook(
        hook, with_kwargs=True
    )
    try:
        backbone.model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if "query" not in captured:
        raise RuntimeError("full-context teacher layer hook was not called")
    return captured["query"]


def checkpoint_payload(adapter, optimizer, history, step, best):
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
        backbone, args.injection_layer, args.heads,
        layer_matched_write=args.layer_matched_write,
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

    validation_seeds = [410000000 + seed * 10000 + i for i in range(args.validation_samples)]
    for step in range(start + 1, args.steps + 1):
        example_seeds = [
            420000000 + seed * 10000000 + step * args.batch + i
            for i in range(args.batch)
        ]
        ids, target = make_batch(tokenizer, example_seeds, "train", device)
        teacher = full_context_layer_target(backbone, ids, adapter.injection_layer)
        first, second = ids.split(CHUNK, dim=1)
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, state = adapter(first, None)
            normal_logits, _ = adapter(second, state, intervention="normal")
            student_pre = adapter.last_pre_injection
            student_post = adapter.last_post_injection
            if student_pre is None or student_post is None:
                raise RuntimeError("student layer injection hook was not called")
            normal_candidates = normal_logits[:, -1, labels.to(device)]
            task_loss = F.cross_entropy(normal_candidates, target)

            representation_cosine = (
                1 - F.cosine_similarity(student_post.float(), teacher.float(), dim=-1)
            ).mean()
            representation_mse = F.mse_loss(student_post.float(), teacher.float())
            teacher_delta = teacher.float() - student_pre.detach().float()
            student_delta = student_post.float() - student_pre.float()
            delta_cosine = (
                1 - F.cosine_similarity(student_delta, teacher_delta, dim=-1)
            ).mean()
            delta_mse = F.mse_loss(student_delta, teacher_delta)

            swapped_logits, _ = adapter(second, state, intervention="swap_fast")
            swapped_candidates = swapped_logits[:, -1, labels.to(device)]
            normal_logp = F.log_softmax(normal_candidates.float(), dim=-1)
            swapped_logp = F.log_softmax(swapped_candidates.float(), dim=-1)
            normal_target = normal_logp.gather(1, target[:, None]).squeeze(1)
            swapped_target = swapped_logp.gather(1, target[:, None]).squeeze(1)
            informative = target != torch.roll(target, 1, dims=0)
            if informative.any():
                binding_loss = F.relu(
                    args.binding_margin - (normal_target - swapped_target)
                )[informative].mean()
            else:
                binding_loss = normal_target.sum() * 0.0

            loss = (
                task_loss
                + args.representation_cosine_weight * representation_cosine
                + args.representation_mse_weight * representation_mse
                + args.delta_cosine_weight * delta_cosine
                + args.delta_mse_weight * delta_mse
                + args.binding_weight * binding_loss
            )
        loss.backward()
        raw_grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "representation_cosine_loss": float(representation_cosine.detach()),
                "representation_mse_loss": float(representation_mse.detach()),
                "delta_cosine_loss": float(delta_cosine.detach()),
                "delta_mse_loss": float(delta_mse.detach()),
                "binding_loss": float(binding_loss.detach()),
                "batch_accuracy": float((normal_candidates.argmax(-1) == target).float().mean()),
                "raw_trainable_grad": raw_grad,
                "injection_scale": float(torch.tanh(adapter.injection_scale.detach())),
                "injection_norm": adapter.last_injection_norm,
                "teacher_delta_norm": float(teacher_delta.detach().norm(dim=-1).mean()),
                "student_delta_norm": float(student_delta.detach().norm(dim=-1).mean()),
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
            atomic_torch(
                resume, checkpoint_payload(adapter, optimizer, history, step, best)
            )

    selected = torch.load(folder / "best.pt", map_location=device, weights_only=False)
    adapter.load_trainable_state_dict(selected["adapter"])
    heldout_seeds = [430000000 + seed * 10000 + i for i in range(args.heldout_samples)]
    adapter.eval()
    heldout = evaluate(
        backbone, adapter, tokenizer, labels, heldout_seeds,
        "held_out", device, args.eval_batch,
    )
    return adapter, {"seed": seed, "best": best, "history": history, "heldout": heldout}


def main(default_layer_matched_write=False,
         default_output="experiments/pretrained_base/layer_memory_0_5_2/formal",
         stage="Frozen Memory 0.5.2"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--layer-matched-write", action=argparse.BooleanOptionalAction,
        default=default_layer_matched_write,
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--heldout-samples", type=int, default=128)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--representation-cosine-weight", type=float, default=.5)
    parser.add_argument("--representation-mse-weight", type=float, default=.05)
    parser.add_argument("--delta-cosine-weight", type=float, default=1.0)
    parser.add_argument("--delta-mse-weight", type=float, default=.1)
    parser.add_argument("--binding-weight", type=float, default=.25)
    parser.add_argument("--binding-margin", type=float, default=1.0)
    parser.add_argument("--output", default=default_output)
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
    weights = {
        "answer_ce": 1.0,
        "layer_representation_cosine": args.representation_cosine_weight,
        "layer_representation_mse": args.representation_mse_weight,
        "missing_delta_cosine": args.delta_cosine_weight,
        "missing_delta_mse": args.delta_mse_weight,
        "swap_binding": args.binding_weight,
        "swap_margin": args.binding_margin,
    }
    protocol = {
        "stage": stage,
        "task": "layer-matched write/read with full-context teacher distillation" if args.layer_matched_write else "layer-aligned full-context teacher distillation",
        "model_id": args.model_id,
        "teacher": "frozen Qwen full 1024 layer-input query representation",
        "student": "frozen Qwen 2x512 with historical Fast Memory",
        "distance": 1024,
        "chunk_size": CHUNK,
        "seeds": args.seeds,
        "fresh_adapter_per_seed": True,
        "freeze_backbone": True,
        "injection_layer_requested": args.injection_layer,
        "memory_write_representation": "injection-layer input" if args.layer_matched_write else "final backbone hidden",
        "memory_read_representation": "injection-layer input",
        "layer_matched_write_read": args.layer_matched_write,
        "steps_per_seed": args.steps,
        "batch": args.batch,
        "unique_training_examples_per_seed": args.steps * args.batch,
        "lr": args.lr,
        "loss_weights": weights,
        "validation_samples": args.validation_samples,
        "heldout_samples_per_seed": args.heldout_samples,
        "checkpoint_selection": "max normal-minus-max(zero_fast, reset_memory, swap_fast)",
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
            value for run in runs
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
        "# Frozen Memory 0.5.2\n\nFull-context teacher and chunked student are aligned "
        "at the actual in-layer Memory injection boundary.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
