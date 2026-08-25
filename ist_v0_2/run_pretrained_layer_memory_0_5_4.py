"""Frozen Memory 0.5.4: Fast-Memory content-collapse tomography."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment_utils import ROOT, atomic_json, run_metadata
from pretrained_layer_memory_adapter import FrozenLayerInjectedIST
from pretrained_memory_adapter import load_qwen
from run_pretrained_base_smoke import MODEL_ID, candidate_ids, make_tokens
from run_pretrained_frozen_memory_0_4 import CHUNK, SEEDS


DEFAULT_SOURCE = ROOT / "experiments/pretrained_base/layer_memory_0_5_3/formal"


def off_diagonal_mean(matrix):
    count = matrix.numel() - matrix.size(0)
    return float((matrix.sum() - matrix.diagonal().sum()) / max(count, 1))


def effective_rank(matrix):
    centered = matrix.float() - matrix.float().mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))
    # Centering removes one sample degree of freedom.
    maximum = max(1, min(matrix.shape[0] - 1, matrix.shape[1]))
    return {
        "effective_rank": rank,
        "maximum_rank": maximum,
        "normalized_effective_rank": rank / maximum,
        "top1_energy_fraction": float(probabilities[0]) if probabilities.numel() else 0.0,
    }


def batch_rows(tokenizer, seeds, split, device):
    rows = [make_tokens(tokenizer, seed, split, 1024) for seed in seeds]
    ids = torch.stack([row[0] for row in rows]).to(device)
    targets = torch.tensor([row[1] for row in rows], device=device)
    fact_lengths = [
        len(tokenizer.encode(row[2]["fact"] + "\n", add_special_tokens=False))
        for row in rows
    ]
    return ids, targets, fact_lengths


@torch.no_grad()
def diagnose_seed(backbone, adapter, tokenizer, labels, seed, args, device):
    example_seeds = [440000000 + seed * 10000 + i for i in range(args.samples)]
    all_fast = []
    all_writer_entropy = []
    all_fact_mass = []
    all_fact_enrichment = []
    all_fact_top_hits = []
    all_reader_entropy = []
    all_reader_top1 = []
    all_normal_correct = []
    all_swap_correct = []
    all_prediction_changed = []
    all_swap_same_label = []
    all_symmetric_kl = []
    all_logit_l2 = []
    all_memory_swap_l2 = []

    labels = labels.to(device)
    for start in range(0, len(example_seeds), args.batch):
        seeds = example_seeds[start:start + args.batch]
        ids, target, fact_lengths = batch_rows(tokenizer, seeds, "held_out", device)
        first, second = ids.split(CHUNK, dim=1)
        _, state = adapter(first, None, detach_state=True)
        fast = state["fast"].detach()
        all_fast.append(fast.float().cpu())

        writer = adapter.memory.fast_writer.last_diagnostics
        compression = writer["compression_weights"].float()  # [B, slots, tokens]
        entropy = writer["attention_entropy"].float() / math.log(compression.size(-1))
        all_writer_entropy.extend(entropy.mean(-1).cpu().tolist())
        for index, fact_length in enumerate(fact_lengths):
            mass = compression[index, :, :fact_length].sum(-1)
            expected = fact_length / compression.size(-1)
            all_fact_mass.append(float(mass.mean()))
            all_fact_enrichment.append(float((mass / expected).mean()))
            top_token = compression[index].argmax(-1)
            all_fact_top_hits.append(float((top_token < fact_length).float().mean()))

        normal_logits, _ = adapter(second, state, intervention="normal", detach_state=True)
        pre_hidden = adapter.last_layer_input_sequence.detach()
        query = adapter.query_norm(pre_hidden)
        memory = adapter.memory_norm(fast.to(pre_hidden.dtype))
        _, read_weights = adapter.layer_read(
            query, memory, memory, need_weights=True, average_attn_weights=False
        )
        final_weights = read_weights[:, :, -1].float().mean(1)  # [B, slots]
        read_entropy = -(
            final_weights.clamp_min(1e-12) * final_weights.clamp_min(1e-12).log()
        ).sum(-1) / math.log(final_weights.size(-1))
        all_reader_entropy.extend(read_entropy.cpu().tolist())
        all_reader_top1.extend(final_weights.max(-1).values.cpu().tolist())

        swapped_logits, _ = adapter(second, state, intervention="swap_fast", detach_state=True)
        normal_candidates = normal_logits[:, -1, labels].float()
        swapped_candidates = swapped_logits[:, -1, labels].float()
        normal_prediction = normal_candidates.argmax(-1)
        swapped_prediction = swapped_candidates.argmax(-1)
        all_normal_correct.extend((normal_prediction == target).int().cpu().tolist())
        all_swap_correct.extend((swapped_prediction == target).int().cpu().tolist())
        all_prediction_changed.extend((normal_prediction != swapped_prediction).int().cpu().tolist())
        rolled_target = torch.roll(target, 1, dims=0)
        all_swap_same_label.extend((target == rolled_target).int().cpu().tolist())
        p = F.softmax(normal_candidates, dim=-1)
        q = F.softmax(swapped_candidates, dim=-1)
        log_p = F.log_softmax(normal_candidates, dim=-1)
        log_q = F.log_softmax(swapped_candidates, dim=-1)
        symmetric = .5 * (
            F.kl_div(log_p, q, reduction="none").sum(-1)
            + F.kl_div(log_q, p, reduction="none").sum(-1)
        )
        all_symmetric_kl.extend(symmetric.cpu().tolist())
        all_logit_l2.extend((normal_candidates - swapped_candidates).norm(dim=-1).cpu().tolist())
        swapped_fast = torch.roll(fast.float(), 1, dims=0)
        all_memory_swap_l2.extend(
            (fast.float() - swapped_fast).flatten(1).norm(dim=-1).cpu().tolist()
        )

    fast = torch.cat(all_fast, dim=0)
    flat = fast.flatten(1)
    normalized_flat = F.normalize(flat, dim=-1)
    example_cosine = normalized_flat @ normalized_flat.T
    normalized_slots = F.normalize(fast, dim=-1)
    slot_gram = normalized_slots @ normalized_slots.transpose(1, 2)
    slot_offdiag = (
        slot_gram.sum((1, 2)) - slot_gram.diagonal(dim1=1, dim2=2).sum(1)
    ) / (fast.size(1) * (fast.size(1) - 1))
    variance = flat.var(0, unbiased=False).mean()
    mean_square = flat.square().mean()
    rank = effective_rank(flat)

    def mean(values):
        return sum(values) / len(values)

    result = {
        "seed": seed,
        "checkpoint": str(args.source / f"seed{seed}" / "best.pt"),
        "samples": args.samples,
        "fast_geometry": {
            "cross_example_cosine_mean": off_diagonal_mean(example_cosine),
            "within_example_slot_cosine_mean": float(slot_offdiag.mean()),
            "between_example_variance": float(variance),
            "variance_to_mean_square_ratio": float(variance / mean_square.clamp_min(1e-12)),
            "mean_slot_norm": float(fast.norm(dim=-1).mean()),
            **rank,
        },
        "writer": {
            "normalized_attention_entropy": mean(all_writer_entropy),
            "fact_attention_mass": mean(all_fact_mass),
            "fact_attention_enrichment_over_uniform": mean(all_fact_enrichment),
            "slot_top_token_inside_fact_rate": mean(all_fact_top_hits),
        },
        "reader": {
            "normalized_query_slot_entropy": mean(all_reader_entropy),
            "query_top1_slot_weight": mean(all_reader_top1),
        },
        "swap_response": {
            "normal_accuracy": mean(all_normal_correct),
            "swap_accuracy": mean(all_swap_correct),
            "prediction_flip_rate": mean(all_prediction_changed),
            "same_answer_label_swap_rate": mean(all_swap_same_label),
            "candidate_symmetric_kl": mean(all_symmetric_kl),
            "candidate_logit_l2": mean(all_logit_l2),
            "fast_state_swap_l2": mean(all_memory_swap_l2),
        },
    }
    return result


def aggregate(runs):
    sections = ("fast_geometry", "writer", "reader", "swap_response")
    return {
        section: {
            key: sum(run[section][key] for run in runs) / len(runs)
            for key in runs[0][section]
        }
        for section in sections
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output", default="experiments/pretrained_base/layer_memory_0_5_4/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.seeds = [2026]
        args.samples = 4
        args.batch = 2
        if args.output.endswith("formal"):
            args.output = args.output[:-6] + "smoke"
    protocol = {
        "stage": "Frozen Memory 0.5.4",
        "task": "frozen Fast-Memory content-collapse tomography",
        "model_id": args.model_id,
        "source": str(args.source),
        "seeds": args.seeds,
        "heldout_samples_per_seed": args.samples,
        "batch": args.batch,
        "training": False,
        "measurements": [
            "cross-example slot geometry", "effective rank", "fact attention enrichment",
            "query-to-slot entropy", "swap logit response",
        ],
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    missing = [
        str(args.source / f"seed{seed}" / "best.pt")
        for seed in args.seeds
        if not (args.source / f"seed{seed}" / "best.pt").exists()
    ]
    if missing:
        raise FileNotFoundError("missing Level 0.5.3 checkpoints: " + ", ".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    tokenizer, backbone = load_qwen(args.model_id, dtype, device, args.local_files_only)
    labels = candidate_ids(tokenizer)
    runs = []
    for seed in args.seeds:
        adapter = FrozenLayerInjectedIST(
            backbone, injection_layer=-4, heads=8, layer_matched_write=True
        ).to(device=device, dtype=dtype)
        adapter.injection_scale.data = adapter.injection_scale.data.float()
        checkpoint = torch.load(
            args.source / f"seed{seed}" / "best.pt", map_location=device, weights_only=False
        )
        adapter.load_trainable_state_dict(checkpoint["adapter"])
        adapter.eval()
        row = diagnose_seed(backbone, adapter, tokenizer, labels, seed, args, device)
        runs.append(row)
        print(f"seed={seed} " + json.dumps(row), flush=True)
        del adapter
        torch.cuda.empty_cache()
    summary = aggregate(runs)
    diagnosis = {
        "high_cross_example_similarity": summary["fast_geometry"]["cross_example_cosine_mean"] > .9,
        "low_effective_rank": summary["fast_geometry"]["normalized_effective_rank"] < .25,
        "writer_not_fact_enriched": summary["writer"]["fact_attention_enrichment_over_uniform"] < 1.5,
        "reader_nearly_uniform": summary["reader"]["normalized_query_slot_entropy"] > .9,
        "swap_output_insensitive": summary["swap_response"]["prediction_flip_rate"] < .1,
    }
    protocol["resolved_revision"] = getattr(backbone.config, "_commit_hash", None)
    result = {
        "status": "complete",
        "summary": summary,
        "diagnosis_flags": diagnosis,
        "runs": runs,
        "protocol": protocol,
    }
    atomic_json(root / "config.json", protocol)
    atomic_json(root / "run_metadata.json", run_metadata(device, args.seeds))
    atomic_json(root / "raw_results.json", result)
    atomic_json(root / "result.json", result)
    lines = [
        "# Frozen Memory 0.5.4", "", "Frozen tomography of Level 0.5.3 best checkpoints.", "",
        "## Diagnosis flags", "",
    ] + [f"- {key}: {value}" for key, value in diagnosis.items()]
    (root / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
