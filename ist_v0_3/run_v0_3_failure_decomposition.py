"""Milestone 2.2.1: decompose strict OOD failure into Writer/Reader/Decoder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from config import SourceTokenMemoryConfig
from pretrained_token_memory_adapter import FrozenTokenMemoryIST
from run_v0_3_coverage_gate import DEFAULT_MODEL, Tee, load_model
from run_v0_3_retrieval_gate import left_pad
from run_v0_3_strict_generalization import choose_answers, strict_stream


def signature_mask(state, row, signature):
    mask = torch.zeros_like(state["valid"][row])
    for position, token_id in signature:
        mask |= (state["positions"][row] == position) & (state["token_ids"][row] == token_id)
    return mask & state["valid"][row]


def oracle_state(state, signatures):
    result = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }
    for row, signature in enumerate(signatures):
        result["valid"][row] = signature_mask(state, row, signature)
    return result


@torch.no_grad()
def evaluate(args):
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    answers, answer_token_ids = choose_answers(tokenizer)
    config = SourceTokenMemoryConfig(
        capacity=args.capacity, writes_per_chunk=args.writes_per_chunk,
        reads_per_query=args.reads_per_query, heads=args.heads,
        injection_layer=args.injection_layer,
    )
    model = FrozenTokenMemoryIST(backbone, config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["adapter"], strict=False)
    candidates = torch.tensor(answer_token_ids, device=device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = []
    for chunks in args.chunks:
        counts = {key: 0.0 for key in (
            "writer_available", "writer_recall", "reader_hit", "reader_hit_when_available",
            "normal_accuracy", "oracle_accuracy", "oracle_logit_gain", "distractor_available",
        )}
        available_examples = examples = 0
        for start in range(0, args.samples, 2):
            if start + 1 >= args.samples:
                break
            batch = [
                strict_stream(tokenizer, args.chunk_size, chunks,
                              args.seed + chunks * 1000 + start + offset,
                              answers[(start + offset) % len(answers)])
                for offset in range(2)
            ]
            streams = torch.stack([item[0] for item in batch]).to(device)
            queries = left_pad([item[1] for item in batch], pad).to(device)
            fact_signatures, distractor_signatures = [], []
            for row, item in enumerate(batch):
                fact_signatures.append({(position, int(streams[row, position])) for position in item[2]})
                distractor_signatures.append({(position, int(streams[row, position])) for position in item[3]})
            state = None
            for chunk_id in range(chunks):
                begin = chunk_id * args.chunk_size
                _, state = model(streams[:, begin:begin + args.chunk_size], state,
                                 chunk_id, begin, "normal", True)
            target_masks = [signature_mask(state, row, signature) for row, signature in enumerate(fact_signatures)]
            distractor_masks = [signature_mask(state, row, signature) for row, signature in enumerate(distractor_signatures)]
            normal_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, "normal", True)
            normal_provenance = model.last_provenance
            zero_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, "zero", True)
            filtered = oracle_state(state, fact_signatures)
            oracle_logits, _ = model(queries, filtered, chunks, chunks * args.chunk_size, "normal", True)
            for row, item in enumerate(batch):
                _, _, fact_positions, _, metadata = item
                available = bool(target_masks[row].any())
                recall = float(target_masks[row].sum()) / max(1, len(fact_positions))
                counts["writer_available"] += available
                counts["writer_recall"] += recall
                counts["distractor_available"] += bool(distractor_masks[row].any())
                pairs = set(zip(
                    normal_provenance["positions"][row, -1].cpu().tolist(),
                    normal_provenance["token_ids"][row, -1].cpu().tolist(),
                ))
                hit = bool(pairs.intersection(fact_signatures[row]))
                counts["reader_hit"] += hit
                if available:
                    available_examples += 1
                    counts["reader_hit_when_available"] += hit
                target_index = answers.index(metadata["answer"])
                counts["normal_accuracy"] += int(
                    normal_logits[row, -1, candidates].argmax().item() == target_index
                )
                counts["oracle_accuracy"] += int(
                    oracle_logits[row, -1, candidates].argmax().item() == target_index
                )
                target_id = candidates[target_index]
                counts["oracle_logit_gain"] += float(
                    oracle_logits[row, -1, target_id] - zero_logits[row, -1, target_id]
                )
                examples += 1
        result = {
            "chunks": chunks, "examples": examples,
            "writer_fact_availability": counts["writer_available"] / examples,
            "writer_fact_token_recall": counts["writer_recall"] / examples,
            "distractor_availability": counts["distractor_available"] / examples,
            "reader_fact_hit_rate": counts["reader_hit"] / examples,
            "reader_hit_given_writer_available": counts["reader_hit_when_available"] / max(1, available_examples),
            "normal_answer_accuracy": counts["normal_accuracy"] / examples,
            "oracle_answer_accuracy": counts["oracle_accuracy"] / examples,
            "oracle_minus_zero_target_logit": counts["oracle_logit_gain"] / examples,
        }
        rows.append(result); print(json.dumps(result, ensure_ascii=False), flush=True)
    return {"status": "complete", "answers": answers, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/reader_alignment/reader_step_000400.pt"))
    parser.add_argument("--chunks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--writes-per-chunk", type=int, default=8)
    parser.add_argument("--reads-per-query", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--seed", type=int, default=3303)
    parser.add_argument("--output", type=Path, default=Path("experiments/failure_decomposition/results.json"))
    parser.add_argument("--log", type=Path, default=Path("experiments/failure_decomposition/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2)); return 0
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": protocol, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("v0_3_FAILURE_DECOMPOSITION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
