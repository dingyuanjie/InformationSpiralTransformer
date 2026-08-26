"""Milestone 2: diagnose query retrieval and causal Memory use."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from config import SourceTokenMemoryConfig
from pretrained_token_memory_adapter import FrozenTokenMemoryIST
from run_v0_3_coverage_gate import DEFAULT_MODEL, Tee, load_model, make_stream


ANSWERS = ["Kestrel", "Amber", "Nimbus", "Quartz", "Cobalt"]


def left_pad(rows, pad_id):
    width = max(row.numel() for row in rows)
    result = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for index, row in enumerate(rows):
        result[index, -row.numel():] = row
    return result


@torch.no_grad()
def evaluate(args):
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    config = SourceTokenMemoryConfig(
        capacity=args.capacity, writes_per_chunk=args.writes_per_chunk,
        reads_per_query=args.reads_per_query, heads=args.heads,
        injection_layer=args.injection_layer,
    )
    model = FrozenTokenMemoryIST(backbone, config).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(payload["adapter"], strict=False)
        unexpected = [key for key in unexpected if not key.startswith("backbone.")]
        if unexpected:
            raise RuntimeError(f"unexpected adapter keys: {unexpected}")
        print(f"loaded_reader_checkpoint={args.checkpoint}", flush=True)
    candidate_ids = torch.tensor(
        [tokenizer.encode(answer, add_special_tokens=False)[0] for answer in ANSWERS],
        device=device,
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    rows = []
    for chunk_count in args.chunks:
        normal_hits = swap_hits = examples = correct = 0
        normal_mass = swap_mass = causal_margin = 0.0
        for start_sample in range(0, args.samples, 2):
            batch_size = min(2, args.samples - start_sample)
            if batch_size < 2:
                break
            streams, facts, metadata = [], [], []
            for offset in range(batch_size):
                stream, fact, meta = make_stream(
                    tokenizer, args.chunk_size, chunk_count,
                    args.seed + start_sample + offset,
                )
                streams.append(stream)
                facts.append(fact)
                metadata.append(meta)
            stream_batch = torch.stack(streams).to(device)
            state = None
            for chunk_id in range(chunk_count):
                begin = chunk_id * args.chunk_size
                piece = stream_batch[:, begin:begin + args.chunk_size]
                _, state = model(piece, state, chunk_id, begin, detach_state=True)
            queries = [
                torch.tensor(tokenizer.encode(
                    f"Question: what is the access code for {meta['entity']}? Answer:",
                    add_special_tokens=False,
                ), dtype=torch.long)
                for meta in metadata
            ]
            query_batch = left_pad(queries, pad_id).to(device)
            normal_logits, _ = model(query_batch, state, chunk_count,
                                     chunk_count * args.chunk_size, "normal", True)
            normal_provenance = model.last_provenance
            zero_logits, _ = model(query_batch, state, chunk_count,
                                   chunk_count * args.chunk_size, "zero", True)
            swap_logits, _ = model(query_batch, state, chunk_count,
                                   chunk_count * args.chunk_size, "swap", True)
            swap_provenance = model.last_provenance
            for row, meta in enumerate(metadata):
                positions = normal_provenance["positions"][row, -1].cpu().tolist()
                weights = normal_provenance["weights"][row, -1].cpu().tolist()
                swapped_positions = swap_provenance["positions"][row, -1].cpu().tolist()
                swapped_weights = swap_provenance["weights"][row, -1].cpu().tolist()
                fact = facts[row]
                normal_hits += bool(fact.intersection(positions))
                swap_hits += bool(fact.intersection(swapped_positions))
                normal_mass += sum(weight for position, weight in zip(positions, weights) if position in fact)
                swap_mass += sum(weight for position, weight in zip(swapped_positions, swapped_weights) if position in fact)
                answer_index = ANSWERS.index(meta["answer"])
                scores = normal_logits[row, -1, candidate_ids].float()
                correct += int(scores.argmax().item() == answer_index)
                target = candidate_ids[answer_index]
                causal_margin += float(normal_logits[row, -1, target] - zero_logits[row, -1, target])
                examples += 1
        row = {
            "chunks": chunk_count,
            "examples": examples,
            "normal_fact_hit_rate": normal_hits / max(1, examples),
            "swap_self_fact_hit_rate": swap_hits / max(1, examples),
            "normal_fact_weight_mass": normal_mass / max(1, examples),
            "swap_self_fact_weight_mass": swap_mass / max(1, examples),
            "normal_answer_accuracy": correct / max(1, examples),
            "normal_minus_zero_target_logit": causal_margin / max(1, examples),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    passed = all(
        row["normal_fact_hit_rate"] >= args.min_retrieval_hit
        and row["normal_fact_hit_rate"] - row["swap_self_fact_hit_rate"] >= args.min_swap_gap
        and row["normal_minus_zero_target_logit"] >= args.min_logit_effect
        for row in rows
    )
    return {"status": "pass" if passed else "fail", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--chunks", nargs="+", type=int, default=[2, 8, 16])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--writes-per-chunk", type=int, default=8)
    parser.add_argument("--reads-per-query", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--seed", type=int, default=1303)
    parser.add_argument("--min-retrieval-hit", type=float, default=0.5)
    parser.add_argument("--min-swap-gap", type=float, default=0.2)
    parser.add_argument("--min-logit-effect", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("experiments/retrieval_gate/results.json"))
    parser.add_argument("--log", type=Path, default=Path("experiments/retrieval_gate/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2))
        return 0
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": protocol, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v0_3_RETRIEVAL_GATE_{result['status'].upper()}")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
