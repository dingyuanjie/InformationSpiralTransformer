"""Milestone 2.2: locked out-of-distribution Reader generalization gate."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from config import SourceTokenMemoryConfig
from pretrained_token_memory_adapter import FrozenTokenMemoryIST
from run_v0_3_coverage_gate import DEFAULT_MODEL, Tee, load_model
from run_v0_3_retrieval_gate import left_pad


ENTITIES = ["Aster", "Borin", "Cyra", "Dalen", "Eris", "Faron", "Galen", "Hesta"]
ANSWER_POOL = ["Falcon", "Saffron", "Obsidian", "Willow", "Harbor", "Lantern", "Crimson", "Marble"]
FACT_TEMPLATES = [
    ("In the sealed registry, {entity} has verification phrase ", ". End of entry."),
    ("Audit memorandum: use ", " when validating {entity}."),
    ("The private credential assigned to {entity} reads ", ". Keep it confidential."),
]
QUERY_TEMPLATES = [
    "Retrieve the verification phrase belonging to {entity}. Response:",
    "Which private credential was assigned to {entity}? Response:",
    "According to the registry, validate {entity} using:",
]
FILLER = " Routine shipping minutes discuss weather, schedules, invoices, and unrelated maintenance."


def encode_parts(tokenizer, entity, answer, template_index):
    before, after = FACT_TEMPLATES[template_index % len(FACT_TEMPLATES)]
    prefix = tokenizer.encode(before.format(entity=entity), add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    suffix = tokenizer.encode(after.format(entity=entity), add_special_tokens=False)
    return prefix + answer_ids + suffix, len(prefix), len(answer_ids)


def non_overlapping_start(rng, total, length, occupied):
    for _ in range(256):
        start = rng.randrange(8, max(9, total - length - 1))
        span = set(range(start, start + length))
        if not span.intersection(occupied):
            occupied.update(span)
            return start
    raise RuntimeError("could not place a non-overlapping fact span")


def strict_stream(tokenizer, chunk_size, chunks, seed, answer):
    rng = random.Random(seed)
    total = chunk_size * chunks
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    stream = (filler * (total // len(filler) + 1))[:total]
    entity = rng.choice(ENTITIES)
    occupied = set()
    target, answer_offset, answer_length = encode_parts(tokenizer, entity, answer, seed)
    target_start = non_overlapping_start(rng, total, len(target), occupied)
    stream[target_start:target_start + len(target)] = target
    fact_positions = set(range(
        target_start + answer_offset,
        target_start + answer_offset + answer_length,
    ))
    distractor_positions = set()
    other_entities = [item for item in ENTITIES if item != entity]
    other_answers = [item for item in ANSWER_POOL if item != answer]
    for index in range(3):
        distractor, offset, length = encode_parts(
            tokenizer, other_entities[index], other_answers[(seed + index) % len(other_answers)],
            seed + index + 1,
        )
        start = non_overlapping_start(rng, total, len(distractor), occupied)
        stream[start:start + len(distractor)] = distractor
        distractor_positions.update(range(start + offset, start + offset + length))
    query = tokenizer.encode(
        QUERY_TEMPLATES[(seed // 3) % len(QUERY_TEMPLATES)].format(entity=entity),
        add_special_tokens=False,
    )
    return (
        torch.tensor(stream, dtype=torch.long), torch.tensor(query, dtype=torch.long),
        fact_positions, distractor_positions,
        {"entity": entity, "answer": answer, "target_start": target_start},
    )


def choose_answers(tokenizer, count=5):
    chosen, ids = [], []
    for answer in ANSWER_POOL:
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if token_ids and token_ids[0] not in ids:
            chosen.append(answer); ids.append(token_ids[0])
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError("held-out answer pool does not provide five unique first tokens")
    return chosen, ids


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
    candidate_ids = torch.tensor(answer_token_ids, device=device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = []
    for chunks in args.chunks:
        totals = {key: 0.0 for key in ("hit", "swap_hit", "mass", "swap_mass", "distractor_mass", "accuracy", "logit")}
        examples = 0
        for start in range(0, args.samples, 2):
            if start + 1 >= args.samples:
                break
            batch = []
            for offset in range(2):
                answer = answers[(start + offset) % len(answers)]
                batch.append(strict_stream(tokenizer, args.chunk_size, chunks,
                                           args.seed + chunks * 1000 + start + offset, answer))
            streams = torch.stack([item[0] for item in batch]).to(device)
            queries = left_pad([item[1] for item in batch], pad).to(device)
            state = None
            for chunk_id in range(chunks):
                begin = chunk_id * args.chunk_size
                _, state = model(streams[:, begin:begin + args.chunk_size], state,
                                 chunk_id, begin, "normal", True)
            normal_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, "normal", True)
            normal = model.last_provenance
            zero_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, "zero", True)
            model(queries, state, chunks, chunks * args.chunk_size, "swap", True)
            swapped = model.last_provenance
            for row, item in enumerate(batch):
                _, _, fact, distractors, meta = item
                signature = {(position, int(streams[row, position])) for position in fact}
                distractor_signature = {(position, int(streams[row, position])) for position in distractors}
                pairs = list(zip(normal["positions"][row, -1].cpu().tolist(), normal["token_ids"][row, -1].cpu().tolist()))
                swap_pairs = list(zip(swapped["positions"][row, -1].cpu().tolist(), swapped["token_ids"][row, -1].cpu().tolist()))
                weights = normal["weights"][row, -1].cpu().tolist()
                swap_weights = swapped["weights"][row, -1].cpu().tolist()
                totals["hit"] += any(pair in signature for pair in pairs)
                totals["swap_hit"] += any(pair in signature for pair in swap_pairs)
                totals["mass"] += sum(weight for pair, weight in zip(pairs, weights) if pair in signature)
                totals["swap_mass"] += sum(weight for pair, weight in zip(swap_pairs, swap_weights) if pair in signature)
                totals["distractor_mass"] += sum(weight for pair, weight in zip(pairs, weights) if pair in distractor_signature)
                target_index = answers.index(meta["answer"])
                totals["accuracy"] += int(normal_logits[row, -1, candidate_ids].argmax().item() == target_index)
                target_id = candidate_ids[target_index]
                totals["logit"] += float(normal_logits[row, -1, target_id] - zero_logits[row, -1, target_id])
                examples += 1
        result = {
            "chunks": chunks, "examples": examples,
            "fact_hit_rate": totals["hit"] / examples,
            "swap_self_fact_hit_rate": totals["swap_hit"] / examples,
            "fact_weight_mass": totals["mass"] / examples,
            "distractor_weight_mass": totals["distractor_mass"] / examples,
            "answer_accuracy": totals["accuracy"] / examples,
            "normal_minus_zero_target_logit": totals["logit"] / examples,
        }
        rows.append(result); print(json.dumps(result, ensure_ascii=False), flush=True)
    passed = all(
        row["fact_hit_rate"] >= args.min_hit
        and row["answer_accuracy"] >= args.min_accuracy
        and row["swap_self_fact_hit_rate"] <= args.max_swap_hit
        and row["normal_minus_zero_target_logit"] >= args.min_logit_effect
        for row in rows
    )
    return {"status": "pass" if passed else "fail", "answers": answers, "rows": rows}


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
    parser.add_argument("--min-hit", type=float, default=0.75)
    parser.add_argument("--min-accuracy", type=float, default=0.6)
    parser.add_argument("--max-swap-hit", type=float, default=0.2)
    parser.add_argument("--min-logit-effect", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("experiments/strict_generalization/results.json"))
    parser.add_argument("--log", type=Path, default=Path("experiments/strict_generalization/run.log"))
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
    print(f"v0_3_STRICT_GENERALIZATION_{result['status'].upper()}", flush=True)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
