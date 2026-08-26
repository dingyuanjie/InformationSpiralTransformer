"""Milestone 2.2.2: orthogonal OOD factors and 32-chunk capacity scaling."""
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
from run_v0_3_failure_decomposition import oracle_state, signature_mask
from run_v0_3_retrieval_gate import ANSWERS as OLD_ANSWERS, left_pad
from run_v0_3_strict_generalization import (
    ANSWER_POOL, ENTITIES, FACT_TEMPLATES, FILLER, QUERY_TEMPLATES,
    choose_answers, encode_parts, non_overlapping_start,
)


CONDITIONS = {
    "train_distribution": dict(new_answers=False, new_template=False, random_position=False, distractors=False),
    "new_template_only": dict(new_answers=False, new_template=True, random_position=False, distractors=False),
    "new_answers_only": dict(new_answers=True, new_template=False, random_position=False, distractors=False),
    "random_position_only": dict(new_answers=False, new_template=False, random_position=True, distractors=False),
    "distractors_only": dict(new_answers=False, new_template=False, random_position=False, distractors=True),
    "strict_full": dict(new_answers=True, new_template=True, random_position=True, distractors=True),
}


def old_parts(tokenizer, entity, answer):
    prefix = tokenizer.encode(f"Archive record. The access code for {entity} is ", add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    suffix = tokenizer.encode(". Preserve this record.", add_special_tokens=False)
    return prefix + answer_ids + suffix, len(prefix), len(answer_ids)


def factor_stream(tokenizer, chunk_size, chunks, seed, answer, condition):
    rng = random.Random(seed)
    total = chunk_size * chunks
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    stream = (filler * (total // len(filler) + 1))[:total]
    entity = ENTITIES[seed % len(ENTITIES)] if condition["new_template"] else ["Mira", "Tovan", "Selka", "Orin", "Vela"][seed % 5]
    parts = encode_parts(tokenizer, entity, answer, seed) if condition["new_template"] else old_parts(tokenizer, entity, answer)
    target, offset, length = parts
    occupied = set()
    start = non_overlapping_start(rng, total, len(target), occupied) if condition["random_position"] else 0
    occupied.update(range(start, start + len(target)))
    stream[start:start + len(target)] = target
    fact = set(range(start + offset, start + offset + length))
    distractor_positions = set()
    if condition["distractors"]:
        pool = ANSWER_POOL if condition["new_answers"] else OLD_ANSWERS
        others = [item for item in pool if item != answer]
        for index in range(3):
            other_entity = ENTITIES[(seed + index + 1) % len(ENTITIES)]
            distractor = encode_parts(tokenizer, other_entity, others[index % len(others)], seed + index + 7) if condition["new_template"] else old_parts(tokenizer, other_entity, others[index % len(others)])
            tokens, answer_offset, answer_length = distractor
            location = non_overlapping_start(rng, total, len(tokens), occupied)
            stream[location:location + len(tokens)] = tokens
            distractor_positions.update(range(location + answer_offset, location + answer_offset + answer_length))
    query_text = (
        QUERY_TEMPLATES[(seed // 3) % len(QUERY_TEMPLATES)].format(entity=entity)
        if condition["new_template"]
        else f"Question: what is the access code for {entity}? Answer:"
    )
    query = tokenizer.encode(query_text, add_special_tokens=False)
    return torch.tensor(stream), torch.tensor(query), fact, distractor_positions, {"answer": answer}


@torch.no_grad()
def evaluate_setting(tokenizer, backbone, payload, device, args, name, condition, chunks, capacity):
    answers = choose_answers(tokenizer)[0] if condition["new_answers"] else list(OLD_ANSWERS)
    candidate_ids = torch.tensor([tokenizer.encode(answer, add_special_tokens=False)[0] for answer in answers], device=device)
    config = SourceTokenMemoryConfig(capacity=capacity, writes_per_chunk=args.writes_per_chunk,
                                     reads_per_query=args.reads_per_query, heads=args.heads,
                                     injection_layer=args.injection_layer)
    model = FrozenTokenMemoryIST(backbone, config).to(device)
    model.load_state_dict(payload["adapter"], strict=False)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    totals = {key: 0.0 for key in ("available", "reader", "accuracy", "oracle", "distractor_mass")}
    available_count = examples = 0
    for start in range(0, args.samples, 2):
        if start + 1 >= args.samples: break
        batch = [factor_stream(tokenizer, args.chunk_size, chunks,
                               args.seed + start + offset + chunks * 1000,
                               answers[(start + offset) % len(answers)], condition) for offset in range(2)]
        streams = torch.stack([item[0] for item in batch]).to(device)
        queries = left_pad([item[1] for item in batch], pad).to(device)
        signatures, distractor_signatures = [], []
        for row, item in enumerate(batch):
            signatures.append({(position, int(streams[row, position])) for position in item[2]})
            distractor_signatures.append({(position, int(streams[row, position])) for position in item[3]})
        state = None
        for chunk_id in range(chunks):
            begin = chunk_id * args.chunk_size
            _, state = model(streams[:, begin:begin + args.chunk_size], state, chunk_id, begin, "normal", True)
        normal_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, "normal", True)
        provenance = model.last_provenance
        filtered = oracle_state(state, signatures)
        oracle_logits, _ = model(queries, filtered, chunks, chunks * args.chunk_size, "normal", True)
        for row, item in enumerate(batch):
            available = bool(signature_mask(state, row, signatures[row]).any())
            totals["available"] += available
            pairs = list(zip(provenance["positions"][row, -1].cpu().tolist(), provenance["token_ids"][row, -1].cpu().tolist()))
            weights = provenance["weights"][row, -1].cpu().tolist()
            hit = any(pair in signatures[row] for pair in pairs)
            if available:
                available_count += 1; totals["reader"] += hit
            totals["distractor_mass"] += sum(weight for pair, weight in zip(pairs, weights) if pair in distractor_signatures[row])
            target = answers.index(item[4]["answer"])
            totals["accuracy"] += int(normal_logits[row, -1, candidate_ids].argmax().item() == target)
            totals["oracle"] += int(oracle_logits[row, -1, candidate_ids].argmax().item() == target)
            examples += 1
    result = {
        "setting": name, "chunks": chunks, "capacity": capacity, "examples": examples,
        "writer_availability": totals["available"] / examples,
        "reader_hit_given_available": totals["reader"] / max(1, available_count),
        "answer_accuracy": totals["accuracy"] / examples,
        "oracle_answer_accuracy": totals["oracle"] / examples,
        "distractor_weight_mass": totals["distractor_mass"] / examples,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/reader_alignment/reader_step_000400.pt"))
    parser.add_argument("--factor-chunks", type=int, default=8)
    parser.add_argument("--capacity-chunks", type=int, default=32)
    parser.add_argument("--capacities", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--writes-per-chunk", type=int, default=8)
    parser.add_argument("--reads-per-query", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--seed", type=int, default=4303)
    parser.add_argument("--output", type=Path, default=Path("experiments/factor_ablation/results.json"))
    parser.add_argument("--log", type=Path, default=Path("experiments/factor_ablation/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2)); return 0
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    factor_rows = [evaluate_setting(tokenizer, backbone, payload, device, args, name, CONDITIONS[name], args.factor_chunks, 64) for name in args.conditions]
    capacity_rows = [evaluate_setting(tokenizer, backbone, payload, device, args, f"strict_capacity_{capacity}", CONDITIONS["strict_full"], args.capacity_chunks, capacity) for capacity in args.capacities]
    result = {"status": "complete", "factor_rows": factor_rows, "capacity_rows": capacity_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": protocol, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("v0_3_FACTOR_ABLATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
