"""Milestone 2.2.1: verify entity-relation-value event completeness."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from cognitive_event_memory import CognitiveEventMemory
from config import CognitiveMemoryConfig
from run_v0_4_pretrained_writer_gate import Tee, open_tokens


ENTITIES = ["Maren Vale", "Doctor Amina Rho", "Basil North", "Celine Ardent",
            "Dorian Pike", "Elara Stone", "Felix Rowan", "Greta Sol"]
TEMPLATES = [
    ("Audit registry states that ", " has private verification token ", ". End record."),
    ("For secure validation, the credential assigned to ", " is exactly ", ". Preserve it."),
    ("During the review, officers confirmed ", " should be identified by code ", "."),
]
FILLER = " routine schedules weather invoices shipping maintenance notes"


def tokenizer_local(model_id, local_files_only):
    from transformers import AutoTokenizer
    try: return AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
    except OSError:
        if local_files_only: raise
        return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def example(tokenizer, seed, answer_id, tokens):
    rng = random.Random(seed); entity = ENTITIES[seed % len(ENTITIES)]
    before, relation, after = TEMPLATES[seed % len(TEMPLATES)]
    prefix = tokenizer.encode(before, add_special_tokens=False)
    entity_ids = tokenizer.encode(entity, add_special_tokens=False)
    middle = tokenizer.encode(relation, add_special_tokens=False)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    fact = prefix + entity_ids + middle + [answer_id] + suffix
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    stream = (filler * (tokens // len(filler) + 1))[:tokens]
    start = rng.randrange(0, tokens - len(fact))
    stream[start:start + len(fact)] = fact
    entity_positions = set(range(start + len(prefix), start + len(prefix) + len(entity_ids)))
    answer_position = start + len(prefix) + len(entity_ids) + len(middle)
    required = entity_positions | {answer_position}
    return torch.tensor(stream)[None], required


def coverage(tokenizer, args, span, stride, targets):
    memory = CognitiveEventMemory(4, CognitiveMemoryConfig(
        event_span=span, event_stride=stride, working_events=2,
        episodic_events=32, semantic_slots=2, admissions_per_chunk=1, retrieved_events=1))
    hits = 0; event_counts = []
    for sample in range(args.samples):
        ids, required = example(tokenizer, args.seed + sample, targets[sample % len(targets)], args.chunk_tokens)
        hidden = torch.zeros(1, args.chunk_tokens, 4)
        _, _, positions, valid, _, _ = memory._events(hidden, ids, 0)
        complete = any(required.issubset(set(row[row >= 0].tolist())) for row in positions[0])
        hits += complete; event_counts.append(int(valid.any(-1).sum()))
    return {"span": span, "stride": stride, "relation_complete_rate": hits / args.samples,
            "mean_events_per_chunk": sum(event_counts) / len(event_counts),
            "overlap_factor": span / stride}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=5404)
    parser.add_argument("--min-complete", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=Path("experiments/relation_coverage/results.json"))
    parser.add_argument("--log", type=Path, default=Path("experiments/relation_coverage/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2)); return 0
    tokenizer = tokenizer_local(args.model_id, args.local_files_only)
    targets = [token_id for token_id, _ in open_tokens(tokenizer, max(8, args.samples))]
    rows = [coverage(tokenizer, args, span, stride, targets)
            for span, stride in ((8, 8), (16, 16), (16, 8), (24, 8))]
    for row in rows: print(json.dumps(row), flush=True)
    overlap = rows[-1]
    status = "pass" if overlap["relation_complete_rate"] >= args.min_complete else "fail"
    result = {"status": status, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": protocol, "result": result}, indent=2), encoding="utf-8")
    print(f"v0_4_RELATION_COVERAGE_{status.upper()}", flush=True)
    return 0 if status == "pass" else 2


if __name__ == "__main__": raise SystemExit(main())
