"""Milestone 1: selective event encoding, rehearsal and forgetting gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from cognitive_event_memory import CognitiveEventMemory
from config import CognitiveMemoryConfig


class Tee:
    def __init__(self, terminal, log_file):
        self.terminal, self.log_file = terminal, log_file

    def write(self, text):
        self.terminal.write(text); self.log_file.write(text); self.log_file.flush()
        return len(text)

    def flush(self):
        self.terminal.flush(); self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


def target_slots(store, target_id):
    return store["valid"] & (store["token_ids"] == target_id).any(-1)


def make_chunk(seed, chunk, tokens, hidden_size, target_id, mode, repeat_chunks):
    generator = torch.Generator().manual_seed(seed * 100003 + chunk)
    hidden = torch.randn(1, tokens, hidden_size, generator=generator) * 0.08
    ids = torch.arange(chunk * tokens, (chunk + 1) * tokens)[None] + 10000
    write_target = chunk == 0 or (mode == "repeated" and chunk in repeat_chunks)
    target_span = None
    if write_target:
        span = 8
        event_index = (seed + chunk) % (tokens // span)
        begin = event_index * span
        ids[0, begin + 3] = target_id
        target_span = set(ids[0, begin:begin + span].tolist())
        if mode != "incidental":
            direction = torch.linspace(-3.0, 3.0, span)[:, None]
            signature = torch.zeros(span, hidden_size)
            signature[:, (seed * 7) % hidden_size] = direction[:, 0]
            signature[:, (seed * 7 + 3) % hidden_size] = direction.flip(0)[:, 0]
            hidden[0, begin:begin + span] += signature
    return hidden, ids, target_span


def run_one(args, length, seed, mode):
    torch.manual_seed(args.seed + seed)
    config = CognitiveMemoryConfig(
        event_span=8, working_events=args.working_events,
        episodic_events=args.episodic_events, semantic_slots=args.semantic_slots,
        admissions_per_chunk=args.admissions_per_chunk,
        retrieved_events=3, age_decay=args.age_decay,
        access_bonus=args.access_bonus, consolidation_accesses=args.consolidation_accesses,
    )
    memory = CognitiveEventMemory(args.hidden_size, config)
    target_id = 900000 + seed
    repeat_chunks = {length // 3, (2 * length) // 3}
    state = None
    source_spans = []
    admitted_initially = False
    for chunk in range(length):
        hidden, ids, span = make_chunk(args.seed + seed, chunk, args.chunk_tokens,
                                       args.hidden_size, target_id, mode, repeat_chunks)
        if span is not None:
            source_spans.append(span)
        state = memory.write(hidden, ids, state, chunk, chunk * args.chunk_tokens)
        present = target_slots(state["episodic"], target_id)
        if chunk == 0:
            admitted_initially = bool(present.any())
        if mode == "reinforced" and present.any() and (chunk == 0 or (chunk + 1) % args.rehearse_every == 0):
            slots = torch.where(present[0])[0][None]
            for _ in range(args.rehearsals_per_visit):
                state = memory.reinforce(state, slots)
    working = bool(target_slots(state["working"], target_id).any())
    episodic_mask = target_slots(state["episodic"], target_id)
    episodic = bool(episodic_mask.any())
    span_intact = False
    if episodic and source_spans:
        for slot in torch.where(episodic_mask[0])[0].tolist():
            retained = set(state["episodic"]["token_ids"][0, slot][state["episodic"]["token_valid"][0, slot]].tolist())
            span_intact |= retained in source_spans
    consolidated = bool(state["semantic"]["valid"].any())
    return {"admitted": admitted_initially, "working": working, "episodic": episodic,
            "span_intact": span_intact, "consolidated": consolidated}


def evaluate(args):
    rows = []
    modes = ("incidental", "distinctive", "repeated", "reinforced")
    for length in args.lengths:
        outcomes = {mode: [run_one(args, length, seed, mode) for seed in range(args.samples)] for mode in modes}
        row = {"chunks": length, "samples": args.samples}
        for mode in modes:
            for metric in ("admitted", "working", "episodic", "span_intact", "consolidated"):
                row[f"{mode}_{metric}_rate"] = sum(item[metric] for item in outcomes[mode]) / args.samples
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    passed = all(
        row["distinctive_admitted_rate"] >= args.min_admission
        and row["reinforced_episodic_rate"] >= args.min_reinforced_retention
        and row["reinforced_span_intact_rate"] >= args.min_reinforced_retention
        and row["reinforced_consolidated_rate"] >= args.min_consolidation
        and row["reinforced_episodic_rate"] >= row["incidental_episodic_rate"]
        for row in rows
    )
    return {"status": "pass" if passed else "fail", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--working-events", type=int, default=4)
    parser.add_argument("--episodic-events", type=int, default=16)
    parser.add_argument("--semantic-slots", type=int, default=8)
    parser.add_argument("--admissions-per-chunk", type=int, default=2)
    parser.add_argument("--age-decay", type=float, default=0.04)
    parser.add_argument("--access-bonus", type=float, default=0.4)
    parser.add_argument("--consolidation-accesses", type=int, default=3)
    parser.add_argument("--rehearse-every", type=int, default=4)
    parser.add_argument("--rehearsals-per-visit", type=int, default=2)
    parser.add_argument("--min-admission", type=float, default=0.9)
    parser.add_argument("--min-reinforced-retention", type=float, default=0.8)
    parser.add_argument("--min-consolidation", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1404)
    parser.add_argument("--output", type=Path, default=Path("experiments/lifecycle_gate/results.json"))
    parser.add_argument("--log", type=Path, default=Path("experiments/lifecycle_gate/run.log"))
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
    print(f"v0_4_LIFECYCLE_GATE_{result['status'].upper()}", flush=True)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
