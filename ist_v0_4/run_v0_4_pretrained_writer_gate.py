"""Milestone 2: frozen-Qwen natural-language event Writer lifecycle gate."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from config import CognitiveMemoryConfig
from pretrained_cognitive_adapter import FrozenCognitiveIST


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
FILLER = " Routine reports discuss schedules, maintenance, weather, invoices, and unrelated shipping notes."


class Tee:
    def __init__(self, terminal, log_file): self.terminal, self.log_file = terminal, log_file
    def write(self, text):
        self.terminal.write(text); self.log_file.write(text); self.log_file.flush(); return len(text)
    def flush(self): self.terminal.flush(); self.log_file.flush()
    def isatty(self): return self.terminal.isatty()


def load_model(model_id, local_files_only):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, dtype=dtype, attn_implementation="sdpa").to(device)
    except OSError:
        if local_files_only: raise
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation="sdpa").to(device)
    return tokenizer, backbone, device


def open_tokens(tokenizer, count):
    result = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        word = text.strip()
        if 4 <= len(word) <= 10 and word.isascii() and word.isalpha():
            result.append((token_id, text))
    result.sort(key=lambda item: ((item[0] * 2654435761) & 0xFFFFFFFF, item[0]))
    if len(result) < count: raise RuntimeError("not enough open tokens")
    return result[:count]


def target_slots(store, target_id):
    return store["valid"] & (store["token_ids"] == target_id).any(-1)


def build_stream(tokenizer, chunks, chunk_size, seed, target_id, mode):
    rng = random.Random(seed)
    total = chunks * chunk_size
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    stream = (filler * (total // len(filler) + 1))[:total]
    repeat_chunks = [0] if mode in {"single", "reinforced"} else [0, chunks // 3, 2 * chunks // 3]
    source_spans = []
    for occurrence, chunk in enumerate(sorted(set(repeat_chunks))):
        prefix = tokenizer.encode(" Critical registry event. The private verification token is", add_special_tokens=False)
        suffix = tokenizer.encode(". Preserve this exact event for later verification.", add_special_tokens=False)
        fact = prefix + [target_id] + suffix
        event_span = 8
        local_event = rng.randrange(max(1, chunk_size // event_span - 2))
        start = chunk * chunk_size + local_event * event_span
        fact = fact[:event_span]
        if target_id not in fact:
            fact[-2] = target_id
        stream[start:start + event_span] = fact
        source_spans.append(set(range(start, start + event_span)))
    return torch.tensor(stream, dtype=torch.long), source_spans


@torch.no_grad()
def run_one(model, tokenizer, device, args, length, sample, mode, target_id):
    stream, source_spans = build_stream(tokenizer, length, args.chunk_size,
                                        args.seed + sample + length * 1000, target_id, mode)
    state = None; admitted = False
    for chunk in range(length):
        begin = chunk * args.chunk_size
        piece = stream[begin:begin + args.chunk_size][None].to(device)
        _, state = model(piece, state, chunk, begin, "zero", True)
        present = target_slots(state["episodic"], target_id)
        if chunk == 0: admitted = bool(present.any())
        if mode == "reinforced" and present.any() and (chunk == 0 or (chunk + 1) % args.rehearse_every == 0):
            slots = torch.where(present[0])[0][None]
            for _ in range(args.rehearsals_per_visit): state = model.memory.reinforce(state, slots)
    mask = target_slots(state["episodic"], target_id)
    retained = bool(mask.any()); intact = False
    if retained:
        for slot in torch.where(mask[0])[0].tolist():
            positions = set(state["episodic"]["positions"][0, slot][state["episodic"]["token_valid"][0, slot]].tolist())
            intact |= positions in source_spans
    return {"admitted": admitted, "retained": retained, "intact": intact,
            "consolidated": bool(state["semantic"]["valid"].any())}


@torch.no_grad()
def evaluate(args):
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    config = CognitiveMemoryConfig(
        event_span=8, working_events=args.working_events, episodic_events=args.episodic_events,
        semantic_slots=args.semantic_slots, admissions_per_chunk=args.admissions_per_chunk,
        retrieved_events=3, age_decay=args.age_decay, access_bonus=args.access_bonus,
        consolidation_accesses=args.consolidation_accesses)
    model = FrozenCognitiveIST(backbone, config, args.injection_layer).to(device)
    targets = open_tokens(tokenizer, args.samples)
    rows = []
    for length in args.lengths:
        row = {"chunks": length, "samples": args.samples}
        for mode in ("single", "repeated", "reinforced"):
            outcomes = [run_one(model, tokenizer, device, args, length, sample, mode, targets[sample][0])
                        for sample in range(args.samples)]
            for metric in ("admitted", "retained", "intact", "consolidated"):
                row[f"{mode}_{metric}_rate"] = sum(item[metric] for item in outcomes) / args.samples
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    passed = all(row["repeated_retained_rate"] >= args.min_repeated
                 and row["reinforced_retained_rate"] >= args.min_reinforced
                 and row["reinforced_intact_rate"] >= args.min_reinforced
                 and row["reinforced_consolidated_rate"] >= args.min_consolidation for row in rows)
    return {"status": "pass" if passed else "fail", "target_tokens": [text for _, text in targets], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4, 16, 32])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--working-events", type=int, default=4)
    parser.add_argument("--episodic-events", type=int, default=32)
    parser.add_argument("--semantic-slots", type=int, default=8)
    # A 128-token chunk contains sixteen eight-token events. Encode them all
    # initially; lifecycle utility, interference and rehearsal decide what
    # survives. Pre-filtering four events recreated v0.3's missing-token flaw.
    parser.add_argument("--admissions-per-chunk", type=int, default=16)
    parser.add_argument("--age-decay", type=float, default=0.02)
    parser.add_argument("--access-bonus", type=float, default=0.4)
    parser.add_argument("--consolidation-accesses", type=int, default=3)
    parser.add_argument("--rehearse-every", type=int, default=4)
    parser.add_argument("--rehearsals-per-visit", type=int, default=2)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--min-repeated", type=float, default=0.75)
    parser.add_argument("--min-reinforced", type=float, default=0.75)
    parser.add_argument("--min-consolidation", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2404)
    parser.add_argument("--output", type=Path, default=Path("experiments/pretrained_writer_gate/results.json"))
    parser.add_argument("--log", type=Path, default=Path("experiments/pretrained_writer_gate/run.log"))
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
    print(f"v0_4_PRETRAINED_WRITER_GATE_{result['status'].upper()}", flush=True)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__": raise SystemExit(main())
