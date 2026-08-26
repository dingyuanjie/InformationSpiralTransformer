"""Milestone 1: verify that source-token Memory retains the answer span."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from config import SourceTokenMemoryConfig
from pretrained_token_memory_adapter import FrozenTokenMemoryIST
from source_token_memory import SourceTokenMemory


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def synthetic_smoke() -> dict:
    torch.manual_seed(303)
    config = SourceTokenMemoryConfig(capacity=16, writes_per_chunk=4, reads_per_query=2, heads=4)
    memory = SourceTokenMemory(16, config)
    state = None
    fact_positions = {5, 6}
    for chunk_id in range(4):
        hidden = torch.randn(2, 8, 16) * 0.05
        if chunk_id == 0:
            hidden[:, 5:7] += 8.0
        ids = torch.arange(chunk_id * 8, (chunk_id + 1) * 8).repeat(2, 1)
        state = memory.write(hidden, ids, state, chunk_id, chunk_id * 8)
    retained = []
    for row in state["positions"]:
        retained.append(bool(fact_positions.intersection(row[row >= 0].tolist())))
    return {"status": "pass" if all(retained) else "fail", "fact_retained": retained}


def load_model(model_id: str, local_files_only: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, use_fast=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=local_files_only, torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    return tokenizer, backbone, device


def make_stream(tokenizer, chunk_size: int, chunks: int, seed: int):
    rng = random.Random(seed)
    names = ["Mira", "Tovan", "Selka", "Orin", "Vela"]
    codes = ["Kestrel", "Amber", "Nimbus", "Quartz", "Cobalt"]
    name, code = rng.choice(names), rng.choice(codes)
    prefix = f"Archive record. The access code for {name} is "
    answer = code
    suffix = ". Preserve this record."
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    source = prefix_ids + answer_ids + tokenizer.encode(suffix, add_special_tokens=False)
    filler_ids = tokenizer.encode(
        " Background notes contain ordinary unrelated administrative text.", add_special_tokens=False
    )
    total = chunk_size * chunks
    stream = list(source)
    while len(stream) < total:
        stream.extend(filler_ids)
    stream = stream[:total]
    fact = set(range(len(prefix_ids), len(prefix_ids) + len(answer_ids)))
    return torch.tensor(stream, dtype=torch.long), fact, {"entity": name, "answer": answer}


@torch.no_grad()
def evaluate(args) -> dict:
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    config = SourceTokenMemoryConfig(
        capacity=args.capacity, writes_per_chunk=args.writes_per_chunk,
        reads_per_query=args.reads_per_query,
        heads=args.heads, injection_layer=args.injection_layer,
    )
    model = FrozenTokenMemoryIST(backbone, config).to(device)
    rows = []
    for chunks in args.chunks:
        hits = 0
        token_recall = 0.0
        for sample in range(args.samples):
            ids, fact, metadata = make_stream(tokenizer, args.chunk_size, chunks, args.seed + sample)
            state = None
            for chunk_id, start in enumerate(range(0, ids.numel(), args.chunk_size)):
                piece = ids[start:start + args.chunk_size][None].to(device)
                _, state = model(piece, state, chunk_id, start, detach_state=True)
            retained = set(state["positions"][0][state["valid"][0]].cpu().tolist())
            overlap = retained.intersection(fact)
            hits += bool(overlap)
            token_recall += len(overlap) / max(1, len(fact))
        rows.append({
            "chunks": chunks,
            "distance_tokens": chunks * args.chunk_size,
            "span_hit_rate": hits / args.samples,
            "fact_token_recall": token_recall / args.samples,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    passed = all(row["span_hit_rate"] >= args.min_hit_rate for row in rows)
    return {"status": "pass" if passed else "fail", "gate": args.min_hit_rate, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--chunks", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--writes-per-chunk", type=int, default=8)
    parser.add_argument("--reads-per-query", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--min-hit-rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--output", type=Path, default=Path("experiments/coverage_gate/results.json"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    protocol = vars(args).copy()
    protocol["output"] = str(protocol["output"])
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2))
        return 0
    result = synthetic_smoke() if args.smoke_test else evaluate(args)
    payload = {"protocol": protocol, "result": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v0_3_COVERAGE_GATE_{result['status'].upper()}")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
