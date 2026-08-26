"""Milestone 2.1: align the Reader while freezing backbone and Writer."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from config import SourceTokenMemoryConfig
from pretrained_token_memory_adapter import FrozenTokenMemoryIST
from run_v0_3_coverage_gate import DEFAULT_MODEL, Tee, load_model, make_stream
from run_v0_3_retrieval_gate import ANSWERS, left_pad


def adapter_state(model):
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }


def latest_checkpoint(folder: Path):
    candidates = list(folder.glob("reader_step_*.pt"))
    return max(candidates, key=lambda path: int(path.stem.rsplit("_", 1)[1])) if candidates else None


def save_checkpoint(folder, step, model, optimizer, history, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"reader_step_{step:06d}.pt"
    if path.exists():
        return path
    torch.save({
        "step": step, "adapter": adapter_state(model),
        "optimizer": optimizer.state_dict(), "history": history,
        "protocol": protocol, "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)
    return path


def make_batch(tokenizer, chunk_size, chunks, batch, seed, device):
    streams, facts, metadata, queries = [], [], [], []
    for row in range(batch):
        stream, fact, meta = make_stream(tokenizer, chunk_size, chunks, seed + row)
        streams.append(stream); facts.append(fact); metadata.append(meta)
        queries.append(torch.tensor(tokenizer.encode(
            f"Question: what is the access code for {meta['entity']}? Answer:",
            add_special_tokens=False,
        ), dtype=torch.long))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return torch.stack(streams).to(device), left_pad(queries, pad).to(device), facts, metadata


def retrieval_loss(scores, positions, facts):
    losses = []
    scores = scores[:, -1]
    for row, fact in enumerate(facts):
        valid = positions[row] >= 0
        target = torch.zeros_like(valid)
        for position in fact:
            target |= positions[row] == position
        target &= valid
        if target.any():
            losses.append(torch.logsumexp(scores[row, valid], 0) - torch.logsumexp(scores[row, target], 0))
    return torch.stack(losses).mean() if losses else scores.sum() * 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--curriculum-chunks", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--writes-per-chunk", type=int, default=8)
    parser.add_argument("--reads-per-query", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--retrieval-weight", type=float, default=0.5)
    parser.add_argument("--causal-weight", type=float, default=0.25)
    parser.add_argument("--causal-margin", type=float, default=0.25)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2303)
    parser.add_argument("--output", type=Path, default=Path("experiments/reader_alignment"))
    parser.add_argument("--log", type=Path, default=Path("experiments/reader_alignment/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("a" if args.resume else "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2))
        return 0
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    config = SourceTokenMemoryConfig(
        capacity=args.capacity, writes_per_chunk=args.writes_per_chunk,
        reads_per_query=args.reads_per_query, heads=args.heads,
        injection_layer=args.injection_layer,
    )
    model = FrozenTokenMemoryIST(backbone, config).to(device)
    model.memory.salience.weight.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    start, history = 0, []
    checkpoint = latest_checkpoint(args.output) if args.resume else None
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["adapter"], strict=False)
        optimizer.load_state_dict(payload["optimizer"])
        start, history = int(payload["step"]), payload["history"]
        print(f"resumed={checkpoint} step={start}", flush=True)
    candidate_ids = torch.tensor(
        [tokenizer.encode(answer, add_special_tokens=False)[0] for answer in ANSWERS], device=device
    )
    for step in range(start + 1, args.steps + 1):
        chunks = random.choice(args.curriculum_chunks)
        streams, queries, facts, metadata = make_batch(
            tokenizer, args.chunk_size, chunks, args.batch,
            args.seed + step * args.batch, device,
        )
        state = None
        with torch.no_grad():
            for chunk_id in range(chunks):
                begin = chunk_id * args.chunk_size
                _, state = model(streams[:, begin:begin + args.chunk_size], state,
                                 chunk_id, begin, "zero", True)
        normal_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, "normal", True)
        read_scores = model.memory.last_read_scores
        read_positions = model.memory.last_read_positions
        targets = torch.tensor([ANSWERS.index(meta["answer"]) for meta in metadata], device=device)
        candidate_logits = normal_logits[:, -1, candidate_ids].float()
        answer_loss = F.cross_entropy(candidate_logits, targets)
        retrieve_loss = retrieval_loss(read_scores, read_positions, facts)
        condition = "swap" if step % 2 == 0 else "zero"
        intervened_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, condition, True)
        row_ids = torch.arange(args.batch, device=device)
        token_targets = candidate_ids[targets]
        normal_target = normal_logits[row_ids, -1, token_targets].float()
        intervened_target = intervened_logits[row_ids, -1, token_targets].float()
        causal_loss = F.relu(args.causal_margin - (normal_target - intervened_target)).mean()
        loss = answer_loss + args.retrieval_weight * retrieve_loss + args.causal_weight * causal_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {
            "step": step, "chunks": chunks, "loss": float(loss.detach()),
            "answer_loss": float(answer_loss.detach()), "retrieval_loss": float(retrieve_loss.detach()),
            "causal_loss": float(causal_loss.detach()),
            "accuracy": float((candidate_logits.argmax(-1) == targets).float().mean()),
            "gate": float(torch.tanh(model.injection_gate.detach().float())),
        }
        history.append(row)
        if step == 1 or step % 10 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            path = save_checkpoint(args.output, step, model, optimizer, history, protocol)
            print(f"checkpoint={path}", flush=True)
    summary = {"status": "complete", "final": history[-1], "checkpoint": str(path)}
    (args.output / "training_results.json").write_text(
        json.dumps({"protocol": protocol, "summary": summary, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("v0_3_READER_ALIGNMENT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
