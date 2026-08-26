"""Milestone 2.2: train natural-language Query-to-event retrieval only."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from config import CognitiveMemoryConfig
from pretrained_cognitive_adapter import FrozenCognitiveIST
from run_v0_4_pretrained_writer_gate import Tee, FILLER, load_model, open_tokens


ENTITIES = ["Amina", "Basil", "Celine", "Dorian", "Elara", "Felix", "Greta", "Hector",
            "Iris", "Jonas", "Kara", "Lucan", "Maren", "Nolan", "Opal", "Pavel"]
QUERIES = ["Which verification token belongs to {entity}? Answer:",
           "Retrieve the private code assigned to {entity}. Answer:",
           "According to the registry, {entity} uses token:"]


def make_example(tokenizer, chunks, chunk_size, seed, answer_ids):
    rng = random.Random(seed); total = chunks * chunk_size
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    stream = (filler * (total // len(filler) + 1))[:total]
    entities = rng.sample(ENTITIES, 4); answers = rng.sample(answer_ids, 4)
    target = rng.randrange(4); occupied = set(); target_position = None
    for index, (entity, answer_id) in enumerate(zip(entities, answers)):
        prefix = tokenizer.encode(f" Registry entry: {entity} has private verification token", add_special_tokens=False)
        suffix = tokenizer.encode(". End record.", add_special_tokens=False)
        event = (prefix + [answer_id] + suffix)[-8:]
        if answer_id not in event: event[-2] = answer_id
        for _ in range(100):
            start = rng.randrange(total // 8) * 8
            if not set(range(start, start + 8)).intersection(occupied): break
        occupied.update(range(start, start + 8)); stream[start:start + 8] = event
        if index == target: target_position = start + event.index(answer_id)
    query = tokenizer.encode(QUERIES[seed % len(QUERIES)].format(entity=entities[target]), add_special_tokens=False)
    return torch.tensor(stream), torch.tensor(query), target_position


def left_pad(rows, pad):
    width = max(row.numel() for row in rows); result = torch.full((len(rows), width), pad, dtype=torch.long)
    for index, row in enumerate(rows): result[index, -row.numel():] = row
    return result


def retrieval_loss(scores, positions, targets):
    scores = scores[:, -1]; losses = []; available = 0
    for row, target in enumerate(targets):
        valid = positions[row].ge(0)
        target_events = (positions[row] == target).any(-1) & valid.any(-1)
        valid_events = valid.any(-1)
        if target_events.any():
            available += 1
            losses.append(torch.logsumexp(scores[row, valid_events], 0) - torch.logsumexp(scores[row, target_events], 0))
    zero = scores[torch.isfinite(scores)].sum() * 0 if torch.isfinite(scores).any() else scores.new_zeros(())
    return (torch.stack(losses).mean() if losses else zero), available


def adapter_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not key.startswith("backbone.")}


def latest(folder):
    files = list(folder.glob("query_step_*.pt"))
    return max(files, key=lambda path: int(path.stem.rsplit("_", 1)[1])) if files else None


def save(folder, step, model, optimizer, history, protocol):
    folder.mkdir(parents=True, exist_ok=True); path = folder / f"query_step_{step:06d}.pt"
    if not path.exists():
        torch.save({"step": step, "adapter": adapter_state(model), "optimizer": optimizer.state_dict(),
                    "history": history, "protocol": protocol}, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--curriculum-chunks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--train-token-count", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--seed", type=int, default=4404)
    parser.add_argument("--output", type=Path, default=Path("experiments/query_alignment"))
    parser.add_argument("--log", type=Path, default=Path("experiments/query_alignment/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("a" if args.resume else "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2)); return 0
    torch.manual_seed(args.seed); random.seed(args.seed)
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    answer_ids = [token_id for token_id, _ in open_tokens(tokenizer, args.train_token_count)]
    config = CognitiveMemoryConfig(event_span=8, working_events=4, episodic_events=args.capacity,
                                   semantic_slots=8, admissions_per_chunk=16, retrieved_events=3)
    model = FrozenCognitiveIST(backbone, config, args.injection_layer).to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for module in (model.memory.query, model.memory.event_key, model.query_norm):
        for parameter in module.parameters(): parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    start, history = 0, []; checkpoint = latest(args.output) if args.resume else None
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["adapter"], strict=False); optimizer.load_state_dict(payload["optimizer"])
        start, history = int(payload["step"]), payload["history"]
        print(f"resumed={checkpoint} step={start}", flush=True)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    path = None
    for step in range(start + 1, args.steps + 1):
        chunks = random.choice(args.curriculum_chunks)
        examples = [make_example(tokenizer, chunks, args.chunk_size, args.seed + step * args.batch + row, answer_ids)
                    for row in range(args.batch)]
        streams = torch.stack([item[0] for item in examples]).to(device)
        queries = left_pad([item[1] for item in examples], pad).to(device)
        targets = [item[2] for item in examples]
        state = None
        with torch.no_grad():
            for chunk in range(chunks):
                begin = chunk * args.chunk_size
                _, state = model(streams[:, begin:begin + args.chunk_size], state, chunk, begin, "zero", True)
        model(queries, state, chunks, chunks * args.chunk_size, "normal", True)
        loss, available = retrieval_loss(model.memory.last_event_scores, model.memory.last_event_positions, targets)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
        with torch.no_grad():
            top = model.memory.last_event_scores[:, -1].argmax(-1)
            hit = torch.stack([(model.memory.last_event_positions[row, top[row]] == targets[row]).any()
                               for row in range(args.batch)]).float().mean()
        row = {"step": step, "chunks": chunks, "loss": float(loss.detach()),
               "writer_available_rate": available / args.batch, "top1_event_accuracy": float(hit)}
        history.append(row)
        if step == 1 or step % 10 == 0: print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            path = save(args.output, step, model, optimizer, history, protocol); print(f"checkpoint={path}", flush=True)
    summary = {"status": "complete", "final": history[-1], "checkpoint": str(path)}
    (args.output / "training_results.json").write_text(
        json.dumps({"protocol": protocol, "summary": summary, "history": history}, indent=2), encoding="utf-8")
    print("v0_4_QUERY_ALIGNMENT_COMPLETE", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
