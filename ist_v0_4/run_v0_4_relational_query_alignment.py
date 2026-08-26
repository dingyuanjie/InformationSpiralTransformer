"""Milestone 2.2.2: retrain Query on relation-complete overlapping events."""
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
from run_v0_4_query_alignment import adapter_state, latest, save, left_pad


TRAIN_ENTITIES = ["Amina Rho", "Basil North", "Celine Ardent", "Dorian Pike",
                  "Elara Stone", "Felix Rowan", "Greta Sol", "Hector Vale",
                  "Iris Dawn", "Jonas Reed", "Kara Flint", "Lucan Shore"]
HELDOUT_ENTITIES = ["Maren Quill", "Nolan Crest", "Opal Wren", "Pavel Hart"]
TRAIN_QUERIES = ["Which verification token belongs to {entity}? Answer:",
                 "Retrieve the private code assigned to {entity}. Answer:"]
HELDOUT_QUERIES = ["According to the registry, identify {entity} using token:"]


def make_example(tokenizer, chunks, chunk_size, seed, entities, answer_ids, queries):
    rng = random.Random(seed); total = chunks * chunk_size
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    stream = (filler * (total // len(filler) + 1))[:total]
    chosen_entities = rng.sample(entities, 4); chosen_answers = rng.sample(answer_ids, 4)
    target = rng.randrange(4); occupied = set(); required = None
    for index, (entity, answer_id) in enumerate(zip(chosen_entities, chosen_answers)):
        prefix = tokenizer.encode(" Audit registry states that ", add_special_tokens=False)
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        relation = tokenizer.encode(" has private verification token ", add_special_tokens=False)
        suffix = tokenizer.encode(". End record.", add_special_tokens=False)
        fact = prefix + entity_ids + relation + [answer_id] + suffix
        for _ in range(256):
            start = rng.randrange(0, total - len(fact))
            span = set(range(start, start + len(fact)))
            if not span.intersection(occupied): break
        occupied.update(span); stream[start:start + len(fact)] = fact
        if index == target:
            entity_positions = set(range(start + len(prefix), start + len(prefix) + len(entity_ids)))
            answer_position = start + len(prefix) + len(entity_ids) + len(relation)
            required = entity_positions | {answer_position}
    query = tokenizer.encode(rng.choice(queries).format(entity=chosen_entities[target]), add_special_tokens=False)
    return torch.tensor(stream), torch.tensor(query), required


def target_event_mask(positions, required):
    mask = torch.ones(positions.size(0), dtype=torch.bool, device=positions.device)
    for position in required: mask &= (positions == position).any(-1)
    return mask & (positions >= 0).any(-1)


def relation_loss(scores, positions, required_sets):
    scores = scores[:, -1]; losses = []; available = 0
    for row, required in enumerate(required_sets):
        valid = (positions[row] >= 0).any(-1)
        targets = target_event_mask(positions[row], required)
        if targets.any():
            available += 1
            losses.append(torch.logsumexp(scores[row, valid], 0) - torch.logsumexp(scores[row, targets], 0))
    finite = scores[torch.isfinite(scores)]
    zero = finite.sum() * 0 if finite.numel() else scores.new_zeros(())
    return (torch.stack(losses).mean() if losses else zero), available


@torch.no_grad()
def validate(model, tokenizer, device, args, answer_ids):
    rows = []; pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for chunks in args.validation_chunks:
        hits = available = examples_count = 0
        for start in range(0, args.validation_samples, args.batch):
            batch_size = min(args.batch, args.validation_samples - start)
            examples = [make_example(tokenizer, chunks, args.chunk_size,
                                     args.seed + 900000 + chunks * 1000 + start + row,
                                     HELDOUT_ENTITIES, answer_ids, HELDOUT_QUERIES)
                        for row in range(batch_size)]
            streams = torch.stack([item[0] for item in examples]).to(device)
            queries = left_pad([item[1] for item in examples], pad).to(device)
            required = [item[2] for item in examples]; state = None
            for chunk in range(chunks):
                begin = chunk * args.chunk_size
                _, state = model(streams[:, begin:begin + args.chunk_size], state, chunk, begin, "zero", True)
            model(queries, state, chunks, chunks * args.chunk_size, "normal", True)
            scores, positions = model.memory.last_event_scores[:, -1], model.memory.last_event_positions
            top = scores.argmax(-1)
            for row in range(batch_size):
                mask = target_event_mask(positions[row], required[row]); available += bool(mask.any())
                hits += bool(mask[top[row]]) if mask.any() else False; examples_count += 1
        result = {"chunks": chunks, "examples": examples_count,
                  "writer_relation_availability": available / examples_count,
                  "top1_relation_accuracy": hits / examples_count}
        rows.append(result); print("validation=" + json.dumps(result), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--curriculum-chunks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--validation-chunks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--train-token-count", type=int, default=128)
    parser.add_argument("--heldout-token-count", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--seed", type=int, default=6404)
    parser.add_argument("--output", type=Path, default=Path("experiments/relational_query_alignment"))
    parser.add_argument("--log", type=Path, default=Path("experiments/relational_query_alignment/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("a" if args.resume else "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, indent=2)); return 0
    torch.manual_seed(args.seed); random.seed(args.seed)
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    pool = open_tokens(tokenizer, args.train_token_count + args.heldout_token_count)
    train_ids = [item[0] for item in pool[:args.train_token_count]]
    heldout_ids = [item[0] for item in pool[args.train_token_count:]]
    config = CognitiveMemoryConfig(event_span=24, event_stride=8, working_events=4,
                                   episodic_events=args.capacity, semantic_slots=8,
                                   admissions_per_chunk=16, retrieved_events=3,
                                   redundancy_weight=0.25)
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
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id; path = None
    for step in range(start + 1, args.steps + 1):
        chunks = random.choice(args.curriculum_chunks)
        examples = [make_example(tokenizer, chunks, args.chunk_size, args.seed + step * args.batch + row,
                                 TRAIN_ENTITIES, train_ids, TRAIN_QUERIES) for row in range(args.batch)]
        streams = torch.stack([item[0] for item in examples]).to(device)
        queries = left_pad([item[1] for item in examples], pad).to(device)
        required = [item[2] for item in examples]; state = None
        with torch.no_grad():
            for chunk in range(chunks):
                begin = chunk * args.chunk_size
                _, state = model(streams[:, begin:begin + args.chunk_size], state, chunk, begin, "zero", True)
        model(queries, state, chunks, chunks * args.chunk_size, "normal", True)
        loss, available = relation_loss(model.memory.last_event_scores, model.memory.last_event_positions, required)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
        with torch.no_grad():
            top = model.memory.last_event_scores[:, -1].argmax(-1)
            hit = sum(bool(target_event_mask(model.memory.last_event_positions[row], required[row])[top[row]])
                      for row in range(args.batch)) / args.batch
        row = {"step": step, "chunks": chunks, "loss": float(loss.detach()),
               "writer_relation_availability": available / args.batch, "top1_relation_accuracy": hit}
        history.append(row)
        if step == 1 or step % 10 == 0: print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            path = save(args.output, step, model, optimizer, history, protocol); print(f"checkpoint={path}", flush=True)
    validation = validate(model, tokenizer, device, args, heldout_ids)
    summary = {"status": "complete", "final": history[-1], "checkpoint": str(path), "validation": validation}
    (args.output / "training_results.json").write_text(
        json.dumps({"protocol": protocol, "summary": summary, "history": history}, indent=2), encoding="utf-8")
    print("v0_4_RELATIONAL_QUERY_ALIGNMENT_COMPLETE", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
