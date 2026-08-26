"""Milestone 2.3: open-vocabulary copy and entity-conditioned Reader alignment."""
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
from run_v0_3_coverage_gate import DEFAULT_MODEL, Tee, load_model
from run_v0_3_reader_alignment import adapter_state, latest_checkpoint, retrieval_loss, save_checkpoint
from run_v0_3_retrieval_gate import left_pad
from run_v0_3_strict_generalization import FACT_TEMPLATES, FILLER, QUERY_TEMPLATES, non_overlapping_start


TRAIN_ENTITIES = [
    "Amina", "Basil", "Celine", "Dorian", "Elara", "Felix", "Greta", "Hector",
    "Iris", "Jonas", "Kara", "Lucan", "Maren", "Nolan", "Opal", "Pavel",
]


def token_pool(tokenizer, train_size=128, heldout_size=32):
    candidates = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        word = text.strip()
        if 4 <= len(word) <= 10 and word.isascii() and word.isalpha():
            candidates.append((token_id, text))
    # Fixed arithmetic ordering makes the split independent of tokenizer vocab order.
    candidates.sort(key=lambda item: ((item[0] * 2654435761) & 0xFFFFFFFF, item[0]))
    required = train_size + heldout_size
    if len(candidates) < required:
        raise RuntimeError(f"only {len(candidates)} suitable single tokens; need {required}")
    return candidates[:train_size], candidates[train_size:required]


def fact_tokens(tokenizer, entity, answer_id, template_index):
    before, after = FACT_TEMPLATES[template_index % len(FACT_TEMPLATES)]
    prefix = tokenizer.encode(before.format(entity=entity), add_special_tokens=False)
    suffix = tokenizer.encode(after.format(entity=entity), add_special_tokens=False)
    return prefix + [answer_id] + suffix, len(prefix)


def make_example(tokenizer, chunk_size, chunks, seed, answer_ids):
    rng = random.Random(seed)
    total = chunk_size * chunks
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    stream = (filler * (total // len(filler) + 1))[:total]
    entities = rng.sample(TRAIN_ENTITIES, 4)
    tokens = rng.sample(answer_ids, 4)
    target_index = rng.randrange(4)
    occupied = set()
    target_position = None
    for index, (entity, answer_id) in enumerate(zip(entities, tokens)):
        fact, offset = fact_tokens(tokenizer, entity, answer_id, seed + index)
        start = non_overlapping_start(rng, total, len(fact), occupied)
        stream[start:start + len(fact)] = fact
        if index == target_index:
            target_position = start + offset
    query = tokenizer.encode(
        QUERY_TEMPLATES[(seed // 7) % len(QUERY_TEMPLATES)].format(entity=entities[target_index]),
        add_special_tokens=False,
    )
    return torch.tensor(stream), torch.tensor(query), {target_position}, tokens[target_index]


def make_batch(tokenizer, args, chunks, seed, answer_ids, device):
    examples = [make_example(tokenizer, args.chunk_size, chunks, seed + row, answer_ids)
                for row in range(args.batch)]
    # Ensure swap is a genuine wrong-answer intervention.
    if args.batch > 1 and examples[0][3] == examples[1][3]:
        examples[1] = make_example(tokenizer, args.chunk_size, chunks, seed + 100003, answer_ids)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return (
        torch.stack([item[0] for item in examples]).to(device),
        left_pad([item[1] for item in examples], pad).to(device),
        [item[2] for item in examples],
        torch.tensor([item[3] for item in examples], device=device),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--initial-checkpoint", type=Path, default=Path("experiments/reader_alignment/reader_step_000400.pt"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--curriculum-chunks", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--writes-per-chunk", type=int, default=8)
    parser.add_argument("--reads-per-query", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--train-token-count", type=int, default=128)
    parser.add_argument("--heldout-token-count", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--retrieval-weight", type=float, default=0.5)
    parser.add_argument("--causal-weight", type=float, default=0.25)
    parser.add_argument("--causal-margin", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5303)
    parser.add_argument("--output", type=Path, default=Path("experiments/open_copy_alignment"))
    parser.add_argument("--log", type=Path, default=Path("experiments/open_copy_alignment/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("a" if args.resume else "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2)); return 0
    torch.manual_seed(args.seed); random.seed(args.seed)
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    train_pool, heldout_pool = token_pool(tokenizer, args.train_token_count, args.heldout_token_count)
    train_ids = [item[0] for item in train_pool]
    config = SourceTokenMemoryConfig(capacity=args.capacity, writes_per_chunk=args.writes_per_chunk,
                                     reads_per_query=args.reads_per_query, heads=args.heads,
                                     injection_layer=args.injection_layer)
    model = FrozenTokenMemoryIST(backbone, config).to(device)
    model.memory.salience.weight.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    start, history = 0, []
    resume_checkpoint = latest_checkpoint(args.output) if args.resume else None
    if resume_checkpoint:
        payload = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["adapter"], strict=False)
        optimizer.load_state_dict(payload["optimizer"])
        start, history = int(payload["step"]), payload["history"]
        print(f"resumed={resume_checkpoint} step={start}", flush=True)
    else:
        payload = torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["adapter"], strict=False)
        print(f"initialized={args.initial_checkpoint}", flush=True)
    protocol["train_token_ids"] = train_ids
    protocol["heldout_token_ids"] = [item[0] for item in heldout_pool]
    protocol["heldout_token_text"] = [item[1] for item in heldout_pool]
    path = None
    for step in range(start + 1, args.steps + 1):
        chunks = random.choice(args.curriculum_chunks)
        streams, queries, fact_positions, target_ids = make_batch(
            tokenizer, args, chunks, args.seed + step * args.batch, train_ids, device
        )
        state = None
        with torch.no_grad():
            for chunk_id in range(chunks):
                begin = chunk_id * args.chunk_size
                _, state = model(streams[:, begin:begin + args.chunk_size], state,
                                 chunk_id, begin, "zero", True)
        normal_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, "normal", True)
        read_scores, read_positions = model.memory.last_read_scores, model.memory.last_read_positions
        answer_loss = F.cross_entropy(normal_logits[:, -1].float(), target_ids)
        retrieve_loss = retrieval_loss(read_scores, read_positions, fact_positions)
        condition = "swap" if step % 2 == 0 else "zero"
        intervened_logits, _ = model(queries, state, chunks, chunks * args.chunk_size, condition, True)
        rows = torch.arange(args.batch, device=device)
        normal_target = normal_logits[rows, -1, target_ids].float()
        intervened_target = intervened_logits[rows, -1, target_ids].float()
        causal_loss = F.relu(args.causal_margin - (normal_target - intervened_target)).mean()
        loss = answer_loss + args.retrieval_weight * retrieve_loss + args.causal_weight * causal_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
        prediction = normal_logits[:, -1].argmax(-1)
        row = {"step": step, "chunks": chunks, "loss": float(loss.detach()),
               "answer_loss": float(answer_loss.detach()), "retrieval_loss": float(retrieve_loss.detach()),
               "causal_loss": float(causal_loss.detach()),
               "exact_token_accuracy": float((prediction == target_ids).float().mean()),
               "gate": float(torch.tanh(model.injection_gate.detach().float()))}
        history.append(row)
        if step == 1 or step % 10 == 0: print(json.dumps(row, ensure_ascii=False), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            path = save_checkpoint(args.output, step, model, optimizer, history, protocol)
            print(f"checkpoint={path}", flush=True)
    summary = {"status": "complete", "final": history[-1], "checkpoint": str(path)}
    (args.output / "training_results.json").write_text(
        json.dumps({"protocol": protocol, "summary": summary, "history": history}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("v0_3_OPEN_COPY_ALIGNMENT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
