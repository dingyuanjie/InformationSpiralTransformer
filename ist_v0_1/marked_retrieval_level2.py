import argparse
import json

import torch
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer


def make_batch(batch_size, length, needle_range, vocab_size, device):
    mask, needle, query = vocab_size, vocab_size + 1, vocab_size + 2
    targets = torch.randint(vocab_size, (batch_size,), device=device)
    tokens = torch.randint(vocab_size, (batch_size, length), device=device)
    positions = torch.randint(0, min(needle_range, length - 3), (batch_size,), device=device)
    rows = torch.arange(batch_size, device=device)
    tokens[rows, positions] = needle
    tokens[rows, positions + 1] = targets
    tokens[:, -2] = query
    tokens[:, -1] = mask
    return tokens, targets, positions


def losses(model, tokens, targets, positions, vocab_size):
    logits = model(tokens)[..., :vocab_size]
    rows = torch.arange(tokens.size(0), device=tokens.device)
    query_loss = F.cross_entropy(logits[:, -1], targets)
    local_loss = F.cross_entropy(logits[rows, positions], targets)
    predictions = logits[:, -1].argmax(-1)
    return query_loss, local_loss, predictions


@torch.no_grad()
def evaluate(model, args, device):
    model.eval(); query_correct = local_correct = total = 0
    query_loss_sum = local_loss_sum = 0.0
    for _ in range(args.validation_batches):
        tokens, targets, positions = make_batch(
            args.batch_size, args.length, args.needle_range, args.vocab_size, device
        )
        logits = model(tokens)[..., :args.vocab_size]
        rows = torch.arange(args.batch_size, device=device)
        query_loss_sum += F.cross_entropy(logits[:, -1], targets).item()
        local_loss_sum += F.cross_entropy(logits[rows, positions], targets).item()
        query_correct += (logits[:, -1].argmax(-1) == targets).sum().item()
        local_correct += (logits[rows, positions].argmax(-1) == targets).sum().item()
        total += args.batch_size
    return {"query_loss": query_loss_sum / args.validation_batches,
            "local_loss": local_loss_sum / args.validation_batches,
            "query_accuracy": query_correct / total,
            "local_accuracy": local_correct / total}


def main():
    parser = argparse.ArgumentParser(description="Level 2 local random marked retrieval")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--needle-range", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--position-encoding", default="absolute",
                        choices=["absolute", "sinusoidal", "rope", "dynamic_rope"])
    parser.add_argument("--output", default="experiments/results/level2_results.json")
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model = InformationSpiralTransformer(args.vocab_size + 3, 64, args.layers, args.length,
                                         args.position_encoding).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); history = []
    for step in range(1, args.steps + 1):
        model.train(); tokens, targets, positions = make_batch(
            args.batch_size, args.length, args.needle_range, args.vocab_size, device)
        optimizer.zero_grad(set_to_none=True)
        query_loss, local_loss, predictions = losses(
            model, tokens, targets, positions, args.vocab_size)
        loss = query_loss + 0.5 * local_loss + 0.1 * model.memory_diversity_loss()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 50 == 0:
            accuracy = (predictions == targets).float().mean().item()
            history.append({"step": step, "query_loss": query_loss.item(),
                            "local_loss": local_loss.item(), "query_accuracy": accuracy})
            print(f"step={step:03d} query_loss={query_loss.item():.4f} "
                  f"local_loss={local_loss.item():.4f} query_acc={accuracy:.2%}")
    validation = evaluate(model, args, device)
    passed = validation["query_accuracy"] >= 0.90
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump({"config": vars(args), "history": history,
                   "validation": validation, "passed": passed}, output, indent=2)
    print("validation", validation); print("LEVEL2_PASS" if passed else "LEVEL2_FAIL")


if __name__ == "__main__": main()
