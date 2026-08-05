import argparse
import json

import torch
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer


def make_batch(batch_size, length, vocab_size, device):
    mask, needle, query = vocab_size, vocab_size + 1, vocab_size + 2
    targets = torch.randint(vocab_size, (batch_size,), device=device)
    tokens = torch.randint(vocab_size, (batch_size, length), device=device)
    tokens[:, 0] = needle
    tokens[:, 1] = targets
    tokens[:, -2] = query
    tokens[:, -1] = mask
    return tokens, targets


@torch.no_grad()
def evaluate(model, batches, batch_size, length, vocab_size, device):
    model.eval(); correct = total = 0; loss_sum = 0.0
    for _ in range(batches):
        tokens, targets = make_batch(batch_size, length, vocab_size, device)
        logits = model(tokens)[:, -1, :vocab_size]
        loss_sum += F.cross_entropy(logits, targets).item()
        correct += (logits.argmax(-1) == targets).sum().item(); total += batch_size
    return loss_sum / batches, correct / total


def main():
    parser = argparse.ArgumentParser(description="Level 1 fixed marked retrieval")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--position-encoding", default="absolute",
                        choices=["absolute", "sinusoidal", "rope", "dynamic_rope"])
    parser.add_argument("--output", default="experiments/results/level1_results.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model = InformationSpiralTransformer(
        args.vocab_size + 3, 64, 2, args.length, args.position_encoding
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for step in range(1, args.steps + 1):
        model.train(); tokens, targets = make_batch(
            args.batch_size, args.length, args.vocab_size, device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)[:, -1, :args.vocab_size]
        task_loss = F.cross_entropy(logits, targets)
        loss = task_loss + 0.1 * model.memory_diversity_loss()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            accuracy = (logits.argmax(-1) == targets).float().mean().item()
            history.append({"step": step, "loss": task_loss.item(), "accuracy": accuracy})
            print(f"step={step:03d} task_loss={task_loss.item():.4f} accuracy={accuracy:.2%}")
    validation_loss, validation_accuracy = evaluate(
        model, 20, args.batch_size, args.length, args.vocab_size, device
    )
    passed = history[-1]["accuracy"] >= 0.99 and validation_accuracy >= 0.95
    result = {"config": vars(args), "history": history,
              "validation_loss": validation_loss,
              "validation_accuracy": validation_accuracy, "passed": passed}
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(f"validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.2%}")
    print("LEVEL1_PASS" if passed else "LEVEL1_FAIL")


if __name__ == "__main__":
    main()
