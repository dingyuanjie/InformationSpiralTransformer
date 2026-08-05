import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from baseline_transformer import StandardTransformer
from model import InformationSpiralTransformer


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_needle_batch(batch_size, sequence_length, vocab_size, device):
    """Put a random target at position 0 and ask for it at the final [MASK]."""
    targets = torch.randint(vocab_size, (batch_size,), device=device)
    tokens = torch.randint(
        vocab_size, (batch_size, sequence_length), device=device
    )
    tokens[:, 0] = targets
    tokens[:, -1] = vocab_size  # [MASK]
    return tokens, targets


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(model, args, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    started = time.perf_counter()
    for step in range(1, args.train_steps + 1):
        length = random.choice(args.train_lengths)
        tokens, targets = make_needle_batch(
            args.batch_size, length, args.vocab_size, device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)[:, -1, : args.vocab_size]
        loss = F.cross_entropy(logits, targets)
        if hasattr(model, "memory_diversity_loss"):
            loss = loss + args.diversity_weight * model.memory_diversity_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            accuracy = (logits.argmax(-1) == targets).float().mean().item()
            print(
                f"step={step:04d}/{args.train_steps} length={length:4d} "
                f"loss={loss.item():.4f} acc={accuracy:.2%}"
            )
    synchronize(device)
    return time.perf_counter() - started


@torch.no_grad()
def evaluate_length(model, length, args, device):
    model.eval()
    correct = 0
    total = 0
    latency_sum = 0.0
    peak_memory = 0

    # Warm up kernels without including them in latency.
    warm_tokens, _ = make_needle_batch(
        min(args.eval_batch_size, 8), length, args.vocab_size, device
    )
    model(warm_tokens)
    synchronize(device)

    for _ in range(args.eval_batches):
        tokens, targets = make_needle_batch(
            args.eval_batch_size, length, args.vocab_size, device
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        logits = model(tokens)[:, -1, : args.vocab_size]
        synchronize(device)
        latency_sum += time.perf_counter() - started
        correct += (logits.argmax(-1) == targets).sum().item()
        total += targets.numel()
        if device.type == "cuda":
            peak_memory = max(
                peak_memory, torch.cuda.max_memory_allocated(device)
            )

    return {
        "length": length,
        "accuracy": correct / total,
        "latency_ms_per_batch": 1000 * latency_sum / args.eval_batches,
        "tokens_per_second": total * length / latency_sum,
        "peak_memory_mb": peak_memory / (1024**2) if peak_memory else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Needle-retrieval long-context test for IST v0.3."
    )
    parser.add_argument("--train-lengths", type=int, nargs="+", default=[32, 64])
    parser.add_argument(
        "--test-lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512]
    )
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    parser.add_argument(
        "--position-encoding",
        choices=("rope", "scaled_rope", "dynamic_rope", "sinusoidal", "absolute"),
        default="sinusoidal",
    )
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="experiments/results/long_context_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hidden_size % 8:
        raise ValueError("--hidden-size must be divisible by 8")
    maximum_length = max(args.train_lengths + args.test_lengths)
    if min(args.train_lengths + args.test_lengths) < 2:
        raise ValueError("all sequence lengths must be at least 2")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    constructors = [
        (
            "IST-v0.3",
            lambda: InformationSpiralTransformer(
                args.vocab_size + 1,
                args.hidden_size,
                args.layers,
                maximum_length,
                args.position_encoding,
            ),
        ),
        (
            "Transformer",
            lambda: StandardTransformer(
                args.vocab_size + 1,
                args.hidden_size,
                args.layers,
                max_sequence_length=maximum_length,
                position_encoding=args.position_encoding,
            ),
        ),
    ]
    results = []
    print(
        f"device={device} train_lengths={args.train_lengths} "
        f"test_lengths={args.test_lengths}"
    )
    for name, constructor in constructors:
        print(f"\nTraining {name}")
        set_seed(args.seed)
        model = constructor().to(device)
        training_seconds = train(model, args, device)
        measurements = []
        for length in args.test_lengths:
            measurement = evaluate_length(model, length, args, device)
            measurements.append(measurement)
            print(
                f"{name:12s} length={length:4d} "
                f"acc={measurement['accuracy']:.2%} "
                f"latency={measurement['latency_ms_per_batch']:.2f}ms "
                f"peak_memory={measurement['peak_memory_mb'] or 0:.1f}MB"
            )
        results.append(
            {
                "model": name,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_seconds": training_seconds,
                "measurements": measurements,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "task": "first-token needle retrieval at final masked position",
        "chance_accuracy": 1 / args.vocab_size,
        "config": vars(args),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    print(f"\nsaved={args.output}")


if __name__ == "__main__":
    main()
