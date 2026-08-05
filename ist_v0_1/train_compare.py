import argparse
import json
import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from baseline_transformer import StandardTransformer
from model import InformationSpiralTransformer


@dataclass
class Metrics:
    model: str
    parameters: int
    best_validation_loss: float
    best_validation_accuracy: float
    training_seconds: float


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_recursive_sequences(count, length, vocab_size, seed):
    """Create x[t] = (x[t-1] + x[t-3]) mod vocab_size sequences."""
    generator = torch.Generator().manual_seed(seed)
    sequences = torch.randint(vocab_size, (count, length), generator=generator)
    for index in range(3, length):
        sequences[:, index] = (
            sequences[:, index - 1] + sequences[:, index - 3]
        ) % vocab_size
    return sequences


def mask_batch(sequences, mask_token, mask_rate, generator):
    mask = torch.rand(sequences.shape, generator=generator) < mask_rate
    # Every sample must contribute at least one supervised token.
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        random_positions = torch.randint(
            sequences.size(1),
            (int(empty_rows.sum()),),
            generator=generator,
        )
        mask[empty_rows, random_positions] = True
    inputs = sequences.clone()
    inputs[mask] = mask_token
    return inputs, mask


def batches(sequences, batch_size, shuffle, generator):
    indices = torch.arange(len(sequences))
    if shuffle:
        indices = indices[torch.randperm(len(indices), generator=generator)]
    for start in range(0, len(indices), batch_size):
        yield sequences[indices[start : start + batch_size]]


@torch.no_grad()
def evaluate(model, sequences, args, device, seed):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for targets in batches(sequences, args.batch_size, False, generator):
        inputs, mask = mask_batch(
            targets, args.vocab_size, args.mask_rate, generator
        )
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits[mask], targets[mask], reduction="sum")
        total_loss += loss.item()
        total_correct += (logits[mask].argmax(-1) == targets[mask]).sum().item()
        total_tokens += mask.sum().item()
    return total_loss / total_tokens, total_correct / total_tokens


def train_one(name, model, train_data, validation_data, args, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed + 100)
    best_loss = float("inf")
    best_accuracy = 0.0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        token_count = 0
        for targets in batches(train_data, args.batch_size, True, generator):
            inputs, mask = mask_batch(
                targets, args.vocab_size, args.mask_rate, generator
            )
            inputs, targets, mask = (
                inputs.to(device),
                targets.to(device),
                mask.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits[mask], targets[mask])
            if hasattr(model, "memory_diversity_loss"):
                loss = loss + args.diversity_weight * model.memory_diversity_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            supervised = mask.sum().item()
            loss_sum += loss.item() * supervised
            token_count += supervised

        validation_loss, validation_accuracy = evaluate(
            model, validation_data, args, device, args.seed + 1000
        )
        best_loss = min(best_loss, validation_loss)
        best_accuracy = max(best_accuracy, validation_accuracy)
        print(
            f"{name:12s} epoch={epoch:02d} "
            f"train_loss={loss_sum / token_count:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.2%}"
        )

    return Metrics(
        model=name,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        best_validation_loss=best_loss,
        best_validation_accuracy=best_accuracy,
        training_seconds=time.perf_counter() - started,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train IST v0.3 and compare it with a standard Transformer."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mask-rate", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    parser.add_argument(
        "--position-encoding",
        choices=("rope", "scaled_rope", "dynamic_rope", "sinusoidal", "absolute"),
        default="sinusoidal",
    )
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="experiments/results/comparison_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hidden_size % 8:
        raise ValueError("--hidden-size must be divisible by 8")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    set_seed(args.seed)
    train_data = make_recursive_sequences(
        args.train_samples, args.sequence_length, args.vocab_size, args.seed
    )
    validation_data = make_recursive_sequences(
        args.validation_samples,
        args.sequence_length,
        args.vocab_size,
        args.seed + 1,
    )
    model_vocab_size = args.vocab_size + 1  # One additional [MASK] token.
    constructors = [
        (
            "IST-v0.3",
            lambda: InformationSpiralTransformer(
                model_vocab_size,
                args.hidden_size,
                args.layers,
                args.sequence_length,
                args.position_encoding,
            ),
        ),
        (
            "Transformer",
            lambda: StandardTransformer(
                model_vocab_size,
                args.hidden_size,
                args.layers,
                max_sequence_length=args.sequence_length,
                position_encoding=args.position_encoding,
            ),
        ),
    ]
    results = []
    print(f"device={device} task=recursive-masked-token-recovery")
    for name, constructor in constructors:
        set_seed(args.seed)
        results.append(
            train_one(
                name,
                constructor(),
                train_data,
                validation_data,
                args,
                device,
            )
        )

    payload = {"config": vars(args), "results": [asdict(item) for item in results]}
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    print("\nSummary")
    for item in results:
        print(
            f"{item.model:12s} params={item.parameters:,} "
            f"best_loss={item.best_validation_loss:.4f} "
            f"best_acc={item.best_validation_accuracy:.2%} "
            f"seconds={item.training_seconds:.1f}"
        )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
