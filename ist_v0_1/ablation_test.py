import argparse
import json
from types import SimpleNamespace

import torch

from long_context_test import evaluate_length, set_seed, train
from model import InformationSpiralTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="IST v0.3 position/diversity ablation")
    parser.add_argument("--position-encodings", nargs="+", default=["rope", "absolute"])
    parser.add_argument(
        "--diversity-weights", type=float, nargs="+",
        default=[0.0, 0.001, 0.01, 0.05, 0.1],
    )
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--test-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="experiments/results/ablation_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    maximum_length = max(args.train_lengths + args.test_lengths)
    results = []
    for encoding in args.position_encodings:
        if encoding not in ("rope", "absolute"):
            raise ValueError(f"unsupported position encoding: {encoding}")
        for weight in args.diversity_weights:
            print(f"\nAblation position={encoding} diversity_weight={weight}")
            set_seed(args.seed)
            model = InformationSpiralTransformer(
                args.vocab_size + 1,
                args.hidden_size,
                args.layers,
                maximum_length,
                encoding,
            ).to(device)
            train_args = SimpleNamespace(**vars(args))
            train_args.diversity_weight = weight
            train_args.log_every = args.train_steps
            training_seconds = train(model, train_args, device)
            measurements = [
                evaluate_length(model, length, train_args, device)
                for length in args.test_lengths
            ]
            score = sum(item["accuracy"] for item in measurements) / len(measurements)
            slot_similarity = sum(
                block.memory.last_diagnostics["diversity_loss"].item()
                for block in model.blocks
            ) / len(model.blocks)
            result = {
                "position_encoding": encoding,
                "diversity_weight": weight,
                "mean_long_context_accuracy": score,
                "slot_diversity_loss": slot_similarity,
                "training_seconds": training_seconds,
                "measurements": measurements,
            }
            results.append(result)
            print(
                f"mean_accuracy={score:.2%} diversity_loss={slot_similarity:.4f}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Prefer lower slot redundancy when task accuracy is tied.
    results.sort(
        key=lambda item: (
            item["mean_long_context_accuracy"],
            -item["slot_diversity_loss"],
        ),
        reverse=True,
    )
    payload = {"config": vars(args), "best": results[0], "results": results}
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    best = results[0]
    print(
        f"\nBEST position={best['position_encoding']} "
        f"weight={best['diversity_weight']} "
        f"mean_accuracy={best['mean_long_context_accuracy']:.2%}"
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
