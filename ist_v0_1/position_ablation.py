import argparse
import json
import statistics
from types import SimpleNamespace

import torch

from long_context_test import evaluate_length, set_seed, train
from model import InformationSpiralTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed position encoding ablation")
    parser.add_argument(
        "--encodings", nargs="+",
        default=["absolute", "sinusoidal", "rope", "scaled_rope"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[313, 42, 2026, 7, 1234])
    parser.add_argument("--train-lengths", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--test-lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="experiments/results/position_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    maximum_length = max(args.train_lengths + args.test_lengths)
    grouped = []
    for encoding in args.encodings:
        runs = []
        for seed in args.seeds:
            print(f"\nposition={encoding} seed={seed}")
            set_seed(seed)
            model = InformationSpiralTransformer(
                args.vocab_size + 1, args.hidden_size, args.layers,
                maximum_length, encoding,
            ).to(device)
            run_args = SimpleNamespace(**vars(args), seed=seed, log_every=args.train_steps)
            train(model, run_args, device)
            measurements = [
                evaluate_length(model, length, run_args, device)
                for length in args.test_lengths
            ]
            score = statistics.mean(item["accuracy"] for item in measurements)
            runs.append({"seed": seed, "mean_accuracy": score, "measurements": measurements})
            print(f"mean_long_context_accuracy={score:.2%}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        scores = [run["mean_accuracy"] for run in runs]
        grouped.append({
            "position_encoding": encoding,
            "mean_accuracy": statistics.mean(scores),
            "std_accuracy": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "minimum_accuracy": min(scores),
            "runs": runs,
        })
    grouped.sort(key=lambda item: (item["mean_accuracy"], -item["std_accuracy"]), reverse=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump({"config": vars(args), "best": grouped[0], "results": grouped}, output_file, indent=2)
    for item in grouped:
        print(
            f"{item['position_encoding']:12s} mean={item['mean_accuracy']:.2%} "
            f"std={item['std_accuracy']:.2%} min={item['minimum_accuracy']:.2%}"
        )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
