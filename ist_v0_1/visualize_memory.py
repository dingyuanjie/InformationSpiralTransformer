import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from long_context_test import make_needle_batch, set_seed, train
from model import InformationSpiralTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Spiral Memory compression, slots and update gates."
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=16)
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
    parser.add_argument("--output", default="experiments/results/memory_visualization.png")
    parser.add_argument("--data-output", default="experiments/results/memory_visualization.json")
    return parser.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def collect_diagnostics(model, args, device):
    model.eval()
    tokens, targets = make_needle_batch(
        args.sample_batch_size,
        args.sequence_length,
        args.vocab_size,
        device,
    )
    logits = model(tokens)[:, -1, : args.vocab_size]
    accuracy = (logits.argmax(-1) == targets).float().mean().item()
    diagnostics = []
    for layer_index, block in enumerate(model.blocks):
        item = block.memory.last_diagnostics
        memory = item["new_memory"].float()
        normalized = F.normalize(memory, dim=-1)
        similarity = normalized @ normalized.transpose(-1, -2)
        slots = similarity.size(-1)
        off_diagonal = (
            similarity.sum(dim=(-1, -2)) - slots
        ) / max(slots * (slots - 1), 1)
        diagnostics.append(
            {
                "layer": layer_index + 1,
                "compression_weights": item["compression_weights"]
                .float()
                .mean(dim=0)
                .cpu(),
                "slot_norms": memory.norm(dim=-1).mean(dim=0).cpu(),
                "gate_mean": item["update_gate"].float().mean().item(),
                "gate_std": item["update_gate"].float().std().item(),
                "slot_cosine_similarity": off_diagonal.mean().item(),
                "attention_entropy": item["attention_entropy"].float().mean().item(),
                "effective_context_tokens": item["attention_entropy"]
                .float()
                .exp()
                .mean()
                .item(),
                "diversity_loss": item["diversity_loss"].float().item(),
            }
        )
    return accuracy, diagnostics


def render(diagnostics, accuracy, output_path, sequence_length):
    layer_count = len(diagnostics)
    slot_count = len(diagnostics[0]["slot_norms"])
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
        f"Spiral Memory diagnostics | context={sequence_length} | "
        f"retrieval accuracy={accuracy:.1%}",
        fontsize=14,
    )

    compression = diagnostics[-1]["compression_weights"].numpy()
    image = axes[0, 0].imshow(compression, aspect="auto", cmap="viridis")
    axes[0, 0].set_title(f"Layer {layer_count}: compression attention")
    axes[0, 0].set_xlabel("Input token position")
    axes[0, 0].set_ylabel("Memory slot")
    figure.colorbar(image, ax=axes[0, 0], label="Mean attention weight")

    norms = torch.stack([item["slot_norms"] for item in diagnostics]).numpy()
    image = axes[0, 1].imshow(norms, aspect="auto", cmap="magma")
    axes[0, 1].set_title("Memory slot activation")
    axes[0, 1].set_xlabel("Memory slot")
    axes[0, 1].set_ylabel("Layer")
    axes[0, 1].set_yticks(range(layer_count), range(1, layer_count + 1))
    figure.colorbar(image, ax=axes[0, 1], label="Mean L2 norm")

    layers = [item["layer"] for item in diagnostics]
    gate_means = [item["gate_mean"] for item in diagnostics]
    gate_stds = [item["gate_std"] for item in diagnostics]
    axes[1, 0].bar(layers, gate_means, yerr=gate_stds, capsize=5)
    axes[1, 0].set_title("Memory update gate")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Gate value (1 = replace, 0 = retain)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xticks(layers)
    axes[1, 0].grid(axis="y", alpha=0.25)

    similarities = [item["slot_cosine_similarity"] for item in diagnostics]
    axes[1, 1].plot(layers, similarities, marker="o", linewidth=2)
    axes[1, 1].set_title("Average similarity between memory slots")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Off-diagonal cosine similarity")
    axes[1, 1].set_ylim(-1, 1)
    axes[1, 1].set_xticks(layers)
    axes[1, 1].axhline(0, color="gray", linewidth=1)
    axes[1, 1].grid(alpha=0.25)

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.hidden_size % 8:
        raise ValueError("--hidden-size must be divisible by 8")
    maximum_length = max([args.sequence_length] + args.train_lengths)
    device = choose_device(args.device)
    set_seed(args.seed)
    model = InformationSpiralTransformer(
        args.vocab_size + 1,
        args.hidden_size,
        args.layers,
        maximum_length,
        args.position_encoding,
    ).to(device)
    print(f"device={device} training model before visualization")
    train(model, args, device)
    accuracy, diagnostics = collect_diagnostics(model, args, device)
    render(diagnostics, accuracy, args.output, args.sequence_length)

    serializable = {
        "sequence_length": args.sequence_length,
        "retrieval_accuracy": accuracy,
        "layers": [
            {
                "layer": item["layer"],
                "gate_mean": item["gate_mean"],
                "gate_std": item["gate_std"],
                "slot_cosine_similarity": item["slot_cosine_similarity"],
                "attention_entropy": item["attention_entropy"],
                "effective_context_tokens": item["effective_context_tokens"],
                "diversity_loss": item["diversity_loss"],
                "slot_norms": item["slot_norms"].tolist(),
            }
            for item in diagnostics
        ],
    }
    Path(args.data_output).write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved={args.output}")
    print(f"saved={args.data_output}")


if __name__ == "__main__":
    main()
