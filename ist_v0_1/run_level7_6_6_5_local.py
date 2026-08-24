"""Level 7.6.6.5: held-out 32K Memory-content decodability audit."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level7_1_local import atomic_save
from run_level7_4_1_local import atomic_torch_save


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
LENGTH = 32768
WINDOWS = {"near": (16, 4095), "far": (16384, 32766)}
SEEDS = (313, 42, 2026, 7, 1234)
HIGH_SEEDS = (2026, 7)
LOW_SEEDS = (313, 1234)
SAMPLES_PER_WINDOW = 160
RIDGE = 1.0


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, LENGTH, "rope", True).to(device).eval()
    checkpoint = PARENT / f"ist-full_seed{seed}" / "stage_4096.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return model


def make_example(window: tuple[int, int], sample: int, seed: int, device: torch.device):
    set_seed(seed)
    target_value = sample % 16
    target = torch.tensor([target_value], device=device)
    tokens = torch.randint(16, (1, LENGTH), device=device)
    distance = int(torch.randint(window[0], window[1] + 1, (1,), device=device).item())
    position = LENGTH - 2 - distance
    tokens[0, position] = 17
    tokens[0, position + 1] = target
    tokens[0, -2] = 18
    tokens[0, -1] = 16
    return tokens, target


@torch.no_grad()
def capture(seed: int, device: torch.device, dtype: torch.dtype) -> dict:
    model = build(seed, device)
    features = []
    labels = []
    windows = []
    correctness = []
    for window_index, (window_name, window) in enumerate(WINDOWS.items()):
        for sample in range(SAMPLES_PER_WINDOW):
            example_seed = 766500000 + seed * 1000 + window_index * SAMPLES_PER_WINDOW + sample
            tokens, target = make_example(window, sample, example_seed, device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(tokens)
            states = torch.stack([block.memory.last_diagnostics["new_memory"][0]
                                  for block in model.blocks]).to(device="cpu", dtype=torch.float16)
            features.append(states)
            labels.append(int(target.item()))
            windows.append(window_index)
            correctness.append(int(logits[..., :16][:, -1].argmax(-1).item() == target.item()))
            if (sample + 1) % 20 == 0:
                print(f"capture seed={seed} window={window_name} sample={sample + 1}/{SAMPLES_PER_WINDOW}", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"seed": seed, "features": torch.stack(features),
            "labels": torch.tensor(labels, dtype=torch.long),
            "window_ids": torch.tensor(windows, dtype=torch.long),
            "correctness": torch.tensor(correctness, dtype=torch.uint8)}


def stratified_split(labels: torch.Tensor, seed: int):
    rng = random.Random(766510000 + seed)
    train, validation, test = [], [], []
    for label in range(16):
        indices = torch.where(labels == label)[0].tolist()
        rng.shuffle(indices)
        # 20 examples/class -> 12 train, 4 validation, 4 test.
        train.extend(indices[:12])
        validation.extend(indices[12:16])
        test.extend(indices[16:20])
    rng.shuffle(train); rng.shuffle(validation); rng.shuffle(test)
    return {"train": torch.tensor(train), "validation": torch.tensor(validation),
            "test": torch.tensor(test)}


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float = RIDGE):
    x = x.float()
    mean = x.mean(0, keepdim=True)
    scale = x.std(0, keepdim=True).clamp_min(1e-5)
    z = (x - mean) / scale
    z = torch.cat([z, torch.ones(len(z), 1)], dim=1)
    targets = torch.nn.functional.one_hot(y, 16).float()
    if z.size(1) <= z.size(0):
        weight = torch.linalg.solve(z.T @ z + ridge * torch.eye(z.size(1)), z.T @ targets)
    else:
        weight = z.T @ torch.linalg.solve(z @ z.T + ridge * torch.eye(z.size(0)), targets)
    return mean, scale, weight


def ridge_predict(model, x: torch.Tensor) -> torch.Tensor:
    mean, scale, weight = model
    z = (x.float() - mean) / scale
    z = torch.cat([z, torch.ones(len(z), 1)], dim=1)
    return (z @ weight).argmax(-1)


def accuracy(prediction: torch.Tensor, labels: torch.Tensor) -> float:
    return float((prediction == labels).float().mean())


class ProbeMLP(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(size, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 16))

    def forward(self, x):
        return self.network(x)


def fit_mlp(x: torch.Tensor, y: torch.Tensor, split: dict, seed: int,
            device: torch.device) -> dict:
    torch.manual_seed(766520000 + seed)
    train_x = x[split["train"]].float()
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    model = ProbeMLP(x.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best_accuracy = -1.0
    best_state = None
    patience = 0
    for epoch in range(150):
        model.train()
        order = split["train"][torch.randperm(len(split["train"]))]
        for start in range(0, len(order), 64):
            index = order[start:start + 64]
            batch_x = ((x[index].float() - mean) / scale).to(device)
            batch_y = y[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_x = ((x[split["validation"]].float() - mean) / scale).to(device)
            validation_prediction = model(validation_x).argmax(-1).cpu()
        value = accuracy(validation_prediction, y[split["validation"]])
        if value > best_accuracy:
            best_accuracy = value
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 15:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_x = ((x[split["test"]].float() - mean) / scale).to(device)
        test_prediction = model(test_x).argmax(-1).cpu()
    return {"validation_accuracy": best_accuracy,
            "test_accuracy": accuracy(test_prediction, y[split["test"]]),
            "epochs": epoch + 1}


def probe_seed(dataset: dict, device: torch.device) -> dict:
    seed = int(dataset["seed"])
    features = dataset["features"].float()
    labels = dataset["labels"]
    split = stratified_split(labels, seed)
    train = split["train"]
    validation = split["validation"]
    test = split["test"]
    slot_rows = []
    for layer in range(3):
        for slot in range(32):
            x = features[:, layer, slot, :]
            model = fit_ridge(x[train], labels[train])
            slot_rows.append({"layer": layer, "slot": slot,
                              "validation_accuracy": accuracy(ridge_predict(model, x[validation]), labels[validation]),
                              "test_accuracy": accuracy(ridge_predict(model, x[test]), labels[test])})
    selected_slot = max(slot_rows, key=lambda row: row["validation_accuracy"])
    layer_rows = []
    for layer in range(3):
        x = features[:, layer].flatten(1)
        model = fit_ridge(x[train], labels[train])
        layer_rows.append({"layer": layer,
                           "validation_accuracy": accuracy(ridge_predict(model, x[validation]), labels[validation]),
                           "test_accuracy": accuracy(ridge_predict(model, x[test]), labels[test])})
    all_x = features.flatten(1)
    all_model = fit_ridge(all_x[train], labels[train])
    all_linear = {"validation_accuracy": accuracy(ridge_predict(all_model, all_x[validation]), labels[validation]),
                  "test_accuracy": accuracy(ridge_predict(all_model, all_x[test]), labels[test])}
    all_mlp = fit_mlp(all_x, labels, split, seed, device)
    return {"seed": seed, "group": "high" if seed in HIGH_SEEDS else "low" if seed in LOW_SEEDS else "intermediate",
            "samples": len(labels), "split_sizes": {key: len(value) for key, value in split.items()},
            "model_accuracy": float(dataset["correctness"].float().mean()),
            "selected_slot_by_validation": selected_slot,
            "slot_linear": slot_rows, "layer_concat_linear": layer_rows,
            "all_layers_concat_linear": all_linear, "all_layers_mlp": all_mlp}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-seed-window", type=int, default=SAMPLES_PER_WINDOW)
    parser.add_argument("--output", default="experiments/level7_6_6_5/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"variant": "ist-full", "length": LENGTH, "windows": WINDOWS, "seeds": SEEDS,
                "groups": {"high": HIGH_SEEDS, "low": LOW_SEEDS, "intermediate": (42,)},
                "samples_per_seed_window": args.samples_per_seed_window,
                "balanced_labels": True, "split_per_class": {"train": 12, "validation": 4, "test": 4},
                "probes": ["96 slot-linear", "3 layer-concat-linear", "all-concat-linear", "all-concat-MLP"],
                "ridge": RIDGE, "test_set_selection_forbidden": True}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    if args.samples_per_seed_window != SAMPLES_PER_WINDOW:
        raise ValueError(f"Formal protocol locks --samples-per-seed-window={SAMPLES_PER_WINDOW}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol, "gpu": torch.cuda.get_device_name(device),
                                           "torch": torch.__version__, "dtype": str(dtype)})
    results = []
    for seed in SEEDS:
        dataset_path = root / f"seed{seed}_memory_dataset.pt"
        if dataset_path.exists() and not args.force:
            dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
        else:
            dataset = capture(seed, device, dtype)
            atomic_torch_save(dataset_path, dataset)
        probe_path = root / f"seed{seed}_probes.json"
        if probe_path.exists() and not args.force:
            result = json.loads(probe_path.read_text(encoding="utf-8"))
        else:
            result = probe_seed(dataset, device)
            atomic_save(probe_path, result)
        results.append(result)
        atomic_save(root / "runs.partial.json", results)
        print(f"seed={seed} group={result['group']} model={result['model_accuracy']:.2%} "
              f"slot={result['selected_slot_by_validation']['test_accuracy']:.2%} "
              f"linear={result['all_layers_concat_linear']['test_accuracy']:.2%} "
              f"mlp={result['all_layers_mlp']['test_accuracy']:.2%}", flush=True)
    group_summary = []
    for group in ("high", "low", "intermediate"):
        selected = [row for row in results if row["group"] == group]
        group_summary.append({"group": group, "seeds": [row["seed"] for row in selected],
                              "mean_model_accuracy": sum(row["model_accuracy"] for row in selected) / len(selected),
                              "mean_selected_slot_test_accuracy": sum(row["selected_slot_by_validation"]["test_accuracy"] for row in selected) / len(selected),
                              "mean_all_linear_test_accuracy": sum(row["all_layers_concat_linear"]["test_accuracy"] for row in selected) / len(selected),
                              "mean_all_mlp_test_accuracy": sum(row["all_layers_mlp"]["test_accuracy"] for row in selected) / len(selected)})
    result = {"protocol": protocol, "group_summary": group_summary, "seeds": results}
    atomic_save(root / "result.json", result)
    print(json.dumps(group_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
