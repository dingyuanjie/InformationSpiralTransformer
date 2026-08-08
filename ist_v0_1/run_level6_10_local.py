import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks, vector


SEEDS = [606, 808, 1001]


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@torch.no_grad()
def collect(model, original_probe, count, samples, batch_size, chunk_size, device, dtype, seed):
    set_seed(seed); model.eval(); original_probe.eval()
    states = []; labels = []; query = probe = total = 0
    while total < samples:
        batch = min(batch_size, samples - total)
        chunks, target, _ = make_chunks(batch, count, chunk_size, device)
        memory = None
        with torch.autocast(device_type="cuda", dtype=dtype):
            for index in range(count):
                logits, memory = model(chunks[:, index], memory=memory, return_memory=True,
                                       per_layer_memory=True)
            original = original_probe(vector(memory))
        states.append(torch.stack(memory, dim=1).detach().to("cpu", torch.float16))
        labels.append(target.cpu())
        query += (logits[:, -1, :16].argmax(-1) == target).sum().item()
        probe += (original.argmax(-1) == target).sum().item(); total += batch
    return torch.cat(states), torch.cat(labels), {"query": query / total,
                                                   "original_probe": probe / total,
                                                   "samples": total}


def accuracy(logits, labels):
    # logits: [N, K, C]
    return (logits.argmax(-1) == labels[:, None]).float().mean(dim=0)


def fit_linear(train_x, train_y, val_x, val_y, test_x, test_y, args, device, seed):
    # X is [N, K, F]; all K independent linear probes train in parallel.
    set_seed(seed); features = train_x.shape[-1]; heads = train_x.shape[1]
    mean = train_x.float().mean(dim=0).to(device)
    std = train_x.float().std(dim=0).clamp_min(1e-4).to(device)
    weight = nn.Parameter(torch.empty(heads, features, 16, device=device))
    bias = nn.Parameter(torch.zeros(heads, 16, device=device))
    nn.init.normal_(weight, std=0.02)
    optimizer = torch.optim.AdamW([weight, bias], lr=args.probe_lr, weight_decay=1e-4)
    best_score = -1.0; best = None; patience = 0; best_epoch = 0
    for epoch in range(1, args.probe_epochs + 1):
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), args.probe_batch_size):
            ids = order[start:start + args.probe_batch_size]
            x = train_x[ids].to(device, torch.float32); y = train_y[ids].to(device)
            x = (x - mean) / std
            logits = torch.einsum("nkf,kfc->nkc", x, weight) + bias
            loss = F.cross_entropy(logits.flatten(0, 1), y[:, None].expand(-1, heads).reshape(-1))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            logits = []
            for start in range(0, len(val_y), args.probe_batch_size):
                x = val_x[start:start + args.probe_batch_size].to(device, torch.float32)
                x = (x - mean) / std
                logits.append((torch.einsum("nkf,kfc->nkc", x, weight) + bias).cpu())
            scores = accuracy(torch.cat(logits), val_y); score = scores.mean().item()
        if score > best_score + 1e-5:
            best_score = score; best_epoch = epoch; patience = 0
            best = (weight.detach().cpu().clone(), bias.detach().cpu().clone())
        else:
            patience += 1
            if patience >= args.patience: break
    weight_data, bias_data = best; weight.data.copy_(weight_data.to(device)); bias.data.copy_(bias_data.to(device))
    with torch.no_grad():
        logits = []
        for start in range(0, len(test_y), args.probe_batch_size):
            x = test_x[start:start + args.probe_batch_size].to(device, torch.float32)
            x = (x - mean) / std
            logits.append((torch.einsum("nkf,kfc->nkc", x, weight) + bias).cpu())
    scores = accuracy(torch.cat(logits), test_y).tolist()
    return {"test_accuracies": scores, "mean_test_accuracy": sum(scores) / len(scores),
            "best_test_accuracy": max(scores), "best_head": scores.index(max(scores)),
            "best_val_mean": best_score, "best_epoch": best_epoch, "heads": heads,
            "features_per_head": features}


class MLP(nn.Module):
    def __init__(self, width, hidden):
        super().__init__(); self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                                     nn.LayerNorm(hidden), nn.Linear(hidden, 16))
    def forward(self, x): return self.net(x)


def fit_mlp(train_x, train_y, val_x, val_y, test_x, test_y, args, device, seed):
    set_seed(seed); train_x = train_x[:, 0]; val_x = val_x[:, 0]; test_x = test_x[:, 0]
    mean = train_x.float().mean(0).to(device); std = train_x.float().std(0).clamp_min(1e-4).to(device)
    model = MLP(train_x.shape[-1], args.mlp_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    best_score = -1.0; best = None; patience = 0; best_epoch = 0
    for epoch in range(1, args.probe_epochs + 1):
        order = torch.randperm(len(train_y))
        model.train()
        for start in range(0, len(order), args.probe_batch_size):
            ids = order[start:start + args.probe_batch_size]
            x = (train_x[ids].to(device, torch.float32) - mean) / std; y = train_y[ids].to(device)
            loss = F.cross_entropy(model(x), y); optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for start in range(0, len(val_y), args.probe_batch_size):
                x = (val_x[start:start + args.probe_batch_size].to(device, torch.float32) - mean) / std
                y = val_y[start:start + args.probe_batch_size].to(device)
                correct += (model(x).argmax(-1) == y).sum().item(); total += len(y)
        score = correct / total
        if score > best_score + 1e-5:
            best_score = score; best_epoch = epoch; patience = 0
            best = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= args.patience: break
    model.load_state_dict(best); model.eval(); correct = total = 0
    with torch.no_grad():
        for start in range(0, len(test_y), args.probe_batch_size):
            x = (test_x[start:start + args.probe_batch_size].to(device, torch.float32) - mean) / std
            y = test_y[start:start + args.probe_batch_size].to(device)
            correct += (model(x).argmax(-1) == y).sum().item(); total += len(y)
    return {"test_accuracy": correct / total, "best_val_accuracy": best_score,
            "best_epoch": best_epoch, "hidden": args.mlp_hidden,
            "features": train_x.shape[-1]}


def features(states, kind):
    # states: [N, layer, slot, dim]
    n, layers, slots, dim = states.shape
    if kind == "mean_concat": return states.mean(2).reshape(n, 1, layers * dim)
    if kind == "layer_mean": return states.mean(2)
    if kind == "slot": return states.reshape(n, layers * slots, dim)
    if kind == "layer_concat": return states.reshape(n, layers, slots * dim)
    if kind == "all_concat": return states.reshape(n, 1, layers * slots * dim)
    raise ValueError(kind)


def run_one(seed, count, args, device, dtype, root):
    folder = root / f"seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"chunks{count}.json"
    if path.exists() and not args.force: return json.loads(path.read_text(encoding="utf-8"))
    checkpoint_path = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    original_probe = nn.Linear(192, 16).to(device)
    model.load_state_dict(state["model"]); original_probe.load_state_dict(state["probe"])
    base = args.dataset_seed_base + seed * 1000 + count * 10
    train_s, train_y, train_behavior = collect(model, original_probe, count, args.train_samples,
                                               args.extract_batch_size, args.chunk_size, device, dtype, base + 1)
    val_s, val_y, val_behavior = collect(model, original_probe, count, args.val_samples,
                                         args.extract_batch_size, args.chunk_size, device, dtype, base + 2)
    test_s, test_y, test_behavior = collect(model, original_probe, count, args.test_samples,
                                            args.extract_batch_size, args.chunk_size, device, dtype, base + 3)
    del model, original_probe; torch.cuda.empty_cache()
    probes = {}
    for index, kind in enumerate(["mean_concat", "layer_mean", "slot", "layer_concat", "all_concat"]):
        result = fit_linear(features(train_s, kind), train_y, features(val_s, kind), val_y,
                            features(test_s, kind), test_y, args, device, base + 100 + index)
        if kind == "layer_mean" or kind == "layer_concat":
            result["head_map"] = [{"layer": i} for i in range(train_s.shape[1])]
        elif kind == "slot":
            result["head_map"] = [{"layer": i // train_s.shape[2], "slot": i % train_s.shape[2]}
                                  for i in range(train_s.shape[1] * train_s.shape[2])]
        probes[kind] = result
    probes["nonlinear_mlp"] = fit_mlp(features(train_s, "all_concat"), train_y,
                                       features(val_s, "all_concat"), val_y,
                                       features(test_s, "all_concat"), test_y,
                                       args, device, base + 200)
    result = {"seed": seed, "chunks": count, "state_shape": list(train_s.shape[1:]),
              "train_behavior": train_behavior, "val_behavior": val_behavior,
              "test_behavior": test_behavior, "probes": probes}
    save(path, result); return result


def main():
    p = argparse.ArgumentParser(description="Level 6.10 frozen memory tomography")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--chunks", nargs="+", type=int, default=[2, 4, 8, 16])
    p.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--train-samples", type=int, default=2048)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--extract-batch-size", type=int, default=16)
    p.add_argument("--probe-batch-size", type=int, default=128)
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--dataset-seed-base", type=int, default=610000)
    p.add_argument("--output", default="experiments/level6_10/formal")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda"); dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True); results = []
    for seed in args.seeds:
        for count in args.chunks:
            result = run_one(seed, count, args, device, dtype, root); results.append(result)
            save(root / "runs.partial.json", results); torch.cuda.empty_cache()
            print(f"completed seed={seed} chunks={count} behavior={result['test_behavior']['query']:.2%}", flush=True)
    summary = []
    for count in args.chunks:
        selected = [r for r in results if r["chunks"] == count]
        row = {"chunks": count}
        for name in ["mean_concat", "layer_mean", "slot", "layer_concat", "all_concat"]:
            row[name] = sum(r["probes"][name]["best_test_accuracy"] for r in selected) / len(selected)
        row["nonlinear_mlp"] = sum(r["probes"]["nonlinear_mlp"]["test_accuracy"] for r in selected) / len(selected)
        row["behavior"] = sum(r["test_behavior"]["query"] for r in selected) / len(selected)
        row["original_probe"] = sum(r["test_behavior"]["original_probe"] for r in selected) / len(selected)
        summary.append(row)
    save(root / "summary.json", {"protocol": vars(args), "summary": summary, "runs": results})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
