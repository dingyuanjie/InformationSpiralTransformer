import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks
from run_level6_13_1_local import bootstrap_mean_ci, holm, mcnemar, save
from run_level6_14_1_local import choose_candidates, swap_slots


SEEDS = [606, 808, 1001]
SWAP_TIMES = [4, 8]
FEATURE_NAMES = [
    "encoded_norm", "attended_norm", "propagation_ratio",
    "gate_mean", "gate_std", "rewrite_relative", "rewrite_cosine",
    "old_memory_norm", "new_memory_norm", "slot_norm_std",
    "compression_entropy_mean", "compression_entropy_std",
]


def reset_control(model):
    memory = model.blocks[2].memory
    memory.propagation_scale = 1.0
    memory.propagation_relative_cap = None
    memory.propagation_consistency_threshold = None
    model.blocks[2].historical_read_scale = 1.0


def observable_features(model):
    diag = model.blocks[2].memory.last_diagnostics
    old = diag["old_memory"].float()
    new = diag["new_memory"].float()
    gate = diag["update_gate"].float()
    entropy = diag["attention_entropy"].float()
    old_norm = old.norm(dim=-1)
    new_norm = new.norm(dim=-1)
    values = [
        diag["encoded_norm"].float(),
        diag["attended_memory_norm"].float(),
        diag["propagation_ratio"].float(),
        gate.mean(dim=(1, 2)),
        gate.std(dim=(1, 2)),
        (new - old).norm(dim=-1).mean(dim=-1)
        / old_norm.mean(dim=-1).clamp_min(1e-8),
        F.cosine_similarity(old, new, dim=-1).mean(dim=-1),
        old_norm.mean(dim=-1),
        new_norm.mean(dim=-1),
        new_norm.std(dim=-1),
        entropy.mean(dim=-1),
        entropy.std(dim=-1),
    ]
    return torch.stack(values, dim=-1)


def load_model(seed, args, device):
    checkpoint = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def collect_detector_data(model, seed, candidates, args, device, dtype):
    splits = {name: {"x": [], "y": [], "group": []}
              for name in ["train", "validation", "test"]}
    seed_groups = {
        "train": [args.detector_seed_base + seed * 100 + i
                  for i in range(args.detector_train_seeds)],
        "validation": [args.detector_seed_base + seed * 100 + 20 + i
                       for i in range(args.detector_validation_seeds)],
        "test": [args.detector_seed_base + seed * 100 + 40 + i
                 for i in range(args.detector_test_seeds)],
    }
    batches = args.detector_samples_per_seed // args.eval_batch_size
    for split, eval_seeds in seed_groups.items():
        for candidate in candidates:
            for swap_after in args.swap_times:
                group = f"{candidate['key']}@{swap_after}"
                for eval_seed in eval_seeds:
                    set_seed(eval_seed)
                    for _ in range(batches):
                        chunks, _, _ = make_chunks(
                            args.eval_batch_size, args.chunks, args.chunk_size, device)
                        clean_memory = None
                        polluted_memory = None
                        clean_feature = None
                        polluted_feature = None
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            for chunk_index in range(swap_after + 1):
                                reset_control(model)
                                _, clean_memory = model(
                                    chunks[:, chunk_index], memory=clean_memory,
                                    return_memory=True, per_layer_memory=True)
                                if chunk_index + 1 == swap_after + 1:
                                    clean_feature = observable_features(model)
                                reset_control(model)
                                _, polluted_memory = model(
                                    chunks[:, chunk_index], memory=polluted_memory,
                                    return_memory=True, per_layer_memory=True)
                                if chunk_index + 1 == swap_after:
                                    polluted_memory = swap_slots(polluted_memory, candidate["slots"])
                                if chunk_index + 1 == swap_after + 1:
                                    polluted_feature = observable_features(model)
                        count = args.eval_batch_size
                        splits[split]["x"].extend(clean_feature.cpu().tolist())
                        splits[split]["y"].extend([0] * count)
                        splits[split]["group"].extend([group] * count)
                        splits[split]["x"].extend(polluted_feature.cpu().tolist())
                        splits[split]["y"].extend([1] * count)
                        splits[split]["group"].extend([group] * count)
    return {"seed": seed, "feature_names": FEATURE_NAMES,
            "seed_groups": seed_groups, "splits": splits}


def standardize_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(x, y, args):
    mean, std = standardize_fit(x)
    xt = torch.tensor((x - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    weight = torch.zeros(xt.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=args.probe_lr,
                                 weight_decay=args.probe_weight_decay)
    for _ in range(args.probe_steps):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xt @ weight + bias, yt)
        loss.backward()
        optimizer.step()
    return {"mean": mean, "std": std,
            "weight": weight.detach().numpy(), "bias": float(bias.detach())}


def probe_scores(probe, x):
    z = (x - probe["mean"]) / probe["std"]
    return sigmoid(z @ probe["weight"] + probe["bias"])


def roc_auc(y, score):
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    positives = y == 1
    n_pos = positives.sum()
    n_neg = len(y) - n_pos
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def select_threshold(y, score, max_fpr):
    clean = score[y == 0]
    polluted = score[y == 1]
    candidates = np.unique(score)
    best = None
    for threshold in candidates:
        fpr = float((clean >= threshold).mean())
        tpr = float((polluted >= threshold).mean())
        if fpr <= max_fpr + 1e-12 and (best is None or tpr > best["tpr"]):
            best = {"threshold": float(threshold), "fpr": fpr, "tpr": tpr}
    if best is None:
        threshold = float(np.nextafter(clean.max(), np.inf))
        best = {"threshold": threshold, "fpr": 0.0,
                "tpr": float((polluted >= threshold).mean())}
    return best


def evaluate_probe(probe, threshold, split):
    x = np.asarray(split["x"], dtype=np.float32)
    y = np.asarray(split["y"], dtype=np.int8)
    score = probe_scores(probe, x)
    prediction = score >= threshold
    return {"samples": int(len(y)), "auc": roc_auc(y, score),
            "fpr": float(prediction[y == 0].mean()),
            "tpr": float(prediction[y == 1].mean()),
            "accuracy": float((prediction == y).mean())}


def train_probes(datasets, args):
    probes = {}
    for seed, data in datasets.items():
        train = data["splits"]["train"]
        x = np.asarray(train["x"], dtype=np.float32)
        y = np.asarray(train["y"], dtype=np.float32)
        probe = fit_logistic(x, y, args)
        validation = data["splits"]["validation"]
        vx = np.asarray(validation["x"], dtype=np.float32)
        vy = np.asarray(validation["y"], dtype=np.int8)
        selection = select_threshold(vy, probe_scores(probe, vx), args.max_clean_fpr)
        probes[seed] = {"probe": probe, "selection": selection,
                        "validation": evaluate_probe(probe, selection["threshold"], validation),
                        "test": evaluate_probe(probe, selection["threshold"],
                                               data["splits"]["test"])}
    transfer = []
    for held_out in datasets:
        train_x, train_y = [], []
        for seed, data in datasets.items():
            if seed == held_out:
                continue
            train_x.extend(data["splits"]["train"]["x"])
            train_y.extend(data["splits"]["train"]["y"])
        probe = fit_logistic(np.asarray(train_x, dtype=np.float32),
                             np.asarray(train_y, dtype=np.float32), args)
        pooled_validation = {"x": [], "y": []}
        for seed, data in datasets.items():
            if seed != held_out:
                pooled_validation["x"].extend(data["splits"]["validation"]["x"])
                pooled_validation["y"].extend(data["splits"]["validation"]["y"])
        vx = np.asarray(pooled_validation["x"], dtype=np.float32)
        vy = np.asarray(pooled_validation["y"], dtype=np.int8)
        selection = select_threshold(vy, probe_scores(probe, vx), args.max_clean_fpr)
        transfer.append({"held_out_seed": held_out, "selection": selection,
                         "test": evaluate_probe(probe, selection["threshold"],
                                                datasets[held_out]["splits"]["test"])})
    return probes, transfer


def serialize_probe(item):
    probe = item["probe"]
    return {**{k: v for k, v in item.items() if k != "probe"},
            "probe": {"mean": probe["mean"].tolist(), "std": probe["std"].tolist(),
                      "weight": probe["weight"].tolist(), "bias": probe["bias"]}}


@torch.no_grad()
def conditional_evaluate(model, probe, threshold, seed, candidate, swap_after,
                         args, device, dtype):
    output = {name: [] for name in ["target", "donor", "baseline_prediction",
                                     "conditional_prediction", "clean_prediction",
                                     "trigger_polluted", "trigger_clean"]}
    eval_seeds = [args.intervention_seed_base + seed * 100 + i
                  for i in range(args.intervention_eval_seeds)]
    batches = args.intervention_samples_per_seed // args.eval_batch_size
    mean = torch.tensor(probe["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(probe["std"], device=device, dtype=torch.float32)
    weight = torch.tensor(probe["weight"], device=device, dtype=torch.float32)
    bias = torch.tensor(probe["bias"], device=device, dtype=torch.float32)
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device)
            baseline_memory = None
            conditional_memory = None
            clean_memory = None
            polluted_trigger = torch.zeros(args.eval_batch_size, dtype=torch.bool, device=device)
            clean_trigger = torch.zeros_like(polluted_trigger)
            with torch.autocast(device_type="cuda", dtype=dtype):
                for chunk_index in range(args.chunks):
                    chunk_number = chunk_index + 1
                    reset_control(model)
                    baseline_logits, baseline_memory = model(
                        chunks[:, chunk_index], memory=baseline_memory,
                        return_memory=True, per_layer_memory=True)
                    reset_control(model)
                    conditional_logits, proposed_conditional = model(
                        chunks[:, chunk_index], memory=conditional_memory,
                        return_memory=True, per_layer_memory=True)
                    if chunk_number == swap_after + 1:
                        feature = observable_features(model).float()
                        score = torch.sigmoid(((feature - mean) / std) @ weight + bias)
                        polluted_trigger = score >= threshold
                        model.blocks[2].memory.propagation_scale = torch.where(
                            polluted_trigger, args.intervention_scale, 1.0)[:, None, None]
                        conditional_logits, proposed_conditional = model(
                            chunks[:, chunk_index], memory=conditional_memory,
                            return_memory=True, per_layer_memory=True)
                    conditional_memory = proposed_conditional
                    reset_control(model)
                    clean_logits, proposed_clean = model(
                        chunks[:, chunk_index], memory=clean_memory,
                        return_memory=True, per_layer_memory=True)
                    if chunk_number == swap_after + 1:
                        feature = observable_features(model).float()
                        score = torch.sigmoid(((feature - mean) / std) @ weight + bias)
                        clean_trigger = score >= threshold
                        model.blocks[2].memory.propagation_scale = torch.where(
                            clean_trigger, args.intervention_scale, 1.0)[:, None, None]
                        clean_logits, proposed_clean = model(
                            chunks[:, chunk_index], memory=clean_memory,
                            return_memory=True, per_layer_memory=True)
                    clean_memory = proposed_clean
                    if chunk_number == swap_after:
                        baseline_memory = swap_slots(baseline_memory, candidate["slots"])
                        conditional_memory = swap_slots(conditional_memory, candidate["slots"])
            output["target"].extend(target.cpu().tolist())
            output["donor"].extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
            output["baseline_prediction"].extend(
                baseline_logits[:, -1, :16].argmax(-1).cpu().tolist())
            output["conditional_prediction"].extend(
                conditional_logits[:, -1, :16].argmax(-1).cpu().tolist())
            output["clean_prediction"].extend(clean_logits[:, -1, :16].argmax(-1).cpu().tolist())
            output["trigger_polluted"].extend(polluted_trigger.cpu().tolist())
            output["trigger_clean"].extend(clean_trigger.cpu().tolist())
    reset_control(model)
    arrays = {key: np.asarray(value) for key, value in output.items()}
    baseline_correct = (arrays["baseline_prediction"] == arrays["target"]).astype(np.int8)
    conditional_correct = (arrays["conditional_prediction"] == arrays["target"]).astype(np.int8)
    clean_correct = (arrays["clean_prediction"] == arrays["target"]).astype(np.int8)
    donor_mask = arrays["donor"] != arrays["target"]
    baseline_donor = ((arrays["baseline_prediction"] == arrays["donor"]) & donor_mask).astype(np.int8)
    conditional_donor = ((arrays["conditional_prediction"] == arrays["donor"]) & donor_mask).astype(np.int8)
    return {"samples": int(len(arrays["target"])),
            "baseline_accuracy": float(baseline_correct.mean()),
            "conditional_accuracy": float(conditional_correct.mean()),
            "clean_accuracy": float(clean_correct.mean()),
            "polluted_trigger_rate": float(arrays["trigger_polluted"].mean()),
            "clean_trigger_rate": float(arrays["trigger_clean"].mean()),
            "baseline_correct": baseline_correct.tolist(),
            "conditional_correct": conditional_correct.tolist(),
            "baseline_donor": baseline_donor[donor_mask].tolist(),
            "conditional_donor": conditional_donor[donor_mask].tolist()}


def analyze_interventions(rows, args):
    analyzed, accuracy_tests, donor_tests = [], [], []
    for index, row in enumerate(rows):
        baseline = np.asarray(row["baseline_correct"], dtype=np.int8)
        conditional = np.asarray(row["conditional_correct"], dtype=np.int8)
        base_donor = np.asarray(row["baseline_donor"], dtype=np.int8)
        cond_donor = np.asarray(row["conditional_donor"], dtype=np.int8)
        atest = mcnemar(conditional, baseline)
        dtest = mcnemar(cond_donor, base_donor)
        accuracy_tests.append(atest); donor_tests.append(dtest)
        analyzed.append({k: v for k, v in row.items()
                         if k not in ["baseline_correct", "conditional_correct",
                                      "baseline_donor", "conditional_donor"]} | {
            "accuracy_gain": bootstrap_mean_ci(
                conditional - baseline, args.bootstrap_seed + index,
                args.bootstrap_iterations),
            "corrected_samples": int(((conditional == 1) & (baseline == 0)).sum()),
            "harmed_samples": int(((conditional == 0) & (baseline == 1)).sum()),
            "donor_reduction": bootstrap_mean_ci(
                base_donor - cond_donor, args.bootstrap_seed + 10000 + index,
                args.bootstrap_iterations),
            "mcnemar_accuracy": atest, "mcnemar_donor": dtest})
    holm(accuracy_tests); holm(donor_tests)
    for row, atest, dtest in zip(analyzed, accuracy_tests, donor_tests):
        row["mcnemar_accuracy"] = atest; row["mcnemar_donor"] = dtest
    return analyzed


def plot_summary(probe_summary, interventions, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    seeds = [str(x["seed"]) for x in probe_summary]
    axes[0].bar(np.arange(len(seeds)) - 0.18, [x["test"]["tpr"] * 100 for x in probe_summary],
                width=0.36, label="Polluted TPR")
    axes[0].bar(np.arange(len(seeds)) + 0.18, [x["test"]["fpr"] * 100 for x in probe_summary],
                width=0.36, label="Clean FPR")
    axes[0].set_xticks(range(len(seeds)), seeds)
    axes[0].set(title="Held-out detector performance", xlabel="Model seed", ylabel="Rate (%)")
    axes[0].legend()
    gains = []
    for seed in [606, 808, 1001]:
        values = [x["accuracy_gain"]["estimate"] * 100 for x in interventions
                  if x["seed"] == seed]
        gains.append(float(np.mean(values)))
    axes[1].bar(["606", "808", "1001"], gains)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(title="Conditional intervention", xlabel="Model seed",
                ylabel="Accuracy gain (pp)")
    for axis in axes: axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Level 6.16 sample-level pollution detector")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--detector-samples-per-seed", type=int, default=200)
    parser.add_argument("--detector-train-seeds", type=int, default=2)
    parser.add_argument("--detector-validation-seeds", type=int, default=1)
    parser.add_argument("--detector-test-seeds", type=int, default=1)
    parser.add_argument("--detector-seed-base", type=int, default=7750000)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--probe-lr", type=float, default=0.03)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-clean-fpr", type=float, default=0.05)
    parser.add_argument("--intervention-scale", type=float, default=0.20)
    parser.add_argument("--intervention-samples-per-seed", type=int, default=400)
    parser.add_argument("--intervention-eval-seeds", type=int, default=3)
    parser.add_argument("--intervention-seed-base", type=int, default=7950000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=616000)
    parser.add_argument("--output", default="experiments/level6_16/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for value in [args.detector_samples_per_seed, args.intervention_samples_per_seed]:
        if value % args.eval_batch_size:
            raise ValueError("sample counts must be divisible by eval-batch-size")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    save(root / "preregistration.json", {
        "primary_endpoint": "conditional accuracy gain with clean FPR <= 5%",
        "hard_model": 808, "features": FEATURE_NAMES,
        "forbidden_features": ["donor target", "ground-truth target", "clean reference"],
        "intervention": {"pathway": "MemoryAttention", "scale": args.intervention_scale,
                         "time": "first successor only"}, "protocol": vars(args)})
    dataset_path = root / "detector_datasets.json"
    if dataset_path.exists() and not args.force:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        datasets = {int(k): v for k, v in raw.items()}
    else:
        datasets = {}
        for seed in args.seeds:
            model = load_model(seed, args, device)
            datasets[seed] = collect_detector_data(
                model, seed, choose_candidates(registration, seed), args, device, dtype)
            save(dataset_path, datasets); torch.cuda.empty_cache()
    probes, transfer = train_probes(datasets, args)
    probe_summary = [{"seed": seed, "selection": item["selection"],
                      "validation": item["validation"], "test": item["test"]}
                     for seed, item in probes.items()]
    save(root / "probes.json", {str(seed): serialize_probe(item)
                                for seed, item in probes.items()})
    intervention_path = root / "interventions.partial.json"
    rows = [] if args.force or not intervention_path.exists() else json.loads(
        intervention_path.read_text(encoding="utf-8"))
    done = {(x["seed"], x["pair"], x["swap_after"]) for x in rows}
    for seed in args.seeds:
        model = load_model(seed, args, device)
        item = probes[seed]
        for candidate in choose_candidates(registration, seed):
            for swap_after in args.swap_times:
                key = (seed, candidate["key"], swap_after)
                if key in done: continue
                metric = conditional_evaluate(
                    model, item["probe"], item["selection"]["threshold"], seed,
                    candidate, swap_after, args, device, dtype)
                rows.append({"seed": seed, "pair": candidate["key"],
                             "slots": candidate["slots"], "swap_after": swap_after, **metric})
                save(intervention_path, rows)
                print(f"seed={seed} pair={candidate['key']} swap={swap_after} "
                      f"base={metric['baseline_accuracy']:.2%} "
                      f"conditional={metric['conditional_accuracy']:.2%} "
                      f"triggers={metric['polluted_trigger_rate']:.2%}/"
                      f"{metric['clean_trigger_rate']:.2%}", flush=True)
        torch.cuda.empty_cache()
    analyzed = analyze_interventions(rows, args)
    summary = {"protocol": vars(args), "probe_summary": probe_summary,
               "leave_one_model_out": transfer, "interventions": analyzed}
    save(root / "summary.json", summary)
    plot_summary(probe_summary, analyzed, root / "probe_and_intervention.png")
    print(json.dumps({"probe_summary": probe_summary,
                      "intervention_conditions": len(analyzed)}, indent=2))


if __name__ == "__main__": main()
