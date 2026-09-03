"""Train and audit IST v0.5 Level A shared-vocabulary unseen bindings."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from config import V05Config
from model import HybridIST
from strict_data import SCENARIOS, VALUE_IDS, assert_no_leakage, make_batch, split_audit


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path); return
        except PermissionError:
            if attempt == 7: raise
            time.sleep(0.05 * (attempt + 1))


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "memory": sum(p.numel() for p in model.memory.parameters()),
    }


def train_one(config, variant, seed, steps, batch_size, train_chunks, device, learning_rate):
    torch.manual_seed(seed); random.seed(seed)
    model = HybridIST(config, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []; tokens_seen = 0; started = time.perf_counter()
    for step in range(1, steps + 1):
        chunks = random.choice(train_chunks)
        batch = make_batch(batch_size, chunks, config.chunk_size, seed * 100000 + step, "train").to(device)
        logits, _ = model(batch.history, batch.query)
        loss = F.cross_entropy(logits, batch.answers)
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        tokens_seen += batch.history.numel() + batch.query.numel()
        if step == 1 or step % max(1, steps // 10) == 0:
            accuracy = float(logits.argmax(-1).eq(batch.answers).float().mean())
            row = {"step": step, "loss": float(loss.detach()), "accuracy": accuracy,
                   "grad_norm": grad_norm, "chunks": chunks}
            history.append(row); print(f"{variant} seed={seed} {row}", flush=True)
    return model, history, tokens_seen, time.perf_counter() - started


@torch.no_grad()
def evaluate(model, config, seed, chunks, samples, device, intervention="normal", scenario="multi_fact"):
    model.eval(); correct = writer_hit = reader_hit = 0; latency = 0.0
    peak = 0; valid_slots = duplicate_slots = age_total = 0.0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, samples, 32):
        size = min(32, samples - start)
        batch = make_batch(size, chunks, config.chunk_size,
                           seed * 1000000 + chunks * 10000 + start, "strict", scenario=scenario).to(device)
        if device.type == "cuda": torch.cuda.synchronize()
        tick = time.perf_counter()
        if intervention == "delete_target_source":
            state = model.build_state(batch.history)
            logits, state = model(batch.history, batch.query, state=state,
                                  intervention="delete_source", source_chunk=batch.target_chunks)
        else:
            logits, state = model(batch.history, batch.query, intervention=intervention)
        if device.type == "cuda": torch.cuda.synchronize()
        latency += time.perf_counter() - tick
        correct += int(logits.argmax(-1).eq(batch.answers).sum())
        evidence = state["evidence"]
        for row in range(size):
            target = batch.target_entities[row]
            answer = batch.answers[row]
            available = ((evidence["token_ids"][row] == target).any(-1)
                         & (evidence["token_ids"][row] == answer).any(-1)
                         & evidence["valid"][row])
            writer_hit += int(available.any())
            valid_rows = evidence["token_ids"][row][evidence["valid"][row]]
            valid_slots += valid_rows.size(0)
            if valid_rows.size(0):
                duplicate_slots += valid_rows.size(0) - torch.unique(valid_rows, dim=0).size(0)
                age_total += float((int(state["clock"]) - evidence["born"][row][evidence["valid"][row]]).float().sum())
            if model.last_provenance is not None:
                read_ids = model.last_provenance["token_ids"][row]
                reader_hit += int(((read_ids == target).any(-1) & (read_ids == answer).any(-1)).any())
    if device.type == "cuda": peak = int(torch.cuda.max_memory_allocated(device))
    return {"chunks": chunks, "samples": samples, "intervention": intervention, "scenario": scenario,
            "accuracy": correct / samples, "writer_relation_recall": writer_hit / samples,
            "reader_relation_hit": reader_hit / samples,
            "slot_utilization": valid_slots / (samples * config.evidence_capacity),
            "slot_duplicate_rate": duplicate_slots / max(1.0, valid_slots),
            "mean_memory_age": age_total / max(1.0, valid_slots),
            "latency_ms_per_example": latency * 1000 / samples,
            "peak_memory_bytes": peak}


def causal_panel(model, config, seed, chunks, samples, device):
    names = ["normal", "zero", "zero_evidence", "zero_core", "swap", "shuffle",
             "delete_target_source", "block_writer", "block_reader", "corrupt_identity"]
    return [evaluate(model, config, seed + 77, chunks, samples, device, name) for name in names]


def aggregate(runs):
    grouped = {}
    for run in runs:
        for row in run["evaluation"]:
            key = (run["variant"], row["chunks"])
            grouped.setdefault(key, []).append(row["accuracy"])
    rows = []
    for (variant, chunks), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        variance = sum((item - mean) ** 2 for item in values) / max(1, len(values) - 1)
        rows.append({"variant": variant, "chunks": chunks, "seeds": len(values),
                     "mean_accuracy": mean, "std_accuracy": math.sqrt(variance),
                     "chance": 1 / len(VALUE_IDS), "delta_vs_strong_baseline": 0.0})
    for row in rows:
        baselines = [item["mean_accuracy"] for item in rows
                     if item["chunks"] == row["chunks"] and item["variant"] in {"no_memory", "last_k"}]
        row["delta_vs_strong_baseline"] = row["mean_accuracy"] - max(baselines) if baselines else 0.0
    return rows


def write_reports(output, payload):
    atomic_json(output / "results.json", payload)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["aggregate"][0]))
        writer.writeheader(); writer.writerows(payload["aggregate"])
    lines = ["# IST v0.5 Level A result", "", f"Status: `{payload['status']}`.", "",
             "| Variant | Chunks | Seeds | Mean accuracy | Std | Delta vs baseline |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in payload["aggregate"]:
        lines.append(f"| {row['variant']} | {row['chunks']} | {row['seeds']} | "
                     f"{row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} | "
                     f"{row['delta_vs_strong_baseline']:+.4f} |")
    lines += ["", "Chance is 0.0625 (16 possible values). Smoke results validate code paths only.",
              "Causal effects are interpretable only when the normal condition learned the task."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        for variant in sorted({row["variant"] for row in payload["aggregate"]}):
            selected = [row for row in payload["aggregate"] if row["variant"] == variant]
            plt.plot([row["chunks"] for row in selected], [row["mean_accuracy"] for row in selected],
                     marker="o", label=variant)
        plt.axhline(1 / len(VALUE_IDS), linestyle="--", color="black", label="chance")
        plt.xlabel("Chunks"); plt.ylabel("Strict held-out accuracy"); plt.legend(); plt.tight_layout()
        plt.savefig(output / "accuracy_by_distance.png", dpi=160); plt.close()
    except ImportError:
        payload["notes"].append("matplotlib unavailable; PNG curve was not generated")
        atomic_json(output / "results.json", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level_a.json"))
    parser.add_argument("--output", type=Path, default=Path("results/v0_5/level_a"))
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = V05Config.from_json(args.config)
    variants = args.variants or raw["variants"]; seeds = args.seeds or raw["seeds"]
    steps = args.steps or raw["train_steps"]; samples = args.samples
    train_chunks, test_chunks = raw["train_chunks"], raw["test_chunks"]
    if args.smoke_test:
        variants, seeds, steps, samples, train_chunks, test_chunks = ["hybrid"], [505], 3, 8, [2], [2, 4]
        args.output = args.output / "smoke"
    protocol = {"config": config.to_dict(), "variants": variants, "seeds": seeds,
                "steps": steps, "samples": samples, "train_chunks": train_chunks,
                "test_chunks": test_chunks, "split_audit": split_audit()}
    assert_no_leakage()
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, indent=2)); return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for variant in variants:
        for seed in seeds:
            model, history, tokens, seconds = train_one(
                config, variant, seed, steps, raw["batch_size"], train_chunks, device,
                raw["learning_rate"])
            evaluation = [evaluate(model, config, seed, length, samples, device) for length in test_chunks]
            causal = causal_panel(model, config, seed, max(test_chunks), min(samples, 64), device)
            scenario_panel = [evaluate(model, config, seed + 131, min(8, max(test_chunks)),
                                       min(samples, 16), device, scenario=scenario)
                              for scenario in SCENARIOS]
            runs.append({"variant": variant, "seed": seed, "history": history,
                         "training_tokens": tokens, "training_seconds": seconds,
                         "parameters": parameter_counts(model), "evaluation": evaluation,
                         "causal": causal, "scenario_panel": scenario_panel})
    payload = {"status": "complete", "device": str(device), "protocol": protocol,
               "aggregate": aggregate(runs), "runs": runs,
               "notes": ["All variants instantiate the same parameter envelope; inactive paths are reported, not removed."]}
    write_reports(args.output, payload)
    print(f"V0_5_LEVEL_A_COMPLETE output={args.output}", flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
