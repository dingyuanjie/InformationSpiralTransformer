"""v0.5.1: capacity ceiling, Oracle Evidence, Reader stability and binding causality."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from config import V05Config
from model import HybridIST
from run_level_a import atomic_json, parameter_counts
from strict_data import VALUE_IDS, assert_no_leakage, make_batch, split_audit


def condition_config(base, condition):
    if condition == "oracle_current":
        return replace(base, evidence_gate_init=0.0, core_gate_init=0.0,
                       reader_temperature=1.0, reranker_weight=0.0)
    if condition == "oracle_stable":
        return replace(base, evidence_gate_init=2.0, core_gate_init=-4.0,
                       reader_temperature=0.7, reranker_weight=1.0)
    raise ValueError(condition)


def train_reader(base, condition, seed, protocol, device):
    torch.manual_seed(seed); random.seed(seed)
    config = condition_config(base, condition)
    model = HybridIST(config, "evidence_only").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"])
    history = []; started = time.perf_counter()
    for step in range(1, protocol["steps"] + 1):
        chunks = random.choice(protocol["train_chunks"])
        batch = make_batch(protocol["batch_size"], chunks, config.chunk_size,
                           seed * 100000 + step, "train").to(device)
        logits, _ = model(batch.history, batch.query, oracle_positions=batch.fact_positions)
        task_loss = F.cross_entropy(logits, batch.answers)
        retrieval_loss = F.cross_entropy(model.memory.last_span_scores,
                                         torch.full_like(batch.answers, config.evidence_capacity - 1))
        weight = protocol["contrastive_weight"] if condition == "oracle_stable" else 0.0
        loss = task_loss + weight * retrieval_loss
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss: {condition} seed={seed} step={step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); optimizer.step()
        if step == 1 or step % max(1, protocol["steps"] // 10) == 0:
            history.append({"step": step, "loss": float(loss.detach()),
                            "task_loss": float(task_loss.detach()),
                            "retrieval_loss": float(retrieval_loss.detach()),
                            "accuracy": float(logits.argmax(-1).eq(batch.answers).float().mean()),
                            "gradient_norm": grad_norm,
                            "evidence_gate": float(torch.sigmoid(model.memory.evidence_gate).detach())})
            print(condition, seed, history[-1], flush=True)
    return model, history, time.perf_counter() - started


def exact_masks(evidence, batch):
    expected = batch.fact_positions[:, None, None] + torch.arange(
        evidence["positions"].size(-1), device=batch.history.device)[None, None]
    exact = evidence["positions"].eq(expected).all(-1) & evidence["valid"]
    same = ((evidence["token_ids"] == batch.target_entities[:, None, None]).any(-1)
            & (evidence["token_ids"] == batch.answers[:, None, None]).any(-1)
            & evidence["valid"])
    return exact, same


@torch.no_grad()
def evaluate(model, config, seed, chunks, samples, device, oracle=False):
    counts = {"correct": 0, "exact": 0, "same": 0, "correct_exact": 0,
              "correct_not_exact": 0, "reader_exact": 0, "reader_same": 0}
    entropy_total = 0.0
    for start in range(0, samples, 32):
        size = min(32, samples - start)
        batch = make_batch(size, chunks, config.chunk_size,
                           seed * 1000000 + chunks * 10000 + start, "strict").to(device)
        positions = batch.fact_positions if oracle else None
        logits, state = model(batch.history, batch.query, oracle_positions=positions)
        correct = logits.argmax(-1).eq(batch.answers); exact, same = exact_masks(state["evidence"], batch)
        exact_any, same_any = exact.any(-1), same.any(-1)
        selected = model.last_provenance["slots"]
        read_exact = exact.gather(1, selected).any(-1); read_same = same.gather(1, selected).any(-1)
        weights = model.last_provenance["weights"].clamp_min(1e-8)
        entropy_total += float((-(weights * weights.log()).sum(-1)).sum())
        counts["correct"] += int(correct.sum()); counts["exact"] += int(exact_any.sum())
        counts["same"] += int(same_any.sum()); counts["correct_exact"] += int((correct & exact_any).sum())
        counts["correct_not_exact"] += int((correct & ~exact_any).sum())
        counts["reader_exact"] += int(read_exact.sum()); counts["reader_same"] += int(read_same.sum())
    exact_n, absent_n = counts["exact"], samples - counts["exact"]
    facts = chunks * 2
    return {"chunks": chunks, "capacity": config.evidence_capacity, "facts": facts,
            "capacity_fraction": min(1.0, config.evidence_capacity / facts), "oracle": oracle,
            "accuracy": counts["correct"] / samples,
            "exact_target_retention": counts["exact"] / samples,
            "same_binding_retention": counts["same"] / samples,
            "accuracy_given_exact_retained": counts["correct_exact"] / max(1, exact_n),
            "accuracy_given_exact_absent": counts["correct_not_exact"] / max(1, absent_n),
            "reader_exact_hit": counts["reader_exact"] / samples,
            "reader_same_binding_hit": counts["reader_same"] / samples,
            "mean_read_entropy": entropy_total / samples,
            "evidence_gate": float(torch.sigmoid(model.memory.evidence_gate))}


@torch.no_grad()
def binding_causality(model, config, seed, samples, device):
    batch = make_batch(samples, 8, config.chunk_size, seed + 990000, "strict").to(device)
    state = model.build_state(batch.history, oracle_positions=batch.fact_positions)
    original, _ = model(batch.history, batch.query, state=state)
    original_prediction = original.argmax(-1)
    rows = []
    for intervention in ("swap_entities", "swap_answers", "rebind", "corrupt_roles"):
        changed, _ = model(batch.history, batch.query, state=state, intervention=intervention)
        prediction = changed.argmax(-1)
        evidence, _ = model.memory._intervene(state, intervention, None)
        entity_match = evidence["token_ids"][:, :, 1].eq(batch.target_entities[:, None]) & evidence["valid"]
        has = entity_match.any(-1)
        slot = entity_match.float().argmax(-1)
        counterfactual = evidence["token_ids"][:, :, 3].gather(1, slot[:, None]).squeeze(1)
        rows.append({"intervention": intervention,
                     "original_accuracy": float(original_prediction.eq(batch.answers).float().mean()),
                     "prediction_change_rate": float(prediction.ne(original_prediction).float().mean()),
                     "counterfactual_available": float(has.float().mean()),
                     "counterfactual_follow_rate": float((prediction.eq(counterfactual) & has).float().sum() / has.sum().clamp_min(1))})
    return rows


def summarize(runs):
    grouped = {}
    for run in runs:
        for row in run["oracle_evaluation"]:
            grouped.setdefault((run["condition"], row["chunks"]), []).append(row["accuracy"])
    return [{"condition": key[0], "chunks": key[1], "mean_accuracy": sum(values) / len(values),
             "std_accuracy": math.sqrt(sum((x - sum(values) / len(values)) ** 2 for x in values) /
                                       max(1, len(values) - 1)), "seeds": len(values)}
            for key, values in sorted(grouped.items())]


def write_reports(output, payload):
    output.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    with (output / "reader_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    capacity_rows = []
    for condition in sorted({run["condition"] for run in payload["runs"]}):
        selected = [run for run in payload["runs"] if run["condition"] == condition]
        for capacity in payload["protocol"]["capacities"]:
            rows = [row for run in selected for row in run["capacity_curve"] if row["capacity"] == capacity]
            capacity_rows.append({"condition": condition, "capacity": capacity,
                                  "capacity_fraction": rows[0]["capacity_fraction"],
                                  "mean_exact_retention": sum(row["exact_target_retention"] for row in rows) / len(rows),
                                  "mean_same_binding_retention": sum(row["same_binding_retention"] for row in rows) / len(rows),
                                  "mean_accuracy": sum(row["accuracy"] for row in rows) / len(rows),
                                  "mean_accuracy_given_retained": sum(row["accuracy_given_exact_retained"] for row in rows) / len(rows),
                                  "mean_accuracy_given_absent": sum(row["accuracy_given_exact_absent"] for row in rows) / len(rows)})
    with (output / "capacity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(capacity_rows[0])); writer.writeheader(); writer.writerows(capacity_rows)
    lines = ["# IST v0.5.1 result", "", f"Status: `{payload['status']}`", "",
             "## Oracle Reader stability", "", "| Condition | Chunks | Seeds | Accuracy | Std |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['condition']} | {row['chunks']} | {row['seeds']} | "
                     f"{row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} |")
    lines += ["", "## 32-chunk capacity audit", "",
              "The stream contains 64 fact occurrences. `K/N` is the exact-occurrence ceiling for an unbiased query-blind reservoir.", "",
              "| Condition | K | K/N | Exact retained | Same binding | Accuracy | Acc given retained | Acc given absent |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in capacity_rows:
        lines.append(f"| {row['condition']} | {row['capacity']} | {row['capacity_fraction']:.4f} | "
                     f"{row['mean_exact_retention']:.4f} | {row['mean_same_binding_retention']:.4f} | "
                     f"{row['mean_accuracy']:.4f} | {row['mean_accuracy_given_retained']:.4f} | "
                     f"{row['mean_accuracy_given_absent']:.4f} |")
    lines += ["", "Oracle rows force the supervised exact target into Memory and are diagnostics, not deployable scores.",
              "Binding interventions are valid only when original accuracy is above chance."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/v0_5_1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/v0_5_1"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    assert_no_leakage(); base = V05Config.from_json(protocol["base_config"])
    if args.smoke_test:
        protocol.update({"capacities": [4, 12], "reader_conditions": ["oracle_current", "oracle_stable"],
                         "seeds": [505], "steps": 3, "batch_size": 4,
                         "evaluation_chunks": [2, 4], "evaluation_samples": 8})
        args.output = args.output / "smoke"
    registered = {"status": "protocol-pass", "protocol": protocol, "split_audit": split_audit()}
    if args.dry_run: print(json.dumps(registered, indent=2)); return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); runs = []
    for condition in protocol["reader_conditions"]:
        for seed in protocol["seeds"]:
            model, history, seconds = train_reader(base, condition, seed, protocol, device)
            oracle_rows = [evaluate(model, model.config, seed, chunks,
                                    protocol["evaluation_samples"], device, oracle=True)
                           for chunks in protocol["evaluation_chunks"]]
            capacity_rows = []
            for capacity in protocol["capacities"]:
                config = replace(model.config, evidence_capacity=capacity,
                                 reads_per_query=min(model.config.reads_per_query, capacity))
                scaled = HybridIST(config, "evidence_only").to(device)
                scaled.load_state_dict(model.state_dict(), strict=True)
                capacity_rows.append(evaluate(scaled, config, seed, 32,
                                              protocol["evaluation_samples"], device, oracle=False))
            runs.append({"condition": condition, "seed": seed, "history": history,
                         "training_seconds": seconds, "parameters": parameter_counts(model),
                         "oracle_evaluation": oracle_rows, "capacity_curve": capacity_rows,
                         "binding_causality": binding_causality(
                             model, model.config, seed, min(64, protocol["evaluation_samples"]), device)})
    payload = {**registered, "status": "complete", "device": str(device),
               "summary": summarize(runs), "runs": runs}
    atomic_json(args.output / "results.json", payload)
    write_reports(args.output, payload)
    print(f"V0_5_1_COMPLETE output={args.output}", flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
