"""Milestone 2.1: paired lifecycle, reinforcement and capacity tomography."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from config import CognitiveMemoryConfig
from pretrained_cognitive_adapter import FrozenCognitiveIST
from run_v0_4_pretrained_writer_gate import Tee, build_stream, load_model, open_tokens, target_slots


def snapshot(state, target_id, config):
    episodic = state["episodic"]
    mask = target_slots(episodic, target_id)
    clock = int(state["clock"])
    age = (clock - episodic["born"]).clamp_min(0).float()
    idle = (clock - episodic["last_access"]).clamp_min(0).float()
    utility = (episodic["strength"] + config.access_bonus * torch.log1p(episodic["accesses"].float())
               - config.age_decay * (age + 0.5 * idle))
    valid_utility = utility[episodic["valid"]]
    target_utility = utility[mask]
    semantic = state["semantic"]
    semantic_exact = bool(
        semantic["valid"].any()
        and ((semantic["source_token_ids"] == target_id) & semantic["source_valid"]).any()
    )
    return {
        "episodic": bool(mask.any()),
        "semantic_exact": semantic_exact,
        "target_utility": float(target_utility.max()) if target_utility.numel() else None,
        "eviction_floor": float(valid_utility.min()) if valid_utility.numel() else None,
        "accesses": int(episodic["accesses"][mask].max()) if mask.any() else 0,
    }


@torch.no_grad()
def run_trace(backbone, tokenizer, device, args, capacity, policy, sample, target_id):
    maximum = max(args.lengths)
    config = CognitiveMemoryConfig(
        event_span=8, working_events=4, episodic_events=capacity,
        semantic_slots=args.semantic_slots, admissions_per_chunk=16,
        retrieved_events=3, age_decay=args.age_decay,
        access_bonus=args.access_bonus, consolidation_accesses=args.consolidation_accesses)
    model = FrozenCognitiveIST(backbone, config, args.injection_layer).to(device)
    stream, _ = build_stream(tokenizer, maximum, args.chunk_size,
                             args.seed + sample, target_id, "reinforced")
    state = None; points = {}
    for chunk in range(maximum):
        begin = chunk * args.chunk_size
        _, state = model(stream[begin:begin + args.chunk_size][None].to(device), state,
                         chunk, begin, "zero", True)
        present = target_slots(state["episodic"], target_id)
        if present.any() and (chunk == 0 or (chunk + 1) % args.rehearse_every == 0):
            slots = torch.where(present[0])[0][None]
            for _ in range(args.rehearsals_per_visit):
                state = model.memory.reinforce(state, slots, mode=policy)
        if chunk + 1 in args.lengths:
            points[chunk + 1] = snapshot(state, target_id, config)
    return points


@torch.no_grad()
def evaluate(args):
    tokenizer, backbone, device = load_model(args.model_id, args.local_files_only)
    targets = open_tokens(tokenizer, args.samples)
    rows = []
    for capacity in args.capacities:
        for policy in args.policies:
            traces = [run_trace(backbone, tokenizer, device, args, capacity, policy,
                                sample, targets[sample][0]) for sample in range(args.samples)]
            for length in args.lengths:
                items = [trace[length] for trace in traces]
                retained = [item for item in items if item["episodic"]]
                row = {
                    "capacity": capacity, "policy": policy, "chunks": length,
                    "samples": args.samples,
                    "episodic_retention_rate": len(retained) / args.samples,
                    "semantic_exact_rate": sum(item["semantic_exact"] for item in items) / args.samples,
                    "mean_accesses": sum(item["accesses"] for item in items) / args.samples,
                    "mean_target_utility_when_retained": (
                        sum(item["target_utility"] for item in retained) / len(retained) if retained else None),
                    "mean_eviction_floor": sum(item["eviction_floor"] for item in items) / len(items),
                }
                rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    passed = all(
        any(row["capacity"] == capacity and row["policy"] == "relative"
            and row["chunks"] == max(args.lengths)
            and row["semantic_exact_rate"] >= args.min_semantic_exact
            for row in rows)
        for capacity in args.capacities
    ) and any(row["capacity"] == max(args.capacities) and row["policy"] == "relative"
              and row["chunks"] == max(args.lengths)
              and row["episodic_retention_rate"] >= args.min_retention for row in rows)
    return {"status": "pass" if passed else "fail", "target_tokens": [text for _, text in targets], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lengths", nargs="+", type=int, default=[4, 16, 32])
    parser.add_argument("--capacities", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--policies", nargs="+", choices=["fixed", "relative"], default=["fixed", "relative"])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--semantic-slots", type=int, default=8)
    parser.add_argument("--age-decay", type=float, default=0.02)
    parser.add_argument("--access-bonus", type=float, default=0.4)
    parser.add_argument("--consolidation-accesses", type=int, default=3)
    parser.add_argument("--rehearse-every", type=int, default=4)
    parser.add_argument("--rehearsals-per-visit", type=int, default=2)
    parser.add_argument("--injection-layer", type=int, default=-4)
    parser.add_argument("--min-retention", type=float, default=0.75)
    parser.add_argument("--min-semantic-exact", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=3404)
    parser.add_argument("--output", type=Path, default=Path("experiments/paired_tomography/results.json"))
    parser.add_argument("--log", type=Path, default=Path("experiments/paired_tomography/run.log"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps({"status": "protocol-pass", "protocol": protocol}, ensure_ascii=False, indent=2)); return 0
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": protocol, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v0_4_PAIRED_TOMOGRAPHY_{result['status'].upper()}", flush=True)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__": raise SystemExit(main())
