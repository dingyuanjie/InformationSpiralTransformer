"""Level 8.2: important-vs-noise routing and causal Memory diagnosis.

Starts from completed Level 8.1 hierarchical checkpoints, performs a matched
importance calibration, then separates write-time (freeze) from query-time
(zero) causal effects for Slow and Episodic Memory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import HierarchicalMemoryConfig
from experiment_utils import ROOT, atomic_json, atomic_torch, run_metadata
from model import build_model


SOURCE = ROOT / "experiments/level8_1/formal/hierarchical_v0_2"
SEEDS = (313, 42, 2026, 7, 1234)
CHUNK_SIZE = 512
TRAIN_STEPS = 100
TRAIN_CHUNKS = 16
EVAL_BATCH = 16
MILESTONES = (16, 128, 512, 1000)
INTERVENTIONS = ("normal", "zero_slow", "zero_episodic", "freeze_slow", "freeze_episodic")


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def clone_memory(memory):
    return [{k: v.clone() if torch.is_tensor(v) else v for k, v in layer.items()} for layer in memory]


def build(seed, mode, device):
    config = HierarchicalMemoryConfig.from_dict({"router": {"mode": mode}})
    model = build_model("hierarchical_v0_2", vocab_size=19, hidden_size=64, layers=3,
                        max_sequence_length=CHUNK_SIZE, position_encoding="rope",
                        hierarchical_config=config).to(device)
    source = SOURCE / f"seed{seed}" / "stage4.pt"
    if not source.exists():
        raise FileNotFoundError(f"Level 8.1 checkpoint missing: {source}")
    model.load_state_dict(torch.load(source, map_location=device, weights_only=False)["model"], strict=True)
    return model, config


def paired_chunk(batch, target, important, device, base=None):
    tokens = (base.clone() if base is not None else
              torch.randint(16, (batch, CHUNK_SIZE), device=device))
    rows = torch.arange(batch, device=device)
    positions = 16 + rows * 29 % (CHUNK_SIZE - 32)
    if important:
        tokens[rows, positions] = 17
        tokens[rows, positions + 1] = target
    else:
        # Matched content and position, but no learned importance marker and never queried.
        tokens[rows, positions] = target
        tokens[rows, positions + 1] = torch.roll(target, 1)
    return tokens, positions


def route_snapshot(model):
    rows = []
    for layer, block in enumerate(model.blocks):
        d = block.memory.last_diagnostics
        rows.append({
            "layer": layer,
            "router_distribution": d["router_distribution"].float().cpu().tolist(),
            "write_rate": {"fast": float(d["fast_write_rate"].cpu()),
                           "slow": float(d["slow_write_rate"].cpu()),
                           "episodic": float(d["episodic_write_rate"].cpu())},
            "retention_gate": float(d["retention_gate"].cpu()),
            "slot_usage": {k: float(v.float().mean().cpu()) for k, v in d["slot_usage"].items()},
            "slot_age": {k: float(v.float().mean().cpu()) for k, v in d["slot_age"].items()},
            "slot_replacement": d["slot_replacement"].cpu().tolist(),
            "target_similarity": d["memory_similarity_to_target_encoding"].float().mean(0).cpu().tolist(),
        })
    return rows


def calibrate(model, seed, folder, steps, device, dtype, force):
    final = folder / "calibrated.pt"; resume = folder / "calibration_resume.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4); history = []; start = 0
    if final.exists() and not force:
        state = torch.load(final, map_location=device, weights_only=False); model.load_state_dict(state["model"])
        return state["history"]
    if resume.exists() and not force:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start = state["step"]; history = state["history"]
    for step in range(start + 1, steps + 1):
        set_seed(82000000 + seed * 1000 + step)
        batch = 4; target = torch.randint(16, (batch,), device=device)
        base = torch.randint(16, (batch, CHUNK_SIZE), device=device)
        first, positions = paired_chunk(batch, target, True, device, base)
        noise, noise_positions = paired_chunk(batch, target, False, device, base)
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            important_logits, important_memory = model(first, return_memory=True)
            noise_logits, _ = model(noise, return_memory=True)
            memory = important_memory
            for chunk in range(2, TRAIN_CHUNKS + 1):
                tokens = torch.randint(16, (batch, CHUNK_SIZE), device=device)
                if chunk == TRAIN_CHUNKS: tokens[:, -2] = 18; tokens[:, -1] = 16
                logits, memory = model(tokens, memory=memory, return_memory=True)
            rows = torch.arange(batch, device=device)
            query_loss = F.cross_entropy(logits[:, -1, :16], target)
            important_local = F.cross_entropy(important_logits[rows, positions, :16], target)
            noise_local = F.cross_entropy(noise_logits[rows, noise_positions, :16], target)
            loss = query_loss + .25 * important_local + .25 * noise_local + .1 * model.memory_diversity_loss()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 10 == 0:
            row = {"step": step, "loss": float(loss.detach()), "query_loss": float(query_loss.detach()),
                   "accuracy": float((logits[:, -1, :16].argmax(-1) == target).float().mean())}
            history.append(row); print(f"seed={seed} calibration step={step} loss={row['loss']:.4f} "
                                       f"query={row['accuracy']:.2%}", flush=True)
            atomic_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                  "step": step, "history": history})
    atomic_torch(final, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "step": steps, "history": history})
    return history


@torch.no_grad()
def evaluate(model, seed, device):
    target = torch.arange(EVAL_BATCH, device=device) % 16
    base = torch.randint(16, (EVAL_BATCH, CHUNK_SIZE), device=device)
    important, _ = paired_chunk(EVAL_BATCH, target, True, device, base)
    noise, _ = paired_chunk(EVAL_BATCH, target, False, device, base)
    model.eval(); model.clear_memory_interventions()
    _, important_memory = model(important, return_memory=True); important_routes = route_snapshot(model)
    _, _ = model(noise, return_memory=True); noise_routes = route_snapshot(model)
    states = {"normal": clone_memory(important_memory), "freeze_slow": clone_memory(important_memory),
              "freeze_episodic": clone_memory(important_memory)}
    rows = []
    for chunk in range(2, MILESTONES[-1] + 1):
        set_seed(82100000 + seed * 2000 + chunk)
        tokens = torch.randint(16, (EVAL_BATCH, CHUNK_SIZE), device=device)
        for condition in tuple(states):
            model.set_memory_intervention(condition)
            _, states[condition] = model(tokens, memory=states[condition], return_memory=True, detach_memory=True)
        if chunk in MILESTONES:
            query = torch.randint(16, (EVAL_BATCH, CHUNK_SIZE), device=device)
            query[:, -2] = 18; query[:, -1] = 16
            for condition in INTERVENTIONS:
                source = condition if condition.startswith("freeze_") else "normal"
                model.set_memory_intervention(condition)
                logits, _ = model(query, memory=clone_memory(states[source]), return_memory=True,
                                  detach_memory=True)
                correct = (logits[:, -1, :16].argmax(-1) == target).int().cpu().tolist()
                rows.append({"chunks": chunk, "condition": condition, "correctness": correct,
                             "accuracy": sum(correct) / len(correct), "diagnostics": route_snapshot(model)})
                print(f"seed={seed} chunks={chunk} condition={condition} "
                      f"accuracy={sum(correct)/len(correct):.2%}", flush=True)
    model.clear_memory_interventions()
    return {"seed": seed, "important_routes": important_routes, "noise_routes": noise_routes, "rows": rows}


def paired_effect(runs, chunks, condition):
    normal=[]; treatment=[]
    for run in runs:
        normal += next(x["correctness"] for x in run["rows"] if x["chunks"] == chunks and x["condition"] == "normal")
        treatment += next(x["correctness"] for x in run["rows"] if x["chunks"] == chunks and x["condition"] == condition)
    return {"chunks": chunks, "condition": condition, "normal_accuracy": sum(normal)/len(normal),
            "treatment_accuracy": sum(treatment)/len(treatment),
            "causal_effect": (sum(treatment)-sum(normal))/len(normal)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--router-mode", choices=("soft", "hard_straight_through", "disabled"), default="soft")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--output", default="experiments/level8_2/formal")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args=parser.parse_args()
    global TRAIN_CHUNKS, MILESTONES, EVAL_BATCH
    if args.smoke_test:
        args.seeds=[2026]; args.steps=2; TRAIN_CHUNKS=2; MILESTONES=(2,4); EVAL_BATCH=4
        if args.output == "experiments/level8_2/formal": args.output="experiments/level8_2/smoke"
    protocol={"task":"important_vs_noise", "seeds":args.seeds, "router_mode":args.router_mode,
              "calibration_steps":args.steps, "train_chunks":TRAIN_CHUNKS, "milestones":MILESTONES,
              "interventions":INTERVENTIONS, "source":"level8_1/formal hierarchical checkpoints"}
    if args.dry_run: print(json.dumps(protocol, indent=2)); return 0
    if not torch.cuda.is_available(): raise RuntimeError("Level 8.2 requires CUDA")
    device=torch.device("cuda"); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root=ROOT/args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/"config.json",protocol); atomic_json(root/"run_metadata.json",run_metadata(device,args.seeds))
    runs=[]; training=[]
    for seed in args.seeds:
        folder=root/f"seed{seed}"; folder.mkdir(parents=True,exist_ok=True)
        model,config=build(seed,args.router_mode,device)
        history=calibrate(model,seed,folder,args.steps,device,dtype,args.force); training.append({"seed":seed,"history":history})
        output=folder/"evaluation.json"
        if output.exists() and not args.force: row=json.loads(output.read_text(encoding="utf-8"))
        else: row=evaluate(model,seed,device); atomic_json(output,row)
        runs.append(row); atomic_json(root/"runs.partial.json",runs)
        del model; torch.cuda.empty_cache()
    route_deltas=[]
    for layer in range(3):
        important=[run["important_routes"][layer]["router_distribution"] for run in runs]
        noise=[run["noise_routes"][layer]["router_distribution"] for run in runs]
        means=lambda values:[sum(row[i] for row in values)/len(values) for i in range(4)]
        a,b=means(important),means(noise)
        route_deltas.append({"layer":layer,"important":a,"noise":b,
                             "important_minus_noise":[x-y for x,y in zip(a,b)]})
    effects=[paired_effect(runs,chunks,condition) for chunks in MILESTONES
             for condition in INTERVENTIONS if condition != "normal"]
    result={"status":"complete","protocol":protocol,"route_deltas":route_deltas,
            "causal_effects":effects,"training":training,"runs":runs}
    atomic_json(root/"raw_results.json",result); atomic_json(root/"result.json",result)
    (root/"ANALYSIS.md").write_text("# Level 8.2 Analysis\n\nRun complete. Interpret route deltas together with zero/read and freeze/write causal effects.\n",encoding="utf-8")
    print(json.dumps({"status":"complete","route_deltas":route_deltas,"causal_effects":effects},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
