"""Level 8.1: matched-training old-information retention through 1000 chunks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import HierarchicalMemoryConfig
from experiment_utils import ROOT, atomic_json, atomic_torch, parameter_count, run_metadata
from hierarchical_model import transfer_v0_1_weights
from model import build_model


SOURCE = ROOT.parent / "ist_v0_1/experiments/level7_6_4/formal"
ARCHITECTURES = ("v0_1", "hierarchical_v0_2")
SEEDS = (313, 42, 2026, 7, 1234)
CHUNK_SIZE = 512
STAGES = ((2, 200, 4), (4, 150, 2), (8, 100, 1), (16, 100, 1))
MILESTONES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)
SAMPLES = 32
EVAL_BATCH = 16
CHANCE = 1 / 16


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build(architecture, seed, config, device):
    old = torch.load(SOURCE / f"ist-full_seed{seed}" / "stage_4096.pt",
                     map_location=device, weights_only=False)["model"]
    if architecture == "v0_1":
        model = build_model("v0_1", vocab_size=19, hidden_size=64, layers=3,
                            max_sequence_length=CHUNK_SIZE, position_encoding="rope",
                            use_memory_fusion=True).to(device)
        model.load_state_dict(old, strict=True); transfer = None
    else:
        model = build_model("hierarchical_v0_2", vocab_size=19, hidden_size=64, layers=3,
                            max_sequence_length=CHUNK_SIZE, position_encoding="rope",
                            hierarchical_config=config).to(device)
        transfer = transfer_v0_1_weights(model, old)
    return model, transfer


def make_training_stream(batch, chunks, device):
    target = torch.randint(16, (batch,), device=device)
    stream = [torch.randint(16, (batch, CHUNK_SIZE), device=device) for _ in range(chunks)]
    rows = torch.arange(batch, device=device)
    positions = torch.randint(8, CHUNK_SIZE - 8, (batch,), device=device)
    stream[0][rows, positions] = 17; stream[0][rows, positions + 1] = target
    stream[-1][:, -2] = 18; stream[-1][:, -1] = 16
    return stream, target, positions


def forward_chunk(model, architecture, tokens, memory, detach=False):
    if architecture == "v0_1":
        return model(tokens, memory=memory, return_memory=True, detach_memory=detach,
                     per_layer_memory=True)
    return model(tokens, memory=memory, return_memory=True, detach_memory=detach)


def train(model, architecture, seed, folder, device, dtype, force):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4); history = []
    for stage_index, (chunks, steps, batch) in enumerate(STAGES, 1):
        final = folder / f"stage{stage_index}.pt"; resume = folder / f"stage{stage_index}_resume.pt"
        if final.exists() and not force:
            state = torch.load(final, map_location=device, weights_only=False)
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
            print(f"arch={architecture} seed={seed} chunks={chunks} already complete", flush=True); continue
        start = 0
        candidates = [path for path in (resume, resume.with_suffix(".pt.tmp")) if path.exists()]
        if candidates and not force:
            loaded = [(torch.load(path, map_location=device, weights_only=False), path) for path in candidates]
            state, selected = max(loaded, key=lambda item: int(item[0]["step"]))
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start = int(state["step"]); history = state["history"]
            print(f"resume arch={architecture} seed={seed} chunks={chunks} step={start} "
                  f"source={selected.name}", flush=True)
        for step in range(start + 1, steps + 1):
            set_seed(810000000 + seed * 10000 + stage_index * 1000 + step)
            stream, target, positions = make_training_stream(batch, chunks, device)
            model.train(); optimizer.zero_grad(set_to_none=True); memory = None; first_logits = None
            with torch.autocast(device_type="cuda", dtype=dtype):
                for index, tokens in enumerate(stream):
                    logits, memory = forward_chunk(model, architecture, tokens, memory)
                    if index == 0: first_logits = logits
                rows = torch.arange(batch, device=device)
                query_loss = F.cross_entropy(logits[:, -1, :16], target)
                local_loss = F.cross_entropy(first_logits[rows, positions, :16], target)
                loss = query_loss + 0.5 * local_loss + 0.1 * model.memory_diversity_loss()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step == 1 or step % 25 == 0:
                row = {"stage": stage_index, "chunks": chunks, "step": step,
                       "loss": float(loss.detach()), "query_loss": float(query_loss.detach()),
                       "local_loss": float(local_loss.detach()),
                       "accuracy": float((logits[:, -1, :16].argmax(-1) == target).float().mean())}
                history.append(row)
                atomic_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                      "step": step, "history": history})
                print(f"arch={architecture} seed={seed} chunks={chunks} step={step} "
                      f"loss={row['loss']:.4f} accuracy={row['accuracy']:.2%}", flush=True)
        atomic_torch(final, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "step": steps, "history": history})
    return history


def clone_memory(memory):
    if isinstance(memory[0], dict):
        return [{key: value.clone() if torch.is_tensor(value) else value for key, value in layer.items()}
                for layer in memory]
    return [item.clone() for item in memory]


def compact_diagnostics(model):
    if not hasattr(model, "memory_arch") or model.memory_arch != "hierarchical_v0_2": return None
    rows = []
    for layer, block in enumerate(model.blocks):
        d = block.memory.last_diagnostics
        rows.append({"layer": layer,
                     "fast_memory_norm": float(d["fast_memory_norm"].cpu()),
                     "slow_memory_norm": float(d["slow_memory_norm"].cpu()),
                     "episodic_memory_norm": float(d["episodic_memory_norm"].cpu()),
                     "fast_write_rate": float(d["fast_write_rate"].cpu()),
                     "slow_write_rate": float(d["slow_write_rate"].cpu()),
                     "episodic_write_rate": float(d["episodic_write_rate"].cpu()),
                     "retention_gate": float(d["retention_gate"].cpu()),
                     "importance_score": float(d["importance_score"].cpu()),
                     "router_distribution": d["router_distribution"].cpu().tolist(),
                     "slot_replacement": d["slot_replacement"].cpu().tolist(),
                     "memory_similarity_to_initial": d["memory_similarity_to_initial"].mean(0).cpu().tolist(),
                     "memory_similarity_to_target_encoding": d["memory_similarity_to_target_encoding"].mean(0).cpu().tolist()})
    return rows


@torch.no_grad()
def evaluate_replicate(model, architecture, seed, replicate, device, dtype):
    target = (torch.arange(EVAL_BATCH, device=device) + replicate * 7) % 16
    memory = None; rows = []
    for chunk in range(1, MILESTONES[-1] + 1):
        set_seed(811000000 + seed * 100000 + replicate * 2000 + chunk)
        tokens = torch.randint(16, (EVAL_BATCH, CHUNK_SIZE), device=device)
        if chunk == 1:
            positions = 16 + (torch.arange(EVAL_BATCH, device=device) * 29) % (CHUNK_SIZE - 32)
            batch_rows = torch.arange(EVAL_BATCH, device=device)
            tokens[batch_rows, positions] = 17; tokens[batch_rows, positions + 1] = target
        _, memory = forward_chunk(model, architecture, tokens, memory, detach=True)
        if chunk in MILESTONES:
            diagnostics = compact_diagnostics(model)
            set_seed(811900000 + seed * 100000 + replicate * 2000 + chunk)
            query = torch.randint(16, (EVAL_BATCH, CHUNK_SIZE), device=device)
            query[:, -2] = 18; query[:, -1] = 16
            logits, _ = forward_chunk(model, architecture, query, clone_memory(memory), detach=True)
            prediction = logits[:, -1, :16].argmax(-1)
            rows.append({"chunks": chunk, "correctness": (prediction == target).int().cpu().tolist(),
                         "predictions": prediction.cpu().tolist(), "targets": target.cpu().tolist(),
                         "diagnostics": diagnostics})
            print(f"arch={architecture} seed={seed} replicate={replicate} chunks={chunk} "
                  f"accuracy={(prediction == target).float().mean():.2%}", flush=True)
    return {"architecture": architecture, "seed": seed, "replicate": replicate,
            "samples": EVAL_BATCH, "rows": rows}


def wilson(correct, samples, z=1.959963984540054):
    p=correct/samples; scale=1+z*z/samples; middle=(p+z*z/(2*samples))/scale
    half=z*math.sqrt(p*(1-p)/samples+z*z/(4*samples*samples))/scale
    return [middle-half,middle+half]


def paired_exact(treatment, control):
    improved=sum(a==1 and b==0 for a,b in zip(treatment,control)); harmed=sum(a==0 and b==1 for a,b in zip(treatment,control)); n=improved+harmed
    tail=sum(math.comb(n,k) for k in range(min(improved,harmed)+1))/2**n if n else .5
    return {"difference":(sum(treatment)-sum(control))/len(control),"improved":improved,
            "harmed":harmed,"ties":len(control)-n,"mcnemar_exact_p":min(1.0,2*tail)}


def main() -> int:
    global STAGES, MILESTONES, SAMPLES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    parser.add_argument("--output",default="experiments/level8_1/formal")
    parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--force",action="store_true")
    parser.add_argument("--smoke-test",action="store_true")
    args=parser.parse_args(); config=HierarchicalMemoryConfig()
    if args.smoke_test:
        args.seeds=[2026]; STAGES=((2,2,2),); MILESTONES=(1,2); SAMPLES=16
        if args.output=="experiments/level8_1/formal": args.output="experiments/level8_1/smoke"
    protocol={"architectures":list(ARCHITECTURES),"seeds":args.seeds,"chunk_size":CHUNK_SIZE,
              "matched_training_stages":STAGES,"milestones":MILESTONES,"samples_per_seed":SAMPLES,
              "chance":CHANCE,"hierarchical_config":config.to_dict(),
              "primary":"paired accuracy curve and rightmost Wilson lower bound above chance",
              "identical_training_and_evaluation_streams":True}
    if args.dry_run: print(json.dumps(protocol,indent=2)); return 0
    if not torch.cuda.is_available(): raise RuntimeError("Level 8.1 formal run requires CUDA")
    device=torch.device("cuda"); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root=ROOT/args.output; root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/"config.json",protocol); atomic_json(root/"run_metadata.json",run_metadata(device,args.seeds))
    runs=[]; training=[]; model_rows=[]
    for architecture in ARCHITECTURES:
        for seed in args.seeds:
            folder=root/architecture/f"seed{seed}"; folder.mkdir(parents=True,exist_ok=True)
            model,transfer=build(architecture,seed,config,device)
            history=train(model,architecture,seed,folder,device,dtype,args.force)
            training.append({"architecture":architecture,"seed":seed,"history":history})
            model_rows.append({"architecture":architecture,"seed":seed,"parameters":parameter_count(model),
                               "transferred":len(transfer["transferred"]) if transfer else None})
            model.eval()
            for replicate in range(SAMPLES//EVAL_BATCH):
                output=folder/f"eval_replicate{replicate}.json"
                if output.exists() and not args.force: row=json.loads(output.read_text(encoding="utf-8"))
                else: row=evaluate_replicate(model,architecture,seed,replicate,device,dtype); atomic_json(output,row)
                runs.append(row); atomic_json(root/"runs.partial.json",runs)
            del model; torch.cuda.empty_cache()
    summary=[]
    for architecture in ARCHITECTURES:
        for chunks in MILESTONES:
            values=[v for run in runs if run["architecture"]==architecture for row in run["rows"]
                    if row["chunks"]==chunks for v in row["correctness"]]
            interval=wilson(sum(values),len(values))
            summary.append({"architecture":architecture,"chunks":chunks,"correct":sum(values),
                            "samples":len(values),"accuracy":sum(values)/len(values),"wilson95":interval,
                            "above_chance":interval[0]>CHANCE})
    comparisons=[]
    keyed={(run["architecture"],run["seed"],run["replicate"]):run for run in runs}
    for chunks in MILESTONES:
        treatment=[];control=[]
        for seed in args.seeds:
            for replicate in range(SAMPLES//EVAL_BATCH):
                for arch,destination in (("hierarchical_v0_2",treatment),("v0_1",control)):
                    row=next(x for x in keyed[(arch,seed,replicate)]["rows"] if x["chunks"]==chunks)
                    destination+=row["correctness"]
        comparison=paired_exact(treatment,control);comparison["chunks"]=chunks;comparisons.append(comparison)
    lifetimes={arch:max((row["chunks"] for row in summary if row["architecture"]==arch and row["above_chance"]),default=0)
               for arch in ARCHITECTURES}
    result={"protocol":protocol,"lifetimes":lifetimes,"summary":summary,"paired_comparisons":comparisons,
            "models":model_rows,"training":training,"runs":runs}
    atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result)
    (root/"ANALYSIS.md").write_text("# Level 8.1 Analysis\n\nRun complete. Compare `lifetimes`, Wilson intervals, paired tests, and router diagnostics in `result.json`.\n",encoding="utf-8")
    print(json.dumps({"lifetimes":lifetimes,"summary":summary,"paired_comparisons":comparisons},indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
