"""Level 7.7: mechanism-driven Memory-bank-dropout stabilization training."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from long_context_test import set_seed
from marked_retrieval_level2 import make_batch
from model import InformationSpiralTransformer
from run_level7_1_local import atomic_save
from run_level7_4_1_local import atomic_torch_save
from run_level7_6_local import SEEDS


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/level7_6_4/formal"
BRANCHES = {
    "control_continue": {"dropout_probability": 0.0, "dropout_size": 8},
    "bankdrop_k8_p50": {"dropout_probability": 0.5, "dropout_size": 8},
}
TRAIN_LENGTH = 4096
TRAIN_STEPS = 200
EVAL_LENGTH = 32768
WINDOWS = {"near": (16, 4095), "far": (16384, 32766)}
EVAL_SAMPLES = 100
CHANCE = 1 / 16


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, EVAL_LENGTH, "rope", True).to(device)
    checkpoint = torch.load(PARENT / f"ist-full_seed{seed}" / "stage_4096.pt",
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def configure(model, branch: str) -> None:
    settings = BRANCHES[branch]
    for block in model.blocks:
        block.memory_bank_dropout_probability = settings["dropout_probability"]
        block.memory_bank_dropout_size = settings["dropout_size"]


def save_training(path: Path, model, optimizer, step: int, history: list) -> None:
    atomic_torch_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "step": step, "history": history})


def train(model, optimizer, branch: str, seed: int, folder: Path,
          device: torch.device, dtype: torch.dtype, force: bool) -> list:
    final = folder / "trained.pt"
    resume = folder / "resume.pt"
    if final.exists() and not force:
        state = torch.load(final, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        return state["history"]
    start, history = 0, []
    candidates = [path for path in (resume, resume.with_suffix(".pt.tmp")) if path.exists()]
    if candidates and not force:
        loaded = [(torch.load(path, map_location=device, weights_only=False), path) for path in candidates]
        state, selected = max(loaded, key=lambda item: int(item[0]["step"]))
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start, history = int(state["step"]), state["history"]
        print(f"resume branch={branch} seed={seed} step={start} source={selected.name}", flush=True)
    configure(model, branch)
    for step in range(start + 1, TRAIN_STEPS + 1):
        # Reset before batch creation: both branches see identical examples at every step.
        set_seed(770000000 + seed * 1000 + step)
        model.train()
        tokens, target, position = make_batch(1, TRAIN_LENGTH, TRAIN_LENGTH - 3, 16, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(tokens)[..., :16]
            query_loss = F.cross_entropy(logits[:, -1], target)
            local_loss = F.cross_entropy(logits[torch.arange(len(target), device=device), position], target)
            loss = query_loss + 0.5 * local_loss + 0.1 * model.memory_diversity_loss()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            row = {"step": step, "loss": float(loss.detach()),
                   "query_loss": float(query_loss.detach()), "local_loss": float(local_loss.detach())}
            history.append(row); save_training(resume, model, optimizer, step, history)
            print(f"branch={branch} seed={seed} step={step} loss={row['loss']:.4f}", flush=True)
    save_training(final, model, optimizer, TRAIN_STEPS, history)
    return history


def make_example(window: tuple[int, int], sample: int, seed: int, device: torch.device):
    set_seed(seed)
    target_value = sample % 16
    target = torch.tensor([target_value], device=device)
    tokens = torch.randint(16, (1, EVAL_LENGTH), device=device)
    distance = int(torch.randint(window[0], window[1] + 1, (1,), device=device).item())
    position = EVAL_LENGTH - 2 - distance
    tokens[0, position] = 17; tokens[0, position + 1] = target
    tokens[0, -2] = 18; tokens[0, -1] = 16
    return tokens, target


@torch.no_grad()
def evaluate(model, branch: str, seed: int, window_name: str,
             device: torch.device, dtype: torch.dtype) -> dict:
    model.eval()  # Structured dropout is training-only by construction.
    correctness = []
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    for sample in range(EVAL_SAMPLES):
        example_seed = 770100000 + seed * 1000 + (0 if window_name == "near" else 500) + sample
        tokens, target = make_example(WINDOWS[window_name], sample, example_seed, device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            prediction = model(tokens)[..., :16][:, -1].argmax(-1)
        correctness.append(int(prediction.item() == target.item()))
    torch.cuda.synchronize(); seconds = time.perf_counter() - started
    return {"branch": branch, "seed": seed, "window": window_name, "samples": EVAL_SAMPLES,
            "correct": sum(correctness), "accuracy": sum(correctness) / EVAL_SAMPLES,
            "correctness": correctness, "seconds": seconds,
            "tokens_per_second": EVAL_SAMPLES * EVAL_LENGTH / seconds,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1048576}


def wilson(correct: int, samples: int, z: float = 1.959963984540054):
    p = correct / samples; scale = 1 + z*z/samples
    middle = (p + z*z/(2*samples))/scale
    half = z*math.sqrt(p*(1-p)/samples + z*z/(4*samples*samples))/scale
    return [middle-half, middle+half]


def paired_exact(treatment: list[int], control: list[int]):
    improved = sum(a == 1 and b == 0 for a,b in zip(treatment,control))
    harmed = sum(a == 0 and b == 1 for a,b in zip(treatment,control)); n=improved+harmed
    tail = sum(math.comb(n,k) for k in range(min(improved,harmed)+1))/2**n if n else .5
    return {"difference": (sum(treatment)-sum(control))/len(control), "improved": improved,
            "harmed": harmed, "ties": len(control)-n, "mcnemar_exact_p": min(1.0,2*tail)}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="experiments/level7_7/formal")
    parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    protocol={"branches":BRANCHES,"seeds":SEEDS,"source":"level7_6_4/stage_4096.pt",
              "train":{"length":TRAIN_LENGTH,"steps":TRAIN_STEPS,"batch":1,"identical_examples":True},
              "eval":{"length":EVAL_LENGTH,"windows":WINDOWS,"samples_per_seed_window":EVAL_SAMPLES,
                      "balanced_targets":True,"dropout_disabled":True},
              "primary":"paired bankdrop vs control 32K accuracy and successful-seed count"}
    if args.dry_run: print(json.dumps(protocol,indent=2)); return 0
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    device=torch.device("cuda"); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root=ROOT/args.output; root.mkdir(parents=True,exist_ok=True)
    atomic_save(root/"protocol.json",{"protocol":protocol,"gpu":torch.cuda.get_device_name(device),"torch":torch.__version__,"dtype":str(dtype)})
    runs=[]
    for branch in BRANCHES:
        for seed in SEEDS:
            folder=root/f"{branch}_seed{seed}"; folder.mkdir(parents=True,exist_ok=True)
            model,optimizer=build(seed,device); history=train(model,optimizer,branch,seed,folder,device,dtype,args.force)
            tests=[]
            for window_name in WINDOWS:
                output=folder/f"eval_{window_name}.json"
                if output.exists() and not args.force: row=json.loads(output.read_text(encoding="utf-8"))
                else: row=evaluate(model,branch,seed,window_name,device,dtype); atomic_save(output,row)
                tests.append(row); print(f"branch={branch} seed={seed} window={window_name} accuracy={row['accuracy']:.2%}",flush=True)
            run={"branch":branch,"seed":seed,"history":history,"tests":tests}; runs.append(run)
            atomic_save(root/"runs.partial.json",runs); del model,optimizer; torch.cuda.empty_cache()
    summary=[]
    for branch in BRANCHES:
        selected=[run for run in runs if run["branch"]==branch]
        seed_rows=[]
        for run in selected:
            c=sum(x["correct"] for x in run["tests"]); n=sum(x["samples"] for x in run["tests"]); interval=wilson(c,n)
            seed_rows.append({"seed":run["seed"],"correct":c,"samples":n,"accuracy":c/n,
                              "wilson95":interval,"above_chance":interval[0]>CHANCE})
        summary.append({"branch":branch,"seed_results":seed_rows,
                        "successful_seed_count":sum(x["above_chance"] for x in seed_rows),
                        "mean_seed_accuracy":sum(x["accuracy"] for x in seed_rows)/len(seed_rows)})
    keyed={(run["branch"],run["seed"]):run for run in runs}; treatment=[]; control=[]
    for seed in SEEDS:
        treatment += [v for row in keyed[("bankdrop_k8_p50",seed)]["tests"] for v in row["correctness"]]
        control += [v for row in keyed[("control_continue",seed)]["tests"] for v in row["correctness"]]
    comparison=paired_exact(treatment,control)
    result={"protocol":protocol,"summary":summary,"paired_bankdrop_vs_control":comparison,"runs":runs}
    atomic_save(root/"result.json",result); print(json.dumps({"summary":summary,"comparison":comparison},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
