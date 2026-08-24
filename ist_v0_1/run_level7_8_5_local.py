"""Level 7.8.5: all-key counterfactual training of the frozen IST readout side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level7_1_local import atomic_save
from run_level7_4_1_local import atomic_torch_save
from run_level7_7_local import paired_exact
from run_level7_8_4_local import (CHUNK_SIZE, FILLER_CHUNKS, SEEDS, VOCAB,
                                  forward_one, make_values, query_tokens)
from run_level7_8_4_1_local import (SAMPLES_PER_KEY_PER_SEED, evaluate_key,
                                    make_memory, summarize as strict_summarize)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/level7_8_4/formal"
BRANCHES = ("all_key_counterfactual", "single_key_equal_compute")
LOADS = (2, 4)
STEPS = {2: 400, 4: 400}
MEMORY_BATCH = 8
LR = 5e-4


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, CHUNK_SIZE, "rope", True).to(device)
    path = SOURCE / f"seed{seed}" / "stage_load16.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model


def configure_readout_only(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = model.blocks[2]
    modules = (layer.memory_read, layer.memory_fusion_gate, layer.ffn, layer.norm2, model.output)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def expand_memory(memory, repeats: int):
    return [item[:, None].expand(-1, repeats, -1, -1).reshape(
                item.size(0) * repeats, item.size(1), item.size(2)).detach()
            for item in memory]


def make_queries(values, load: int, branch: str, device):
    batch = values.size(0)
    if branch == "all_key_counterfactual":
        keys = torch.arange(load, device=device)[None, :].expand(batch, -1)
        targets = values
    else:
        selected = torch.randint(load, (batch,), device=device)
        keys = selected[:, None].expand(-1, load)
        targets = values[torch.arange(batch, device=device), selected][:, None].expand(-1, load)
    return keys.reshape(-1), targets.reshape(-1)


def save_state(path: Path, model, optimizer, load: int, step: int, history: list):
    atomic_torch_save(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "load": load, "step": step, "history": history})


def train(model, seed: int, branch: str, folder: Path, device, dtype, force: bool):
    trainable = configure_readout_only(model)
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    history = []
    for load in LOADS:
        final = folder / f"stage_load{load}.pt"
        resume = folder / f"stage_load{load}_resume.pt"
        if final.exists() and not force:
            state = torch.load(final, map_location=device, weights_only=False)
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
            print(f"branch={branch} seed={seed} load={load} already complete", flush=True)
            continue
        start = 0
        candidates = [path for path in (resume, resume.with_suffix(".pt.tmp")) if path.exists()]
        if candidates and not force:
            loaded = [(torch.load(path, map_location=device, weights_only=False), path) for path in candidates]
            state, selected = max(loaded, key=lambda item: int(item[0]["step"]))
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start = int(state["step"]); history = state["history"]
            print(f"resume branch={branch} seed={seed} load={load} step={start} "
                  f"source={selected.name}", flush=True)
        for step in range(start + 1, STEPS[load] + 1):
            set_seed(785000000 + seed * 10000 + load * 1000 + step)
            # The writer is locked and executed without a graph. Only the same
            # stored state is queried under different Key counterfactuals.
            model.eval()
            values = make_values(MEMORY_BATCH, load, device)
            with torch.no_grad():
                memory = make_memory(model, values, load, device, dtype)
            keys, targets = make_queries(values, load, branch, device)
            # Lock the random query background across both branches; only the
            # supervised Key topology is allowed to differ.
            set_seed(785500000 + seed * 10000 + load * 1000 + step)
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, _ = forward_one(model, query_tokens(keys, device),
                                    expand_memory(memory, load), dtype, detach=False)
            logits = logits[:, -1, :VOCAB]
            loss = F.cross_entropy(logits, targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
            if step == 1 or step % 25 == 0:
                row = {"load": load, "step": step, "loss": float(loss.detach()),
                       "accuracy": float((logits.argmax(-1) == targets).float().mean().detach()),
                       "unique_keys_per_memory": load if branch == "all_key_counterfactual" else 1,
                       "query_rows": MEMORY_BATCH * load}
                history.append(row); save_state(resume, model, optimizer, load, step, history)
                print(f"branch={branch} seed={seed} load={load} step={step} "
                      f"loss={row['loss']:.4f} accuracy={row['accuracy']:.2%}", flush=True)
        save_state(final, model, optimizer, load, STEPS[load], history)
    return history


def branch_comparison(runs):
    keyed = {(run["branch"], run["seed"], run["load"], run["key"]): run for run in runs}
    rows = []
    for load in LOADS:
        for key in range(load):
            treatment, control = [], []
            treatment_changed, control_changed = [], []
            for seed in SEEDS:
                treatment += keyed[("all_key_counterfactual", seed, load, key)]["correctness"]
                control += keyed[("single_key_equal_compute", seed, load, key)]["correctness"]
                treatment_changed += keyed[("all_key_counterfactual", seed, load, key)]["prediction_changed"]
                control_changed += keyed[("single_key_equal_compute", seed, load, key)]["prediction_changed"]
            result = paired_exact(treatment, control)
            result.update({"load": load, "key": key,
                           "all_key_accuracy": sum(treatment) / len(treatment),
                           "single_key_accuracy": sum(control) / len(control),
                           "all_key_prediction_change_rate": sum(treatment_changed) / len(treatment_changed),
                           "single_key_prediction_change_rate": sum(control_changed) / len(control_changed)})
            rows.append(result)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level7_8_5/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"source": "locked level7_8_4 stage_load16.pt", "seeds": list(SEEDS),
                "branches": list(BRANCHES), "loads": list(LOADS), "steps": STEPS,
                "memory_batch": MEMORY_BATCH, "learning_rate": LR,
                "writer": "frozen; Memory construction has no gradient",
                "trainable": "L3 memory_read, fusion gate, FFN, norm2, and output head",
                "all_key_branch": "every stored Memory is queried once for every Key",
                "control_branch": "one random Key per Memory repeated to equal query rows",
                "eval_samples_per_key_per_seed": SAMPLES_PER_KEY_PER_SEED,
                "primary": "strict per-key and Holm-corrected query-switch confirmation",
                "paired_queries": "identical query except one Key token"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2)); return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol,
                "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "dtype": str(dtype)})
    runs = []; training = []
    for branch in BRANCHES:
        for seed in SEEDS:
            folder = root / branch / f"seed{seed}"; folder.mkdir(parents=True, exist_ok=True)
            model = build(seed, device)
            history = train(model, seed, branch, folder, device, dtype, args.force)
            training.append({"branch": branch, "seed": seed, "history": history})
            model.eval()
            for load in LOADS:
                for key in range(load):
                    output = folder / f"eval_load{load}_key{key}.json"
                    if output.exists() and not args.force:
                        row = json.loads(output.read_text(encoding="utf-8"))
                    else:
                        row = evaluate_key(model, seed, load, key, device, dtype)
                        atomic_save(output, row)
                    row["branch"] = branch; runs.append(row)
                    atomic_save(root / "runs.partial.json", runs)
                    print(f"branch={branch} seed={seed} load={load} key={key} "
                          f"accuracy={sum(row['correctness']) / len(row['correctness']):.2%}", flush=True)
            del model; torch.cuda.empty_cache()
    branch_results = {}
    for branch in BRANCHES:
        selected = [run for run in runs if run["branch"] == branch]
        per_key, switches, strict = strict_summarize(selected)
        maximum = max((int(load) for load, row in strict.items() if row["strictly_confirmed"]), default=0)
        branch_results[branch] = {"strict_maximum_confirmed_load": maximum,
                                  "strict_decisions": strict, "per_key_summary": per_key,
                                  "query_switch_controls": switches}
    result = {"protocol": protocol, "branch_results": branch_results,
              "paired_branch_comparisons": branch_comparison(runs),
              "training": training, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"branch_results": branch_results,
                      "paired_branch_comparisons": result["paired_branch_comparisons"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
