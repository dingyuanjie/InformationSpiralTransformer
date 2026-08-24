"""Level 7.8.4.1: locked per-key confirmation and query-key switching control."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from long_context_test import set_seed
from run_level7_1_local import atomic_save
from run_level7_7_local import paired_exact
from run_level7_8_4_local import (CHANCE, CHUNK_SIZE, FILLER_CHUNKS, PARENT,
                                  SEEDS, VOCAB, base_chunk, forward_one,
                                  make_values, route_memory,
                                  write_tokens)
from model import InformationSpiralTransformer


ROOT = Path(__file__).resolve().parent
TRAINED = ROOT / "experiments/level7_8_4/formal"
LOADS = (2, 4)
SAMPLES_PER_KEY_PER_SEED = 128
BATCH = 8


def build(seed: int, device: torch.device):
    model = InformationSpiralTransformer(19, 64, 3, CHUNK_SIZE, "rope", True).to(device).eval()
    path = TRAINED / f"seed{seed}" / "stage_load16.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model


@torch.no_grad()
def make_memory(model, values, load: int, device, dtype):
    batch = values.size(0); keys = torch.arange(load, device=device); memory = None
    for write_index in range(load):
        tokens, _ = write_tokens(keys[write_index].expand(batch), values[:, write_index],
                                 write_index, device)
        _, candidate = forward_one(model, tokens, memory, dtype, detach=True)
        memory = route_memory("oracle_marker_l3_gate", memory, candidate, True, write_index == 0)
    for _ in range(FILLER_CHUNKS):
        _, candidate = forward_one(model, base_chunk(batch, device), memory, dtype, detach=True)
        memory = route_memory("oracle_marker_l3_gate", memory, candidate, False, False)
    return memory


@torch.no_grad()
def query(model, memory, tokens, dtype):
    logits, _ = forward_one(model, tokens,
                            [item.clone() for item in memory], dtype, detach=True)
    return logits[:, -1, :VOCAB].argmax(-1)


@torch.no_grad()
def evaluate_key(model, seed: int, load: int, key: int, device, dtype):
    correct = []; switched_correct = []; switched_old = []; changed = []
    predictions = []; switched_predictions = []; targets = []; switched_targets = []
    switch_key = (key + 1) % load
    for start in range(0, SAMPLES_PER_KEY_PER_SEED, BATCH):
        set_seed(784910000 + seed * 10000 + load * 1000 + key * 200 + start)
        values = make_values(BATCH, load, device)
        memory = make_memory(model, values, load, device, dtype)
        # The paired queries differ in exactly one token: the requested Key.
        original_tokens = base_chunk(BATCH, device)
        original_tokens[:, -3] = 18; original_tokens[:, -2] = key; original_tokens[:, -1] = 16
        alternate_tokens = original_tokens.clone(); alternate_tokens[:, -2] = switch_key
        original_prediction = query(model, memory, original_tokens, dtype)
        alternate_prediction = query(model, memory, alternate_tokens, dtype)
        original_target = values[:, key]; alternate_target = values[:, switch_key]
        correct += (original_prediction == original_target).int().cpu().tolist()
        switched_correct += (alternate_prediction == alternate_target).int().cpu().tolist()
        switched_old += (alternate_prediction == original_target).int().cpu().tolist()
        changed += (alternate_prediction != original_prediction).int().cpu().tolist()
        predictions += original_prediction.cpu().tolist()
        switched_predictions += alternate_prediction.cpu().tolist()
        targets += original_target.cpu().tolist(); switched_targets += alternate_target.cpu().tolist()
    return {"seed": seed, "load": load, "key": key, "switch_key": switch_key,
            "samples": SAMPLES_PER_KEY_PER_SEED, "correctness": correct,
            "switched_correctness": switched_correct,
            "switched_old_correctness": switched_old, "prediction_changed": changed,
            "predictions": predictions, "switched_predictions": switched_predictions,
            "targets": targets, "switched_targets": switched_targets}


def wilson(correct: int, samples: int, z: float = 1.959963984540054):
    p = correct / samples; scale = 1 + z*z/samples
    middle = (p + z*z/(2*samples))/scale
    half = z*math.sqrt(p*(1-p)/samples + z*z/(4*samples*samples))/scale
    return [middle-half, middle+half]


def metric(values):
    correct, samples = sum(values), len(values); interval = wilson(correct, samples)
    return {"correct": correct, "samples": samples, "accuracy": correct / samples,
            "wilson95": interval, "above_chance": interval[0] > CHANCE}


def holm(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["mcnemar_exact_p"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["mcnemar_exact_p"])
        running = max(running, adjusted)
        rows[index]["holm_p"] = running


def summarize(runs):
    per_key = []
    switch_controls = []
    strict = {}
    for load in LOADS:
        load_rows = []
        for key in range(load):
            selected = [run for run in runs if run["load"] == load and run["key"] == key]
            correct = [v for run in selected for v in run["correctness"]]
            switched = [v for run in selected for v in run["switched_correctness"]]
            old = [v for run in selected for v in run["switched_old_correctness"]]
            changed = [v for run in selected for v in run["prediction_changed"]]
            row = {"load": load, "key": key, "switch_key": (key + 1) % load,
                   "correct_query": metric(correct), "switched_query": metric(switched),
                   "switched_still_old": metric(old),
                   "prediction_change_rate": sum(changed) / len(changed),
                   "seed_accuracies": {str(run["seed"]): sum(run["correctness"]) / len(run["correctness"])
                                       for run in selected}}
            per_key.append(row); load_rows.append(row)
            causal = paired_exact(switched, old)
            causal.update({"load": load, "source_key": key, "switch_key": (key + 1) % load,
                           "switched_target_accuracy": sum(switched) / len(switched),
                           "old_target_leakage": sum(old) / len(old)})
            switch_controls.append(causal)
        load_controls = [row for row in switch_controls if row["load"] == load]
        holm(load_controls)
        strict[str(load)] = {
            "all_keys_above_chance": all(row["correct_query"]["above_chance"] for row in load_rows),
            "all_switches_above_chance": all(row["switched_query"]["above_chance"] for row in load_rows),
            "all_switches_prefer_new_key": all(
                row["switched_target_accuracy"] > row["old_target_leakage"] and
                row["holm_p"] < 0.05 for row in load_controls),
        }
        strict[str(load)]["strictly_confirmed"] = all(strict[str(load)].values())
    return per_key, switch_controls, strict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level7_8_4_1/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = {"model": "locked level7_8_4 stage_load16.pt", "no_training": True,
                "seeds": list(SEEDS), "loads": list(LOADS),
                "samples_per_key_per_seed": SAMPLES_PER_KEY_PER_SEED,
                "pooled_samples_per_key": SAMPLES_PER_KEY_PER_SEED * len(SEEDS),
                "routing": "oracle marker L3 gate", "filler_chunks": FILLER_CHUNKS,
                "primary": "every key's Wilson lower bound exceeds 1/16 chance",
                "switch_control": "on identical Memory, change query key k to (k+1)%load",
                "strict_success": "all keys and switched queries above chance, and every switch "
                "significantly favors the switched key value over old-key leakage after within-load Holm correction"}
    if args.dry_run:
        print(json.dumps(protocol, indent=2)); return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root = ROOT / args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_save(root / "protocol.json", {"protocol": protocol,
                "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "dtype": str(dtype)})
    runs = []
    for seed in SEEDS:
        model = build(seed, device)
        for load in LOADS:
            for key in range(load):
                output = root / f"seed{seed}_load{load}_key{key}.json"
                if output.exists() and not args.force:
                    row = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row = evaluate_key(model, seed, load, key, device, dtype)
                    atomic_save(output, row)
                runs.append(row); atomic_save(root / "runs.partial.json", runs)
                print(f"seed={seed} load={load} key={key} "
                      f"accuracy={sum(row['correctness']) / len(row['correctness']):.2%}", flush=True)
        del model; torch.cuda.empty_cache()
    per_key, switches, strict = summarize(runs)
    maximum = max((int(load) for load, row in strict.items() if row["strictly_confirmed"]), default=0)
    result = {"protocol": protocol, "strict_maximum_confirmed_load": maximum,
              "strict_decisions": strict, "per_key_summary": per_key,
              "query_switch_controls": switches, "runs": runs}
    atomic_save(root / "result.json", result)
    print(json.dumps({"strict_maximum_confirmed_load": maximum,
                      "strict_decisions": strict, "per_key_summary": per_key,
                      "query_switch_controls": switches}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
