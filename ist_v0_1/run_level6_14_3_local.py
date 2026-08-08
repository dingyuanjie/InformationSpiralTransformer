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

from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks
from run_level6_13_1_local import bootstrap_mean_ci, holm, mcnemar, save
from run_level6_14_1_local import choose_candidates, swap_slots


SEEDS = [606, 808, 1001]
SLOTS = 32
SWAP_TIMES = [4, 8]


def restore(memory, clean_memory, slots):
    output = [item.clone() for item in memory]
    output[2][:, slots, :] = clean_memory[2][:, slots, :]
    return output


def correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values))
    return result


def propagation_diagnostic(clean, polluted, block, source_slots):
    clean_final = clean[2].float()
    polluted_final = polluted[2].float()
    donor = torch.roll(clean_final, shifts=1, dims=0)
    delta = polluted_final - clean_final
    donor_delta = donor - clean_final
    delta_norm = delta.norm(dim=-1)
    relative = (delta_norm / clean_final.norm(dim=-1).clamp_min(1e-8)).mean(dim=0)
    dot = (delta * donor_delta).sum(dim=-1)
    projection = (dot / donor_delta.square().sum(dim=-1).clamp_min(1e-8)).mean(dim=0)
    diagnostics = block.memory.last_diagnostics
    compression = diagnostics["compression_weights"].float()  # [B, dest, token]
    memory_attention = diagnostics["memory_attention_weights"].float().mean(dim=1)
    routing = torch.bmm(compression, memory_attention).mean(dim=0)  # [dest, source]
    gate = diagnostics["update_gate"].float().mean(dim=(0, 2))
    source_routing = routing[:, source_slots].T
    return {
        "relative_l2": relative.cpu().tolist(),
        "donor_projection": projection.cpu().tolist(),
        "source_to_destination_routing": source_routing.cpu().tolist(),
        "source_routing_sum": source_routing.sum(dim=0).cpu().tolist(),
        "update_gate": gate.cpu().tolist(),
    }


@torch.no_grad()
def evaluate(model, args, condition, device, dtype, eval_seeds, capture=False):
    predictions = []
    targets = []
    donor_targets = []
    diagnostic_sums = None
    diagnostic_batches = 0
    batches = args.samples_per_eval_seed // args.eval_batch_size
    for eval_seed in eval_seeds:
        set_seed(eval_seed)
        for _ in range(batches):
            chunks, target, _ = make_chunks(
                args.eval_batch_size, args.chunks, args.chunk_size, device
            )
            clean_memory = None
            polluted_memory = None
            for chunk_index in range(args.chunks):
                chunk_number = chunk_index + 1
                needs_clean = (
                    condition["mode"] == "intact"
                    or condition["restore_slots"] is not None
                    or (capture and chunk_number <= condition["swap_after"] + 1)
                )
                with torch.autocast(device_type="cuda", dtype=dtype):
                    clean_logits = None
                    clean_produced = None
                    if needs_clean:
                        clean_logits, clean_produced = model(
                            chunks[:, chunk_index], memory=clean_memory, return_memory=True,
                            per_layer_memory=True,
                        )
                        clean_memory = clean_produced
                    if condition["mode"] == "intact":
                        polluted_logits, polluted_produced = clean_logits, clean_produced
                    else:
                        polluted_logits, polluted_produced = model(
                            chunks[:, chunk_index], memory=polluted_memory, return_memory=True,
                            per_layer_memory=True,
                        )
                    polluted_memory = polluted_produced
                    if condition["mode"] == "pollution":
                        if chunk_number == condition["swap_after"]:
                            polluted_memory = swap_slots(polluted_memory, condition["source_slots"])
                        propagation_chunk = condition["swap_after"] + 1
                        if capture and chunk_number == propagation_chunk:
                            diagnostic = propagation_diagnostic(
                                clean_memory, polluted_memory, model.blocks[2],
                                condition["source_slots"],
                            )
                            arrays = {key: np.asarray(value, dtype=float)
                                      for key, value in diagnostic.items()}
                            if diagnostic_sums is None:
                                diagnostic_sums = arrays
                            else:
                                for key in diagnostic_sums:
                                    diagnostic_sums[key] += arrays[key]
                            diagnostic_batches += 1
                        if (condition["restore_slots"] is not None
                                and chunk_number == propagation_chunk):
                            polluted_memory = restore(
                                polluted_memory, clean_memory, condition["restore_slots"]
                            )
            logits = clean_logits if condition["mode"] == "intact" else polluted_logits
            prediction = logits[:, -1, :16].argmax(-1)
            predictions.extend(prediction.cpu().tolist())
            targets.extend(target.cpu().tolist())
            donor_targets.extend(torch.roll(target, shifts=1, dims=0).cpu().tolist())
    prediction = np.asarray(predictions)
    target = np.asarray(targets)
    donor = np.asarray(donor_targets)
    correct = (prediction == target).astype(np.int8)
    mismatch = donor != target
    donor_hit = ((prediction == donor) & mismatch).astype(np.int8)
    result = {
        "samples": int(len(target)),
        "accuracy": float(correct.mean()),
        "donor_attraction": float(donor_hit.sum() / mismatch.sum()),
        "correct": correct.tolist(),
        "donor_hit": donor_hit.tolist(),
        "donor_mismatch": mismatch.astype(np.int8).tolist(),
    }
    if capture:
        result["propagation_diagnostic"] = {
            key: (value / diagnostic_batches).tolist()
            for key, value in diagnostic_sums.items()
        }
    return result


def destination_slots(candidate, limit=None):
    slots = [slot for slot in range(SLOTS) if slot not in candidate["slots"]]
    return slots if limit is None else slots[:limit]


def make_conditions(candidates, swap_times, destination_limit=None):
    output = [{"name": "intact", "mode": "intact", "pair": None,
               "source_slots": None, "restore_slots": None, "swap_after": None,
               "category": "baseline"}]
    for candidate in candidates:
        destinations = destination_slots(candidate, destination_limit)
        for swap_after in swap_times:
            base = {"mode": "pollution", "pair": candidate["key"],
                    "source_slots": candidate["slots"], "swap_after": swap_after,
                    "category": candidate["category"]}
            output.append({"name": f"{candidate['key']}__swap{swap_after}__hold",
                           **base, "restore_slots": None})
            output.append({"name": f"{candidate['key']}__swap{swap_after}__restore_source",
                           **base, "restore_slots": candidate["slots"]})
            output.append({"name": f"{candidate['key']}__swap{swap_after}__restore_all_final",
                           **base, "restore_slots": list(range(SLOTS))})
            for destination in destinations:
                output.append({
                    "name": f"{candidate['key']}__swap{swap_after}__restore_dest{destination}",
                    **base, "restore_slots": [destination],
                })
    return output


def masked_donor(row):
    mask = np.asarray(row["donor_mismatch"], dtype=bool)
    return np.asarray(row["donor_hit"], dtype=np.int8)[mask]


def analyze(seed, candidates, rows, args):
    table = {row["name"]: row for row in rows}
    intact = table["intact"]
    graphs = []
    for candidate in candidates:
        destinations = destination_slots(candidate, args.destination_limit)
        for swap_after in args.swap_times:
            prefix = f"{candidate['key']}__swap{swap_after}"
            hold = table[f"{prefix}__hold"]
            source = table[f"{prefix}__restore_source"]
            all_final = table[f"{prefix}__restore_all_final"]
            hold_correct = np.asarray(hold["correct"], dtype=np.int8)
            hold_donor = masked_donor(hold)
            tests_accuracy = []
            tests_donor = []
            edges = []
            for index, destination in enumerate(destinations):
                row = table[f"{prefix}__restore_dest{destination}"]
                corrected = np.asarray(row["correct"], dtype=np.int8)
                corrected_donor = masked_donor(row)
                accuracy_test = mcnemar(corrected, hold_correct)
                donor_test = mcnemar(corrected_donor, hold_donor)
                tests_accuracy.append(accuracy_test)
                tests_donor.append(donor_test)
                edges.append({
                    "destination": destination,
                    "accuracy": row["accuracy"],
                    "donor_attraction": row["donor_attraction"],
                    "accuracy_gain": bootstrap_mean_ci(
                        corrected - hold_correct,
                        args.bootstrap_seed + seed * 10000 + swap_after * 100 + index,
                        args.bootstrap_iterations,
                    ),
                    "donor_reduction": bootstrap_mean_ci(
                        hold_donor - corrected_donor,
                        args.bootstrap_seed + seed * 10000 + 5000 + swap_after * 100 + index,
                        args.bootstrap_iterations,
                    ),
                    "mcnemar_accuracy": accuracy_test,
                    "mcnemar_donor": donor_test,
                })
            holm(tests_accuracy)
            holm(tests_donor)
            diagnostic = hold["propagation_diagnostic"]
            for edge, accuracy_test, donor_test in zip(edges, tests_accuracy, tests_donor):
                destination = edge["destination"]
                edge["mcnemar_accuracy"] = accuracy_test
                edge["mcnemar_donor"] = donor_test
                edge["relative_l2"] = diagnostic["relative_l2"][destination]
                edge["donor_projection"] = diagnostic["donor_projection"][destination]
                edge["source_routing"] = [
                    row[destination] for row in diagnostic["source_to_destination_routing"]
                ]
                edge["source_routing_sum"] = diagnostic["source_routing_sum"][destination]
                edge["update_gate"] = diagnostic["update_gate"][destination]
            gain = [edge["accuracy_gain"]["estimate"] for edge in edges]
            donor_reduction = [edge["donor_reduction"]["estimate"] for edge in edges]
            predictors = {
                "relative_l2": [edge["relative_l2"] for edge in edges],
                "donor_projection": [edge["donor_projection"] for edge in edges],
                "source_routing_sum": [edge["source_routing_sum"] for edge in edges],
                "update_gate": [edge["update_gate"] for edge in edges],
            }
            correlations = {}
            for name, values in predictors.items():
                correlations[name] = {
                    "accuracy_gain_pearson": correlation(values, gain),
                    "accuracy_gain_spearman": correlation(ranks(values), ranks(gain)),
                    "donor_reduction_pearson": correlation(values, donor_reduction),
                }
            graphs.append({
                **candidate,
                "swap_after": swap_after,
                "hold_accuracy": hold["accuracy"],
                "hold_donor_attraction": hold["donor_attraction"],
                "restore_source_accuracy": source["accuracy"],
                "restore_all_final_accuracy": all_final["accuracy"],
                "diagnostic": diagnostic,
                "edges": edges,
                "correlations": correlations,
                "top_accuracy_destinations": sorted(
                    edges, key=lambda edge: edge["accuracy_gain"]["estimate"], reverse=True
                )[:8],
                "top_donor_destinations": sorted(
                    edges, key=lambda edge: edge["donor_reduction"]["estimate"], reverse=True
                )[:8],
            })
    return {
        "seed": seed,
        "samples_per_condition": intact["samples"],
        "intact_accuracy": intact["accuracy"],
        "candidates": candidates,
        "graphs": graphs,
    }


def plot_pair(result, candidate, path, swap_times):
    fig, axes = plt.subplots(len(swap_times), 4, figsize=(17, 4 * len(swap_times)),
                             squeeze=False)
    for row_index, swap_after in enumerate(swap_times):
        graph = next(x for x in result["graphs"]
                     if x["key"] == candidate["key"] and x["swap_after"] == swap_after)
        edges = sorted(graph["edges"], key=lambda edge: edge["destination"])
        slots = [edge["destination"] for edge in edges]
        gain = np.array([edge["accuracy_gain"]["estimate"] for edge in edges]) * 100
        donor = np.array([edge["donor_reduction"]["estimate"] for edge in edges]) * 100
        routing = np.array([edge["source_routing"] for edge in edges]).T
        representation = np.array([edge["donor_projection"] for edge in edges])
        route_sum = np.array([edge["source_routing_sum"] for edge in edges])
        axes[row_index, 0].bar(slots, gain)
        axes[row_index, 0].axhline(0, color="black", linewidth=0.8)
        axes[row_index, 0].set_ylabel(f"Swap {swap_after}\nAccuracy gain (pp)")
        axes[row_index, 0].set_title("Restore-one causal effect")
        image = axes[row_index, 1].imshow(routing, aspect="auto", cmap="viridis")
        axes[row_index, 1].set_yticks(range(len(candidate["slots"])), candidate["slots"])
        axes[row_index, 1].set_xticks(range(0, len(slots), 4), slots[::4])
        axes[row_index, 1].set_title("Source → destination routing")
        fig.colorbar(image, ax=axes[row_index, 1], fraction=0.046, pad=0.04)
        axes[row_index, 2].scatter(representation, gain)
        axes[row_index, 2].set_xlabel("Donor projection")
        axes[row_index, 2].set_ylabel("Accuracy gain (pp)")
        axes[row_index, 2].set_title("Representation vs causality")
        axes[row_index, 3].scatter(route_sum, gain, c=donor, cmap="coolwarm")
        axes[row_index, 3].set_xlabel("Source routing sum")
        axes[row_index, 3].set_ylabel("Accuracy gain (pp)")
        axes[row_index, 3].set_title("Routing vs causality")
        for axis in axes[row_index]:
            axis.grid(alpha=0.15)
    fig.suptitle(f"IST seed {result['seed']} pair {candidate['key']}: slot propagation graph")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_seed(seed, candidates, args, device, dtype, root):
    folder = root / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists() and not args.force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    source = Path(args.level6_8_root) / f"seed{seed}" / "withdrawal_phase3.pt"
    state = torch.load(source, map_location=device, weights_only=False)
    model = InformationSpiralTransformer(19, 64, 3, args.chunk_size, "rope", True).to(device)
    model.load_state_dict(state["model"])
    model.blocks[2].memory.capture_memory_attention_weights = True
    eval_seeds = [args.eval_seed_base + seed * 100 + index
                  for index in range(args.eval_seed_count)]
    specs = make_conditions(candidates, args.swap_times, args.destination_limit)
    progress_path = folder / "predictions.json"
    rows = [] if args.force or not progress_path.exists() else json.loads(
        progress_path.read_text(encoding="utf-8")
    )
    done = {row["name"] for row in rows}
    for index, condition in enumerate(specs, start=1):
        if condition["name"] in done:
            continue
        capture = condition["restore_slots"] is None and condition["mode"] == "pollution"
        metric = evaluate(model, args, condition, device, dtype, eval_seeds, capture=capture)
        rows.append({**condition, **metric})
        save(progress_path, rows)
        print(f"seed={seed} [{index}/{len(specs)}] {condition['name']} "
              f"acc={metric['accuracy']:.2%} donor={metric['donor_attraction']:.2%}", flush=True)
    result = analyze(seed, candidates, rows, args)
    result["eval_seeds"] = eval_seeds
    save(result_path, result)
    for candidate in candidates:
        plot_pair(result, candidate, folder / f"pair_{candidate['key']}_propagation_graph.png",
                  args.swap_times)
    return result


def main():
    parser = argparse.ArgumentParser(description="Level 6.14.3 slot-to-slot causal propagation graph")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level6-8-root", default="experiments/level6_8/formal")
    parser.add_argument("--level6-13-1-root", default="experiments/level6_13_1/formal")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--swap-times", nargs="+", type=int, default=SWAP_TIMES)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-eval-seed", type=int, default=400)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--eval-seed-base", type=int, default=7143000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=614300)
    parser.add_argument("--destination-limit", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", default="experiments/level6_14_3/formal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.samples_per_eval_seed % args.eval_batch_size:
        raise ValueError("samples-per-eval-seed must be divisible by eval-batch-size")
    if any(time < 1 or time >= args.chunks - 1 for time in args.swap_times):
        raise ValueError("swap times must leave at least one propagation and one query chunk")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    registration = json.loads((Path(args.level6_13_1_root)
                               / "preregistered_candidates.json").read_text(encoding="utf-8"))
    results = []
    for seed in args.seeds:
        candidates = choose_candidates(registration, seed)
        results.append(run_seed(seed, candidates, args, device, dtype, root))
        save(root / "runs.partial.json", results)
        torch.cuda.empty_cache()
    save(root / "summary.json", {"protocol": vars(args), "runs": results})
    print(json.dumps({str(result["seed"]): {
        "intact": result["intact_accuracy"],
        "graphs": len(result["graphs"]),
    } for result in results}, indent=2))


if __name__ == "__main__":
    main()
