"""Level 8.0: hierarchical Memory smoke, checkpoint, intervention, and v0.1 regression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from config import HierarchicalMemoryConfig
from experiment_utils import (ROOT, atomic_json, atomic_torch, parameter_count,
                              run_metadata, tensor_to_json)
from hierarchical_model import transfer_v0_1_weights
from model import build_model


OLD_CHECKPOINT = ROOT.parent / "ist_v0_1/experiments/level7_6_4/formal/ist-full_seed2026/stage_4096.pt"
INTERVENTIONS = ("normal", "zero_fast", "zero_slow", "zero_episodic",
                 "freeze_fast", "freeze_slow", "freeze_episodic",
                 "keep_only_fast", "keep_only_slow", "keep_only_episodic",
                 "roll_fast", "roll_slow", "roll_episodic",
                 "swap_fast", "swap_slow", "swap_episodic")


def state_shapes(state):
    keys = ("fast", "slow", "episodic_keys", "episodic_values", "episodic_usage",
            "episodic_age", "episodic_importance", "episodic_occupied")
    return [{key: list(layer[key].shape) for key in keys} for layer in state]


def measure_single_forward_peak(model, tokens, hierarchical: bool):
    if tokens.device.type != "cuda":
        return None
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if hierarchical:
            model(tokens, return_memory=True)
        else:
            model(tokens, return_memory=True, per_layer_memory=True)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1048576


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/level8_0/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = HierarchicalMemoryConfig()
    protocol = {"memory_arch": "hierarchical_v0_2", "batch": 2, "chunks": 2,
                "chunk_size": 64, "hidden_size": 64, "layers": 3,
                "config": config.to_dict(), "interventions": list(INTERVENTIONS),
                "checks": ["forward", "backward", "state shapes", "cross-chunk change",
                           "checkpoint exact recovery", "module ablations", "causal interventions",
                           "old checkpoint strict load and roundtrip"]}
    if args.dry_run:
        print(json.dumps(protocol, indent=2)); return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32)
    torch.manual_seed(800001); root = ROOT / args.output; root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "config.json", protocol); atomic_json(root / "run_metadata.json", run_metadata(device, 800001))
    model = build_model("hierarchical_v0_2", vocab_size=19, hidden_size=64, layers=3,
                        max_sequence_length=64, position_encoding="rope",
                        hierarchical_config=config).to(device)
    legacy = build_model("v0_1", vocab_size=19, hidden_size=64, layers=3,
                         max_sequence_length=64, position_encoding="rope", use_memory_fusion=True).to(device)
    parameter_rows = {"v0_1": parameter_count(legacy), "hierarchical_v0_2": parameter_count(model)}
    parameter_rows["added"] = parameter_rows["hierarchical_v0_2"] - parameter_rows["v0_1"]
    tokens1 = torch.randint(19, (2, 64), device=device); tokens2 = torch.randint(19, (2, 64), device=device)
    legacy_peak = measure_single_forward_peak(legacy, tokens1, False)
    v0_2_peak = measure_single_forward_peak(model, tokens1, True)
    forward_memory = {"v0_1_mb": legacy_peak, "hierarchical_v0_2_mb": v0_2_peak,
                      "delta_mb": None if legacy_peak is None else v0_2_peak - legacy_peak}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        logits1, state1, diagnostics1 = model(tokens1, return_memory=True, return_diagnostics=True)
        logits2, state2, diagnostics2 = model(tokens2, memory=state1, return_memory=True,
                                               return_diagnostics=True)
        target = torch.randint(16, (2,), device=device)
        loss = F.cross_entropy(logits2[:, -1, :16], target) + 0.1 * model.memory_diversity_loss()
    loss.backward(); gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); optimizer.step()
    state_changed = [float((b["slow"] - a["slow"]).abs().mean().detach().cpu())
                     for a, b in zip(state1, state2)]
    peak_mb = torch.cuda.max_memory_allocated() / 1048576 if device.type == "cuda" else None

    checkpoint_path = root / "checkpoint.pt"
    atomic_torch(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "memory": model.blocks[0].memory.detach_state(state2[0]),
                                   "config": config.to_dict()})
    recovered = build_model("hierarchical_v0_2", vocab_size=19, hidden_size=64, layers=3,
                            max_sequence_length=64, position_encoding="rope",
                            hierarchical_config=config).to(device).eval()
    recovered.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    model.eval()
    with torch.no_grad():
        reference, _ = model(tokens2, memory=state2, return_memory=True)
        restored, _ = recovered(tokens2, memory=state2, return_memory=True)
    recovery_delta = float((reference - restored).abs().max().cpu())

    intervention_rows = []
    for name in INTERVENTIONS:
        recovered.set_memory_intervention(name)
        with torch.no_grad(): output, _ = recovered(tokens2, memory=state2, return_memory=True)
        intervention_rows.append({"intervention": name,
                                  "max_logit_delta_vs_normal": float((output - reference).abs().max().cpu()),
                                  "finite": bool(torch.isfinite(output).all())})
    recovered.clear_memory_interventions()

    ablations = []
    for component in ("fast", "slow", "episodic", "router", "consolidation"):
        value = config.to_dict(); value[component]["enabled"] = False
        candidate = build_model("hierarchical_v0_2", vocab_size=19, hidden_size=64, layers=1,
                                max_sequence_length=64, hierarchical_config=value).to(device)
        with torch.no_grad(): output, memory = candidate(tokens1, return_memory=True)
        ablations.append({"disabled": component, "finite": bool(torch.isfinite(output).all()),
                          "state_shapes": state_shapes(memory)})

    regression = {"checkpoint_present": OLD_CHECKPOINT.exists(), "strict_load": False,
                  "roundtrip_max_delta": None, "transfer_count": 0}
    if OLD_CHECKPOINT.exists():
        old = torch.load(OLD_CHECKPOINT, map_location=device, weights_only=False)["model"]
        legacy.load_state_dict(old, strict=True); legacy.eval()
        clone = build_model("v0_1", vocab_size=19, hidden_size=64, layers=3,
                            max_sequence_length=64, position_encoding="rope", use_memory_fusion=True).to(device).eval()
        clone.load_state_dict(legacy.state_dict(), strict=True)
        regression_tokens = torch.randint(19, (1, 64), device=device)
        with torch.no_grad(): a, b = legacy(regression_tokens), clone(regression_tokens)
        regression.update({"strict_load": True, "roundtrip_max_delta": float((a-b).abs().max().cpu())})
        transfer_target = build_model("hierarchical_v0_2", vocab_size=19, hidden_size=64, layers=3,
                                      max_sequence_length=64, hierarchical_config=config).to(device)
        transfer = transfer_v0_1_weights(transfer_target, old)
        regression["transfer_count"] = len(transfer["transferred"])

    result = {"status": "pass" if recovery_delta == 0 and regression["strict_load"] and
              all(row["finite"] for row in intervention_rows + ablations) else "fail",
              "parameters": parameter_rows, "peak_memory_mb": peak_mb,
              "single_forward_peak_memory": forward_memory,
              "loss": float(loss.detach().cpu()), "gradient_norm": gradient_norm,
              "state_shapes": state_shapes(state2), "slow_state_change_by_layer": state_changed,
              "checkpoint_recovery_max_delta": recovery_delta, "v0_1_regression": regression,
              "interventions": intervention_rows, "ablations": ablations,
              "diagnostics_chunk1": tensor_to_json(diagnostics1),
              "diagnostics_chunk2": tensor_to_json(diagnostics2)}
    atomic_json(root / "raw_results.json", result); atomic_json(root / "result.json", result)
    analysis = ("# Level 8.0 Analysis\n\n"
                f"Status: **{result['status']}**\n\n"
                f"- v0.2 parameters: {parameter_rows['hierarchical_v0_2']:,}\n"
                f"- added parameters: {parameter_rows['added']:,}\n"
                f"- checkpoint recovery max delta: {recovery_delta}\n"
                f"- v0.1 strict checkpoint load: {regression['strict_load']}\n"
                f"- peak allocated Memory: {peak_mb} MB\n")
    (root / "ANALYSIS.md").write_text(analysis, encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "parameters", "peak_memory_mb",
                                                    "state_shapes", "checkpoint_recovery_max_delta",
                                                    "v0_1_regression")}, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
