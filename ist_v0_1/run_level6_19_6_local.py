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

from run_level6_18_6_local import configure_cuda, paired_statistics, save
from run_level6_18_7_local import holm_adjust
from run_level6_18_8_local import continuous_effect
from run_level6_19_1_local import load_frozen, tensor_fingerprint
from run_level6_19_4_local import parameter_fingerprint, query_downstream
from run_level6_19_5_local import (
    DirectionDistiller,
    ObservableScalarProbe,
    binary_observability_metrics,
    cache_batch,
    collect_cache,
    load_parent_routers,
    router_inputs,
)


LEVEL = "6.19.6"
SEED = 707
CHUNKS = 16
DIAGNOSTIC_SEED = 6196100
ANALYSIS_SEED = 6196200
RECOVERY_THRESHOLD = 0.25
FULL_ACCURACY_NONINFERIORITY = 0.0025

CONDITIONS = [
    "source",
    "frozen_signed_router",
    "factorized_signed_candidate",
    "factorized_residual_control",
    "factorized_signed_shuffled_memory",
    "factorized_signed_rolled_delta",
    "factorized_signed_head_permuted",
    "full_signed_oracle",
]

SPECIFICITY_CONTROLS = [
    "source",
    "frozen_signed_router",
    "factorized_residual_control",
    "factorized_signed_shuffled_memory",
    "factorized_signed_rolled_delta",
    "factorized_signed_head_permuted",
]


def load_level6_19_5_probes(args, device):
    result = json.loads(
        Path(args.parent_result).read_text(encoding="utf-8")
    )
    if (
        not result.get("integrity", {}).get("passed")
        or result.get("analysis", {}).get("diagnosis", {}).get(
            "classification"
        ) != "joint_calibration_coupling_bottleneck"
    ):
        raise RuntimeError("Level 6.19.5 canonical parent did not pass")
    saved = torch.load(
        args.parent_probes, map_location="cpu", weights_only=False
    )
    parent_protocol = result["protocol"]["protocol"]
    hidden = parent_protocol["probe_hidden"]
    basis_seed = parent_protocol["residual_basis_seed"]
    dose_cap = parent_protocol["dose_prediction_cap"]
    if abs(dose_cap - args.dose_prediction_cap) > 1e-12:
        raise RuntimeError("Level 6.19.5 dose cap does not match protocol")
    head_mask = saved["head_mask"].float().to(device)
    probes = {
        "classifier": ObservableScalarProbe(hidden).to(device),
        "dose": ObservableScalarProbe(hidden).to(device),
        "signed_direction": DirectionDistiller(
            "signed", hidden, head_mask, basis_seed
        ).to(device),
        "residual_direction": DirectionDistiller(
            "residual", hidden, head_mask, basis_seed
        ).to(device),
    }
    for name, probe in probes.items():
        probe.load_state_dict(saved["states"][name])
        probe.eval()
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
    return probes, saved["dose_stats"], result, saved


def predicted_dose(probe, inputs, dose_stats, cap):
    raw = probe(*inputs)
    return torch.expm1(
        raw * dose_stats["std"] + dose_stats["mean"]
    ).clamp(min=0.0, max=cap)


def append_condition(parts, name, logits, labels, competitor, delta):
    rows = torch.arange(len(labels), device=labels.device)
    fixed_margin = logits[rows, labels] - logits[rows, competitor]
    parts[name]["predictions"].append(logits.argmax(dim=-1).cpu())
    parts[name]["fixed_margin"].append(fixed_margin.cpu())
    parts[name]["context_delta_norm"].append(delta.norm(dim=-1).cpu())


def evaluate_formal(model, parent_routers, probes, dose_stats, cache, args,
                    device, dtype, root):
    parts = {
        name: {
            field: [] for field in [
                "predictions", "fixed_margin", "context_delta_norm"
            ]
        }
        for name in CONDITIONS
    }
    classifier_logits = []
    factorized_dose = []
    labels_parts = []
    competitor_parts = []
    memory_prediction_parts = []
    candidate_l2_error = 0.0
    residual_l2_error = 0.0
    shuffled_l2_error = 0.0
    rolled_l2_error = 0.0
    head_permuted_l2_error = 0.0
    signed_unit_error = 0.0
    residual_unit_error = 0.0
    for start in range(0, len(cache["labels"]), args.probe_batch_size):
        end = min(start + args.probe_batch_size, len(cache["labels"]))
        batch = cache_batch(cache, slice(start, end), device)
        inputs = router_inputs(batch)
        with torch.no_grad():
            classifier = probes["classifier"](*inputs)
            dose = predicted_dose(
                probes["dose"], inputs, dose_stats,
                args.dose_prediction_cap,
            )
            signed = probes["signed_direction"](*inputs)
            residual = probes["residual_direction"](*inputs)
            frozen = parent_routers["signed"](*inputs)["delta"]
            shuffled_inputs = (
                batch["query"],
                batch["pre_fusion"],
                batch["source_context"].roll(1, 0),
                batch["source_attention"].roll(1, 0),
                batch["atoms"].roll(1, 0),
            )
            shuffled_dose = predicted_dose(
                probes["dose"], shuffled_inputs, dose_stats,
                args.dose_prediction_cap,
            )
            shuffled_unit = probes["signed_direction"](
                *shuffled_inputs
            )["unit"]
        candidate_delta = signed["unit"] * dose[:, None]
        residual_delta = residual["unit"] * dose[:, None]
        shuffled_delta = shuffled_unit * shuffled_dose[:, None]
        rolled_delta = candidate_delta.roll(1, 0)
        permuted_coefficients = signed["coefficients"].roll(1, 1)
        permuted_direction = torch.einsum(
            "bhs,bhsd->bd", permuted_coefficients, batch["atoms"]
        )
        permuted_unit = permuted_direction / permuted_direction.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        permuted_delta = permuted_unit * dose[:, None]
        oracle_delta = batch["oracle_delta"]
        deltas = {
            "source": torch.zeros_like(candidate_delta),
            "frozen_signed_router": frozen,
            "factorized_signed_candidate": candidate_delta,
            "factorized_residual_control": residual_delta,
            "factorized_signed_shuffled_memory": shuffled_delta,
            "factorized_signed_rolled_delta": rolled_delta,
            "factorized_signed_head_permuted": permuted_delta,
            "full_signed_oracle": oracle_delta,
        }
        signed_unit_error = max(
            signed_unit_error,
            (signed["unit"].norm(dim=-1) - 1.0).abs().max().item(),
        )
        residual_unit_error = max(
            residual_unit_error,
            (residual["unit"].norm(dim=-1) - 1.0).abs().max().item(),
        )
        candidate_l2_error = max(
            candidate_l2_error,
            (candidate_delta.norm(dim=-1) - dose).abs().max().item(),
        )
        residual_l2_error = max(
            residual_l2_error,
            (residual_delta.norm(dim=-1) - dose).abs().max().item(),
        )
        shuffled_l2_error = max(
            shuffled_l2_error,
            (
                shuffled_delta.norm(dim=-1) - shuffled_dose
            ).abs().max().item(),
        )
        rolled_l2_error = max(
            rolled_l2_error,
            (
                rolled_delta.norm(dim=-1) - dose.roll(1, 0)
            ).abs().max().item(),
        )
        head_permuted_l2_error = max(
            head_permuted_l2_error,
            (permuted_delta.norm(dim=-1) - dose).abs().max().item(),
        )
        source_replay = query_downstream(
            model, batch["query"], batch["pre_fusion"],
            batch["source_context"], dtype,
        )
        for name, delta in deltas.items():
            if name == "source":
                logits = batch["source_logits"]
            else:
                updated = query_downstream(
                    model, batch["query"], batch["pre_fusion"],
                    batch["source_context"] + delta, dtype,
                )
                logits = batch["source_logits"] + updated - source_replay
            append_condition(
                parts, name, logits, batch["labels"],
                batch["competitor"], delta,
            )
        classifier_logits.append(classifier.cpu())
        factorized_dose.append(dose.cpu())
        labels_parts.append(batch["labels"].cpu())
        competitor_parts.append(batch["competitor"].cpu())
        memory_prediction_parts.append(batch["memory_predictions"].cpu())
        if end == args.probe_batch_size or end % args.log_every_samples == 0:
            print(
                f"Level {LEVEL} formal={end}/{len(cache['labels'])}",
                flush=True,
            )
            save(root / "progress.json", {
                "stage": "formal_diagnostic",
                "samples_complete": end,
                "samples_total": len(cache["labels"]),
            })
    return {
        "labels": torch.cat(labels_parts),
        "competitor": torch.cat(competitor_parts),
        "memory_predictions": torch.cat(memory_prediction_parts),
        "classifier_logits": torch.cat(classifier_logits),
        "factorized_dose": torch.cat(factorized_dose),
        "conditions": {
            name: {
                field: torch.cat(values) for field, values in row.items()
            }
            for name, row in parts.items()
        },
        "l2_audit": {
            "signed_unit_max_abs_error": signed_unit_error,
            "residual_unit_max_abs_error": residual_unit_error,
            "candidate_dose_max_abs_error": candidate_l2_error,
            "residual_dose_max_abs_error": residual_l2_error,
            "shuffled_dose_max_abs_error": shuffled_l2_error,
            "rolled_dose_max_abs_error": rolled_l2_error,
            "head_permuted_dose_max_abs_error": head_permuted_l2_error,
        },
    }


def condition_metrics(row, source, labels, primary):
    ids = torch.where(primary)[0]
    source_correct = source["predictions"] == labels
    updated_correct = row["predictions"] == labels
    return {
        "full_accuracy": updated_correct.float().mean().item(),
        "primary_accuracy": (
            row["predictions"][ids] == labels[ids]
        ).float().mean().item(),
        "full_fixed_margin_mean": row["fixed_margin"].mean().item(),
        "primary_fixed_margin_mean": row["fixed_margin"][ids].mean().item(),
        "full_context_l2_mean": row["context_delta_norm"].mean().item(),
        "primary_context_l2_mean": row[
            "context_delta_norm"
        ][ids].mean().item(),
        "corrections": int((~source_correct & updated_correct).sum().item()),
        "regressions": int((source_correct & ~updated_correct).sum().item()),
        "primary_corrections": int((
            row["predictions"][ids] == labels[ids]
        ).sum().item()),
    }


def analyze_formal(collected, args):
    labels = collected["labels"]
    source = collected["conditions"]["source"]
    source_wrong = source["predictions"] != labels
    primary = source_wrong & (collected["memory_predictions"] == labels)
    ids = torch.where(primary)[0]
    if len(ids) < args.minimum_primary:
        raise RuntimeError(
            f"formal primary population {len(ids)} < {args.minimum_primary}"
        )
    if int(source_wrong.sum()) < args.minimum_errors:
        raise RuntimeError("formal source-error population is below minimum")
    metrics = {
        name: condition_metrics(row, source, labels, primary)
        for name, row in collected["conditions"].items()
    }
    source_primary_margin = source["fixed_margin"][ids]
    oracle_gain_values = (
        collected["conditions"]["full_signed_oracle"]["fixed_margin"][ids]
        - source_primary_margin
    )
    oracle_gain = oracle_gain_values.mean().item()
    candidate_gain_values = (
        collected["conditions"]["factorized_signed_candidate"][
            "fixed_margin"
        ][ids] - source_primary_margin
    )
    candidate_gain = candidate_gain_values.mean().item()
    recovery = candidate_gain / max(oracle_gain, 1e-12)
    candidate_margin = collected["conditions"][
        "factorized_signed_candidate"
    ]["fixed_margin"][ids]
    specificity = {}
    p_values = {}
    for offset, control in enumerate(SPECIFICITY_CONTROLS):
        control_margin = collected["conditions"][control][
            "fixed_margin"
        ][ids]
        effect = continuous_effect(
            (candidate_margin.double() - control_margin.double()).numpy(),
            args, args.analysis_seed + offset * 100,
        )
        effect["candidate_mean"] = candidate_margin.double().mean().item()
        effect["control_mean"] = control_margin.double().mean().item()
        key = f"candidate_vs_{control}"
        specificity[key] = effect
        p_values[key] = effect["sign_flip_p_two_sided"]
    adjusted = holm_adjust(p_values)
    for name, result in specificity.items():
        result["multiplicity"] = adjusted[name]
    specificity_passed = all(
        row["estimate"] > 0
        and row["multiplicity"]["significant_0.05"]
        for row in specificity.values()
    )
    full_accuracy = paired_statistics(
        source["predictions"],
        collected["conditions"]["factorized_signed_candidate"]["predictions"],
        labels, args, args.analysis_seed + 10000,
    )
    accuracy_noninferiority = (
        full_accuracy["accuracy_change"]["ci95"][0]
        >= -args.full_accuracy_noninferiority
    )
    effects = {}
    for name in CONDITIONS:
        if name == "source":
            continue
        gain = (
            collected["conditions"][name]["fixed_margin"][ids]
            - source_primary_margin
        ).mean().item()
        effects[name] = {
            "primary_margin_gain": gain,
            "full_oracle_recovery": gain / max(oracle_gain, 1e-12),
            "full_accuracy": paired_statistics(
                source["predictions"],
                collected["conditions"][name]["predictions"],
                labels, args,
                args.analysis_seed + 20000 + CONDITIONS.index(name) * 100,
            ),
        }
    classifier = binary_observability_metrics(
        collected["classifier_logits"], primary
    )
    registered_without_integrity = (
        recovery >= args.recovery_threshold
        and specificity_passed
        and accuracy_noninferiority
    )
    return {
        "population": {
            "samples": len(labels),
            "source_accuracy": (~source_wrong).float().mean().item(),
            "source_errors": int(source_wrong.sum().item()),
            "memory_decodable_source_errors": len(ids),
        },
        "metrics": metrics,
        "effects": effects,
        "classifier_enrichment_audit": classifier,
        "candidate": {
            "primary_margin_gain": candidate_gain,
            "full_oracle_margin_gain": oracle_gain,
            "full_oracle_recovery": recovery,
            "recovery_threshold_passed": recovery >= args.recovery_threshold,
            "specificity": specificity,
            "specificity_passed": specificity_passed,
            "full_accuracy": full_accuracy,
            "full_accuracy_noninferiority_passed": accuracy_noninferiority,
            "registered_gate_without_integrity": registered_without_integrity,
        },
        "diagnosis": {
            "classification": (
                "factorized_signed_read_supported"
                if registered_without_integrity
                else "factorized_repair_failed_stop_branch"
            ),
            "factorized_signed_recovery": recovery,
            "specificity_passed": specificity_passed,
            "full_accuracy_noninferiority_passed": accuracy_noninferiority,
            "registered_next_boundary": (
                "If integrity passes, repeat the frozen factorized reader "
                "across independent probe initializations before seed909."
                if registered_without_integrity
                else "Stop this router-repair branch; do not introduce a "
                "second composition formula or open seed909."
            ),
        },
    }


def plot_result(analysis, path):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    names = [
        "frozen_signed_router",
        "factorized_signed_candidate",
        "factorized_residual_control",
        "factorized_signed_shuffled_memory",
        "factorized_signed_rolled_delta",
        "factorized_signed_head_permuted",
        "full_signed_oracle",
    ]
    labels = [
        "Frozen", "Factorized", "Residual", "Shuffled", "Rolled",
        "Head perm.", "Oracle",
    ]
    recovery = [
        analysis["effects"][name]["full_oracle_recovery"] for name in names
    ]
    axes[0].bar(labels, recovery, color="#4E79A7")
    axes[0].axhline(
        RECOVERY_THRESHOLD, color="#E15759", linestyle="--", label="25%"
    )
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Full-Oracle margin recovery")
    axes[0].set_title("Fresh primary mechanism panel")
    axes[0].legend()

    keys = [f"candidate_vs_{name}" for name in SPECIFICITY_CONTROLS]
    estimates = [
        analysis["candidate"]["specificity"][key]["estimate"] for key in keys
    ]
    axes[1].bar(
        ["Source", "Frozen", "Residual", "Shuffled", "Rolled", "Head perm."],
        estimates, color="#59A14F",
    )
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("Candidate margin contrast")
    axes[1].set_title("Registered specificity")

    classifier = analysis["classifier_enrichment_audit"]
    axes[2].bar(
        ["AUROC", "AUPRC", "Top-k precision"],
        [classifier["auroc"], classifier["average_precision"],
         classifier["fixed_prevalence_precision"]],
        color=["#59A14F", "#F28E2B", "#B07AA1"],
    )
    axes[2].axhline(
        classifier["prevalence"], color="#777777", linestyle="--",
        label="prevalence",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Frozen classifier audit (not a gate)")
    axes[2].legend()
    figure.suptitle(
        "IST Level 6.19.6: One Frozen Factorized Signed Read", fontsize=15
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def preregistration(args):
    return {
        "level": LEVEL,
        "status": "one frozen factorized signed-read composition test",
        "parent": "Level 6.19.5 joint_calibration_coupling_bottleneck",
        "formal_split": {
            "samples": args.diagnostic_samples,
            "seed": args.diagnostic_seed,
            "new_relative_to_Level_6_19_5": True,
            "opened_once": True,
        },
        "single_candidate": {
            "formula": (
                "frozen predicted Oracle dose times frozen signed distilled "
                "unit direction"
            ),
            "dose_cap": args.dose_prediction_cap,
            "uses_target_label": False,
            "uses_rival_class": False,
            "uses_oracle_dose_or_direction": False,
            "uses_correctness_flag": False,
            "error_classifier_used_as_gate": False,
        },
        "controls": CONDITIONS,
        "specificity_family": [
            f"candidate versus {name}" for name in SPECIFICITY_CONTROLS
        ],
        "success_gate": {
            "full_oracle_recovery": args.recovery_threshold,
            "specificity": (
                "all six candidate margin contrasts positive after Holm "
                "correction at 0.05"
            ),
            "full_accuracy_noninferiority": (
                args.full_accuracy_noninferiority
            ),
            "integrity_required": True,
        },
        "locks": {
            "seed707_trunk_and_existing_probes_frozen": True,
            "Level_6_19_4_parent_routers_frozen": True,
            "Level_6_19_5_final_epoch_probes_frozen": True,
            "no_training": True,
            "no_calibration_or_threshold_search": True,
            "no_second_repair_candidate": True,
            "failed_Level_6_18_9_candidate_not_used": True,
            "seed909_locked": True,
            "protected_tests_not_used": True,
            "optimizer_and_model_search_closed": True,
        },
        "decision_rule": {
            "pass": (
                "repeat across independent probe initializations before "
                "opening seed909"
            ),
            "fail": (
                "stop router repair; no second composition formula and no "
                "seed909"
            ),
        },
        "protocol": {
            key: value for key, value in vars(args).items()
            if key not in {"dry_run", "smoke_test", "force"}
        },
    }


def validate(args):
    if args.seed != SEED or args.chunks != CHUNKS:
        raise ValueError(f"Level {LEVEL} is fixed to seed707 at 16 chunks")
    for path in (
        args.checkpoint,
        args.probes,
        args.level6_19_4_result,
        args.level6_19_4_routers,
        args.parent_result,
        args.parent_probes,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if min(
        args.diagnostic_samples, args.eval_batch_size,
        args.probe_batch_size, args.bootstrap_iterations,
        args.sign_flip_iterations,
    ) <= 0:
        raise ValueError("sample, batch, and iteration counts must be positive")
    if args.diagnostic_samples % args.eval_batch_size:
        raise ValueError("diagnostic samples must divide by eval-batch-size")
    if not args.smoke_test and (
        args.diagnostic_samples != 4096
        or args.diagnostic_seed != DIAGNOSTIC_SEED
        or args.analysis_seed != ANALYSIS_SEED
    ):
        raise ValueError(
            f"Formal Level {LEVEL} size and seeds are fixed; use "
            "--smoke-test for implementation checks"
        )
    parent = json.loads(
        Path(args.parent_result).read_text(encoding="utf-8")
    )
    if (
        not parent.get("integrity", {}).get("passed")
        or parent.get("analysis", {}).get("diagnosis", {}).get(
            "classification"
        ) != "joint_calibration_coupling_bottleneck"
    ):
        raise RuntimeError("Level 6.19.5 parent is not the canonical result")


def serializable_predictions(collected):
    return {
        "labels": collected["labels"].tolist(),
        "competitor": collected["competitor"].tolist(),
        "memory_predictions": collected["memory_predictions"].tolist(),
        "classifier_logits": collected["classifier_logits"].tolist(),
        "factorized_dose": collected["factorized_dose"].tolist(),
        "conditions": {
            name: {field: value.tolist() for field, value in row.items()}
            for name, row in collected["conditions"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Level 6.19.6 one frozen factorized signed read"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint",
        default="experiments/level6_18_3/formal/rescued_head_checkpoint.pt",
    )
    parser.add_argument(
        "--probes", default="experiments/level6_19/formal/linear_probes.pt"
    )
    parser.add_argument(
        "--level6-19-4-result",
        default="experiments/level6_19_4/formal_recovery/result.json",
    )
    parser.add_argument(
        "--level6-19-4-routers",
        default="experiments/level6_19_4/formal_recovery/router_checkpoint.pt",
    )
    parser.add_argument(
        "--parent-result",
        default="experiments/level6_19_5/formal/result.json",
    )
    parser.add_argument(
        "--parent-probes",
        default="experiments/level6_19_5/formal/diagnostic_probes.pt",
    )
    parser.add_argument("--diagnostic-samples", type=int, default=4096)
    parser.add_argument("--diagnostic-seed", type=int, default=DIAGNOSTIC_SEED)
    parser.add_argument("--analysis-seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--dose-prediction-cap", type=float, default=8.0)
    parser.add_argument(
        "--recovery-threshold", type=float, default=RECOVERY_THRESHOLD
    )
    parser.add_argument(
        "--full-accuracy-noninferiority",
        type=float, default=FULL_ACCURACY_NONINFERIORITY,
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=10000)
    parser.add_argument("--minimum-errors", type=int, default=200)
    parser.add_argument("--minimum-primary", type=int, default=150)
    parser.add_argument("--log-every-samples", type=int, default=256)
    parser.add_argument("--output", default="experiments/level6_19_6/formal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.diagnostic_samples = min(args.diagnostic_samples, 128)
        args.bootstrap_iterations = min(args.bootstrap_iterations, 100)
        args.sign_flip_iterations = min(args.sign_flip_iterations, 100)
        args.minimum_errors = 1
        args.minimum_primary = 1
        args.diagnostic_seed += 50_000_000
        args.analysis_seed += 50_000_000
        if args.output == "experiments/level6_19_6/formal":
            args.output = "experiments/level6_19_6/smoke"
    validate(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    protocol = preregistration(args)
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    result_path = root / "result.json"
    if result_path.exists() and not args.force:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps(result["analysis"]["diagnosis"], indent=2))
        return
    save(root / "preregistration.json", protocol)
    configure_cuda()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, original_probe, existing_probes, checkpoint_meta = load_frozen(
        args, device
    )
    router_args = argparse.Namespace(**vars(args))
    router_args.parent_result = args.level6_19_4_result
    router_args.parent_routers = args.level6_19_4_routers
    parent_routers, head_mask, router_meta, level6_19_4_result = (
        load_parent_routers(router_args, device)
    )
    diagnostic_probes, dose_stats, parent_result, parent_probe_meta = (
        load_level6_19_5_probes(args, device)
    )
    frozen_modules = {
        "model": model,
        "original_probe": original_probe,
        **{
            f"existing_probe_{name}": row["probe"]
            for name, row in existing_probes.items()
        },
        **{
            f"parent_router_{name}": router
            for name, router in parent_routers.items()
        },
        **{
            f"level6_19_5_probe_{name}": probe
            for name, probe in diagnostic_probes.items()
        },
    }
    before = {
        name: tensor_fingerprint(module)
        for name, module in frozen_modules.items()
    }
    router_before = {
        name: parameter_fingerprint(router)
        for name, router in parent_routers.items()
    }
    diagnostic = collect_cache(
        model, existing_probes, args, args.diagnostic_samples,
        args.diagnostic_seed, device, dtype, root, "formal_diagnostic",
        display_level=LEVEL,
    )
    collected = evaluate_formal(
        model, parent_routers, diagnostic_probes, dose_stats,
        diagnostic["cache"], args, device, dtype, root,
    )
    analysis = analyze_formal(collected, args)
    after = {
        name: tensor_fingerprint(module)
        for name, module in frozen_modules.items()
    }
    router_after = {
        name: parameter_fingerprint(router)
        for name, router in parent_routers.items()
    }
    l2_max = max(collected["l2_audit"].values())
    parent_seeds = parent_result["protocol"]["protocol"]
    prior_split_seeds = {
        parent_seeds["train_seed"], parent_seeds["validation_seed"],
        parent_seeds["diagnostic_seed"],
    }
    integrity = {
        "frozen_module_fingerprints_unchanged": before == after,
        "parent_router_fingerprints_unchanged": router_before == router_after,
        "frozen_parameters_remain_frozen": all(
            not parameter.requires_grad
            for module in frozen_modules.values()
            for parameter in module.parameters()
        ),
        "oracle_l2_max_abs_error": diagnostic["oracle_l2_max_abs_error"],
        "oracle_l2_passed": diagnostic["oracle_l2_max_abs_error"] <= 1e-5,
        "factorized_l2_audit": collected["l2_audit"],
        "factorized_l2_passed": l2_max <= 1e-5,
        "source_replay_max_abs": diagnostic["source_replay_max_abs"],
        "fresh_diagnostic_seed": args.diagnostic_seed not in prior_split_seeds,
        "no_training_or_calibration": True,
        "single_candidate_only": True,
        "candidate_inference_uses_target_label": False,
        "candidate_inference_uses_rival_class": False,
        "candidate_inference_uses_oracle": False,
        "classifier_used_as_gate": False,
        "failed_candidate_not_used": True,
        "seed909_locked": True,
        "protected_tests_not_used": True,
        "optimizer_search_closed": True,
    }
    integrity["passed"] = all([
        integrity["frozen_module_fingerprints_unchanged"],
        integrity["parent_router_fingerprints_unchanged"],
        integrity["frozen_parameters_remain_frozen"],
        integrity["oracle_l2_passed"],
        integrity["factorized_l2_passed"],
        integrity["fresh_diagnostic_seed"],
        integrity["no_training_or_calibration"],
        integrity["single_candidate_only"],
        not integrity["candidate_inference_uses_target_label"],
        not integrity["candidate_inference_uses_rival_class"],
        not integrity["candidate_inference_uses_oracle"],
        not integrity["classifier_used_as_gate"],
    ])
    final_pass = (
        integrity["passed"]
        and analysis["candidate"]["registered_gate_without_integrity"]
    )
    analysis["diagnosis"]["integrity_passed"] = integrity["passed"]
    analysis["diagnosis"]["factorized_signed_read_passed"] = final_pass
    if not integrity["passed"]:
        analysis["diagnosis"]["classification"] = "integrity_failure"
        analysis["diagnosis"]["registered_next_boundary"] = (
            f"Stop; repair the Level {LEVEL} implementation."
        )
    elif final_pass:
        analysis["diagnosis"]["classification"] = (
            "factorized_signed_read_supported"
        )
    else:
        analysis["diagnosis"]["classification"] = (
            "factorized_repair_failed_stop_branch"
        )
    result = {
        "protocol": protocol,
        "checkpoint_meta": checkpoint_meta,
        "parent_level6_19_5_diagnosis": parent_result["analysis"]["diagnosis"],
        "parent_level6_19_4_diagnosis": level6_19_4_result[
            "analysis"
        ]["diagnosis"],
        "frozen_probe_meta": {
            "dose_stats": dose_stats,
            "selected_heads": router_meta["selected_heads"],
            "head_mask": head_mask.cpu().tolist(),
            "probe_level": parent_probe_meta["level"],
        },
        "integrity": integrity,
        "analysis": analysis,
    }
    save(root / "result.json", result)
    save(root / "summary.json", {
        "integrity": integrity,
        "population": analysis["population"],
        "candidate": analysis["candidate"],
        "diagnosis": analysis["diagnosis"],
    })
    save(root / "predictions.json", serializable_predictions(collected))
    plot_result(analysis, root / "factorized_signed_read.png")
    save(root / "progress.json", {
        "stage": "complete",
        "integrity_passed": integrity["passed"],
        "classification": analysis["diagnosis"]["classification"],
        "factorized_signed_read_passed": final_pass,
    })
    print(json.dumps(analysis["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
