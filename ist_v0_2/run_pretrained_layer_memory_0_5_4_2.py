"""Frozen Memory 0.5.4.2: locked independent intervention confirmation."""
from run_pretrained_layer_memory_0_5_4_1 import main


if __name__ == "__main__":
    raise SystemExit(main(
        default_methods=["baseline", "prototype_center", "prototype_pc1_topk4"],
        default_calibration_samples=64,
        default_samples=256,
        default_output="experiments/pretrained_base/layer_memory_0_5_4_2/formal",
        stage="Frozen Memory 0.5.4.2",
        calibration_seed_base=470000000,
        heldout_seed_base=480000000,
        primary_method="prototype_pc1_topk4",
    ))

