"""Frozen Memory 0.5.3: layer-matched Fast-Memory write and read."""
from run_pretrained_layer_memory_0_5_2 import main


if __name__ == "__main__":
    raise SystemExit(main(
        default_layer_matched_write=True,
        default_output="experiments/pretrained_base/layer_memory_0_5_3/formal",
        stage="Frozen Memory 0.5.3",
    ))

