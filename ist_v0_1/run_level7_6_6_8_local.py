"""Level 7.6.6.8: cross-seed confirmation of probe-selected L3 top-4 causality (seed 2026)."""
from __future__ import annotations

import sys

import run_level7_6_6_7_local as confirmation


# Freeze the independent validation selection made in Level 7.6.6.5.
confirmation.SEED = 2026
confirmation.PARENT = (
    confirmation.ROOT / "experiments/level7_6_4/formal/ist-full_seed2026/stage_4096.pt"
)
confirmation.PROBE_TOP4 = (31, 3, 5, 12)
confirmation.RANDOM_CONTROLS = {
    "random4_a_l3_ablate": (0, 7, 14, 20),
    "random4_b_l3_ablate": (1, 8, 16, 24),
    "random4_c_l3_ablate": (2, 9, 18, 27),
}
confirmation.CONDITIONS = (
    "intact", "probe_top4_l3_ablate", *confirmation.RANDOM_CONTROLS,
    "probe_indices_l2_ablate", "probe_top4_l3_boost_2",
)


def main() -> int:
    if not any(arg == "--output" or arg.startswith("--output=") for arg in sys.argv[1:]):
        sys.argv.extend(["--output", "experiments/level7_6_6_8/formal"])
    return confirmation.main()


if __name__ == "__main__":
    raise SystemExit(main())
