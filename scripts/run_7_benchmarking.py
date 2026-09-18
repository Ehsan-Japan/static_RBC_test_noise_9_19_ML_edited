"""
RUN 7 — BENCHMARK: corner-fan rays vs parallel diagonal lines, SAME budget.

    python scripts/run_7_benchmarking.py

Every arm gets the same n_rays x n_points measured points on the SAME
devices; only WHERE the points go differs.  Each arm is an ordinary
configuration folder under training_data/ (built and trained with the same
library functions run_1/2/3 use, and reused if already on disk), and the
comparison lands in ONE results folder ready for the paper / slides:

    results/benchmark_<rays>_rays_<points>_points_<train>_samples[_thr…]/
        benchmark.csv / .txt / metrics.json     the table
        README.txt                              what each figure shows
        fig1_measurement_masks.png     where each geometry puts its points
        fig2_f1_bars.png               THE headline figure
        fig3_f1_vs_tolerance.png       how much error is sub-pixel
        fig4_precision_recall.png
        fig5_per_device_scatter.png    fan vs the others, paired per device
        fig6_f1_vs_threshold.png       the PROB_THRESHOLD knob, per arm
        fig7_gallery_sample_<i>.png    measurement -> probability -> prediction

Restartable and additive: re-running reuses datasets and checkpoints, and a
changed PROB_THRESHOLD writes a NEW folder instead of overwriting the old.
"""
import _common  # noqa: F401  (import path + headless plotting)
from _common import banner
from dqd.study import benchmarking
from run_1_generate_dataset import CONFIG as TEMPLATE

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# The arms.  Order = order in every table and figure.  "rays" is our method;
# "parallel_diag" is the same budget on parallel oblique lines.  Add
# "hcuts" for the Hernandes-style horizontal line-cut baseline, or any of:
# vcuts, random_rays, grid, random  (see study/sampling.py).
STRATEGIES = ["rays", "parallel_diag"]

# The shared measurement budget: every arm gets N_RAYS x N_POINTS points.
N_RAYS = 8
N_POINTS = 40

# Devices — identical for every arm, so the arms differ ONLY in geometry.
N_TRAIN = 150
N_TEST = 30

# Training epochs per arm.
EPOCHS = 40

# ── THE U-NET PROBABILITY THRESHOLD ──────────────────────────────────────
# The network outputs a probability per pixel; predictions exist only after
# cutting that map at a threshold.
#   None    each arm uses the cut chosen on ITS validation split during
#           training (stored in the checkpoint) — the honest default,
#           nothing tuned on test devices
#   0.5     (or any number) the SAME fixed cut for every arm — a sensitivity
#           check, labelled as an override in every output and written to a
#           separate _thr<value> folder
# fig6 shows the whole F1-vs-threshold curve per arm either way.
PROB_THRESHOLD = None

# True reuses datasets and checkpoints already on disk; False retrains the
# arms (datasets and devices are still reused — they never change).
SKIP_EXISTING = True

# Which held-out devices get the fig7 side-by-side gallery (1-based).
GALLERY_DEVICES = [1, 2, 3]

# Devices used for the fig6 threshold scan (speed knob; the curve barely
# moves past ~40) — and figure resolution: 300 for JJAP, 150 for a look.
SCAN_DEVICES = 40
FIGURE_DPI = 300

# Results folder name under results/.  None builds it from the settings.
OUT_NAME = None

# ══════════════════════════════════════════════════════════════════════════


def main():
    banner("RUN 7 — benchmark: ray fan vs parallel diagonal, same budget")
    benchmarking.run(
        TEMPLATE,
        strategies=STRATEGIES,
        n_rays=N_RAYS, n_points=N_POINTS,
        n_train=N_TRAIN, n_test=N_TEST, epochs=EPOCHS,
        threshold=PROB_THRESHOLD,
        skip_existing=SKIP_EXISTING,
        gallery_devices=GALLERY_DEVICES,
        scan_devices=SCAN_DEVICES,
        dpi=FIGURE_DPI,
        figure_devices=TEMPLATE.figure_devices,
        name=OUT_NAME,
    )


if __name__ == "__main__":
    main()
