"""
STEP 1 of 4 — build the dataset for ONE measurement budget.

    python scripts/run_1_generate_dataset.py

Writes training_data/<rays>_rays_<points>_points_<n>_samples/ containing
config.json (steps 2-4 read it back), train.npz / test.npz, a
dataset_summary.txt with the numbers to quote in the paper, and figures/.

The simulated devices are cached once in training_data/_device_pools/ and
reused by every budget, so only the first configuration pays for simulation.
The train/test split is made on DEVICE ids, stored with the pool, so no
device can appear on both sides in any budget.

NEXT:  python scripts/run_2_train_model.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import matplotlib
matplotlib.use("Agg")            # no display: this can run unattended

from dqd.study import dataset
from dqd.config import log
from dqd.study.config import StudyConfig

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS — steps 2-4 read these back out of config.json, so edit them HERE
# ══════════════════════════════════════════════════════════════════════════

CONFIG = StudyConfig(

    # the measurement budget — what the whole study is about
    n_rays=3,               # rays fired across the diagram
    n_points=40,            # points sampled along each ray

    # how much data
    n_train=500,            # devices the model trains on
    n_test=100,             # held-out devices, same distribution

    # the devices
    resolution=100,         # stability-diagram side length, in pixels
    split_seed=12345,       # which devices are held out (never redraws them)
    seed=0,                 # same seed = the same devices, every time
    voltage_window=(-1.0, 1.0, -1.0, 1.0),
    offset_scale=0.35,      # per-device shift of the window, as a fraction of
                            # its width — slides the honeycomb off a common
                            # origin, without which the dataset looks far more
                            # uniform than it is
    coulomb_peak_width=0.01,
    temperature=0.00001,

    epochs=40,              # used by run_2, recorded here so all four agree

    # ── per-device pictures ──────────────────────────────────────────────
    # Pure .png output; free to turn on and off, and run_5 redraws them
    # later without regenerating data or retraining.
    save_device_figures=True,
    device_figures={
        "charge_sensor":          True,   # the raw simulated sensor image
        "charge_sensor_gradient": True,   # ... with the smooth gate background
                                          # differenced away — where the
                                          # honeycomb is actually visible
        "stability_diagram":      True,   # the binary ground truth
        "rays":                   True,   # rays + detected peaks, over sensor
        "rays_on_truth":          True,   # the same rays over the ground truth
        "panel":                  True,   # four of the above, side by side
        "measurement":            False,  # only the visited pixels
        "ray_traces":             False,  # the 1-D signal along each ray

        # per-RAY colouring: "which ray found which line", rather than
        # "how much of the plane was measured"
        "all_rays_peaks_overlay":     True,
        "ml_measurement":             True,
        "summary_total":              True,
        "summary_total_all_crosses":  True,   # the paper figure
    },

    # Which devices to draw:  "ALL" | "NONE" | [1, 2, 5]
    # "ALL" on a 500-device split is a few hundred files; the count and a
    # time estimate are printed before anything is drawn.
    figure_devices={"train": [1, 2, 3], "test": [1, 2, 3]},

    figure_dpi=200,          # 300 for the paper, 150 for a quick look
    figure_size_in=8.0,
    figure_cell_grid=True,   # voltage-grid cells behind the binary maps
)

# Re-cut train.npz / test.npz even if they exist.  Only needed after changing
# how the measurement is cut; the devices are reused either way.
REBUILD_ARRAYS = False

# ══════════════════════════════════════════════════════════════════════════


def main():
    log.say("=" * 74)
    log.say("STEP 1 of 4 — dataset")
    log.say("=" * 74)
    log.detail(CONFIG.describe())

    if os.path.isfile(os.path.join(CONFIG.dir, "config.json")):
        diff = CONFIG.differences(StudyConfig.load(CONFIG.dir))
        if diff:
            log.say("\n[note] this folder exists with different settings; "
                    "it will be updated:")
            for key, (new, was) in diff.items():
                log.say(f"       {key}: {was}  ->  {new}")

    report = dataset.build(CONFIG, rebuild=REBUILD_ARRAYS)

    log.say("\n" + "=" * 74)
    if report["all_passed"]:
        log.say("every split check passed.")
    else:
        log.warn("!! A SPLIT CHECK FAILED — see dataset_summary.json.  Do not "
                 "publish results from this dataset.")
    log.say(f"folder:  {os.path.abspath(CONFIG.dir)}")
    log.say("next:    python scripts/run_2_train_model.py")
    log.say("=" * 74)


if __name__ == "__main__":
    main()
