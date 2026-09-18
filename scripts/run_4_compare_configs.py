"""
STEP 4 of 4 — every configuration side by side.

    python scripts/run_4_compare_configs.py

Retrains nothing and touches nothing: it reads the metrics.json each
configuration's step 3 wrote and collects them into a folder NAMED AFTER THE
SWEEP, so a different setup writes a different folder and nothing is
overwritten:

    results/3-5-8-12_rays_40_points_500_samples/
        comparison.csv / .txt      one row per configuration
        figures/f1_vs_rays.png     THE figure of the study
                f1_vs_coverage.png the same on the honest x-axis: how much
                                   of the grid was actually measured
                f1_heatmap.png     rays x points (needs a 2-D sweep)
                f1_vs_tolerance.png   how much of the error is sub-pixel
                f1_vs_train_size.png  would more devices have helped?

Configurations that have not been evaluated are listed as missing, not
silently dropped — a half-finished sweep must not turn into a
complete-looking figure.  Safe to re-run at any time, including mid-sweep.
"""
from _common import banner, configs
from dqd.study import comparison

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# The configurations to compare.  They appear in the table and the figures IN
# THE ORDER YOU LIST THEM.  "ALL" is every folder in training_data/.
CONFIG_NAMES = "ALL"

# Folder name under results/.  None builds it from the sweep itself, which is
# what you want unless a particular comparison deserves its own name.
OUT_NAME = None

# ══════════════════════════════════════════════════════════════════════════


def main():
    banner("STEP 4 of 4 — compare")
    comparison.run(configs=configs(CONFIG_NAMES), name=OUT_NAME)


if __name__ == "__main__":
    main()
