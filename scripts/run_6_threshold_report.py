"""
EXTRA — show the probability maps and the threshold that turns them into the
binary predictions run_3 scores.  Trains nothing, re-tunes nothing.

    python scripts/run_6_threshold_report.py

The U-Net outputs a probability per pixel; run_3's scores only exist after
that map is cut at a threshold.  This puts the probability map, the ground
truth and the cut side by side, so the threshold is something a reader can
see rather than a number they have to trust.

The threshold is NOT chosen here and is not 0.5: ml/grid_train.py scans
0.3 … 0.99 on a validation split carved out of the TRAINING devices, keeps
the best tolerant F1@1, and saves it inside model/unet.pt.  This page also
re-scans it on the test devices, but only to report how much the honest
choice gave up — that hindsight number is labelled as such and used by
nothing else.

Writes results/threshold_report/<configuration>/ (curves, threshold_scan.csv,
and per-device overview / threshold-strip / histogram pages).  Nothing under
training_data/ is written to.
"""
import os

from _common import banner, configs as resolve, for_each
from dqd.study import threshold_report
from dqd.config import log

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

CONFIG_NAMES = "ALL"        # folders that have been through run_1 and run_2

# Which held-out devices get their own folder of pictures.
#   None       the configuration's own figure_devices["test"] list, so the
#              same devices are followed through the whole study
#   [4, 5, 6]  those, by their sample_<i> name;  "ALL" is 3 files per device
TEST_DEVICES = None

# How many devices the AGGREGATE curves are scanned over.  Every threshold
# costs one distance transform per device, so scanning all of them is slow
# and moves the curves by nothing.
MAX_SCAN_DEVICES = 60

# ══════════════════════════════════════════════════════════════════════════


def main():
    banner("EXTRA — probability maps and the binarisation threshold")
    done, failed = for_each(
        resolve(CONFIG_NAMES),
        lambda cfg: threshold_report.run(cfg, devices=TEST_DEVICES,
                                         max_scan_devices=MAX_SCAN_DEVICES))

    log.say()
    for _cfg, path in done:
        log.say(f"  wrote {os.path.abspath(path)}")
    for name, why in failed:
        log.warn(f"  FAILED {name}: {why.splitlines()[0]}")


if __name__ == "__main__":
    main()
