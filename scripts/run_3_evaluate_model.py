"""
STEP 3 of 4 — score the models on the held-out devices.  Trains nothing.

    python scripts/run_3_evaluate_model.py

For each configuration, three files are read, all from inside that
configuration's own folder — config.json, model/unet.pt and test.npz.
train.npz is never read, and no other configuration's folder is touched.
Nothing is tuned on the test devices: the binarisation threshold comes out
of the checkpoint, and the checkpoint's budget is checked against the
folder's, so a model can never be scored on a measurement it was not
trained for.

Writes training_data/<configuration>/evaluation/: results.txt,
metrics.json, per_device.csv (the spread, not just the mean), and figures/
(predictions on the best / median / worst device, F1 vs tolerance, F1 per
device, and the raw probability map with the threshold marked).

With COMPARE_AFTER on it then does step 4 for the listed configurations, so
for most sweeps you never need to run run_4 separately.
"""
from _common import banner, configs, for_each, table
from dqd.study import comparison, evaluation
from dqd.config import log

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# Folder names that have been through run_1 and run_2.  Scored in the order
# given, or "ALL" for every folder in training_data/.
CONFIG_NAMES = [
    "3_rays_40_points_500_samples",
]

# Put the listed configurations side by side afterwards and write
# results/<sweep name>/.  Exactly what run_4 does; this just saves a command.
COMPARE_AFTER = True

# ══════════════════════════════════════════════════════════════════════════


def main():
    banner("STEP 3 of 4 — evaluate")
    done, failed = for_each(configs(CONFIG_NAMES), evaluation.run)

    table([("configuration", 36), ("devices", 8), ("coverage", 10),
           ("F1@1", 8), ("IoU", 8)],
          [(cfg.name, str(m["n_test_devices"]), f"{100 * m['coverage']:.2f}%",
            f"{m['f1@1']:.3f}", f"{m['iou']:.3f}") for cfg, m in done],
          failed)

    if COMPARE_AFTER and done:
        banner("comparing the configurations you listed")
        comparison.run(configs=[cfg for cfg, _ in done])
    else:
        log.say("next:    python scripts/run_4_compare_configs.py")


if __name__ == "__main__":
    main()
