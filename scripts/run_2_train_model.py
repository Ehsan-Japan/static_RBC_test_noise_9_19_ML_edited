"""
STEP 2 of 4 — train the U-Net on one or more configurations.

    python scripts/run_2_train_model.py

Reads the dataset run_1 built and writes into the same folder, under model/:
unet.pt (the checkpoint, which remembers its budget), training_curve.png,
model_structure.yaml and training_summary.json.

The test devices are NOT touched.  The validation slice that picks the best
epoch and the binarisation threshold is carved out of the TRAINING devices,
so step 3's number is a genuine held-out result.

Everything except the number of epochs is deliberately not a knob — the
architecture, learning rate, batch size, loss and validation fraction are
constants in src/dqd/ml/.  Comparing two budgets only means something if
nothing else changed.

NEXT:  python scripts/run_3_evaluate_model.py
"""
from _common import banner, configs, for_each, table
from dqd.study import training
from dqd.config import log

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# The folder names run_1 created.  A list, trained in the order given, or
# "ALL" for every folder in training_data/.
CONFIG_NAMES = [
    "3_rays_40_points_500_samples",
]

# None uses each folder's own config.json; a number overrides it everywhere.
# More devices need FEWER epochs, not more.
EPOCHS = None

# ══════════════════════════════════════════════════════════════════════════


def _train(cfg):
    if EPOCHS is not None:
        cfg.epochs = EPOCHS
        cfg.save()
    _checkpoint, summary = training.train(cfg)
    return summary


def main():
    banner("STEP 2 of 4 — train")
    done, failed = for_each(configs(CONFIG_NAMES), _train)

    table([("configuration", 36), ("best val F1@1", 15), ("threshold", 11)],
          [(cfg.name, f"{s['best_val_f1']:.3f}", f"{s['threshold']:g}")
           for cfg, s in done],
          failed)
    log.say("next:    python scripts/run_3_evaluate_model.py")


if __name__ == "__main__":
    main()
