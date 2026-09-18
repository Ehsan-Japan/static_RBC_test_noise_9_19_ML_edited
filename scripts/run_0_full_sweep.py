"""
THE LAZY BUTTON — steps 1-4 for a whole sweep, in one command.

    python scripts/run_0_full_sweep.py

ONE RUN, ONE FOLDER.  Everything a run produces — the datasets, the models,
the evaluations, the comparison table and figures, plus model_structure.yaml
and hyperparameters.yaml describing what was run — is written inside

    results/<rays>_rays_<points>_points_<samples>_samples/

With FRESH_START below, results/ is emptied first, so that folder is the
whole run and nothing older can be mistaken for part of it.  With
FRESH_START = False the old behaviour is back: nothing is overwritten and a
re-run only pays for the cells that are new.

Every setting NOT listed below is taken from run_1_generate_dataset.py's
CONFIG, and it calls the same library functions steps 1-4 do, in the same
order — so this cannot produce a different answer than running them by hand.
"""
import _common  # noqa: F401  (import path + headless plotting)
from dqd.study import sweep
from run_1_generate_dataset import CONFIG as TEMPLATE

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

# START FROM EMPTY.  True deletes results/ before anything runs, so what is
# on disk afterwards came from THIS run and nothing else — no stale figures,
# no half-finished cell from a setting you have since changed.
FRESH_START = True

# True ALSO empties training_data/, which is where the simulated DEVICE POOLS
# live.  Those are the expensive part and they do not depend on the budget,
# so leaving this False keeps the devices and still gives a clean results/ —
# the same fresh answer, minutes faster.
WIPE_TRAINING_DATA = True

# The measurement budget.  Every (rays, points) combination is one cell:
# one folder, one trained model, one row in the final table.
RAYS = [4,5,6,7,8]
POINTS = [40,50,60]

# Dataset sizes.  The SAME for every cell on purpose, so the cells differ
# only in how the devices were measured — never in which devices, or how many.
N_TRAIN = 500
N_TEST = 50

# How long to train each cell.
EPOCHS =50

# True reuses checkpoints already on disk.  (Datasets are reused either way:
# devices are never re-simulated.)
SKIP_EXISTING = True

# Per-device pictures per cell.  Keep it small; run_5 draws the full set.
FIGURE_DEVICES = {"train": [1, 2, 3,4,5,6,7,8,9,10], "test": [1, 2, 3,4,5,6,7,8,9,10]}

# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sweep.run(TEMPLATE,
              rays=RAYS, points=POINTS,
              n_train=N_TRAIN, n_test=N_TEST, epochs=EPOCHS,
              skip_existing=SKIP_EXISTING, figure_devices=FIGURE_DEVICES,
              fresh_start=FRESH_START,
              wipe_training_data=WIPE_TRAINING_DATA)
