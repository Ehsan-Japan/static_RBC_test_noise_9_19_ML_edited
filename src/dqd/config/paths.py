"""
paths.py — the project's directory layout, in one place.

Everything the code writes or reads lives INSIDE the project folder, and
every path here is anchored to this file, not to the current working
directory.  So a program behaves the same whether it is started from the
project root, from scripts/, or from anywhere else:

    static_RBC_test_noise_7_25_ML/       PROJECT_ROOT
      src/                               the code
      scripts/                           the main programs
      training_data/                     TRAINING_DATA — the device folders
      runs/                              RUNS_ROOT     — one folder per trial
      grid_cache/                        GRID_CACHE    — shared measured rays
      results/                           RESULTS

Use these constants instead of writing "../training_data": a relative path
means a different folder for every caller, and once meant the sibling
projects/training_data/ outside this project entirely.
"""
import os

# .../src/dqd/config/paths.py -> .../src/dqd/config -> src/dqd -> src -> root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

TRAINING_DATA = os.path.join(PROJECT_ROOT, "training_data")
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
GRID_CACHE = os.path.join(PROJECT_ROOT, "grid_cache")
RESULTS = os.path.join(PROJECT_ROOT, "results")


# Where the CONFIGURATION folders (one per measurement budget) are written.
# Overridable, because run_0 gives each sweep its own folder under results/ so
# that one run of the program is one self-contained folder:
#
#     results/4-7-8_rays_40_points_150_samples/
#         4_rays_40_points_150_samples/     <- a configuration folder
#         7_rays_40_points_150_samples/
#         comparison.csv, figures/, model_structure.yaml, hyperparameters.yaml
#
# The DEVICE POOLS are deliberately NOT moved: they are the expensive part,
# they do not depend on the budget, and they stay shared in training_data/.
CONFIG_ROOT = TRAINING_DATA


def set_config_root(path: str) -> str:
    """Point the configuration folders at *path* (run_0 does this)."""
    global CONFIG_ROOT
    CONFIG_ROOT = path
    os.makedirs(CONFIG_ROOT, exist_ok=True)
    return CONFIG_ROOT


def config_root(*parts: str) -> str:
    """Path to a configuration folder under the current CONFIG_ROOT."""
    return os.path.join(CONFIG_ROOT, *parts)


def training_data(*parts: str) -> str:
    """Path to a dataset folder inside training_data/, e.g.

        training_data("ml_train_split_n2000_res100")
    """
    return os.path.join(TRAINING_DATA, *parts)

# This is a convenience function. 
# The *parts syntax allows you to pass in as many string arguments as you want.
# The function will take the absolute path of your TRAINING_DATA folder and glue all those pieces onto the end of it.



# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.abspath(__file__)))))
# This block calculates the exact location of your main project folder.

# os.path.abspath(__file__): Gets the full, absolute path to this file. (e.g., /Users/name/static_RBC_test_noise_7_25_ML/src/dqd/config/paths.py)

# os.path.dirname(...): Strips off the last piece of the path to get the parent folder. Because paths.py is buried 4 folders deep inside the project root, the code calls dirname 4 times to climb up:

# Up to config/

# Up to dqd/

# Up to src/

# Up to static_RBC_test_noise_7_25_ML/ (This becomes PROJECT_ROOT)