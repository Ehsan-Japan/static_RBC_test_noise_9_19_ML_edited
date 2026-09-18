"""
benchmark.py — four ways to spend the SAME measurement budget.

    python scripts/benchmark.py

Same devices, same network, same training, same metric.  Only WHERE the
measured points go changes, so a difference in the score is a difference in
geometry and nothing else.

    rays           a fan of oblique lines from one corner   <- ours
    hcuts          horizontal line cuts                     <- Hernandes et al.
    parallel_diag  oblique lines, all the same angle        <- the fan removed
    grid           the same points on a lattice             <- not lines at all

Each scenario differs from `rays` by ONE property, so the ordering says which
property does the work: being lines, being oblique, or fanning out.

Writes results/benchmark/  ->  table.csv, per_device.csv, f1.png
Restartable: a scenario already trained is loaded from disk, so adding a
fifth costs only that one.
"""
import csv
import os

import numpy as np

from _common import banner
from dqd.config import paths
from dqd.config import log
from dqd.ml import grid_metrics, grid_train
from dqd.study import dataset, sampling
from dqd.study.config import StudyConfig

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════

SCENARIOS = ["rays", "hcuts", "parallel_diag", "grid"]

N_RAYS, N_POINTS = 8, 40        # the budget: 8 x 40 = 320 measured points
N_TRAIN, N_TEST = 500, 100      # devices; reuses the cached pool
EPOCHS = 40

OUT = os.path.join(paths.RESULTS, "benchmark")

# ══════════════════════════════════════════════════════════════════════════

DEVICES = StudyConfig(n_rays=N_RAYS, n_points=N_POINTS,
                      n_train=N_TRAIN, n_test=N_TEST)


_DIRS = {}


def device_dirs(split):
    """The train / test device folders — resolved once, then remembered.

    The devices are simulated once and cached on disk, and the split is read
    from the pool rather than redrawn, so every scenario below is measured on
    exactly the same devices.
    """
    if not _DIRS:
        pool, _ = dataset.make_devices(DEVICES)
        train_ids, test_ids, _ = dataset.split_devices(DEVICES, pool)
        _DIRS["train"] = dataset.sample_dirs_for(pool, train_ids)
        _DIRS["test"] = dataset.sample_dirs_for(pool, test_ids)
    return _DIRS[split]


def measure(scenario, split):
    """(X, Y) for one side of the split, measured by one scenario.

    Only the measurement is redone per scenario — which is exactly what has
    to differ, and nothing else does.
    """
    return sampling.build(device_dirs(split), N_RAYS, N_POINTS, scenario,
                          verbose=False)


def score(scenario):
    """Train (or reload) one scenario and score it device by device."""
    checkpoint = os.path.join(OUT, f"{scenario}.pt")

    if os.path.isfile(checkpoint):
        log.detail(f"  reusing {checkpoint}")
        net, meta = grid_train.load(checkpoint)
        threshold = meta["threshold"]
    else:
        Xtr, Ytr = measure(scenario, "train")
        net, threshold, _ = grid_train.train(Xtr, Ytr, epochs=EPOCHS)
        grid_train.save(net, threshold, checkpoint, N_RAYS, N_POINTS)

    Xte, Yte = measure(scenario, "test")

    pred = grid_train.predict(net, Xte) > threshold
    metrics = grid_metrics.evaluate(pred, Yte)
    per_device = [grid_metrics.tolerant_f1(p, y, 1)["f1"]
                  for p, y in zip(pred, Yte)]
    coverage = float(Xte[:, 1].mean())          # ch1 is the visited mask
    return metrics, per_device, coverage


def figure(rows, per_device, path):
    """One bar per scenario, with the device-to-device spread on it."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [r["scenario"] for r in rows]
    means = [r["f1@1"] for r in rows]
    sds = [float(np.std(per_device[n])) for n in names]
    colours = ["#1f5fa8"] + ["#9aa4b0"] * (len(names) - 1)
    ax.bar(names, means, yerr=sds, capsize=4, color=colours, width=0.6)
    for x, m in enumerate(means):
        ax.text(x, m + 0.015, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("F1@1 on held-out devices")
    ax.set_title(f"{N_RAYS * N_POINTS} measured points, "
                 f"{N_TEST} held-out devices", fontsize=10)
    ax.set_ylim(0, max(means) * 1.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    banner(f"benchmark — {N_RAYS} x {N_POINTS} = {N_RAYS * N_POINTS} "
           f"measured points, {len(SCENARIOS)} scenarios")
    os.makedirs(OUT, exist_ok=True)

    rows, per_device = [], {}
    for scenario in SCENARIOS:
        log.say(f"\n--- {scenario} " + "-" * 55)
        metrics, f1s, coverage = score(scenario)
        per_device[scenario] = f1s
        rows.append({"scenario": scenario,
                     "description": sampling.describe(scenario),
                     "coverage": coverage,
                     "f1@1": metrics["f1@1"],
                     "precision@1": metrics["precision@1"],
                     "recall@1": metrics["recall@1"],
                     "iou": metrics["iou"]})

    # ── the table ────────────────────────────────────────────────────────
    base = np.array(per_device[SCENARIOS[0]])
    log.say("\n" + "=" * 74)
    log.say(f"{'scenario':<16}{'coverage':>10}{'F1@1':>8}{'prec':>8}"
            f"{'recall':>8}{'IoU':>8}{'vs rays':>10}{'wins':>7}")
    for r in rows:
        d = base - np.array(per_device[r["scenario"]])
        gap = "" if r["scenario"] == SCENARIOS[0] else f"{d.mean():+.3f}"
        wins = "" if r["scenario"] == SCENARIOS[0] else \
            f"{100 * (d > 0).mean():.0f}%"
        log.say(f"{r['scenario']:<16}{100 * r['coverage']:>9.2f}%"
                f"{r['f1@1']:>8.3f}{r['precision@1']:>8.3f}"
                f"{r['recall@1']:>8.3f}{r['iou']:>8.3f}{gap:>10}{wins:>7}")
    log.say("=" * 74)
    log.say("'vs rays' is the mean F1@1 difference taken device by device;")
    log.say("'wins' is the fraction of held-out devices the rays come out ahead")
    log.say("on.  One training run per scenario — repeat at another seed before")
    log.say("claiming a margin this size.")

    # ── the files ────────────────────────────────────────────────────────
    with open(os.path.join(OUT, "table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "per_device.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", *SCENARIOS])
        for i in range(len(base)):
            w.writerow([i, *[f"{per_device[s][i]:.4f}" for s in SCENARIOS]])
    figure(rows, per_device, os.path.join(OUT, "f1.png"))
    log.say(f"\nwrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
