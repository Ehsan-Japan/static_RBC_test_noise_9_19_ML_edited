"""
benchmarking.py — the machinery behind scripts/run_7_benchmarking.py.

THE QUESTION: at the SAME measurement budget (n_rays x n_points points), does
the corner FAN of rays beat n_rays PARALLEL DIAGONAL lines?  Same devices,
same network, same training constants, same metrics — the arms differ ONLY in
where the points are put (study/sampling.py).

Each arm is an ordinary configuration folder (…_parallel_diag beside the ray
one), built with dataset.build and trained with training.train — the same
functions the four-step study uses — so the benchmark cannot produce a
different answer than running the steps by hand.

THE THRESHOLD KNOB.  The U-Net outputs a probability per pixel; a prediction
only exists after that map is cut at a threshold.  By default the cut is the
one chosen on the VALIDATION split during training and stored in the
checkpoint (nothing tuned on test).  run_7's PROB_THRESHOLD overrides it with
one fixed value for EVERY arm — a fair sensitivity check, clearly labelled in
every output as an override.  The threshold scan figure shows how the whole
comparison moves as that knob turns.

Writes results/benchmark_<budget>[ _thr<value> ]/ — table, metrics.json, and
the publication figures (see FIGURES in the code below).

CONTEXT FOR THE PAPER.  Hernandes et al. reconstruct charge stability
diagrams from sparse LINE-CUT masks with a deep network; our parallel /
line-cut arms spend the budget the way their masks do, while the fan aims
every line obliquely across the honeycomb from one corner.  Adding "hcuts"
to the arm list gives the Hernandes-style baseline directly.
"""
import csv
import json
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import log, paths
from ..ml import grid_train
from ..ml.grid_metrics import evaluate, tolerant_f1
from . import dataset, sampling, training
from .config import StudyConfig
from .dataset import load_split
from .evaluation import TAUS, _device_extent, per_device_rows

_RULE = "=" * 74

# One colour per arm, fixed, so the same arm is the same colour in every
# figure of the report (and every re-run of it).
ARM_COLORS = {
    "rays": "#1f5fa8",
    "parallel_diag": "#c0392b",
    "hcuts": "#7d3c98",
    "vcuts": "#b9770e",
    "random_rays": "#148f77",
    "grid": "#5d6d7e",
    "random": "#884ea0",
}
ARM_LABELS = {
    "rays": "corner fan (ours)",
    "parallel_diag": "parallel diagonal",
    "hcuts": "horizontal cuts (Hernandes-style)",
    "vcuts": "vertical cuts",
    "random_rays": "random lines",
    "grid": "lattice",
    "random": "random pixels",
}

THRESHOLD_GRID = np.round(np.arange(0.30, 0.96, 0.05), 2)


def _label(strategy: str) -> str:
    return ARM_LABELS.get(strategy, strategy)


def _color(strategy: str) -> str:
    return ARM_COLORS.get(strategy, "#333333")


def arm_configs(template: StudyConfig, strategies: Sequence[str],
                n_rays: int, n_points: int, n_train: int, n_test: int,
                epochs: int, figure_devices: Dict) -> List[StudyConfig]:
    """One StudyConfig per arm — identical except `sampling`."""
    for s in strategies:
        if s not in sampling.STRATEGIES:
            raise KeyError(f"unknown strategy {s!r}; available: "
                           + ", ".join(sampling.STRATEGIES))
    return [replace(template, sampling=s,
                    n_rays=n_rays, n_points=n_points,
                    n_train=n_train, n_test=n_test, epochs=epochs,
                    figure_devices=dict(figure_devices))
            for s in strategies]


# ── scoring one arm, with the threshold knob ─────────────────────────────

def score_arm(cfg: StudyConfig, threshold: Optional[float]) -> Dict:
    """
    Predict on the arm's held-out devices and measure.

    threshold=None uses the checkpoint's own validation-chosen cut (the
    honest default); a number overrides it, identically for every arm, and
    the override is recorded in the metrics so no figure can pass it off as
    the validation choice.
    """
    X, Y, sample_dirs = load_split(cfg.test_npz)
    net, ck = grid_train.load(cfg.checkpoint)
    val_thr = float(ck["threshold"])
    thr = val_thr if threshold is None else float(threshold)

    log.say(f"scoring '{cfg.sampling}': {len(X)} held-out devices, "
            f"threshold {thr:g}"
            + ("" if threshold is None else
               f" (OVERRIDE — validation chose {val_thr:g})"))

    prob = grid_train.predict(net, X)
    pred = prob > thr
    rows = per_device_rows(pred, Y, sample_dirs)

    metrics = {
        "strategy": cfg.sampling,
        "configuration": cfg.name,
        "n_rays": cfg.n_rays,
        "n_points": cfg.n_points,
        "budget": int(cfg.n_rays * cfg.n_points),
        "coverage": float(X[:, 1].mean()),
        "threshold": thr,
        "threshold_source": ("validation (checkpoint)" if threshold is None
                             else "manual override (run_7 PROB_THRESHOLD)"),
        "validation_threshold": val_thr,
        "n_test_devices": int(len(X)),
        **evaluate(pred, Y, taus=TAUS),
    }
    f1s = np.array([r["f1@1"] for r in rows])
    metrics["f1@1_std"] = float(f1s.std())
    metrics["f1@1_min"] = float(f1s.min())
    metrics["f1@1_max"] = float(f1s.max())
    return {"cfg": cfg, "metrics": metrics, "rows": rows,
            "X": X, "Y": Y, "prob": prob, "pred": pred,
            "sample_dirs": sample_dirs}


def threshold_scan(arm: Dict, max_devices: int = 40) -> List[Dict]:
    """Mean F1@1 across the first `max_devices` test devices, per threshold."""
    prob, Y = arm["prob"][:max_devices], arm["Y"][:max_devices]
    out = []
    for t in THRESHOLD_GRID:
        f1 = np.mean([tolerant_f1(p > t, y, 1)["f1"]
                      for p, y in zip(prob, Y)])
        out.append({"threshold": float(t), "f1@1": float(f1)})
    return out


# ── figures ───────────────────────────────────────────────────────────────

def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def _save(fig, path: str, dpi: int):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.detail(f"  wrote {os.path.basename(path)}")


def fig_measurement_masks(arms: List[Dict], out: str, dpi: int):
    """WHERE each arm puts the budget — the geometry itself, one device."""
    fig, axes = plt.subplots(1, len(arms),
                             figsize=(4.2 * len(arms), 4.4), squeeze=False)
    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("white")
    for ax, arm in zip(axes[0], arms):
        X = arm["X"]
        visited = X[0, 1] > 0.5
        ext = _device_extent(arm["sample_dirs"][0])
        ax.imshow(np.where(visited, X[0, 0], np.nan), origin="lower",
                  cmap=cm, vmin=0, vmax=1, extent=ext, aspect="auto",
                  interpolation="nearest")
        m = arm["metrics"]
        ax.set_title(f"{_label(m['strategy'])}\n"
                     f"{m['budget']} points, {100 * m['coverage']:.1f}% of "
                     f"the grid", fontsize=10)
        if ext is None:
            _blank(ax)
    _save(fig, os.path.join(out, "fig1_measurement_masks.png"), dpi)


def fig_f1_bars(arms: List[Dict], out: str, dpi: int):
    """THE figure: F1@1 per arm, mean bar + every held-out device as a dot."""
    fig, ax = plt.subplots(figsize=(1.9 * len(arms) + 2.4, 4.2))
    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        m, rows = arm["metrics"], arm["rows"]
        f1s = np.array([r["f1@1"] for r in rows])
        ax.bar(i, m["f1@1"], width=0.55, color=_color(m["strategy"]),
               alpha=0.75, yerr=m["f1@1_std"], capsize=4,
               error_kw=dict(lw=1.1))
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(f1s)), f1s, s=12,
                   color="#222222", alpha=0.45, zorder=3)
        ax.text(i, 0.02, f"{m['f1@1']:.3f}", ha="center", fontsize=9,
                color="white", weight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([_label(a["metrics"]["strategy"]) for a in arms],
                       fontsize=9)
    ax.set_ylabel("F1 @ tolerance 1 px (held-out devices)")
    ax.set_ylim(0, 1)
    _despine(ax)
    thr = arms[0]["metrics"]
    ax.set_title(f"same budget: {thr['n_rays']} lines x {thr['n_points']} "
                 f"points, threshold {thr['threshold']:g}", fontsize=10)
    _save(fig, os.path.join(out, "fig2_f1_bars.png"), dpi)


def fig_f1_vs_tolerance(arms: List[Dict], out: str, dpi: int):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for arm in arms:
        m = arm["metrics"]
        ax.plot(TAUS, [m[f"f1@{t}"] for t in TAUS], "o-", lw=1.8, ms=5,
                color=_color(m["strategy"]), label=_label(m["strategy"]))
    ax.set_xlabel("tolerance tau (pixels)")
    ax.set_ylabel("F1 on the held-out devices")
    ax.set_xticks(list(TAUS))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    _despine(ax)
    _save(fig, os.path.join(out, "fig3_f1_vs_tolerance.png"), dpi)


def fig_precision_recall(arms: List[Dict], out: str, dpi: int):
    keys = ("precision@1", "recall@1", "f1@1")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        m = arm["metrics"]
        x = np.arange(len(keys)) + (i - (len(arms) - 1) / 2) * width
        ax.bar(x, [m[k] for k in keys], width=width * 0.92,
               color=_color(m["strategy"]), alpha=0.8,
               label=_label(m["strategy"]))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["precision", "recall", "F1"], fontsize=10)
    ax.set_ylabel("score @ tolerance 1 px")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    _despine(ax)
    _save(fig, os.path.join(out, "fig4_precision_recall.png"), dpi)


def fig_per_device_scatter(arms: List[Dict], out: str, dpi: int):
    """Rays vs each other arm, PAIRED on the same device: above the diagonal
    the fan won on that device, below it lost.  The pairing removes the
    device-to-device spread that a bar chart mixes into the error bar."""
    others = [a for a in arms if a["metrics"]["strategy"] != "rays"]
    base = next((a for a in arms if a["metrics"]["strategy"] == "rays"), None)
    if base is None or not others:
        return
    fig, axes = plt.subplots(1, len(others),
                             figsize=(4.4 * len(others), 4.4), squeeze=False)
    ray_f1 = np.array([r["f1@1"] for r in base["rows"]])
    for ax, arm in zip(axes[0], others):
        m = arm["metrics"]
        f1 = np.array([r["f1@1"] for r in arm["rows"]])
        n = min(len(ray_f1), len(f1))
        ax.plot([0, 1], [0, 1], "-", color="#999999", lw=1)
        ax.scatter(f1[:n], ray_f1[:n], s=18, color=_color(m["strategy"]),
                   alpha=0.6)
        wins = int((ray_f1[:n] > f1[:n]).sum())
        ax.set_xlabel(f"F1@1 — {_label(m['strategy'])}")
        ax.set_ylabel("F1@1 — corner fan")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"fan better on {wins}/{n} devices", fontsize=10)
        _despine(ax)
    _save(fig, os.path.join(out, "fig5_per_device_scatter.png"), dpi)


def fig_threshold_curves(arms: List[Dict], scans: Dict[str, List[Dict]],
                         out: str, dpi: int):
    """F1@1 against the binarisation threshold, per arm — the knob's effect.
    If the arms keep their order across the whole scan, the comparison does
    not depend on where PROB_THRESHOLD was set; this figure is that proof."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for arm in arms:
        m = arm["metrics"]
        rows = scans[m["strategy"]]
        ax.plot([r["threshold"] for r in rows], [r["f1@1"] for r in rows],
                "o-", lw=1.8, ms=4, color=_color(m["strategy"]),
                label=_label(m["strategy"]))
        ax.axvline(m["threshold"], color=_color(m["strategy"]), lw=1.0,
                   ls="--", alpha=0.6)
    ax.set_xlabel("binarisation threshold on the U-Net probability")
    ax.set_ylabel("F1 @ tolerance 1 px")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("dashed: the threshold each arm was scored at", fontsize=9)
    _despine(ax)
    _save(fig, os.path.join(out, "fig6_f1_vs_threshold.png"), dpi)


def fig_gallery(arms: List[Dict], device_idx: int, out: str, dpi: int):
    """One test device across every arm: measurement / probability /
    prediction per arm, plus the shared ground truth — the picture that shows
    WHY one geometry wins, not only that it does."""
    n = len(arms)
    fig, axes = plt.subplots(n, 4, figsize=(15.5, 3.7 * n), squeeze=False)
    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("white")
    name = os.path.basename(arms[0]["sample_dirs"][device_idx])
    for r, arm in enumerate(arms):
        m = arm["metrics"]
        X, Y = arm["X"], arm["Y"]
        visited = X[device_idx, 1] > 0.5
        row_f1 = arm["rows"][device_idx]["f1@1"]
        panels = [
            (np.where(visited, X[device_idx, 0], np.nan),
             dict(cmap=cm, vmin=0, vmax=1), "measurement"),
            (arm["prob"][device_idx],
             dict(cmap="magma", vmin=0, vmax=1), "U-Net probability"),
            (1 - arm["pred"][device_idx],
             dict(cmap="gray"), f"prediction  (F1@1 {row_f1:.3f})"),
            (1 - (Y[device_idx] > 0.5),
             dict(cmap="gray"), "ground truth"),
        ]
        for c, (img, kw, title) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(img, origin="lower", interpolation="nearest",
                      aspect="auto", **kw)
            _blank(ax)
            ax.set_title(title, fontsize=9)
        axes[r][0].set_ylabel(_label(m["strategy"]), fontsize=11)
    fig.suptitle(f"{name} — same device, same budget, different geometry",
                 fontsize=12, weight="bold")
    _save(fig, os.path.join(out, f"fig7_gallery_{name}.png"), dpi)


# ── table, json, readme ──────────────────────────────────────────────────

_COLUMNS = ("strategy", "budget", "coverage", "threshold",
            "precision@1", "recall@1", "f1@1", "f1@1_std", "iou")


def write_table(arms: List[Dict], out: str) -> str:
    rows = [{k: a["metrics"][k] for k in _COLUMNS} for a in arms]
    with open(os.path.join(out, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    lines = [_RULE, "BENCHMARK — same budget, different geometry", _RULE, "",
             f"{'arm':<28}{'coverage':>10}{'thr':>7}{'P@1':>8}{'R@1':>8}"
             f"{'F1@1':>8}{'+/-sd':>8}{'IoU':>8}"]
    for a in arms:
        m = a["metrics"]
        lines.append(f"{_label(m['strategy']):<28}"
                     f"{100 * m['coverage']:>9.2f}%{m['threshold']:>7.2f}"
                     f"{m['precision@1']:>8.3f}{m['recall@1']:>8.3f}"
                     f"{m['f1@1']:>8.3f}{m['f1@1_std']:>8.3f}"
                     f"{m['iou']:>8.3f}")
    lines.append(_RULE)
    text = "\n".join(lines)
    with open(os.path.join(out, "benchmark.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump([a["metrics"] for a in arms], f, indent=2)
    return text


def write_readme(arms: List[Dict], out: str):
    m0 = arms[0]["metrics"]
    text = f"""WHAT THIS FOLDER IS
Every arm measured the SAME {m0['n_rays']} x {m0['n_points']} =
{m0['budget']}-point budget on the SAME devices and trained the SAME network;
only WHERE the points were put differs.  Binarisation threshold:
{m0['threshold']:g} ({m0['threshold_source']}).

fig1  where each geometry puts its points (one test device)
fig2  F1@1 per arm, every held-out device as a dot        <- headline
fig3  F1 against tolerance: how much error is sub-pixel
fig4  precision / recall / F1 side by side
fig5  paired per-device scatter, fan vs each other arm
fig6  F1 against the threshold knob, per arm
fig7  one device across all arms: measurement -> probability -> prediction

RELATION TO HERNANDES ET AL.  Their reconstruction of charge stability
diagrams feeds a deep network sparse LINE-CUT masks (our 'hcuts' arm spends
the budget exactly that way).  The parallel-diagonal arm is the strongest
line-cut variant — oblique like the fan, so it crosses both families of
honeycomb lines — which makes the fan-vs-parallel gap attributable to the
corner-fan geometry itself rather than to line direction.
"""
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(text)


# ── the whole benchmark ──────────────────────────────────────────────────

def out_dir(n_rays: int, n_points: int, n_train: int,
            threshold: Optional[float], name: Optional[str]) -> str:
    """results/benchmark_<budget>[_thr<value>]/ — a threshold override gets
    its OWN folder, so scoring at 0.5 cannot overwrite the honest run."""
    if name:
        return os.path.join(paths.RESULTS, name)
    base = f"benchmark_{n_rays}_rays_{n_points}_points_{n_train}_samples"
    if threshold is not None:
        base += f"_thr{threshold:g}"
    return os.path.join(paths.RESULTS, base)


def run(template: StudyConfig, strategies: Sequence[str],
        n_rays: int, n_points: int, n_train: int, n_test: int, epochs: int,
        threshold: Optional[float] = None, skip_existing: bool = True,
        gallery_devices: Sequence[int] = (1, 2, 3),
        scan_devices: int = 40, dpi: int = 300,
        figure_devices: Optional[Dict] = None,
        name: Optional[str] = None) -> List[Dict]:
    """Stages 1-3 per arm (reusing anything on disk), then every figure."""
    t0 = time.time()
    cfgs = arm_configs(template, strategies, n_rays, n_points, n_train,
                       n_test, epochs, figure_devices or {})

    log.say(_RULE)
    log.say(f"run_7 — {' vs '.join(_label(s) for s in strategies)} at "
            f"{n_rays} x {n_points} = {n_rays * n_points} points")
    log.say(_RULE)

    arms = []
    for i, cfg in enumerate(cfgs, 1):
        log.say(f"\n[{i}/{len(cfgs)}] arm '{cfg.sampling}' — {cfg.name}")
        report = dataset.build(cfg)
        if not report["all_passed"]:
            raise RuntimeError(f"dataset split check FAILED for {cfg.name}")
        if skip_existing and os.path.isfile(cfg.checkpoint):
            log.say(f"reusing {os.path.abspath(cfg.checkpoint)}")
        else:
            training.train(cfg)
        arms.append(score_arm(cfg, threshold))

    out = out_dir(n_rays, n_points, n_train, threshold, name)
    os.makedirs(out, exist_ok=True)
    log.say(f"\nfigures and table -> {os.path.abspath(out)}")

    scans = {a["metrics"]["strategy"]: threshold_scan(a, scan_devices)
             for a in arms}
    fig_measurement_masks(arms, out, dpi)
    fig_f1_bars(arms, out, dpi)
    fig_f1_vs_tolerance(arms, out, dpi)
    fig_precision_recall(arms, out, dpi)
    fig_per_device_scatter(arms, out, dpi)
    fig_threshold_curves(arms, scans, out, dpi)
    n_avail = len(arms[0]["X"])
    for d in gallery_devices:
        idx = int(d) - 1
        if 0 <= idx < n_avail:
            fig_gallery(arms, idx, out, dpi)
        else:
            log.detail(f"  [skip] gallery device {d}: only {n_avail} "
                       f"test devices")
    write_readme(arms, out)
    log.say("\n" + write_table(arms, out))
    log.say(f"total time: {(time.time() - t0) / 60:.1f} min")
    return arms
