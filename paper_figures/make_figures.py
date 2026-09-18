"""
paper_figures/make_figures.py — publication figures, drawn from the real
arrays in the device pool rather than mocked up.

    fig_network_input   the TWO maps the U-Net is actually given
    fig_unet            a compact schematic of the network

Run from the project root:   python paper_figures/make_figures.py

Every figure is written as .png (600 dpi, for slides) and .pdf (vector, for
the manuscript).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from dqd.ml import ray_peaks, grid_train                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
POOL = os.path.join(ROOT, "training_data", "_device_pools",
                    "devices_n550_res100_c1df7b6bf")

# Every figure in the deck shows ONE device: #33 from the charge-sensor
# gallery, simulated in its 0.8 mV window.  It is a freshly generated device
# — not in the 550-device pool at all — so the network has never seen it.
# NOTE: the network was trained on 2 mV windows.  At 0.8 mV the honeycomb is
# coarser than anything it was fitted on, so the reconstruction here scores
# below the test-set mean.  See picked_device_33/scores.csv.
PICK = os.path.join(ROOT, "results",
                    "4-5-6-7-8_rays_40-50-60_points_500_samples",
                    "picked_device_33", "device_0.8mV")
DEVICE = PICK
N_RAYS, N_POINTS = 8, 60

INK = "#111111"
MUT = "#5a606a"

# Error colours for the reconstruction panels, taken from the deck's own
# theme (teal / red / plum) rather than green and orange.
HIT_HEX, MISS_HEX, FALSE_HEX = "#1B587C", "#9F2936", "#7A5C99"
HIT_RGB = (0.106, 0.345, 0.486)
MISS_RGB = (0.624, 0.161, 0.212)
FALSE_RGB = (0.478, 0.361, 0.600)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.edgecolor": "#999999",
    "savefig.facecolor": "white",
})


def save(fig, name):
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {})):
        p = os.path.join(HERE, name + ext)
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("  ->", os.path.relpath(p, ROOT))
    plt.close(fig)


# ── figure 1: the two input maps ──────────────────────────────────────────
def fig_network_input(compact=False):
    m = ray_peaks.measure(DEVICE, N_RAYS, N_POINTS)
    ux, uy, Z = ray_peaks.load_grid(DEVICE)
    H, W = Z.shape
    ch = ray_peaks.to_channels(m, (H, W))
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]
    cov = 100.0 * vis.mean()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))

    # channel 1 — the measured value, shown only where a ray passed
    cm = plt.get_cmap("hot").copy()
    cm.set_bad("#f2f2f2")
    ax = axes[0]
    im = ax.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("channel 1   —   measured sensor signal", fontsize=11,
                 color=INK, pad=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("normalised charge-sensor signal", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("grey = never measured", fontsize=9, color=MUT)

    # channel 2 — where we looked at all
    ax = axes[1]
    ax.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title("channel 2   —   visited mask", fontsize=11, color=INK,
                 pad=9)
    ax.set_xlabel(f"black = a ray passed here  ({cov:.1f} % of the grid)",
                  fontsize=9, color=MUT)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#999999")

    if not compact:
        # the manuscript version carries its own title and caption; the slide
        # version does not, because the slide already says both
        fig.suptitle(f"What the network is given — {N_RAYS} rays × "
                     f"{N_POINTS} points, one held-out device",
                     fontsize=12.5, color=INK, y=1.0)
        fig.text(0.5, -0.045,
                 "Channel 2 is what separates “measured here, and the signal "
                 "was low” from “never looked here”.  Without it a zero in "
                 "channel 1 is ambiguous everywhere.",
                 ha="center", fontsize=9.5, color=MUT)
    fig.tight_layout()
    save(fig, "fig_network_input_slide" if compact else "fig_network_input")


# ── figure 2: compact U-Net schematic ─────────────────────────────────────
ENC = "#4f81a8"
BOT = "#3d4a5c"
DEC = "#c07a4a"
IO = "#5f8a5f"


_BLOCK_FS = 1.0          # set by draw_unet so sub-captions scale with fs


def _block(ax, x, y, w, h, color, label, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=0, facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y - 0.055, sub, ha="center", va="top",
                fontsize=8.5 * _BLOCK_FS, color=MUT)


def _arrow(ax, p, q, color, style="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=style,
                                 shrinkA=1, shrinkB=1, zorder=1))


def draw_unet(ax, fs=1.0, io_labels=True):
    """Draw the U-Net schematic into an existing axes.  fs scales the fonts;
    io_labels=False drops the input/output captions, for when the figure
    already shows the real input and output either side."""
    global _BLOCK_FS
    _BLOCK_FS = fs
    ax.set_xlim(0, 10); ax.set_ylim(-0.32, 1.25); ax.axis("off")

    # x, y, w, h, colour, label, caption
    lvl = [(0.72, 0.85), (0.56, 0.60), (0.40, 0.35)]      # h, y per depth
    enc_x = [1.35, 2.35, 3.35]
    dec_x = [5.30, 6.30, 7.30]
    widths = ["32", "64", "128"]
    grids = ["100²", "50²", "25²"]

    _block(ax, 0.15, 0.30, 0.55, 0.72, IO, "2", "input")
    for i, (x, w, g) in enumerate(zip(enc_x, widths, grids)):
        h, yt = lvl[i]
        _block(ax, x, yt - h, 0.72, h, ENC, w, g)
    _block(ax, 4.30, 0.00, 0.72, 0.28, BOT, "256", "12²")
    for i, (x, w, g) in enumerate(zip(dec_x, reversed(widths),
                                      reversed(grids))):
        h, yt = lvl[2 - i]
        _block(ax, x, yt - h, 0.72, h, DEC, w, g)
    _block(ax, 8.40, 0.30, 0.55, 0.72, IO, "1", "output")

    # down / up path
    _arrow(ax, (0.72, 0.66), (1.33, 0.66), INK)
    for i in range(2):
        h0, y0 = lvl[i]; h1, y1 = lvl[i + 1]
        _arrow(ax, (enc_x[i] + 0.72, y0 - h0 / 2),
               (enc_x[i + 1], y1 - h1 / 2), ENC)
    h, y = lvl[2]
    _arrow(ax, (enc_x[2] + 0.72, y - h / 2), (4.30, 0.14), ENC)
    _arrow(ax, (5.02, 0.14), (dec_x[0], y - h / 2), DEC)
    for i in range(2):
        h0, y0 = lvl[2 - i]; h1, y1 = lvl[1 - i]
        _arrow(ax, (dec_x[i] + 0.72, y0 - h0 / 2),
               (dec_x[i + 1], y1 - h1 / 2), DEC)
    _arrow(ax, (dec_x[2] + 0.72, 0.66), (8.38, 0.66), INK)

    # skip connections - the deepest one is routed clear of the bottleneck
    skip_y = [lvl[0][1] - 0.06, lvl[1][1] - 0.06, 0.33]
    for i in range(3):
        _arrow(ax, (enc_x[i] + 0.72, skip_y[i]),
               (dec_x[2 - i], skip_y[i]), "#9aa2ac", style=(0, (4, 3)),
               lw=1.2)
    ax.text(4.66, lvl[0][1] + 0.02, "skip connections",
            ha="center", fontsize=8.5 * fs, color=MUT, style="italic")

    if io_labels:
        ax.text(0.42, 0.16, "channel 1  ray signal\nchannel 2  visited mask",
                ha="center", va="top", fontsize=8.5 * fs, color=MUT)
        ax.text(8.68, 0.16, "P(transition line)\nper pixel",
                ha="center", va="top", fontsize=8.5 * fs, color=MUT)
    ax.text(2.43, 1.16, "encoder", fontsize=10.5 * fs, color=ENC,
            fontweight="bold", ha="center")
    ax.text(6.38, 1.16, "decoder", fontsize=10.5 * fs, color=DEC,
            fontweight="bold", ha="center")
    ax.text(4.66, -0.14, "bottleneck", fontsize=9.5 * fs, color=BOT,
            fontweight="bold", ha="center", va="top")

    return ax


def fig_unet():
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    draw_unet(ax)
    save(fig, "fig_unet")




# ── figure 3: the probability map, and where the threshold comes from ─────
# No input channels here — those live on the model slide.  This figure is
# only about turning a continuous map into lines, and about the fact that
# the cut is chosen on a VALIDATION split, never on the test devices.
CONFIG = "8_rays_60_points_500_samples"
RUN_DIR = os.path.join(ROOT, "results",
                       "4-5-6-7-8_rays_40-50-60_points_500_samples")
LADDER = (0.4, 0.9)

# Which held-out device the results figure uses.  Index into test.npz:
#   0 -> figures/test/sample_1 (pool sample_5)
#   1 -> figures/test/sample_2 (pool sample_10)   <- the one we show
# Both are in test_ids, so the network never saw them during training.
RESULT_DEVICE = 1


def pick_case(net, n_rays=8, n_points=60):
    """(channels, truth, probability) for the picked device."""
    m = ray_peaks.measure(PICK, n_rays, n_points)
    ux, uy, Z = ray_peaks.load_grid(PICK)
    ch = ray_peaks.to_channels(m, Z.shape)
    truth = ray_peaks.load_ground_truth(PICK)
    prob = grid_train.predict(net, ch[None, :ray_peaks.NET_CHANNELS])[0]
    return ch, truth, prob


def _validation_curve(net, cfg_dir):
    """
    Re-derive the exact validation split the training used and score every
    candidate threshold on it.  grid_train carves it out with
    default_rng(SEED) before training, so this is reproducible.
    """
    from dqd.ml import grid_train
    from dqd.ml.grid_metrics import tolerant_f1
    from dqd.study.dataset import load_split

    Xtr, Ytr, _ = load_split(os.path.join(cfg_dir, "train.npz"))
    rng = np.random.default_rng(grid_train.SEED)
    idx = rng.permutation(len(Xtr))
    n_val = max(1, int(grid_train.VAL_FRACTION * len(Xtr)))
    vi = idx[:n_val]
    pv, Yv = grid_train.predict(net, Xtr[vi]), Ytr[vi]
    cand = np.array(grid_train.THRESHOLDS)
    f1 = np.array([np.mean([tolerant_f1(pv[j] > t, Yv[j], 1.0)["f1"]
                            for j in range(len(Yv))]) for t in cand])
    return cand, f1, len(vi)


def fig_probability_to_lines(device_index=RESULT_DEVICE, ladder=LADDER):
    from dqd.ml import grid_train
    from dqd.ml.grid_metrics import tolerant_f1
    from dqd.study.dataset import load_split
    from scipy.ndimage import distance_transform_edt

    cfg_dir = os.path.join(RUN_DIR, CONFIG)
    net, ck = grid_train.load(os.path.join(cfg_dir, "model", "unet.pt"))
    thr = float(ck["threshold"])
    ch, Yt, p = pick_case(net)
    Y = {0: Yt}                      # keep the indexing below unchanged
    i = 0
    truth = Yt > 0.5
    print(f"  picked device #33 (0.8 mV window)   threshold {thr:g}")

    TAUS = (0, 1, 3)
    d_true_by_tau = {}
    fig = plt.figure(figsize=(12.6, 8.6))
    # three panels per row, so each is as large as the slide allows
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          hspace=0.40, wspace=0.12,
                          left=0.035, right=0.99, top=0.88, bottom=0.075)

    def blank(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999999")

    def err_rgb(pred, tau):
        """teal = found within tau, red = missed line, plum = false line."""
        if tau not in d_true_by_tau:
            d_true_by_tau[tau] = (distance_transform_edt(~truth)
                                  if truth.any()
                                  else np.full(truth.shape, np.inf))
        dt = d_true_by_tau[tau]
        dp = (distance_transform_edt(~pred) if pred.any()
              else np.full(pred.shape, np.inf))
        rgb = np.ones(truth.shape + (3,))
        rgb[pred & (dt > tau)] = FALSE_RGB
        rgb[truth & (dp > tau)] = MISS_RGB
        rgb[pred & (dt <= tau)] = HIT_RGB
        return rgb

    # ── row 1: the output, the truth, and the map cut four ways ──────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(p, origin="lower", cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title("U-Net output\nP(transition line)", fontsize=11.5,
                 color="#7d2b3a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=6.5)
    blank(ax)

    for k, t in enumerate(ladder):
        pred = p > t
        m = tolerant_f1(pred, Yt, 1.0)
        ax = fig.add_subplot(gs[0, 1 + k])
        ax.imshow(err_rgb(pred, 1.0), origin="lower",
                  interpolation="nearest")
        chosen = abs(t - thr) < 1e-9
        ax.set_title(f"P > {t:g}" + ("  (chosen)" if chosen else "") +
                     f"\nF1@1 {m['f1']:.3f}", fontsize=11.5,
                     color="#c0392b" if chosen else INK,
                     fontweight="bold" if chosen else "normal")
        blank(ax)
        if chosen:
            for sp in ax.spines.values():
                sp.set_color("#c0392b"); sp.set_linewidth(2.2)

    # ── row 2: ONE prediction, scored at three tolerances ─────────────────
    pred = p > thr
    for k, tau in enumerate(TAUS):
        m = tolerant_f1(pred, Yt, float(tau))
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(err_rgb(pred, float(tau)), origin="lower",
                  interpolation="nearest")
        head = {0: "τ = 0   strict"}.get(tau, f"τ = {tau}")
        ax.set_title(f"{head}\nF1@{tau} {m['f1']:.3f}", fontsize=11.5,
                     color="#1f5fa8" if tau == 1 else INK,
                     fontweight="bold" if tau == 1 else "normal")
        blank(ax)
        if tau == 1:
            for sp in ax.spines.values():
                sp.set_color("#1f5fa8"); sp.set_linewidth(2.2)

    fig.text(0.5, 0.975, "cutting the probability map at two thresholds",
             ha="center", fontsize=10.5, color=MUT)
    fig.text(0.5, 0.492, "the SAME prediction (P > "
             f"{thr:g}), scored at three tolerances τ — a predicted pixel "
             "counts as correct when a true line lies within τ pixels",
             ha="center", fontsize=10.5, color=MUT)
    for _x, _t, _c in (
            (0.22, "found  — a predicted line that really is there", HIT_HEX),
            (0.53, "missed  — a real line the model did not draw", MISS_HEX),
            (0.83, "false  — a line drawn where there is none", FALSE_HEX)):
        fig.text(_x, 0.018, "■  " + _t, ha="center", fontsize=10.5,
                 color=_c, fontweight="bold")

    save(fig, "fig_probability_to_lines")


# ── figure 4: the whole flow — input | network | output ───────────────────
def fig_model_flow(device_index=RESULT_DEVICE):
    """
    The two input maps on the left, the network in the middle, the
    probability map on the right — one picture of what the model does.
    """
    from dqd.ml import grid_train
    from dqd.study.dataset import load_split

    cfg_dir = os.path.join(RUN_DIR, CONFIG)
    net, ck = grid_train.load(os.path.join(cfg_dir, "model", "unet.pt"))
    ch, _, p = pick_case(net)
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]

    fig = plt.figure(figsize=(13.0, 3.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 3.45, 1.25],
                          wspace=0.30, hspace=0.42,
                          left=0.02, right=0.985, top=0.86, bottom=0.06)

    def blank(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999999")

    cm = plt.get_cmap("hot").copy(); cm.set_bad("#f2f2f2")
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.where(vis > 0.5, sig, np.nan), origin="lower", cmap=cm,
               vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("channel 1 · measured signal", fontsize=9.5, pad=5)
    blank(ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(1.0 - vis, origin="lower", cmap="gray", vmin=0, vmax=1,
               interpolation="nearest")
    ax2.set_title(f"channel 2 · visited mask ({100 * vis.mean():.1f} %)",
                  fontsize=9.5, pad=5)
    blank(ax2)

    axn = fig.add_subplot(gs[:, 1])
    draw_unet(axn, fs=0.92, io_labels=False)

    axo = fig.add_subplot(gs[:, 2])
    im = axo.imshow(p, origin="lower", cmap="magma", vmin=0, vmax=1,
                    interpolation="nearest")
    axo.set_title("P(transition line) per pixel", fontsize=10.5,
                  color="#7d2b3a", pad=6)
    fig.colorbar(im, ax=axo, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=7.5)
    blank(axo)

    # arrows in figure coordinates: inputs -> network -> output
    def farrow(x0, y0, x1, y1):
        fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=15,
                                       linewidth=1.8, color="#444444"))
    b1, b2 = ax1.get_position(), ax2.get_position()
    bn, bo = axn.get_position(), axo.get_position()
    farrow(b1.x1 + 0.004, b1.y0 + b1.height * 0.5,
           bn.x0 + 0.012, bn.y0 + bn.height * 0.62)
    farrow(b2.x1 + 0.004, b2.y0 + b2.height * 0.5,
           bn.x0 + 0.012, bn.y0 + bn.height * 0.62)
    farrow(bn.x1 - 0.010, bn.y0 + bn.height * 0.62,
           bo.x0 - 0.006, bo.y0 + bo.height * 0.5)

    fig.text(0.5, 0.955, "Two measured maps in, one probability map out",
             ha="center", fontsize=12.5, color=INK)
    save(fig, "fig_model_flow")


# ── figure 5: the same three maps, each as its own standalone panel ───────
# Separate files so a slide can lay them out itself, with its own arrows,
# instead of being handed one merged image.
def fig_panels(device_index=RESULT_DEVICE):
    from dqd.ml import grid_train
    from dqd.study.dataset import load_split

    cfg_dir = os.path.join(RUN_DIR, CONFIG)
    net, ck = grid_train.load(os.path.join(cfg_dir, "model", "unet.pt"))
    ch, _, prob = pick_case(net)
    sig, vis = ch[ray_peaks.CH_SIGNAL], ch[ray_peaks.CH_VISITED]

    def panel(draw, title, name, colour=INK, cbar=None):
        fig, ax = plt.subplots(figsize=(2.7, 2.95))
        im = draw(ax)
        ax.set_title(title, fontsize=11, color=colour, pad=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999999")
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        save(fig, name)

    cm = plt.get_cmap("hot").copy(); cm.set_bad("#f2f2f2")
    panel(lambda ax: ax.imshow(np.where(vis > 0.5, sig, np.nan),
                               origin="lower", cmap=cm, vmin=0, vmax=1,
                               interpolation="nearest"),
          "channel 1\nmeasured sensor signal", "panel_channel1")
    panel(lambda ax: ax.imshow(1.0 - vis, origin="lower", cmap="gray",
                               vmin=0, vmax=1, interpolation="nearest"),
          f"channel 2\nvisited mask ({100 * vis.mean():.1f} % of the grid)",
          "panel_channel2")
    panel(lambda ax: ax.imshow(prob, origin="lower", cmap=J_CMAP, vmin=0,
                               vmax=1, interpolation="nearest"),
          "output\nP(transition line) per pixel", "panel_probability",
          colour=J_PTITLE, cbar=True)


# ── figure 6: the charge sensor, raw and with its background removed ──────
# The raw sensor signal is a large smooth gradient with the transition lines
# riding on it as a small modulation — which is why the raw map looks like a
# featureless wash.  Showing both is the honest way to present it, and it is
# also the reason the whole problem is hard.
def fig_charge_sensor(device=None):
    from scipy.ndimage import gaussian_filter

    device = device or os.path.join(POOL, "sample_10")   # = test/sample_2
    ux, uy, Z = ray_peaks.load_grid(device)
    gy, gx = np.gradient(Z)
    g = gx + gy
    detail = g - gaussian_filter(g, 3.0)                 # high-pass
    lim = np.percentile(np.abs(detail), 99.0)

    ext = [ux.min(), ux.max(), uy.min(), uy.max()]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1))

    ax = axes[0]
    im = ax.imshow(Z, origin="lower", cmap="hot", extent=ext,
                   aspect="auto", interpolation="nearest")
    ax.set_title("as measured\na large smooth background", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=8)

    ax = axes[1]
    im = ax.imshow(detail, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=ext, aspect="auto", interpolation="nearest")
    ax.set_title("same data, slow background removed\nthe transition "
                 "lines appear", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=8)

    for ax in axes:
        ax.set_xlabel("$V_1$ (mV)", fontsize=9.5)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#999999")
    axes[0].set_ylabel("$V_2$ (mV)", fontsize=9.5)

    fig.tight_layout()
    save(fig, "panel_charge_sensor")


# ── figure 7: the measurement story for the picked device ────────────────
def fig_measurement_panel(n_rays=8, n_points=60):
    """charge sensor | ground truth | rays and peaks | what the net is shown"""
    ux, uy, Z = ray_peaks.load_grid(PICK)
    m = ray_peaks.measure(PICK, n_rays, n_points)
    ch = ray_peaks.to_channels(m, Z.shape)
    truth = ray_peaks.load_ground_truth(PICK) > 0.5
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]
    sx = (ux.max() - ux.min()) / (len(ux) - 1)
    sy = (uy.max() - uy.min()) / (len(uy) - 1)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))

    axes[0].imshow(Z, origin="lower", cmap="hot", extent=ext, aspect="auto",
                   interpolation="nearest")
    axes[0].set_title("charge sensor", fontsize=11)

    axes[1].imshow(1 - truth, origin="lower", cmap="gray", extent=ext,
                   aspect="auto", interpolation="nearest")
    axes[1].set_title("stability diagram\n(ground truth)", fontsize=11)

    cm = plt.get_cmap("hot").copy(); cm.set_bad("#f2f2f2")
    vis = ch[ray_peaks.CH_VISITED]
    axes[2].imshow(np.where(vis > 0.5, ch[ray_peaks.CH_SIGNAL], np.nan),
                   origin="lower", cmap=cm, vmin=0, vmax=1, extent=ext,
                   aspect="auto", interpolation="nearest")
    axes[2].set_title(f"{n_rays} rays × {n_points} points\nwhat the network "
                      f"is shown  ({100 * vis.mean():.1f} % of the grid)",
                      fontsize=11)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#999999")
    fig.tight_layout()
    save(fig, "fig_measurement_panel")


if __name__ == "__main__":
    print("figures ->", HERE)
    fig_network_input()
    fig_network_input(compact=True)
    fig_unet()
    fig_probability_to_lines(RESULT_DEVICE)
    fig_probability_panels()
    fig_results_f1_vs_coverage()
    fig_data_split()
    fig_tau_metrics()
    fig_model_flow()
    fig_panels()
    fig_charge_sensor()
    fig_measurement_panel()


# ── figure 8: slide 9 / manuscript, one file per panel ────────────────────
# TWO colours, plus a tolerance band.  The panels answer one question — how
# close is the model's output to the truth — so they carry exactly the two
# things being compared:
#
#     black  the ground truth transition line (1 pixel wide)
#     red    the model's output after thresholding
#     grey   the tolerance band: every pixel within tau of the truth
#
# The band is what makes tau visible.  Note that tau is symmetric: for
# PRECISION the truth is dilated (is this red pixel near a true line?), for
# RECALL the prediction is dilated (is this true pixel near a red one?).
# The band drawn here is the precision side, which is the visible one.
J_TRUTH, J_PRED, J_BAND = "#000000", "#C00000", "#D9D9D9"
J_TRUTH_RGB = (0.000, 0.000, 0.000)
J_PRED_RGB = (0.753, 0.000, 0.000)
J_BAND_RGB = (0.851, 0.851, 0.851)
J_CMAP = "magma"                     # black = P 0, yellow = P 1
J_PTITLE = "#7d2b3a"                 # probability-panel title
J_GRID = "#9a9a9a"


# Titles on the "From probability to lines" panels can be left OFF, so the
# deck can carry them as real text boxes the author may delete.  The blank
# title keeps the same number of lines, so the reserved space -- and with
# it the figure's proportions and the slide geometry -- does not move.
P2L_TITLES_OFF = os.environ.get("P2L_TITLES") == "off"
P2L_TITLES = {}


def _p2l_name(stem):
    """Blanked panels get their own filenames, so turning titles off
    never overwrites the titled versions the manuscript uses."""
    return stem + "_notitle" if P2L_TITLES_OFF else stem


def _p2l_title(stem, text, colour, size, bold=False):
    """Record a panel title, and blank it when the deck supplies it."""
    P2L_TITLES[stem] = {"text": text.replace("$", ""), "colour": colour,
                        "size": size, "bold": bold}
    if not P2L_TITLES_OFF:
        return text
    # blanks of the SAME line count: an empty string has no height at
    # all, and matplotlib then reclaims the gap the box has to sit in
    return "\n".join([" "] * (text.count("\n") + 1))


def fig_probability_panels(ladder=LADDER, taus=(0, 1, 3)):
    import csv

    from dqd.ml import grid_train
    from dqd.ml.grid_metrics import tolerant_f1
    from scipy.ndimage import distance_transform_edt

    cfg_dir = os.path.join(RUN_DIR, CONFIG)
    net, ck = grid_train.load(os.path.join(cfg_dir, "model", "unet.pt"))
    thr = float(ck["threshold"])
    _ch, Yt, p = pick_case(net)
    truth = Yt > 0.5
    d_true = distance_transform_edt(~truth)

    ux, uy, _Z = ray_peaks.load_grid(PICK)
    ext = [ux.min(), ux.max(), uy.min(), uy.max()]

    def overlay(pred, tau):
        """grey band (truth +/- tau), then the red output, then black truth."""
        rgb = np.ones(truth.shape + (3,))
        rgb[d_true <= tau] = J_BAND_RGB
        rgb[pred] = J_PRED_RGB
        rgb[truth] = J_TRUTH_RGB
        return rgb

    def axes_style(ax):
        ax.set_xlabel("$V_1$ (mV)", fontsize=9, color=INK, labelpad=2)
        ax.set_ylabel("$V_2$ (mV)", fontsize=9, color=INK, labelpad=2)
        ax.tick_params(labelsize=7.5, length=3, width=0.7, direction="out",
                       colors=INK)
        ax.set_xticks(np.round(np.linspace(ext[0], ext[1], 5), 2))
        ax.set_yticks(np.round(np.linspace(ext[2], ext[3], 5), 2))
        ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
                alpha=0.85)
        ax.set_axisbelow(False)           # the grid sits ON TOP of the image
        for sp in ax.spines.values():
            sp.set_color("#444444")
            sp.set_linewidth(0.8)

    def panel(name, draw, title, colour=INK, boxed=None, cbar=False):
        fig, ax = plt.subplots(figsize=(2.55, 2.85))
        im = draw(ax)
        ax.set_title(_p2l_title(name, title, colour, 10.5, bool(boxed)),
                     fontsize=10.5, color=colour, pad=6,
                     fontweight="bold" if boxed else "normal")
        axes_style(ax)
        if boxed:
            for sp in ax.spines.values():
                sp.set_color(boxed)
                sp.set_linewidth(2.2)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7.5)
            cb.outline.set_linewidth(0.6)
        fig.tight_layout()
        save(fig, _p2l_name(name))

    def show(ax, rgb):
        return ax.imshow(rgb, origin="lower", extent=ext, aspect="auto",
                         interpolation="nearest")

    # 1 — the raw probability map
    panel("p2l_1_probability",
          lambda ax: ax.imshow(p, origin="lower", cmap=J_CMAP, vmin=0,
                               vmax=1, extent=ext, aspect="auto",
                               interpolation="nearest"),
          "U-Net output\n$P$(transition line)", colour=J_PTITLE, cbar=True)

    # 2, 3 — the same map cut at two thresholds.
    #
    # Two versions of each.  The tau = 1 pair is the manuscript's.  The
    # tau = 0 pair is for the slide, whose top row is about WHERE to cut and
    # so must show no tolerance band: tau has not been introduced yet at
    # that point in the talk.  Their titles say "exact match" rather than
    # "F1@0" for the same reason.
    for k, t in enumerate(ladder):
        pred_t = p > t
        m = tolerant_f1(pred_t, Yt, 1.0)
        m0 = tolerant_f1(pred_t, Yt, 0.0)
        chosen = abs(t - thr) < 1e-9
        head = f"$P$ > {t:g}" + ("  (chosen)" if chosen else "")
        stem = f"p2l_{2 + k}_threshold_{t:g}".replace(".", "p")
        panel(stem,
              lambda ax, pr=pred_t: show(ax, overlay(pr, 1.0)),
              head + f"\nF1@1 = {m['f1']:.3f}",
              colour=J_PRED if chosen else INK,
              boxed=J_PRED if chosen else None)
        panel(stem + "_tau0",
              lambda ax, pr=pred_t: show(ax, overlay(pr, 0.0)),
              head + f"\nF1 = {m0['f1']:.3f}  (exact match)",
              colour=J_PRED if chosen else INK,
              boxed=J_PRED if chosen else None)
        print("    P > %g:  F1@1 = %.3f   exact-match F1 = %.3f"
              % (t, m["f1"], m0["f1"]))
    # 4, 5, 6 — ONE prediction, three tolerances: only the band changes
    pred = p > thr
    for k, tau in enumerate(taus):
        m = tolerant_f1(pred, Yt, float(tau))
        head = "tau = 0  (no band)" if tau == 0 else f"tau = {tau}"
        head = head.replace("tau", "τ")
        panel(f"p2l_{4 + k}_tau{tau}",
              lambda ax, tt=tau: show(ax, overlay(pred, float(tt))),
              f"{head}\nF1@{tau} = {m['f1']:.3f}",
              colour=J_PRED if tau == 1 else INK,
              boxed=J_PRED if tau == 1 else None)

    # 7 — the tolerance curve: F1, and the precision / recall behind it
    with open(os.path.join(RUN_DIR, "comparison.csv"), newline="") as fh:
        row = next(r for r in csv.DictReader(fh)
                   if r["configuration"] == CONFIG)
    T = (0, 1, 2, 3)
    f1s = [float(row[f"f1@{t}"]) for t in T]
    prs = [float(row[f"precision@{t}"]) for t in T]
    rcs = [float(row[f"recall@{t}"]) for t in T]

    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    ax.plot(T, prs, "--s", color="#777777", linewidth=1.0, markersize=3.6,
            label="precision", zorder=2)
    ax.plot(T, rcs, ":^", color="#777777", linewidth=1.0, markersize=3.8,
            label="recall", zorder=2)
    ax.plot(T, f1s, "-o", color=J_TRUTH, linewidth=1.8, markersize=5,
            label="F1", zorder=3)
    ax.plot([1], [f1s[1]], "o", color=J_PRED, markersize=12,
            markerfacecolor="none", markeredgewidth=1.8, zorder=4)
    for x, y in zip(T, f1s):
        off, ha = {0: ((13, -4), "left"),
                   1: ((0, 16), "center")}.get(x, ((0, -16), "center"))
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=8.5,
                    color=J_PRED if x == 1 else INK)
    ax.set_xticks(list(T))
    ax.set_ylim(0.30, 1.02)
    ax.set_xlabel("tolerance τ  (pixels)", fontsize=9.5, color=INK)
    ax.set_ylabel("score on 50 held-out devices", fontsize=9.5, color=INK)
    ax.set_title(_p2l_title("p2l_7_tolerance_curve",
                            "τ = 0 → 1 is the big jump;\n"
                            "beyond that the curve flattens", INK, 10),
                 fontsize=10, color=INK, pad=6)
    ax.tick_params(labelsize=8.5, colors=INK)
    ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
            alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right",
              handlelength=2.2, borderpad=0.2, labelspacing=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#444444")
        ax.spines[sp].set_linewidth(0.8)
    fig.tight_layout()
    save(fig, _p2l_name("p2l_7_tolerance_curve"))

    # 8 — the colour key
    fig, ax = plt.subplots(figsize=(9.0, 0.40))
    ax.axis("off")
    KEY = ((0.000, "ground truth — the real transition line",
            J_TRUTH, J_TRUTH),
           (0.360, "model output — after the P > 0.4 cut",
            J_PRED, J_PRED),
           (0.700, "τ band — within τ pixels of the truth",
            "#D9D9D9", INK))
    for x, txt, swatch, tcol in KEY:
        ax.add_patch(plt.Rectangle((x, 0.34), 0.013, 0.34, facecolor=swatch,
                                   edgecolor="#888888", linewidth=0.6,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.020, 0.5, txt, transform=ax.transAxes, va="center",
                fontsize=9.0, color=tcol, fontweight="bold")
    save(fig, "p2l_8_legend")

    # 9 — the point of the whole slide, zoomed until pixels are visible:
    #     the band grows with tau, the black and the red never move
    r0, c0, S = 30, 26, 40
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 3.00))
    for ax, tau in zip(axes, (0, 1, 2, 3)):
        ax.imshow(overlay(pred, float(tau))[r0:r0 + S, c0:c0 + S],
                  origin="lower", interpolation="nearest")
        m = tolerant_f1(pred, Yt, float(tau))
        ax.set_title(f"τ = {tau}\nP {m['precision']:.2f}   "
                     f"R {m['recall']:.2f}   F1 {m['f1']:.2f}",
                     fontsize=11, color=J_PRED if tau == 1 else INK,
                     fontweight="bold" if tau == 1 else "normal", pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(J_PRED if tau == 1 else "#777777")
            sp.set_linewidth(1.8 if tau == 1 else 0.8)
    fig.suptitle(_p2l_title("p2l_9_tau_zoom",
                            "Same truth, same output — only the "
                            "tolerance band grows", INK, 14),
                 fontsize=14, color=INK, y=0.985)
    # the colour key rides inside this figure, so the slide does not need a
    # separate legend strip and the panels can be larger
    fig.tight_layout(rect=(0, 0.105, 1, 0.945))
    for x, txt, swatch, tcol in (
            (0.030, "ground truth — the real transition line",
             J_TRUTH, J_TRUTH),
            (0.375, "model output — after the P > 0.4 cut", J_PRED, J_PRED),
            (0.700, "τ band — within τ pixels of the truth", "#D9D9D9", INK)):
        fig.patches.append(plt.Rectangle(
            (x, 0.030), 0.0125, 0.042, facecolor=swatch,
            edgecolor="#888888", linewidth=0.6,
            transform=fig.transFigure, figure=fig))
        fig.text(x + 0.020, 0.051, txt, va="center", fontsize=10.5,
                 color=tcol, fontweight="bold")
    save(fig, _p2l_name("p2l_9_tau_zoom"))

    with open(os.path.join(HERE, "p2l_titles.json"), "w",
              encoding="utf-8") as fh:
        json.dump(P2L_TITLES, fh, ensure_ascii=False, indent=2)
    print("  titles %s -> p2l_titles.json"
          % ("blanked" if P2L_TITLES_OFF else "kept"))
    print(f"  threshold {thr:g}")
    print("  tau:        " + "  ".join(f"{t}" for t in T))
    print("  F1:         " + "  ".join(f"{v:.3f}" for v in f1s))
    print("  precision:  " + "  ".join(f"{v:.3f}" for v in prs))
    print("  recall:     " + "  ".join(f"{v:.3f}" for v in rcs))


# ── figure 9: the results chart for the slide ─────────────────────────────
# The gallery version labels all fifteen points and the labels collide.  Here
# the budgets are GROUPED BY RAY COUNT — one line each — so the reading
# "more rays beats more points at the same coverage" is the shape of the
# plot rather than something the caption has to assert.
def fig_results_f1_vs_coverage():
    import csv

    with open(os.path.join(RUN_DIR, "comparison.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_rays = {}
    for r in rows:
        by_rays.setdefault(int(r["n_rays"]), []).append(
            (100.0 * float(r["coverage"]), float(r["f1@1"]),
             int(r["n_points"])))
    for v in by_rays.values():
        v.sort()

    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    # Ray count is ORDERED, so the series run light-blue -> blue -> navy ->
    # black -> red: still a ramp, but each step changes hue as well as
    # lightness, and each carries its own marker.  Four greys were
    # indistinguishable on a projector, and marker shape survives greyscale
    # printing and the common colour-vision deficiencies.
    STYLE = {4: ("#9ECAE1", "o"), 5: ("#4292C6", "s"), 6: ("#08519C", "^"),
             7: ("#000000", "D"), 8: (J_PRED, "o")}
    for n in sorted(by_rays):
        x = [c for c, _f, _p in by_rays[n]]
        y = [f for _c, f, _p in by_rays[n]]
        colour, marker = STYLE[n]
        best = n == max(by_rays)
        ax.plot(x, y, marker=marker, linestyle="-", color=colour,
                linewidth=2.2 if best else 1.5,
                markersize=6.5 if best else 5.0,
                markeredgecolor="white" if best else colour,
                markeredgewidth=0.8 if best else 0.6,
                zorder=4 if best else 2, label=f"{n} rays")

    # the honest comparison: same coverage, more rays
    ax.annotate("", xy=(2.33, 0.752), xytext=(2.35, 0.691),
                arrowprops=dict(arrowstyle="-|>", color=J_TRUTH,
                                linewidth=1.2, shrinkA=3, shrinkB=3))
    ax.text(2.45, 0.719, "same coverage,\nmore rays", fontsize=8.5,
            color=J_TRUTH, va="center")

    ax.annotate("8 × 60\nF1 0.849 at 4.7 %", xy=(4.66, 0.849),
                xytext=(3.95, 0.885), fontsize=9, color=J_PRED,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=J_PRED, linewidth=0.9))

    ax.set_xlabel("fraction of the grid measured  (%)", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("F1 @ 1 px tolerance", fontsize=10.5, color=INK)
    ax.set_xlim(1.2, 5.1)
    ax.set_ylim(0.63, 0.91)
    ax.tick_params(labelsize=9, colors=INK)
    ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
            alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.92,
              edgecolor="#cccccc", loc="lower right", handlelength=1.9,
              labelspacing=0.35, borderpad=0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#444444"); ax.spines[sp].set_linewidth(0.8)
    # the title is a PowerPoint text box — see make_slide_pictures.py
    fig.tight_layout()
    save(fig, "results_f1_vs_coverage")


# ── figure 10: where the validation devices come from ─────────────────────
# The dataset table says 500 / 50 and the validation split is invisible in
# it, which is the one thing a reviewer will ask about: the threshold is a
# fitted quantity, so WHERE it was fitted has to be on the slide.
#   550 devices -> 500 training pool + 50 test          (split_seed 12345)
#   500 pool    -> 425 fit weights + 75 validation      (default_rng(0))
SPLIT_LABEL = []
SPLIT_LABEL_OFF = os.environ.get("SPLIT_LABEL") == "off"


def fig_data_split():
    from dqd.ml import grid_train

    N_TOTAL, N_TEST = 550, 50
    n_pool = N_TOTAL - N_TEST
    n_val = max(1, int(grid_train.VAL_FRACTION * n_pool))
    n_fit = n_pool - n_val

    POOL, TEST = "#c8c8c8", "#1a1a1a"
    FIT, VAL = "#e2e2e2", J_PRED

    # Only the bars are drawn here.  The heading, the colour key
    # and the footnote are PowerPoint text boxes placed by
    # make_measurement_slide.py, so they can be edited or deleted
    # on the slide without regenerating a figure.
    fig, ax = plt.subplots(figsize=(7.4, 1.30))
    ax.set_xlim(0, N_TOTAL)
    ax.set_ylim(0.22, 0.88)
    ax.axis("off")

    def bar(x0, w, y, h, fc):
        ax.add_patch(plt.Rectangle((x0, y), w, h, facecolor=fc,
                                   edgecolor="#444444", linewidth=0.9))

    # ── level 1: the split that is stored with the device pool ───────────
    bar(0, n_pool, 0.63, 0.22, POOL)
    bar(n_pool, N_TEST, 0.63, 0.22, TEST)
    ax.text(n_pool / 2, 0.74, f"{n_pool} training devices", ha="center",
            va="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(n_pool + N_TEST / 2, 0.74, f"{N_TEST}" + chr(10) + "test", ha="center",
            va="center", fontsize=9, color="white", fontweight="bold",
            linespacing=1.2)

    # ── the carve-out, before any training ───────────────────────────────
    for x in (0, n_pool):
        ax.plot([x, x], [0.63, 0.52], color="#999999", linewidth=0.8,
                linestyle=(0, (2, 2)))
    ax.annotate("", xy=(n_pool / 2, 0.50), xytext=(n_pool / 2, 0.61),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                linewidth=1.1))
    # The label is NOT drawn when the deck will carry it as a text box; its
    # position is recorded in figure fractions instead, so the box can be
    # placed exactly where the annotation was.
    SPLIT_LABEL[:] = [n_pool / 2 + 10, 0.555]

    # ── level 2: inside the training devices ─────────────────────────────
    bar(0, n_fit, 0.26, 0.22, FIT)
    bar(n_fit, n_val, 0.26, 0.22, VAL)
    ax.text(n_fit / 2, 0.37, f"{n_fit}  fit the weights", ha="center",
            va="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(n_fit + n_val / 2, 0.37, f"{n_val}" + chr(10) + "validation", ha="center",
            va="center", fontsize=9, color="white", fontweight="bold",
            linespacing=1.2)

    fig.tight_layout(pad=0.2)
    if SPLIT_LABEL_OFF:
        fig.canvas.draw()
        px = ax.transData.transform(tuple(SPLIT_LABEL))
        fx, fy = fig.transFigure.inverted().transform(px)
        with open(os.path.join(HERE, "data_split_label.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"x": float(fx), "y": float(fy)}, fh, indent=2)
        print("    label slot at x=%.3f y=%.3f of the figure" % (fx, fy))
        save(fig, "fig_data_split_notext")
    else:
        ax.text(SPLIT_LABEL[0], SPLIT_LABEL[1],
                "15 % carved out, before any training",
                fontsize=8.8, color=MUT, va="center")
        save(fig, "fig_data_split")


# ── figure 11: every metric against tolerance, three budgets ──────────────
# The gallery version (figures/40_tau_all_metrics.png) draws all fifteen
# budgets and four metrics: sixteen indistinguishable lines per panel.  This
# keeps the smallest budget, a middle one and the best, and drops the pixel
# accuracy panel — accuracy is high whatever the model does, which is a point
# to make out loud, not a panel to squint at.
def fig_tau_metrics(budgets=((4, 40), (6, 50), (8, 60))):
    import csv

    with open(os.path.join(RUN_DIR, "comparison.csv"), newline="") as fh:
        rows = {(int(r["n_rays"]), int(r["n_points"])): r
                for r in csv.DictReader(fh)}

    STYLE = {4: ("#9ECAE1", "o"), 5: ("#4292C6", "s"), 6: ("#08519C", "^"),
             7: ("#000000", "D"), 8: (J_PRED, "o")}
    T = (0, 1, 2, 3)
    METRICS = (("f1", "F1"), ("precision", "precision"), ("recall", "recall"))

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.15), sharey=True)
    for ax, (key, label) in zip(axes, METRICS):
        for n_rays, n_pts in budgets:
            r = rows[(n_rays, n_pts)]
            y = [float(r[f"{key}@{t}"]) for t in T]
            colour, marker = STYLE[n_rays]
            best = (n_rays, n_pts) == max(budgets)
            ax.plot(T, y, marker=marker, linestyle="-", color=colour,
                    linewidth=2.0 if best else 1.4,
                    markersize=5.5 if best else 4.5,
                    markeredgecolor="white" if best else colour,
                    markeredgewidth=0.8 if best else 0.6,
                    zorder=4 if best else 2,
                    label=f"{n_rays} × {n_pts}   "
                          f"({100 * float(r['coverage']):.1f} %)")
        ax.set_title(label, fontsize=10.5, color=INK, pad=5, loc="left")
        ax.set_xlabel("tolerance τ  (pixels)", fontsize=9.5, color=INK)
        ax.set_xticks(list(T))
        ax.set_ylim(0.25, 1.02)
        ax.tick_params(labelsize=8.5, colors=INK, labelleft=True)
        ax.set_ylabel("score on 50" + chr(10) + "held-out devices",
                      fontsize=8.5, color=INK, linespacing=1.3)
        ax.grid(True, color=J_GRID, linewidth=0.5, linestyle=(0, (1, 3)),
                alpha=0.85)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#444444")
            ax.spines[sp].set_linewidth(0.8)

    axes[0].legend(fontsize=8, frameon=True, framealpha=0.92,
                   edgecolor="#cccccc", loc="lower right", handlelength=1.8,
                   labelspacing=0.3, borderpad=0.3,
                   title="rays × points  (coverage)", title_fontsize=8)
    fig.tight_layout(pad=0.6, w_pad=1.4)
    save(fig, "fig_tau_metrics")
