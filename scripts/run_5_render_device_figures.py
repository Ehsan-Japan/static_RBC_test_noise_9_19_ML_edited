"""
EXTRA — redraw the per-device pictures.  No data is regenerated, nothing is
retrained; this exists so you can change your mind about figures (more
devices, 300 dpi for the paper) cheaply.

    python scripts/run_5_render_device_figures.py

Files land in training_data/<configuration>/figures/<split>/sample_<i>/.
The available figures and what each one shows are printed when it runs.
"""
from _common import banner, configs as resolve
from dqd.study import device_figures
from dqd.config import log

# ══════════════════════════════════════════════════════════════════════════
#  SETTINGS  —  None means "use what run_1 recorded in config.json"
# ══════════════════════════════════════════════════════════════════════════

CONFIG_NAMES = ["3_rays_40_points_500_samples"]      # or "ALL"

# Overrides for this run only; nothing is written back to config.json.
FIGURES = None        # e.g. {"charge_sensor": True, "rays_on_truth": True}
DEVICES = None        # e.g. {"train": "NONE", "test": "ALL"}
DPI = None            # e.g. 300 for the paper

# ══════════════════════════════════════════════════════════════════════════


def main():
    banner("per-device figures")
    log.say("available figures:")
    for name, description in device_figures.FIGURE_KINDS:
        log.say(f"  {name:<24}{description}")
    log.say()

    total = 0
    for cfg in resolve(CONFIG_NAMES):
        log.say(cfg.name)
        cfg.save_device_figures = True
        if FIGURES is not None:
            cfg.device_figures = {k: bool(FIGURES.get(k, False))
                                  for k in device_figures.DEFAULT_DEVICE_FIGURES}
        if DEVICES is not None:
            cfg.figure_devices = dict(DEVICES)
        if DPI is not None:
            cfg.figure_dpi = DPI
        total += device_figures.render_config(cfg)

    banner(f"{total} figure(s) written")


if __name__ == "__main__":
    main()
