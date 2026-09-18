"""
Shared boilerplate for the run_* programs — import path, headless plotting,
and the "resolve a list of configuration names, do a thing to each one, and
report" loop that steps 2 and 3 both need.

Nothing here is a setting.  The settings live at the top of each run_* file.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib
matplotlib.use("Agg")            # no display: these can run unattended

from dqd.study.config import resolve_configs   # noqa: E402  (needs the path)
from dqd.config import log

RULE = "=" * 74


def banner(text):
    log.say(RULE, text, RULE, sep="\n")


def configs(names):
    """The StudyConfigs for `names`, or exit with a useful message.

    A name that does not exist is an error listing the ones that do, rather
    than being silently skipped: a comparison quietly missing a
    configuration is a comparison that says something untrue.
    """
    try:
        found = resolve_configs(names)
    except KeyError as exc:
        sys.exit(str(exc).strip('"'))
    if not found:
        sys.exit("no configuration folders found — run "
                 "scripts/run_1_generate_dataset.py first")
    log.say(f"{len(found)} configuration(s):")
    for c in found:
        log.say(f"  {c.name}")
    return found


def for_each(cfgs, action):
    """Run `action(cfg)` on every configuration, in order.

    One configuration failing must not throw away the ones already done —
    they are on disk and still worth having — so failures are collected and
    reported at the end instead of stopping the run.

    Returns (done, failed): [(cfg, result), ...] and [(name, why), ...].
    """
    done, failed = [], []
    for n, cfg in enumerate(cfgs, 1):
        log.say(f"\n{RULE}\n[{n}/{len(cfgs)}] {cfg.name}\n{RULE}")
        log.detail(cfg.describe())
        try:
            done.append((cfg, action(cfg)))
        except Exception as exc:
            log.warn(f"[FAILED] {cfg.name}: {exc}")
            failed.append((cfg.name, str(exc)))
    return done, failed


def table(headers, rows, failed=()):
    """A fixed-width table: headers is [(title, width), ...]."""
    log.say("\n" + RULE)
    log.say("".join(f"{t:>{w}}" if i else f"{t:<{w}}"
                    for i, (t, w) in enumerate(headers)))
    for row in rows:
        log.say("".join(f"{v:>{w}}" if i else f"{v:<{w}}"
                        for i, (v, (_, w)) in enumerate(zip(row, headers))))
    for name, why in failed:
        log.say(f"{name:<{headers[0][1]}}FAILED: {why.splitlines()[0]}")
    log.say(RULE)
