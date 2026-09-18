"""
log.py — how much the programs say while they run.

Two levels, one switch:

    say(...)     the headline.  Always printed.  One line per stage, so a
                 four-cell sweep is a dozen lines you can actually read.
    detail(...)  the working — per-device counts, paths, full reports.
                 Printed only when VERBOSE is on.

Nothing is lost by staying quiet: every report printed at detail level is
ALSO written to a file (dataset_summary.txt, results.txt, comparison.txt),
which is where you read it afterwards anyway.  The terminal is for watching
progress, not for archiving evidence.

Turn the working back on for one run without editing anything:

    DQD_VERBOSE=1 python scripts/run_0_full_sweep.py

or from a program, before it starts:

    from dqd.config import log
    log.VERBOSE = True
"""
import os
import sys

VERBOSE = os.environ.get("DQD_VERBOSE", "").lower() in ("1", "true", "yes")


def say(*args, **kwargs):
    """A headline: always printed."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def detail(*args, **kwargs):
    """The working: printed only when VERBOSE."""
    if VERBOSE:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def warn(*args, **kwargs):
    """Something the reader must see even when quiet."""
    kwargs.setdefault("flush", True)
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)
