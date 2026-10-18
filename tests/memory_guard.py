"""Stop a Python process before it takes the machine's memory.

A sympy formula once grew to 160 GB while being turned into numpy code,
and macOS only killed it after swapping the whole machine. ``start()``
runs a watchdog thread that ends the process, with every thread's
traceback, once its peak memory passes ``LIMIT``. ``conftest.py`` starts
it for every pytest run; a sympy experiment outside pytest starts it by
hand (see CLAUDE.md). macOS ignores ``resource.RLIMIT_AS``, so a hard
limit is not an option.
"""

from __future__ import annotations

import faulthandler
import os
import resource
import sys
import threading
import time

# The whole suite peaks at about 200 MB (2026-10-18).
LIMIT = 2 * 1024**3
INTERVAL = 0.2
EXIT_STATUS = 3


def peak_memory() -> int:
    """Return this process's peak resident memory in bytes.

    Returns
    -------
    int
        ``ru_maxrss``, which macOS gives in bytes and Linux in kilobytes.
    """
    unit = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit


def stop() -> None:
    """End the process at once, saying why and where it was."""
    sys.stderr.write(
        f"\nmemory_guard: peak memory {peak_memory() / 1024**3:.1f} GB passed "
        f"the {LIMIT / 1024**3:.1f} GB limit; stopping. Tracebacks:\n",
    )
    faulthandler.dump_traceback(all_threads=True)
    os._exit(EXIT_STATUS)


def watch(limit: int) -> None:
    """Wait until the peak passes ``limit`` bytes, then ``stop``.

    The limit is an argument, not read from ``LIMIT`` on each check, so a
    test that calls ``watch`` with a small limit cannot trip the guard
    already running in the same process.
    """
    while peak_memory() <= limit:
        time.sleep(INTERVAL)
    stop()


def start() -> None:
    """Run ``watch`` with ``LIMIT`` in a daemon thread."""
    threading.Thread(
        target=watch,
        args=(LIMIT,),
        name="memory_guard",
        daemon=True,
    ).start()
