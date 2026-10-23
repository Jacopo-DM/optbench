"""Time every problem on a batch of points (spec criterion 2).

Run with ``uv run python -m optibench.scripts.timing``. Each problem
evaluates 100,000 uniform random points in its domain (free dimensions at
10, Powell at 8); the time is CPU seconds, the minimum of 5 repeats.
Prints a table, writes ``__data__/timing.csv`` and exits with status 1 if
any problem is over the 0.1 s budget, except the stated exceptions in
``EXCEPTIONS``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from optibench.model import Model
from optibench.scripts.common import DATA, Integer, every_problem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optibench import Problem

POINTS = 100_000
REPEATS = 5
BUDGET = 0.1

# Problems allowed over the budget (the user, 2026-10-17): Whitley has d^2
# terms per point, and their cosines alone take most of the budget at d = 10.
EXCEPTIONS = frozenset({"whitley"})


class Arguments(Model):
    """The command-line arguments, parsed from their text."""

    points: Integer = POINTS
    repeats: Integer = REPEATS


class Timing(Model):
    """One problem's time, a row of the table and of ``timing.csv``."""

    name: str
    dimension: int
    seconds: float


def parse_arguments(argv: Sequence[str] | None = None) -> Arguments:
    """Read ``--points`` and ``--repeats`` from the command line.

    Returns
    -------
    Arguments
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument("--points", default=POINTS)
    parser.add_argument("--repeats", default=REPEATS)
    return Arguments.model_validate(vars(parser.parse_args(argv)))


def time_problem(problem: Problem, points: int, repeats: int) -> float:
    """Return the fastest of ``repeats`` evaluations of a batch, in CPU s.

    Returns
    -------
    float
        CPU seconds for one call on ``points`` points.
    """
    bounds = problem.bounds
    rng = np.random.default_rng(0)
    x = rng.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(points, problem.dimension),
    )
    times: list[float] = []
    for _ in range(repeats):
        start = time.process_time()
        problem(x)
        times.append(time.process_time() - start)
    return min(times)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Returns
    -------
    int
        0 if every problem is within the budget, else 1.
    """
    args = parse_arguments(argv)

    # Time every problem, slowest last in the table.
    rows = [
        Timing(
            name=p.name,
            dimension=p.dimension,
            seconds=time_problem(p, args.points, args.repeats),
        )
        for p in every_problem()
    ]
    rows.sort(key=lambda row: row.seconds)
    for row in rows:
        flag = ""
        if row.seconds > BUDGET:
            excused = row.name in EXCEPTIONS
            flag = "  over budget, stated exception" if excused else "  OVER"
        ms = 1000 * row.seconds
        print(f"{row.name:26} d={row.dimension:<3} {ms:7.1f} ms{flag}")  # ruff: ignore[print]

    # Save them for plotting.
    DATA.mkdir(exist_ok=True)
    out = DATA / "timing.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "d", "seconds"])
        writer.writerows([row.name, row.dimension, row.seconds] for row in rows)
    over = sum(
        row.seconds > BUDGET and row.name not in EXCEPTIONS for row in rows
    )
    print(f"{over} of {len(rows)} over {1000 * BUDGET:.0f} ms; saved {out}")  # ruff: ignore[print]
    return 1 if over else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
