"""Run the random-search baseline on every problem.

Run with ``uv run python -m optbench.scripts.baseline``. Each problem (free
dimensions at 10, Powell at 8) gets 10 seeds of 10,000 uniform random
points. Prints, per problem, the known minimum and the median, best and
worst final value over the seeds, and writes every best-so-far curve to
``__data__/baseline/<name>_seed<n>.csv`` (columns: evaluations, best).
"""

from __future__ import annotations

import argparse
import statistics
from typing import TYPE_CHECKING

import numpy as np

from optbench import random_search
from optbench.scripts.common import DATA, every_problem

if TYPE_CHECKING:
    from collections.abc import Sequence

BUDGET = 10_000
SEEDS = 10


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    args = parser.parse_args(argv)
    folder = DATA / "baseline"
    folder.mkdir(parents=True, exist_ok=True)

    # One row per problem: the known minimum and the seeds' final values.
    columns = ("minimum", "median", "best", "worst")
    header = "".join(f"{column:>13}" for column in columns)
    print(f"{'problem':26} {'d':>3}{header}")  # ruff: ignore[print]
    for problem in every_problem():
        finals: list[float] = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            tracked = random_search(problem, args.budget, rng)
            finals.append(tracked.best_value)
            out = folder / f"{problem.name}_seed{seed}.csv"
            np.savetxt(
                out,
                tracked.curve,
                delimiter=",",
                header="evaluations,best",
                comments="",
            )
        minimum = problem.minimum
        known = (
            f"{minimum.value:12.6g}" if minimum is not None else f"{'?':>12}"
        )
        print(  # ruff: ignore[print]
            f"{problem.name:26} {problem.dimension:3} {known} "
            f"{statistics.median(finals):12.6g} {min(finals):12.6g} "
            f"{max(finals):12.6g}",
        )
    print(f"curves saved in {folder}")  # ruff: ignore[print]


if __name__ == "__main__":  # pragma: no cover
    main()
