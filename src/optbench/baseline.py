"""Random search: the baseline any optimiser should beat."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .tracking import Tracked

if TYPE_CHECKING:
    import numpy as np

    from .problem import Problem


def random_search(
    problem: Problem,
    budget: int,
    rng: np.random.Generator,
    batch: int = 1_000,
) -> Tracked:
    """Evaluate uniform random points in the domain and keep the best.

    Parameters
    ----------
    problem
        The problem to minimise, over its ``bounds``.
    budget
        Number of points to evaluate, exactly.
    rng
        Source of the points.
    batch
        Points per call; the last call takes what is left of the budget.

    Returns
    -------
    Tracked
        The record of the run: ``best_value``, ``best_point`` and the
        best-so-far ``curve``, one row per batch.

    Raises
    ------
    ValueError
        If ``budget`` or ``batch`` is below 1.
    """
    if budget < 1:
        msg = f"budget must be at least 1, got {budget}"
        raise ValueError(msg)
    if batch < 1:
        msg = f"batch must be at least 1, got {batch}"
        raise ValueError(msg)
    tracked = Tracked(problem)
    bounds = problem.bounds
    while tracked.evaluations < budget:
        n = min(batch, budget - tracked.evaluations)
        size = (n, problem.dimension)
        tracked(rng.uniform(bounds[:, 0], bounds[:, 1], size=size))
    return tracked
