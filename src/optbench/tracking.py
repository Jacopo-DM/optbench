"""Counting evaluations while an optimiser runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .problem import Problem


class Tracked:
    """Wraps a problem to count evaluations and keep the best point so far.

    Call it like the problem. After every call, ``curve`` gains one row
    ``(evaluations so far, best value so far)``, so an optimiser's progress
    can be plotted or saved (``numpy.savetxt``) afterwards. NaN values never
    count as best.

    Parameters
    ----------
    problem
        The problem to evaluate.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.evaluations = 0
        self.best_value = np.inf
        self.best_point: NDArray[np.float64] | None = None
        self._curve: list[tuple[float, float]] = []

    def __call__(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate a batch and record it.

        Returns
        -------
        numpy.ndarray
            The problem's values, shaped ``(n,)``.
        """
        points = np.asarray(x, dtype=np.float64)
        values = self.problem(points)
        self.evaluations += len(values)
        if np.any(~np.isnan(values)):
            best = int(np.nanargmin(values))
            if values[best] < self.best_value:
                self.best_value = float(values[best])
                self.best_point = points[best].copy()
        self._curve.append((self.evaluations, self.best_value))
        return values

    @property
    def curve(self) -> NDArray[np.float64]:
        """Rows of (evaluations, best value so far), one per call."""
        return np.array(self._curve, dtype=np.float64).reshape(-1, 2)
