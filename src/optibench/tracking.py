"""Counting evaluations while an optimiser runs."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .model import Model
from .problem import Problem


class Tracked(Model):
    """Wraps a problem to count evaluations and keep the best point so far.

    Call it like the problem. After every call, ``curve`` gains one row
    ``(evaluations so far, best value so far)``, so an optimiser's progress
    can be plotted or saved (``numpy.savetxt``) afterwards. NaN values never
    count as best.

    A mutable model, the standard's exception for a record that must
    change: it counts as it goes.

    Parameters
    ----------
    problem
        The problem to evaluate.
    """

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    problem: Problem
    evaluations: int = 0
    best_value: float = np.inf
    best_point: NDArray[np.float64] | None = None
    _curve: list[tuple[float, float]] = PrivateAttr(
        default_factory=list[tuple[float, float]],
    )

    def __init__(self, problem: Problem) -> None:
        # BaseModel's own __init__: the one type checkers synthesise for
        # Model has no fields, so it would refuse ``problem``.
        BaseModel.__init__(self, problem=problem)

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
