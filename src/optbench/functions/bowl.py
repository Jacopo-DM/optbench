"""Bowl-shaped functions: one minimum, no local minima.

https://www.sfu.ca/~ssurjano/optimization.html, group "Bowl-Shaped".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import AnyDimension, Group, Minimum, box

from .references import DIXON_SZEGO, MOLGA, PICHENY

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class Sphere(AnyDimension):
    """Sphere function, ``f(x) = sum_i x_i^2``.

    Continuous, convex and unimodal. It has no constants.
    """

    name: ClassVar[str] = "sphere"
    group: ClassVar[Group] = Group.BOWL
    url: ClassVar[str] = "https://www.sfu.ca/~ssurjano/spheref.html"
    references: ClassVar[tuple[str, ...]] = (DIXON_SZEGO, MOLGA, PICHENY)

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.12, 5.12, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin, for every ``d``, on any domain holding it."""
        bounds = self.bounds
        if np.any(bounds[:, 0] > 0) or np.any(bounds[:, 1] < 0):
            return None
        return Minimum(0.0, np.zeros((1, self.d)), "formula", tolerance=0.0)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.einsum("ij,ij->i", x, x)
