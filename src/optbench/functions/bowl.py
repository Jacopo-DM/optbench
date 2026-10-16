"""Bowl-shaped functions: one minimum, no local minima.

https://www.sfu.ca/~ssurjano/optimization.html, group "Bowl-Shaped".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import Group, Minimum, Problem, box

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class Sphere(Problem):
    """Sphere function, ``f(x) = sum_i x_i^2``.

    Continuous, convex and unimodal. It has no constants.

    Attributes
    ----------
    d
        Number of variables.
    """

    name: ClassVar[str] = "sphere"
    group: ClassVar[Group] = Group.BOWL
    url: ClassVar[str] = "https://www.sfu.ca/~ssurjano/spheref.html"
    references: ClassVar[tuple[str, ...]] = (
        (
            "Dixon, L. C. W., & Szego, G. P. (1978). The global optimization "
            "problem: an introduction. Towards global optimization, 2, 1-15."
        ),
        (
            "Molga, M., & Smutnicki, C. (2005). Test functions for "
            "optimization needs."
        ),
    )

    d: int = 2

    def _check(self) -> None:
        if self.d < 1:
            msg = f"d must be at least 1, got {self.d}"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        return self.d

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
