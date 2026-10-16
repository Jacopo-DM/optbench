"""Functions the SFU page groups as "Other".

https://www.sfu.ca/~ssurjano/optimization.html, group "Other".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import Group, Minimum, Problem, box

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MOLGA = (
    "Molga, M., & Smutnicki, C. (2005). Test functions for optimization needs."
)
_HEDAR = "Hedar, A. Global Optimization Test Problems (retrieved June 2013)."


@dataclass(frozen=True, kw_only=True)
class Branin(Problem):
    """Branin (Branin-Hoo) function, two variables, three global minima.

    ``f(x) = a (x2 - b x1^2 + c x1 - r)^2 + s (1 - t) cos(x1) + s``

    Attributes
    ----------
    a, b, c, r, s, t
        The constants, defaulting to the recommended ``a = 1``,
        ``b = 5.1 / (4 pi^2)``, ``c = 5 / pi``, ``r = 6``, ``s = 10`` and
        ``t = 1 / (8 pi)``.
    """

    name: ClassVar[str] = "branin"
    group: ClassVar[Group] = Group.OTHER
    url: ClassVar[str] = "https://www.sfu.ca/~ssurjano/branin.html"
    references: ClassVar[tuple[str, ...]] = (
        (
            "Dixon, L. C. W., & Szego, G. P. (1978). The global optimization "
            "problem: an introduction. Towards global optimization, 2, 1-15."
        ),
        (
            "Forrester, A., Sobester, A., & Keane, A. (2008). Engineering "
            "design via surrogate modelling: a practical guide. Wiley."
        ),
        _HEDAR,
        _MOLGA,
        (
            "Picheny, V., Wagner, T., & Ginsbourger, D. (2012). A benchmark of "
            "kriging-based infill criteria for noisy optimization."
        ),
    )

    a: float = 1.0
    b: float = 5.1 / (4 * math.pi**2)
    c: float = 5 / math.pi
    r: float = 6.0
    s: float = 10.0
    t: float = 1 / (8 * math.pi)

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return np.array([[-5.0, 10.0], [0.0, 15.0]])

    @property
    def minimum(self) -> Minimum | None:
        """``s t``, where ``cos(x1) = -1`` and the square is zero.

        Holds for ``a >= 0``, ``s > 0`` and ``0 <= t <= 1``: the square is
        then at least zero and ``s (1 - t) cos(x1)`` at least ``-s (1 - t)``.
        The minimisers are ``x1`` an odd multiple of pi and
        ``x2 = b x1^2 - c x1 + r``, those inside the domain (three for the
        defaults).
        """
        if self.a < 0 or self.s <= 0 or not 0 <= self.t <= 1:
            return None
        low, high = self.bounds[0]
        first = math.ceil((low / math.pi - 1) / 2)
        last = math.floor((high / math.pi - 1) / 2)
        x1 = np.array([(2 * k + 1) * math.pi for k in range(first, last + 1)])
        x2 = self.b * x1**2 - self.c * x1 + self.r
        y_low, y_high = self.bounds[1]
        inside = (x2 >= y_low) & (x2 <= y_high)
        if not np.any(inside):
            return None
        points = np.column_stack([x1[inside], x2[inside]])
        return Minimum(self.s * self.t, points, "formula")

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        square = (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2
        return self.a * square + self.s * (1 - self.t) * np.cos(x1) + self.s


_SHEKEL_BETA = (0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5)
_SHEKEL_C = (
    (4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
    (4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
    (4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
    (4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
)

# Minimisers of the published Shekel functions (m = 5, 7, 10), found by
# Newton steps from the published (4, 4, 4, 4) until the gradient was below
# 1e-13; the values agree with the published ones to their four decimals.
_SHEKEL_MINIMA = {
    5: (
        -10.153199679058227,
        (4.000037152819676, 4.00013327659156) * 2,
    ),
    7: (
        -10.402915336777744,
        (4.000572819251117, 3.9996062096096887) * 2,
    ),
    10: (
        -10.536443153483528,
        (4.000746868270634, 3.9995094800857736) * 2,
    ),
}


@dataclass(frozen=True, kw_only=True)
class Shekel(Problem):
    """Shekel function: ``m`` wells, the deepest near (4, 4, 4, 4).

    ``f(x) = -sum_{i<=m} 1 / (sum_j (x_j - C_ji)^2 + beta_i)``

    Attributes
    ----------
    m
        Number of wells used: the first ``m`` entries of ``beta`` and columns
        of ``c``. The literature uses 5, 7 or 10.
    beta
        Well widths, at least ``m`` of them.
    c
        Well centres, one row per variable and at least ``m`` columns. The
        number of rows is the dimension (4 for the published function).
    """

    name: ClassVar[str] = "shekel"
    group: ClassVar[Group] = Group.OTHER
    url: ClassVar[str] = "https://www.sfu.ca/~ssurjano/shekel.html"
    references: ClassVar[tuple[str, ...]] = (_HEDAR, _MOLGA)

    m: int = 10
    beta: tuple[float, ...] = _SHEKEL_BETA
    c: tuple[tuple[float, ...], ...] = _SHEKEL_C

    def _check(self) -> None:
        centres = np.asarray(self.c, dtype=np.float64)
        if centres.ndim != 2:  # ruff: ignore[magic-value-comparison]
            msg = f"c must be a matrix, got shape {centres.shape}"
            raise ValueError(msg)
        if not 1 <= self.m <= min(len(self.beta), centres.shape[1]):
            msg = (
                f"m must be between 1 and the number of wells "
                f"({min(len(self.beta), centres.shape[1])}), got {self.m}"
            )
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        return len(self.c)

    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, 10.0, self.dimension)

    @property
    def minimum(self) -> Minimum | None:
        """Known for the published wells with m = 5, 7 or 10."""
        published = self.beta == _SHEKEL_BETA and self.c == _SHEKEL_C
        if not (
            published and self.m in _SHEKEL_MINIMA and self.uses_default_domain
        ):
            return None
        value, point = _SHEKEL_MINIMA[self.m]
        return Minimum(
            value,
            np.array([point]),
            "Newton refinement of the published (4, 4, 4, 4)",
            tolerance=1e-12,
        )

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        beta = np.asarray(self.beta[: self.m], dtype=np.float64)
        centres = np.asarray(self.c, dtype=np.float64)[:, : self.m]
        diff = x[:, :, np.newaxis] - centres
        squares = np.einsum("ijk,ijk->ik", diff, diff)
        wells: NDArray[np.float64] = 1 / (squares + beta)
        return -wells.sum(axis=1)
