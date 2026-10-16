"""Functions the SFU page groups as "Other", Beale to Goldstein-Price.

https://www.sfu.ca/~ssurjano/optimization.html, group "Other". The rest of
the group is in ``other_hartmann`` and ``other_perm_to_styblinski_tang``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import Group, Minimum, Problem, box, formula_minimum

from .known_minima import FORRESTER_2008, GOLDSTEIN_PRICE, at_defaults
from .references import DIXON_SZEGO, HEDAR, MOLGA, PICHENY

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.OTHER
_SFU = "https://www.sfu.ca/~ssurjano/"
_FORRESTER = (
    "Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design "
    "via surrogate modelling: a practical guide. Wiley."
)


@dataclass(frozen=True, kw_only=True)
class Beale(Problem):
    """Beale function: sharp peaks at the corners of the domain.

    ``f(x) = sum_k (c_k - x1 + x1 x2^k)^2`` for ``k = 1, 2, 3``

    Attributes
    ----------
    c
        The published (1.5, 2.25, 2.625).
    """

    name: ClassVar[str] = "beale"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "beale.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    c: tuple[float, float, float] = (1.5, 2.25, 2.625)

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-4.5, 4.5, 2)

    @property
    def minimum(self) -> Minimum | None:
        """Zero where all three brackets vanish, if they can.

        The first two give ``x2 = c_2 / c_1 - 1`` and ``x1 = c_1 / (1 - x2)``
        ((3, 0.5) for the defaults); the third must then hold too.
        """
        c1, c2, c3 = self.c
        if c1 == 0:
            return None
        x2 = c2 / c1 - 1
        if x2 == 1:
            return None
        x1 = c1 / (1 - x2)
        if not math.isclose(c3, x1 * (1 - x2**3), rel_tol=1e-12, abs_tol=1e-12):
            return None
        return formula_minimum(self, 0.0, np.array([[x1, x2]]))

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        total = np.zeros(len(x))
        for k, c in enumerate(self.c, start=1):
            total += (c - x1 + x1 * x2**k) ** 2
        return total


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
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "branin.html"
    references: ClassVar[tuple[str, ...]] = (
        DIXON_SZEGO,
        (
            "Forrester, A., Sobester, A., & Keane, A. (2008). Engineering "
            "design via surrogate modelling: a practical guide. Wiley."
        ),
        HEDAR,
        MOLGA,
        PICHENY,
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


@dataclass(frozen=True, kw_only=True)
class Colville(Problem):
    """Colville function, four variables.

    ``f(x) = a (x1^2 - x2)^2 + (x1 - 1)^2 + (x3 - 1)^2 + b (x3^2 - x4)^2
    + c ((x2 - 1)^2 + (x4 - 1)^2) + e (x2 - 1) (x4 - 1)``

    Attributes
    ----------
    a, b, c, e
        The published 100, 90, 10.1 and 19.8.
    """

    name: ClassVar[str] = "colville"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "colville.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    a: float = 100.0
    b: float = 90.0
    c: float = 10.1
    e: float = 19.8

    @property
    def dimension(self) -> int:
        return 4

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, 4)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at (1, 1, 1, 1) when no term can be negative.

        That needs ``a, b >= 0`` and ``c >= |e| / 2``, so that the last two
        terms form a non-negative quadratic in ``x2 - 1`` and ``x4 - 1``.
        """
        if min(self.a, self.b) < 0 or self.c < abs(self.e) / 2:
            return None
        return formula_minimum(self, 0.0, np.ones((1, 4)))

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        u, v = x2 - 1, x4 - 1
        return (
            self.a * (x1**2 - x2) ** 2
            + (x1 - 1) ** 2
            + (x3 - 1) ** 2
            + self.b * (x3**2 - x4) ** 2
            + self.c * (u**2 + v**2)
            + self.e * u * v
        )


@dataclass(frozen=True, kw_only=True)
class Forrester2008(Problem):
    """Forrester et al. (2008) function, one variable.

    ``f(x) = (a x - b)^2 sin(c x - e)``

    Attributes
    ----------
    a, b, c, e
        The published 6, 2, 12 and 4.
    """

    name: ClassVar[str] = "forrester_2008"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "forretal08.html"
    references: ClassVar[tuple[str, ...]] = (_FORRESTER,)

    a: float = 6.0
    b: float = 2.0
    c: float = 12.0
    e: float = 4.0

    @property
    def dimension(self) -> int:
        return 1

    def default_bounds(self) -> NDArray[np.float64]:
        return np.array([[0.0, 1.0]])

    @property
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, FORRESTER_2008)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        t = x[:, 0]
        return (self.a * t - self.b) ** 2 * np.sin(self.c * t - self.e)


@dataclass(frozen=True, kw_only=True)
class GoldsteinPrice(Problem):
    """Goldstein-Price function: several local minima.

    ``f(x) = (a + (x1 + x2 + 1)^2 P(x)) (b + (2 x1 - 3 x2)^2 Q(x))``, where
    ``P`` and ``Q`` are quadratics with coefficients ``p`` and ``q`` for the
    terms ``(1, x1, x1^2, x2, x1 x2, x2^2)``.

    Attributes
    ----------
    a, b
        The published 1 and 30.
    p, q
        The published (19, -14, 3, -14, 6, 3) and (18, -32, 12, 48, -36, 27).
    """

    name: ClassVar[str] = "goldstein_price"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "goldpr.html"
    references: ClassVar[tuple[str, ...]] = (DIXON_SZEGO, MOLGA, PICHENY)

    a: float = 1.0
    b: float = 30.0
    p: tuple[float, ...] = (19.0, -14.0, 3.0, -14.0, 6.0, 3.0)
    q: tuple[float, ...] = (18.0, -32.0, 12.0, 48.0, -36.0, 27.0)

    def _check(self) -> None:
        if len(self.p) != 6 or len(self.q) != 6:  # ruff: ignore[magic-value-comparison]
            msg = "p and q need six coefficients each"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-2.0, 2.0, 2)

    @property
    def minimum(self) -> Minimum | None:
        """3 at (0, -1), for the published constants."""
        return at_defaults(self, GOLDSTEIN_PRICE)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        terms = np.stack([np.ones(len(x)), x1, x1**2, x2, x1 * x2, x2**2])
        first = self.a + (x1 + x2 + 1) ** 2 * (np.asarray(self.p) @ terms)
        second = self.b + (2 * x1 - 3 * x2) ** 2 * (np.asarray(self.q) @ terms)
        return first * second
