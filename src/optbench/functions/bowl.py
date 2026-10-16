"""Bowl-shaped functions: one minimum, no local minima.

https://www.sfu.ca/~ssurjano/optimization.html, group "Bowl-Shaped".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import AnyDimension, Group, Minimum, Problem, box

from .references import ADORIO, DIXON_SZEGO, HEDAR, MOLGA, PICHENY

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.BOWL
_SFU = "https://www.sfu.ca/~ssurjano/"


def formula_minimum(
    problem: Problem,
    value: float,
    points: NDArray[np.float64],
) -> Minimum | None:
    """Return a minimum known from a formula, if it lies in the domain.

    Returns
    -------
    Minimum or None
        The minimum, or ``None`` when a minimiser is outside the domain.
    """
    if not problem.contains(points):
        return None
    return Minimum(value, points, "formula")


@dataclass(frozen=True, kw_only=True)
class Sphere(AnyDimension):
    """Sphere function, ``f(x) = sum_i x_i^2``.

    Continuous, convex and unimodal. It has no constants.
    """

    name: ClassVar[str] = "sphere"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "spheref.html"
    references: ClassVar[tuple[str, ...]] = (DIXON_SZEGO, MOLGA, PICHENY)

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.12, 5.12, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin, for every ``d``, on any domain holding it."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.einsum("ij,ij->i", x, x)


@dataclass(frozen=True, kw_only=True)
class _Bohachevsky(Problem):
    """What the three Bohachevsky functions share.

    Each is ``x1^2 + weight x2^2 - (cosine terms) + offset``, with cosines of
    ``p pi x1`` and ``q pi x2``.

    Attributes
    ----------
    weight, a, p, q, offset
        The published 2, 0.3, 3, 4 and the offset of each function.
    """

    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "boha.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    weight: float = 2.0
    a: float = 0.3
    p: float = 3.0
    q: float = 4.0
    offset: float = 0.3

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-100.0, 100.0, 2)

    def _cosine_weights(self) -> tuple[float, ...]:
        """Return the weights of the cosine terms.

        Returns
        -------
        tuple of float
            One weight per cosine term.
        """
        return (self.a,)

    @property
    def minimum(self) -> Minimum | None:
        """``offset`` minus the cosine weights, at the origin.

        Holds when ``weight`` and the cosine weights are not negative: the
        squares are then smallest at the origin, where every cosine is 1.
        """
        cosines = self._cosine_weights()
        if self.weight < 0 or min(cosines) < 0:
            return None
        value = self.offset - sum(cosines)
        return formula_minimum(self, value, np.zeros((1, 2)))

    def _squares(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x[:, 0] ** 2 + self.weight * x[:, 1] ** 2


@dataclass(frozen=True, kw_only=True)
class Bohachevsky1(_Bohachevsky):
    """Bohachevsky function 1.

    ``f(x) = x1^2 + weight x2^2 - a cos(p pi x1) - b cos(q pi x2) + offset``

    Attributes
    ----------
    b, offset
        The published 0.4 and 0.7.
    """

    name: ClassVar[str] = "bohachevsky1"

    b: float = 0.4
    offset: float = 0.7

    def _cosine_weights(self) -> tuple[float, ...]:
        return (self.a, self.b)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        waves = self.a * np.cos(self.p * math.pi * x1) + self.b * np.cos(
            self.q * math.pi * x2,
        )
        return self._squares(x) - waves + self.offset


@dataclass(frozen=True, kw_only=True)
class Bohachevsky2(_Bohachevsky):
    """Bohachevsky function 2.

    ``f(x) = x1^2 + weight x2^2 - a cos(p pi x1) cos(q pi x2) + offset``
    """

    name: ClassVar[str] = "bohachevsky2"

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        wave = np.cos(self.p * math.pi * x1) * np.cos(self.q * math.pi * x2)
        return self._squares(x) - self.a * wave + self.offset


@dataclass(frozen=True, kw_only=True)
class Bohachevsky3(_Bohachevsky):
    """Bohachevsky function 3.

    ``f(x) = x1^2 + weight x2^2 - a cos(p pi x1 + q pi x2) + offset``
    """

    name: ClassVar[str] = "bohachevsky3"

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        wave = np.cos(self.p * math.pi * x1 + self.q * math.pi * x2)
        return self._squares(x) - self.a * wave + self.offset


@dataclass(frozen=True, kw_only=True)
class Perm0(AnyDimension):
    """Perm function 0, d, beta.

    ``f(x) = sum_i (sum_j (j + beta) (x_j^i - 1 / j^i))^2``

    Attributes
    ----------
    beta
        10, the default of the SFU page's own code (the page gives none).
    """

    name: ClassVar[str] = "perm0"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "perm0db.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    beta: float = 10.0

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-self.d, self.d, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at ``x_j = 1 / j``, for every ``beta``: each inner sum is 0."""
        points = 1 / np.arange(1, self.d + 1, dtype=np.float64)[np.newaxis]
        return formula_minimum(self, 0.0, points)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        d = self.d
        j = np.arange(1, d + 1, dtype=np.float64)

        # powers[n, i, j] = x_j^(i+1), built by repeated multiplication, which
        # is several times faster than calling power for every entry.
        repeated = np.broadcast_to(x[:, np.newaxis, :], (len(x), d, d))
        powers = np.cumprod(repeated, axis=1)
        targets = np.cumprod(np.broadcast_to(1 / j, (d, d)), axis=0)
        inner: NDArray[np.float64] = np.einsum(
            "nij,j->ni",
            powers - targets,
            j + self.beta,
        )
        return np.sum(inner**2, axis=1)


@dataclass(frozen=True, kw_only=True)
class RotatedHyperEllipsoid(AnyDimension):
    """Rotated hyper-ellipsoid function, ``f(x) = sum_i sum_{j<=i} x_j^2``.

    Continuous, convex and unimodal. It has no constants.
    """

    name: ClassVar[str] = "rotated_hyper_ellipsoid"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "rothyp.html"
    references: ClassVar[tuple[str, ...]] = (MOLGA,)

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-65.536, 65.536, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # x_j^2 appears in the sums for i = j, ..., d: d - j + 1 times.
        counts = np.arange(self.d, 0, -1, dtype=np.float64)
        return np.sum(counts * x**2, axis=1)


@dataclass(frozen=True, kw_only=True)
class SumOfDifferentPowers(AnyDimension):
    """Sum of different powers, ``f(x) = sum_i |x_i|^(i + power_shift)``.

    Attributes
    ----------
    power_shift
        The published 1.
    """

    name: ClassVar[str] = "sum_of_different_powers"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "sumpow.html"
    references: ClassVar[tuple[str, ...]] = (MOLGA,)

    power_shift: float = 1.0

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-1.0, 1.0, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin while every exponent is positive."""
        if 1 + self.power_shift <= 0:
            return None
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        powers = np.arange(1, self.d + 1, dtype=np.float64) + self.power_shift
        return np.sum(np.abs(x) ** powers, axis=1)


@dataclass(frozen=True, kw_only=True)
class SumSquares(AnyDimension):
    """Sum squares (axis-parallel hyper-ellipsoid), ``sum_i i x_i^2``.

    Continuous, convex and unimodal. It has no constants.
    """

    name: ClassVar[str] = "sum_squares"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "sumsqu.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR, MOLGA)

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        weights = np.arange(1, self.d + 1, dtype=np.float64)
        return np.sum(weights * x**2, axis=1)


@dataclass(frozen=True, kw_only=True)
class Trid(AnyDimension):
    """Trid function, ``f(x) = sum_i (x_i - shift)^2 - sum_i x_i x_{i-1}``.

    Attributes
    ----------
    shift
        The published 1.
    """

    name: ClassVar[str] = "trid"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "trid.html"
    references: ClassVar[tuple[str, ...]] = (ADORIO, HEDAR)

    shift: float = 1.0

    def default_bounds(self) -> NDArray[np.float64]:
        square = float(self.d**2)
        return box(-square, square, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """``-shift^2 d (d + 4) (d - 1) / 6`` at ``x_i = shift i (d + 1 - i)``.

        The published minimum is for ``shift = 1``. The function is a convex
        quadratic, and replacing ``x`` by ``shift x`` scales it by
        ``shift^2``, so the minimiser scales by ``shift`` and the minimum by
        ``shift^2``.
        """
        d = self.d
        i = np.arange(1, d + 1, dtype=np.float64)
        points = (self.shift * i * (d + 1 - i))[np.newaxis]
        value = -(self.shift**2) * d * (d + 4) * (d - 1) / 6
        return formula_minimum(self, value, points)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        squares = np.sum((x - self.shift) ** 2, axis=1)
        neighbours = np.sum(x[:, 1:] * x[:, :-1], axis=1)
        return squares - neighbours
