"""Bowl-shaped functions, Perm 0 to Trid: one minimum, no local minima.

https://www.sfu.ca/~ssurjano/optimization.html, group "Bowl-Shaped". Sphere
and Bohachevsky are in ``bowl_sphere_to_bohachevsky``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optibench.problem import (
    AnyDimension,
    Group,
    Minimum,
    box,
    formula_minimum,
)
from optibench.symbolic import absolute, total, variables

from .references import ADORIO, HEDAR, MOLGA

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optibench.symbolic import Expr

_GROUP = Group.BOWL
_SFU = "https://www.sfu.ca/~ssurjano/"


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

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-self.d, self.d, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at ``x_j = 1 / j``, for every ``beta``: each inner sum is 0."""
        points = 1 / np.arange(1, self.d + 1, dtype=np.float64)[np.newaxis]
        return formula_minimum(self, 0.0, points)

    @override
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

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        js = range(1, self.d + 1)
        return total(
            total((j + self.beta) * (x[j - 1] ** i - 1 / j**i) for j in js) ** 2
            for i in js
        )


class RotatedHyperEllipsoid(AnyDimension):
    """Rotated hyper-ellipsoid function, ``f(x) = sum_i sum_{j<=i} x_j^2``.

    Continuous, convex and unimodal. It has no constants.
    """

    name: ClassVar[str] = "rotated_hyper_ellipsoid"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "rothyp.html"
    references: ClassVar[tuple[str, ...]] = (MOLGA,)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-65.536, 65.536, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # x_j^2 appears in the sums for i = j, ..., d: d - j + 1 times.
        counts = np.arange(self.d, 0, -1, dtype=np.float64)
        return np.sum(counts * x**2, axis=1)

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        return total(
            total(x[j] ** 2 for j in range(i + 1)) for i in range(self.d)
        )


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

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-1.0, 1.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin while every exponent is positive."""
        if 1 + self.power_shift <= 0:
            return None
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        powers = np.arange(1, self.d + 1, dtype=np.float64) + self.power_shift
        return np.sum(np.abs(x) ** powers, axis=1)

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        return total(
            absolute(xi) ** (i + self.power_shift)
            for i, xi in enumerate(x, start=1)
        )


class SumSquares(AnyDimension):
    """Sum squares (axis-parallel hyper-ellipsoid), ``sum_i i x_i^2``.

    Continuous, convex and unimodal. It has no constants.
    """

    name: ClassVar[str] = "sum_squares"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "sumsqu.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR, MOLGA)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        weights = np.arange(1, self.d + 1, dtype=np.float64)
        return np.sum(weights * x**2, axis=1)

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        return total(i * xi**2 for i, xi in enumerate(x, start=1))


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

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        square = float(self.d**2)
        return box(-square, square, self.d)

    @property
    @override
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

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        squares = np.sum((x - self.shift) ** 2, axis=1)
        neighbours = np.sum(x[:, 1:] * x[:, :-1], axis=1)
        return squares - neighbours

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        squares = total((xi - self.shift) ** 2 for xi in x)
        neighbours = total(x[i] * x[i - 1] for i in range(1, self.d))
        return squares - neighbours
