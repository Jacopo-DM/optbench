"""Plate-shaped functions: a flat, plate-like surface around the minimum.

https://www.sfu.ca/~ssurjano/optimization.html, group "Plate-Shaped".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import (
    AnyDimension,
    Group,
    Minimum,
    Problem,
    box,
    formula_minimum,
)

from .known_minima import POWER_SUM, at_defaults
from .references import ADORIO, HEDAR

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.PLATE
_SFU = "https://www.sfu.ca/~ssurjano/"


@dataclass(frozen=True, kw_only=True)
class Booth(Problem):
    """Booth function.

    ``f(x) = (x1 + weight x2 - a)^2 + (weight x1 + x2 - b)^2``

    Attributes
    ----------
    weight, a, b
        The published 2, 7 and 5.
    """

    name: ClassVar[str] = "booth"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "booth.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    weight: float = 2.0
    a: float = 7.0
    b: float = 5.0

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, 2)

    @property
    def minimum(self) -> Minimum | None:
        """Zero where both brackets vanish: one point unless ``weight = +-1``.

        Solving ``x1 + w x2 = a`` and ``w x1 + x2 = b`` gives
        ``x1 = (a - w b) / (1 - w^2)`` and ``x2 = (b - w a) / (1 - w^2)``;
        for the defaults that is (1, 3).
        """
        w = self.weight
        det = 1 - w**2
        if abs(det) < 1e-12:  # ruff: ignore[magic-value-comparison]
            return None
        point = np.array([
            [(self.a - w * self.b) / det, (self.b - w * self.a) / det],
        ])
        return formula_minimum(self, 0.0, point)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        first = (x1 + self.weight * x2 - self.a) ** 2
        second = (self.weight * x1 + x2 - self.b) ** 2
        return first + second


@dataclass(frozen=True, kw_only=True)
class Matyas(Problem):
    """Matyas function, ``f(x) = a (x1^2 + x2^2) - b x1 x2``.

    Attributes
    ----------
    a, b
        The published 0.26 and 0.48.
    """

    name: ClassVar[str] = "matyas"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "matya.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    a: float = 0.26
    b: float = 0.48

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, 2)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin while the quadratic form is not negative.

        That holds for ``a >= |b| / 2``; otherwise the function falls
        without bound along a diagonal.
        """
        if self.a < abs(self.b) / 2:
            return None
        return formula_minimum(self, 0.0, np.zeros((1, 2)))

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        return self.a * (x1**2 + x2**2) - self.b * x1 * x2


@dataclass(frozen=True, kw_only=True)
class McCormick(Problem):
    """McCormick function.

    ``f(x) = sin(x1 + x2) + (x1 - x2)^2 - a x1 + b x2 + offset``

    Attributes
    ----------
    a, b, offset
        The published 1.5, 2.5 and 1.
    """

    name: ClassVar[str] = "mccormick"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "mccorm.html"
    references: ClassVar[tuple[str, ...]] = (ADORIO,)

    a: float = 1.5
    b: float = 2.5
    offset: float = 1.0

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return np.array([[-1.5, 4.0], [-3.0, 4.0]])

    @property
    def minimum(self) -> Minimum | None:
        """``offset - 1 - sqrt(3) / 2 - pi / 3``, for the published a and b.

        With ``u = x1 + x2`` and ``v = x1 - x2`` the published function is
        ``sin(u) + u / 2 + (v - 1)^2 + offset - 1``: the minimum on the
        default domain has ``v = 1`` and ``cos(u) = -1/2``, ``u = -2 pi / 3``
        (the SFU page gives -1.9133 at (-0.54719, -1.54719)). Other ``a``
        and ``b`` move the minimum onto the domain's edge for some values, so
        they are not covered.
        """
        if (self.a, self.b) != (1.5, 2.5) or not self.uses_default_domain:
            return None
        u, v = -2 * math.pi / 3, 1.0
        value = self.offset - 1 - math.sqrt(3) / 2 - math.pi / 3
        return Minimum(value, np.array([[(u + v) / 2, (u - v) / 2]]), "formula")

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        linear = -self.a * x1 + self.b * x2 + self.offset
        return np.sin(x1 + x2) + (x1 - x2) ** 2 + linear


@dataclass(frozen=True, kw_only=True)
class PowerSum(Problem):
    """Power sum function, ``f(x) = sum_i (sum_j x_j^i - b_i)^2``.

    The dimension is the length of ``b``.

    Attributes
    ----------
    b
        The published (8, 18, 44, 114), for four variables.
    """

    name: ClassVar[str] = "power_sum"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "powersum.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    b: tuple[float, ...] = (8.0, 18.0, 44.0, 114.0)

    def _check(self) -> None:
        if not self.b:
            msg = "b must have at least one entry"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        return len(self.b)

    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, float(self.dimension), self.dimension)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the orderings of (1, 2, 2, 3), for the published b."""
        return at_defaults(self, POWER_SUM)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        d = self.dimension
        repeated = np.broadcast_to(x[:, np.newaxis, :], (len(x), d, d))
        sums = np.sum(np.cumprod(repeated, axis=1), axis=2)
        return np.sum((sums - np.asarray(self.b)) ** 2, axis=1)


@dataclass(frozen=True, kw_only=True)
class Zakharov(AnyDimension):
    """Zakharov function, with ``s = sum_i coefficient i x_i``.

    ``f(x) = sum_i x_i^2 + s^2 + s^4``

    Attributes
    ----------
    coefficient
        The published 0.5.
    """

    name: ClassVar[str] = "zakharov"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "zakharov.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    coefficient: float = 0.5

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.0, 10.0, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin, for every coefficient: no term is negative."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        i = np.arange(1, self.d + 1, dtype=np.float64)
        s = x @ (self.coefficient * i)
        return np.sum(x**2, axis=1) + s**2 + s**4
