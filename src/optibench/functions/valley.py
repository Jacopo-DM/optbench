"""Valley-shaped functions: the minimum lies in a long, flat valley.

https://www.sfu.ca/~ssurjano/optimization.html, group "Valley-Shaped".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optibench.problem import (
    AnyDimension,
    Group,
    Minimum,
    Problem,
    box,
    formula_minimum,
)
from optibench.symbolic import total, variables

from .known_minima import SIX_HUMP_CAMEL, THREE_HUMP_CAMEL, at_defaults
from .references import DIXON_SZEGO, GAVANA, HEDAR, MOLGA, PICHENY, WIKIPEDIA

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optibench.symbolic import Expr

_GROUP = Group.VALLEY
_SFU = "https://www.sfu.ca/~ssurjano/"


class ThreeHumpCamel(Problem):
    """Three-hump camel function: three local minima.

    ``f(x) = a x1^2 - b x1^4 + c x1^6 + x1 x2 + x2^2``

    Attributes
    ----------
    a, b, c
        The published 2, 1.05 and 1/6.
    """

    name: ClassVar[str] = "three_hump_camel"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "camel3.html"
    references: ClassVar[tuple[str, ...]] = (WIKIPEDIA,)

    a: float = 2.0
    b: float = 1.05
    c: float = 1 / 6

    @property
    @override
    def dimension(self) -> int:
        return 2

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.0, 5.0, 2)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin, for the published constants."""
        return at_defaults(self, THREE_HUMP_CAMEL)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        squared = x1**2
        polynomial = (self.a - (self.b - self.c * squared) * squared) * squared
        return polynomial + x1 * x2 + x2**2

    @override
    def formula(self) -> Expr:
        x1, x2 = variables(2)
        squared = x1**2
        polynomial = (self.a - (self.b - self.c * squared) * squared) * squared
        return polynomial + x1 * x2 + x2**2


class SixHumpCamel(Problem):
    """Six-hump camel function: six local minima, two of them global.

    ``f(x) = (a - b x1^2 + c x1^4) x1^2 + x1 x2 + (-e + g x2^2) x2^2``

    Attributes
    ----------
    a, b, c, e, g
        The published 4, 2.1, 1/3, 4 and 4.
    """

    name: ClassVar[str] = "six_hump_camel"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "camel6.html"
    references: ClassVar[tuple[str, ...]] = (MOLGA,)

    a: float = 4.0
    b: float = 2.1
    c: float = 1 / 3
    e: float = 4.0
    g: float = 4.0

    @property
    @override
    def dimension(self) -> int:
        return 2

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return np.array([[-3.0, 3.0], [-2.0, 2.0]])

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, SIX_HUMP_CAMEL)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        first = (self.a - self.b * x1**2 + self.c * x1**4) * x1**2
        last = (-self.e + self.g * x2**2) * x2**2
        return first + x1 * x2 + last

    @override
    def formula(self) -> Expr:
        x1, x2 = variables(2)
        first = (self.a - self.b * x1**2 + self.c * x1**4) * x1**2
        last = (-self.e + self.g * x2**2) * x2**2
        return first + x1 * x2 + last


class DixonPrice(AnyDimension):
    """Dixon-Price function.

    ``f(x) = (x1 - shift)^2 + sum_{i>=2} i (weight x_i^2 - x_{i-1})^2``

    Attributes
    ----------
    shift, weight
        The published 1 and 2.
    """

    name: ClassVar[str] = "dixon_price"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "dixonpr.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA,)

    shift: float = 1.0
    weight: float = 2.0

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero where ``x1 = shift`` and ``x_i = sqrt(x_{i-1} / weight)``.

        Every term then vanishes, which needs ``shift >= 0`` and
        ``weight > 0``. Only ``x_d`` may also be negative, because every
        other ``x_i`` appears unsquared in the next term: two minimisers for
        ``d >= 2``. The all-positive one is the one the SFU page gives
        (``x_i = 2^(-(2^i - 2) / 2^i)`` for the defaults).
        """
        if self.shift < 0 or self.weight <= 0:
            return None
        point = np.empty(self.d)
        point[0] = self.shift
        for i in range(1, self.d):
            point[i] = np.sqrt(point[i - 1] / self.weight)
        points = point[np.newaxis]
        if self.d >= 2 and point[-1] != 0:  # ruff: ignore[magic-value-comparison]
            flipped = point.copy()
            flipped[-1] = -flipped[-1]
            points = np.vstack([point, flipped])
        return formula_minimum(self, 0.0, points)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        first = (x[:, 0] - self.shift) ** 2
        i = np.arange(2, self.d + 1, dtype=np.float64)
        terms = i * (self.weight * x[:, 1:] ** 2 - x[:, :-1]) ** 2
        return first + np.sum(terms, axis=1)

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        first = (x[0] - self.shift) ** 2
        return first + total(
            i * (self.weight * x[i - 1] ** 2 - x[i - 2]) ** 2
            for i in range(2, self.d + 1)
        )


class Rosenbrock(AnyDimension):
    """Rosenbrock (valley, banana) function.

    ``f(x) = sum_{i<d} (a (x_{i+1} - x_i^2)^2 + (x_i - b)^2)``

    Attributes
    ----------
    a, b
        The published 100 and 1.
    """

    name: ClassVar[str] = "rosenbrock"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "rosen.html"
    references: ClassVar[tuple[str, ...]] = (DIXON_SZEGO, HEDAR, MOLGA, PICHENY)
    smallest_dimension: ClassVar[int] = 2

    a: float = 100.0
    b: float = 1.0

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.0, 10.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at ``(b, ..., b, b^2)`` when every term can vanish.

        For ``a >= 0`` that needs ``x_1 .. x_{d-1} = b`` and
        ``x_{i+1} = x_i^2``, so ``b = b^2`` (``b`` is 0 or 1) unless ``d = 2``.
        """
        if self.a < 0 or (self.d > 2 and self.b not in {0.0, 1.0}):  # ruff: ignore[magic-value-comparison]
            return None
        point = np.full(self.d, self.b)
        point[-1] = self.b**2
        return formula_minimum(self, 0.0, point[np.newaxis])

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        head, tail = x[:, :-1], x[:, 1:]
        terms = self.a * (tail - head**2) ** 2 + (head - self.b) ** 2
        return np.sum(terms, axis=1)

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        return total(
            self.a * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - self.b) ** 2
            for i in range(self.d - 1)
        )
