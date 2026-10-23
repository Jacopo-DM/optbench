"""Well-known test functions not on the SFU pages: Qing to Whitley.

Each is written from its formula in Jamil and Yang (2013), Gavana's AMPGO
pages, Wikipedia or its original paper (the ``references`` of each class).
They have ``suite = "classic"`` and the SFU shape group that fits them.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optibench.problem import (
    AnyDimension,
    Group,
    Minimum,
    box,
    formula_minimum,
)
from optibench.symbolic import (
    absolute,
    cos,
    maximum,
    product,
    sqrt,
    total,
    variables,
)

from .references import GAVANA, JAMIL_YANG

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optibench.symbolic import Expr

_SUITE = "classic"
_GAVANA = "https://infinity77.net/global_optimization/test_functions_nd_"


class Qing(AnyDimension):
    """Qing function, ``f(x) = sum_i (x_i^2 - i)^2``. It has no constants."""

    name: ClassVar[str] = "qing"
    group: ClassVar[Group] = Group.OTHER
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = _GAVANA + "Q.html"
    references: ClassVar[tuple[str, ...]] = (JAMIL_YANG, GAVANA)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-500.0, 500.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at every ``x_i = +-sqrt(i)``: ``2^d`` minimisers.

        They are built when this is read, ``8 d 2^d`` bytes: 80 kB at
        ``d = 10``, 168 MB at ``d = 20``.
        """
        roots = np.sqrt(np.arange(1, self.d + 1, dtype=np.float64))
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=self.d)))
        return formula_minimum(self, 0.0, roots * signs)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        i = np.arange(1, self.d + 1, dtype=np.float64)
        return np.sum((x**2 - i) ** 2, axis=1)

    @override
    def formula(self) -> Expr:
        return total(
            (xi**2 - float(i)) ** 2
            for i, xi in enumerate(variables(self.d), start=1)
        )


class Salomon(AnyDimension):
    """Salomon function, ``f(x) = 1 - cos(frequency |x|) + a |x|``.

    Attributes
    ----------
    frequency, a
        The published ``2 pi`` and 0.1.
    """

    name: ClassVar[str] = "salomon"
    group: ClassVar[Group] = Group.MANY_LOCAL_MINIMA
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = _GAVANA + "S.html"
    references: ClassVar[tuple[str, ...]] = (JAMIL_YANG, GAVANA)

    frequency: float = 2 * math.pi
    a: float = 0.1

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-100.0, 100.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin when ``a > 0``: both terms are then >= 0."""
        if self.a <= 0:
            return None
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        radius = np.sqrt(np.einsum("ij,ij->i", x, x))
        result: NDArray[np.float64] = (
            1 - np.cos(self.frequency * radius) + self.a * radius
        )
        return result

    @override
    def formula(self) -> Expr:
        radius = sqrt(total(xi**2 for xi in variables(self.d)))
        return 1 - cos(self.frequency * radius) + self.a * radius


class Schwefel221(AnyDimension):
    """Schwefel function 2.21, ``f(x) = max_i |x_i|``. No constants."""

    name: ClassVar[str] = "schwefel_2_21"
    group: ClassVar[Group] = Group.BOWL
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = _GAVANA + "S.html"
    references: ClassVar[tuple[str, ...]] = (JAMIL_YANG, GAVANA)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-100.0, 100.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.max(np.abs(x), axis=1)

    @override
    def formula(self) -> Expr:
        return maximum(absolute(xi) for xi in variables(self.d))


class Schwefel222(AnyDimension):
    """Schwefel function 2.22, ``sum_i |x_i| + prod_i |x_i|``. No constants."""

    name: ClassVar[str] = "schwefel_2_22"
    group: ClassVar[Group] = Group.BOWL
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = _GAVANA + "S.html"
    references: ClassVar[tuple[str, ...]] = (JAMIL_YANG, GAVANA)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-100.0, 100.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        size = np.abs(x)
        result: NDArray[np.float64] = np.sum(size, axis=1) + np.prod(
            size, axis=1
        )
        return result

    @override
    def formula(self) -> Expr:
        size = [absolute(xi) for xi in variables(self.d)]
        return total(size) + product(size)


class Whitley(AnyDimension):
    """Whitley function.

    With ``y_ij = a (x_i^2 - x_j)^2 + (1 - x_j)^2``:
    ``f(x) = sum_i sum_j (y_ij^2 / divisor - cos(y_ij) + 1)``

    Attributes
    ----------
    a, divisor
        The published 100 and 4000.
    """

    name: ClassVar[str] = "whitley"
    group: ClassVar[Group] = Group.MANY_LOCAL_MINIMA
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = _GAVANA + "W.html"
    references: ClassVar[tuple[str, ...]] = (JAMIL_YANG, GAVANA)

    a: float = 100.0
    divisor: float = 4000.0

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.24, 10.24, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at ``x_i = 1`` when ``a >= 0`` and ``divisor > 0``.

        Each term ``y^2 / divisor + 1 - cos(y)`` is then at least 0, and 0
        only at ``y = 0``, which needs every ``x_j = 1``.
        """
        if self.a < 0 or self.divisor <= 0:
            return None
        return formula_minimum(self, 0.0, np.ones((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        y = (
            self.a * (x[:, :, np.newaxis] ** 2 - x[:, np.newaxis, :]) ** 2
            + (1 - x[:, np.newaxis, :]) ** 2
        )
        terms = y**2 / self.divisor - np.cos(y) + 1
        return np.sum(terms, axis=(1, 2))

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        ys = [
            self.a * (xi**2 - xj) ** 2 + (1 - xj) ** 2 for xi in x for xj in x
        ]
        return total(y**2 / self.divisor - cos(y) + 1 for y in ys)
