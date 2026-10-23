"""Functions with many local minima, Ackley to Langermann.

https://www.sfu.ca/~ssurjano/optimization.html, group "Many Local Minima".
Ackley and Griewank are defined for any number of variables. Levy to
Schwefel are in ``many_local_minima_levy_to_schwefel``, and the other
fixed-dimension ones in ``many_local_minima_bukin_to_gramacy`` and
``many_local_minima_holder_to_shubert``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optibench.problem import (
    AnyDimension,
    Group,
    Minimum,
    Problem,
    box,
    constant_point,
)
from optibench.symbolic import cos, exp, pi, product, sqrt, total, variables

from .known_minima import LANGERMANN, at_defaults
from .references import (
    ADORIO,
    BACK,
    HEDAR,
    MOLGA,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optibench.symbolic import Expr

_GROUP = Group.MANY_LOCAL_MINIMA
_SFU = "https://www.sfu.ca/~ssurjano/"


class Ackley(AnyDimension):
    """Ackley function: a nearly flat outer region and a hole at the centre.

    ``f(x) = -a exp(-b sqrt(mean(x^2))) - exp(mean(cos(c x))) + a + e``

    Attributes
    ----------
    a, b, c
        The recommended ``a = 20``, ``b = 0.2`` and ``c = 2 pi``.
    """

    name: ClassVar[str] = "ackley"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "ackley.html"
    references: ClassVar[tuple[str, ...]] = (ADORIO, MOLGA, BACK)

    a: float = 20.0
    b: float = 0.2
    c: float = 2 * math.pi

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-32.768, 32.768, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin when ``a`` and ``b`` are not negative.

        The first term is then at least ``-a`` and the second at least
        ``-e``, and both reach it at the origin.
        """
        points = constant_point(0.0, self.d)
        if self.a < 0 or self.b < 0 or not self.contains(points):
            return None
        return Minimum(
            value=0.0,
            points=points,
            source="formula",
            tolerance=0.0,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        root_mean_square = np.sqrt(np.mean(x**2, axis=1))
        mean_cos = np.mean(np.cos(self.c * x), axis=1)
        return (
            -self.a * np.exp(-self.b * root_mean_square)
            - np.exp(mean_cos)
            + self.a
            + math.e
        )

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        root_mean_square = sqrt(total(xi**2 for xi in x) / self.d)
        mean_cos = total(cos(self.c * xi) for xi in x) / self.d
        return (
            -self.a * exp(-self.b * root_mean_square)
            - exp(mean_cos)
            + self.a
            + exp(1)
        )


class Griewank(AnyDimension):
    """Griewank function: many regularly spaced local minima.

    ``f(x) = sum(x_i^2) / divisor - prod(cos(x_i / sqrt(i))) + 1``

    Attributes
    ----------
    divisor
        The published 4000.
    """

    name: ClassVar[str] = "griewank"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "griewank.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR, MOLGA)

    divisor: float = 4000.0

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-600.0, 600.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin for a positive divisor.

        The sum is then at least zero and the product at most one.
        """
        points = constant_point(0.0, self.d)
        if self.divisor <= 0 or not self.contains(points):
            return None
        return Minimum(
            value=0.0,
            points=points,
            source="formula",
            tolerance=0.0,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        roots = np.sqrt(np.arange(1, self.d + 1))
        product = np.prod(np.cos(x / roots), axis=1)
        result: NDArray[np.float64] = (
            np.sum(x**2, axis=1) / self.divisor - product + 1
        )
        return result

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        waves = product(cos(xi / sqrt(i)) for i, xi in enumerate(x, start=1))
        return total(xi**2 for xi in x) / self.divisor - waves + 1


# Langermann's published constants (Molga and Smutnicki, for d = 2).
_LANGERMANN_C = (1.0, 2.0, 5.0, 2.0, 3.0)
_LANGERMANN_A = ((3.0, 5.0), (5.0, 2.0), (2.0, 1.0), (1.0, 4.0), (7.0, 9.0))


class Langermann(Problem):
    """Langermann function: unevenly spread local minima.

    ``f(x) = sum_i c_i exp(-|x - A_i|^2 / pi) cos(pi |x - A_i|^2)``

    The dimension is the number of columns of ``a``; the published constants
    are for two variables.

    Attributes
    ----------
    m
        Number of terms: the first ``m`` entries of ``c`` and rows of ``a``.
    c
        Term weights.
    a
        Term centres, one row per term.
    """

    name: ClassVar[str] = "langermann"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "langer.html"
    references: ClassVar[tuple[str, ...]] = (ADORIO, MOLGA)

    m: int = 5
    c: tuple[float, ...] = _LANGERMANN_C
    a: tuple[tuple[float, ...], ...] = _LANGERMANN_A

    @override
    def _check(self) -> None:
        centres = np.asarray(self.a, dtype=np.float64)
        if centres.ndim != 2:  # ruff: ignore[magic-value-comparison]
            msg = f"a must be a matrix, got shape {centres.shape}"
            raise ValueError(msg)
        terms = min(len(self.c), centres.shape[0])
        if not 1 <= self.m <= terms:
            msg = f"m must be between 1 and {terms}, got {self.m}"
            raise ValueError(msg)

    @property
    @override
    def dimension(self) -> int:
        return len(self.a[0])

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, 10.0, self.dimension)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, LANGERMANN)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        weights = np.asarray(self.c[: self.m], dtype=np.float64)
        centres = np.asarray(self.a, dtype=np.float64)[: self.m]
        diff = x[:, np.newaxis, :] - centres
        squares = np.einsum("ikj,ikj->ik", diff, diff)
        terms: NDArray[np.float64] = (
            weights * np.exp(-squares / math.pi) * np.cos(math.pi * squares)
        )
        return terms.sum(axis=1)

    @override
    def formula(self) -> Expr:
        x = variables(self.dimension)
        squares = [
            total((xi - aj) ** 2 for xi, aj in zip(x, centre, strict=True))
            for centre in self.a[: self.m]
        ]
        return total(
            weight * exp(-square / pi()) * cos(pi() * square)
            for weight, square in zip(self.c[: self.m], squares, strict=True)
        )
