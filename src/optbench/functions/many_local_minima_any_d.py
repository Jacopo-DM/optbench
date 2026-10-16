"""Functions with many local minima, defined for any number of variables.

https://www.sfu.ca/~ssurjano/optimization.html, group "Many Local Minima".
The fixed-dimension ones of the group are in
``many_local_minima_bukin_to_gramacy`` and
``many_local_minima_holder_to_shubert``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import AnyDimension, Group, Minimum, Problem, box

from .known_minima import LANGERMANN, at_defaults
from .references import (
    ADORIO,
    BACK,
    GAVANA,
    HEDAR,
    LAGUNA_MARTI,
    MOLGA,
    POHLHEIM,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.MANY_LOCAL_MINIMA
_SFU = "https://www.sfu.ca/~ssurjano/"


def _at(point: float, d: int) -> NDArray[np.float64]:
    """Return the single minimiser with every coordinate equal to ``point``.

    Returns
    -------
    numpy.ndarray
        Shaped ``(1, d)``.
    """
    return np.full((1, d), point)


@dataclass(frozen=True, kw_only=True)
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

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-32.768, 32.768, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin when ``a`` and ``b`` are not negative.

        The first term is then at least ``-a`` and the second at least
        ``-e``, and both reach it at the origin.
        """
        points = _at(0.0, self.d)
        if self.a < 0 or self.b < 0 or not self.contains(points):
            return None
        return Minimum(0.0, points, "formula", tolerance=0.0)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        root_mean_square = np.sqrt(np.mean(x**2, axis=1))
        mean_cos = np.mean(np.cos(self.c * x), axis=1)
        return (
            -self.a * np.exp(-self.b * root_mean_square)
            - np.exp(mean_cos)
            + self.a
            + math.e
        )


@dataclass(frozen=True, kw_only=True)
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

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-600.0, 600.0, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin for a positive divisor.

        The sum is then at least zero and the product at most one.
        """
        points = _at(0.0, self.d)
        if self.divisor <= 0 or not self.contains(points):
            return None
        return Minimum(0.0, points, "formula", tolerance=0.0)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        roots = np.sqrt(np.arange(1, self.d + 1))
        product = np.prod(np.cos(x / roots), axis=1)
        return np.sum(x**2, axis=1) / self.divisor - product + 1


# Langermann's published constants (Molga and Smutnicki, for d = 2).
_LANGERMANN_C = (1.0, 2.0, 5.0, 2.0, 3.0)
_LANGERMANN_A = ((3.0, 5.0), (5.0, 2.0), (2.0, 1.0), (1.0, 4.0), (7.0, 9.0))


@dataclass(frozen=True, kw_only=True)
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
    def dimension(self) -> int:
        return len(self.a[0])

    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, 10.0, self.dimension)

    @property
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, LANGERMANN)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        weights = np.asarray(self.c[: self.m], dtype=np.float64)
        centres = np.asarray(self.a, dtype=np.float64)[: self.m]
        diff = x[:, np.newaxis, :] - centres
        squares = np.einsum("ikj,ikj->ik", diff, diff)
        terms: NDArray[np.float64] = (
            weights * np.exp(-squares / math.pi) * np.cos(math.pi * squares)
        )
        return terms.sum(axis=1)


@dataclass(frozen=True, kw_only=True)
class Levy(AnyDimension):
    """Levy function.

    With ``w_i = 1 + (x_i - shift) / scale``:
    ``f(x) = sin^2(pi w_1)
    + sum_{i<d} (w_i - 1)^2 (1 + weight sin^2(pi w_i + 1))
    + (w_d - 1)^2 (1 + sin^2(2 pi w_d))``

    Attributes
    ----------
    shift, scale, weight
        The published ``shift = 1``, ``scale = 4`` and ``weight = 10``.
    """

    name: ClassVar[str] = "levy"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "levy.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA, LAGUNA_MARTI)

    shift: float = 1.0
    scale: float = 4.0
    weight: float = 10.0

    def _check(self) -> None:
        super()._check()
        if self.scale == 0:
            msg = "scale must not be zero"
            raise ValueError(msg)

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at ``x_i = shift`` when ``weight >= -1``.

        Every term is then at least zero, and all vanish where ``w = 1``.
        """
        points = _at(self.shift, self.d)
        if self.weight < -1 or not self.contains(points):
            return None
        return Minimum(0.0, points, "formula", tolerance=0.0)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        w = 1 + (x - self.shift) / self.scale
        first = np.sin(math.pi * w[:, 0]) ** 2
        inner = w[:, :-1]
        middle = np.sum(
            (inner - 1) ** 2
            * (1 + self.weight * np.sin(math.pi * inner + 1) ** 2),
            axis=1,
        )
        last = w[:, -1]
        end = (last - 1) ** 2 * (1 + np.sin(2 * math.pi * last) ** 2)
        return first + middle + end


@dataclass(frozen=True, kw_only=True)
class Rastrigin(AnyDimension):
    """Rastrigin function: regularly spaced local minima.

    ``f(x) = a d + sum(x_i^2 - a cos(2 pi x_i))``

    Attributes
    ----------
    a
        The published 10.
    """

    name: ClassVar[str] = "rastrigin"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "rastr.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR, POHLHEIM)

    a: float = 10.0

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.12, 5.12, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """Zero at the origin when ``a`` is not negative.

        Each term ``x^2 + a (1 - cos(2 pi x))`` is then at least zero.
        """
        points = _at(0.0, self.d)
        if self.a < 0 or not self.contains(points):
            return None
        return Minimum(0.0, points, "formula", tolerance=0.0)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        terms = x**2 - self.a * np.cos(2 * math.pi * x)
        return self.a * self.d + np.sum(terms, axis=1)


# The root of sin(sqrt(x)) + sqrt(x) cos(sqrt(x)) / 2 = 0 near 420.97, where
# x sin(sqrt(x)) is largest on [-500, 500] (bisection to machine precision;
# a grid of 2,000,001 points finds no higher value), and the value there.
_SCHWEFEL_ARGMAX = 420.968746359982
_SCHWEFEL_PEAK = 418.98288727243374


@dataclass(frozen=True, kw_only=True)
class Schwefel(AnyDimension):
    """Schwefel function: the best point lies far from the next best.

    ``f(x) = offset d - sum(x_i sin(sqrt(|x_i|)))``

    The published ``offset = 418.9829`` rounds the true peak of
    ``x sin(sqrt(|x|))``, so the minimum is about ``1.27e-5 d``, not zero.

    Attributes
    ----------
    offset
        The published 418.9829.
    """

    name: ClassVar[str] = "schwefel"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "schwef.html"
    references: ClassVar[tuple[str, ...]] = (
        POHLHEIM,
        GAVANA,
        HEDAR,
        LAGUNA_MARTI,
    )

    offset: float = 418.9829

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-500.0, 500.0, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """``d (offset - 418.98288727...)`` at ``x_i = 420.96874636...``.

        The terms are independent, so each coordinate sits where
        ``x sin(sqrt(|x|))`` peaks on the default domain.
        """
        if not self.uses_default_domain:
            return None
        return Minimum(
            self.d * (self.offset - _SCHWEFEL_PEAK),
            _at(_SCHWEFEL_ARGMAX, self.d),
            "formula, with the peak found by bisection",
            tolerance=1e-12,
        )

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        terms = x * np.sin(np.sqrt(np.abs(x)))
        return self.offset * self.d - np.sum(terms, axis=1)
