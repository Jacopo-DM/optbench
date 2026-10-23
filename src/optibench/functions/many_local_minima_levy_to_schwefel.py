"""Functions with many local minima, Levy to Schwefel, for any dimension.

https://www.sfu.ca/~ssurjano/optimization.html, group "Many Local Minima".
Ackley to Langermann are in ``many_local_minima_ackley_to_langermann``,
and the fixed-dimension ones in ``many_local_minima_bukin_to_gramacy`` and
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
    box,
    constant_point,
)
from optibench.symbolic import absolute, cos, pi, sin, sqrt, total, variables

from .references import (
    GAVANA,
    HEDAR,
    LAGUNA_MARTI,
    POHLHEIM,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optibench.symbolic import Expr

_GROUP = Group.MANY_LOCAL_MINIMA
_SFU = "https://www.sfu.ca/~ssurjano/"


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

    @override
    def _check(self) -> None:
        super()._check()
        if self.scale == 0:
            msg = "scale must not be zero"
            raise ValueError(msg)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at ``x_i = shift`` when ``weight >= -1``.

        Every term is then at least zero, and all vanish where ``w = 1``.
        """
        points = constant_point(self.shift, self.d)
        if self.weight < -1 or not self.contains(points):
            return None
        return Minimum(
            value=0.0,
            points=points,
            source="formula",
            tolerance=0.0,
        )

    @override
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
        result: NDArray[np.float64] = first + middle + end
        return result

    @override
    def formula(self) -> Expr:
        w = [1 + (xi - self.shift) / self.scale for xi in variables(self.d)]
        first = sin(pi() * w[0]) ** 2
        middle = total(
            (wi - 1) ** 2 * (1 + self.weight * sin(pi() * wi + 1) ** 2)
            for wi in w[:-1]
        )
        end = (w[-1] - 1) ** 2 * (1 + sin(2 * pi() * w[-1]) ** 2)
        return first + middle + end


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

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.12, 5.12, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin when ``a`` is not negative.

        Each term ``x^2 + a (1 - cos(2 pi x))`` is then at least zero.
        """
        points = constant_point(0.0, self.d)
        if self.a < 0 or not self.contains(points):
            return None
        return Minimum(
            value=0.0,
            points=points,
            source="formula",
            tolerance=0.0,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        terms = x**2 - self.a * np.cos(2 * math.pi * x)
        return self.a * self.d + np.sum(terms, axis=1)

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        terms = total(xi**2 - self.a * cos(2 * pi() * xi) for xi in x)
        return self.a * self.d + terms


# The root of sin(sqrt(x)) + sqrt(x) cos(sqrt(x)) / 2 = 0 near 420.97, where
# x sin(sqrt(x)) is largest on [-500, 500] (bisection to machine precision;
# a grid of 2,000,001 points finds no higher value), and the value there.
_SCHWEFEL_ARGMAX = 420.968746359982
_SCHWEFEL_PEAK = 418.98288727243374


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

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-500.0, 500.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``d (offset - 418.98288727...)`` at ``x_i = 420.96874636...``.

        The terms are independent, so each coordinate sits where
        ``x sin(sqrt(|x|))`` peaks on the default domain.
        """
        if not self.uses_default_domain:
            return None
        return Minimum(
            value=self.d * (self.offset - _SCHWEFEL_PEAK),
            points=constant_point(_SCHWEFEL_ARGMAX, self.d),
            source="formula, with the peak found by bisection",
            tolerance=1e-12,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        terms = x * np.sin(np.sqrt(np.abs(x)))
        return self.offset * self.d - np.sum(terms, axis=1)

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        terms = total(xi * sin(sqrt(absolute(xi))) for xi in x)
        return self.offset * self.d - terms
