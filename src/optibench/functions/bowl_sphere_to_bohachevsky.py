"""Bowl-shaped functions, Sphere and Bohachevsky: one minimum, no local minima.

https://www.sfu.ca/~ssurjano/optimization.html, group "Bowl-Shaped". Perm
0 to Trid are in ``bowl_perm_to_trid``.
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
    formula_minimum,
)
from optibench.symbolic import cos, pi, total, variables

from .references import DIXON_SZEGO, HEDAR, MOLGA, PICHENY

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optibench.symbolic import Expr

_GROUP = Group.BOWL
_SFU = "https://www.sfu.ca/~ssurjano/"


class Sphere(AnyDimension):
    """Sphere function, ``f(x) = sum_i x_i^2``.

    Continuous, convex and unimodal. It has no constants.
    """

    name: ClassVar[str] = "sphere"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "spheref.html"
    references: ClassVar[tuple[str, ...]] = (DIXON_SZEGO, MOLGA, PICHENY)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.12, 5.12, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin, for every ``d``, on any domain holding it."""
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        result: NDArray[np.float64] = np.einsum("ij,ij->i", x, x)
        return result

    @override
    def formula(self) -> Expr:
        return total(xi**2 for xi in variables(self.d))


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
    @override
    def dimension(self) -> int:
        return 2

    @override
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
    @override
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

    def _squares_formula(self) -> tuple[Expr, Expr, Expr]:
        """Return ``x1``, ``x2`` and the formula of the two squares.

        Returns
        -------
        tuple of Expr
            ``x1``, ``x2`` and ``x1^2 + weight x2^2``.
        """
        x1, x2 = variables(2)
        return x1, x2, x1**2 + self.weight * x2**2


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

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        waves = self.a * np.cos(self.p * math.pi * x1) + self.b * np.cos(
            self.q * math.pi * x2,
        )
        return self._squares(x) - waves + self.offset

    @override
    def formula(self) -> Expr:
        x1, x2, squares = self._squares_formula()
        waves = self.a * cos(self.p * pi() * x1) + self.b * cos(
            self.q * pi() * x2,
        )
        return squares - waves + self.offset


class Bohachevsky2(_Bohachevsky):
    """Bohachevsky function 2.

    ``f(x) = x1^2 + weight x2^2 - a cos(p pi x1) cos(q pi x2) + offset``
    """

    name: ClassVar[str] = "bohachevsky2"

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        wave = np.cos(self.p * math.pi * x1) * np.cos(self.q * math.pi * x2)
        return self._squares(x) - self.a * wave + self.offset

    @override
    def formula(self) -> Expr:
        x1, x2, squares = self._squares_formula()
        wave = cos(self.p * pi() * x1) * cos(self.q * pi() * x2)
        return squares - self.a * wave + self.offset


class Bohachevsky3(_Bohachevsky):
    """Bohachevsky function 3.

    ``f(x) = x1^2 + weight x2^2 - a cos(p pi x1 + q pi x2) + offset``
    """

    name: ClassVar[str] = "bohachevsky3"

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        wave = np.cos(self.p * math.pi * x1 + self.q * math.pi * x2)
        return self._squares(x) - self.a * wave + self.offset

    @override
    def formula(self) -> Expr:
        x1, x2, squares = self._squares_formula()
        wave = cos(self.p * pi() * x1 + self.q * pi() * x2)
        return squares - self.a * wave + self.offset
