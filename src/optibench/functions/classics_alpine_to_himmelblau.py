"""Well-known test functions not on the SFU pages: Alpine N. 2 to Himmelblau.

Each is written from its formula in Jamil and Yang (2013), Gavana's AMPGO
pages, Wikipedia or its original paper (the ``references`` of each class).
They have ``suite = "classic"`` and the SFU shape group that fits them.
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
from optibench.symbolic import (
    cos,
    exp,
    piecewise,
    product,
    sin,
    sqrt,
    total,
    variables,
)

from .known_minima import BIRD, HIMMELBLAU, at_defaults
from .references import BEYER_FINCK, GAVANA, JAMIL_YANG, WIKIPEDIA

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optibench.symbolic import Expr

_SUITE = "classic"
_GAVANA = "https://infinity77.net/global_optimization/test_functions_nd_"


def _within(problem: Problem, low: float, high: float) -> bool:
    """Whether the domain in use lies inside ``[low, high]`` in every variable.

    Returns
    -------
    bool
        True if every bound is between ``low`` and ``high``.
    """
    bounds = problem.bounds
    return bool(np.all(bounds >= low) and np.all(bounds <= high))


class AlpineN2(AnyDimension):
    """Alpine function N. 2, ``f(x) = -prod_i sqrt(x_i) sin(x_i)``.

    Usually stated as a maximisation of the product; this is its negative.
    It has no constants.
    """

    name: ClassVar[str] = "alpine_n2"
    group: ClassVar[Group] = Group.MANY_LOCAL_MINIMA
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = _GAVANA + "A.html"
    references: ClassVar[tuple[str, ...]] = (JAMIL_YANG, GAVANA)

    # Where sqrt(x) sin(x) peaks on [0, 10] (the root of
    # sin x + 2 x cos x = 0 near 7.92, by bisection; a grid of 2,000,001
    # points finds no higher value), and the peak.
    argmax: ClassVar[float] = 7.917052684666206
    peak: ClassVar[float] = 2.8081311800070052

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, 10.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``-peak^d`` with every ``x_i`` at the peak.

        Holds on any domain inside ``[0, 10]^d`` that contains that point:
        every factor is at most ``peak`` in size there, and an even number
        of negative factors (at most 2.18 in size) gives less.
        """
        points = np.full((1, self.d), self.argmax)
        if not (_within(self, 0.0, 10.0) and self.contains(points)):
            return None
        return Minimum(
            value=-(self.peak**self.d),
            points=points,
            source="formula, with the peak found by bisection",
            tolerance=1e-12,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -np.prod(np.sqrt(x) * np.sin(x), axis=1)

    @override
    def formula(self) -> Expr:
        return -product(sqrt(xi) * sin(xi) for xi in variables(self.d))


class Bird(Problem):
    """Bird function.

    ``f(x) = sin(x1) exp((1 - cos x2)^2) + cos(x2) exp((1 - sin x1)^2)
    + (x1 - x2)^2``. It has no constants.
    """

    name: ClassVar[str] = "bird"
    group: ClassVar[Group] = Group.MANY_LOCAL_MINIMA
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = _GAVANA + "B.html"
    references: ClassVar[tuple[str, ...]] = (JAMIL_YANG, GAVANA)

    @property
    @override
    def dimension(self) -> int:
        return 2

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-2 * math.pi, 2 * math.pi, 2)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Two minimisers, found numerically (see ``known_minima``)."""
        return at_defaults(self, BIRD)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        result: NDArray[np.float64] = (
            np.sin(x1) * np.exp((1 - np.cos(x2)) ** 2)
            + np.cos(x2) * np.exp((1 - np.sin(x1)) ** 2)
            + (x1 - x2) ** 2
        )
        return result

    @override
    def formula(self) -> Expr:
        x1, x2 = variables(2)
        return (
            sin(x1) * exp((1 - cos(x2)) ** 2)
            + cos(x2) * exp((1 - sin(x1)) ** 2)
            + (x1 - x2) ** 2
        )


class Deceptive(AnyDimension):
    """Deceptive function, ``f(x) = -(mean_i g_i(x_i))^beta``.

    Each ``g_i`` is piecewise linear in ``x_i`` with a narrow peak of 1 at
    ``alpha_i`` and a broad slope towards 0.8 at ``x_i = 0`` and ``x_i = 1``,
    which leads a search away from the minimum. Gavana's page leaves out the
    ``+ 1`` and ``+ 4/5`` of the last two pieces, which makes ``g`` jump;
    SciPy's benchmark code, based on Gavana's own code, has them, and so
    does this. Outside ``[0, 1]`` the end pieces continue as straight lines.

    Attributes
    ----------
    alpha
        Where each ``g_i`` peaks, in ``(0, 1)``; ``None`` (the default) means
        ``i / (d + 1)``, as Gavana's code has it.
    beta
        The published 2.
    """

    name: ClassVar[str] = "deceptive"
    group: ClassVar[Group] = Group.OTHER
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = _GAVANA + "D.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA,)

    alpha: tuple[float, ...] | None = None
    beta: float = 2.0

    @override
    def _check(self) -> None:
        super()._check()
        peaks = self._peaks()
        if peaks.shape != (self.d,) or np.any((peaks <= 0) | (peaks >= 1)):
            msg = f"alpha must be {self.d} values strictly between 0 and 1"
            raise ValueError(msg)

    def _peaks(self) -> NDArray[np.float64]:
        if self.alpha is None:
            return np.arange(1, self.d + 1, dtype=np.float64) / (self.d + 1)
        return np.asarray(self.alpha, dtype=np.float64)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, 1.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """-1 at ``x = alpha`` when ``beta > 0`` and the domain is in [0, 1].

        There every ``g_i`` is between 0 and 1 and reaches 1 only at
        ``alpha_i``.
        """
        points = self._peaks()[np.newaxis]
        if self.beta <= 0 or not _within(self, 0.0, 1.0):
            return None
        return formula_minimum(self, -1.0, points)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        a = self._peaks()
        pieces = [
            -x / a + 0.8,
            5 * x / a - 4,
            5 * (x - a) / (a - 1) + 1,
            (x - 1) / (1 - a) + 0.8,
        ]
        conditions = [x < 0.8 * a, x < a, x < (1 + 4 * a) / 5]
        g = np.select(conditions, pieces[:3], default=pieces[3])
        return -(np.mean(g, axis=1) ** self.beta)

    @override
    def formula(self) -> Expr:
        g = [
            piecewise(
                (-xi / a + 0.8, xi < 0.8 * a),
                (5 * xi / a - 4, xi < a),
                (5 * (xi - a) / (a - 1) + 1, xi < (1 + 4 * a) / 5),
                ((xi - 1) / (1 - a) + 0.8, True),
            )
            for xi, a in zip(
                variables(self.d), self._peaks().tolist(), strict=True
            )
        ]
        return -((total(g) / self.d) ** self.beta)


class HappyCat(AnyDimension):
    """HappyCat function, a curved groove around the sphere ``|x|^2 = d``.

    ``f(x) = ((|x|^2 - d)^2)^alpha + (|x|^2 / 2 + sum_i x_i) / d + 1/2``

    Attributes
    ----------
    alpha
        The groove's shape; Beyer and Finck's 1/8.
    """

    name: ClassVar[str] = "happy_cat"
    group: ClassVar[Group] = Group.VALLEY
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = (
        "https://link.springer.com/chapter/10.1007/978-3-642-32937-1_37"
    )
    references: ClassVar[tuple[str, ...]] = (BEYER_FINCK,)

    alpha: float = 0.125

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        # The domain of the CEC 2014 competition, which uses this function.
        return box(-100.0, 100.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at ``x_i = -1`` for ``alpha > 0``.

        The first term is then at least 0, and the rest equals
        ``sum_i (x_i + 1)^2 / (2 d)``; both vanish only there.
        """
        if self.alpha <= 0:
            return None
        return formula_minimum(self, 0.0, np.full((1, self.d), -1.0))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        squares = np.einsum("ij,ij->i", x, x)
        groove = ((squares - self.d) ** 2) ** self.alpha
        result: NDArray[np.float64] = (
            groove + (squares / 2 + np.sum(x, axis=1)) / self.d + 0.5
        )
        return result

    @override
    def formula(self) -> Expr:
        x = variables(self.d)
        squares = total(xi**2 for xi in x)
        groove = ((squares - self.d) ** 2) ** self.alpha
        return groove + (squares / 2 + total(x)) / self.d + 0.5


class Himmelblau(Problem):
    """Himmelblau's function, ``f(x) = (x1^2 + x2 - a)^2 + (x1 + x2^2 - b)^2``.

    Attributes
    ----------
    a, b
        The published 11 and 7.
    """

    name: ClassVar[str] = "himmelblau"
    group: ClassVar[Group] = Group.OTHER
    suite: ClassVar[str] = _SUITE
    url: ClassVar[str] = "https://en.wikipedia.org/wiki/Himmelblau%27s_function"
    references: ClassVar[tuple[str, ...]] = (WIKIPEDIA, JAMIL_YANG)

    a: float = 11.0
    b: float = 7.0

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
        """Zero at four points, for the published constants."""
        return at_defaults(self, HIMMELBLAU)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        return (x1**2 + x2 - self.a) ** 2 + (x1 + x2**2 - self.b) ** 2

    @override
    def formula(self) -> Expr:
        x1, x2 = variables(2)
        return (x1**2 + x2 - self.a) ** 2 + (x1 + x2**2 - self.b) ** 2
