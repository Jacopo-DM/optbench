"""Functions with many local minima in one or two variables, Bukin to Gramacy.

https://www.sfu.ca/~ssurjano/optimization.html, group "Many Local Minima".
Holder Table to Shubert are in ``many_local_minima_holder_to_shubert``, and
the functions for any dimension in ``many_local_minima_any_d``. Minima the
SFU page rounds or omits were found numerically; see ``numerical_minima``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import Group, Minimum, Problem, box

from .numerical_minima import (
    CROSS_IN_TRAY,
    EGGHOLDER,
    GRAMACY_LEE_2012,
    at_defaults,
)
from .references import GAVANA, WIKIPEDIA

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.MANY_LOCAL_MINIMA
_SFU = "https://www.sfu.ca/~ssurjano/"
_GRAMACY_LEE = (
    "Gramacy, R. B., & Lee, H. K. (2012). Cases for the nugget in modeling "
    "computer experiments. Statistics and Computing, 22(3), 713-722."
)
_RANJAN = (
    "Ranjan, P. (2013). Comment: EI Criteria for Noisy Computer Simulators. "
    "Technometrics, 55(1), 24-28."
)


@dataclass(frozen=True, kw_only=True)
class Bukin6(Problem):
    """Bukin function N. 6: local minima along a curved ridge.

    ``f(x) = scale sqrt(|x2 - curvature x1^2|) + slope |x1 + shift|``

    Attributes
    ----------
    scale, curvature, slope, shift
        The published 100, 0.01, 0.01 and 10.
    """

    name: ClassVar[str] = "bukin6"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "bukin6.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA,)

    scale: float = 100.0
    curvature: float = 0.01
    slope: float = 0.01
    shift: float = 10.0

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return np.array([[-15.0, -5.0], [-3.0, 3.0]])

    @property
    def minimum(self) -> Minimum | None:
        """Zero at ``(-shift, curvature shift^2)`` for non-negative weights."""
        points = np.array([[-self.shift, self.curvature * self.shift**2]])
        if self.scale < 0 or self.slope < 0 or not self.contains(points):
            return None
        return Minimum(0.0, points, "formula", tolerance=0.0)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        ridge = np.sqrt(np.abs(x2 - self.curvature * x1**2))
        return self.scale * ridge + self.slope * np.abs(x1 + self.shift)


@dataclass(frozen=True, kw_only=True)
class CrossInTray(Problem):
    """Cross-in-Tray function: four global minima around a cross.

    ``f(x) = -scale (|sin x1 sin x2 exp(|offset - |x| / pi|)| + 1)^power``

    Attributes
    ----------
    scale, offset, power
        The published 0.0001, 100 and 0.1.
    """

    name: ClassVar[str] = "cross_in_tray"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "crossit.html"
    references: ClassVar[tuple[str, ...]] = (WIKIPEDIA,)

    scale: float = 0.0001
    offset: float = 100.0
    power: float = 0.1

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, 2)

    @property
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, CROSS_IN_TRAY)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        radius = np.sqrt(x1**2 + x2**2)
        growth = np.exp(np.abs(self.offset - radius / math.pi))
        inner = np.abs(np.sin(x1) * np.sin(x2) * growth) + 1
        return -self.scale * inner**self.power


@dataclass(frozen=True, kw_only=True)
class DropWave(Problem):
    """Drop-Wave function: rings of local minima around a central drop.

    ``f(x) = -(1 + cos(frequency r)) / (curvature r^2 + offset)``, with
    ``r = |x|``.

    Attributes
    ----------
    frequency, curvature, offset
        The published 12, 0.5 and 2.
    """

    name: ClassVar[str] = "drop_wave"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "drop.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA,)

    frequency: float = 12.0
    curvature: float = 0.5
    offset: float = 2.0

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.12, 5.12, 2)

    @property
    def minimum(self) -> Minimum | None:
        """``-2 / offset`` at the origin, for ``curvature >= 0 < offset``.

        The numerator is at most 2 and the denominator at least ``offset``.
        """
        points = np.zeros((1, 2))
        if self.curvature < 0 or self.offset <= 0 or not self.contains(points):
            return None
        return Minimum(-2 / self.offset, points, "formula")

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        squares = np.sum(x**2, axis=1)
        top = 1 + np.cos(self.frequency * np.sqrt(squares))
        return -top / (self.curvature * squares + self.offset)


@dataclass(frozen=True, kw_only=True)
class Eggholder(Problem):
    """Eggholder function: many deep local minima; the best is on the edge.

    ``f(x) = -(x2 + shift) sin(sqrt(|x2 + ratio x1 + shift|))
    - x1 sin(sqrt(|x1 - (x2 + shift)|))``

    Attributes
    ----------
    shift, ratio
        The published 47 and 1/2.
    """

    name: ClassVar[str] = "eggholder"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "egg.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA,)

    shift: float = 47.0
    ratio: float = 0.5

    @property
    def dimension(self) -> int:
        return 2

    def default_bounds(self) -> NDArray[np.float64]:
        return box(-512.0, 512.0, 2)

    @property
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, EGGHOLDER)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        lifted = x2 + self.shift
        first = lifted * np.sin(np.sqrt(np.abs(lifted + self.ratio * x1)))
        second = x1 * np.sin(np.sqrt(np.abs(x1 - lifted)))
        return -first - second


@dataclass(frozen=True, kw_only=True)
class GramacyLee2012(Problem):
    """Gramacy and Lee (2012) function, one variable.

    ``f(x) = sin(frequency pi x) / (scale x) + (x - shift)^power``

    Attributes
    ----------
    frequency, scale, shift, power
        The published 10, 2, 1 and 4.
    """

    name: ClassVar[str] = "gramacy_lee_2012"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "grlee12.html"
    references: ClassVar[tuple[str, ...]] = (_GRAMACY_LEE, _RANJAN)

    frequency: float = 10.0
    scale: float = 2.0
    shift: float = 1.0
    power: float = 4.0

    @property
    def dimension(self) -> int:
        return 1

    def default_bounds(self) -> NDArray[np.float64]:
        return np.array([[0.5, 2.5]])

    @property
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, GRAMACY_LEE_2012)

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        t = x[:, 0]
        wave = np.sin(self.frequency * math.pi * t) / (self.scale * t)
        return wave + (t - self.shift) ** self.power
