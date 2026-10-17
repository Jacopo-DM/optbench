"""Functions with steep ridges or drops on a mostly flat surface.

https://www.sfu.ca/~ssurjano/optimization.html, group "Steep Ridges/Drops".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optbench.problem import (
    AnyDimension,
    Group,
    Minimum,
    Problem,
    box,
    formula_minimum,
)

from .known_minima import DE_JONG_5, at_defaults
from .references import GAVANA, HEDAR, MOLGA

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.STEEP
_SFU = "https://www.sfu.ca/~ssurjano/"

# The 25 foxhole centres: x1 runs through the five values fastest.
_FOXHOLES = (-32.0, -16.0, 0.0, 16.0, 32.0)
_DE_JONG_A = (
    tuple(_FOXHOLES * 5),
    tuple(v for v in _FOXHOLES for _ in range(5)),
)


class DeJong5(Problem):
    """De Jong function N. 5 (Shekel's foxholes): 25 sharp drops.

    ``f(x) = (offset + sum_i 1 / (i + (x1 - a_1i)^power
    + (x2 - a_2i)^power))^(-1)``

    Attributes
    ----------
    offset, power
        The published 0.002 and 6.
    a
        The drop centres, two rows of 25: a 5 by 5 grid on
        (-32, -16, 0, 16, 32).
    """

    name: ClassVar[str] = "de_jong5"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "dejong5.html"
    references: ClassVar[tuple[str, ...]] = (MOLGA,)

    offset: float = 0.002
    power: float = 6.0
    a: tuple[tuple[float, ...], tuple[float, ...]] = _DE_JONG_A

    @override
    def _check(self) -> None:
        # Exactly two rows is the field's type, so pydantic checks that.
        if len(self.a[0]) != len(self.a[1]):
            msg = "a must have two rows of equal length"
            raise ValueError(msg)

    @property
    @override
    def dimension(self) -> int:
        return 2

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-65.536, 65.536, 2)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, DE_JONG_5)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        centres = np.asarray(self.a, dtype=np.float64)
        i = np.arange(1, centres.shape[1] + 1, dtype=np.float64)
        diff = x[:, :, np.newaxis] - centres
        bottoms = i + np.sum(diff**self.power, axis=1)
        return 1 / (self.offset + np.sum(1 / bottoms, axis=1))


class Easom(Problem):
    """Easom function: a single narrow drop on a flat plane.

    ``f(x) = -cos(x1) cos(x2) exp(-(x1 - centre)^2 - (x2 - centre)^2)``

    Attributes
    ----------
    centre
        The published pi.
    """

    name: ClassVar[str] = "easom"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "easom.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    centre: float = math.pi

    @property
    @override
    def dimension(self) -> int:
        return 2

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-100.0, 100.0, 2)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """-1 at ``(centre, centre)`` when ``centre`` is a multiple of pi.

        ``f >= -exp(-|x - (centre, centre)|^2) >= -1``, and -1 needs both the
        exponential and ``cos(x1) cos(x2)`` to be 1.
        """
        turns = self.centre / math.pi
        if abs(turns - round(turns)) > 1e-12:  # ruff: ignore[magic-value-comparison]
            return None
        points = np.full((1, 2), self.centre)
        return formula_minimum(self, -1.0, points)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        distance = (x1 - self.centre) ** 2 + (x2 - self.centre) ** 2
        return -np.cos(x1) * np.cos(x2) * np.exp(-distance)


# Points per unit length of the grid that brackets each coordinate's
# minimum, and Newton steps after it.
_MICHALEWICZ_GRID = 20_000
_NEWTON_STEPS = 30


class Michalewicz(AnyDimension):
    """Michalewicz function: ``d!`` local minima in steep valleys.

    ``f(x) = -sum_i sin(x_i) sin(i x_i^2 / pi)^(2 m)``

    Attributes
    ----------
    m
        Steepness of the valleys; the published 10.
    """

    name: ClassVar[str] = "michalewicz"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "michal.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA, HEDAR, MOLGA)

    m: float = 10.0

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, math.pi, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Exact for any ``d``, ``m`` and domain: the terms are independent.

        Each coordinate's term is minimised on its own interval, by a grid
        and then Newton steps (the SFU page gives -1.8013, -4.687658 and
        -9.66015 for d = 2, 5 and 10). Needs ``m > 0``.
        """
        if self.m <= 0:
            return None
        point = np.empty(self.d)
        for i, (low, high) in enumerate(self.bounds, start=1):
            point[i - 1] = _michalewicz_argmin(i, self.m, low, high)
        value = float(self(point[np.newaxis])[0])
        return Minimum(
            value=value,
            points=point[np.newaxis],
            source="per-coordinate search",
            tolerance=1e-12,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        i = np.arange(1, self.d + 1, dtype=np.float64)
        return -np.sum(_michalewicz_terms(x, i, self.m), axis=1)


def _michalewicz_terms(
    x: NDArray[np.float64],
    i: NDArray[np.float64] | float,
    m: float,
) -> NDArray[np.float64]:
    """Return ``sin(x) sin(i x^2 / pi)^(2 m)`` element-wise.

    The power is taken as ``(sin^2)^m``, the same for every ``m`` where
    ``sin^(2 m)`` is defined, and also defined for a negative sine when
    ``2 m`` is not a whole number.

    Returns
    -------
    numpy.ndarray
        One term per entry of ``x``.
    """
    return np.sin(x) * (np.sin(i * x**2 / math.pi) ** 2) ** m


def _michalewicz_argmin(i: int, m: float, low: float, high: float) -> float:
    """Find where coordinate ``i``'s term is largest on ``[low, high]``.

    Returns
    -------
    float
        The maximiser of ``sin(x) sin(i x^2 / pi)^(2 m)``: the best grid
        point, refined by Newton steps on the derivative while they stay in
        the grid cell around it.
    """
    count = max(2, math.ceil((high - low) * _MICHALEWICZ_GRID)) + 1
    grid = np.linspace(low, high, count)
    k = int(np.argmax(_michalewicz_terms(grid, i, m)))
    cell_low, cell_high = grid[max(k - 1, 0)], grid[min(k + 1, count - 1)]
    x, h = float(grid[k]), 1e-6

    def term(t: float) -> float:
        return float(_michalewicz_terms(np.array([t]), i, m)[0])

    for _ in range(_NEWTON_STEPS):
        first = (term(x + h) - term(x - h)) / (2 * h)
        second = (term(x + h) - 2 * term(x) + term(x - h)) / h**2
        if second >= 0:
            break
        step = first / second
        if not cell_low <= x - step <= cell_high:
            break
        x -= step
        if abs(step) < 1e-15:  # ruff: ignore[magic-value-comparison]
            break
    return x
