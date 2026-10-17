"""Functions with many local minima in two variables, Holder Table to Shubert.

https://www.sfu.ca/~ssurjano/optimization.html, group "Many Local Minima".
Bukin to Gramacy and Lee are in ``many_local_minima_bukin_to_gramacy``, and
the functions for any dimension in ``many_local_minima_any_d``. Minima the
SFU page rounds or omits were found numerically; see ``known_minima``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optbench.problem import Group, Minimum, Problem, box

from .known_minima import HOLDER_TABLE, SCHAFFER_N4, SHUBERT, at_defaults
from .references import GAVANA, HEDAR, WIKIPEDIA

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.MANY_LOCAL_MINIMA
_SFU = "https://www.sfu.ca/~ssurjano/"


class HolderTable(Problem):
    """Holder Table function: four global minima near the corners.

    ``f(x) = -|sin x1 cos x2 exp(|offset - |x| / pi|)|``

    Attributes
    ----------
    offset
        The published 1.
    """

    name: ClassVar[str] = "holder_table"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "holder.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA, WIKIPEDIA)

    offset: float = 1.0

    @property
    @override
    def dimension(self) -> int:
        return 2

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, 2)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published constant (found numerically)."""
        return at_defaults(self, HOLDER_TABLE)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        radius = np.sqrt(x1**2 + x2**2)
        growth = np.exp(np.abs(self.offset - radius / math.pi))
        return -np.abs(np.sin(x1) * np.cos(x2) * growth)


class LevyN13(Problem):
    """Levy function N. 13.

    ``f(x) = sin^2(a pi x1) + (x1 - 1)^2 (1 + sin^2(b pi x2))
    + (x2 - 1)^2 (1 + sin^2(c pi x2))``

    Attributes
    ----------
    a, b, c
        The published frequencies 3, 3 and 2.
    """

    name: ClassVar[str] = "levy_n13"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "levy13.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA,)

    a: float = 3.0
    b: float = 3.0
    c: float = 2.0

    @property
    @override
    def dimension(self) -> int:
        return 2

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, 2)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at (1, 1) when ``a`` is a whole number.

        Every term is at least zero; the last two vanish at (1, 1) for any
        ``b`` and ``c``, the first only when ``sin(a pi) = 0``.
        """
        points = np.ones((1, 2))
        if not float(self.a).is_integer() or not self.contains(points):
            return None
        return Minimum(
            value=0.0,
            points=points,
            source="formula",
            tolerance=0.0,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        first = np.sin(self.a * math.pi * x1) ** 2
        second = (x1 - 1) ** 2 * (1 + np.sin(self.b * math.pi * x2) ** 2)
        third = (x2 - 1) ** 2 * (1 + np.sin(self.c * math.pi * x2) ** 2)
        return first + second + third


class SchafferN2(Problem):
    """Schaffer function N. 2.

    ``f(x) = 0.5 + (sin^2(x1^2 - x2^2) - 0.5) / (1 + scale |x|^2)^2``

    Attributes
    ----------
    scale
        The published 0.001.
    """

    name: ClassVar[str] = "schaffer_n2"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "schaffer2.html"
    references: ClassVar[tuple[str, ...]] = (WIKIPEDIA,)

    scale: float = 0.001

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
        """Zero at the origin for ``scale >= 0``.

        The fraction's top is at least -0.5 and its bottom at least 1.
        """
        points = np.zeros((1, 2))
        if self.scale < 0 or not self.contains(points):
            return None
        return Minimum(
            value=0.0,
            points=points,
            source="formula",
            tolerance=0.0,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        top = np.sin(x1**2 - x2**2) ** 2 - 0.5
        bottom = (1 + self.scale * (x1**2 + x2**2)) ** 2
        return 0.5 + top / bottom


class SchafferN4(Problem):
    """Schaffer function N. 4.

    ``f(x) = 0.5 + (cos^2(sin(|x1^2 - x2^2|)) - 0.5) / (1 + scale |x|^2)^2``

    Attributes
    ----------
    scale
        The published 0.001.
    """

    name: ClassVar[str] = "schaffer_n4"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "schaffer4.html"
    references: ClassVar[tuple[str, ...]] = (WIKIPEDIA,)

    scale: float = 0.001

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
        """Known for the published constant (found numerically)."""
        return at_defaults(self, SCHAFFER_N4)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2 = x[:, 0], x[:, 1]
        top = np.cos(np.sin(np.abs(x1**2 - x2**2))) ** 2 - 0.5
        bottom = (1 + self.scale * (x1**2 + x2**2)) ** 2
        return 0.5 + top / bottom


class Shubert(Problem):
    """Shubert function: 18 global minima among many local ones.

    ``f(x) = prod_{j=1,2} sum_{i=1}^{terms} i cos((i + 1) x_j + i)``

    Attributes
    ----------
    terms
        The published 5.
    """

    name: ClassVar[str] = "shubert"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "shubert.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    terms: int = 5

    @override
    def _check(self) -> None:
        if self.terms < 1:
            msg = f"terms must be at least 1, got {self.terms}"
            raise ValueError(msg)

    @property
    @override
    def dimension(self) -> int:
        return 2

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-10.0, 10.0, 2)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published constant (found numerically)."""
        return at_defaults(self, SHUBERT)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        i = np.arange(1, self.terms + 1, dtype=np.float64)
        sums = np.sum(i * np.cos((i + 1) * x[:, :, np.newaxis] + i), axis=2)
        return np.prod(sums, axis=1)
