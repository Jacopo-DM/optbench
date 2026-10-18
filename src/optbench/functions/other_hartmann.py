"""The Hartmann functions (3, 4 and 6 variables), from the "Other" group.

https://www.sfu.ca/~ssurjano/optimization.html, group "Other". The rest of
the group is in ``other_beale_to_goldstein_price`` and
``other_perm_to_styblinski_tang``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optbench.problem import Group, Minimum, Problem, box
from optbench.symbolic import exp, total, variables

from .known_minima import HARTMANN_3, HARTMANN_4, HARTMANN_6, at_defaults
from .references import DIXON_SZEGO, HEDAR, PICHENY

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optbench.symbolic import Expr

_GROUP = Group.OTHER
_SFU = "https://www.sfu.ca/~ssurjano/"

_ALPHA = (1.0, 1.2, 3.0, 3.2)
_A3 = (
    (3.0, 10.0, 30.0),
    (0.1, 10.0, 35.0),
    (3.0, 10.0, 30.0),
    (0.1, 10.0, 35.0),
)
_P3 = (
    (0.3689, 0.1170, 0.2673),
    (0.4699, 0.4387, 0.7470),
    (0.1091, 0.8732, 0.5547),
    (0.0381, 0.5743, 0.8828),
)
_A6 = (
    (10.0, 3.0, 17.0, 3.5, 1.7, 8.0),
    (0.05, 10.0, 17.0, 0.1, 8.0, 14.0),
    (3.0, 3.5, 1.7, 10.0, 17.0, 8.0),
    (17.0, 8.0, 0.05, 10.0, 0.1, 14.0),
)
_P6 = (
    (0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886),
    (0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991),
    (0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650),
    (0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381),
)


type _Matrix = tuple[tuple[float, ...], ...]


def _first_four(matrix: _Matrix) -> _Matrix:
    return tuple(row[:4] for row in matrix)


class _Hartmann(Problem):
    """What the Hartmann functions share.

    ``g(x) = sum_i alpha_i exp(-sum_j A_ij (x_j - P_ij)^2)``; the dimension
    is the number of columns of ``a`` and ``p``.

    Attributes
    ----------
    alpha
        Term weights; the published (1, 1.2, 3, 3.2).
    a, p
        Term widths and centres, one row per term.
    """

    group: ClassVar[Group] = _GROUP
    alpha: tuple[float, ...] = _ALPHA
    a: tuple[tuple[float, ...], ...] = _A3
    p: tuple[tuple[float, ...], ...] = _P3

    @override
    def _check(self) -> None:
        widths = np.asarray(self.a, dtype=np.float64)
        centres = np.asarray(self.p, dtype=np.float64)
        if (
            widths.ndim != 2  # ruff: ignore[magic-value-comparison]
            or widths.shape != centres.shape
            or widths.shape[0] != len(self.alpha)
        ):
            msg = "a and p must be matrices with one row per alpha"
            raise ValueError(msg)

    @property
    @override
    def dimension(self) -> int:
        return len(self.a[0])

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, 1.0, self.dimension)

    def _bumps(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``g(x)`` for points shaped ``(n, d)``.

        Returns
        -------
        numpy.ndarray
            One value per point.
        """
        widths = np.asarray(self.a, dtype=np.float64)
        centres = np.asarray(self.p, dtype=np.float64)
        diff = x[:, np.newaxis, :] - centres
        exponents = np.einsum("nij,ij->ni", diff**2, widths)
        bumps: NDArray[np.float64] = np.exp(-exponents) @ np.asarray(self.alpha)
        return bumps

    def _bumps_formula(self) -> Expr:
        """Return ``g(x)`` as a sympy expression.

        Returns
        -------
        Expr
            The sum of the weighted bumps.
        """
        x = variables(self.dimension)
        return total(
            alpha
            * exp(
                -total(
                    w * (xj - c) ** 2
                    for xj, w, c in zip(x, widths, centres, strict=True)
                ),
            )
            for alpha, widths, centres in zip(
                self.alpha,
                self.a,
                self.p,
                strict=True,
            )
        )


class Hartmann3(_Hartmann):
    """Hartmann 3-D function, ``f(x) = -g(x)``: four local minima."""

    name: ClassVar[str] = "hartmann3"
    url: ClassVar[str] = _SFU + "hart3.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, HARTMANN_3)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self._bumps(x)

    @override
    def formula(self) -> Expr:
        return -self._bumps_formula()


class Hartmann4(_Hartmann):
    """Hartmann 4-D function, rescaled to mean 0 and variance 1.

    ``f(x) = (offset - g(x)) / scale``, with the first four columns of the
    6-D function's ``a`` and ``p``, as given by Picheny et al. (2012).

    Attributes
    ----------
    offset, scale
        The published 1.1 and 0.839.
    """

    name: ClassVar[str] = "hartmann4"
    url: ClassVar[str] = _SFU + "hart4.html"
    references: ClassVar[tuple[str, ...]] = (DIXON_SZEGO, PICHENY)

    a: tuple[tuple[float, ...], ...] = _first_four(_A6)
    p: tuple[tuple[float, ...], ...] = _first_four(_P6)
    offset: float = 1.1
    scale: float = 0.839

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, HARTMANN_4)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (self.offset - self._bumps(x)) / self.scale

    @override
    def formula(self) -> Expr:
        return (self.offset - self._bumps_formula()) / self.scale


class Hartmann6(_Hartmann):
    """Hartmann 6-D function, ``f(x) = -g(x)``: six local minima."""

    name: ClassVar[str] = "hartmann6"
    url: ClassVar[str] = _SFU + "hart6.html"
    references: ClassVar[tuple[str, ...]] = (DIXON_SZEGO, HEDAR, PICHENY)

    a: tuple[tuple[float, ...], ...] = _A6
    p: tuple[tuple[float, ...], ...] = _P6

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published constants (found numerically)."""
        return at_defaults(self, HARTMANN_6)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self._bumps(x)

    @override
    def formula(self) -> Expr:
        return -self._bumps_formula()
