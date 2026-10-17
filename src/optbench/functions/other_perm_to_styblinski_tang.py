"""Functions the SFU page groups as "Other", Perm d beta to Styblinski-Tang.

https://www.sfu.ca/~ssurjano/optimization.html, group "Other". The rest of
the group is in ``other_beale_to_goldstein_price`` and ``other_hartmann``.
"""

from __future__ import annotations

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

from .references import GAVANA, HEDAR, LAGUNA_MARTI, MOLGA

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.OTHER
_SFU = "https://www.sfu.ca/~ssurjano/"


class PermDBeta(AnyDimension):
    """Perm function d, beta.

    ``f(x) = sum_i (sum_j (j^i + beta) ((x_j / j)^i - 1))^2``

    Attributes
    ----------
    beta
        0.5, the default of the SFU page's own code (the page gives none).
    """

    name: ClassVar[str] = "perm_d_beta"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "permdb.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR,)

    beta: float = 0.5

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-self.d, self.d, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at ``x_j = j``, for every ``beta``: each inner sum is 0."""
        points = np.arange(1, self.d + 1, dtype=np.float64)[np.newaxis]
        return formula_minimum(self, 0.0, points)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        d = self.d
        j = np.arange(1, d + 1, dtype=np.float64)

        # ratios[n, i, j] = (x_j / j)^(i+1), by repeated multiplication.
        repeated = np.broadcast_to((x / j)[:, np.newaxis, :], (len(x), d, d))
        ratios = np.cumprod(repeated, axis=1)
        weights = np.cumprod(np.broadcast_to(j, (d, d)), axis=0) + self.beta
        inner: NDArray[np.float64] = np.einsum(
            "nij,ij->ni",
            ratios - 1,
            weights,
        )
        return np.sum(inner**2, axis=1)


class Powell(AnyDimension):
    """Powell function, in blocks of four variables.

    Per block ``(x1, x2, x3, x4)``: ``(x1 + a x2)^2 + b (x3 - x4)^2
    + (x2 - c x3)^4 + e (x1 - x4)^4``

    Attributes
    ----------
    a, b, c, e
        The published 10, 5, 2 and 10.
    """

    name: ClassVar[str] = "powell"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "powell.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR, LAGUNA_MARTI)
    smallest_dimension: ClassVar[int] = 4

    d: int = 4
    a: float = 10.0
    b: float = 5.0
    c: float = 2.0
    e: float = 10.0

    @override
    def _check(self) -> None:
        super()._check()
        if self.d % 4:
            msg = f"d must be a multiple of 4, got {self.d}"
            raise ValueError(msg)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-4.0, 5.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Zero at the origin when ``b`` and ``e`` are not negative."""
        if min(self.b, self.e) < 0:
            return None
        return formula_minimum(self, 0.0, np.zeros((1, self.d)))

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x1, x2, x3, x4 = x[:, 0::4], x[:, 1::4], x[:, 2::4], x[:, 3::4]
        blocks = (
            (x1 + self.a * x2) ** 2
            + self.b * (x3 - x4) ** 2
            + (x2 - self.c * x3) ** 4
            + self.e * (x1 - x4) ** 4
        )
        return np.sum(blocks, axis=1)


_SHEKEL_BETA = (0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5)
_SHEKEL_C = (
    (4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
    (4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
    (4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
    (4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
)

# Minimisers of the published Shekel functions (m = 5, 7, 10), found by
# Newton steps from the published (4, 4, 4, 4) until the gradient was below
# 1e-13; the values agree with the published ones to their four decimals.
_SHEKEL_MINIMA = {
    5: (
        -10.153199679058227,
        (4.000037152819676, 4.00013327659156) * 2,
    ),
    7: (
        -10.402915336777744,
        (4.000572819251117, 3.9996062096096887) * 2,
    ),
    10: (
        -10.536443153483528,
        (4.000746868270634, 3.9995094800857736) * 2,
    ),
}


class Shekel(Problem):
    """Shekel function: ``m`` wells, the deepest near (4, 4, 4, 4).

    ``f(x) = -sum_{i<=m} 1 / (sum_j (x_j - C_ji)^2 + beta_i)``

    Attributes
    ----------
    m
        Number of wells used: the first ``m`` entries of ``beta`` and columns
        of ``c``. The literature uses 5, 7 or 10.
    beta
        Well widths, at least ``m`` of them.
    c
        Well centres, one row per variable and at least ``m`` columns. The
        number of rows is the dimension (4 for the published function).
    """

    name: ClassVar[str] = "shekel"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "shekel.html"
    references: ClassVar[tuple[str, ...]] = (HEDAR, MOLGA)

    m: int = 10
    beta: tuple[float, ...] = _SHEKEL_BETA
    c: tuple[tuple[float, ...], ...] = _SHEKEL_C

    @override
    def _check(self) -> None:
        centres = np.asarray(self.c, dtype=np.float64)
        if centres.ndim != 2:  # ruff: ignore[magic-value-comparison]
            msg = f"c must be a matrix, got shape {centres.shape}"
            raise ValueError(msg)
        if not 1 <= self.m <= min(len(self.beta), centres.shape[1]):
            msg = (
                f"m must be between 1 and the number of wells "
                f"({min(len(self.beta), centres.shape[1])}), got {self.m}"
            )
            raise ValueError(msg)

    @property
    @override
    def dimension(self) -> int:
        return len(self.c)

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(0.0, 10.0, self.dimension)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Known for the published wells with m = 5, 7 or 10."""
        published = self.beta == _SHEKEL_BETA and self.c == _SHEKEL_C
        if not (
            published and self.m in _SHEKEL_MINIMA and self.uses_default_domain
        ):
            return None
        value, point = _SHEKEL_MINIMA[self.m]
        return Minimum(
            value=value,
            points=np.array([point]),
            source="Newton refinement of the published (4, 4, 4, 4)",
            tolerance=1e-12,
        )

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        beta = np.asarray(self.beta[: self.m], dtype=np.float64)
        centres = np.asarray(self.c, dtype=np.float64)[:, : self.m]
        diff = x[:, :, np.newaxis] - centres
        squares = np.einsum("ijk,ijk->ik", diff, diff)
        wells: NDArray[np.float64] = 1 / (squares + beta)
        return -wells.sum(axis=1)


class StyblinskiTang(AnyDimension):
    """Styblinski-Tang function.

    ``f(x) = scale sum_i (x_i^4 - a x_i^2 + b x_i)``

    Attributes
    ----------
    scale, a, b
        The published 1/2, 16 and 5.
    """

    name: ClassVar[str] = "styblinski_tang"
    group: ClassVar[Group] = _GROUP
    url: ClassVar[str] = _SFU + "stybtang.html"
    references: ClassVar[tuple[str, ...]] = (GAVANA,)

    scale: float = 0.5
    a: float = 16.0
    b: float = 5.0

    @override
    def default_bounds(self) -> NDArray[np.float64]:
        return box(-5.0, 5.0, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """Exact for any ``a``, ``b``, positive ``scale`` and domain.

        The terms are independent quartics: each coordinate takes the best
        of the real roots of ``4 x^3 - 2 a x + b`` in its interval and the
        interval's ends (the SFU page gives -39.16599 d at x_i = -2.903534).
        """
        if self.scale <= 0:
            return None
        roots = np.roots([4.0, 0.0, -2 * self.a, self.b])
        real = roots[np.abs(roots.imag) < 1e-9].real  # ruff: ignore[magic-value-comparison]
        point = np.empty(self.d)
        for k, (low, high) in enumerate(self.bounds):
            candidates = np.concatenate(
                [real[(real >= low) & (real <= high)], [low, high]],
            )
            point[k] = candidates[np.argmin(self._quartic(candidates))]
        value = self.scale * float(np.sum(self._quartic(point)))
        return Minimum(
            value=value,
            points=point[np.newaxis],
            source="formula",
            tolerance=1e-12,
        )

    def _quartic(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return t**4 - self.a * t**2 + self.b * t

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.scale * np.sum(self._quartic(x), axis=1)
