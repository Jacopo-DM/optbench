"""BBOB functions f1 to f5: the separable group.

From Hansen, Finck, Ros and Auger (2009), INRIA RR-6829, section 1, with
the instances of COCO (see ``bbob_common``). Where COCO's code and the
report differ, the code wins, so values match COCO's published data; each
case is said in the class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from . import bbob_common as bbob

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optbench.problem import Minimum


@dataclass(frozen=True, kw_only=True)
class BbobSphere(bbob.BbobProblem):
    """BBOB f1, sphere: ``f(x) = |x - x_opt|^2 + f_opt``. No constants."""

    name: ClassVar[str] = "bbob_f01_sphere"
    function: ClassVar[int] = 1

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = x - self.x_opt
        return np.einsum("ij,ij->i", z, z) + self.f_opt


@dataclass(frozen=True, kw_only=True)
class BbobEllipsoid(bbob.BbobProblem):
    """BBOB f2, separable ellipsoid.

    ``f(x) = sum_i conditioning^(i/(d-1)) z_i^2 + f_opt``, with
    ``z = T_osz(x - x_opt)`` and ``i`` from 0.

    Attributes
    ----------
    conditioning
        The published 1e6.
    """

    name: ClassVar[str] = "bbob_f02_ellipsoid"
    function: ClassVar[int] = 2

    conditioning: float = 1e6

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` for a positive conditioning."""
        return super().minimum if self.conditioning > 0 else None

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.oscillate(x - self.x_opt)
        weights = bbob.scaling(self.conditioning, self.d) ** 2
        return z**2 @ weights + self.f_opt


@dataclass(frozen=True, kw_only=True)
class BbobRastrigin(bbob.BbobProblem):
    """BBOB f3, separable Rastrigin.

    ``f(x) = a (d - sum_i cos(2 pi z_i)) + |z|^2 + f_opt``, with
    ``z = Lambda^conditioning T_asy^beta(T_osz(x - x_opt))``.

    Attributes
    ----------
    conditioning, beta, a
        The published 10, 0.2 and 10.
    """

    name: ClassVar[str] = "bbob_f03_rastrigin"
    function: ClassVar[int] = 3

    conditioning: float = 10.0
    beta: float = 0.2
    a: float = 10.0

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` while ``a >= 0`` and the scaling is positive.

        Each term ``z_i^2 + a (1 - cos(2 pi z_i))`` is then at least 0.
        """
        if self.a < 0 or self.conditioning <= 0:
            return None
        return super().minimum

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.asymmetric(bbob.oscillate(x - self.x_opt), self.beta)
        z *= bbob.scaling(self.conditioning, self.d)
        return bbob.rastrigin(z, self.a) + self.f_opt


@dataclass(frozen=True, kw_only=True)
class BbobBucheRastrigin(bbob.BbobProblem):
    """BBOB f4, Bueche-Rastrigin.

    ``f(x) = a (d - sum_i cos(2 pi z_i)) + |z|^2 + penalty f_pen(x)
    + f_opt``, with ``z_i = s_i T_osz(x - x_opt)_i`` and
    ``s_i = conditioning^(i / (2 (d - 1)))``, times ``skew`` where ``z_i``
    is positive and ``i`` is even (i from 0: the report's odd i from 1).

    As in COCO's code but not in the report, ``x_opt`` takes the absolute
    value in every even coordinate, and the instance uses function 3's
    seeds.

    Attributes
    ----------
    conditioning, skew, a, penalty
        The published 10, 10, 10 and 100.
    """

    name: ClassVar[str] = "bbob_f04_buche_rastrigin"
    function: ClassVar[int] = 4

    conditioning: float = 10.0
    skew: float = 10.0
    a: float = 10.0
    penalty: float = 100.0

    @property
    def seed(self) -> int:
        """COCO builds f4 from function 3's seed."""
        return 3 + 10_000 * self.instance

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """Function 3's ``x_opt``, made positive in the even coordinates."""
        point = bbob.x_opt(self.seed, self.d).copy()
        point[::2] = np.abs(point[::2])
        return point

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` while the constants keep each term >= 0."""
        if (
            min(self.a, self.penalty) < 0
            or min(self.conditioning, self.skew) <= 0
        ):
            return None
        return super().minimum

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.oscillate(x - self.x_opt)
        even = np.arange(self.d) % 2 == 0
        skewed = np.where((z > 0) & even, self.skew, 1.0)
        z = z * bbob.scaling(self.conditioning, self.d) * skewed
        boundary = self.penalty * bbob.penalty(x)
        return bbob.rastrigin(z, self.a) + boundary + self.f_opt


@dataclass(frozen=True, kw_only=True)
class BbobLinearSlope(bbob.BbobProblem):
    """BBOB f5, linear slope towards a corner of the domain.

    ``f(x) = sum_i 5 |s_i| - s_i z_i + f_opt``, where ``x_opt`` is the
    corner ``5 sign(u)`` for COCO's random point ``u``,
    ``s_i = sign(x_opt_i) conditioning^(i / (2 (d - 1)))``, and
    ``z_i = x_i`` until ``x_i`` passes ``x_opt_i`` (``x_i x_opt_i >= 25``),
    where it stays at ``x_opt_i``: the function is flat beyond the corner.

    Attributes
    ----------
    conditioning
        The published 100 (so the slopes run from 1 to 10).
    """

    name: ClassVar[str] = "bbob_f05_linear_slope"
    function: ClassVar[int] = 5

    conditioning: float = 100.0

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """The corner of [-5, 5]^d on the side of COCO's random point."""
        return np.where(bbob.x_opt(self.seed, self.d) < 0, -5.0, 5.0)

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at the corner for a positive conditioning.

        In [-5, 5]^d each term ``5 |s_i| - s_i z_i`` is at least 0 and is 0
        only at ``z_i = x_opt_i``.
        """
        return super().minimum if self.conditioning > 0 else None

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        corner = self.x_opt
        slopes = np.sign(corner) * bbob.scaling(self.conditioning, self.d)
        # Beyond the corner (x_i x_opt_i >= 5^2) the slope is flat.
        z = np.where(x * corner < corner * corner, x, corner)
        return np.sum(5 * np.abs(slopes) - slopes * z, axis=1) + self.f_opt
