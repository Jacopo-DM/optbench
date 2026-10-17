"""BBOB functions f10 to f14: high conditioning, unimodal.

From Hansen, Finck, Ros and Auger (2009), INRIA RR-6829, section 3, with
the instances of COCO (see ``bbob_common``). Where COCO's code and the
report differ, the code wins, so values match COCO's published data; each
case is said in the class. ``r`` and ``q`` are COCO's two rotations (see
``BbobRotated``); the report calls the single rotation of f10, f11, f12
and f14 ``R``, and COCO builds it as ``q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from optbench.problem import Group, Minimum

from . import bbob_common as bbob

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.BBOB_HIGH_CONDITIONING


@dataclass(frozen=True, kw_only=True)
class BbobEllipsoidRotated(bbob.BbobRotated):
    """BBOB f10, rotated ellipsoid.

    ``f(x) = sum_i conditioning^(i/(d-1)) z_i^2 + f_opt``, with
    ``z = T_osz(q (x - x_opt))`` and ``i`` from 0.

    Attributes
    ----------
    conditioning
        The published 1e6.
    """

    name: ClassVar[str] = "bbob_f10_ellipsoid_rotated"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 10

    conditioning: float = 1e6

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` for a positive conditioning."""
        return super().minimum if self.conditioning > 0 else None

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.oscillate(bbob.rotate(x - self.x_opt, self.q))
        weights = bbob.scaling(self.conditioning, self.d) ** 2
        return bbob.weighted(z**2, weights) + self.f_opt


@dataclass(frozen=True, kw_only=True)
class BbobDiscus(bbob.BbobRotated):
    """BBOB f11, discus: one direction far steeper than the rest.

    ``f(x) = conditioning z_1^2 + sum_{i>1} z_i^2 + f_opt``, with
    ``z = T_osz(q (x - x_opt))``.

    Attributes
    ----------
    conditioning
        The published 1e6.
    """

    name: ClassVar[str] = "bbob_f11_discus"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 11

    conditioning: float = 1e6

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` for a positive conditioning."""
        return super().minimum if self.conditioning > 0 else None

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.oscillate(bbob.rotate(x - self.x_opt, self.q))
        squares = z**2
        rest = np.sum(squares[:, 1:], axis=1)
        return self.conditioning * squares[:, 0] + rest + self.f_opt


@dataclass(frozen=True, kw_only=True)
class BbobBentCigar(bbob.BbobRotated):
    """BBOB f12, bent cigar: one direction far flatter than the rest.

    ``f(x) = z_1^2 + conditioning sum_{i>1} z_i^2 + f_opt``, with
    ``z = q T_asy^beta(q (x - x_opt))``. As in COCO's code but not in the
    report, ``x_opt`` comes from the seed plus 1,000,000, the seed of ``q``.

    Attributes
    ----------
    conditioning, beta
        The published 1e6 and 0.5.
    """

    name: ClassVar[str] = "bbob_f12_bent_cigar"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 12

    conditioning: float = 1e6
    beta: float = 0.5

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """COCO's random point from the seed of ``q``."""
        return bbob.x_opt(self.seed + bbob.SECOND_ROTATION, self.d)

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` for a positive conditioning."""
        return super().minimum if self.conditioning > 0 else None

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        rotated = bbob.rotate(x - self.x_opt, self.q)
        z = bbob.rotate(bbob.asymmetric(rotated, self.beta), self.q)
        squares = z**2
        rest = np.sum(squares[:, 1:], axis=1)
        return squares[:, 0] + self.conditioning * rest + self.f_opt


@dataclass(frozen=True, kw_only=True)
class BbobSharpRidge(bbob.BbobRotated):
    """BBOB f13, sharp ridge.

    ``f(x) = z_1^2 + a sqrt(sum_{i>1} z_i^2) + f_opt``, with
    ``z = q Lambda^conditioning r (x - x_opt)``.

    Attributes
    ----------
    conditioning, a
        The published 10 and 100.
    """

    name: ClassVar[str] = "bbob_f13_sharp_ridge"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 13

    conditioning: float = 10.0
    a: float = 100.0

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` while ``a >= 0`` and the scaling > 0."""
        if self.a < 0 or self.conditioning <= 0:
            return None
        return super().minimum

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        scaling = bbob.scaling(self.conditioning, self.d)
        linear = self.q @ (scaling[:, np.newaxis] * self.r)
        z = bbob.rotate(x - self.x_opt, linear)
        ridge = np.sqrt(np.sum(z[:, 1:] ** 2, axis=1))
        return z[:, 0] ** 2 + self.a * ridge + self.f_opt


@dataclass(frozen=True, kw_only=True)
class BbobDifferentPowers(bbob.BbobRotated):
    """BBOB f14, different powers.

    ``f(x) = sqrt(sum_i |z_i|^(2 + spread i/(d-1))) + f_opt``, with
    ``z = q (x - x_opt)`` and ``i`` from 0.

    Attributes
    ----------
    spread
        How much the exponent grows across the coordinates; the published
        4 (exponents 2 to 6).
    """

    name: ClassVar[str] = "bbob_f14_different_powers"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 14

    spread: float = 4.0

    @property
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` while every exponent is positive."""
        smallest_power = 2 + min(self.spread, 0.0)
        return super().minimum if smallest_power > 0 else None

    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.rotate(x - self.x_opt, self.q)
        ramp = np.arange(self.d, dtype=np.float64) / (self.d - 1)
        powers = np.abs(z) ** (2 + self.spread * ramp)
        return np.sqrt(np.sum(powers, axis=1)) + self.f_opt
