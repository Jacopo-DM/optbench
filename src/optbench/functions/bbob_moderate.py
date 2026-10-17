"""BBOB functions f6 to f9: low or moderate conditioning.

From Hansen, Finck, Ros and Auger (2009), INRIA RR-6829, section 2, with
the instances of COCO (see ``bbob_common``). Where COCO's code and the
report differ, the code wins, so values match COCO's published data; each
case is said in the class. ``R`` is COCO's rotation from the instance's
seed and ``Q`` the one from the seed plus 1,000,000.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optbench.problem import Group, Minimum

from . import bbob_common as bbob

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP = Group.BBOB_MODERATE_CONDITIONING

# Beyond this size f7 rounds a coordinate to an integer, not to 1 / step.
_WHOLE_STEPS = 0.5


class BbobAttractiveSector(bbob.BbobRotated):
    """BBOB f6, attractive sector.

    ``f(x) = T_osz(sum_i (s_i z_i)^2)^power + f_opt``, with
    ``z = Q Lambda^conditioning R (x - x_opt)`` and ``s_i = sector`` where
    ``z_i x_opt_i > 0``, else 1: only a cone around ``x_opt`` is cheap.

    Attributes
    ----------
    conditioning, sector, power
        The published 10, 100 and 0.9.
    """

    name: ClassVar[str] = "bbob_f06_attractive_sector"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 6

    conditioning: float = 10.0
    sector: float = 100.0
    power: float = 0.9

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` for a positive power and a nonzero scaling."""
        if self.power <= 0 or self.conditioning <= 0:
            return None
        return super().minimum

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        scaling = bbob.scaling(self.conditioning, self.d)
        linear = self.q @ (scaling[:, np.newaxis] * self.r)
        z = bbob.rotate(x - self.x_opt, linear)
        s = np.where(z * self.x_opt > 0, self.sector, 1.0)
        squares = np.sum((s * z) ** 2, axis=1)
        return bbob.oscillate(squares) ** self.power + self.f_opt


class BbobStepEllipsoid(bbob.BbobRotated):
    """BBOB f7, step ellipsoid: plateaus on a rotated ellipsoid.

    With ``z_hat = Lambda^conditioning R (x - x_opt)``, each ``z_hat_i`` is
    rounded to an integer where ``|z_hat_i| > 0.5`` and to a multiple of
    ``1 / step`` otherwise (COCO's code tests ``|z_hat_i|``, the report
    ``z_hat_i``; halves round up); ``z = Q`` times the rounded vector, and
    ``f(x) = scale max(|z_hat_1| slope, sum_i weight^(i/(d-1)) z_i^2)
    + penalty f_pen(x) + f_opt``.

    The minimum is reached on a whole plateau, wherever ``z_hat_1 = 0`` and
    every other ``|z_hat_i| < 1 / (2 step)``; ``minimum`` lists ``x_opt``
    only.

    Attributes
    ----------
    conditioning, step, weight, scale, slope, penalty
        The published 10, 10, 100, 0.1, 1e-4 and 1.
    """

    name: ClassVar[str] = "bbob_f07_step_ellipsoid"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 7

    conditioning: float = 10.0
    step: float = 10.0
    weight: float = 100.0
    scale: float = 0.1
    slope: float = 1e-4
    penalty: float = 1.0

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` while every term is at least 0."""
        if min(self.scale, self.slope, self.penalty, self.weight) < 0:
            return None
        return super().minimum

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        scaling = bbob.scaling(self.conditioning, self.d)
        z_hat = bbob.rotate(x - self.x_opt, scaling[:, np.newaxis] * self.r)
        rounded = np.where(
            np.abs(z_hat) > _WHOLE_STEPS,
            np.floor(z_hat + 0.5),
            np.floor(self.step * z_hat + 0.5) / self.step,
        )
        z = bbob.rotate(rounded, self.q)
        weights = bbob.scaling(self.weight, self.d) ** 2
        ellipsoid = bbob.weighted(z**2, weights)
        floor = np.abs(z_hat[:, 0]) * self.slope
        boundary = self.penalty * bbob.penalty(x)
        return self.scale * np.maximum(floor, ellipsoid) + boundary + self.f_opt


def _rosenbrock(z: NDArray[np.float64], a: float) -> NDArray[np.float64]:
    """Rosenbrock's ``sum_i a (z_i^2 - z_{i+1})^2 + (z_i - 1)^2``, per point.

    Returns
    -------
    numpy.ndarray
        Shaped ``(n,)``.
    """
    head, tail = z[:, :-1], z[:, 1:]
    return np.sum(a * (head**2 - tail) ** 2 + (head - 1) ** 2, axis=1)


class BbobRosenbrock(bbob.BbobProblem):
    """BBOB f8, Rosenbrock.

    ``f(x) = sum_i a (z_i^2 - z_{i+1})^2 + (z_i - 1)^2 + f_opt``, with
    ``z = max(1, sqrt(d) / 8) (x - x_opt) + 1`` and ``x_opt`` three
    quarters of COCO's random point (so in [-3, 3]^d).

    Attributes
    ----------
    a
        The published 100.
    """

    name: ClassVar[str] = "bbob_f08_rosenbrock"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 8

    a: float = 100.0

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """Three quarters of COCO's random point."""
        return 0.75 * bbob.x_opt(self.seed, self.d)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` (``z = 1``) for ``a >= 0``."""
        return super().minimum if self.a >= 0 else None

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.rosenbrock_scale(self.d) * (x - self.x_opt) + 1
        return _rosenbrock(z, self.a) + self.f_opt


class BbobRosenbrockRotated(bbob.BbobRotated):
    """BBOB f9, rotated Rosenbrock.

    ``f(x)`` is f8's sum with ``z = max(1, sqrt(d) / 8) R x + 1/2``; the
    minimum is at ``x_opt = R^T 1 / (2 max(1, sqrt(d) / 8))``, where
    ``z = 1``. COCO builds ``R`` from the instance's seed itself, as here.

    Attributes
    ----------
    a
        The published 100.
    """

    name: ClassVar[str] = "bbob_f09_rosenbrock_rotated"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 9

    a: float = 100.0

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """``R^T 1 / (2 max(1, sqrt(d) / 8))``."""
        return self.r.sum(axis=0) / (2 * bbob.rosenbrock_scale(self.d))

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` (``z = 1``) for ``a >= 0``."""
        return super().minimum if self.a >= 0 else None

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.rosenbrock_scale(self.d) * bbob.rotate(x, self.r) + 0.5
        return _rosenbrock(z, self.a) + self.f_opt
