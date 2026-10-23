"""BBOB functions f23 and f24 (Katsuura, Lunacek): weak global structure.

From Hansen, Finck, Ros and Auger (2009), INRIA RR-6829, section 5, with
the instances of COCO (see ``bbob_common``). Where COCO's code and the
report differ, the code wins, so values match COCO's published data; each
case is said in the class. ``r`` and ``q`` are COCO's two rotations (see
``BbobRotated``). f20 to f22 are in
``bbob_multimodal_weak_schwefel_to_gallagher``.
"""

import math
from typing import ClassVar, override

import numpy as np
from numpy.typing import NDArray

from optibench.problem import Group, Minimum
from optibench.symbolic import (
    Expr,
    cos,
    floor,
    minimum,
    pi,
    product,
    total,
    variables,
)

from . import bbob_common as bbob
from . import bbob_symbolic as sym

_GROUP = Group.BBOB_MULTIMODAL_WEAK


class BbobKatsuura(bbob.BbobRotated):
    """BBOB f23, Katsuura: rugged everywhere.

    ``f(x) = 10 / d^2 (prod_i (1 + i sum_{j=1}^{terms}
    |2^j z_i - round(2^j z_i)| / 2^j)^(10 / d^1.2) - 1)
    + penalty f_pen(x) + f_opt``, with
    ``z = q Lambda^conditioning r (x - x_opt)`` and ``i`` from 1.

    Attributes
    ----------
    conditioning, terms, penalty
        The published 100, 32 and 1.
    """

    name: ClassVar[str] = "bbob_f23_katsuura"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 23

    conditioning: float = 100.0
    terms: int = 32
    penalty: float = 1.0

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` (``z = 0``, every distance 0)."""
        if self.conditioning <= 0 or self.penalty < 0:
            return None
        return super().minimum

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        scaling = bbob.scaling(self.conditioning, self.d)
        linear = bbob.linear_map(self.q, scaling, self.r)
        z = bbob.rotate(x - self.x_opt, linear)

        # The distance of 2^j z to the nearest integer, with halves rounded
        # up as COCO does. Doubling the fractional part is exact, so this
        # equals COCO's |2^j z - round(2^j z)| term by term.
        fraction = z - np.floor(z)
        total = np.zeros_like(z)
        for j in range(1, self.terms + 1):
            fraction *= 2
            fraction -= np.floor(fraction)
            distance = np.minimum(fraction, 1 - fraction)
            total += distance / 2.0**j

        index = np.arange(1, self.d + 1, dtype=np.float64)
        factors: NDArray[np.float64] = (1 + index * total) ** (10 / self.d**1.2)
        core = 10 / self.d**2 * (np.prod(factors, axis=1) - 1)
        return core + self.penalty * bbob.penalty(x) + self.f_opt

    @override
    def formula(self) -> Expr:
        scaling = bbob.scaling(self.conditioning, self.d)
        linear = bbob.linear_map(self.q, scaling, self.r)
        z = sym.rotated(linear, sym.shifted(self.x_opt))

        # _evaluate doubles a running fraction; written that way each step
        # would hold the last one twice, 2^32 copies of z when printed.
        # frac(2^j z) is the same number, since doubling a float is exact.
        factors: list[Expr] = []
        for i, zi in enumerate(z, start=1):
            distances: list[Expr] = []
            for j in range(1, self.terms + 1):
                fraction = 2.0**j * zi - floor(2.0**j * zi)
                distances.append(minimum((fraction, 1 - fraction)) / 2.0**j)
            factors.append((1 + i * total(distances)) ** (10 / self.d**1.2))
        core = 10 / self.d**2 * (product(factors) - 1)
        return core + self.penalty * sym.penalty(variables(self.d)) + self.f_opt


class BbobLunacek(bbob.BbobRotated):
    """BBOB f24, Lunacek bi-Rastrigin: two funnels, the better one smaller.

    With ``x_hat = 2 sign(x_opt) x``, ``mu_1 = -sqrt((mu_0^2 - depth) / s)``
    and ``s = 1 - 1 / (2 sqrt(d + 20) - 8.2)``:
    ``f(x) = min(sum (x_hat - mu_0)^2, depth d + s sum (x_hat - mu_1)^2)
    + a (d - sum cos(2 pi z_i)) + penalty f_pen(x) + f_opt``, where
    ``z = q Lambda^conditioning r (x_hat - mu_0)``. ``x_opt`` is
    ``+-mu_0 / 2``, the signs those of COCO's normal numbers.

    Attributes
    ----------
    mu0, depth, conditioning, a, penalty
        The published 2.5, 1, 100, 10 and 1e4.
    """

    name: ClassVar[str] = "bbob_f24_lunacek"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 24

    mu0: float = 2.5
    depth: float = 1.0
    conditioning: float = 100.0
    a: float = 10.0
    penalty: float = 1e4

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """``+-mu_0 / 2``, the signs from COCO's normal numbers."""
        signs = np.where(bbob.gauss(self.d, self.seed) < 0, -1.0, 1.0)
        return signs * 0.5 * self.mu0

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt``, where ``x_hat = mu_0`` and ``z = 0``.

        Holds while the second funnel stays above 0 (``depth > 0``) and the
        other terms are at least 0.
        """
        if (
            min(self.depth, self.conditioning) <= 0
            or min(self.a, self.penalty) < 0
        ):
            return None
        return super().minimum

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        d = self.d
        s = 1 - 0.5 / (math.sqrt(d + 20) - 4.1)
        mu1 = -math.sqrt((self.mu0**2 - self.depth) / s)
        x_hat = 2 * np.sign(self.x_opt) * x
        near = np.sum((x_hat - self.mu0) ** 2, axis=1)
        far = self.depth * d + s * np.sum((x_hat - mu1) ** 2, axis=1)
        scaling = bbob.scaling(self.conditioning, d)
        linear = bbob.linear_map(self.q, scaling, self.r)
        z = bbob.rotate(x_hat - self.mu0, linear)
        waves = self.a * (d - np.sum(np.cos(2 * math.pi * z), axis=1))
        boundary = self.penalty * bbob.penalty(x)
        return np.minimum(near, far) + waves + boundary + self.f_opt

    @override
    def formula(self) -> Expr:
        d = self.d
        s = 1 - 0.5 / (math.sqrt(d + 20) - 4.1)
        mu1 = -math.sqrt((self.mu0**2 - self.depth) / s)
        signs = np.sign(self.x_opt).tolist()
        x_hat = [
            2 * sg * xi for sg, xi in zip(signs, variables(d), strict=True)
        ]
        near = total((xh - self.mu0) ** 2 for xh in x_hat)
        far = self.depth * d + s * total((xh - mu1) ** 2 for xh in x_hat)
        scaling = bbob.scaling(self.conditioning, d)
        linear = bbob.linear_map(self.q, scaling, self.r)
        z = sym.rotated(linear, [xh - self.mu0 for xh in x_hat])
        waves = self.a * (d - total(cos(2 * pi() * zi) for zi in z))
        boundary = self.penalty * sym.penalty(variables(d))
        return minimum((near, far)) + waves + boundary + self.f_opt
