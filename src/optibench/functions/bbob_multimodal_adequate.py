"""BBOB functions f15 to f19: multimodal with adequate global structure.

From Hansen, Finck, Ros and Auger (2009), INRIA RR-6829, section 4, with
the instances of COCO (see ``bbob_common``). Where COCO's code and the
report differ, the code wins, so values match COCO's published data; each
case is said in the class. ``r`` and ``q`` are COCO's two rotations (see
``BbobRotated``); the report's ``R`` is ``q`` here and its ``Q`` is ``r``.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from optibench.problem import Group, Minimum
from optibench.symbolic import cos, pi, sin, sqrt, total, variables

from . import bbob_common as bbob
from . import bbob_symbolic as sym

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optibench.symbolic import Expr

_GROUP = Group.BBOB_MULTIMODAL_ADEQUATE


class BbobRastriginRotated(bbob.BbobRotated):
    """BBOB f15, rotated Rastrigin.

    ``f(x) = a (d - sum_i cos(2 pi z_i)) + |z|^2 + f_opt``, with
    ``z = q Lambda^conditioning r T_asy^beta(T_osz(q (x - x_opt)))``.

    Attributes
    ----------
    conditioning, beta, a
        The published 10, 0.2 and 10.
    """

    name: ClassVar[str] = "bbob_f15_rastrigin_rotated"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 15

    conditioning: float = 10.0
    beta: float = 0.2
    a: float = 10.0

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` while ``a >= 0`` and the scaling > 0."""
        if self.a < 0 or self.conditioning <= 0:
            return None
        return super().minimum

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        inner = bbob.oscillate(bbob.rotate(x - self.x_opt, self.q))
        inner = bbob.asymmetric(inner, self.beta)
        scaling = bbob.scaling(self.conditioning, self.d)
        linear = bbob.linear_map(self.q, scaling, self.r)
        return bbob.rastrigin(bbob.rotate(inner, linear), self.a) + self.f_opt

    @override
    def formula(self) -> Expr:
        rotated = sym.rotated(self.q, sym.shifted(self.x_opt))
        inner = sym.oscillate_asymmetric(rotated, self.beta)
        scaling = bbob.scaling(self.conditioning, self.d)
        linear = bbob.linear_map(self.q, scaling, self.r)
        return sym.rastrigin(sym.rotated(linear, inner), self.a) + self.f_opt


class BbobWeierstrass(bbob.BbobRotated):
    """BBOB f16, Weierstrass: rugged and repetitive.

    ``f(x) = scale (mean_i sum_k a^k cos(2 pi b^k (z_i + 1/2)) - f_0)^3
    + (penalty / d) f_pen(x) + f_opt``, for ``k = 0 .. terms - 1``, with
    ``f_0 = sum_k a^k cos(pi b^k)`` and
    ``z = q Lambda^conditioning r T_osz(q (x - x_opt))``.

    The minimum is reached wherever every ``z_i`` is an integer (for odd
    ``b``), not only at ``x_opt``; ``minimum`` lists ``x_opt`` only.

    Attributes
    ----------
    conditioning, a, b, terms, scale, penalty
        The published 1/100, 1/2, 3, 12, 10 and 10; ``b`` is an integer of
        at least 2.
    """

    name: ClassVar[str] = "bbob_f16_weierstrass"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 16

    conditioning: float = 0.01
    a: float = 0.5
    b: int = 3
    terms: int = 12
    scale: float = 10.0
    penalty: float = 10.0

    @override
    def _check(self) -> None:
        super()._check()
        if self.terms < 1:
            msg = f"terms must be at least 1, got {self.terms}"
            raise ValueError(msg)
        if self.b < 2:  # ruff: ignore[magic-value-comparison]
            msg = f"b must be an integer of at least 2, got {self.b}"
            raise ValueError(msg)

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` for the published ``a``, ``b`` and signs.

        With ``0 < a`` and an odd integer ``b``, every cosine is at least
        -1 and is -1 at ``z_i = 0``, so the mean is at least ``f_0``.
        """
        if self.b % 2 == 0 or self.a <= 0 or self.scale < 0 or self.penalty < 0:
            return None
        if self.conditioning <= 0:
            return None
        return super().minimum

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        inner = bbob.oscillate(bbob.rotate(x - self.x_opt, self.q))
        scaling = bbob.scaling(self.conditioning, self.d)
        linear = bbob.linear_map(self.q, scaling, self.r)
        z = bbob.rotate(inner, linear)

        # cos(2 pi b^k u) is the real part of w^(b^k), w = exp(2 pi i u):
        # raising to the power b each term needs b - 1 products instead of
        # a cosine, about twice as fast, and like the cosine it multiplies
        # the angle's rounding error by b per term.
        w: NDArray[np.complex128] = np.exp(2j * math.pi * (z + 0.5))
        waves: NDArray[np.float64] = np.zeros(z.shape)
        for k in range(self.terms):
            waves += self.a**k * np.real(w)

            # A new array before any in-place product, so w is not changed.
            power = w * w
            for _ in range(self.b - 2):
                power *= w
            w = power
        f0 = sum(
            self.a**k * math.cos(math.pi * self.b**k) for k in range(self.terms)
        )
        core = self.scale * (np.mean(waves, axis=1) - f0) ** 3
        boundary = self.penalty / self.d * bbob.penalty(x)
        return core + boundary + self.f_opt

    @override
    def formula(self) -> Expr:
        inner = sym.oscillate(sym.rotated(self.q, sym.shifted(self.x_opt)))
        scaling = bbob.scaling(self.conditioning, self.d)
        linear = bbob.linear_map(self.q, scaling, self.r)
        z = sym.rotated(linear, inner)
        waves = total(
            self.a**k * cos(2 * pi() * self.b**k * (zi + 0.5))
            for zi in z
            for k in range(self.terms)
        )
        f0 = sum(
            self.a**k * math.cos(math.pi * self.b**k) for k in range(self.terms)
        )
        core = self.scale * (waves / self.d - f0) ** 3
        boundary = self.penalty / self.d * sym.penalty(variables(self.d))
        return core + boundary + self.f_opt


class BbobSchaffersF7(bbob.BbobRotated):
    """BBOB f17, Schaffers F7.

    ``f(x) = (mean_{i<d} sqrt(s_i) (1 + sin^2(frequency s_i^(1/5))))^2
    + penalty f_pen(x) + f_opt``, with ``s_i = sqrt(z_i^2 + z_{i+1}^2)``
    and ``z = Lambda^conditioning r T_asy^beta(q (x - x_opt))``.

    Attributes
    ----------
    conditioning, beta, frequency, penalty
        The published 10, 0.5, 50 and 10.
    """

    name: ClassVar[str] = "bbob_f17_schaffers_f7"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 17

    conditioning: float = 10.0
    beta: float = 0.5
    frequency: float = 50.0
    penalty: float = 10.0

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` for a positive scaling and penalty >= 0."""
        if self.conditioning <= 0 or self.penalty < 0:
            return None
        return super().minimum

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        inner = bbob.asymmetric(bbob.rotate(x - self.x_opt, self.q), self.beta)
        scaling = bbob.scaling(self.conditioning, self.d)
        z = bbob.rotate(inner, scaling[:, np.newaxis] * self.r)
        squares = z[:, :-1] ** 2 + z[:, 1:] ** 2
        root = np.sqrt(np.sqrt(squares))
        ripple = np.sin(self.frequency * squares**0.1) ** 2
        core = np.mean(root * (1 + ripple), axis=1) ** 2
        return core + self.penalty * bbob.penalty(x) + self.f_opt

    @override
    def formula(self) -> Expr:
        rotated = sym.rotated(self.q, sym.shifted(self.x_opt))
        inner = sym.asymmetric(rotated, self.beta)
        scaling = bbob.scaling(self.conditioning, self.d)
        z = sym.rotated(scaling[:, np.newaxis] * self.r, inner)
        terms: list[Expr] = []
        for head, tail in itertools.pairwise(z):
            squares = head**2 + tail**2
            ripple = sin(self.frequency * squares**0.1) ** 2
            terms.append(sqrt(sqrt(squares)) * (1 + ripple))
        core = (total(terms) / (self.d - 1)) ** 2
        return core + self.penalty * sym.penalty(variables(self.d)) + self.f_opt


class BbobSchaffersF7Ill(BbobSchaffersF7):
    """BBOB f18, Schaffers F7 with ``conditioning`` 1000.

    The same formula as f17. As COCO does, its instance is built from f17's
    seeds (``x_opt``, the rotations and ``f_opt``), so f17 and f18 differ
    only in conditioning.
    """

    name: ClassVar[str] = "bbob_f18_schaffers_f7_ill"
    function: ClassVar[int] = 18

    conditioning: float = 1000.0

    @property
    def seed(self) -> int:
        """COCO builds f18 from function 17's seed."""
        return 17 + 10_000 * self.instance


class BbobGriewankRosenbrock(bbob.BbobRotated):
    """BBOB f19, composite Griewank-Rosenbrock F8F2.

    With ``t_i = a (z_i^2 - z_{i+1})^2 + (z_i - 1)^2``:
    ``f(x) = scale + scale mean_{i<d} (t_i / divisor - cos(t_i)) + f_opt``,
    and ``z = max(1, sqrt(d) / 8) r x + 1/2``. The minimum is at
    ``x_opt = r^T 1 / (2 max(1, sqrt(d) / 8))``, where ``z = 1``. (COCO
    reports ``r^T 1 / 2`` as its best point, which is the same for
    ``d <= 64``.)

    Attributes
    ----------
    a, divisor, scale
        The published 100, 4000 and 10.
    """

    name: ClassVar[str] = "bbob_f19_griewank_rosenbrock"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 19

    a: float = 100.0
    divisor: float = 4000.0
    scale: float = 10.0

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """``r^T 1 / (2 max(1, sqrt(d) / 8))``."""
        return self.r.sum(axis=0) / (2 * bbob.rosenbrock_scale(self.d))

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt`` while each term is at least -1.

        ``t / divisor - cos(t)`` is at least -1 for ``t >= 0`` when
        ``divisor > 0``, and -1 only at ``t = 0``.
        """
        if self.a < 0 or self.divisor <= 0 or self.scale < 0:
            return None
        return super().minimum

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = bbob.rosenbrock_scale(self.d) * bbob.rotate(x, self.r) + 0.5
        head, tail = z[:, :-1], z[:, 1:]
        t = self.a * (head**2 - tail) ** 2 + (head - 1) ** 2
        terms = np.mean(t / self.divisor - np.cos(t), axis=1)
        return self.scale + self.scale * terms + self.f_opt

    @override
    def formula(self) -> Expr:
        scale = bbob.rosenbrock_scale(self.d)
        z = [scale * zi + 0.5 for zi in sym.rotated(self.r, variables(self.d))]
        ts = [
            self.a * (head**2 - tail) ** 2 + (head - 1) ** 2
            for head, tail in itertools.pairwise(z)
        ]
        terms = total(t / self.divisor - cos(t) for t in ts) / (self.d - 1)
        return self.scale + self.scale * terms + self.f_opt
