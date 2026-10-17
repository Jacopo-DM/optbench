"""BBOB functions f20 to f24: multimodal with weak global structure.

From Hansen, Finck, Ros and Auger (2009), INRIA RR-6829, section 5, with
the instances of COCO (see ``bbob_common``). Where COCO's code and the
report differ, the code wins, so values match COCO's published data; each
case is said in the class. ``r`` and ``q`` are COCO's two rotations (see
``BbobRotated``).
"""

import functools
import math
from typing import ClassVar, override

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from optbench.model import Model
from optbench.problem import Group, Minimum, formula_minimum

from . import bbob_common as bbob

_GROUP = Group.BBOB_MULTIMODAL_WEAK

# COCO's x_opt for f20: half of 4.2096874637 (z = 420.96874637 is where
# z sin(sqrt|z|) peaks), and its offset, that peak to 16 digits.
_SCHWEFEL_HALF_PEAK = 0.5 * 4.2096874637
_SCHWEFEL_OFFSET = 418.9828872724339

# COCO flips a coordinate's sign where its uniform number is below this.
_FLIP_BELOW = 0.5


class BbobSchwefel(bbob.BbobProblem):
    """BBOB f20, Schwefel x sin(x).

    With ``x_hat = 2 sign(x_opt) x``, ``z_hat_1 = x_hat_1`` and
    ``z_hat_i = x_hat_i + coupling (x_hat_{i-1} - 2 |x_opt_{i-1}|)``,
    ``z = 100 (Lambda^conditioning (z_hat - 2 |x_opt|) + 2 |x_opt|)`` and
    ``f(x) = 0.01 (offset - mean_i z_i sin(sqrt|z_i|) + f_pen500(z))
    + f_opt``, where ``f_pen500`` is the squared distance outside
    [-500, 500]^d. ``x_opt`` is ``+-4.2096874637 / 2`` in each coordinate,
    the sign drawn by COCO's generator.

    Attributes
    ----------
    conditioning, coupling, offset
        The published 10, 1/4 and 418.9828872724339.
    """

    name: ClassVar[str] = "bbob_f20_schwefel"
    group: ClassVar[Group] = _GROUP
    function: ClassVar[int] = 20

    conditioning: float = 10.0
    coupling: float = 0.25
    offset: float = _SCHWEFEL_OFFSET

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """``+-4.2096874637 / 2``, the signs from COCO's uniform numbers."""
        flip = bbob.uniform(self.d, self.seed) < _FLIP_BELOW
        signs = np.where(flip, -1.0, 1.0)
        return signs * _SCHWEFEL_HALF_PEAK

    @property
    @override
    def minimum(self) -> Minimum | None:
        """At ``x_opt``, where every ``z_i`` is 420.96874637.

        That is where ``z sin(sqrt|z|)`` peaks (to 1e-8 in ``z``, so to
        about 1e-17 in value) for any positive conditioning and coupling.
        """
        if self.conditioning <= 0:
            return None
        peak = 2 * _SCHWEFEL_HALF_PEAK * 100
        value = 0.01 * (self.offset - peak * math.sin(math.sqrt(peak)))
        points = self.x_opt[np.newaxis]
        return formula_minimum(self, value + self.f_opt, points)

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        corner = self.x_opt
        x_hat = 2 * np.sign(corner) * x
        twice = 2 * np.abs(corner)
        z_hat = x_hat.copy()
        z_hat[:, 1:] += self.coupling * (x_hat[:, :-1] - twice[:-1])
        scaling = bbob.scaling(self.conditioning, self.d)
        z = 100 * (scaling * (z_hat - twice) + twice)
        outside = np.maximum(0.0, np.abs(z) - 500)
        boundary = np.einsum("ij,ij->i", outside, outside)
        waves = np.mean(z * np.sin(np.sqrt(np.abs(z))), axis=1)
        result: NDArray[np.float64] = (
            0.01 * (boundary + self.offset - waves) + self.f_opt
        )
        return result


class _Peaks(Model):
    """Gallagher's peaks for one instance and dimension."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x_opt: NDArray[np.float64]
    centres: NDArray[np.float64]
    weights: NDArray[np.float64]
    log_heights: NDArray[np.float64]
    constants: NDArray[np.float64]


class _Shape(Model):
    """What sets f21 and f22 apart.

    Attributes
    ----------
    peaks
        Number of peaks.
    spread, centre
        Centres are ``spread u - centre`` for COCO's uniform ``u``.
    top_condition
        The highest peak's condition number.
    """

    peaks: int
    spread: float
    centre: float
    top_condition: float


@functools.cache
def _gallagher_peaks(seed: int, d: int, kind: "type[_Gallagher]") -> _Peaks:
    """Build Gallagher's peaks as COCO's ``f_gallagher_bbob_problem_allocate``.

    Each peak ``p`` has a height (10 for the global one, then 1.1 to 9.1),
    a centre (rotated by ``r``) and per-coordinate weights, a
    condition number ``c_p`` raised to a random permutation of
    ``i / (d - 1) - 1/2``. The cache key is the class (f21 or f22), whose
    ``shape`` is fixed, not the ``_Shape`` itself: pydantic adds a frozen
    model's ``__hash__`` only at run time, so type checkers would refuse it.

    Returns
    -------
    _Peaks
        ``x_opt``; the rotated centres ``(peaks, d)``; the weights
        ``(peaks, 2 d)`` and constants ``(peaks,)`` with which
        ``sum_j w_j (t_j - centre_j)^2`` is one product with
        ``[t^2, t]``; and the log heights.
    """
    shape = kind.shape
    rotation = bbob.rotation(seed, d)

    # Condition numbers: the global peak's, then 1000 to the power of a
    # random permutation of 0 .. 1.
    order = np.argsort(bbob.uniform(shape.peaks - 1, seed), kind="stable")
    conditions = np.concatenate([
        [shape.top_condition],
        1000.0 ** (order / (shape.peaks - 2)),
    ])
    heights = np.concatenate([
        [10.0],
        1.1 + 8.0 * np.arange(shape.peaks - 1) / (shape.peaks - 2),
    ])

    # Each peak's scales: its condition number to a permuted ramp.
    scales = np.empty((shape.peaks, d))
    for p in range(shape.peaks):
        ramp = np.argsort(bbob.uniform(d, seed + 1000 * p), kind="stable")
        scales[p] = conditions[p] ** (ramp / (d - 1) - 0.5)

    # Centres, uniform in [-centre, spread - centre], the first shrunk by
    # 0.8; that one, unrotated, is x_opt.
    draws = bbob.uniform(d * shape.peaks, seed).reshape(shape.peaks, d)
    raw = shape.spread * draws - shape.centre
    raw[0] *= 0.8
    centres = bbob.rotate(raw, rotation)
    weights = np.hstack([scales, -2 * scales * centres])
    constants = np.einsum("pj,pj->p", scales, centres * centres)
    return _Peaks(
        x_opt=raw[0].copy(),
        centres=centres,
        weights=weights,
        log_heights=np.log(heights),
        constants=constants,
    )


class _Gallagher(bbob.BbobRotated):
    """What f21 and f22 share: the peaks differ in count and spread.

    ``f(x) = T_osz(10 - max_p h_p exp(-sum_j w_pj (t_j - c_pj)^2 / (2 d)))^2
    + penalty f_pen(x) + f_opt``, with ``t = r x``.

    Attributes
    ----------
    penalty
        The published 1.
    """

    group: ClassVar[Group] = _GROUP
    shape: ClassVar[_Shape]

    penalty: float = 1.0

    @property
    def _peaks(self) -> _Peaks:
        return _gallagher_peaks(self.seed, self.d, type(self))

    @property
    def x_opt(self) -> NDArray[np.float64]:
        """The centre of the highest peak, before rotation."""
        return self._peaks.x_opt

    @property
    @override
    def minimum(self) -> Minimum | None:
        """``f_opt`` at ``x_opt``: only the highest peak reaches 10."""
        return super().minimum if self.penalty >= 0 else None

    @override
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        peaks = self._peaks
        t = bbob.rotate(x, self.r)
        features = np.hstack([t * t, t])
        squares: NDArray[np.float64] = np.einsum(
            "nk,pk->np",
            features,
            peaks.weights,
        )
        squares += peaks.constants
        exponents = peaks.log_heights - squares / (2 * self.d)
        highest = np.exp(np.max(exponents, axis=1))
        core = bbob.oscillate(10 - highest) ** 2
        return core + self.penalty * bbob.penalty(x) + self.f_opt


class BbobGallagher101(_Gallagher):
    """BBOB f21, Gallagher's Gaussian 101-me peaks.

    101 peaks, centres in [-4, 4]^d (times 0.8 for the highest), the
    highest with condition number sqrt(1000).
    """

    name: ClassVar[str] = "bbob_f21_gallagher_101"
    function: ClassVar[int] = 21
    shape: ClassVar[_Shape] = _Shape(
        peaks=101,
        spread=10.0,
        centre=5.0,
        top_condition=math.sqrt(1000.0),
    )


class BbobGallagher21(_Gallagher):
    """BBOB f22, Gallagher's Gaussian 21-hi peaks.

    21 peaks, centres in [-3.92, 3.92]^d (times 0.8 for the highest), the
    highest with condition number 1000.
    """

    name: ClassVar[str] = "bbob_f22_gallagher_21"
    function: ClassVar[int] = 22
    shape: ClassVar[_Shape] = _Shape(
        peaks=21,
        spread=9.8,
        centre=4.9,
        top_condition=1000.0,
    )


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
        linear = self.q @ (scaling[:, np.newaxis] * self.r)
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
        linear = self.q @ (scaling[:, np.newaxis] * self.r)
        z = bbob.rotate(x_hat - self.mu0, linear)
        waves = self.a * (d - np.sum(np.cos(2 * math.pi * z), axis=1))
        boundary = self.penalty * bbob.penalty(x)
        return np.minimum(near, far) + waves + boundary + self.f_opt
