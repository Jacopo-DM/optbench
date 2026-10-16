"""What the BBOB functions share: instances and transformations.

From Hansen, Finck, Ros and Auger (2009), "Real-Parameter Black-Box
Optimization Benchmarking 2009: Noiseless Functions Definitions", INRIA
RR-6829, section 0. An instance's optimum ``x_opt``, its value ``f_opt`` and
its rotations come from COCO's own random generator and seeds (COCO's
``suite_bbob_legacy_code.c``, BSD-3-Clause), so instance ``k`` of a function
here is instance ``k`` in COCO and in its published results.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# COCO's generator: a Park-Miller "minimal standard" sequence shuffled
# through a table of 32 (Numerical Recipes' ran1).
_MULTIPLIER = 16807
_MODULUS = 2147483647
_QUOTIENT = 127773
_REMAINDER = 2836
_TABLE = 32
_WARM_UP = 40
_SHUFFLE = 67108865


def _next(seed: int) -> int:
    """Advance the Park-Miller sequence by one step (Schrage's method).

    Returns
    -------
    int
        The next value, between 1 and ``2^31 - 2``.
    """
    high = seed // _QUOTIENT
    seed = _MULTIPLIER * (seed - high * _QUOTIENT) - _REMAINDER * high
    return seed + _MODULUS if seed < 0 else seed


def uniform(n: int, seed: int) -> NDArray[np.float64]:
    """Return ``n`` uniform numbers in (0, 1), as COCO's ``bbob2009_unif``.

    Returns
    -------
    numpy.ndarray
        Shaped ``(n,)``; a zero is replaced by 1e-99, as COCO does.
    """
    state = max(abs(seed), 1)
    table = [0] * _TABLE
    for i in range(_WARM_UP - 1, -1, -1):
        state = _next(state)
        if i < _TABLE:
            table[i] = state
    pick = table[0]
    values = np.empty(n)
    for i in range(n):
        state = _next(state)
        slot = pick // _SHUFFLE
        pick = table[slot]
        table[slot] = state
        values[i] = pick / 2.147483647e9
    values[values == 0] = 1e-99
    return values


def gauss(n: int, seed: int) -> NDArray[np.float64]:
    """Return ``n`` standard normal numbers, as COCO's ``bbob2009_gauss``.

    Box-Muller on ``2 n`` uniform numbers: the first half gives the radius,
    the second the angle.

    Returns
    -------
    numpy.ndarray
        Shaped ``(n,)``; a zero is replaced by 1e-99, as COCO does.
    """
    u = uniform(2 * n, seed)
    values = np.sqrt(-2 * np.log(u[:n])) * np.cos(2 * math.pi * u[n:])
    values[values == 0] = 1e-99
    return values


@functools.cache
def x_opt(seed: int, d: int) -> NDArray[np.float64]:
    """Return an instance's optimum: uniform in [-4, 4], on a 8e-4 grid.

    Returns
    -------
    numpy.ndarray
        Shaped ``(d,)``, read-only (it is cached); a zero becomes -1e-5, as
        COCO does.
    """
    point = 8 * np.floor(1e4 * uniform(d, seed)) / 1e4 - 4
    point[point == 0] = -1e-5
    point.flags.writeable = False
    return point


@functools.cache
def f_opt(function: int, instance: int) -> float:
    """Return an instance's optimal value, as COCO's ``bbob2009_compute_fopt``.

    A ratio of two normal numbers (Cauchy distributed), rounded to two
    decimals and clipped to [-1000, 1000]. Functions 4 and 18 use the seeds
    of 3 and 17.

    Returns
    -------
    float
        The value.
    """
    base = {4: 3, 18: 17}.get(function, function)
    seed = base + 10_000 * instance
    ratio = gauss(1, seed)[0] / gauss(1, seed + 1)[0]
    rounded = math.floor(100 * 100 * ratio + 0.5) / 100
    return min(1000.0, max(-1000.0, rounded))


def oscillate(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """T_osz: a smooth, coordinate-wise oscillation about the identity.

    ``sign(x) exp(log|x| + 0.049 (sin(c1 log|x|) + sin(c2 log|x|)))``, with
    ``c1, c2 = 10, 7.9`` for positive ``x`` and ``5.5, 3.1`` otherwise; 0
    stays 0.

    Returns
    -------
    numpy.ndarray
        The same shape as ``x``.
    """
    size = np.abs(x)
    log = np.log(np.where(size > 0, size, 1.0))
    positive = x > 0
    c1 = np.where(positive, 10.0, 5.5)
    c2 = np.where(positive, 7.9, 3.1)
    wave = 0.049 * (np.sin(c1 * log) + np.sin(c2 * log))
    return np.sign(x) * np.exp(log + wave)


def asymmetric(x: NDArray[np.float64], beta: float) -> NDArray[np.float64]:
    """T_asy: raise positive coordinates to ``1 + beta i/(d-1) sqrt(x_i)``.

    Negative coordinates are unchanged; ``i`` counts from 0.

    Returns
    -------
    numpy.ndarray
        The same shape as ``x``.
    """
    d: int = x.shape[-1]
    ramp = np.arange(d, dtype=np.float64) / (d - 1)
    positive = np.where(x > 0, x, 0.0)
    raised = positive ** (1 + beta * ramp * np.sqrt(positive))
    return np.where(x > 0, raised, x)


def scaling(alpha: float, d: int) -> NDArray[np.float64]:
    """Return the diagonal of Lambda^alpha, ``alpha^(i / (2 (d - 1)))``.

    ``i`` counts from 0.

    Returns
    -------
    numpy.ndarray
        Shaped ``(d,)``.
    """
    return alpha ** (0.5 * np.arange(d) / (d - 1))


def penalty(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """f_pen: the squared distance outside [-5, 5]^d, per point.

    Returns
    -------
    numpy.ndarray
        Shaped ``(n,)``.
    """
    outside = np.maximum(0.0, np.abs(x) - 5)
    return np.einsum("ij,ij->i", outside, outside)
