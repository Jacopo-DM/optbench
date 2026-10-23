"""Symbolic twins of the BBOB transformations in ``bbob_common``.

Each takes and returns one sympy expression per coordinate, so a BBOB
``formula()`` reads like its ``_evaluate``. They follow the numpy code's
order of operations, so the formula's values match to the tests'
tolerances. The twins live apart from ``bbob_common`` only to keep that
file under 400 lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optibench.symbolic import (
    absolute,
    cos,
    exp,
    log,
    maximum,
    pi,
    piecewise,
    sin,
    sqrt,
    total,
    variables,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from optibench.symbolic import Expr


def shifted(x_opt: NDArray[np.float64]) -> list[Expr]:
    """Return ``x - x_opt``, one expression per coordinate.

    Returns
    -------
    list of Expr
        ``x(j+1) - x_opt[j]`` for each ``j``.
    """
    return [
        xj - c
        for xj, c in zip(variables(len(x_opt)), x_opt.tolist(), strict=True)
    ]


def rotated(matrix: NDArray[np.float64], x: Sequence[Expr]) -> list[Expr]:
    """Return ``matrix`` applied to ``x``, as ``bbob_common.rotate``.

    Returns
    -------
    list of Expr
        ``z_i = sum_j matrix[i, j] x_j``, one per row.
    """
    return [
        total(m * xj for m, xj in zip(row, x, strict=True))
        for row in matrix.tolist()
    ]


def _wave(log_size: Expr, c1: float, c2: float) -> Expr:
    return log_size + 0.049 * (sin(c1 * log_size) + sin(c2 * log_size))


def oscillate_pieces(xi: Expr) -> tuple[Expr, Expr]:
    """T_osz of one coordinate, as its piece for ``xi > 0`` and ``xi < 0``.

    T_osz keeps the sign of its input (and 0 stays 0), so a transform
    applied after it can test ``xi`` itself. A piecewise inside another
    piecewise's condition makes sympy's printer fold the two, which ran out
    of memory for f3 at d = 2.

    Returns
    -------
    tuple of Expr
        The positive piece, then the negative one.
    """
    positive = exp(_wave(log(xi), 10.0, 7.9))
    negative = -exp(_wave(log(-xi), 5.5, 3.1))
    return positive, negative


def oscillate(x: Sequence[Expr]) -> list[Expr]:
    """T_osz, as ``bbob_common.oscillate``: 0 stays 0.

    Returns
    -------
    list of Expr
        One piecewise expression per coordinate.
    """
    result: list[Expr] = []
    for xi in x:
        positive, negative = oscillate_pieces(xi)
        result.append(
            piecewise((positive, xi > 0), (negative, xi < 0), (0.0, True)),
        )
    return result


def oscillate_asymmetric(x: Sequence[Expr], beta: float) -> list[Expr]:
    """T_asy applied to T_osz, as ``asymmetric(oscillate(x), beta)``.

    T_asy raises the positive coordinates to
    ``1 + beta i / (d - 1) sqrt(z_i)`` and leaves the others alone.

    Returns
    -------
    list of Expr
        One piecewise expression per coordinate, its conditions on ``x``.
    """
    d = len(x)
    result: list[Expr] = []
    for i, xi in enumerate(x):
        positive, negative = oscillate_pieces(xi)
        raised = positive ** (1 + beta * (i / (d - 1)) * sqrt(positive))
        result.append(
            piecewise((raised, xi > 0), (negative, xi < 0), (0.0, True)),
        )
    return result


def asymmetric(x: Sequence[Expr], beta: float) -> list[Expr]:
    """T_asy, as ``bbob_common.asymmetric``, for ``x`` with no piecewise.

    A piecewise ``x`` would put a piecewise in a condition (see
    ``oscillate_pieces``); after T_osz use ``oscillate_asymmetric``.

    Returns
    -------
    list of Expr
        Positive coordinates raised to ``1 + beta i / (d - 1) sqrt(x_i)``.
    """
    d = len(x)
    return [
        piecewise(
            (xi ** (1 + beta * (i / (d - 1)) * sqrt(xi)), xi > 0),
            (xi, True),
        )
        for i, xi in enumerate(x)
    ]


def penalty(x: Sequence[Expr]) -> Expr:
    """f_pen, as ``bbob_common.penalty``.

    Returns
    -------
    Expr
        The squared distance outside [-5, 5]^d.
    """
    return total(maximum((0.0, absolute(xi) - 5)) ** 2 for xi in x)


def rastrigin(z: Sequence[Expr], a: float) -> Expr:
    """Rastrigin's ``a (d - sum cos(2 pi z_i)) + |z|^2``.

    Returns
    -------
    Expr
        The value.
    """
    waves = total(cos(2 * pi() * zi) for zi in z)
    return a * (len(z) - waves) + total(zi**2 for zi in z)
