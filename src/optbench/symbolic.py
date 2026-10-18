"""A typed view of the part of sympy that the formulas use.

sympy ships no type information (1.14 has no ``py.typed``), so this module
reaches it through Protocols with one ``cast``, the standard's pattern for
an untyped library, and nothing else in optbench imports sympy. sympy is
imported on first use, so ``import optbench`` stays fast.
"""

from __future__ import annotations

import functools
import importlib
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


class Expr(Protocol):
    """A sympy expression, with the arithmetic the formulas use."""

    def __add__(self, other: Expr | float, /) -> Expr: ...
    def __radd__(self, other: float, /) -> Expr: ...
    def __sub__(self, other: Expr | float, /) -> Expr: ...
    def __rsub__(self, other: float, /) -> Expr: ...
    def __mul__(self, other: Expr | float, /) -> Expr: ...
    def __rmul__(self, other: float, /) -> Expr: ...
    def __truediv__(self, other: Expr | float, /) -> Expr: ...
    def __rtruediv__(self, other: float, /) -> Expr: ...
    def __pow__(self, other: Expr | float, /) -> Expr: ...
    def __rpow__(self, other: float, /) -> Expr: ...
    def __neg__(self) -> Expr: ...
    def __lt__(self, other: Expr | float, /) -> Condition: ...


class Condition(Protocol):
    """A sympy relation such as ``x1 < 0.5``, for ``piecewise``."""


class _Lambdified(Protocol):
    """A formula turned into a numpy function: one array per variable."""

    def __call__(self, *columns: NDArray[np.float64]) -> object: ...


class _Sympy(Protocol):
    """The part of the ``sympy`` module this file calls."""

    pi: Expr

    def symbols(self, names: str, /, *, real: bool) -> tuple[Expr, ...]: ...
    def sin(self, x: Expr | float, /) -> Expr: ...
    def cos(self, x: Expr | float, /) -> Expr: ...
    def exp(self, x: Expr | float, /) -> Expr: ...
    def sqrt(self, x: Expr | float, /) -> Expr: ...
    def Abs(self, x: Expr | float, /) -> Expr: ...  # ruff: ignore[invalid-function-name]  # sympy's own name
    def Add(self, *terms: Expr | float) -> Expr: ...  # ruff: ignore[invalid-function-name]  # sympy's own name
    def Mul(self, *factors: Expr | float) -> Expr: ...  # ruff: ignore[invalid-function-name]  # sympy's own name
    def Max(self, *args: Expr | float) -> Expr: ...  # ruff: ignore[invalid-function-name]  # sympy's own name
    def Piecewise(  # ruff: ignore[invalid-function-name]  # sympy's own name
        self,
        *pieces: tuple[Expr | float, Condition | bool],
    ) -> Expr: ...
    def lambdify(
        self,
        args: tuple[Expr, ...],
        expr: Expr,
        /,
        *,
        modules: str,
    ) -> _Lambdified: ...


@functools.cache
def _sympy() -> _Sympy:
    """Import sympy once, typed as the part this module uses.

    Returns
    -------
    _Sympy
        The ``sympy`` module.
    """
    return cast("_Sympy", importlib.import_module("sympy"))


def variables(d: int) -> tuple[Expr, ...]:
    """Return the variables ``x1`` to ``xd`` that every formula uses.

    Column ``j`` of a batch of points is ``x(j+1)``, as the SFU pages
    number the variables.

    Returns
    -------
    tuple of Expr
        ``d`` real sympy symbols.
    """
    return _sympy().symbols(f"x1:{d + 1}", real=True)


def total(terms: Iterable[Expr | float]) -> Expr:
    """Return the sum of ``terms`` as one sympy expression.

    Returns
    -------
    Expr
        The sum.
    """
    return _sympy().Add(*terms)


def product(factors: Iterable[Expr | float]) -> Expr:
    """Return the product of ``factors`` as one sympy expression.

    Returns
    -------
    Expr
        The product.
    """
    return _sympy().Mul(*factors)


def pi() -> Expr:
    """Return sympy's exact pi.

    Returns
    -------
    Expr
        ``sympy.pi``.
    """
    return _sympy().pi


def sin(x: Expr | float) -> Expr:
    """Return ``sin(x)``.

    Returns
    -------
    Expr
        The sine.
    """
    return _sympy().sin(x)


def cos(x: Expr | float) -> Expr:
    """Return ``cos(x)``.

    Returns
    -------
    Expr
        The cosine.
    """
    return _sympy().cos(x)


def exp(x: Expr | float) -> Expr:
    """Return ``e^x``.

    Returns
    -------
    Expr
        The exponential.
    """
    return _sympy().exp(x)


def sqrt(x: Expr | float) -> Expr:
    """Return the square root of ``x``.

    Returns
    -------
    Expr
        The square root.
    """
    return _sympy().sqrt(x)


def absolute(x: Expr | float) -> Expr:
    """Return ``|x|``.

    Returns
    -------
    Expr
        The absolute value.
    """
    return _sympy().Abs(x)


def maximum(args: Iterable[Expr | float]) -> Expr:
    """Return the largest of ``args`` as one sympy expression.

    Returns
    -------
    Expr
        ``sympy.Max`` of the arguments.
    """
    return _sympy().Max(*args)


def piecewise(*pieces: tuple[Expr | float, Condition | bool]) -> Expr:
    """Return the first piece whose condition holds, as ``numpy.select``.

    Parameters
    ----------
    *pieces
        ``(value, condition)`` pairs; the last condition is usually ``True``.

    Returns
    -------
    Expr
        ``sympy.Piecewise`` of the pieces.
    """
    return _sympy().Piecewise(*pieces)


def evaluate_formula(
    formula: Expr,
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a formula at a batch of points with numpy.

    Parameters
    ----------
    formula
        An expression in ``variables(d)``.
    points
        Shaped ``(n, d)``; column ``j`` is ``x(j+1)``.

    Returns
    -------
    numpy.ndarray
        One value per point, shaped ``(n,)``.
    """
    n, d = points.shape
    function = _sympy().lambdify(variables(d), formula, modules="numpy")
    values = np.asarray(function(*points.T), dtype=np.float64)

    # A formula with no variable left gives one number; spread it out.
    return np.broadcast_to(values, (n,)).copy()
