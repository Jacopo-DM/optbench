"""Every formula equals its problem's numpy values (module formula)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import optbench
from optbench import AnyDimension, Problem
from optbench.functions.bbob_common import BbobProblem
from optbench.symbolic import piecewise

if TYPE_CHECKING:
    from numpy.typing import NDArray

DIMENSIONS = range(1, 7)
# BBOB problems are checked at these COCO instances (spec criterion 2).
INSTANCES = range(1, 6)
RTOL = 1e-12
# Spec criterion 2: these BBOB functions use test_bbob.py's looser 1e-9;
# they magnify the last bit of z (Weierstrass's 3^11, Katsuura's 2^32,
# Schaffers' high powers).
LOOSE = {
    "bbob_f16_weierstrass",
    "bbob_f17_schaffers_f7",
    "bbob_f18_schaffers_f7_ill",
    "bbob_f19_griewank_rosenbrock",
    "bbob_f23_katsuura",
}
LOOSE_RTOL = 1e-9
ATOL = 1e-10
INSIDE = 20
OUTSIDE = 5
# Points outside the domain reach this far past its edges, as a share of
# each side's width.
REACH = 0.3


def _allowed(problem: AnyDimension, d: int) -> bool:
    """Whether ``problem`` takes ``d`` variables (Powell needs multiples of 4).

    Returns
    -------
    bool
        False when ``replace(d=d)`` refuses ``d``.
    """
    try:
        problem.replace(d=d)
    except ValueError:
        return False
    return True


def _problems() -> list[Problem]:
    """Every registered problem, any-d ones at each allowed d up to 6.

    BBOB problems are checked at each of ``INSTANCES`` too.

    Returns
    -------
    list of Problem
        The problems to check, at their default constants.
    """
    problems: list[Problem] = []
    for name in optbench.names():
        problem = optbench.make(name)
        if isinstance(problem, BbobProblem):
            problems += [
                problem.replace(d=d, instance=k)
                for d in DIMENSIONS
                if _allowed(problem, d)
                for k in INSTANCES
            ]
        elif isinstance(problem, AnyDimension):
            problems += [
                problem.replace(d=d) for d in DIMENSIONS if _allowed(problem, d)
            ]
        else:
            problems.append(problem)
    return problems


def _points(problem: Problem) -> NDArray[np.float64]:
    """Points in the domain, a few outside it, and the optimum.

    Returns
    -------
    numpy.ndarray
        Shaped ``(n, d)``.
    """
    bounds = problem.bounds
    low, high = bounds[:, 0], bounds[:, 1]
    margin = REACH * (high - low)
    rng = np.random.default_rng(len(problem.name) * 100 + problem.dimension)
    inside = rng.uniform(low, high, size=(INSIDE, problem.dimension))
    outside = rng.uniform(
        low - margin,
        high + margin,
        size=(OUTSIDE, problem.dimension),
    )
    minimum = problem.minimum
    optimum = (
        minimum.points
        if minimum is not None
        else np.empty((0, problem.dimension))
    )
    return np.vstack([inside, outside, optimum])


PROBLEMS = _problems()
IDS = [
    f"{p.name}-d{p.dimension}"
    + (f"-i{p.instance}" if isinstance(p, BbobProblem) else "")
    for p in PROBLEMS
]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_the_formula_equals_the_numpy_values(problem: Problem) -> None:
    x = _points(problem)

    # Outside the domain some functions are undefined (Alpine N. 2 takes
    # sqrt(x) below 0); both sides then give NaN, which counts as equal.
    with np.errstate(invalid="ignore"):
        expected = problem(x)
        values = optbench.evaluate_formula(problem.formula(), x)
    # A BBOB value is core + f_opt, a random constant; compare the cores so
    # the error is measured against the function's own size, not against
    # what is left when f_opt happens to cancel it (f19, d = 5, instance 4:
    # 58.55 - 58.48).
    offset = problem.f_opt if isinstance(problem, BbobProblem) else 0.0
    rtol = LOOSE_RTOL if problem.name in LOOSE else RTOL
    np.testing.assert_allclose(
        values - offset,
        expected - offset,
        rtol=rtol,
        atol=ATOL,
    )


def test_the_variables_are_x1_to_xd() -> None:
    assert [str(v) for v in optbench.variables(3)] == ["x1", "x2", "x3"]
    assert [str(v) for v in optbench.variables(1)] == ["x1"]


def test_a_constant_formula_gives_one_value_per_point() -> None:
    # A formula with no variable left (here 0 * x1) still gives (n,) values.
    (x1,) = optbench.variables(1)
    values = optbench.evaluate_formula(0 * x1 + 2.5, np.zeros((4, 1)))
    np.testing.assert_array_equal(values, [2.5] * 4)


def test_booth_prints_as_the_sfu_page_writes_it() -> None:
    # The constants are float fields, so they print with a decimal point.
    assert str(optbench.make("booth").formula()) == (
        "(x1 + 2.0*x2 - 7.0)**2 + (2.0*x1 + x2 - 5.0)**2"
    )


def test_a_piecewise_inside_a_condition_is_refused() -> None:
    # Turning this into numpy code made sympy grow to 160 GB (BBOB f3).
    (x1,) = optbench.variables(1)
    inner = piecewise((x1, x1 > 0), (-x1, True))
    outer = piecewise((x1, inner > 1), (0.0, True))
    with pytest.raises(ValueError, match="exhaust memory"):
        optbench.evaluate_formula(outer, np.zeros((2, 1)))
