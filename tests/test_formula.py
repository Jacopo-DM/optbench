"""Every formula equals its problem's numpy values (module formula)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import optbench
from optbench import AnyDimension, Group, Problem
from optbench.functions.bbob_common import BbobProblem
from optbench.symbolic import piecewise

if TYPE_CHECKING:
    from numpy.typing import NDArray

# The (suite, group) pairs whose problems have a formula so far; each task
# of module formula adds its own.
COVERED = (
    ("bbob", Group.BBOB_SEPARABLE),
    ("classic", Group.BOWL),
    ("classic", Group.MANY_LOCAL_MINIMA),
    ("classic", Group.OTHER),
    ("classic", Group.VALLEY),
    ("sfu", Group.BOWL),
    ("sfu", Group.MANY_LOCAL_MINIMA),
    ("sfu", Group.OTHER),
    ("sfu", Group.PLATE),
    ("sfu", Group.STEEP),
    ("sfu", Group.VALLEY),
)
DIMENSIONS = range(1, 7)
# BBOB problems are checked at these COCO instances (spec criterion 2).
INSTANCES = range(1, 6)
RTOL = 1e-12
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
    """Every problem of ``COVERED``, any-d ones at each allowed d up to 6.

    BBOB problems are checked at each of ``INSTANCES`` too.

    Returns
    -------
    list of Problem
        The problems to check, at their default constants.
    """
    names = [
        name
        for suite, group in COVERED
        for name in optbench.names(group=group, suite=suite)
    ]
    problems: list[Problem] = []
    for name in names:
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
    np.testing.assert_allclose(values, expected, rtol=RTOL, atol=ATOL)


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


def test_a_problem_without_a_formula_yet_says_so() -> None:
    with pytest.raises(NotImplementedError, match="no formula yet"):
        optbench.make("bbob_f24_lunacek").formula()


def test_a_piecewise_inside_a_condition_is_refused() -> None:
    # Turning this into numpy code made sympy grow to 160 GB (BBOB f3).
    (x1,) = optbench.variables(1)
    inner = piecewise((x1, x1 > 0), (-x1, True))
    outer = piecewise((x1, inner > 1), (0.0, True))
    with pytest.raises(ValueError, match="exhaust memory"):
        optbench.evaluate_formula(outer, np.zeros((2, 1)))
