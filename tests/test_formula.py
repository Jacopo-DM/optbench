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
# BBOB problems are checked at every allowed d for instance 1, and at every
# one of INSTANCES for d = EVERY_INSTANCE_D (spec criterion 2, as the user
# amended it): the instance changes only x_opt, f_opt and the rotations,
# not the code path, and all 25 pairs made the suite take 9 minutes.
INSTANCES = range(1, 6)
EVERY_INSTANCE_D = 3
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

    BBOB problems are checked at instance 1, and at each of ``INSTANCES``
    for ``d = EVERY_INSTANCE_D``.

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
                if k == 1 or d == EVERY_INSTANCE_D
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


# Spec criterion 3: one problem per function file, with constants changed
# from the published ones (the spec's example is Ackley with a = 10).
CHANGED: dict[str, dict[str, float]] = {
    "ackley": {"a": 10.0},
    "bbob_f02_ellipsoid": {"conditioning": 1000.0},
    "bbob_f06_attractive_sector": {"sector": 50.0, "power": 0.8},
    "bbob_f10_ellipsoid_rotated": {"conditioning": 100.0},
    "bbob_f15_rastrigin_rotated": {"beta": 0.3, "a": 5.0},
    "bbob_f20_schwefel": {"coupling": 0.5},
    "bbob_f23_katsuura": {"conditioning": 10.0},
    "bohachevsky1": {"a": 0.5, "q": 3.0},
    "booth": {"a": 3.0},
    "branin": {"s": 5.0, "t": 0.2},
    "bukin6": {"scale": 50.0},
    "de_jong5": {"offset": 0.01},
    "deceptive": {"beta": 3.0},
    "dixon_price": {"weight": 3.0},
    "hartmann4": {"scale": 1.0},
    "holder_table": {"offset": 2.0},
    "levy": {"weight": 5.0},
    "perm0": {"beta": 5.0},
    "perm_d_beta": {"beta": 2.0},
    "salomon": {"a": 0.2},
}


def _assert_formula_matches(problem: Problem) -> None:
    """Check that ``problem``'s formula equals its numpy values."""
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


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_the_formula_equals_the_numpy_values(problem: Problem) -> None:
    _assert_formula_matches(problem)


@pytest.mark.parametrize(("name", "changes"), CHANGED.items())
def test_the_formula_follows_changed_constants(
    name: str,
    changes: dict[str, float],
) -> None:
    changed = optbench.make(name, **changes)
    assert str(changed.formula()) != str(optbench.make(name).formula())
    _assert_formula_matches(changed)


def test_every_function_file_has_a_changed_constants_check() -> None:
    files = {type(optbench.make(name)).__module__ for name in optbench.names()}
    checked = {type(optbench.make(name)).__module__ for name in CHANGED}
    assert checked == files


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
