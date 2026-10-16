"""The bowl-shaped group: published values, parameters, refusals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from optbench import (
    Bohachevsky1,
    Bohachevsky2,
    Bohachevsky3,
    Perm0,
    RotatedHyperEllipsoid,
    SumOfDifferentPowers,
    SumSquares,
    Trid,
)

if TYPE_CHECKING:
    from optbench import Problem


def test_published_constants() -> None:
    first = Bohachevsky1()
    assert (first.weight, first.a, first.b, first.p, first.q, first.offset) == (
        2.0,
        0.3,
        0.4,
        3.0,
        4.0,
        0.7,
    )
    for other in (Bohachevsky2(), Bohachevsky3()):
        assert (other.weight, other.a, other.offset) == (2.0, 0.3, 0.3)
    assert Perm0().beta == pytest.approx(10.0)
    assert SumOfDifferentPowers().power_shift == pytest.approx(1.0)
    assert Trid().shift == pytest.approx(1.0)


@pytest.mark.parametrize(("d", "value"), [(2, -2.0), (6, -50.0), (10, -210.0)])
def test_trid_matches_its_published_formula(d: int, value: float) -> None:
    # SFU: -d (d + 4) (d - 1) / 6 at x_i = i (d + 1 - i).
    minimum = Trid(d=d).minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(value)
    i = np.arange(1, d + 1)
    np.testing.assert_allclose(minimum.points[0], i * (d + 1 - i))


def test_the_hyper_ellipsoid_is_the_double_sum() -> None:
    x = np.random.default_rng(4).normal(size=(30, 6))
    double_sum = np.zeros(len(x))
    for i in range(6):
        double_sum += np.sum(x[:, : i + 1] ** 2, axis=1)
    np.testing.assert_allclose(RotatedHyperEllipsoid(d=6)(x), double_sum)


def test_sum_squares_and_powers_on_a_hand_worked_point() -> None:
    point = np.array([[1.0, -2.0, 0.5]])
    assert SumSquares(d=3)(point)[0] == pytest.approx(1 + 8 + 0.75)
    assert SumOfDifferentPowers(d=3)(point)[0] == pytest.approx(1 + 8 + 0.0625)


VARIANTS = [
    Bohachevsky1(a=0.5, b=0.1, offset=1.0, weight=1.0),
    Bohachevsky2(a=0.2, p=1.0),
    Bohachevsky3(weight=0.5, q=2.0),
    Perm0(d=3, beta=0.5),
    SumOfDifferentPowers(d=3, power_shift=0.5),
    Trid(d=4, shift=2.0),
    Trid(d=3, shift=-1.0),
]


@pytest.mark.parametrize("problem", VARIANTS, ids=repr)
def test_minimum_follows_the_parameters(problem: Problem) -> None:
    minimum = problem.minimum
    assert minimum is not None
    np.testing.assert_allclose(
        problem(minimum.points),
        minimum.value,
        rtol=1e-12,
        atol=1e-12,
    )
    bounds = problem.bounds
    size = (20_000, problem.dimension)
    x = np.random.default_rng(5).uniform(bounds[:, 0], bounds[:, 1], size=size)
    assert np.all(problem(x) >= minimum.value - 1e-9)


@pytest.mark.parametrize(
    "problem",
    [
        Bohachevsky1(b=-0.1),
        Bohachevsky2(weight=-1.0),
        Bohachevsky3(a=-0.3),
        SumOfDifferentPowers(power_shift=-1.0),
        Trid(d=3, shift=10.0),
        Perm0(d=2, domain=[[2.0, 3.0], [2.0, 3.0]]),
    ],
    ids=repr,
)
def test_minimum_is_unknown_where_no_formula_covers_it(
    problem: Problem,
) -> None:
    assert problem.minimum is None
