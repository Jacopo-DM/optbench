"""The plate-shaped group: published values, parameters, refusals."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from optibench import Booth, Matyas, McCormick, PowerSum, Zakharov

if TYPE_CHECKING:
    from optibench import Problem


def test_published_constants() -> None:
    booth = Booth()
    assert (booth.weight, booth.a, booth.b) == (2.0, 7.0, 5.0)
    matyas = Matyas()
    assert (matyas.a, matyas.b) == (0.26, 0.48)
    mccormick = McCormick()
    assert (mccormick.a, mccormick.b, mccormick.offset) == (1.5, 2.5, 1.0)
    assert PowerSum().b == (8.0, 18.0, 44.0, 114.0)
    assert Zakharov().coefficient == pytest.approx(0.5)


def test_booth_matches_the_published_minimiser() -> None:
    # SFU: 0 at (1, 3).
    minimum = Booth().minimum
    assert minimum is not None
    np.testing.assert_allclose(minimum.points, [[1.0, 3.0]])


def test_mccormick_matches_the_published_minimum() -> None:
    # SFU: -1.9133 at (-0.54719, -1.54719).
    minimum = McCormick().minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(-1.9133, abs=1e-4)
    published = [[-0.54719, -1.54719]]
    np.testing.assert_allclose(minimum.points, published, atol=1e-5)
    shifted = McCormick(offset=3.0).minimum
    assert shifted is not None
    assert shifted.value == pytest.approx(minimum.value + 2)


def test_power_sum_minimisers_are_the_orderings_of_1_2_2_3() -> None:
    minimum = PowerSum().minimum
    assert minimum is not None
    expected = sorted(set(itertools.permutations((1.0, 2.0, 2.0, 3.0))))
    assert [tuple(p) for p in minimum.points.tolist()] == expected


def test_zakharov_on_a_hand_worked_point() -> None:
    # s = 0.5 (1 * 1 + 2 * 1) = 1.5; f = 2 + 1.5^2 + 1.5^4.
    value = Zakharov(d=2)(np.array([[1.0, 1.0]]))[0]
    assert value == pytest.approx(2 + 1.5**2 + 1.5**4)


VARIANTS = [
    Booth(weight=0.5, a=1.0, b=2.0),
    Matyas(a=1.0, b=-1.5),
    Zakharov(d=4, coefficient=-2.0),
    PowerSum(domain=((0.0, 4.0),) * 4),
]


@pytest.mark.parametrize("problem", VARIANTS, ids=repr)
def test_minimum_follows_the_parameters(problem: Problem) -> None:
    minimum = problem.minimum
    assert minimum is not None
    values = problem(minimum.points)
    np.testing.assert_allclose(values, minimum.value, atol=1e-12)
    bounds = problem.bounds
    size = (20_000, problem.dimension)
    x = np.random.default_rng(7).uniform(bounds[:, 0], bounds[:, 1], size=size)
    assert np.all(problem(x) >= minimum.value - 1e-9)


@pytest.mark.parametrize(
    "problem",
    [
        Booth(weight=1.0),
        Booth(a=100.0),  # the minimiser leaves the domain
        Matyas(a=0.1),
        McCormick(a=2.0),
        McCormick(domain=((0.0, 4.0), (-3.0, 4.0))),
        PowerSum(b=(1.0, 2.0, 3.0)),
    ],
    ids=repr,
)
def test_minimum_is_unknown_where_nothing_covers_it(problem: Problem) -> None:
    assert problem.minimum is None


def test_power_sum_takes_other_dimensions_and_refuses_none() -> None:
    problem = PowerSum(b=(3.0, 5.0))
    point = np.array([[1.0, 2.0]])
    assert problem.dimension == point.shape[1]
    assert problem(point)[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="at least one entry"):
        PowerSum(b=())
    assert math.isclose(PowerSum().bounds[0, 1], 4.0)
