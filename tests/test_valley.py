"""The valley-shaped group: published values, parameters, refusals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from optbench import DixonPrice, Rosenbrock, SixHumpCamel, ThreeHumpCamel

if TYPE_CHECKING:
    from optbench import Problem


def test_published_constants() -> None:
    three = ThreeHumpCamel()
    assert (three.a, three.b) == (2.0, 1.05)
    assert three.c == pytest.approx(1 / 6)
    six = SixHumpCamel()
    assert (six.a, six.b, six.e, six.g) == (4.0, 2.1, 4.0, 4.0)
    assert six.c == pytest.approx(1 / 3)
    dixon = DixonPrice()
    assert (dixon.shift, dixon.weight) == (1.0, 2.0)
    rosenbrock = Rosenbrock()
    assert (rosenbrock.a, rosenbrock.b) == (100.0, 1.0)


def test_six_hump_camel_matches_the_published_minimum() -> None:
    # SFU: -1.0316 at (0.0898, -0.7126) and (-0.0898, 0.7126).
    minimum = SixHumpCamel().minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(-1.0316, abs=1e-4)
    np.testing.assert_allclose(
        minimum.points,
        [[0.0898, -0.7126], [-0.0898, 0.7126]],
        atol=1e-4,
    )


@pytest.mark.parametrize("d", [1, 2, 5, 8])
def test_dixon_price_matches_the_published_minimiser(d: int) -> None:
    # SFU: 0 at x_i = 2^(-(2^i - 2) / 2^i).
    minimum = DixonPrice(d=d).minimum
    assert minimum is not None
    i = np.arange(1, d + 1, dtype=np.float64)
    np.testing.assert_allclose(minimum.points[0], 2 ** (-(2**i - 2) / 2**i))


def test_dixon_price_has_two_minimisers_differing_in_the_last_sign() -> None:
    problem = DixonPrice(d=4)
    minimum = problem.minimum
    assert minimum is not None
    last_flipped = minimum.points[0] * np.array([1.0, 1.0, 1.0, -1.0])
    np.testing.assert_allclose(minimum.points[1], last_flipped)
    np.testing.assert_allclose(problem(minimum.points), 0.0, atol=1e-12)

    # Flipping any earlier coordinate is not a minimiser.
    flipped = minimum.points[:1] * np.array([1.0, -1.0, 1.0, 1.0])
    assert problem(flipped)[0] > 0.1
    assert len(DixonPrice(d=1).minimum.points) == 1  # pyright: ignore[reportOptionalMemberAccess]


def test_rosenbrock_on_a_hand_worked_point() -> None:
    # (x2 - x1^2) = -1 and (x1 - 1) = 0 for the first term; (x3 - x2^2) = 1
    # and (x2 - 1) = -1 for the second: 100 + 0 + 100 + 1.
    assert Rosenbrock(d=3)(np.array([[1.0, 0.0, 1.0]]))[0] == pytest.approx(201)


VARIANTS = [
    DixonPrice(d=4, shift=3.0, weight=0.5),
    Rosenbrock(d=2, a=10.0, b=1.5),
    Rosenbrock(d=4, a=5.0, b=0.0),
]


@pytest.mark.parametrize("problem", VARIANTS, ids=repr)
def test_minimum_follows_the_parameters(problem: Problem) -> None:
    minimum = problem.minimum
    assert minimum is not None
    values = problem(minimum.points)
    np.testing.assert_allclose(values, minimum.value, atol=1e-12)
    bounds = problem.bounds
    size = (20_000, problem.dimension)
    x = np.random.default_rng(6).uniform(bounds[:, 0], bounds[:, 1], size=size)
    assert np.all(problem(x) >= minimum.value - 1e-9)


@pytest.mark.parametrize(
    "problem",
    [
        DixonPrice(shift=-1.0),
        DixonPrice(weight=0.0),
        Rosenbrock(d=3, b=2.0),
        Rosenbrock(a=-1.0),
        Rosenbrock(d=2, b=4.0),  # the minimiser (4, 16) leaves the domain
        ThreeHumpCamel(c=0.2),
        SixHumpCamel(a=3.0),
    ],
    ids=repr,
)
def test_minimum_is_unknown_where_nothing_covers_it(problem: Problem) -> None:
    assert problem.minimum is None


def test_rosenbrock_needs_two_variables() -> None:
    with pytest.raises(ValueError, match="d must be at least 2"):
        Rosenbrock(d=1)
