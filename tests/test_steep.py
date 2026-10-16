"""The steep ridges and drops group: published values, parameters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from optbench import DeJong5, Easom, Michalewicz

if TYPE_CHECKING:
    from optbench import Problem


def test_published_constants() -> None:
    de_jong = DeJong5()
    assert (de_jong.offset, de_jong.power) == (0.002, 6.0)
    assert de_jong.a[0][:6] == (-32.0, -16.0, 0.0, 16.0, 32.0, -32.0)
    assert de_jong.a[1][:6] == (-32.0, -32.0, -32.0, -32.0, -32.0, -16.0)
    assert (de_jong.a[0][-1], de_jong.a[1][-1]) == (32.0, 32.0)
    assert Easom().centre == pytest.approx(math.pi)
    assert Michalewicz().m == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("d", "value"),
    [(2, -1.8013), (5, -4.687658), (10, -9.66015)],
)
def test_michalewicz_matches_the_published_minima(d: int, value: float) -> None:
    minimum = Michalewicz(d=d).minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(value, abs=1e-4)


def test_michalewicz_published_minimiser_for_two_variables() -> None:
    # SFU: at (2.20, 1.57).
    minimum = Michalewicz(d=2).minimum
    assert minimum is not None
    np.testing.assert_allclose(minimum.points, [[2.20, 1.57]], atol=5e-3)


def test_easom_matches_the_published_minimum() -> None:
    # SFU: -1 at (pi, pi).
    minimum = Easom().minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(-1.0)
    np.testing.assert_allclose(minimum.points, [[math.pi, math.pi]])


VARIANTS = [
    Michalewicz(d=3, m=2.5),
    Michalewicz(d=2, m=1.0, domain=[[1.0, 3.0], [0.5, 2.0]]),
    Easom(centre=2 * math.pi),
    Easom(centre=0.0),
]


@pytest.mark.parametrize("problem", VARIANTS, ids=repr)
def test_minimum_follows_the_parameters(problem: Problem) -> None:
    minimum = problem.minimum
    assert minimum is not None
    values = problem(minimum.points)
    np.testing.assert_allclose(values, minimum.value, atol=1e-12)
    bounds = problem.bounds
    size = (50_000, problem.dimension)
    x = np.random.default_rng(8).uniform(bounds[:, 0], bounds[:, 1], size=size)
    assert np.all(problem(x) >= minimum.value - 1e-9)


@pytest.mark.parametrize(
    "problem",
    [
        Easom(centre=1.0),
        Easom(centre=200.0),  # outside the domain
        Michalewicz(m=0.0),
        DeJong5(offset=0.01),
    ],
    ids=repr,
)
def test_minimum_is_unknown_where_nothing_covers_it(problem: Problem) -> None:
    assert problem.minimum is None


def test_de_jong5_refuses_misshapen_centres() -> None:
    with pytest.raises(ValueError, match="two rows of equal length"):
        DeJong5(a=((1.0, 2.0), (1.0,)))


@pytest.mark.parametrize(
    "domain",
    [
        [[1.0, 1.5]],  # the term rises across it: best at the right edge
        [[1.0, 2.1]],  # the peak (2.20) is just outside: Newton would leave
        [[0.0, 0.1]],  # the term is flat (below 1e-40): no curvature
    ],
)
def test_michalewicz_minimum_on_awkward_domains(
    domain: list[list[float]],
) -> None:
    problem = Michalewicz(d=1, domain=domain)
    minimum = problem.minimum
    assert minimum is not None
    grid = np.linspace(domain[0][0], domain[0][1], 200_001)[:, np.newaxis]
    assert minimum.value <= problem(grid).min() + 1e-15
