"""The random-search baseline."""

from __future__ import annotations

import numpy as np
import pytest

from optibench import Branin, Sphere, random_search


def test_uses_exactly_the_budget() -> None:
    budget = 2_500
    tracked = random_search(Sphere(d=3), budget, np.random.default_rng(0))
    assert tracked.evaluations == budget

    # Batches of 1,000 and a last one of 500.
    np.testing.assert_array_equal(tracked.curve[:, 0], [1_000, 2_000, 2_500])


def test_best_is_the_lowest_of_the_points_it_drew() -> None:
    problem = Branin()
    tracked = random_search(problem, 300, np.random.default_rng(1), batch=300)
    bounds = problem.bounds
    x = np.random.default_rng(1).uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(300, 2),
    )
    assert tracked.best_value == pytest.approx(problem(x).min())
    assert tracked.best_point is not None
    assert problem.contains(tracked.best_point)


def test_same_seed_same_result() -> None:
    problem = Sphere(d=5, domain=((0.5, 1.0),) * 5)
    first = random_search(problem, 1_000, np.random.default_rng(7), batch=64)
    again = random_search(problem, 1_000, np.random.default_rng(7), batch=64)
    np.testing.assert_array_equal(first.curve, again.curve)
    assert first.best_value >= 5 * 0.5**2


def test_budget_and_batch_must_be_positive() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="budget"):
        random_search(Sphere(), 0, rng)
    with pytest.raises(ValueError, match="batch"):
        random_search(Sphere(), 10, rng, batch=0)
