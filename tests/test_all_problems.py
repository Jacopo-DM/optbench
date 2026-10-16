"""Checks every registered problem must pass (spec criterion 6)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import optbench
from optbench import Problem

SAMPLES = 10_000


def _problems() -> list[Problem]:
    """Every registered problem at its defaults, free dimensions also at 5.

    Returns
    -------
    list of Problem
        The problems to check.
    """
    problems = [optbench.make(name) for name in optbench.names()]
    free = [p for p in problems if hasattr(p, "d")]
    problems += [dataclasses.replace(p, d=5) for p in free]
    return problems


PROBLEMS = _problems()
IDS = [f"{p.name}-d{p.dimension}" for p in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_value_at_every_minimiser_is_the_minimum(problem: Problem) -> None:
    minimum = problem.minimum
    assert minimum is not None
    assert minimum.points.shape[1] == problem.dimension
    np.testing.assert_allclose(
        problem(minimum.points),
        minimum.value,
        rtol=max(minimum.tolerance, 1e-15),
        atol=1e-12,
    )


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_no_random_point_is_below_the_minimum(problem: Problem) -> None:
    minimum = problem.minimum
    assert minimum is not None
    bounds = problem.bounds
    rng = np.random.default_rng(0)
    size = (SAMPLES, problem.dimension)
    x = rng.uniform(bounds[:, 0], bounds[:, 1], size=size)
    slack = 1e-9 * max(1.0, abs(minimum.value))
    assert np.all(problem(x) >= minimum.value - slack)


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_a_batch_equals_one_point_at_a_time(problem: Problem) -> None:
    bounds = problem.bounds
    rng = np.random.default_rng(1)
    x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(20, problem.dimension))
    one_by_one = np.array([problem(point[np.newaxis])[0] for point in x])
    batch = problem(x)
    assert batch.shape == (20,)
    assert batch.dtype == np.float64
    np.testing.assert_allclose(batch, one_by_one, rtol=1e-14, atol=0)


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_wrong_shapes_are_refused(problem: Problem) -> None:
    d = problem.dimension
    for bad in (np.zeros(d), np.zeros((3, d + 1)), np.zeros((1, 1, d))):
        with pytest.raises(ValueError, match="expects points shaped"):
            problem(bad)


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_the_metadata_is_complete(problem: Problem) -> None:
    assert problem.url.startswith("https://www.sfu.ca/~ssurjano/")
    assert problem.references
    assert problem.group in optbench.Group
    assert problem.bounds.shape == (problem.dimension, 2)
