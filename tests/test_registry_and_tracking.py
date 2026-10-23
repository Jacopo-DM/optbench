"""The registry and the evaluation counter."""

from __future__ import annotations

import numpy as np
import pytest

import optibench
from optibench import Group, Sphere, Tracked


def test_names_lists_every_exported_problem_and_filters_by_group() -> None:
    exported = {
        getattr(optibench, name).name
        for name in optibench.__all__
        if isinstance(getattr(optibench, name), type)
        and issubclass(getattr(optibench, name), optibench.Problem)
        and hasattr(getattr(optibench, name), "name")
    }
    assert optibench.names() == sorted(exported)
    for group in Group:
        names = optibench.names(group)
        assert all(optibench.make(name).group == group for name in names)
    assert "sphere" in optibench.names(Group.BOWL)


def test_make_builds_with_changed_parameters() -> None:
    branin = optibench.make("branin", s=5.0)
    assert isinstance(branin, optibench.Branin)
    assert branin.s == pytest.approx(5.0)
    with pytest.raises(KeyError, match="no problem named"):
        optibench.make("nope")


def test_tracked_counts_and_keeps_the_best() -> None:
    tracked = Tracked(Sphere(d=2))
    batches = [
        np.array([[3.0, 0.0], [1.0, 1.0]]),
        np.array([[2.0, 2.0]]),
        np.array([[0.5, 0.0], [4.0, 0.0], [1.0, 0.0]]),
    ]
    for batch in batches:
        tracked(batch)
    assert tracked.evaluations == sum(len(batch) for batch in batches)
    assert tracked.best_value == pytest.approx(0.25)
    assert tracked.best_point is not None
    np.testing.assert_array_equal(tracked.best_point, [0.5, 0.0])
    np.testing.assert_allclose(tracked.curve, [[2, 2.0], [3, 2.0], [6, 0.25]])


def test_tracked_ignores_nan_and_starts_empty() -> None:
    tracked = Tracked(Sphere(d=1))
    assert tracked.curve.shape == (0, 2)
    tracked(np.array([[np.nan]]))
    assert tracked.best_point is None
    assert tracked.best_value == np.inf
    tracked(np.array([[np.nan], [2.0]]))
    assert tracked.best_value == pytest.approx(4.0)
