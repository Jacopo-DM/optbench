"""The registry and the evaluation counter."""

from __future__ import annotations

import numpy as np
import pytest

import optbench
from optbench import Group, Sphere, Tracked


def test_names_lists_every_exported_problem_and_filters_by_group() -> None:
    exported = {
        getattr(optbench, name).name
        for name in optbench.__all__
        if isinstance(getattr(optbench, name), type)
        and issubclass(getattr(optbench, name), optbench.Problem)
        and hasattr(getattr(optbench, name), "name")
    }
    assert optbench.names() == sorted(exported)
    for group in Group:
        names = optbench.names(group)
        assert all(optbench.make(name).group == group for name in names)
    assert "sphere" in optbench.names(Group.BOWL)


def test_make_builds_with_changed_parameters() -> None:
    branin = optbench.make("branin", s=5.0)
    assert isinstance(branin, optbench.Branin)
    assert branin.s == pytest.approx(5.0)
    with pytest.raises(KeyError, match="no problem named"):
        optbench.make("nope")


def test_tracked_counts_and_keeps_the_best() -> None:
    tracked = Tracked(Sphere(d=2))
    tracked(np.array([[3.0, 0.0], [1.0, 1.0]]))
    tracked(np.array([[2.0, 2.0]]))
    tracked(np.array([[0.5, 0.0], [4.0, 0.0], [1.0, 0.0]]))
    assert tracked.evaluations == 6
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
