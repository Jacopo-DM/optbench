"""The registry and the evaluation counter."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pydantic import ValidationError

import optibench
from optibench import Group, Sphere, Tracked

# Every (problem, parameter) pair, and values of every kind a caller might
# pass. Integers stay small: a large ``d`` would build huge arrays.
PARAMETERS = sorted(
    (name, field)
    for name in optibench.names()
    for field in type(optibench.make(name)).model_fields
)
VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-3, max_value=12),
    st.floats(),
    st.text(max_size=5),
    st.lists(st.floats(), max_size=4),
    st.lists(st.lists(st.floats(), max_size=3), max_size=4),
)


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


@given(st.sampled_from(PARAMETERS), VALUES)
@example(("branin", "s"), 5.0)
def test_make_builds_a_problem_or_raises_validation_error(
    parameter: tuple[str, str],
    value: object,
) -> None:
    """Any value for any parameter gives a problem or a ValidationError."""
    name, field = parameter
    try:
        problem = optibench.make(name, **{field: value})
    except ValidationError:
        return
    assert problem.name == name


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
