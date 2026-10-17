"""The pydantic base: variants, the stored domain and read-only minima."""

import numpy as np
import pytest

from optbench import Branin, Sphere


def test_replace_makes_a_checked_variant() -> None:
    problem = Branin()
    variant = problem.replace(s=5.0)
    assert variant.s == pytest.approx(5.0)
    assert problem.s != variant.s
    assert type(variant) is Branin
    with pytest.raises(ValueError, match="d must be at least 1"):
        Sphere(d=3).replace(d=0)


def test_a_domain_from_an_array_equals_one_from_tuples() -> None:
    rows = ((0.0, 1.0), (2.0, 3.0))
    from_array = Sphere.model_validate({"d": 2, "domain": np.array(rows)})
    from_lists = Sphere.model_validate({"d": 2, "domain": [[0, 1], [2, 3]]})
    assert from_array.domain == rows
    assert from_array == from_lists == Sphere(d=2, domain=rows)
    assert hash(from_array) == hash(from_lists)
    np.testing.assert_array_equal(from_array.bounds, np.array(rows))


def test_a_domain_must_be_rows() -> None:
    with pytest.raises(ValueError, match=r"domain must be rows .* \(2,\)"):
        Sphere.model_validate({"d": 2, "domain": [0.0, 1.0]})


def test_minimum_points_are_read_only() -> None:
    minimum = Sphere(d=3).minimum
    assert minimum is not None
    with pytest.raises(ValueError, match="read-only"):
        minimum.points[0, 0] = 9.0


def test_minima_compare_by_identity() -> None:
    problem = Sphere(d=2)
    first, second = problem.minimum, problem.minimum
    assert first is not None
    same = first
    assert first == same
    assert first != second
    assert hash(first) != hash(second)
