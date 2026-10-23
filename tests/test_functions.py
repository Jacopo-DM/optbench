"""Function-specific checks: published constants, minima, parameters."""

from __future__ import annotations

import math

import numpy as np
import pytest

from optibench import Branin, Shekel, Sphere


def test_sphere_has_its_minimum_for_any_d_and_none_off_the_domain() -> None:
    assert Sphere(d=1)(np.array([[3.0]]))[0] == pytest.approx(9.0)
    assert Sphere(d=7).minimum is not None
    assert Sphere(d=1, domain=((1.0, 2.0),)).minimum is None
    with pytest.raises(ValueError, match="at least 1"):
        Sphere(d=0)


def test_branin_defaults_are_the_published_constants() -> None:
    branin = Branin()
    assert (branin.a, branin.r, branin.s) == (1.0, 6.0, 10.0)
    assert branin.b == pytest.approx(5.1 / (4 * math.pi**2))
    assert branin.c == pytest.approx(5 / math.pi)
    assert branin.t == pytest.approx(1 / (8 * math.pi))


def test_branin_matches_the_published_minimum() -> None:
    # SFU: f = 0.397887 at (-pi, 12.275), (pi, 2.275) and (9.42478, 2.475).
    published = np.array(
        [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]],
    )
    branin = Branin()
    np.testing.assert_allclose(branin(published), 0.397887, atol=1e-5)
    minimum = branin.minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(0.397887, abs=1e-6)
    np.testing.assert_allclose(minimum.points, published, atol=1e-5)


def test_branin_minimum_follows_its_parameters() -> None:
    variant = Branin().replace(s=5.0, t=0.2, r=4.0)
    minimum = variant.minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(1.0)
    np.testing.assert_allclose(variant(minimum.points), 1.0)
    x = np.random.default_rng(2).uniform([-5, 0], [10, 15], size=(10_000, 2))
    assert np.all(variant(x) >= 1.0 - 1e-12)
    assert Branin(t=2.0).minimum is None


def test_branin_minimisers_follow_the_domain() -> None:
    minimum = Branin(domain=((0.0, 5.0), (0.0, 15.0))).minimum
    assert minimum is not None
    np.testing.assert_allclose(minimum.points, [[math.pi, 2.275]], atol=1e-5)
    assert Branin(domain=((0.0, 1.0), (0.0, 15.0))).minimum is None


def test_shekel_matches_the_published_minima() -> None:
    # SFU: m = 5, 7, 10 give -10.1532, -10.4029, -10.5364 at (4, 4, 4, 4).
    for m, published in ((5, -10.1532), (7, -10.4029), (10, -10.5364)):
        shekel = Shekel(m=m)
        minimum = shekel.minimum
        assert minimum is not None
        assert minimum.value == pytest.approx(published, abs=1e-4)
        at_four = shekel(np.full((1, 4), 4.0))[0]
        assert at_four == pytest.approx(published, abs=2e-4)


def test_shekel_minimum_is_unknown_for_changed_wells() -> None:
    assert Shekel(m=3).minimum is None
    assert Shekel(beta=(0.2,) * 10).minimum is None
    assert Shekel(domain=((0.0, 5.0),) * 4).minimum is None


def test_shekel_takes_other_wells_and_dimensions() -> None:
    shekel = Shekel(m=1, beta=(0.5,), c=((1.0,), (2.0,)))
    point = np.array([[1.0, 2.0]])
    assert shekel.dimension == point.shape[1]
    assert shekel(point)[0] == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"m": 11}, "m must be between"),
        ({"m": 0}, "m must be between"),
        ({"c": ()}, "c must be a matrix"),
    ],
)
def test_shekel_refuses_bad_parameters(
    params: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Shekel.model_validate(params)


def test_a_domain_must_fit_the_dimension() -> None:
    with pytest.raises(ValueError, match="domain must be"):
        Branin(domain=((0.0, 1.0),))
    with pytest.raises(ValueError, match="domain must be"):
        Branin(domain=((1.0, 0.0), (0.0, 1.0)))
