"""The classic functions that are not on the SFU pages."""

from __future__ import annotations

import math

import numpy as np
import pytest

import optbench
from optbench import (
    AlpineN2,
    Bird,
    Deceptive,
    Group,
    HappyCat,
    Himmelblau,
    Minimum,
    Problem,
    Qing,
    Salomon,
    Schwefel221,
    Schwefel222,
    Whitley,
)


def _known(problem: Problem) -> Minimum:
    """Return the problem's minimum, which the test expects to be known.

    Returns
    -------
    Minimum
        The minimum.
    """
    minimum = problem.minimum
    assert minimum is not None
    return minimum


# The SFU pages' functions, which the classics must not repeat.
SFU_FUNCTIONS = 49

CLASSICS = [
    "alpine_n2",
    "bird",
    "deceptive",
    "happy_cat",
    "himmelblau",
    "qing",
    "salomon",
    "schwefel_2_21",
    "schwefel_2_22",
    "whitley",
]


def test_the_registry_lists_them_as_a_suite() -> None:
    assert optbench.names(suite="classic") == CLASSICS
    assert not set(CLASSICS) & set(optbench.names(suite="sfu"))
    assert len(optbench.names(suite="sfu")) == SFU_FUNCTIONS
    assert optbench.names(Group.BOWL, suite="classic") == [
        "schwefel_2_21",
        "schwefel_2_22",
    ]


def test_published_values() -> None:
    # Gavana's Bird minimisers give his value to 3e-8.
    gavana = np.array([[4.701055751981055, 3.152946019601391]])
    assert Bird()(gavana)[0] == pytest.approx(-106.7645367198034, abs=1e-9)

    # Himmelblau's minima as Wikipedia rounds them.
    rounded = [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186)]
    np.testing.assert_allclose(Himmelblau()(rounded), 0.0, atol=1e-9)
    assert Himmelblau()(np.zeros((1, 2)))[0] == pytest.approx(170.0)

    # Alpine N. 2 peaks near the published 7.917 with 2.808 per variable.
    assert AlpineN2.argmax == pytest.approx(7.917, abs=1e-3)
    per_variable = (-_known(AlpineN2(d=3)).value) ** (1 / 3)
    assert per_variable == pytest.approx(2.808, abs=1e-3)

    assert Salomon()(np.array([[3.0, 4.0]]))[0] == pytest.approx(0.5)
    assert Schwefel221(d=3)(np.array([[1.0, -5.0, 2.0]]))[0] == pytest.approx(
        5.0,
    )
    assert Schwefel222(d=3)(np.array([[1.0, -2.0, 3.0]]))[0] == pytest.approx(
        12.0,
    )
    assert Qing(d=2)(np.array([[1.0, 1.0]]))[0] == pytest.approx(1.0)
    assert HappyCat(d=2)(np.zeros((1, 2)))[0] == pytest.approx(2**0.25 + 0.5)
    assert Whitley(d=2)(np.zeros((1, 2)))[0] == pytest.approx(
        4 * (1 / 4000 - math.cos(1) + 1),
    )


def test_deceptive_pieces_join_and_peak_at_alpha() -> None:
    problem = Deceptive(d=1, alpha=(0.4,))
    knots = np.array([[0.0], [0.32], [0.4], [0.52], [1.0]])
    np.testing.assert_allclose(
        -problem(knots),
        np.array([0.8, 0.0, 1.0, 0.0, 0.8]) ** 2,
        atol=1e-12,
    )

    # Just either side of each knot, g barely moves: no jumps.
    for knot in (0.32, 0.4, 0.52):
        sides = np.array([[knot - 1e-9], [knot + 1e-9]])
        values = np.sqrt(-problem(sides))
        assert values[0] == pytest.approx(values[1], rel=0, abs=1e-7)

    default = Deceptive(d=3)
    np.testing.assert_allclose(_known(default).points, [[0.25, 0.5, 0.75]])


def test_minima_follow_the_parameters() -> None:
    assert Himmelblau(a=12.0).minimum is None
    assert Salomon(a=0.0).minimum is None
    assert _known(Salomon(a=0.5, frequency=1.0)).value == pytest.approx(0.0)
    assert HappyCat(alpha=0.0).minimum is None
    assert _known(HappyCat(d=3, alpha=0.25)).value == pytest.approx(0.0)
    assert Whitley(a=-1.0).minimum is None
    assert Whitley(divisor=0.0).minimum is None
    assert Deceptive(beta=0.0).minimum is None
    assert _known(Deceptive(d=2, alpha=(0.3, 0.6))).value == pytest.approx(-1)
    assert _known(Qing(d=3)).points.shape == (8, 3)


def test_minima_follow_the_domain() -> None:
    assert AlpineN2(d=2, domain=((0.0, 12.0),) * 2).minimum is None
    assert AlpineN2(d=2, domain=((7.0, 9.0),) * 2).minimum is not None
    assert Deceptive(d=2, domain=((-1.0, 1.0),) * 2).minimum is None
    assert Bird(domain=((0.0, 1.0),) * 2).minimum is None
    assert Schwefel221(d=2, domain=((1.0, 2.0),) * 2).minimum is None


def test_deceptive_refuses_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        Deceptive(d=2, alpha=(0.5,))
    with pytest.raises(ValueError, match="alpha"):
        Deceptive(d=2, alpha=(0.5, 1.0))
    with pytest.raises(ValueError, match="d must be"):
        Deceptive(d=0)
