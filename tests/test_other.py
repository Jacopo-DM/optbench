"""The "other" group (apart from Branin and Shekel): values, parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from optbench import (
    Beale,
    Colville,
    Forrester2008,
    GoldsteinPrice,
    Hartmann3,
    Hartmann4,
    Hartmann6,
    PermDBeta,
    Powell,
    StyblinskiTang,
)

if TYPE_CHECKING:
    from optbench import Problem

# SFU's published minimum value and minimiser, rounded as printed there.
PUBLISHED = [
    (Beale(), 0.0, (3.0, 0.5)),
    (Colville(), 0.0, (1.0, 1.0, 1.0, 1.0)),
    (GoldsteinPrice(), 3.0, (0.0, -1.0)),
    (Hartmann3(), -3.86278, (0.114614, 0.555649, 0.852547)),
    (
        Hartmann6(),
        -3.32237,
        (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    ),
    (PermDBeta(d=3), 0.0, (1.0, 2.0, 3.0)),
    (Powell(d=8), 0.0, (0.0,) * 8),
]


@pytest.mark.parametrize(
    ("problem", "value", "point"),
    PUBLISHED,
    ids=[p.name for p, _, _ in PUBLISHED],
)
def test_minimum_agrees_with_the_sfu_page(
    problem: Problem,
    value: float,
    point: tuple[float, ...],
) -> None:
    minimum = problem.minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(value, abs=1e-4)
    np.testing.assert_allclose(minimum.points[0], point, atol=1e-4)


def test_published_constants() -> None:
    assert Beale().c == (1.5, 2.25, 2.625)
    colville = Colville()
    assert (colville.a, colville.b, colville.c, colville.e) == (
        100.0,
        90.0,
        10.1,
        19.8,
    )
    forrester = Forrester2008()
    assert (forrester.a, forrester.b, forrester.c, forrester.e) == (
        6.0,
        2.0,
        12.0,
        4.0,
    )
    goldstein = GoldsteinPrice()
    assert (goldstein.a, goldstein.b) == (1.0, 30.0)
    assert goldstein.p == (19.0, -14.0, 3.0, -14.0, 6.0, 3.0)
    assert goldstein.q == (18.0, -32.0, 12.0, 48.0, -36.0, 27.0)
    assert Hartmann3().alpha == (1.0, 1.2, 3.0, 3.2)
    assert Hartmann3().p[3] == (0.0381, 0.5743, 0.8828)
    assert Hartmann6().a[3] == (17.0, 8.0, 0.05, 10.0, 0.1, 14.0)
    four = Hartmann4()
    assert (four.offset, four.scale) == (1.1, 0.839)
    assert four.a == tuple(row[:4] for row in Hartmann6().a)
    assert PermDBeta().beta == pytest.approx(0.5)
    powell = Powell()
    assert (powell.a, powell.b, powell.c, powell.e) == (10.0, 5.0, 2.0, 10.0)
    tang = StyblinskiTang()
    assert (tang.scale, tang.a, tang.b) == (0.5, 16.0, 5.0)


def test_styblinski_tang_minimum_is_exact_and_the_sfu_value_rounded() -> None:
    # SFU: -39.16599 d at x_i = -2.903534. The exact per-coordinate minimum
    # is -39.1661657..., so the printed value is 1.7e-4 too high.
    minimum = StyblinskiTang(d=3).minimum
    assert minimum is not None
    np.testing.assert_allclose(minimum.points[0], -2.903534, atol=1e-6)
    assert minimum.value / 3 == pytest.approx(-39.16616570377142, abs=1e-10)
    assert minimum.value / 3 == pytest.approx(-39.16599, abs=2e-4)


def test_goldstein_price_on_a_hand_worked_point() -> None:
    # At the origin: (1 + 1 * 19) (30 + 0) = 600.
    assert GoldsteinPrice()(np.zeros((1, 2)))[0] == pytest.approx(600.0)


def test_forrester_on_a_hand_worked_point() -> None:
    # At x = 0.5: (3 - 2)^2 sin(6 - 4) = sin(2).
    value = Forrester2008()(np.array([[0.5]]))[0]
    assert value == pytest.approx(np.sin(2.0))


VARIANTS = [
    Beale(c=(2.0, 3.0, 3.5)),  # x2 = 0.5, x1 = 4: the third holds too
    Beale(c=(1.0, 1.0, 1.0)),  # x2 = 0, x1 = 1
    Colville(a=1.0, b=2.0, c=1.0, e=-2.0),
    PermDBeta(d=4, beta=3.0),
    Powell(d=4, a=1.0, c=0.5),
    StyblinskiTang(d=3, a=4.0, b=-2.0, scale=2.0),
    StyblinskiTang(d=2, domain=[[0.0, 1.0], [-1.0, 1.0]]),
]


@pytest.mark.parametrize("problem", VARIANTS, ids=repr)
def test_minimum_follows_the_parameters(problem: Problem) -> None:
    minimum = problem.minimum
    assert minimum is not None
    values = problem(minimum.points)
    np.testing.assert_allclose(values, minimum.value, rtol=1e-12, atol=1e-12)
    bounds = problem.bounds
    size = (50_000, problem.dimension)
    x = np.random.default_rng(9).uniform(bounds[:, 0], bounds[:, 1], size=size)
    assert np.all(problem(x) >= minimum.value - 1e-9)


@pytest.mark.parametrize(
    "problem",
    [
        Beale(c=(0.0, 1.0, 1.0)),  # c_1 = 0
        Beale(c=(1.0, 2.0, 3.0)),  # x2 would be 1
        Beale(c=(1.0, 1.5, 2.0)),  # the third bracket cannot vanish too
        Colville(c=1.0),
        Forrester2008(a=5.0),
        GoldsteinPrice(b=20.0),
        Hartmann3(alpha=(1.0, 1.0, 1.0, 1.0)),
        Hartmann4(scale=1.0),
        Hartmann6(domain=[[0.0, 0.5]] * 6),
        Powell(b=-1.0),
        StyblinskiTang(scale=-0.5),
    ],
    ids=repr,
)
def test_minimum_is_unknown_where_nothing_covers_it(problem: Problem) -> None:
    assert problem.minimum is None


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Powell(d=6), "multiple of 4"),
        (lambda: Powell(d=0), "at least 4"),
        (lambda: GoldsteinPrice(p=(1.0, 2.0)), "six coefficients"),
        (lambda: Hartmann3(a=((1.0, 2.0, 3.0),)), "one row per alpha"),
        (lambda: Hartmann3(p=((0.5,) * 2,) * 4), "one row per alpha"),
    ],
)
def test_bad_parameters_are_refused(make: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make()  # pyright: ignore[reportCallIssue]
