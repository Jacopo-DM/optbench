"""The "many local minima" group: published values, parameters, refusals."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

import optibench
from optibench import (
    Ackley,
    Bukin6,
    CrossInTray,
    DropWave,
    Eggholder,
    GramacyLee2012,
    Griewank,
    HolderTable,
    Langermann,
    Levy,
    LevyN13,
    Rastrigin,
    SchafferN2,
    SchafferN4,
    Schwefel,
    Shubert,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from optibench import Problem

# SFU's published minimum values (rounded as printed there), where it gives
# one, with the number of minimisers it lists.
# Shubert's published number of terms.
SHUBERT_TERMS = 5

PUBLISHED = [
    (Ackley(), 0.0, 1),
    (Bukin6(), 0.0, 1),
    (CrossInTray(), -2.06261, 4),
    (DropWave(), -1.0, 1),
    (Eggholder(), -959.6407, 1),
    (Griewank(), 0.0, 1),
    (HolderTable(), -19.2085, 4),
    (Levy(), 0.0, 1),
    (LevyN13(), 0.0, 1),
    (Rastrigin(), 0.0, 1),
    (SchafferN2(), 0.0, 1),
    (Schwefel(), 0.0, 1),
    (Shubert(), -186.7309, 18),
]


@pytest.mark.parametrize(
    ("problem", "value", "count"),
    PUBLISHED,
    ids=[p.name for p, _, _ in PUBLISHED],
)
def test_minimum_agrees_with_the_sfu_page(
    problem: Problem,
    value: float,
    count: int,
) -> None:
    minimum = problem.minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(value, abs=1e-4)
    assert len(minimum.points) == count


def test_published_constants() -> None:
    ackley = Ackley()
    assert (ackley.a, ackley.b, ackley.c) == (20.0, 0.2, 2 * math.pi)
    assert Griewank().divisor == pytest.approx(4000.0)
    levy = Levy()
    assert (levy.shift, levy.scale, levy.weight) == (1.0, 4.0, 10.0)
    assert Rastrigin().a == pytest.approx(10.0)
    assert Schwefel().offset == pytest.approx(418.9829)
    bukin = Bukin6()
    assert (bukin.scale, bukin.curvature, bukin.slope, bukin.shift) == (
        100.0,
        0.01,
        0.01,
        10.0,
    )
    cross = CrossInTray()
    assert (cross.scale, cross.offset, cross.power) == (0.0001, 100.0, 0.1)
    drop = DropWave()
    assert (drop.frequency, drop.curvature, drop.offset) == (12.0, 0.5, 2.0)
    egg = Eggholder()
    assert (egg.shift, egg.ratio) == (47.0, 0.5)
    gramacy = GramacyLee2012()
    assert (gramacy.frequency, gramacy.scale, gramacy.shift, gramacy.power) == (
        10.0,
        2.0,
        1.0,
        4.0,
    )
    assert HolderTable().offset == pytest.approx(1.0)
    levy13 = LevyN13()
    assert (levy13.a, levy13.b, levy13.c) == (3.0, 3.0, 2.0)
    assert SchafferN2().scale == pytest.approx(0.001)
    assert SchafferN4().scale == pytest.approx(0.001)
    assert Shubert().terms == SHUBERT_TERMS
    langermann = Langermann()
    assert langermann.c == (1.0, 2.0, 5.0, 2.0, 3.0)
    assert langermann.a == ((3, 5), (5, 2), (2, 1), (1, 4), (7, 9))


def test_schwefel_published_offset_leaves_a_small_positive_minimum() -> None:
    minimum = Schwefel(d=3).minimum
    assert minimum is not None
    assert minimum.value == pytest.approx(3 * 1.2727566e-5, rel=1e-4)
    np.testing.assert_allclose(minimum.points, 420.9687, atol=1e-4)


# Minima that follow the parameters: each variant's minimum is checked at
# its minimisers and against random points.
VARIANTS = [
    Ackley(d=3, a=5.0, b=1.0, c=1.0),
    Bukin6(shift=8.0, curvature=0.02),
    DropWave(offset=4.0, frequency=3.0),
    Griewank(d=3, divisor=100.0),
    Levy(d=3, shift=2.0, scale=2.0, weight=-0.5),
    LevyN13(a=4.0, b=1.5, c=0.5),
    Rastrigin(d=3, a=3.0),
    SchafferN2(scale=0.1),
    Schwefel(d=2, offset=500.0),
]


@pytest.mark.parametrize("problem", VARIANTS, ids=repr)
def test_minimum_follows_the_parameters(problem: Problem) -> None:
    minimum = problem.minimum
    assert minimum is not None
    np.testing.assert_allclose(
        problem(minimum.points),
        minimum.value,
        rtol=1e-12,
        atol=1e-12,
    )
    bounds = problem.bounds
    rng = np.random.default_rng(3)
    size = (20_000, problem.dimension)
    x = rng.uniform(bounds[:, 0], bounds[:, 1], size=size)
    assert np.all(problem(x) >= minimum.value - 1e-9)


@pytest.mark.parametrize(
    "problem",
    [
        Ackley(a=-1.0),
        Griewank(divisor=-1.0),
        Levy(weight=-2.0),
        Rastrigin(a=-1.0),
        Schwefel(domain=((0.0, 100.0),) * 2),
        Bukin6(scale=-1.0),
        Bukin6(shift=0.0),
        DropWave(offset=0.0),
        LevyN13(a=2.5),
        SchafferN2(scale=-1.0),
        CrossInTray(power=0.2),
        Shubert(terms=4),
        Langermann(m=4),
    ],
    ids=repr,
)
def test_minimum_is_unknown_where_no_formula_or_search_covers_it(
    problem: Problem,
) -> None:
    assert problem.minimum is None


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Levy(scale=0.0), "scale must not be zero"),
        (lambda: Langermann(m=6), "m must be between"),
        (lambda: Langermann(a=()), "a must be a matrix"),
        (lambda: Shubert(terms=0), "terms must be at least 1"),
        (lambda: Rastrigin(d=0), "d must be at least 1"),
    ],
)
def test_bad_parameters_are_refused(
    make: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make()


def test_langermann_takes_other_dimensions() -> None:
    problem = optibench.make("langermann", m=1, c=(2.0,), a=((1.0, 2.0, 3.0),))
    point = np.array([[1.0, 2.0, 3.0]])
    assert problem.dimension == point.shape[1]
    assert problem(point)[0] == pytest.approx(2.0)
