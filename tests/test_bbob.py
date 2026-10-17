"""The BBOB functions against COCO's own implementation (``cocoex``).

``cocoex`` has no type stubs; the tests reach it through two ``Protocol``
views and one ``cast`` of its ``Suite`` class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import cocoex
import numpy as np
import pytest

import optbench
from optbench import (
    BbobAttractiveSector,
    BbobBentCigar,
    BbobBucheRastrigin,
    BbobDifferentPowers,
    BbobDiscus,
    BbobEllipsoid,
    BbobEllipsoidRotated,
    BbobGallagher21,
    BbobGallagher101,
    BbobGriewankRosenbrock,
    BbobKatsuura,
    BbobLinearSlope,
    BbobLunacek,
    BbobRastrigin,
    BbobRastriginRotated,
    BbobRosenbrock,
    BbobRosenbrockRotated,
    BbobSchaffersF7,
    BbobSchaffersF7Ill,
    BbobSchwefel,
    BbobSharpRidge,
    BbobSphere,
    BbobStepEllipsoid,
    BbobWeierstrass,
)
from optbench.functions import bbob_common as bbob

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class _CocoProblem(Protocol):
    """The part of a ``cocoex`` problem the tests use."""

    def __call__(self, x: NDArray[np.float64], /) -> float:
        """Return the value at one point."""
        ...


class _CocoSuite(Protocol):
    """The part of a ``cocoex.Suite`` the tests use."""

    def get_problem_by_function_dimension_instance(
        self,
        function: int,
        dimension: int,
        instance: int,
        /,
    ) -> _CocoProblem:
        """Return one problem of the suite."""
        ...


FUNCTIONS: dict[int, type[bbob.BbobProblem]] = {
    1: BbobSphere,
    2: BbobEllipsoid,
    3: BbobRastrigin,
    4: BbobBucheRastrigin,
    5: BbobLinearSlope,
    6: BbobAttractiveSector,
    7: BbobStepEllipsoid,
    8: BbobRosenbrock,
    9: BbobRosenbrockRotated,
    10: BbobEllipsoidRotated,
    11: BbobDiscus,
    12: BbobBentCigar,
    13: BbobSharpRidge,
    14: BbobDifferentPowers,
    15: BbobRastriginRotated,
    16: BbobWeierstrass,
    17: BbobSchaffersF7,
    18: BbobSchaffersF7Ill,
    19: BbobGriewankRosenbrock,
    20: BbobSchwefel,
    21: BbobGallagher101,
    22: BbobGallagher21,
    23: BbobKatsuura,
    24: BbobLunacek,
}
DIMENSIONS = (2, 3, 5, 10, 20, 40)

# Relative tolerance against COCO. optbench sums matrix products in its own
# order and COCO in its C loop's, so z differs in the last bit; f16 to f19
# take sines and cosines of large arguments (up to 3^11 z in f16, about
# 1e5 in f19) and f23 uses 2^j z up to j = 32, which magnify that to at
# most 7e-10 (measured 2026-10-17).
RTOL = (
    dict.fromkeys(FUNCTIONS, 1e-12)
    | dict.fromkeys(range(16, 20), 1e-9)
    | {23: 1e-9}
)
INSTANCES = (1, 2, 7, 15, 100)

_make_suite = cast("Callable[[str, str, str], _CocoSuite]", cocoex.Suite)
SUITE = _make_suite(
    "bbob",
    "instances: " + ",".join(str(i) for i in INSTANCES),
    "dimensions: " + ",".join(str(d) for d in DIMENSIONS),
)


def _coco(function: int, d: int, instance: int) -> _CocoProblem:
    """Return COCO's problem for a function, dimension and instance.

    Returns
    -------
    _CocoProblem
        A ``cocoex`` problem; call it with one point.
    """
    return SUITE.get_problem_by_function_dimension_instance(
        function,
        d,
        instance,
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("instance", INSTANCES)
def test_values_match_coco(function: int, d: int, instance: int) -> None:
    problem = FUNCTIONS[function](d=d, instance=instance)
    reference = _coco(function, d, instance)
    rng = np.random.default_rng(function * 1000 + d + instance)

    # Points inside the domain, a few outside it, and the optimum itself.
    x = np.vstack([
        rng.uniform(-5, 5, size=(20, d)),
        rng.uniform(-8, 8, size=(5, d)),
        problem.x_opt[np.newaxis],
    ])
    expected = np.array([reference(point) for point in x], dtype=np.float64)
    np.testing.assert_allclose(
        problem(x),
        expected,
        rtol=RTOL[function],
        atol=1e-10,
    )


def test_the_generator_is_uniform_and_normal() -> None:
    u = bbob.uniform(1000, 12345)
    assert np.all((u > 0) & (u < 1))
    assert u.mean() == pytest.approx(0.5, abs=0.05)
    np.testing.assert_array_equal(u, bbob.uniform(1000, 12345))
    g = bbob.gauss(2000, 7)
    assert g.mean() == pytest.approx(0.0, abs=0.1)
    assert g.std() == pytest.approx(1.0, abs=0.1)


def test_minimum_is_f_opt_at_x_opt() -> None:
    for cls in FUNCTIONS.values():
        problem = cls(d=10, instance=3)
        minimum = problem.minimum
        assert minimum is not None
        assert problem(minimum.points)[0] == pytest.approx(minimum.value)
    np.testing.assert_array_equal(
        np.abs(BbobLinearSlope(d=6).x_opt),
        np.full(6, 5.0),
    )
    assert np.all(BbobBucheRastrigin(d=7).x_opt[::2] > 0)


def test_minimum_follows_the_constants() -> None:
    assert BbobEllipsoid(conditioning=0.0).minimum is None
    assert BbobRastrigin(a=-1.0).minimum is None
    assert BbobBucheRastrigin(skew=0.0).minimum is None
    assert BbobLinearSlope(conditioning=-1.0).minimum is None
    assert BbobSphere(d=2, domain=((4.5, 5.0),) * 2).minimum is None
    assert BbobAttractiveSector(power=0.0).minimum is None
    assert BbobStepEllipsoid(slope=-1.0).minimum is None
    assert BbobRosenbrock(a=-1.0).minimum is None
    assert BbobRosenbrockRotated(a=-1.0).minimum is None
    assert BbobEllipsoidRotated(conditioning=0.0).minimum is None
    assert BbobDiscus(conditioning=0.0).minimum is None
    assert BbobBentCigar(conditioning=0.0).minimum is None
    assert BbobSharpRidge(a=-1.0).minimum is None
    assert BbobDifferentPowers(spread=-2.0).minimum is None
    assert BbobRastriginRotated(a=-1.0).minimum is None
    assert BbobWeierstrass(b=2).minimum is None
    assert BbobWeierstrass(conditioning=0.0).minimum is None
    assert BbobSchaffersF7(penalty=-1.0).minimum is None
    assert BbobSchaffersF7Ill(conditioning=0.0).minimum is None
    assert BbobGriewankRosenbrock(divisor=0.0).minimum is None
    assert BbobSchwefel(conditioning=0.0).minimum is None
    assert BbobGallagher21(penalty=-1.0).minimum is None
    assert BbobKatsuura(penalty=-1.0).minimum is None
    assert BbobLunacek(depth=0.0).minimum is None


def test_the_step_ellipsoid_has_a_plateau_at_its_minimum() -> None:
    problem = BbobStepEllipsoid(d=4, instance=1)

    # Move along a direction that keeps z_hat_1 at 0 and the rest small.
    scaling = bbob.scaling(problem.conditioning, problem.d)
    linear = scaling[:, np.newaxis] * problem.r
    z_hat = np.array([0.0, 0.03, -0.02, 0.04])
    x = problem.x_opt + np.linalg.solve(linear, z_hat)
    assert problem(x[np.newaxis])[0] == pytest.approx(problem.f_opt)

    # x_opt is the listed minimiser, and the rotations are orthonormal.
    np.testing.assert_allclose(problem.r @ problem.r.T, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(problem.q @ problem.q.T, np.eye(4), atol=1e-14)


def test_refuses_bad_arguments() -> None:
    with pytest.raises(ValueError, match="instance"):
        BbobSphere(instance=0)
    with pytest.raises(ValueError, match="d must be at least 2"):
        BbobSphere(d=1)
    with pytest.raises(ValueError, match="terms"):
        BbobWeierstrass(terms=0)
    with pytest.raises(ValueError, match="b must be"):
        BbobWeierstrass(b=1)


def test_f18_is_f17_with_more_conditioning() -> None:
    f17 = BbobSchaffersF7(d=5, instance=4)
    f18 = BbobSchaffersF7Ill(d=5, instance=4)
    np.testing.assert_array_equal(f17.x_opt, f18.x_opt)
    assert f17.f_opt == f18.f_opt
    same = BbobSchaffersF7(d=5, instance=4, conditioning=1000.0)
    x = np.random.default_rng(0).uniform(-5, 5, size=(10, 5))
    np.testing.assert_array_equal(same(x), f18(x))


def test_the_registry_lists_the_suite() -> None:
    names = sorted(cls.name for cls in FUNCTIONS.values())
    assert optbench.names(suite="bbob") == names
    groups = {
        optbench.Group.BBOB_SEPARABLE: range(1, 6),
        optbench.Group.BBOB_MODERATE_CONDITIONING: range(6, 10),
        optbench.Group.BBOB_HIGH_CONDITIONING: range(10, 15),
        optbench.Group.BBOB_MULTIMODAL_ADEQUATE: range(15, 20),
        optbench.Group.BBOB_MULTIMODAL_WEAK: range(20, 25),
    }
    for group, functions in groups.items():
        expected = sorted(FUNCTIONS[f].name for f in functions)
        assert optbench.names(group) == expected
