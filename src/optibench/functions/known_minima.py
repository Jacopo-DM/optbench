"""Global minima known only for the published constants.

Most were found numerically: a grid search over the function's default
domain, then Newton steps (central differences) from the best grid points
until the step was below 1e-15, checked against the SFU page's rounded value
where it gives one. Others are published but not proven for other constants.
Every one holds for the default constants and default domain only.
"""

from __future__ import annotations

import itertools

import numpy as np

from optibench.model import Model
from optibench.problem import Minimum, Problem


class KnownMinimum(Model):
    """A minimum known only for a function's published constants.

    Attributes
    ----------
    value
        The lowest value.
    points
        Every known minimiser.
    source
        Where it comes from, or how it was found.
    tolerance
        Relative accuracy of ``value``.
    """

    value: float
    points: tuple[tuple[float, ...], ...]
    source: str
    tolerance: float


def at_defaults(problem: Problem, found: KnownMinimum) -> Minimum | None:
    """Return a numerically found minimum if it applies to ``problem``.

    Returns
    -------
    Minimum or None
        The minimum if the problem has its published constants (``found``
        was computed for them) and its default domain, else ``None``.
    """
    published = problem.replace(domain=None) == type(problem)()
    if not (published and problem.uses_default_domain):
        return None
    return Minimum(
        value=found.value,
        points=np.array(found.points),
        source=found.source,
        tolerance=found.tolerance,
    )


_SEARCH = "grid search and Newton refinement"

# SFU: -2.06261 at (+-1.3491, +-1.3491). Grid step 0.01.
_CROSS = 1.3494066169385899
CROSS_IN_TRAY = KnownMinimum(
    value=-2.062611870822739,
    points=(
        (_CROSS, _CROSS),
        (_CROSS, -_CROSS),
        (-_CROSS, _CROSS),
        (-_CROSS, -_CROSS),
    ),
    source=_SEARCH + " (SFU: -2.06261 at (+-1.3491, +-1.3491))",
    tolerance=1e-12,
)

# SFU: -959.6407 at (512, 404.2319). The minimum lies on the edge x1 = 512
# (outside the domain the function goes lower); grid step 0.1 over the whole
# domain, then Newton in x2 along the edge.
EGGHOLDER = KnownMinimum(
    value=-959.640662720851,
    points=((512.0, 404.2318051120689),),
    source=_SEARCH
    + " along the edge x1 = 512 (SFU: -959.6407 at (512, 404.2319))",
    tolerance=1e-12,
)

# The SFU page states no minimum. Grid step 1e-6 over [0.5, 2.5].
GRAMACY_LEE_2012 = KnownMinimum(
    value=-0.8690111349894998,
    points=((0.5485634445950892,),),
    source=_SEARCH + " (the SFU page states no minimum)",
    tolerance=1e-12,
)

# SFU: -19.2085 at (+-8.05502, +-9.66459). Grid step 0.01.
_HOLDER_1, _HOLDER_2 = 8.055023475738732, 9.664590019226269
HOLDER_TABLE = KnownMinimum(
    value=-19.20850256788673,
    points=(
        (_HOLDER_1, _HOLDER_2),
        (_HOLDER_1, -_HOLDER_2),
        (-_HOLDER_1, _HOLDER_2),
        (-_HOLDER_1, -_HOLDER_2),
    ),
    source=_SEARCH + " (SFU: -19.2085 at (+-8.05502, +-9.66459))",
    tolerance=1e-12,
)

# The SFU page states no minimum. Grid step 0.05 over the whole domain, then
# 0.001 over [-3, 3]^2; the function is even in each variable.
_SCHAFFER = 1.2531318314239779
SCHAFFER_N4 = KnownMinimum(
    value=0.29257863203598056,
    points=(
        (0.0, _SCHAFFER),
        (0.0, -_SCHAFFER),
        (_SCHAFFER, 0.0),
        (-_SCHAFFER, 0.0),
    ),
    source=_SEARCH + " (the SFU page states no minimum)",
    tolerance=1e-12,
)

# SFU: -186.7309. Grid step 0.005; every grid point within 2 of the lowest
# was refined, and 18 distinct minimisers share the lowest value. Their
# coordinates take six values; each pairs one value with three others.
_SHUBERT = (
    -7.708313735514333,
    -7.083506407644529,
    -1.425128428334852,
    -0.8003211004648427,
    4.858056878844753,
    5.48286420671475,
)
_SHUBERT_PAIRS = (
    (0, 1),
    (0, 3),
    (0, 5),
    (1, 0),
    (1, 2),
    (1, 4),
    (2, 1),
    (2, 3),
    (2, 5),
    (3, 0),
    (3, 2),
    (3, 4),
    (4, 1),
    (4, 3),
    (4, 5),
    (5, 0),
    (5, 2),
    (5, 4),
)
SHUBERT = KnownMinimum(
    value=-186.73090883102392,
    points=tuple((_SHUBERT[i], _SHUBERT[j]) for i, j in _SHUBERT_PAIRS),
    source=_SEARCH + " (SFU: -186.7309)",
    tolerance=1e-12,
)

# The SFU page states no minimum. Grid step 0.005 over [0, 10]^2.
LANGERMANN = KnownMinimum(
    value=-4.155809291847785,
    points=((2.793402208625887, 1.5972325012426074),),
    source=_SEARCH + " (the SFU page states no minimum)",
    tolerance=1e-12,
)

# SFU: 0 at (0, 0). Published, and a grid over the whole domain (step 0.002)
# finds nothing lower; for other constants it is not proven.
THREE_HUMP_CAMEL = KnownMinimum(
    value=0.0,
    points=((0.0, 0.0),),
    source="published (SFU: 0 at (0, 0)), checked by a grid over the domain",
    tolerance=0.0,
)

# SFU: -1.0316 at (0.0898, -0.7126) and (-0.0898, 0.7126). Grid step 0.002;
# the function is unchanged by x -> -x, so the two minimisers are opposite.
_CAMEL_1, _CAMEL_2 = 0.0898420131014, -0.7126564029511063
SIX_HUMP_CAMEL = KnownMinimum(
    value=-1.0316284534898774,
    points=((_CAMEL_1, _CAMEL_2), (-_CAMEL_1, -_CAMEL_2)),
    source=_SEARCH + " (SFU: -1.0316 at (0.0898, -0.7126) and its opposite)",
    tolerance=1e-12,
)

# The SFU page states no minimum. The published b = (8, 18, 44, 114) holds
# the power sums of (1, 2, 2, 3), so f is 0 there; by Newton's identities
# the power sums 1 to 4 fix the four numbers up to order, so the minimisers
# are exactly the 12 distinct orderings.
POWER_SUM = KnownMinimum(
    value=0.0,
    points=tuple(sorted(set(itertools.permutations((1.0, 2.0, 2.0, 3.0))))),
    source="exact: the power sums of (1, 2, 2, 3) (the SFU page states none)",
    tolerance=0.0,
)

# The SFU page states no minimum. Grid step 0.02 over the whole domain; the
# drop at (-32, -32) is the deepest (the next, at (-16, -32), is near 1.99).
# The two coordinates differ slightly: the drops are weighted by their
# index, which is not symmetric in x1 and x2.
DE_JONG_5 = KnownMinimum(
    value=0.99800383779445,
    points=((-31.978334183274413, -31.97833392314706),),
    source=_SEARCH + " (the SFU page states no minimum)",
    tolerance=1e-12,
)

# The SFU page states no minimum. Grid step 1e-6 over [0, 1].
FORRESTER_2008 = KnownMinimum(
    value=-6.020740055767083,
    points=((0.757248757699377,),),
    source=_SEARCH + " (the SFU page states no minimum)",
    tolerance=1e-12,
)

# SFU: 3 at (0, -1). Exact there: x1 + x2 + 1 = 0 makes the first factor 1,
# and 2 x1 - 3 x2 = 3 makes the second 30 + 9 (18 - 48 + 27) = 3. A grid
# with step 0.001 over the whole domain finds nothing lower.
GOLDSTEIN_PRICE = KnownMinimum(
    value=3.0,
    points=((0.0, -1.0),),
    source="published (SFU: 3 at (0, -1)), checked by a grid over the domain",
    tolerance=1e-12,
)

# SFU: -3.86278 at (0.114614, 0.555649, 0.852547). Best of 200,000 random
# points, Newton steps from the 25 best distinct ones.
HARTMANN_3 = KnownMinimum(
    value=-3.862779787332663,
    points=((0.11458887666777617, 0.5556488946139297, 0.8525469846234259),),
    source="random search and Newton refinement (SFU: -3.86278)",
    tolerance=1e-12,
)

# The SFU page states no minimum. Same search as Hartmann 3-D.
HARTMANN_4 = KnownMinimum(
    value=-3.1344941412224,
    points=(
        (
            0.1873952729895325,
            0.19415152929501436,
            0.5579177800547765,
            0.26477962415486545,
        ),
    ),
    source=(
        "random search and Newton refinement (the SFU page states no minimum)"
    ),
    tolerance=1e-12,
)

# SFU: -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652,
# 0.6573). Same search as Hartmann 3-D.
HARTMANN_6 = KnownMinimum(
    value=-3.322368011415515,
    points=(
        (
            0.2016895110259998,
            0.15001069182070456,
            0.4768739741457662,
            0.2753324304886124,
            0.3116516166096781,
            0.6573005340675282,
        ),
    ),
    source="random search and Newton refinement (SFU: -3.32237)",
    tolerance=1e-12,
)

# Gavana: -106.7645367198034 at (4.701055751981055, 3.152946019601391) and
# (-1.582142172055011, -3.130246799635430). Grid step 0.002 over the whole
# domain, then Newton on the exact gradient (below 1e-13 at both points); the
# published value is 3e-8 too high.
BIRD = KnownMinimum(
    value=-106.7645367492647,
    points=(
        (4.701043130249553, 3.15293850372493),
        (-1.5821421769300335, -3.1302468034546562),
    ),
    source="grid search and Newton refinement on the exact gradient (Gavana: "
    "-106.7645367198034)",
    tolerance=1e-12,
)

# Wikipedia: 0 at (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186) and
# (3.584428, -1.848126). Each is a zero of both squared terms; Newton on that
# pair of equations with the exact Jacobian, and a grid of step 0.002 over
# the domain finds no fifth.
HIMMELBLAU = KnownMinimum(
    value=0.0,
    points=(
        (3.0, 2.0),
        (-2.805118086952745, 3.131312518250573),
        (-3.779310253377747, -3.283185991286169),
        (3.5844283403304917, -1.8481265269644036),
    ),
    source="published (Wikipedia), refined by Newton on the two squared terms",
    tolerance=1e-12,
)
