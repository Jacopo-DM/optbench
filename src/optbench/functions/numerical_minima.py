"""Global minima found numerically, for the published constants.

Each was found by a grid search over the function's default domain, then
Newton steps (central differences) from the best grid points until the step
was below 1e-15, and checked against the SFU page's rounded value where it
gives one. The numbers hold for the default constants and default domain
only.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from optbench.problem import Minimum, Problem


@dataclass(frozen=True)
class NumericalMinimum:
    """A minimum found numerically.

    Attributes
    ----------
    value
        The lowest value found.
    points
        Every minimiser found.
    source
        How it was found.
    tolerance
        Relative accuracy of ``value``.
    """

    value: float
    points: tuple[tuple[float, ...], ...]
    source: str
    tolerance: float


def at_defaults(problem: Problem, found: NumericalMinimum) -> Minimum | None:
    """Return a numerically found minimum if it applies to ``problem``.

    Returns
    -------
    Minimum or None
        The minimum if the problem has its published constants (``found``
        was computed for them) and its default domain, else ``None``.
    """
    published = dataclasses.replace(problem, domain=None) == type(problem)()
    if not (published and problem.uses_default_domain):
        return None
    return Minimum(
        found.value,
        np.array(found.points),
        found.source,
        tolerance=found.tolerance,
    )


_SEARCH = "grid search and Newton refinement"

# SFU: -2.06261 at (+-1.3491, +-1.3491). Grid step 0.01.
_CROSS = 1.3494066169385899
CROSS_IN_TRAY = NumericalMinimum(
    -2.062611870822739,
    (
        (_CROSS, _CROSS),
        (_CROSS, -_CROSS),
        (-_CROSS, _CROSS),
        (-_CROSS, -_CROSS),
    ),
    _SEARCH + " (SFU: -2.06261 at (+-1.3491, +-1.3491))",
    1e-12,
)

# SFU: -959.6407 at (512, 404.2319). The minimum lies on the edge x1 = 512
# (outside the domain the function goes lower); grid step 0.1 over the whole
# domain, then Newton in x2 along the edge.
EGGHOLDER = NumericalMinimum(
    -959.640662720851,
    ((512.0, 404.2318051120689),),
    _SEARCH + " along the edge x1 = 512 (SFU: -959.6407 at (512, 404.2319))",
    1e-12,
)

# The SFU page states no minimum. Grid step 1e-6 over [0.5, 2.5].
GRAMACY_LEE_2012 = NumericalMinimum(
    -0.8690111349894998,
    ((0.5485634445950892,),),
    _SEARCH + " (the SFU page states no minimum)",
    1e-12,
)

# SFU: -19.2085 at (+-8.05502, +-9.66459). Grid step 0.01.
_HOLDER_1, _HOLDER_2 = 8.055023475738732, 9.664590019226269
HOLDER_TABLE = NumericalMinimum(
    -19.20850256788673,
    (
        (_HOLDER_1, _HOLDER_2),
        (_HOLDER_1, -_HOLDER_2),
        (-_HOLDER_1, _HOLDER_2),
        (-_HOLDER_1, -_HOLDER_2),
    ),
    _SEARCH + " (SFU: -19.2085 at (+-8.05502, +-9.66459))",
    1e-12,
)

# The SFU page states no minimum. Grid step 0.05 over the whole domain, then
# 0.001 over [-3, 3]^2; the function is even in each variable.
_SCHAFFER = 1.2531318314239779
SCHAFFER_N4 = NumericalMinimum(
    0.29257863203598056,
    ((0.0, _SCHAFFER), (0.0, -_SCHAFFER), (_SCHAFFER, 0.0), (-_SCHAFFER, 0.0)),
    _SEARCH + " (the SFU page states no minimum)",
    1e-12,
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
SHUBERT = NumericalMinimum(
    -186.73090883102392,
    tuple((_SHUBERT[i], _SHUBERT[j]) for i, j in _SHUBERT_PAIRS),
    _SEARCH + " (SFU: -186.7309)",
    1e-12,
)

# The SFU page states no minimum. Grid step 0.005 over [0, 10]^2.
LANGERMANN = NumericalMinimum(
    -4.155809291847785,
    ((2.793402208625887, 1.5972325012426074),),
    _SEARCH + " (the SFU page states no minimum)",
    1e-12,
)
