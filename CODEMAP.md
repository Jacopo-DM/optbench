# CODEMAP.md

Structure of the code, derived from it and regenerable. No plans or decisions
here; reasons live in DESIGN.md. Line counts are Python source lines at the
commit this file was last regenerated.

One language (Python 3.13, uv). No cross-language boundary and no data
files. Other code imports the package only through `optbench/__init__.py`.

## `src/optbench` (2562 lines, 14 files)

Purpose: optimisation test functions with domains and known minima.

Public interface (`optbench/__init__.py`):
- `Problem`: abstract frozen dataclass. Call it with `(n, d)` points to get
  `(n,)` values. `dimension`, `bounds` (the `domain` field or
  `default_bounds()`), `uses_default_domain`, `contains(points)`, `minimum`
  (a `Minimum` or `None`), class variables `name`, `group`, `url`,
  `references`.
- `AnyDimension`: a `Problem` with a `d` field (default 2), for functions
  defined for any number of variables.
- `Minimum`: `value`, `points` (`(k, d)`), `source`, `tolerance`.
- `Group`: the six SFU groups (`many_local_minima`, `bowl`, `plate`,
  `valley`, `steep`, `other`).
- Functions (36), by group:
  - many local minima: `Ackley(d, a, b, c)`, `Griewank(d, divisor)`,
    `Langermann(m, c, a)`, `Levy(d, shift, scale, weight)`,
    `Rastrigin(d, a)`, `Schwefel(d, offset)`, `Bukin6(scale, curvature,
    slope, shift)`, `CrossInTray(scale, offset, power)`, `DropWave(frequency,
    curvature, offset)`, `Eggholder(shift, ratio)`, `GramacyLee2012(frequency,
    scale, shift, power)`, `HolderTable(offset)`, `LevyN13(a, b, c)`,
    `SchafferN2(scale)`, `SchafferN4(scale)`, `Shubert(terms)`;
  - bowl: `Sphere(d)`, `Bohachevsky1(weight, a, b, p, q, offset)`,
    `Bohachevsky2` and `Bohachevsky3(weight, a, p, q, offset)`,
    `Perm0(d, beta)`, `RotatedHyperEllipsoid(d)`,
    `SumOfDifferentPowers(d, power_shift)`, `SumSquares(d)`, `Trid(d, shift)`;
  - plate: `Booth(weight, a, b)`, `Matyas(a, b)`, `McCormick(a, b, offset)`,
    `PowerSum(b)`, `Zakharov(d, coefficient)`;
  - valley: `ThreeHumpCamel(a, b, c)`, `SixHumpCamel(a, b, c, e, g)`,
    `DixonPrice(d, shift, weight)`, `Rosenbrock(d, a, b)` (d >= 2);
  - other: `Branin(a, b, c, r, s, t)`, `Shekel(m, beta, c)`.
- `names(group=None)`, `make(name, **params)`: the registry.
- `Tracked(problem)`: counts evaluations, keeps `best_value` and
  `best_point`, and `curve` (rows of evaluations, best so far).

Depends on: numpy.

Files: `problem.py` (`Problem`, `AnyDimension`, `Minimum`, `Group`, `box`,
`formula_minimum`: a minimum from a formula if its minimisers lie in the
domain);
`functions/`: `bowl.py`, `plate.py`, `valley.py`, `other.py`, `many_local_minima_any_d.py`,
`many_local_minima_bukin_to_gramacy.py`,
`many_local_minima_holder_to_shubert.py`, `known_minima.py`
(`KnownMinimum`, `at_defaults` and the minima known only for the published
constants), `references.py`
(shared citations), `__init__.py` (the list the registry reads);
`registry.py`; `tracking.py`; `__init__.py`. Order: `problem` is the leaf;
`references` and `known_minima` use only `problem`; the group modules use
those; `registry` reads `functions.__all__`; `tracking` uses only the
`Problem` type.

## `tests` (751 lines, 7 files)

`test_all_problems.py` (spec criterion 6 for every registered problem, at
its defaults and free dimensions at d = 5), `test_functions.py` (Sphere,
Branin, Shekel: published constants and minima, parameter and domain
handling), `test_bowl.py`, `test_plate.py` and `test_valley.py` (those
groups), `test_many_local_minima.py` (the group's published values,
constants, minima that follow parameters, refusals),
`test_registry_and_tracking.py`. Depends on: `optbench`, numpy, pytest.

## Size checks (400 lines per file)

No file over 400 lines; the largest is
`functions/many_local_minima_any_d.py` (351).
