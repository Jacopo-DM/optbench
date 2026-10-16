# CODEMAP.md

Structure of the code, derived from it and regenerable. No plans or decisions
here; reasons live in DESIGN.md. Line counts are Python source lines at the
commit this file was last regenerated.

One language (Python 3.13, uv). No cross-language boundary and no data
files. Other code imports the package only through `optbench/__init__.py`.

## `src/optbench` (4085 lines, 20 files)

Purpose: optimisation test functions with domains and known minima.

Public interface (`optbench/__init__.py`):
- `Problem`: abstract frozen dataclass. Call it with `(n, d)` points to get
  `(n,)` values. `dimension`, `bounds` (the `domain` field or
  `default_bounds()`), `uses_default_domain`, `contains(points)`, `minimum`
  (a `Minimum` or `None`), class variables `name`, `group`, `url`,
  `references`, `suite` (`"sfu"`, or `"classic"` for functions not on
  the SFU pages).
- `AnyDimension`: a `Problem` with a `d` field (default 2), for functions
  defined for any number of variables.
- `Minimum`: `value`, `points` (`(k, d)`), `source`, `tolerance`.
- `Group`: the six SFU groups (`many_local_minima`, `bowl`, `plate`,
  `valley`, `steep`, `other`).
- Functions (49 SFU and 10 classic), by group:
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
  - steep: `DeJong5(offset, power, a)`, `Easom(centre)`,
    `Michalewicz(d, m)`;
  - valley: `ThreeHumpCamel(a, b, c)`, `SixHumpCamel(a, b, c, e, g)`,
    `DixonPrice(d, shift, weight)`, `Rosenbrock(d, a, b)` (d >= 2);
  - other: `Beale(c)`, `Branin(a, b, c, r, s, t)`, `Colville(a, b, c, e)`,
    `Forrester2008(a, b, c, e)`, `GoldsteinPrice(a, b, p, q)`,
    `Hartmann3(alpha, a, p)`, `Hartmann4(alpha, a, p, offset, scale)`,
    `Hartmann6(alpha, a, p)`, `PermDBeta(d, beta)`, `Powell(d, a, b, c, e)`
    (d a multiple of 4), `Shekel(m, beta, c)`, `StyblinskiTang(d, scale, a,
    b)`.
  - classic (`suite = "classic"`): `AlpineN2(d)`, `Bird()`,
    `Deceptive(d, alpha, beta)`, `HappyCat(d, alpha)`, `Himmelblau(a, b)`,
    `Qing(d)`, `Salomon(d, frequency, a)`, `Schwefel221(d)`,
    `Schwefel222(d)`, `Whitley(d, a, divisor)`.
- `names(group=None, suite=None)`, `make(name, **params)`: the registry.
- `Tracked(problem)`: counts evaluations, keeps `best_value` and
  `best_point`, and `curve` (rows of evaluations, best so far).
- `random_search(problem, budget, rng, batch=1000)`: uniform random points in
  `problem.bounds`, exactly `budget` of them, in batches; returns the
  `Tracked` record of the run.

Depends on: numpy.

Files: `problem.py` (`Problem`, `AnyDimension`, `Minimum`, `Group`, `box`,
`formula_minimum`: a minimum from a formula if its minimisers lie in the
domain);
`functions/`: `bowl.py`, `plate.py`, `steep.py`, `valley.py`,
`other_beale_to_goldstein_price.py`, `other_hartmann.py`,
`other_perm_to_styblinski_tang.py`, `many_local_minima_any_d.py`,
`many_local_minima_bukin_to_gramacy.py`,
`many_local_minima_holder_to_shubert.py`,
`classics_alpine_to_himmelblau.py`, `classics_qing_to_whitley.py`,
`known_minima.py`
(`KnownMinimum`, `at_defaults` and the minima known only for the published
constants), `references.py`
(shared citations), `__init__.py` (the list the registry reads);
`registry.py`; `tracking.py`; `baseline.py` (`random_search`);
`__init__.py`. Order: `problem` is the leaf;
`references` and `known_minima` use only `problem`; the group modules use
those; `registry` reads `functions.__all__`; `tracking` uses only the
`Problem` type; `baseline` uses `tracking`.

## `scripts` (209 lines, 4 files)

Command-line tools, run as `python -m scripts.<name>`; not part of the
package. `common.py` (`D = 10`, `DATA = __data__`, `every_problem(d)`: every
registered problem with free dimensions at `d`, Powell at the largest
multiple of 4 up to it), `timing.py` (`time_problem(problem, points,
repeats)`, `BUDGET`, `EXCEPTIONS` (problems allowed over it), `main`
returning the exit status), `baseline.py` (`main`).
Depends on: `optbench`, numpy.

## `tests` (1290 lines, 12 files)

`test_all_problems.py` (spec criterion 6 for every registered problem, at
its defaults and free dimensions at d = 8), `test_functions.py` (Sphere,
Branin, Shekel: published constants and minima, parameter and domain
handling), `test_bowl.py`, `test_plate.py`, `test_steep.py` and
`test_valley.py` and `test_other.py` (those groups), `test_classics.py`, `test_many_local_minima.py` (the group's published values,
constants, minima that follow parameters, refusals),
`test_registry_and_tracking.py`, `test_baseline.py`, `test_scripts.py`
(both scripts on small budgets in a temporary folder). Depends on: `optbench`, `scripts`, numpy, pytest.

## Size checks (400 lines per file)

No file over 400 lines; the largest is
`functions/many_local_minima_any_d.py` (351).
