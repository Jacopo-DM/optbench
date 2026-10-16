# CODEMAP.md

Structure of the code, derived from it and regenerable. No plans or decisions
here; reasons live in DESIGN.md. Line counts are Python source lines at the
commit this file was last regenerated.

One language (Python 3.13, uv). No cross-language boundary and no data
files. Other code imports the package only through `optbench/__init__.py`.

## `src/optbench` (578 lines, 7 files)

Purpose: optimisation test functions with domains and known minima.

Public interface (`optbench/__init__.py`):
- `Problem`: abstract frozen dataclass. Call it with `(n, d)` points to get
  `(n,)` values. `dimension`, `bounds` (the `domain` field or
  `default_bounds()`), `uses_default_domain`, `minimum` (a `Minimum` or
  `None`), class variables `name`, `group`, `url`, `references`.
- `Minimum`: `value`, `points` (`(k, d)`), `source`, `tolerance`.
- `Group`: the six SFU groups (`many_local_minima`, `bowl`, `plate`,
  `valley`, `steep`, `other`).
- Functions: `Sphere(d)` (bowl), `Branin(a, b, c, r, s, t)` and
  `Shekel(m, beta, c)` (other).
- `names(group=None)`, `make(name, **params)`: the registry.
- `Tracked(problem)`: counts evaluations, keeps `best_value` and
  `best_point`, and `curve` (rows of evaluations, best so far).

Depends on: numpy.

Files: `problem.py` (`Problem`, `Minimum`, `Group`, `box`), `functions/bowl.py`,
`functions/other.py`, `functions/__init__.py`, `registry.py`, `tracking.py`,
`__init__.py`. Order: `problem` is the leaf; `functions` use it; `registry`
uses `functions`; `tracking` uses only the `Problem` type.

## `tests` (233 lines, 3 files)

`test_all_problems.py` (spec criterion 6 for every registered problem, at
its defaults and free dimensions at d = 5), `test_functions.py` (published
constants and minima, parameter and domain handling), `test_registry_and_tracking.py`.
Depends on: `optbench`, numpy, pytest.

## Size checks (400 lines per file)

No file over 400 lines; the largest is `functions/other.py` (198).
