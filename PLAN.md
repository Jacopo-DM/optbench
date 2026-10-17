# PLAN.md

## Goal

A library of optimisation test functions with known answers, built from the
optimisation pages of the SFU Virtual Library of Simulation Experiments, for
testing optimisation algorithms (CGP-based ones from `ailigment`, and any
other). The approved spec and task list are in DESIGN.md.

## Current state (2026-10-17)

- Tier 1 (walking skeleton) done and on `main` (4133d55; `main` had no
  commits, so the branch `tier1-skeleton` simply became `main`).
- Private GitHub repo `Jacopo-DM/optbench` created and `main` pushed
  (2026-10-16, the user's say-so).
- Tier 2, the functions: all six SFU groups are done, 49 functions, on
  `main` and pushed (ced98e3, 2026-10-17): 500 tests, coverage 100%, ruff and
  pyright clean, and every function evaluates 100,000 points in under 55 ms.
  Every feature branch so far was merged and deleted on the user's say-so;
  the working tree is clean.
- Tier 2 is complete (2026-10-17): `optbench.random_search` (the baseline)
  and the timing and baseline scripts (now `src/optbench/scripts/`);
  results in DESIGN.md.
  All 49 functions are under the 100 ms budget. On `main` and pushed
  (2026-10-17); the branch `tier2-baseline` was deleted.
- `classics` done (2026-10-17, branch `classics`): ten functions, 605
  tests, coverage 100%, ruff and pyright clean. Whitley is over the speed
  budget at d = 10, a stated exception (the user, 2026-10-17; DESIGN.md).
- `bbob` complete (2026-10-17, five commits, one per group, each merged
  and its branch deleted on the user's say-so): all 24 functions equal to
  `cocoex` (1e-9 for f16 to f19 and f23, 1e-12 for the rest; DESIGN.md
  results); 83 functions in all, 1578 tests, 100% branch coverage over
  `src` and `tests`; ruff, mypy strict and pyright strict clean.
- Git: `main` is pushed (de53338, 2026-10-17, the user's say-so) and is
  the only local branch. Every push to `main` from f33326f to fb9bd7e
  failed f23 on Linux by 3.3e-9 relative; summing the rotations in COCO's
  order (DESIGN.md Decisions) fixed it, and CI passed on de53338 (1578
  tests, 100% coverage).
  Dependabot's PR #1 (setup-uv 9.0.0 to 10.3.0) was merged by rebase on
  2026-10-17 (2aea30c, the user's say-so), so `ci.yml` now pins a newer
  setup-uv than the user's template.
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy and pydantic only at
  run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

Get the user's approval of the draft spec for module `formula` (below);
`ailigment`'s goal `optbench-symreg` waits on it.

## Next steps in priority order (approved modules, 2026-10-17)

The user approved both specs (now in DESIGN.md, "Approved spec: bbob and
classics") and their task lists on 2026-10-17, with the order: classics,
then BBOB, then the `ailigment` adapters. Downloading RR-6829 (573,076
bytes, from https://inria.hal.science/inria-00362633/document) and adding
COCO's Python package as a dev-only dependency are approved too.

1. done, CI green (2026-10-17, de53338): module `python-standard` (approved
   2026-10-17, commits ae6f494 to the pre-commit and CI commit on branch
   `python-standard`, merged into `main`). The spec, the user's four
   answers and what the work found are in DESIGN.md ("Python standard"
   under Decisions, the results row, the rejected approach).

2. done (2026-10-17): the same migration for `ailigment` (its PLAN.md).
3. draft, awaiting the user's approval: module `formula` (below), asked
   for by `ailigment`'s approved goal `optbench-symreg` (its DESIGN.md).

### Draft spec: module `formula` (2026-10-17)

Kind: build (new behaviour). Goal: every optbench problem returns its
formula as a sympy expression. Users: `ailigment`'s `optbench-symreg`,
which checks whether an evolved formula equals the true one exactly, and
anyone who wants a function's closed form (sympy can print or simplify
it). The user chose to keep the formulas here rather than in `ailigment`,
so they stay next to the functions and change with their parameters.

Acceptance criteria:

1. `Problem.formula()` returns a sympy expression in the variables
   `optbench.variables(d)`, named `x1` to `xd` as the SFU pages write them
   (column j of the points is `x(j+1)`; the user's choice over `x0`), for
   every one of the 83 problems. Example: `Booth().formula()` prints
   `(x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2`.
2. It equals the numpy value: for every problem with default parameters
   (any-d problems at d = 1 to 6 where allowed, BBOB at d = 2 to 6 and
   instances 1 to 5), the formula turned into a numpy function with
   `sympy.lambdify` matches `problem(x)` at the 26 points the BBOB test
   already uses (20 in the domain, 5 outside, the optimum), to the
   relative tolerances of `tests/test_bbob.py` (1e-12, and 1e-9 for f16
   to f19 and f23), absolute 1e-10.
3. It follows the parameters: a problem built with changed constants
   (for example `make("ackley", a=10.0)`) gives a formula that matches its
   own values, tested on one problem per file.
4. sympy is a required dependency of optbench (the user's choice over an
   optional extra: simpler, and no missing-extra path to handle).
5. The standard holds: mypy strict with no `Any`, pyright strict, ruff,
   100% branch coverage, no change to the template configs. sympy has no
   type information (1.14 ships no `py.typed`), so one module,
   `optbench/symbolic.py`, reaches it through typed Protocols with a
   `cast`, as the standard's untyped-library pattern does, and nothing
   else imports sympy. (`untyped_calls_exclude` in `mypy.ini` is the
   standard's other route, but that file is a template copy.)

Non-goals: simplifying the formulas, LaTeX or other printing APIs,
symbolic gradients, evaluating through sympy instead of numpy, formulas
for dimensions beyond what each problem allows.

Dependency: sympy 1.14.0 from PyPI (wheel about 6.3 MB) with mpmath 1.3.0
(about 0.5 MB), in `[project].dependencies`; the user accepted it for this
purpose in `ailigment`'s spec and chose a required dependency here.

Architecture: `symbolic.py` (the typed sympy facade: symbols, the
operations the formulas use, `lambdify`); `formula()` on each problem
class next to its `_evaluate`, written from the same fields; a symbolic
twin of the BBOB transforms (T_osz, T_asy, Lambda, the rotations as
number matrices, f_pen) in `bbob_common.py`. Rejected: generating the
formulas by tracing the numpy code (fragile for the piecewise and rounding
transforms), and one table of formulas apart from the classes (would drift
from the parameters).

Estimated size: about 83 formulas and a facade, roughly 1500 lines with
tests; the BBOB transforms are the largest part.

## Open questions for the user

None.

## Known problems

- BBOB f16 takes about 93 ms per 100,000 points at d = 10, close to the
  0.1 s budget; under heavier load it may cross it (103 ms on 2026-10-17
  with another test suite running, old and new code alike). T_osz (shared by seven
  BBOB functions) and the rotations are 37 ms of that; speeding up T_osz
  would help them all.
- Power Sum's recommended b vector is given for d = 4 only (other b give
  other dimensions, with an unknown minimum). Langermann's constants are for
  d = 2; other dimensions work when the caller passes `a`
  (and its minimum is then unknown).
- Functions whose minimum was found numerically (or is published without a
  proof for other constants) know it only at the
  published constants; any changed constant gives `minimum is None`. A
  caller who wants the minimum of a variant must search for it.
