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
- Made public on 2026-10-18 (the user's choice), so ailigment's CI can
  install optbench as a git dependency without credentials.
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
- Git (2026-10-18): `main` holds the `formula` spec and task list
  (5a6efef to 4df5b43, from `spec-formula`) and was pushed at 4df5b43 (the
  user's say-so); `spec-formula` was deleted (the user's say-so). Module `formula` (569f452 to 0cea41b, then 690f912) was
  fast-forwarded into `main`, pushed at 690f912 and `formula` deleted
  (the user's say-so, 2026-10-18); CI passed on Linux (2005 tests, 100%
  coverage, 2 min 15 s for the tests). 78cd59c and the notes commit after it are on
  `main` but not pushed. Earlier: `main` was pushed (de53338, 2026-10-17,
  the user's say-so). Every push to `main` from f33326f to fb9bd7e
  failed f23 on Linux by 3.3e-9 relative; summing the rotations in COCO's
  order (DESIGN.md Decisions) fixed it, and CI passed on de53338 (1578
  tests, 100% coverage).
  Dependabot's PR #1 (setup-uv 9.0.0 to 10.3.0) was merged by rebase on
  2026-10-17 (2aea30c, the user's say-so), so `ci.yml` now pins a newer
  setup-uv than the user's template.
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).
- Module `formula` done (2026-10-18): all 83 problems have `formula()`,
  abstract on `Problem` (results, decisions and the amended spec in
  DESIGN.md). 2005 tests, 100% coverage; the suite takes about 2 min at
  load 4 to 6 (4 min at load 25).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy and pydantic only at
  run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

`ailigment`'s goal `optbench-symreg`, task S1, which pins optbench at
690f912 (its PLAN.md). No optbench module is open.

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
3. done (2026-10-18, 569f452 to 0cea41b): module `formula` (spec
   in DESIGN.md, "Approved spec: formula"), asked for by `ailigment`'s
   approved goal `optbench-symreg` (its DESIGN.md). Pushed at 690f912.

## Open questions for the user

None.

## Known problems

- BBOB f16 takes about 93 ms per 100,000 points at d = 10, close to the
  0.1 s budget; under heavier load it may cross it (103 ms on 2026-10-17
  with another test suite running, old and new code alike). On 2026-10-18,
  at load 8 from another session's browser tests, f21 (108 ms), f23
  (134 ms) and f16 (147 ms) were over too, with no change to their code;
  rerun at lower load, all 83 passed. T_osz (shared by seven
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
