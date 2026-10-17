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
- Git: `main` is pushed (0e06938, 2026-10-17, the user's say-so); the
  first CI run on it passed every step (cocoex 2.8.2 built from source in
  17 s, 1578 tests, 100% coverage). Dependabot opened PR #1 (setup-uv
  9.0.0 to 10.3.0), not merged. Merged local branches not yet deleted:
  `spec-python-standard`, `notes-push-blocked`, `notes-ci-green`.
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy and pydantic only at
  run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

The `ailigment` migration: write its spec and stop for the user's
approval.

## Next steps in priority order (approved modules, 2026-10-17)

The user approved both specs (now in DESIGN.md, "Approved spec: bbob and
classics") and their task lists on 2026-10-17, with the order: classics,
then BBOB, then the `ailigment` adapters. Downloading RR-6829 (573,076
bytes, from https://inria.hal.science/inria-00362633/document) and adding
COCO's Python package as a dev-only dependency are approved too.

1. done, CI green (2026-10-17): module `python-standard` (approved
   2026-10-17, commits ae6f494 to the pre-commit and CI commit on branch
   `python-standard`, merged into `main`). The spec, the user's four
   answers and what the work found are in DESIGN.md ("Python standard"
   under Decisions, the results row, the rejected approach).

2. later: the same migration for `ailigment`, then its adapters (its
   PLAN.md).

## Open questions for the user

None.

## Known problems

- BBOB f16 takes about 93 ms per 100,000 points at d = 10, close to the
  0.1 s budget; under heavier load it may cross it. T_osz (shared by seven
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
