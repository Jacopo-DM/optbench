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
  and `scripts/timing.py` and `scripts/baseline.py`; results in DESIGN.md.
  All 49 functions are under the 100 ms budget. On `main` and pushed
  (2026-10-17); the branch `tier2-baseline` was deleted.
- `classics` done (2026-10-17, branch `classics`): ten functions, 605
  tests, coverage 100%, ruff and pyright clean. Whitley is over the speed
  budget at d = 10, a stated exception (the user, 2026-10-17; DESIGN.md).
- `bbob` Tier 1 done (2026-10-17, branch `bbob-tier1`): the generator, the
  transforms and f1 to f5, equal to `cocoex` to 1e-12 over 150
  (function, d, instance) combinations; 811 tests, coverage 100%.
- `bbob` f6 to f9 done (2026-10-17, branch `bbob-f6-f9`), with COCO's
  rotations: equal to `cocoex` to 1e-12; 972 tests, coverage 100%.
- `bbob` f10 to f14 done (2026-10-17, branch `bbob-f10-f14`): equal to
  `cocoex` to 1e-12; 1172 tests, coverage 100%.
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy only at run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

`bbob` group f15 to f19 (rotated Rastrigin, Weierstrass, Schaffers F7,
ill-conditioned Schaffers F7, composite Griewank-Rosenbrock), in
`functions/bbob_multimodal_adequate.py`. f18 takes f17's seeds for
`f_opt` (already in `bbob_common.f_opt`) and for its instance (COCO's
`rseed_17` in `suite_bbob.c`). The full order is below.

## Next steps in priority order (approved modules, 2026-10-17)

The user approved both specs (now in DESIGN.md, "Approved spec: bbob and
classics") and their task lists on 2026-10-17, with the order: classics,
then BBOB, then the `ailigment` adapters. Downloading RR-6829 (573,076
bytes, from https://inria.hal.science/inria-00362633/document) and adding
COCO's Python package as a dev-only dependency are approved too.

1. todo: `bbob` Tier 2, one commit per group: f15 to f19, f20 to f24. COCO's sources for each are in its repository,
   `code-experiments/src/f_<name>.c` (read on GitHub; BSD-3-Clause), and
   RR-6829 sections 2 to 5 give the formulas.
2. later: the `ailigment` adapters (its PLAN.md).

## Open questions for the user

None.

## Known problems


- Power Sum's recommended b vector is given for d = 4 only (other b give
  other dimensions, with an unknown minimum). Langermann's constants are for
  d = 2; other dimensions work when the caller passes `a`
  (and its minimum is then unknown).
- Functions whose minimum was found numerically (or is published without a
  proof for other constants) know it only at the
  published constants; any changed constant gives `minimum is None`. A
  caller who wants the minimum of a variant must search for it.
