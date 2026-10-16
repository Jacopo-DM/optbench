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
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy only at run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

The rest of Tier 2: the random-search baseline and the timing script (the
user asked for both on 2026-10-17, then the context-notes hook stopped the
session before they were started). Start a branch `tier2-baseline`.

### Next steps in priority order

0. todo, approved: a random-search baseline and a timing script for spec
   criterion 2 (100,000 points under 0.1 s). The timings so far were taken
   by hand per group (DESIGN.md results), with a one-off loop over
   `optbench.names(group)` that times `problem(x)` on 100,000 uniform points
   (d = 10 where free, Powell d = 8) in CPU time, min of 5 repeats; the
   script should do the same for every function and print a table. The
   baseline is the spec's only optimiser: uniform random points in the
   domain through `Tracked`, a fixed budget, the best value and curve as
   the result. Both are outside the package (a `scripts/` folder, added to
   pyright's `include`), or the baseline inside it if it should be
   importable; decide and record why.
1. later, in `ailigment` (its PLAN.md): the adapters that use optbench, each
   with its own spec.

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
