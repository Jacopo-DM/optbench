# PLAN.md

## Goal

A library of optimisation test functions with known answers, built from the
optimisation pages of the SFU Virtual Library of Simulation Experiments, for
testing optimisation algorithms (CGP-based ones from `ailigment`, and any
other). The approved spec and task list are in DESIGN.md.

## Current state (2026-10-16)

- Tier 1 (walking skeleton) done and on `main` (4133d55; `main` had no
  commits, so the branch `tier1-skeleton` simply became `main`).
- Private GitHub repo `Jacopo-DM/optbench` created and `main` pushed
  (2026-10-16, the user's say-so).
- Tier 2: the "many local minima", bowl-shaped and valley-shaped groups are
  on `main` and pushed (6e3f3cf); the plate-shaped group (5 more, 36
  functions in all) is done on branch `tier2-plate`: 363 tests, coverage
  100%, ruff and pyright clean.
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy only at run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

Merge `tier2-plate`, then the steep group.

### Next steps in priority order

0. todo, approved: Tier 2, one commit per SFU group: steep ridges or drops
   (De Jong N. 5, Easom, Michalewicz); other (Beale, Colville, Forrester et
   al. 2008, Goldstein-Price, Hartmann 3-D, 4-D and 6-D, Perm d beta, Powell,
   Styblinski-Tang).
1. todo, approved: a random-search baseline and a timing script for spec
   criterion 2 (100,000 points under 0.1 s).

## Open questions for the user

None.

## Known problems

- Three SFU pages state no global minimum and are not done yet (De Jong
  N. 5, Forrester et al. 2008, Hartmann 4-D). Their minima must come
  from the literature or a dense search, marked with source and accuracy
  (spec criterion 5), as was done for Gramacy and Lee 2012, Langermann,
  Schaffer N. 4 and Power Sum.
- Power Sum's recommended b vector is given for d = 4 only (other b give
  other dimensions, with an unknown minimum). Langermann's constants are for
  d = 2; other dimensions work when the caller passes `a`
  (and its minimum is then unknown).
- Functions whose minimum was found numerically (or is published without a
  proof for other constants) know it only at the
  published constants; any changed constant gives `minimum is None`. A
  caller who wants the minimum of a variant must search for it.
