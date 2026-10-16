# PLAN.md

## Goal

A library of optimisation test functions with known answers, built from the
optimisation pages of the SFU Virtual Library of Simulation Experiments, for
testing optimisation algorithms (CGP-based ones from `ailigment`, and any
other). The approved spec and task list are in DESIGN.md.

## Current state (2026-10-16)

- Tier 1 (walking skeleton) done on branch `tier1-skeleton`: `Problem`,
  `Minimum`, `Group`, the registry, `Tracked`, and Sphere, Branin and Shekel,
  with the tests of spec criterion 6. 36 tests pass, coverage 100%, ruff and
  pyright clean.
- No GitHub repo yet. Creating the private repo `Jacopo-DM/optbench` and
  pushing need the user's say-so.
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy only at run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

Ask the user to create the private GitHub repo and push, then start Tier 2.

### Next steps in priority order

0. todo, needs say-so: create the private GitHub repo `Jacopo-DM/optbench`
   (`gh repo create Jacopo-DM/optbench --private --source . --remote origin`)
   and push `main`.
1. todo, approved: Tier 2, one commit per SFU group, largest value first:
   many local minima (16 pages: Ackley, Bukin N. 6, Cross-in-Tray, Drop-Wave,
   Eggholder, Gramacy and Lee 2012, Griewank, Holder Table, Langermann, Levy,
   Levy N. 13, Rastrigin, Schaffer N. 2, Schaffer N. 4, Schwefel, Shubert);
   bowl-shaped (the rest: Bohachevsky 1 to 3, Perm 0 d beta, Rotated
   Hyper-Ellipsoid, Sum of Different Powers, Sum Squares, Trid); valley-shaped
   (Three-Hump Camel, Six-Hump Camel, Dixon-Price, Rosenbrock); plate-shaped
   (Booth, Matyas, McCormick, Power Sum, Zakharov); steep ridges or drops (De
   Jong N. 5, Easom, Michalewicz); other (the rest: Beale, Colville, Forrester
   et al. 2008, Goldstein-Price, Hartmann 3-D, 4-D and 6-D, Perm d beta,
   Powell, Styblinski-Tang).
2. todo, approved: a random-search baseline and a timing script for spec
   criterion 2 (100,000 points under 0.1 s).

## Open questions for the user

None.

## Known problems

- Seven SFU pages state no global minimum (Gramacy and Lee 2012, Langermann,
  Schaffer N. 4, Power Sum, De Jong N. 5, Forrester et al. 2008, Hartmann
  4-D). Their minima must come from the literature or a dense search, marked
  with source and accuracy (spec criterion 5).
- Langermann's recommended constants are given for d = 2 only, and Power
  Sum's b vector for d = 4 only; other d need the user to supply them, or a
  rule must be chosen and recorded.
