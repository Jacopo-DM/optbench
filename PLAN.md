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
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy only at run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

Nothing is left in this repo's approved tiers. The next work is in
`ailigment` (its PLAN.md): the adapters that use optbench, each with its own
spec; adding optbench as a dependency there needs the user's approval.

### Next steps in priority order

1. later, in `ailigment` (its PLAN.md): the adapters that use optbench, each
   with its own spec.

## Proposed modules (awaiting the user's approval, 2026-10-17)

The user chose on 2026-10-17 to add the BBOB functions and a few missing
classics (sources reviewed in DESIGN.md, "Other sources of test
functions"). Nothing is built until the user approves each spec and its
task list in words. Both are drafted on branch `spec-extensions`.

### Module `bbob` (build): the 24 noiseless COCO/BBOB functions

Goal: the BBOB 2009 noiseless suite in optbench, so optimisers (CGP-evolved
ones included) can be compared with published COCO results.

Acceptance criteria, each testable:
1. f1 to f24, written from the formulas of Hansen, Finck, Ros and Auger
   (2009), "Real-Parameter Black-Box Optimization Benchmarking 2009:
   Noiseless Functions Definitions", INRIA RR-6829, in their five groups:
   separable (f1 sphere, f2 ellipsoidal, f3 Rastrigin, f4 Buche-Rastrigin,
   f5 linear slope); low or moderate conditioning (f6 attractive sector,
   f7 step ellipsoidal, f8 Rosenbrock, f9 rotated Rosenbrock); high
   conditioning and unimodal (f10 ellipsoidal, f11 discus, f12 bent cigar,
   f13 sharp ridge, f14 different powers); multimodal with adequate global
   structure (f15 Rastrigin, f16 Weierstrass, f17 Schaffers F7, f18
   ill-conditioned Schaffers F7, f19 composite Griewank-Rosenbrock);
   multimodal with weak global structure (f20 Schwefel, f21 Gallagher 101
   peaks, f22 Gallagher 21 peaks, f23 Katsuura, f24 Lunacek bi-Rastrigin).
2. A problem is built from `(d, instance)`: the instance fixes `x_opt`,
   `f_opt` and the rotation matrices, so the same arguments always give the
   same function. Any `d >= 2`; domain [-5, 5]^d.
3. Every constant of a formula is a field, as spec criterion 3 says
   (conditioning 10^6, the asymmetry beta, the Weierstrass and Katsuura
   constants, Gallagher's peak count and so on).
4. The shared transformations (T_osz, T_asy, the diagonal Lambda^alpha, the
   boundary penalty f_pen, the seeded rotations) are one helper each; each
   has several real call sites.
5. Each problem knows its minimum `f_opt` and minimiser (`x_opt` for most;
   from the definitions for f5, f20, f21 and f22).
6. The existing criterion-6 tests run on every BBOB problem; values match
   the reference at fixed points (open question 2).
7. 100,000 points at d = 10 in under 0.1 s, as criterion 2.

Non-goals: the noisy suite, bbob-biobj, large-scale, mixed-integer and
constrained suites; COCO's experiment runner and post-processing.

Architecture: a new module area `functions/bbob_*.py` (split by group to
stay under 400 lines) and `functions/bbob_transforms.py`. A new class
variable `suite` (`"sfu"`, `"bbob"`, `"classic"`) beside `group`, with BBOB's
five groups as their own `Group` values, and `names(group=None, suite=None)`.
Names like `bbob_f01_sphere`.

Tiers: Tier 1, the instance machinery and transforms plus f1 to f5, with
the reference check. Tier 2, one commit per group: f6 to f9, f10 to f14,
f15 to f19, f20 to f24. Tier 3: the non-goals.

Needs approval besides the spec: downloading RR-6829 (a PDF of 573,076 bytes
from https://inria.hal.science/inria-00362633/document) to read the
formulas; and the
answers to open questions 1 and 2.

### Module `classics` (build): ten well-known functions not on the SFU pages

Each from its formula in Jamil and Yang (2013) and Gavana's AMPGO pages,
with every constant a field, its minimum and the criterion-6 tests:
Himmelblau (2-D, four minimisers, 0), Bird (2-D, two minimisers,
-106.7645), Alpine N. 1 (any d, 0 at the origin), Salomon (any d, 0 at the
origin), Schwefel 2.21 (any d, max of |x_i|, 0), Schwefel 2.22 (any d, sum
plus product of |x_i|, 0), Qing (any d, 0 at x_i = +-sqrt(i)), Happy Cat
(any d, 0 at x = -1), Whitley (any d, 0 at x = 1), Deceptive (any d, -1 at
the alpha vector). `suite = "classic"`, and each in the SFU shape group that
fits it. One commit; the minima are checked numerically as for SFU.

## Open questions for the user

1. BBOB instances: reproduce COCO's own instances exactly (its random
   generator and instance-to-seed rule, read from its BSD-licensed code), so
   results compare with published COCO data; or draw them with numpy from a
   seed, which gives the same function family but numbers that do not match
   COCO's.
2. BBOB reference values for tests: add COCO's Python package (`cocoex`) as
   a dev-only dependency and test against it; or check a few values by hand
   against published numbers.
3. Order: the classics (small), then BBOB, then the `ailigment` adapters;
   or the adapters first.

## Known problems

- Power Sum's recommended b vector is given for d = 4 only (other b give
  other dimensions, with an unknown minimum). Langermann's constants are for
  d = 2; other dimensions work when the caller passes `a`
  (and its minimum is then unknown).
- Functions whose minimum was found numerically (or is published without a
  proof for other constants) know it only at the
  published constants; any changed constant gives `minimum is None`. A
  caller who wants the minimum of a variant must search for it.
