# DESIGN.md

Reasoning and reference that stays true after a task is done: the approved
spec, decisions with their reasons, rejected approaches, and results.

## Approved spec (the user, 2026-10-16)

The user's request, made in the `ailigment` repo on 2026-10-16: "a separate
benchmarking library which lets me test optimisation algorithms like the
CGP", based on the optimisation pages of the Virtual Library of Simulation
Experiments (https://www.sfu.ca/~ssurjano/, Surjanovic and Bingham, Simon
Fraser University), "totally distinct from the library we are working on,
but we should use this benchmark lib for CGP". The user approved the spec and
task list below on 2026-10-16 and chose the MIT license. The user's own words
on parameters: every constant in a formula must be editable, "for example
y = 2*x*x where x is the input to the function, 2 should be a parameter".

Users and use case: the user, testing optimisation algorithms (CGP-based ones
from `ailigment`, and any other) on standard test functions with known
answers, and changing the functions' constants to make variants.

Acceptance criteria, each testable:
1. All 49 functions of the SFU optimisation pages exist, each written from
   its published formula (not the site's GPL code), each citing the SFU page
   and its primary reference in its docstring.
2. One call evaluates a batch: `problem(x)` takes an `(n, d)` float array and
   returns an `(n,)` float64 array, with no Python loop over points. Budget:
   100,000 points (d = 10 where d is free) in under 0.1 s on this machine
   (CPU time, min of 5), checked by a timing script, not a test.
3. Every constant in a formula is a keyword argument with the published
   value as default, and the defaults are tested against the published
   values. Example: `Ackley(d=10, a=20.0, b=0.2, c=2*pi)`; Shekel's `beta`
   vector and `C` matrix and Langermann's `m`, `c` and `A` are arrays.
   Arguments are checked on construction (shapes that do not fit `d` raise
   `ValueError`).
4. Each problem knows its domain (`bounds`, an `(d, 2)` array; the usual
   domain by default, overridable) and its global minimum: the value and
   every known minimiser (Branin has three, Holder Table four, Shubert 18).
   Where the minimum depends on the parameters, it is given as a formula of
   them when one is known; otherwise it is known only at the defaults, and
   `minimum` is `None` for other values.
5. Seven pages state no global minimum (Gramacy and Lee 2012, Langermann,
   Schaffer N. 4, Power Sum, De Jong N. 5, Forrester et al. 2008, Hartmann
   4-D). Their minimum is taken from the literature where it gives one, else
   found by a dense search, and marked with its source and accuracy.
6. Tests for every function: the value at each listed minimiser equals the
   minimum (to 1e-6 relative, or the stated accuracy); 10,000 random points
   in the domain are never below it; the batch result equals evaluating the
   points one at a time; a wrong shape raises `ValueError`.
7. Metadata on each problem: name, SFU group, dimension (fixed, or free with
   a smallest d), references, and the page URL. A registry lists and builds
   them: `optbench.names(group=None)`, `optbench.make(name, **params)`.
8. An evaluation counter for running optimisers: `Tracked(problem)` counts
   evaluated points, keeps the best value and point so far, and records the
   best-so-far curve (evaluations, best) that a caller can save as CSV. It is
   the only state in the library; problems themselves are pure.
9. It never imports `ailigment`, and depends on numpy only at run time.

Non-goals for the first version: the emulation, integration and uncertainty
pages; shifted, rotated or noisy variants (CEC style); constrained or
multi-objective problems; plotting; GPU arrays; the optimisers themselves
(beyond a random-search baseline); and the `ailigment` adapters, which live
in `ailigment/benchmarks/` and get their own specs.

Constraints: a separate uv project at
`/Users/jmdm/Documents/projects/code/1xx_projects/optbench`, private GitHub
repo `Jacopo-DM/optbench`; Python >=3.13 (same
interpreter as `ailigment`, which pulls it in); numpy; dev tools as in
`ailigment` (ruff, pyright strict, pytest); its own four state files and
module READMEs; feature branches and the user-level git rules.

Architecture:
- `src/optbench/problem.py`: `Problem`, the base class. Subclasses are frozen
  dataclasses whose fields are the parameters, so `Rastrigin(d=5, a=12.0)`
  needs no hand-written constructor and `dataclasses.replace` makes variants.
  It provides `__call__` (shape check, then the subclass's `_evaluate`),
  `bounds`, `minimum` (a `Minimum`: value, minimisers, source) and metadata.
- `src/optbench/functions/`: one file per SFU group (`many_local_minima.py`,
  `bowl.py`, `plate.py`, `valley.py`, `steep.py`, `other.py`), each under
  400 lines.
- `src/optbench/registry.py`: `names`, `make`.
- `src/optbench/tracking.py`: `Tracked`.
- Rejected: porting the GPL-2.0 R code (ties the license; the user chose
  formulas); one plain function per problem plus a metadata table (the
  parameters, bounds and minimum would live in three places); depending on an
  existing Python collection of test functions (the user asked for an
  own library with every constant editable, and their licences and accuracy
  would need checking function by function).

Task list (approved 2026-10-16 with the spec):
- Tier 1, walking skeleton: create the repo and toolchain, the four state
  files, `Problem`, `Minimum`, the registry, `Tracked`, and three functions
  that span the cases (Sphere: free d, closed-form minimum; Branin: fixed
  d, three minimisers, six parameters; Shekel: array parameters), with the
  tests of criterion 6. Then the private GitHub repo (needs the user's
  say-so to create and push).
- Tier 2, one commit per group, largest value first: many local minima (16),
  bowl-shaped (9 with the three Bohachevsky functions), valley-shaped (4),
  plate-shaped (5), steep ridges (3), other (12). Then a random-search
  baseline and a timing script for criterion 2.
- Tier 3, not built: the non-goals above.

## Decisions and reasons

- Written from the published formulas, not ported: the site's MATLAB and R
  code is GPL-2.0 (each file header says "version 2.0 of the License"), and
  a port would tie this library to that license. The formulas are images on
  each page; they were read from the images and the cited papers.
- MIT license (the user, 2026-10-16): short and permissive, fits a benchmark
  library others may copy or cite.
- Python >=3.13 and numpy only at run time: `ailigment` (Python 3.13, see its
  CLAUDE.md for why) pulls this library in, so the floors must not exceed its
  own.
- Problems are frozen dataclasses: their fields are the constants with the
  published defaults, so `Branin(s=5.0)` and `dataclasses.replace` give
  variants without hand-written constructors, and a problem is immutable and
  pure. Metadata (`name`, `group`, `url`, `references`) are class variables.
  Array constants (Shekel's `beta` and `c`) are tuples so the dataclass stays
  hashable and frozen; they are turned into arrays when evaluating.
- One shape for input: `(n, d)` in, `(n,)` out. A single point is
  `x[None]`; refusing `(d,)` avoids silently treating a batch of 1-D points
  as one point.
- The minimum follows the parameters where a formula gives it (Sphere at the
  origin for every d; Branin `s t` at odd multiples of pi), and is otherwise
  known only at the published constants and the default domain (Shekel).
  `Problem.uses_default_domain` says whether a published minimum still
  applies. A minimiser outside a narrowed domain is dropped, and with none
  left the minimum is `None`.
- Shekel's published minimiser (4, 4, 4, 4) is rounded: the true minimisers
  were found by Newton steps from it (gradient below 1e-13) and are stored to
  full precision with their source; the values agree with the published
  -10.1532, -10.4029 and -10.5364 (m = 5, 7, 10) to their four decimals.
  Shekel's `m` selects the first `m` wells, so the three published variants
  are one class.
- Shekel computes distances from differences, not from the expansion
  `|x|^2 - 2 x.c + |c|^2`, which cancels and loses about 1e-13 near the
  minimum.
- `Tracked` is the only stateful object: it wraps a problem to count
  evaluations and record the best-so-far curve, so problems stay pure and an
  optimiser needs no changes to be measured.

- Which constants are parameters: every number in a formula that sets a
  scale, weight, frequency, shift, offset, exponent or count, named after
  what it does (`Bukin6(scale, curvature, slope, shift)`), or after the
  letter the literature uses (`Ackley(a, b, c)`). Not parameters: pi and e
  where they are part of the formula's structure (except where the
  literature makes them one, such as Ackley's `c = 2 pi`), the square in a
  squared distance, and the 1 in terms like `1 + cos(...)` that keep a term
  non-negative. Applied from the "many local minima" group on (2026-10-16).
- `AnyDimension` (a `Problem` with a `d` field, default 2, and its check)
  for the functions defined for any number of variables; it replaced the
  per-class `d` handling once six classes needed it. `Problem.contains`
  replaced two copies of the same domain check.
- Minima the literature does not give exactly are found once, offline, and
  stored with how they were found (`functions/known_minima.py`, named
  `numerical_minima.py` until the valley group): a grid
  over the whole default domain, then Newton steps with central differences
  from the best grid points. They hold only at the published constants and
  domain (`at_defaults`); a changed constant gives `None`, never a guess.
  The scripts were in a session scratch folder; the method is the record.
- Several published minima are rounded or slightly off, and the stored
  values are the true ones: Shekel (see above); Cross-in-Tray at
  (+-1.349407, +-1.349407), not 1.3491; the Schwefel function's published
  offset 418.9829 rounds the true peak 418.98288727..., so its minimum is
  about 1.27e-5 per dimension, not 0. The SFU values are kept in the tests
  at their printed precision.
- The Eggholder's minimum lies on the domain's edge (x1 = 512); outside the
  box the function goes lower (-976.9 at about (522, 413)), so it was found
  along the edge.
- Langermann takes its dimension from its centre matrix `a` (published for
  d = 2 only), not from a `d` field: other dimensions need other centres.
  Its minimum with the SFU constants, -4.1558 at (2.7934, 1.5972), comes
  from a grid over the whole domain; other collections quote other values
  (such as -5.1621), which must come from other constants or another form
  of the formula, so they do not apply here.

- Bowl-shaped group (2026-10-16): every minimum follows from a formula, so
  each holds for changed constants where the formula's conditions hold
  (non-negative weights, the minimiser inside the domain). The three
  Bohachevsky functions share a private base `_Bohachevsky`; each is its own
  registered problem, as on the SFU page. Perm 0, d, beta takes `beta = 10`
  from the default stated in the SFU page's code header, because the page
  itself gives none (only that stated default was read, not the code). The
  Trid minimum for any `shift`: the function is a convex quadratic, and
  replacing `x` by `shift x` scales it by `shift^2`, so the published minimum
  (`shift = 1`) scales by `shift^2` and its minimiser by `shift`.
- Perm 0, d, beta builds the powers `x_j^i` by repeated multiplication
  (`cumprod`) instead of calling power for each entry: 126 ms down to 43 ms
  for 100,000 points at d = 10, which brought it under the spec's 100 ms
  budget. The results differ from `x ** i` only in the last bits.

- Valley-shaped group (2026-10-16): `functions/numerical_minima.py` became
  `known_minima.py` (`KnownMinimum`), because the Three-Hump Camel's
  minimum (0 at the origin) is published and confirmed by a grid but not
  proven for other constants, and it belongs with the numerically found
  ones: both hold only at the published constants. `formula_minimum` moved
  to `problem.py` once two group modules used it. Dixon-Price has two
  minimisers for d >= 2, the published all-positive one and the same with
  `x_d` negated; flipping an earlier coordinate is not a minimiser, because
  it appears unsquared in the next term (a first version listed `2^(d-1)`
  sign flips; its test caught the error). Rosenbrock's minimum follows `b`
  only where every term can vanish: `(b, ..., b, b^2)` needs `b` equal to 0
  or 1 unless d = 2. The SFU page's rescaled Rosenbrock (Picheny et al.) is
  a variant and is not included.

- Plate-shaped group (2026-10-16): no search was needed. Booth's minimum
  solves its two linear equations for any `weight` other than +-1 (then the
  equations are dependent). Matyas is bounded below only while
  `a >= |b| / 2`. McCormick separates in `u = x1 + x2` and `v = x1 - x2`
  into `sin(u) + u / 2 + (v - 1)^2 + offset - 1`, so its minimum is
  `offset - 1 - sqrt(3) / 2 - pi / 3` at `u = -2 pi / 3`, `v = 1` (checked
  by a grid with step 0.001); other `a` and `b` are not covered because the
  minimum can move onto the domain's edge. The SFU page gives no minimum for
  Power Sum, but its published `b` holds the power sums of (1, 2, 2, 3), so
  the minimum is 0, and by Newton's identities the minimisers are exactly
  the 12 orderings of that vector (2,000,000 random points found nothing
  lower). Power Sum takes its dimension from `b` and builds its powers with
  `cumprod`, like Perm 0.

- Steep group (2026-10-16): Easom's minimum is -1 at `(centre, centre)`
  whenever `centre` is a multiple of pi, because `f >= -exp(-r^2) >= -1`.
  Michalewicz is a sum of independent one-variable terms, so its minimum is
  computed when asked, for any `d`, `m` and domain, by minimising each term
  on its own interval (a grid of 20,000 points per unit length, then Newton
  steps that must stay in the best grid cell); it reproduces the SFU
  values -1.8013, -4.687658 and -9.66015 for d = 2, 5 and 10 and takes about
  13 ms for d = 10. Its power `sin(...)^(2 m)` is computed as
  `(sin(...)^2)^m`, equal wherever the first is defined, so a non-integer
  `m` does not give NaN. De Jong N. 5 has no published minimum; a grid over
  the whole domain puts it at 0.998003837794 near (-32, -32) (the drops are
  weighted by their index, so the two coordinates differ slightly). Its
  drop centres `a` are a parameter, defaulting to the published 5 by 5 grid.

- "Other" group (2026-10-17): split into three modules
  (`other_beale_to_goldstein_price.py`, `other_hartmann.py`,
  `other_perm_to_styblinski_tang.py`), and Branin and Shekel moved there
  from `other.py`. Beale's minimum follows `c` when its three brackets can
  vanish together (the first two fix `x2 = c_2 / c_1 - 1`,
  `x1 = c_1 / (1 - x2)`, and the third must then hold). Colville's is 0 at
  (1, 1, 1, 1) while `c >= |e| / 2`. Goldstein-Price's polynomial
  coefficients are two parameter tuples (`p`, `q`, one entry per term of
  `(1, x1, x1^2, x2, x1 x2, x2^2)`) plus the two offsets, rather than twelve
  separate fields. The Hartmann functions share a base `_Hartmann`; 4-D uses
  the first four columns of the 6-D constants, rescaled as Picheny et al.
  give it. Their minima and Forrester's came from a search (200,000 random
  points, Newton from the 25 best distinct ones); Hartmann 3-D and 6-D match
  the published -3.86278 and -3.32237, and Hartmann 4-D (none published) is
  -3.13449. Perm d, beta takes `beta = 0.5` from the default stated in the
  SFU page's code header. Styblinski-Tang is separable and its minimum is
  exact for any constants and domain (the roots of a cubic per coordinate):
  -39.1661657 per dimension, so the SFU page's -39.16599 is 1.7e-4 too high.
  Powell needs `d` to be a multiple of 4, so `tests/test_all_problems.py`
  checks free-dimension functions at d = 8 (it was 5).

- The random-search baseline (2026-10-17) is `optbench.random_search`, in the
  package rather than in a script, because the `ailigment` adapters will
  import it as the result to beat. It is not an optimiser library: one
  function, uniform points through `Tracked`, so `Tracked` stays the only
  state. Points come in batches of 1,000 (one `Tracked` curve row each) and
  the last batch is cut so the budget is exact.
- The timing and baseline runners are scripts in `scripts/` (a package run
  with `python -m`, checked by pyright and ruff, tested by
  `tests/test_scripts.py` through `pythonpath = ["."]`), not part of
  `optbench`: nobody imports them. They write to `__data__/` (git-ignored),
  as in `ailigment`. The scripts draw no plots, because matplotlib would be a
  new dependency; the charts so far were drawn with `ailigment`'s
  environment from the CSVs.

## Rejected approaches

- Porting the GPL-2.0 R code: see Decisions.
- One plain function per problem plus a metadata table: the parameters,
  bounds and minimum would live in three places.
- Depending on an existing Python collection of test functions: the user
  asked for an own library with every constant editable, and the licences
  and accuracy of those collections would need checking function by function.
- A `check_dimension` helper and an `as_array` helper in `problem.py`
  (Tier 1): each had one real call site, so the checks stay inline until a
  second one exists.

## Results

| What | Measure | Result | Notes |
|------|---------|--------|-------|
| Batch speed, Tier 1 functions (100,000 points, CPU time, min of 5, Apple silicon under load, 2026-10-16) | ms per call | Sphere (d = 10) 0.5; Branin 1.6; Shekel (m = 10) 7.3 | spec criterion 2 asks for under 100 ms |
| Batch speed, "many local minima" group (100,000 points, d = 10 where free, CPU time, min of 5, load average 5 to 9, 2026-10-16) | ms per call | Ackley 11.0; Bukin N. 6 0.3; Cross-in-Tray 2.9; Drop-Wave 1.5; Eggholder 2.0; Gramacy and Lee 1.8; Griewank 10.4; Holder Table 2.1; Langermann 13.2; Levy 13.0; Levy N. 13 2.8; Rastrigin 9.8; Schaffer N. 2 1.1; Schaffer N. 4 1.7; Schwefel 9.6; Shubert 14.5 | all under the 100 ms budget |
| Batch speed, bowl-shaped group (100,000 points, d = 10 where free, CPU time, min of 5, load average 7 to 10, 2026-10-16) | ms per call | Bohachevsky 1 1.9, 2 2.0, 3 1.1; Perm 0, d, beta 43.1 (126.3 with `x ** i`); Rotated Hyper-Ellipsoid 1.6; Sphere 0.4; Sum of Different Powers 9.4; Sum Squares 1.7; Trid 2.8 | all under the 100 ms budget |
| Batch speed, valley-shaped group (100,000 points, d = 10 where free, CPU time, min of 5, load average 4 to 9, 2026-10-16) | ms per call | Dixon-Price 2.9; Rosenbrock 2.9; Six-Hump Camel 1.5; Three-Hump Camel 0.3 | all under the 100 ms budget |
| Batch speed, plate-shaped group (100,000 points, d = 10 where free, CPU time, min of 5, load average 11 to 20, 2026-10-16) | ms per call | Booth 0.3; Matyas 0.2; McCormick 1.0; Power Sum (d = 4) 8.9; Zakharov 2.8 | all under the 100 ms budget |
| Batch speed, steep group (100,000 points, d = 10 where free, CPU time, min of 5, load average 3 to 9, 2026-10-16) | ms per call | De Jong N. 5 53.5; Easom 2.0; Michalewicz 25.2 | all under the 100 ms budget; De Jong pays for a float power over 25 drops |
| Batch speed, "other" group (100,000 points, d = 10 where free, Powell d = 8, CPU time, min of 5, load average 3 to 7, 2026-10-17) | ms per call | Beale 1.7; Branin 1.0; Colville 0.7; Forrester 0.9; Goldstein-Price 0.7; Hartmann 3-D 8.5, 4-D 10.7, 6-D 9.9; Perm d, beta 42.3; Powell 6.3; Shekel 6.8; Styblinski-Tang 13.1 | all under the 100 ms budget; every one of the 49 functions now meets spec criterion 2 |
| Batch speed, all 49 by `scripts.timing` (100,000 points, d = 10 where free, Powell d = 8, CPU time, min of 5, load average 6 to 13, 2026-10-17) | ms per call | fastest Matyas 0.2; median 2.7 (Zakharov); slowest Michalewicz 26.1, Perm d, beta 42.5, Perm 0 43.0, De Jong N. 5 54.5 | 0 of 49 over the 100 ms budget; the per-group rows above agree within 1 ms |
| Random-search baseline, all 49 by `scripts.baseline` (10 seeds of 10,000 points, d = 10 where free, Powell d = 8, 2026-10-17) | median final value minus the minimum | 1-D: about 1e-6 (Forrester, Gramacy and Lee); 2-D: 1e-4 to 10 (Cross-in-Tray 3e-4, Branin 4e-3, Eggholder 16); d = 10: Sphere 13, Rastrigin 72, Rosenbrock 1.1e4, Perm 0 2e14, Perm d, beta 2e15 | the reference an optimiser must beat at the same budget; Michalewicz d = 10 reaches -4.8 of -9.66, Shekel -3.2 of -10.54 |
