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
  results); 83 functions in all, 1578 tests, coverage 100%.
- Git: `main` is pushed up to 1385e82 (2026-10-17, the user's say-so);
  65b1971 (the `python-standard` spec) is on `main`, not pushed. Work on
  `python-standard` is on branch `python-standard`.
- `ailigment` does not use this library yet; its adapters are separate
  modules in that repo (see its PLAN.md).

## Constraints

- See CLAUDE.md rules: no import of `ailigment`, numpy and pydantic only at
  run time,
  formulas not GPL code.
- Python >=3.13 to match `ailigment`.

## Next step

Task 9 of `python-standard` (below): pre-commit and CI.

## Next steps in priority order (approved modules, 2026-10-17)

The user approved both specs (now in DESIGN.md, "Approved spec: bbob and
classics") and their task lists on 2026-10-17, with the order: classics,
then BBOB, then the `ailigment` adapters. Downloading RR-6829 (573,076
bytes, from https://inria.hal.science/inria-00362633/document) and adding
COCO's Python package as a dev-only dependency are approved too.

1. executing (approved by the user 2026-10-17): module `python-standard`
   (kind: change). Tasks 1 to 8 are done. Task 1 took `KnownMinimum`
   from task 2 and the tests of task 4, which it needed to keep the checks
   green; coverage was already 100% of branches over `src` and `tests` once
   `pytest.ini` arrived in task 5. All four standard checks pass since task 7. Spec and task list below; the user's answers to its four
   design questions are in DESIGN.md (Decisions, "Python standard").

2. later: the same migration for `ailigment`, then its adapters (its
   PLAN.md).

### Spec: `python-standard`

Goal: optbench passes the user's Python standard
(`~/.claude/standards/python/core.md` and the files beside it) with the
template configs from `~/Functions/_rsync/python/` unchanged, and every
function returns the same values as before, bit for bit.

Success metric: `uv run ruff format --check`, `uv run ruff check --no-fix`,
`uv run mypy .` and `uv run pytest` (100% branch coverage over `src` and
`tests`) all pass, so the Stop hook `python-gate.sh` lets a turn end; the
value pin below matches; timing stays within its noise.

Measured on 2026-10-17 in a scratch worktree with the template configs
(nothing in the repo changed):

- mypy strict: 33 errors in 18 files. By code: 16 `no-any-return` (numpy
  reductions returned as `float`), 5 `arg-type`, 2 each `operator`,
  `no-untyped-call` (cocoex), `explicit-any` (`tests/test_bbob.py`),
  `call-arg`, `attr-defined`, 1 each `union-attr`, `redundant-expr`. Worst
  files: `bbob_common.py` 10, `tests/test_bbob.py` 7, `tests/test_functions.py` 5.
- ruff, template rules: 70 findings. 30 `no-self-use` (`_evaluate` and
  `default_bounds` overrides that ignore `self`; the repo's `ruff.toml`
  silences this today, the template does not), 23 `banned-api` (every
  `dataclass` import and the two `NamedTuple`s), 17 `magic-value-comparison`
  (mostly tests; the repo silences it in tests, the template does not).
- The repo's `ruff.toml` differs from the template: `target-version` py312,
  rule codes instead of names, the two extra ignores above, a `noxfile.py`
  entry, and no `banned-api` table. `pyproject.toml` carries
  `[tool.pytest.ini_options]` (`pythonpath = ["."]`),
  `[tool.coverage.report]` and `[tool.pyright]`, which the standard moves out.
- Not dependencies yet: `mypy`, `pydantic`, `pre-commit`, `pillow` (all on the
  standard's pre-approved list).
- Probe of a frozen pydantic `Problem` base (pydantic 2, Python 3.13):
  `ABC` with abstract properties, `ClassVar` metadata and `cached_property`
  all work; equality and hashing work with a tuple domain; construction
  takes about 0.6 us; a `ValueError` raised in `model_post_init` reaches the
  caller as `pydantic.ValidationError`, a `ValueError` subclass, so
  `pytest.raises(ValueError, match=...)` still passes. Lax mode turns
  `a=2` into `2.0` (today an int stays an int; values are unchanged).
- `@typing.override` on an override silences `no-self-use` (checked with
  ruff 0.17), so those 30 need no suppression comments.

Found in task 1: the generated `__init__` is typed with the stored
field type, so type checkers want `domain` as a tuple of tuples (tests now
pass tuples; arrays and lists still work at run time, and through
`model_validate`). A flat tuple where a matrix belongs (`Shekel(c=(1.0,
2.0))`) is now refused by pydantic's type check with its own message, not
by the function's "c must be a matrix" check; an empty tuple still reaches
that check. Timing after task 1: total 1270 ms against 1261 ms before
(three runs each, minimum per function), median ratio 1.000.

Behaviour that changes (approved, see DESIGN.md): `problem.replace(...)`
replaces `dataclasses.replace(problem, ...)`; `domain` is stored as a tuple
of `(low, high)` pairs; `Minimum.points` is read-only; bad constructor
arguments raise `pydantic.ValidationError` (still a `ValueError`). Nothing
else may change: every function value, `bounds`, `minimum`, the registry,
`Tracked` and the scripts' output stay as they are.

Non-goals: a Sphinx docs site (`docs.md`; the user's `/zz-python-docs`
prompt adds it to a library when wanted); speeding anything up (f16 stays a
known problem); the `ailigment` migration (its own module afterwards).

Verification for every task:

- Value pin: before the first code change, a scratch script evaluates every
  registered problem (each `d`-free one at d = 2, 5 and 10, BBOB instances
  1 and 7) at 50 fixed points and saves the arrays to the scratchpad; after
  each task the same script must give `np.array_equal` results, and
  `minimum`, `bounds` and `names()` must compare equal. The existing tests
  (cocoex at 1e-12/1e-9, known minima, the 1e-14 batch test) also stay green.
- Timing: `scripts.timing` before and after the model port, three runs
  each; a slowdown beyond the spread stops the work for a look.
- Until the config task lands, each commit passes the current checks
  (`uv run pytest`, `uv run pyright`, `uv run ruff check src tests scripts`);
  from that task on, the four standard commands. The Stop hook blocks on
  the new checks after any Python edit, so the work runs as one pass,
  task by task, on branch `python-standard`.

### Task list: `python-standard` (one commit each, in this order)

9. pre-commit and CI: `templates/pre-commit-config.yaml` to
    `.pre-commit-config.yaml`, `uv add --dev pre-commit`, `uv run
    pre-commit install`; `templates/ci.yml` to `.github/workflows/ci.yml`
    with Python 3.13, plus Dependabot for `github-actions` and `uv`. CI
    compiles cocoex from source on ubuntu-latest; if that fails, stop and
    tell the user. Pushing the branch (to run CI) needs the user's say-so.
    Risk: medium (first CI run).

Skipped, with the reason: replacing `functools.cache` on the BBOB helpers
with `cached_property` on the models (no behaviour or speed gain; the
module cache is shared across instances with the same seed, which is what
BBOB wants).

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
