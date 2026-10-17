# CLAUDE.md

Read PLAN.md first in every session, then DESIGN.md for the approved spec and
the reasoning behind the code, CODEMAP.md for its structure, and
`src/optbench/README.md` before changing the package. If any of them
disagrees with the code, trust the code and fix the file.

## Commands

All commands run from the repo root with uv (Python 3.13).

| Purpose | Command |
|---------|---------|
| Install or rebuild the environment (numpy; dev group: pytest, pytest-cov, pyright, ruff, coco-experiment, which has no Python 3.13 wheel and is compiled from its source archive with Xcode's compiler) | `uv sync` |
| Lint (auto-fixes) | `uv run ruff check src tests scripts` |
| Format | `uv run ruff format src tests scripts` |
| Type check (strict) | `uv run pyright` |
| Tests (about 1 s) | `uv run pytest` |
| Test coverage (`if __name__` guards are excluded in `pyproject.toml`) | `uv run pytest --cov=src/optbench --cov=scripts --cov-report=term-missing:skip-covered` |
| Time every function on 100,000 points, spec criterion 2 (about 5 s; table, `__data__/timing.csv`, exit status 1 if any is over 0.1 s, except the stated exceptions in `scripts/timing.py`) | `uv run python -m scripts.timing [--points N] [--repeats K]` |
| Run the random-search baseline on every function, 10 seeds of 10,000 points (about 1 s; table, curves in `__data__/baseline/`) | `uv run python -m scripts.baseline [--budget N] [--seeds K]` |

Always pass the paths to ruff: a bare `uv run ruff check` also auto-fixes
`ruff.toml` itself, because `fix` and `unsafe-fixes` are on.

Before every commit run lint, type check and the tests above.

## Rules

- Keep the four state files up to date at all times, in the same commit as
  the change that makes them stale: PLAN.md (state, remaining work, known
  problems), CODEMAP.md (modules, interfaces, line counts), DESIGN.md (spec,
  decisions, rejected approaches, results) and this file (commands, rules).
  Each fact lives in exactly one file.
- This library never imports `ailigment` or any other optimiser code, and
  depends on numpy only at run time. A new run-time dependency needs the
  user's approval.
- Every function is written from its published formula, never from the
  SFU site's MATLAB or R code (GPL-2.0). Read the formula image on the
  function's SFU page (`<page>2.png` and so on) and the cited paper.
- BBOB functions are written from RR-6829's formulas, and their instances
  (the random generator, `x_opt`, `f_opt`, rotations) from COCO's BSD code;
  where COCO's code and the report differ, follow the code, say so in the
  class docstring, and add the function to the `cocoex` comparison in
  `tests/test_bbob.py` (every function, d = 2 to 40, instances 1, 2, 7, 15,
  100, to 1e-12, or 1e-9 where `RTOL` there says why). `cocoex` is a test
  reference only; never import it in
  `src`.
- Adding a function (spec criteria 1 to 7): a frozen, keyword-only dataclass
  subclass of `Problem` (or of `AnyDimension` when any `d` works, which
  supplies the `d` field) in the module of its SFU group under
  `src/optbench/functions/` (a function from another collection goes in
  that collection's modules, `classics_*.py`, with its `suite` set); every constant of the formula is a field with
  the published default (see DESIGN.md for which constants count);
  `name`, `group`, `url` and `references` as `ClassVar`s (shared references
  in `functions/references.py`); `dimension`, `default_bounds`, `minimum`
  and `_evaluate`; a `_check` for parameters that can be wrong (call
  `super()._check()` in an `AnyDimension`). Export it from
  `functions/__init__.py` (the registry reads that list) and from
  `optbench/__init__.py`. The tests in `tests/test_all_problems.py` then
  cover it automatically; add tests of its published constants, its
  published minimum and its parameter handling in the group's test file.
- A minimum that holds only for the published constants (found
  numerically, or published without a proof for other constants) goes in
  `functions/known_minima.py` with where it comes from, and the function
  returns
  `at_defaults(self, CONSTANT)`, which gives it only for the published
  constants and domain. Check a candidate with a grid over the whole domain,
  not only near the published point, and keep Newton steps inside the
  domain: the Eggholder's minimum is on the edge, and unconstrained steps
  left the box to a lower value.
- A minimum is only stated when it is certain: from a formula, from the
  literature, or found numerically from a published point (say which in
  `Minimum.source`, and give its accuracy in `Minimum.tolerance`).
- Evaluation is vectorised over the batch: no Python loop over points.
- Keep each file under 400 lines; split a group module before it grows past,
  alphabetically, naming the parts after their first and last function
  (`many_local_minima_bukin_to_gramacy.py`) or after what they share
  (`many_local_minima_any_d.py`).
- No abstraction without two real call sites today. Delete replaced code
  outright: no shims or old and new versions side by side.
- Indentation in Python files, docstrings and comments included, is always a
  multiple of 4 spaces. Check with
  `grep -rnE '^( {4})* {1,3}[^ ]' --include='*.py' src tests scripts`.
- Put one blank line before every comment that introduces a block of code.
- Docstrings follow the numpy convention (see ruff.toml); line length is 80.
- Typed Python, pyright strict, numpy for data, one seeded
  `numpy.random.Generator` in tests.

## Ignore list (never read or edit these when working on features)

| Path | Hand-written or generated | Made by |
|------|---------------------------|---------|
| `.venv/` | generated | `uv sync` / `uv run` |
| `uv.lock` | generated | `uv add`, `uv lock`; never edit by hand |
| `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, `.coverage` | generated | pytest, ruff, the interpreter, pytest-cov |
| `__data__/` | generated, git-ignored (`timing.csv`, `baseline/` curves, `plots/`) | `scripts.timing`, `scripts.baseline` |
| `src/optbench/py.typed` | hand-written, empty marker | n/a |

## Things that bite

- Editing a state file with a Python `str.replace` script does nothing when
  the old text does not match exactly (a reflowed line, a word changed by an
  earlier edit), and raises no error. Put `assert old in text` before every
  replacement. Twice on 2026-10-16/17 a PLAN.md task list stayed stale this
  way until a later read caught it.
- Ruff's auto-fix (ISC004) wraps implicitly joined strings in a tuple in
  parentheses, which pushes lines past 80 characters; rewrap the strings by
  hand afterwards.
- `ruff format` can wrap a long `print(...)  # ruff: ignore[print]` so the
  comment no longer sits on the `print` line; the next `ruff check` then
  deletes the whole call (T201 is an unsafe fix). It removed the baseline
  table's header on 2026-10-17. Keep such lines under 80 characters, and read
  the diff after ruff.
- Ruff's auto-fix (`non-augmented-assignment`) rewrites `y = y * w` into
  `y *= w`, which changes the array in place. When `y` may be the same
  array as `w` or another live array (`power = w`, then a loop), that
  silently changes it: on 2026-10-17 it broke BBOB f16 (values off by up to
  170) until the COCO test caught it. Start such a loop from a new array
  (`power = w * w`) and read the diff after ruff.
- BBOB batch products go through `bbob_common.rotate` and `weighted`
  (`einsum`), never `@`: BLAS may sum one row in another order than a
  batch, and the functions that take powers or cosines of large arguments
  magnify that past the batch test's 1e-14.
