# CLAUDE.md

Read PLAN.md first in every session, then DESIGN.md for the approved spec and
the reasoning behind the code, CODEMAP.md for its structure, and
`src/optbench/README.md` before changing the package. If any of them
disagrees with the code, trust the code and fix the file.

## Commands

All commands run from the repo root with uv (Python 3.13).

| Purpose | Command |
|---------|---------|
| Install or rebuild the environment (numpy; dev group: pytest, pytest-cov, pyright, ruff) | `uv sync` |
| Lint (auto-fixes) | `uv run ruff check src tests` |
| Format | `uv run ruff format src tests` |
| Type check (strict) | `uv run pyright` |
| Tests (about 1 s) | `uv run pytest` |
| Test coverage | `uv run pytest --cov=src/optbench --cov-report=term-missing:skip-covered` |

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
- Adding a function (spec criteria 1 to 7): a frozen, keyword-only dataclass
  subclass of `Problem` in the module of its SFU group under
  `src/optbench/functions/`; every constant of the formula is a field with
  the published default; `name`, `group`, `url` and `references` as
  `ClassVar`s; `dimension`, `default_bounds`, `minimum` and `_evaluate`;
  a `_check` for parameters that can be wrong; register it in
  `registry.py` and export it from both `__init__.py` files. The tests in
  `tests/test_all_problems.py` then cover it automatically; add a test of its
  published constants and published minimum in `tests/test_functions.py`.
- A minimum is only stated when it is certain: from a formula, from the
  literature, or found numerically from a published point (say which in
  `Minimum.source`, and give its accuracy in `Minimum.tolerance`).
- Evaluation is vectorised over the batch: no Python loop over points.
- Keep each file under 400 lines; split a group module before it grows past.
- No abstraction without two real call sites today. Delete replaced code
  outright: no shims or old and new versions side by side.
- Indentation in Python files, docstrings and comments included, is always a
  multiple of 4 spaces. Check with
  `grep -rnE '^( {4})* {1,3}[^ ]' --include='*.py' src tests`.
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
| `src/optbench/py.typed` | hand-written, empty marker | n/a |

## Things that bite

- Ruff's auto-fix (ISC004) wraps implicitly joined strings in a tuple in
  parentheses, which pushes lines past 80 characters; rewrap the strings by
  hand afterwards.
