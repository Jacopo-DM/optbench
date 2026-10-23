# optibench (package)

The test functions, the `Problem` base class, the registry, `Tracked` and
the random-search baseline.

Interface: see [CODEMAP.md](../../CODEMAP.md). Spec and reasons: see
[DESIGN.md](../../DESIGN.md).

## Assumes

- Callers pass points shaped `(n, d)`; one point is `x[None]`.
- A problem is pure: same points, same values; only `Tracked` keeps state.

## Must never depend on

- `ailigment` or any other optimiser code.
- Any run-time library except numpy.
- The SFU site's GPL-2.0 MATLAB and R code (see DESIGN.md).
