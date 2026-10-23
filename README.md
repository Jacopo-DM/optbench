# [optibench](https://github.com/Jacopo-DM/optibench)

Test functions for optimisation algorithms, written from the formulas on the
optimisation pages of the Virtual Library of Simulation Experiments by Sonja
Surjanovic and Derek Bingham (Simon Fraser University,
https://www.sfu.ca/~ssurjano/optimization.html). Every function evaluates a
batch of points in one call, takes its constants as keyword arguments with
the published values as defaults, and knows its domain and global minimum.

## Installation

Python 3.13 or later. Install it from GitHub:

```bash
uv add git+https://github.com/Jacopo-DM/optibench
```

## Example

```python
import numpy as np
import optibench

branin = optibench.make("branin")  # or optibench.Branin()
x = np.array([[np.pi, 2.275], [0.0, 0.0]])  # points shaped (n, d)
branin(x)  # values shaped (n,)
branin.minimum.value  # 0.3978873577...
optibench.Branin(s=5.0, t=0.2).minimum.value  # 1.0: the minimum follows s and t

tracked = optibench.Tracked(optibench.Sphere(d=10))
tracked(np.random.default_rng(0).uniform(-5.12, 5.12, size=(100, 10)))
tracked.evaluations, tracked.best_value, tracked.curve

# The baseline any optimiser should beat: 10,000 uniform random points.
run = optibench.random_search(branin, 10_000, np.random.default_rng(0))
run.best_value, run.best_point, run.curve
```

`optibench.names()` lists the functions: the 49 of the SFU pages
(`optibench.names(suite="sfu")`), ten well-known ones not on them
(`suite="classic"`: Alpine N. 2, Bird, Deceptive, Happy Cat, Himmelblau,
Qing, Salomon, Schwefel 2.21 and 2.22, Whitley), and the COCO/BBOB
noiseless suite (`suite="bbob"`, all 24 functions), whose instances equal
COCO's: `optibench.BbobRastrigin(d=10, instance=3)` is COCO's f3, instance
3, in 10-D. `optibench.names(group)`
lists those of one shape group. MIT license.

## Formulas

Every problem returns its function as a sympy expression in the variables
`x1` to `xd`, built from the same constants as its numpy code, and
`optibench.evaluate_formula` turns one back into values:

```python
booth = optibench.make("booth")
booth.formula()  # (x1 + 2.0*x2 - 7.0)**2 + (2.0*x1 + x2 - 5.0)**2
x = np.array([[1.0, 3.0], [0.0, 0.0]])
optibench.evaluate_formula(booth.formula(), x)  # [0., 74.], as booth(x)
optibench.make("ackley", d=2, a=10.0).formula()  # follows the changed a
```

sympy is a required dependency. Its printer runs out of memory on a
piecewise expression nested inside another one's condition, so
`evaluate_formula` refuses that shape with a ValueError.

## Documentation

There are no built docs yet. Each class's docstring gives its formula,
constants and sources; `CODEMAP.md` lists the modules and their
interfaces, and `DESIGN.md` the decisions and the measured results.
