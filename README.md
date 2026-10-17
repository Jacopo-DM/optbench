# optbench

Test functions for optimisation algorithms, written from the formulas on the
optimisation pages of the Virtual Library of Simulation Experiments by Sonja
Surjanovic and Derek Bingham (Simon Fraser University,
https://www.sfu.ca/~ssurjano/optimization.html). Every function evaluates a
batch of points in one call, takes its constants as keyword arguments with
the published values as defaults, and knows its domain and global minimum.

```python
import numpy as np
import optbench

branin = optbench.make("branin")            # or optbench.Branin()
x = np.array([[np.pi, 2.275], [0.0, 0.0]])  # points shaped (n, d)
branin(x)                                   # values shaped (n,)
branin.minimum.value                        # 0.3978873577...
optbench.Branin(s=5.0, t=0.2).minimum.value # 1.0: the minimum follows s and t

tracked = optbench.Tracked(optbench.Sphere(d=10))
tracked(np.random.default_rng(0).uniform(-5.12, 5.12, size=(100, 10)))
tracked.evaluations, tracked.best_value, tracked.curve

# The baseline any optimiser should beat: 10,000 uniform random points.
run = optbench.random_search(branin, 10_000, np.random.default_rng(0))
run.best_value, run.best_point, run.curve
```

`optbench.names()` lists the functions: the 49 of the SFU pages
(`optbench.names(suite="sfu")`), ten well-known ones not on them
(`suite="classic"`: Alpine N. 2, Bird, Deceptive, Happy Cat, Himmelblau,
Qing, Salomon, Schwefel 2.21 and 2.22, Whitley), and the COCO/BBOB
noiseless suite (`suite="bbob"`; f1 to f19 so far), whose instances equal
COCO's: `optbench.BbobRastrigin(d=10, instance=3)` is COCO's f3, instance
3, in 10-D. `optbench.names(group)`
lists those of one shape group. MIT license.
