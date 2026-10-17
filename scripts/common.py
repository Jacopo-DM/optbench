"""What the scripts share: every problem at one dimension, and the folder."""

from __future__ import annotations

from pathlib import Path

import optbench
from optbench import Problem

# The dimension of spec criterion 2 for functions where it is free.
D = 10

DATA = Path("__data__")


def every_problem(d: int = D) -> list[Problem]:
    """Every registered problem at its defaults, free dimensions at ``d``.

    Powell needs a multiple of 4, so it gets the largest one up to ``d``
    (8 for the default 10).

    Returns
    -------
    list of Problem
        One per registered name, in name order.
    """
    problems: list[Problem] = []
    for name in optbench.names():
        problem = optbench.make(name)
        if hasattr(problem, "d"):
            free = 4 * (d // 4) if name == "powell" else d
            problem = problem.replace(d=free)
        problems.append(problem)
    return problems
