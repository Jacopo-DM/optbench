"""The test functions: one module per SFU group, split before 400 lines."""

from .bowl import Sphere
from .many_local_minima_any_d import (
    Ackley,
    Griewank,
    Langermann,
    Levy,
    Rastrigin,
    Schwefel,
)
from .many_local_minima_bukin_to_gramacy import (
    Bukin6,
    CrossInTray,
    DropWave,
    Eggholder,
    GramacyLee2012,
)
from .many_local_minima_holder_to_shubert import (
    HolderTable,
    LevyN13,
    SchafferN2,
    SchafferN4,
    Shubert,
)
from .other import Branin, Shekel

__all__ = [
    "Ackley",
    "Branin",
    "Bukin6",
    "CrossInTray",
    "DropWave",
    "Eggholder",
    "GramacyLee2012",
    "Griewank",
    "HolderTable",
    "Langermann",
    "Levy",
    "LevyN13",
    "Rastrigin",
    "SchafferN2",
    "SchafferN4",
    "Schwefel",
    "Shekel",
    "Shubert",
    "Sphere",
]
