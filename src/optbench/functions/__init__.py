"""The test functions: one module per SFU group, split before 400 lines."""

from .bowl import (
    Bohachevsky1,
    Bohachevsky2,
    Bohachevsky3,
    Perm0,
    RotatedHyperEllipsoid,
    Sphere,
    SumOfDifferentPowers,
    SumSquares,
    Trid,
)
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
from .plate import Booth, Matyas, McCormick, PowerSum, Zakharov
from .steep import DeJong5, Easom, Michalewicz
from .valley import DixonPrice, Rosenbrock, SixHumpCamel, ThreeHumpCamel

__all__ = [
    "Ackley",
    "Bohachevsky1",
    "Bohachevsky2",
    "Bohachevsky3",
    "Booth",
    "Branin",
    "Bukin6",
    "CrossInTray",
    "DeJong5",
    "DixonPrice",
    "DropWave",
    "Easom",
    "Eggholder",
    "GramacyLee2012",
    "Griewank",
    "HolderTable",
    "Langermann",
    "Levy",
    "LevyN13",
    "Matyas",
    "McCormick",
    "Michalewicz",
    "Perm0",
    "PowerSum",
    "Rastrigin",
    "Rosenbrock",
    "RotatedHyperEllipsoid",
    "SchafferN2",
    "SchafferN4",
    "Schwefel",
    "Shekel",
    "Shubert",
    "SixHumpCamel",
    "Sphere",
    "SumOfDifferentPowers",
    "SumSquares",
    "ThreeHumpCamel",
    "Trid",
    "Zakharov",
]
