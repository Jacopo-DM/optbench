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
from .classics_alpine_to_himmelblau import (
    AlpineN2,
    Bird,
    Deceptive,
    HappyCat,
    Himmelblau,
)
from .classics_qing_to_whitley import (
    Qing,
    Salomon,
    Schwefel221,
    Schwefel222,
    Whitley,
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
from .other_beale_to_goldstein_price import (
    Beale,
    Branin,
    Colville,
    Forrester2008,
    GoldsteinPrice,
)
from .other_hartmann import Hartmann3, Hartmann4, Hartmann6
from .other_perm_to_styblinski_tang import (
    PermDBeta,
    Powell,
    Shekel,
    StyblinskiTang,
)
from .plate import Booth, Matyas, McCormick, PowerSum, Zakharov
from .steep import DeJong5, Easom, Michalewicz
from .valley import DixonPrice, Rosenbrock, SixHumpCamel, ThreeHumpCamel

__all__ = [
    "Ackley",
    "AlpineN2",
    "Beale",
    "Bird",
    "Bohachevsky1",
    "Bohachevsky2",
    "Bohachevsky3",
    "Booth",
    "Branin",
    "Bukin6",
    "Colville",
    "CrossInTray",
    "DeJong5",
    "Deceptive",
    "DixonPrice",
    "DropWave",
    "Easom",
    "Eggholder",
    "Forrester2008",
    "GoldsteinPrice",
    "GramacyLee2012",
    "Griewank",
    "HappyCat",
    "Hartmann3",
    "Hartmann4",
    "Hartmann6",
    "Himmelblau",
    "HolderTable",
    "Langermann",
    "Levy",
    "LevyN13",
    "Matyas",
    "McCormick",
    "Michalewicz",
    "Perm0",
    "PermDBeta",
    "Powell",
    "PowerSum",
    "Qing",
    "Rastrigin",
    "Rosenbrock",
    "RotatedHyperEllipsoid",
    "Salomon",
    "SchafferN2",
    "SchafferN4",
    "Schwefel",
    "Schwefel221",
    "Schwefel222",
    "Shekel",
    "Shubert",
    "SixHumpCamel",
    "Sphere",
    "StyblinskiTang",
    "SumOfDifferentPowers",
    "SumSquares",
    "ThreeHumpCamel",
    "Trid",
    "Whitley",
    "Zakharov",
]
