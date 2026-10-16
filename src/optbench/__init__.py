"""Test functions for optimisation algorithms.

Based on the optimisation pages of the Virtual Library of Simulation
Experiments (Surjanovic and Bingham, https://www.sfu.ca/~ssurjano/), written
from the published formulas. Every function evaluates a batch of points,
takes its constants as keyword arguments and knows its global minimum.
"""

from .functions import (
    Ackley,
    Branin,
    Bukin6,
    CrossInTray,
    DropWave,
    Eggholder,
    GramacyLee2012,
    Griewank,
    HolderTable,
    Langermann,
    Levy,
    LevyN13,
    Rastrigin,
    SchafferN2,
    SchafferN4,
    Schwefel,
    Shekel,
    Shubert,
    Sphere,
)
from .problem import AnyDimension, Group, Minimum, Problem
from .registry import make, names
from .tracking import Tracked

__all__ = [
    "Ackley",
    "AnyDimension",
    "Branin",
    "Bukin6",
    "CrossInTray",
    "DropWave",
    "Eggholder",
    "GramacyLee2012",
    "Griewank",
    "Group",
    "HolderTable",
    "Langermann",
    "Levy",
    "LevyN13",
    "Minimum",
    "Problem",
    "Rastrigin",
    "SchafferN2",
    "SchafferN4",
    "Schwefel",
    "Shekel",
    "Shubert",
    "Sphere",
    "Tracked",
    "make",
    "names",
]
