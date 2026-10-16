"""Test functions for optimisation algorithms.

Based on the optimisation pages of the Virtual Library of Simulation
Experiments (Surjanovic and Bingham, https://www.sfu.ca/~ssurjano/), written
from the published formulas. Every function evaluates a batch of points,
takes its constants as keyword arguments and knows its global minimum.
"""

from .functions import Branin, Shekel, Sphere
from .problem import Group, Minimum, Problem
from .registry import make, names
from .tracking import Tracked

__all__ = [
    "Branin",
    "Group",
    "Minimum",
    "Problem",
    "Shekel",
    "Sphere",
    "Tracked",
    "make",
    "names",
]
