"""The base class every test function derives from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class Group(StrEnum):
    """The six groups of the SFU optimisation page."""

    MANY_LOCAL_MINIMA = "many_local_minima"
    BOWL = "bowl"
    PLATE = "plate"
    VALLEY = "valley"
    STEEP = "steep"
    OTHER = "other"


@dataclass(frozen=True, eq=False)
class Minimum:
    """The global minimum of a problem.

    Attributes
    ----------
    value
        The lowest value the function takes on its domain.
    points
        Every known minimiser, one per row, shaped ``(k, d)``.
    source
        Where the numbers come from, for example a formula, a paper, or a
        numerical refinement of a published point.
    tolerance
        Relative accuracy of ``value`` and ``points``: a test at a minimiser
        accepts this much difference.
    """

    value: float
    points: NDArray[np.float64]
    source: str
    tolerance: float = 1e-9


@dataclass(frozen=True, kw_only=True)
class Problem(ABC):
    """A test function to minimise, with its domain and known answer.

    Subclasses are frozen dataclasses whose fields are the function's
    constants, each defaulting to its published value, so
    ``dataclasses.replace(problem, a=12.0)`` makes a variant. Calling a
    problem evaluates a batch of points.

    Attributes
    ----------
    domain
        Lower and upper bound per dimension, shaped ``(d, 2)``. ``None``
        (the default) means the domain the literature uses.
    """

    name: ClassVar[str]
    group: ClassVar[Group]
    url: ClassVar[str]
    references: ClassVar[tuple[str, ...]]

    domain: ArrayLike | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Check the parameters and the domain.

        Raises
        ------
        ValueError
            If a parameter or the domain does not fit the dimension.
        """
        self._check()
        if self.domain is not None:
            bounds = np.asarray(self.domain, dtype=np.float64)
            if bounds.shape != (self.dimension, 2) or np.any(
                bounds[:, 0] > bounds[:, 1],
            ):
                msg = (
                    f"domain must be {self.dimension} rows of (low, high), "
                    f"got shape {bounds.shape}"
                )
                raise ValueError(msg)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of variables."""

    @abstractmethod
    def default_bounds(self) -> NDArray[np.float64]:
        """Return the literature's domain, shaped ``(d, 2)``."""

    @property
    @abstractmethod
    def minimum(self) -> Minimum | None:
        """The global minimum, or ``None`` where it is not known."""

    @abstractmethod
    def _evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate checked points shaped ``(n, d)``; return ``(n,)``."""

    def _check(self) -> None:  # ruff: ignore[empty-method-without-abstract-decorator]
        """Check the parameters; subclasses with parameters override it."""

    @property
    def bounds(self) -> NDArray[np.float64]:
        """The domain in use, shaped ``(d, 2)``."""
        if self.domain is None:
            return self.default_bounds()
        return np.asarray(self.domain, dtype=np.float64)

    @property
    def uses_default_domain(self) -> bool:
        """Whether the domain is the literature's.

        A known minimum assumes it: a smaller domain may exclude it.
        """
        return self.domain is None or np.array_equal(
            self.bounds,
            self.default_bounds(),
        )

    def contains(self, points: ArrayLike) -> bool:
        """Whether every point lies in the domain in use.

        Returns
        -------
        bool
            True if all coordinates are within ``bounds``.
        """
        bounds = self.bounds
        array = np.asarray(points, dtype=np.float64)
        return bool(np.all((array >= bounds[:, 0]) & (array <= bounds[:, 1])))

    def __call__(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate a batch of points.

        Parameters
        ----------
        x
            Points shaped ``(n, d)``; one point is ``(1, d)``.

        Returns
        -------
        numpy.ndarray
            One float64 value per point, shaped ``(n,)``.

        Raises
        ------
        ValueError
            If ``x`` is not shaped ``(n, d)``.
        """
        points = np.asarray(x, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.dimension:  # ruff: ignore[magic-value-comparison]
            msg = (
                f"{self.name} expects points shaped (n, {self.dimension}), "
                f"got {points.shape}; wrap one point as x[None]"
            )
            raise ValueError(msg)
        return self._evaluate(points)


@dataclass(frozen=True, kw_only=True)
class AnyDimension(Problem):
    """A problem defined for any number of variables.

    Attributes
    ----------
    d
        Number of variables, at least ``smallest_dimension``.
    """

    smallest_dimension: ClassVar[int] = 1

    d: int = 2

    def _check(self) -> None:
        if self.d < self.smallest_dimension:
            msg = f"d must be at least {self.smallest_dimension}, got {self.d}"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        return self.d


def box(low: float, high: float, d: int) -> NDArray[np.float64]:
    """Return the same interval for every dimension.

    Returns
    -------
    numpy.ndarray
        ``(d, 2)`` bounds.
    """
    return np.tile(np.array([low, high], dtype=np.float64), (d, 1))


def formula_minimum(
    problem: Problem,
    value: float,
    points: NDArray[np.float64],
) -> Minimum | None:
    """Return a minimum known from a formula, if it lies in the domain.

    Returns
    -------
    Minimum or None
        The minimum, or ``None`` when a minimiser is outside the domain.
    """
    if not problem.contains(points):
        return None
    return Minimum(value, points, "formula")
