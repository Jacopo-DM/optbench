"""The base class every test function derives from."""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Annotated, ClassVar, Self, override

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BeforeValidator, ConfigDict, Field, PlainValidator

from .model import Model


class Group(StrEnum):
    """Shape groups: the six of the SFU page, then the five of BBOB."""

    MANY_LOCAL_MINIMA = "many_local_minima"
    BOWL = "bowl"
    PLATE = "plate"
    VALLEY = "valley"
    STEEP = "steep"
    OTHER = "other"
    BBOB_SEPARABLE = "bbob_separable"
    BBOB_MODERATE_CONDITIONING = "bbob_moderate_conditioning"
    BBOB_HIGH_CONDITIONING = "bbob_high_conditioning"
    BBOB_MULTIMODAL_ADEQUATE = "bbob_multimodal_adequate"
    BBOB_MULTIMODAL_WEAK = "bbob_multimodal_weak"


def _read_only(points: ArrayLike) -> NDArray[np.float64]:
    """Return a read-only float64 copy of the points.

    Returns
    -------
    numpy.ndarray
        The points, which can no longer be written to.
    """
    array = np.array(points, dtype=np.float64)
    array.flags.writeable = False
    return array


class Minimum(Model):
    """The global minimum of a problem.

    Two minima are equal only if they are the same object, as when this was
    a dataclass with ``eq=False``: ``points`` is an array.

    Attributes
    ----------
    value
        The lowest value the function takes on its domain.
    points
        Every known minimiser, one per row, shaped ``(k, d)``; read-only.
    source
        Where the numbers come from, for example a formula, a paper, or a
        numerical refinement of a published point.
    tolerance
        Relative accuracy of ``value`` and ``points``: a test at a minimiser
        accepts this much difference.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: float
    points: Annotated[NDArray[np.float64], PlainValidator(_read_only)]
    source: str
    tolerance: float = 1e-9

    @override
    def __eq__(self, other: object) -> bool:
        return self is other

    @override
    def __hash__(self) -> int:
        return id(self)


def _pairs(domain: ArrayLike | None) -> tuple[tuple[float, ...], ...] | None:
    """Turn any array-like domain into a tuple of rows.

    Returns
    -------
    tuple of tuple of float, or None
        One row per dimension; ``None`` stays ``None``.

    Raises
    ------
    ValueError
        If the domain is not two-dimensional.
    """
    if domain is None:
        return None
    bounds = np.asarray(domain, dtype=np.float64)
    if bounds.ndim != 2:  # ruff: ignore[magic-value-comparison]
        msg = f"domain must be rows of (low, high), got shape {bounds.shape}"
        raise ValueError(msg)
    return tuple(tuple(row) for row in bounds.tolist())


class Problem(Model, ABC):
    """A test function to minimise, with its domain and known answer.

    Subclasses are frozen pydantic models whose fields are the function's
    constants, each defaulting to its published value, so
    ``problem.replace(a=12.0)`` makes a checked variant. Calling a problem
    evaluates a batch of points.

    Attributes
    ----------
    domain
        Lower and upper bound per dimension, shaped ``(d, 2)``, stored as a
        tuple of ``(low, high)`` rows whatever array-like is passed.
        ``None`` (the default) means the domain the literature uses.
    """

    name: ClassVar[str]
    group: ClassVar[Group]
    url: ClassVar[str]
    references: ClassVar[tuple[str, ...]]
    suite: ClassVar[str] = "sfu"

    domain: Annotated[
        tuple[tuple[float, ...], ...] | None,
        BeforeValidator(_pairs),
    ] = Field(default=None, repr=False)

    @override
    def model_post_init(self, context: object, /) -> None:
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

    def replace(self, **changes: object) -> Self:
        """Return a copy with some fields changed, checked like a new one.

        Example: ``Branin().replace(s=5.0)``.

        Returns
        -------
        Problem
            A new problem of the same class.
        """
        fields = {name: getattr(self, name) for name in type(self).model_fields}
        return self.model_validate(fields | changes)

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

    def _check(self) -> None:
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


class AnyDimension(Problem):
    """A problem defined for any number of variables.

    Attributes
    ----------
    d
        Number of variables, at least ``smallest_dimension``.
    """

    smallest_dimension: ClassVar[int] = 1

    d: int = 2

    @override
    def _check(self) -> None:
        if self.d < self.smallest_dimension:
            msg = f"d must be at least {self.smallest_dimension}, got {self.d}"
            raise ValueError(msg)

    @property
    @override
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
    return Minimum(value=value, points=points, source="formula")
