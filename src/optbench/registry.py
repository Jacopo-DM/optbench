"""Find and build problems by name."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .functions import Branin, Shekel, Sphere

if TYPE_CHECKING:
    from .problem import Group, Problem

_PROBLEMS: dict[str, type[Problem]] = {
    cls.name: cls for cls in (Sphere, Branin, Shekel)
}


def names(group: Group | None = None) -> list[str]:
    """List the problem names, sorted.

    Parameters
    ----------
    group
        Only the problems of this SFU group; all of them by default.

    Returns
    -------
    list of str
        Names accepted by ``make``.
    """
    return sorted(
        name
        for name, cls in _PROBLEMS.items()
        if group is None or cls.group == group
    )


def make(name: str, **params: Any) -> Problem:  # ruff: ignore[any-type]
    """Build a problem by name, with any parameters changed.

    Example: ``make("branin", s=5.0)``.

    Returns
    -------
    Problem
        The problem.

    Raises
    ------
    KeyError
        If no problem has this name.
    """
    if name not in _PROBLEMS:
        msg = f"no problem named {name!r}; see optbench.names()"
        raise KeyError(msg)
    return _PROBLEMS[name](**params)
