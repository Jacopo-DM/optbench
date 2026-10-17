"""Find and build problems by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import functions

if TYPE_CHECKING:
    from .problem import Group, Problem

_PROBLEMS: dict[str, type[Problem]] = {
    getattr(functions, name).name: getattr(functions, name)
    for name in functions.__all__
}


def names(group: Group | None = None, suite: str | None = None) -> list[str]:
    """List the problem names, sorted.

    Parameters
    ----------
    group
        Only the problems of this shape group; all of them by default.
    suite
        Only the problems of this collection: ``"sfu"`` (the SFU pages),
        ``"classic"`` (well-known functions not on them) or ``"bbob"`` (the
        COCO noiseless suite); all by default.

    Returns
    -------
    list of str
        Names accepted by ``make``.
    """
    return sorted(
        name
        for name, cls in _PROBLEMS.items()
        if (group is None or cls.group == group)
        and (suite is None or cls.suite == suite)
    )


def make(name: str, **params: object) -> Problem:
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
    return _PROBLEMS[name].model_validate(params)
