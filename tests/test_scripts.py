"""The command-line scripts, on small budgets in a temporary folder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import optbench
from optbench import Sphere
from optbench.scripts import baseline, common, timing

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "__data__"
    monkeypatch.setattr(timing, "DATA", folder)
    monkeypatch.setattr(baseline, "DATA", folder)
    return folder


def test_every_problem_sets_free_dimensions() -> None:
    problems = common.every_problem()
    assert [p.name for p in problems] == optbench.names()
    dimensions = {p.name: p.dimension for p in problems}
    assert dimensions["ackley"] == common.D
    assert dimensions["powell"] == 4 * (common.D // 4)
    assert dimensions["branin"] == optbench.Branin().dimension


def test_time_problem_is_positive() -> None:
    assert timing.time_problem(Sphere(d=3), 1_000, 2) >= 0


def test_timing_writes_a_table(data: Path) -> None:
    assert timing.main(["--points", "100", "--repeats", "1"]) == 0
    lines = (data / "timing.csv").read_text().splitlines()
    assert lines[0] == "name,d,seconds"
    assert len(lines) == 1 + len(optbench.names())


def test_timing_fails_when_over_budget(
    data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "BUDGET", -1.0)
    assert timing.main(["--points", "10", "--repeats", "1"]) == 1
    assert (data / "timing.csv").exists()


def test_timing_excuses_the_stated_exceptions(
    data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timing, "BUDGET", -1.0)
    monkeypatch.setattr(timing, "EXCEPTIONS", frozenset(optbench.names()))
    assert timing.main(["--points", "10", "--repeats", "1"]) == 0
    assert (data / "timing.csv").exists()


def test_baseline_writes_one_curve_per_seed(data: Path) -> None:
    budget = 50
    baseline.main(["--budget", str(budget), "--seeds", "2"])
    curves = sorted((data / "baseline").glob("*.csv"))
    assert len(curves) == 2 * len(optbench.names())
    curve = np.loadtxt(
        data / "baseline" / "sphere_seed1.csv",
        delimiter=",",
        skiprows=1,
        ndmin=2,
    )
    assert curve[-1, 0] == budget
