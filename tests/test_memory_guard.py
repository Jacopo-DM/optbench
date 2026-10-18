"""The memory watchdog stops a process that passes its limit."""

from __future__ import annotations

import faulthandler
import os

import memory_guard
import pytest


def test_the_peak_is_positive_and_below_the_limit() -> None:
    assert 0 < memory_guard.peak_memory() <= memory_guard.LIMIT


def test_watch_stops_once_the_peak_passes_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[bool] = []
    monkeypatch.setattr(memory_guard, "LIMIT", 0)
    monkeypatch.setattr(memory_guard, "stop", lambda: stopped.append(True))
    memory_guard.watch()
    assert stopped == [True]


def test_stop_reports_and_exits_with_its_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_exit(status: int) -> None:
        raise SystemExit(status)

    def no_traceback(*, all_threads: bool) -> None:
        assert all_threads

    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr(faulthandler, "dump_traceback", no_traceback)
    with pytest.raises(SystemExit) as stopped:
        memory_guard.stop()
    assert stopped.value.code == memory_guard.EXIT_STATUS
    assert "passed the 2.0 GB limit" in capsys.readouterr().err
