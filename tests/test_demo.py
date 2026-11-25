"""Fast tests for demo.py and scripts/run_all.py (no heavy pipeline execution)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import demo
from scripts import run_all

ROOT = Path(__file__).resolve().parent.parent


def test_default_demo_reproduces_saved_result(tmp_path: Path) -> None:
    """Default mode samples the saved best-p circuit and matches stored metrics."""
    saved = demo.load_saved_best(ROOT / "results")
    assert saved is not None
    p = saved["config"]["p"]
    assert saved["optimiser"]["best_value"] == min(
        json.loads(f.read_text(encoding="utf-8"))["optimiser"]["best_value"]
        for f in (ROOT / "results").glob("qaoa_seed0_p*.json")
    )
    rows = demo.run_demo(quick=False, out_dir=tmp_path, results_dir=ROOT / "results")
    qaoa = [r for r in rows if r["method"].startswith("QAOA")]
    assert [r["method"] for r in qaoa] == [f"QAOA p={p}"]
    assert qaoa[0]["p_optimum"] == saved["sampling"]["p_optimal"]
    assert qaoa[0]["gap_pct"] == saved["sampling"]["best_feasible_gap_pct"]
    assert saved["config"]["shots"] == 4096


def test_quick_mode_prints_reduced_budget_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    demo.run_demo(quick=True, out_dir=tmp_path)
    out = capsys.readouterr().out
    assert out.count("REDUCED BUDGET, results will differ") >= 1


def test_default_falls_back_with_banner_when_no_saved_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    demo.run_demo(quick=False, out_dir=tmp_path, results_dir=tmp_path / "empty")
    out = capsys.readouterr().out
    assert "No matching saved QAOA result found" in out
    assert "REDUCED BUDGET" in out


def test_demo_quick_runs_and_table_is_consistent(tmp_path: Path) -> None:
    rows = demo.run_demo(quick=True, out_dir=tmp_path)
    assert (tmp_path / "demo_plot.png").is_file()
    assert [r["method"] for r in rows][:2] == ["Exact brute force", "Greedy"]
    exact = rows[0]["objective"]
    for r in rows:
        if r["objective"] is not None:
            assert r["objective"] >= exact - 1e-9
    assert rows[0]["gap_pct"] == 0.0
    assert "QAOA" in demo.format_table(rows)


def test_honesty_note_is_computed() -> None:
    base = {"p_optimum": None, "depth": None, "runtime_s": 0.0, "objective": 0.0}
    worse = [
        {**base, "method": "Exact brute force", "gap_pct": 0.0},
        {**base, "method": "Greedy", "gap_pct": 1.0},
        {**base, "method": "QAOA p=1", "gap_pct": 5.0},
    ]
    assert "match or beat" in demo.honesty_note(worse)
    better = [
        {**base, "method": "Exact brute force", "gap_pct": 2.0},
        {**base, "method": "Greedy", "gap_pct": 3.0},
        {**base, "method": "QAOA p=1", "gap_pct": 1.0},
    ]
    assert "not evidence" in demo.honesty_note(better)


def test_run_all_help_works() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0
    assert "--skip-tests" in out.stdout


def test_missing_scripts_are_skipped_not_errors(tmp_path: Path) -> None:
    steps = [run_all.Step("ghost", ("ghost.py",), "ghost.py", False)]
    assert not run_all.is_runnable(steps[0], tmp_path)
    assert run_all.run_pipeline(steps, tmp_path) == 0


def test_skip_tests_flag_and_mandatory_set() -> None:
    names = [s.name for s in run_all.build_steps(skip_tests=True)]
    assert "pytest" not in names
    assert "pytest" in [s.name for s in run_all.build_steps()]
    assert not any("hardware" in " ".join(s.args) for s in run_all.build_steps())
    mandatory = [s.name for s in run_all.build_steps() if s.mandatory]
    assert mandatory[:2] == ["pytest", "lambda sweep"]
