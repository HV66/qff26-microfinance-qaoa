"""Tests for compare.py using synthetic inputs."""

import json
from pathlib import Path
from typing import Any

import pytest

import compare


def _fake_qaoa(p: int, obj: float | None, exact: float = -1.0) -> dict[str, Any]:
    return {
        "config": {"seed": 0, "p": p},
        "exact": {"objective": exact, "qubo_min_energy": -1.0},
        "sampling": {
            "best_feasible_objective": obj,
            "best_feasible_gap_pct": None,
            "p_optimal": 0.1,
            "p_feasible_decoded": 0.5,
        },
        "circuit": {"depth": 42},
        "runtime_s": {"total": 1.5},
        "optimiser": {"histories": [[3.0, 2.0], [3.0, 1.0]]},
        "random_null": {"p_optimal": 0.001},
    }


def test_gap_basic() -> None:
    assert compare.gap_pct(-0.9, -1.0) == pytest.approx(10.0)
    assert compare.gap_pct(-1.0, -1.0) == 0.0


def test_gap_zero_optimum_and_none() -> None:
    assert compare.gap_pct(0.0, 0.0) == 0.0
    assert compare.gap_pct(0.5, 0.0) is None
    assert compare.gap_pct(None, -1.0) is None


def test_make_row_max_value() -> None:
    row = compare.make_row("m", -0.8, -1.0)
    assert row["max_value"] == pytest.approx(0.8)
    assert row["gap_pct"] == pytest.approx(20.0)
    assert compare.make_row("m", None, -1.0)["max_value"] is None


def test_qaoa_row_fields() -> None:
    row = compare.qaoa_row(_fake_qaoa(2, -0.5), -1.0)
    assert row["method"] == "QAOA p=2"
    assert row["circuit_depth"] == 42
    assert row["gap_pct"] == pytest.approx(50.0)
    assert row["note"] == ""


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert compare.load_qaoa(0, 1, tmp_path) is None
    assert compare.best_qaoa_p(0, (1, 2), tmp_path) is None


def test_corrupt_file_returns_none(tmp_path: Path) -> None:
    (tmp_path / "qaoa_seed0_p1.json").write_text("{bad", encoding="utf-8")
    assert compare.load_qaoa(0, 1, tmp_path) is None


def test_main_table_skips_missing_qaoa(tmp_path: Path) -> None:
    (tmp_path / "qaoa_seed0_p2.json").write_text(
        json.dumps(_fake_qaoa(2, -0.5)), encoding="utf-8"
    )
    table = compare.build_main_table(results_dir=tmp_path)
    names = [r["method"] for r in table["rows"]]
    assert "QAOA p=2" in names
    assert "QAOA p=1" not in names
    assert table["missing_qaoa_p"] == [1, 3, 4]
    assert table["rows"][0]["gap_pct"] == 0.0  # exact row


def test_summarise_gaps_handles_none() -> None:
    per_seed = [
        {"seed": 0, "method": "a", "gap_pct": 0.0},
        {"seed": 1, "method": "a", "gap_pct": 10.0},
        {"seed": 0, "method": "b", "gap_pct": None},
    ]
    out = {r["method"]: r for r in compare.summarise_gaps(per_seed)}
    assert out["a"]["mean_gap_pct"] == pytest.approx(5.0)
    assert out["a"]["worst_gap_pct"] == 10.0
    assert out["b"]["mean_gap_pct"] is None
    assert out["b"]["n_missing"] == 1


def test_best_qaoa_p_picks_lowest_gap(tmp_path: Path) -> None:
    for p, gap in ((1, 20.0), (2, 5.0)):
        d = _fake_qaoa(p, -0.9)
        d["sampling"]["best_feasible_gap_pct"] = gap
        (tmp_path / f"qaoa_seed0_p{p}.json").write_text(json.dumps(d), encoding="utf-8")
    assert compare.best_qaoa_p(0, (1, 2), tmp_path) == 2
