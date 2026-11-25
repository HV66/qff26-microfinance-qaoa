"""Tests for solve_csv.py (no network, small QAOA)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import solve_csv

EXAMPLE = Path("examples/loans_example.csv")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "loans.csv"
    path.write_text(text, encoding="utf-8")
    return path


def _fast_optimise(q: Any, p: int, seed: int) -> dict[str, Any]:
    from qaoa_run import optimise

    return optimise(q, p, seed, restarts=1, maxiter=20)


def test_example_end_to_end(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(solve_csv, "optimise", _fast_optimise)
    code = solve_csv.main(
        [str(EXAMPLE), "--budget", "12", "--p", "1", "--shots", "256"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "NOT credit advice" in out
    assert "human credit officer" in out
    assert "SAME volatility" in out
    for title in ("Exact (brute force)", "Greedy", "QAOA (best feasible sample)"):
        assert title in out
    assert out.count("sectors with zero funded loans:") == 3
    assert out.count("amount deployed:") == 3
    for line in out.splitlines():
        if line.startswith("amount deployed:"):
            assert int(line.split(":")[1]) <= 12


def test_qaoa_portfolio_is_budget_feasible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solve_csv, "optimise", _fast_optimise)
    amounts, returns, labels = solve_csv.read_loans(EXAMPLE)
    pf, names = solve_csv.build_portfolio(amounts, returns, labels, 12)
    assert names == ["farming", "retail", "services"]
    x = solve_csv.qaoa_portfolio(pf, 0.3, 1, 256, 0)
    assert x is not None
    assert int(pf.amount @ x) <= 12


def test_sigma_matches_data_rule() -> None:
    from data import RHO_CROSS, RHO_SAME, VOL_RANGE

    amounts, returns, labels = solve_csv.read_loans(EXAMPLE)
    pf, _ = solve_csv.build_portfolio(amounts, returns, labels, 12)
    v = (VOL_RANGE[0] + VOL_RANGE[1]) / 2
    assert pf.sigma[0, 0] == pytest.approx(v * v)
    assert pf.sigma[0, 1] == pytest.approx(v * v * RHO_SAME)
    assert pf.sigma[0, 2] == pytest.approx(v * v * RHO_CROSS)


def test_too_many_loans(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = "\n".join(f"{i + 1},0.5,a" for i in range(11))
    path = _write(tmp_path, "amount,expected_return,sector\n" + rows + "\n")
    with pytest.raises(SystemExit) as exc:
        solve_csv.main([str(path), "--budget", "10"])
    assert exc.value.code != 0
    assert "at most 10" in capsys.readouterr().err


def test_bad_column(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, "amount,return,sector\n1,0.5,a\n")
    with pytest.raises(SystemExit) as exc:
        solve_csv.main([str(path), "--budget", "10"])
    assert exc.value.code != 0
    assert "columns must be exactly" in capsys.readouterr().err


@pytest.mark.parametrize(
    "body, message",
    [
        ("-2,0.5,a\n", "must be positive"),
        ("0,0.5,a\n", "must be positive"),
        ("2.5,0.5,a\n", "not an integer"),
        ("2,abc,a\n", "not a number"),
        ("2,-0.1,a\n", "expected_return must be positive"),
        ("2,0.5,\n", "sector is empty"),
        ("", "no loan rows"),
    ],
)
def test_invalid_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    body: str,
    message: str,
) -> None:
    path = _write(tmp_path, "amount,expected_return,sector\n" + body)
    with pytest.raises(SystemExit) as exc:
        solve_csv.main([str(path), "--budget", "10"])
    assert exc.value.code != 0
    assert message in capsys.readouterr().err


def test_budget_below_min_amount(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "amount,expected_return,sector\n5,0.5,a\n6,0.6,b\n")
    with pytest.raises(SystemExit) as exc:
        solve_csv.main([str(path), "--budget", "4"])
    assert exc.value.code != 0
    assert "below the smallest" in capsys.readouterr().err


def test_empty_file_and_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "")
    with pytest.raises(SystemExit):
        solve_csv.main([str(path), "--budget", "4"])
    with pytest.raises(SystemExit):
        solve_csv.main([str(tmp_path / "nope.csv"), "--budget", "4"])
    err = capsys.readouterr().err
    assert "columns must be exactly" in err
    assert "file not found" in err


def test_sector_audit_zero_funded() -> None:
    amounts, returns, labels = solve_csv.read_loans(EXAMPLE)
    pf, names = solve_csv.build_portfolio(amounts, returns, labels, 12)
    x = np.zeros(pf.n, dtype=int)
    x[0] = 1
    audit = solve_csv.sector_audit(pf, names, x)
    assert audit["zero_funded_sectors"] == ["retail", "services"]
    assert audit["deployed"] == 3
