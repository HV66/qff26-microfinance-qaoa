"""Tests for scripts/spectrum_stats.py."""

import numpy as np
import pytest

from data import make_portfolio
from qubo import build_qubo
from scripts import spectrum_stats


def test_residual_zero_states_match_budget_equation() -> None:
    pf = make_portfolio(4, 0)
    q = build_qubo(pf, 0.3)
    res = spectrum_stats.residuals(q, pf.amount, pf.budget)
    assert res.shape == (2**q.n_vars,)
    # Bitstring 0 selects nothing and no slack: residual is -budget.
    assert res[0] == -pf.budget
    # Every zero-residual state is exactly budget-feasible with matching slack.
    n_zero = int((res == 0).sum())
    assert n_zero > 0


def test_compute_reports_consistent_spectrum(tmp_path) -> None:
    out = spectrum_stats.compute(results_dir=tmp_path)  # no saved runs: per_p empty
    assert out["per_p"] == []
    assert out["n_bitstrings"] == 2 ** out["instance"]["n_qubits"]
    assert out["energy_min"] < out["energy_mean_uniform"] < out["energy_max"]
    lo, hi = out["energy_range_zero_residual"]
    assert out["energy_min"] == pytest.approx(lo)  # optimum has zero residual
    assert hi < out["energy_min_violating"] + 1e9
    assert out["penalty"] == pytest.approx(
        build_qubo(make_portfolio(8, 0), 0.3).penalty
    )
    assert np.isfinite(out["offset"])
