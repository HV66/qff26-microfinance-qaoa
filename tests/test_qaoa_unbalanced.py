"""Tests for qaoa_unbalanced.py and scripts/unbalanced_holm.py."""

import numpy as np

from data import make_portfolio
from qaoa_unbalanced import run_one, uniform_null
from scripts.unbalanced_holm import compute


def _tiny(seed: int = 0) -> dict:
    return run_one(seed, 1, n_loans=4, shots=512, restarts=1, maxiter=20)


def test_tiny_run_metric_sanity() -> None:
    r = _tiny()
    s = r["sampling"]
    assert r["config"]["n_qubits"] == 4
    for key in ("p_optimal", "p_feasible_decoded"):
        assert 0.0 <= s[key] <= 1.0
    assert "p_feasible_raw" not in s
    assert s["shots"] == 512
    assert r["null"]["p_optimal"] == 2.0**-4
    assert 0.0 <= r["null"]["p_feasible_decoded"] <= 1.0
    assert r["circuit"]["two_qubit_gates"] >= 0


def test_tiny_run_reproducible() -> None:
    a, b = _tiny(), _tiny()
    assert a["sampling"] == b["sampling"]
    assert a["optimiser"]["best_params"] == b["optimiser"]["best_params"]


def test_uniform_null_matches_enumeration() -> None:
    pf = make_portfolio(6, 1)
    null = uniform_null(pf)
    count = sum(
        int(pf.amount @ np.array([(i >> k) & 1 for k in range(6)]) <= pf.budget)
        for i in range(64)
    )
    assert null["p_feasible_decoded"] == count / 64
    assert null["p_optimal"] == 1 / 64


def test_holm_file_on_synthetic_input() -> None:
    fam = [{"label": f"t{i}", "p_raw": p} for i, p in enumerate([0.001, 0.02, 0.6])]
    out = compute(family=fam)
    assert out["family_size"] == 3
    assert out["n_raw_below_alpha"] == 2
    # Holm: 3*0.001 = 0.003 survives; 2*0.02 = 0.04 survives; 0.6 does not.
    assert out["n_survive_holm"] == 2
    assert [r["rank"] for r in out["rows"]] == [1, 2, 3]
    assert out["rows"][2]["survives"] is False
