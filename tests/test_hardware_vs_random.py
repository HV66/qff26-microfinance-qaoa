"""Tests for scripts/hardware_vs_random.py (offline, uses synthetic counts)."""

import pytest

from baseline import brute_force
from data import make_portfolio
from qubo import build_qubo
from scripts import hardware_vs_random


def _fake_result(n_loans: int, optimum_hits: int, shots: int) -> dict:
    pf = make_portfolio(n_loans, 0)
    q = build_qubo(pf, 0.3)
    bf = brute_force(pf, 0.3)
    bits = [str(int(v)) for v in bf.x][::-1]
    opt_key = "0" * (q.n_vars - q.n_loans) + "".join(bits)
    other_key = "1" * q.n_vars
    counts = {opt_key: optimum_hits, other_key: shots - optimum_hits}
    return {
        "job_id": "test",
        "backend": "none",
        "instance": {"n_loans": n_loans, "seed": 0, "lam": 0.3},
        "metrics": {
            "p_optimal": optimum_hits / shots,
            "p_feasible_decoded": 0.5,
            "p_feasible_raw": 0.1,
        },
        "counts": counts,
    }


def test_clear_excess_is_flagged_above_random() -> None:
    out = hardware_vs_random.compare(_fake_result(5, 400, 4096))
    assert out["verdict"] == "above random"
    assert out["expected_hits_uniform"] == pytest.approx(128.0)
    assert out["shots"] == 4096


def test_random_level_is_indistinguishable() -> None:
    out = hardware_vs_random.compare(_fake_result(5, 128, 4096))
    assert out["verdict"] == "indistinguishable from random"
    assert 0.0 <= out["random_null"]["p_optimum"] <= 1.0
