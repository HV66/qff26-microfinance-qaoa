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
        "instance": {
            "n_loans": n_loans,
            "seed": 0,
            "lam": 0.3,
            "p": 1,
            "params": [0.4, 0.7],
        },
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


def test_compare_all_covers_every_result_file(tmp_path) -> None:
    import json

    for name, seed in (
        ("hardware_result.json", 0),
        ("hardware_result_seed1_n5_p1.json", 1),
    ):
        res = _fake_result(5, 160 + seed, 4096)
        res["instance"]["seed"] = seed
        res["instance"]["p"] = 1
        (tmp_path / name).write_text(json.dumps(res), encoding="utf-8")
    files = hardware_vs_random.hardware_result_files(tmp_path)
    assert [f.name for f in files] == [
        "hardware_result.json",
        "hardware_result_seed1_n5_p1.json",
    ]
    out = hardware_vs_random.compare_all(tmp_path)
    assert out["n_jobs"] == 2 and [j["seed"] for j in out["jobs"]] == [0, 1]
    assert out["job_id"] == out["jobs"][0]["job_id"]  # legacy top-level fields kept


def test_ideal_p_optimum_matches_independent_statevector() -> None:
    """The ideal P(optimum) equals a sum over the Qiskit circuit's probabilities."""
    import numpy as np
    from qiskit.quantum_info import Statevector

    from qaoa_run import build_ansatz

    res = _fake_result(4, 10, 100)
    res["instance"]["params"] = [0.4, 0.7]
    pf = make_portfolio(4, 0)
    q = build_qubo(pf, 0.3)
    bf = brute_force(pf, 0.3)
    probs = Statevector(
        build_ansatz(q, 1).assign_parameters(np.array([0.4, 0.7]))
    ).probabilities()
    opt_index = sum(int(b) << k for k, b in enumerate(bf.x))
    expected = sum(
        p for i, p in enumerate(probs) if (i & (2**q.n_loans - 1)) == opt_index
    )
    assert hardware_vs_random.ideal_p_optimum(res) == pytest.approx(expected, abs=1e-12)
    out = hardware_vs_random.compare(res)
    assert 0.0 <= out["binomial_p_value_below_random"] <= 1.0
    assert out["ideal_noiseless_p_optimum"] == pytest.approx(expected, abs=1e-12)
