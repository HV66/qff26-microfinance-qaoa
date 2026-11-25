"""Tests for qaoa_run.py."""

import numpy as np
import pytest

from baseline import brute_force
from data import make_portfolio
from qaoa_run import (
    build_ansatz,
    energy_diagonal,
    fast_probabilities,
    optimise,
    qubo_to_ising,
    run_one,
    sample_counts,
    sample_metrics,
    scaled_cost_diagonal,
)
from qubo import build_qubo

LAM = 0.3


@pytest.mark.parametrize("seed", range(5))
def test_ising_plus_offset_equals_qubo_energy_all_bitstrings(seed: int) -> None:
    q = build_qubo(make_portfolio(8, seed), LAM)
    op, offset = qubo_to_ising(q)
    assert not op.paulis.z.sum(axis=1).__eq__(0).any()  # no identity left in op
    diag = op.to_matrix(sparse=True).diagonal().real + offset
    assert np.allclose(diag, q.all_energies(), atol=1e-8)
    assert np.allclose(energy_diagonal(q), q.all_energies(), atol=1e-8)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_ansatz_parameter_count_is_2p(p: int) -> None:
    q = build_qubo(make_portfolio(8, 0), LAM)
    ansatz = build_ansatz(q, p)
    assert ansatz.num_parameters == 2 * p
    assert ansatz.num_qubits == q.n_vars


def test_expectation_bounded_by_spectrum() -> None:
    q = build_qubo(make_portfolio(8, 0), LAM)
    e = energy_diagonal(q)
    probs = fast_probabilities(scaled_cost_diagonal(q), q.n_vars, np.array([0.3, 0.7]))
    assert probs.sum() == pytest.approx(1.0)
    val = float(probs @ e)
    assert e.min() - 1e-9 <= val <= e.max() + 1e-9


def test_optimise_is_seed_reproducible_and_improves() -> None:
    q = build_qubo(make_portfolio(8, 0), LAM)
    a = optimise(q, 1, 0, restarts=2, maxiter=30)
    b = optimise(q, 1, 0, restarts=2, maxiter=30)
    assert a["best_value"] == b["best_value"]
    assert a["best_params"] == b["best_params"]
    for h in a["histories"]:
        assert all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1))
    assert a["best_value"] < energy_diagonal(q).mean()


def test_sampling_reproducible_and_counts_total() -> None:
    q = build_qubo(make_portfolio(8, 0), LAM)
    params = np.array([0.4, 0.9])
    c1 = sample_counts(q, 1, params, 512, 3)
    c2 = sample_counts(q, 1, params, 512, 3)
    assert c1 == c2
    assert sum(c1.values()) == 512
    assert all(len(k) == q.n_vars for k in c1)


def test_sample_metrics_known_counts() -> None:
    pf = make_portfolio(8, 0)
    q = build_qubo(pf, LAM)
    bf = brute_force(pf, LAM)
    # Build the Qiskit key (qubit 0 rightmost) for the optimum with zero slack.
    bits = list(bf.x) + [0] * len(q.slack_coeffs)
    key = "".join(str(b) for b in reversed(bits))
    m = sample_metrics(pf, q, {key: 10}, bf.x, bf.objective)
    assert m["p_optimal"] == 1.0
    assert m["p_feasible_decoded"] == 1.0
    assert m["best_feasible_gap_pct"] == pytest.approx(0.0, abs=1e-9)
    empty = sample_metrics(pf, q, {"0" * q.n_vars: 4}, bf.x, bf.objective)
    assert empty["p_optimal"] == 0.0
    assert empty["best_feasible_gap_pct"] >= 0.0


def test_run_one_metric_sanity() -> None:
    r = run_one(seed=0, p=1, shots=512)
    s = r["sampling"]
    for key in ("p_optimal", "p_feasible_raw", "p_feasible_decoded"):
        assert 0.0 <= s[key] <= 1.0
    assert s["p_feasible_raw"] <= s["p_feasible_decoded"]
    assert s["p_optimal"] <= s["p_feasible_decoded"]
    if s["best_feasible_gap_pct"] is not None:
        assert s["best_feasible_gap_pct"] >= -1e-9
    assert 0.0 <= r["optimiser"]["approx_ratio"] <= 1.0
    assert r["circuit"]["depth"] > 0 and r["circuit"]["two_qubit_gates"] > 0
    assert r["config"]["shots"] == r["random_null"]["shots"]


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_fast_simulation_matches_qiskit_circuit(p: int) -> None:
    """The numpy inner loop must equal the Qiskit QAOAAnsatz statevector."""
    from qiskit.quantum_info import Statevector

    q = build_qubo(make_portfolio(4, 0), LAM)
    params = np.random.default_rng(p).uniform(0, 3, 2 * p)
    ref = Statevector(build_ansatz(q, p).assign_parameters(params)).probabilities()
    fast = fast_probabilities(scaled_cost_diagonal(q), q.n_vars, params)
    assert np.allclose(ref, fast, atol=1e-10)


def test_z_diagonal_matches_to_matrix_on_small_instance() -> None:
    from qaoa_run import z_diagonal

    q = build_qubo(make_portfolio(4, 1), LAM)
    op, _ = qubo_to_ising(q)
    assert np.allclose(z_diagonal(op), op.to_matrix(sparse=True).diagonal().real)
