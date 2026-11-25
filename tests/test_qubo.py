"""Tests for qubo.py: QUBO minimum vs brute force, penalty behaviour."""

import numpy as np
import pytest

from data import LoanPortfolio, make_portfolio
from qubo import Qubo, build_qubo, slack_coefficients, to_ising

SEEDS = range(10)
N_LOANS = 8


def x_bits(index: int, n: int) -> np.ndarray:
    """Return loan bits (bit i = loan i) of a bitstring index."""
    return np.array([(index >> i) & 1 for i in range(n)])


def brute_force(p: LoanPortfolio, lam: float) -> tuple[float, np.ndarray]:
    """Independent exhaustive constrained optimum (minimising -objective).

    Never touches slack bits or the penalty, so it is independent of qubo.py.
    """
    best_val, best_x = np.inf, np.zeros(p.n)
    for k in range(2**p.n):
        x = x_bits(k, p.n)
        if p.amount @ x > p.budget:
            continue
        val = -(p.ret @ x) + lam * p.portfolio_risk(x)
        if val < best_val:
            best_val, best_x = val, x
    return float(best_val), best_x


def spends(p: LoanPortfolio) -> np.ndarray:
    """Return total lent amount for every loan-bit pattern."""
    return np.array([p.amount @ x_bits(k, p.n) for k in range(2**p.n)])


def slack_values(q: Qubo) -> np.ndarray:
    """Return the slack value encoded by every slack-bit pattern."""
    m = len(q.slack_coeffs)
    return np.array(
        [sum(q.slack_coeffs[k] for k in range(m) if (s >> k) & 1) for s in range(2**m)]
    )


@pytest.mark.parametrize("budget", [1, 2, 3, 7, 8, 11, 32])
def test_slack_represents_exactly_zero_to_budget(budget: int) -> None:
    c = slack_coefficients(budget)
    assert len(c) == budget.bit_length()
    reachable = {
        int(sum(c[k] for k in range(len(c)) if (m >> k) & 1))
        for m in range(2 ** len(c))
    }
    assert reachable == set(range(budget + 1))


@pytest.mark.parametrize("seed", SEEDS)
def test_qubo_min_equals_brute_force_optimum(seed: int) -> None:
    p = make_portfolio(N_LOANS, seed)
    q = build_qubo(p)
    energies = q.all_energies()
    bf_val, _ = brute_force(p, q.lam)
    assert energies.min() == pytest.approx(bf_val, abs=1e-9)
    x = x_bits(int(np.argmin(energies)) & (2**p.n - 1), p.n)
    assert p.amount @ x <= p.budget
    assert -(p.ret @ x) + q.lam * p.portfolio_risk(x) == pytest.approx(bf_val)


@pytest.mark.parametrize("seed", SEEDS)
def test_infeasible_bitstrings_never_win(seed: int) -> None:
    p = make_portfolio(N_LOANS, seed)
    q = build_qubo(p)
    e = q.all_energies()
    x_idx = np.arange(len(e)) & (2**p.n - 1)
    infeasible = spends(p)[x_idx] > p.budget
    assert infeasible.any()
    assert e[infeasible].min() > e.min() + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_slack_mismatch_is_penalised(seed: int) -> None:
    """For a feasible x, any wrong slack costs more than the matching slack."""
    p = make_portfolio(N_LOANS, seed)
    q = build_qubo(p)
    e = q.all_energies()
    idx = np.arange(len(e))
    x_idx, s_idx = idx & (2**p.n - 1), idx >> p.n
    spend = spends(p)
    resid = spend[x_idx] + slack_values(q)[s_idx] - p.budget
    for k in range(2**p.n):
        if spend[k] > p.budget:
            continue
        mask = x_idx == k
        assert e[mask & (resid != 0)].min() > e[mask & (resid == 0)].min()


def test_penalty_is_necessary() -> None:
    """With P = 0 the constraint is ignored, so an infeasible x wins."""
    p = make_portfolio(N_LOANS, 0)
    e = build_qubo(p, penalty=0.0).all_energies()
    x = x_bits(int(np.argmin(e)) & (2**p.n - 1), p.n)
    assert p.amount @ x > p.budget


def test_auto_penalty_scales_with_lambda() -> None:
    p = make_portfolio(N_LOANS, 0)
    assert build_qubo(p, lam=1.0).penalty > build_qubo(p, lam=0.1).penalty


@pytest.mark.parametrize("seed", SEEDS)
def test_ising_energies_equal_qubo_energies_on_all_bitstrings(seed: int) -> None:
    p = make_portfolio(N_LOANS, seed)
    q = build_qubo(p)
    ising_diag = to_ising(q).to_matrix(sparse=True).diagonal().real
    assert ising_diag.shape == (2**q.n_vars,)
    assert np.allclose(ising_diag, q.all_energies(), atol=1e-8)


def test_ising_is_diagonal_and_real_on_small_random_qubo() -> None:
    rng = np.random.default_rng(7)
    big_q = np.triu(rng.normal(size=(5, 5)))
    q = Qubo(big_q, 1.5, 3, np.array([1, 2]), 1.0, 0.3)
    op = to_ising(q)
    assert all(set(lbl) <= {"I", "Z"} for lbl in op.paulis.to_labels())
    assert np.allclose(op.to_matrix(sparse=True).diagonal().real, q.all_energies())


def test_ising_qubit_order_matches_variable_order() -> None:
    """Only variable 0 has a linear term: bitstring index 1 must pay it."""
    q = Qubo(np.diag([2.0, 0.0, 0.0]), 0.0, 3, np.array([1]), 1.0, 0.3)
    diag = to_ising(q).to_matrix(sparse=True).diagonal().real
    assert diag[1] == pytest.approx(2.0)  # variable 0 set
    assert diag[2] == pytest.approx(0.0)  # variable 1 set
