"""Classical baselines: exact brute force, greedy, uniform random sampling."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from data import LoanPortfolio
from qubo import Qubo, decode_loans, is_feasible, objective, residual


@dataclass(frozen=True)
class SolveResult:
    """Result of a classical solver.

    Attributes:
        x: Selected loans, binary vector of shape (n,).
        objective: Value of the shared objective (minimised; lower is better).
        runtime_s: Wall time in seconds (time.perf_counter).
    """

    x: np.ndarray
    objective: float
    runtime_s: float


@dataclass(frozen=True)
class SamplingResult:
    """Result of uniform random sampling over all QUBO variables.

    Attributes:
        best_objective: Best objective among decoded-feasible samples, or
            None if no sample was feasible.
        best_x: The x achieving best_objective, or None.
        p_feasible_raw: Fraction of full bitstrings with zero penalty.
        p_feasible_decoded: Fraction whose loan bits satisfy the budget.
        p_optimal: Fraction whose decoded loan bits equal the optimal x.
        shots: Number of samples drawn.
    """

    best_objective: float | None
    best_x: np.ndarray | None
    p_feasible_raw: float
    p_feasible_decoded: float
    p_optimal: float
    shots: int


def brute_force(p: LoanPortfolio, lam: float) -> SolveResult:
    """Exact solver: enumerate all 2^n loan vectors (no slack, no penalty).

    Ties go to the smallest bitstring index, so the result is deterministic.

    Args:
        p: Loan portfolio.
        lam: Risk-aversion weight.
    """
    start = time.perf_counter()
    best_val, best_x = np.inf, np.zeros(p.n, dtype=int)
    for k in range(2**p.n):
        x = np.array([(k >> i) & 1 for i in range(p.n)], dtype=int)
        if not is_feasible(p, x):
            continue
        val = objective(p, x, lam)
        if val < best_val:
            best_val, best_x = val, x
    return SolveResult(best_x, float(best_val), time.perf_counter() - start)


def greedy(p: LoanPortfolio, lam: float) -> SolveResult:
    """Greedy heuristic by marginal gain per unit amount.

    Repeatedly adds the loan with the largest
    [ret_i - lam * (risk(x + e_i) - risk(x))] / amount_i, among loans that fit
    the remaining budget and have a positive marginal gain. Stops when none
    qualifies. Ties go to the lowest loan index.

    Args:
        p: Loan portfolio.
        lam: Risk-aversion weight.
    """
    start = time.perf_counter()
    x = np.zeros(p.n, dtype=int)
    remaining = p.budget
    while True:
        base_risk = p.portfolio_risk(x)
        best_i, best_score = -1, 0.0
        for i in range(p.n):
            if x[i] or p.amount[i] > remaining:
                continue
            trial = x.copy()
            trial[i] = 1
            gain = p.ret[i] - lam * (p.portfolio_risk(trial) - base_risk)
            if gain <= 0.0:
                continue
            score = gain / p.amount[i]
            if score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            break
        x[best_i] = 1
        remaining -= int(p.amount[best_i])
    return SolveResult(x, objective(p, x, lam), time.perf_counter() - start)


def random_sampling(p: LoanPortfolio, q: Qubo, shots: int, seed: int) -> SamplingResult:
    """Uniform-random null baseline over ALL QUBO variables (loans + slack).

    This is what QAOA must beat. It uses the same decode functions as
    qaoa_run.py (qubo.decode_loans, qubo.residual).

    Two feasibility views:
      raw: the full bitstring has zero penalty (residual == 0);
      decoded: loan bits alone satisfy sum(amount * x) <= budget.

    Args:
        p: Loan portfolio.
        q: QUBO built from p (supplies lam and variable layout).
        shots: Number of uniform samples.
        seed: RNG seed; same seed gives identical output.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(shots, q.n_vars))
    xs = decode_loans(q, bits)
    raw_ok = residual(p, q, bits) == 0
    dec_ok = (xs @ p.amount) <= p.budget

    best_val, best_x = None, None
    for x in np.unique(xs[dec_ok], axis=0):
        val = objective(p, x, q.lam)
        if best_val is None or val < best_val:
            best_val, best_x = val, x
    opt_x = brute_force(p, q.lam).x
    return SamplingResult(
        best_objective=best_val,
        best_x=best_x,
        p_feasible_raw=float(raw_ok.mean()),
        p_feasible_decoded=float(dec_ok.mean()),
        p_optimal=float(np.mean(np.all(xs == opt_x, axis=1))),
        shots=shots,
    )
