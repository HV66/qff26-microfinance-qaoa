"""Simulated annealing directly on the penalised QUBO energy.

SA works on ALL variables (loans + slack bits), exactly the search space QAOA
sees, with single-bit-flip moves and a geometric cooling schedule.

Fixed schedule (identical for every instance, NOT tuned per instance):
    T_start = T_START_FRAC * max|Q_ij|   (scale-relative, so units cancel)
    T_end   = T_END_FRAC   * max|Q_ij|
    T_k     = T_start * (T_end / T_start) ** (k / (STEPS - 1))
    STEPS   = 4000 single-flip proposals per restart, RESTARTS = 10 restarts.
Metropolis acceptance: accept if dE <= 0 else with probability exp(-dE / T).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from data import LoanPortfolio
from qubo import Qubo, decode_loans, is_feasible, objective

log = logging.getLogger(__name__)

T_START_FRAC = 0.05
T_END_FRAC = 1e-4
STEPS = 4000
RESTARTS = 10


@dataclass(frozen=True)
class SAResult:
    """Result of simulated annealing.

    Attributes:
        bits: Best full bit vector (loans + slack) by QUBO energy.
        x: Decoded loan vector (first n bits).
        energy: QUBO energy of ``bits``.
        feasible: True if the decoded x respects the budget.
        objective: Shared objective of x (lower is better), even if infeasible.
        runtime_s: Wall time in seconds (time.perf_counter).
        seed: RNG seed used.
    """

    bits: np.ndarray
    x: np.ndarray
    energy: float
    feasible: bool
    objective: float
    runtime_s: float
    seed: int


def _anneal_once(
    sym: np.ndarray, q: Qubo, rng: np.random.Generator, t0: float, t1: float
) -> tuple[np.ndarray, float]:
    """Run one annealing restart; return (best bits, best energy w/o offset)."""
    n = q.n_vars
    z = rng.integers(0, 2, size=n).astype(float)
    diag = np.diag(q.Q)
    energy = float(z @ q.Q @ z)
    best_z, best_e = z.copy(), energy
    ratio = (t1 / t0) ** (1.0 / (STEPS - 1))
    temp = t0
    flips = rng.integers(0, n, size=STEPS)
    uniforms = rng.random(STEPS)
    for k in range(STEPS):
        i = int(flips[k])
        # Flip delta: (1 - 2 z_i) * (Q_ii + sum_{j != i} (Q_ij + Q_ji) z_j).
        field = diag[i] + sym[i] @ z - sym[i, i] * z[i]
        delta = (1.0 - 2.0 * z[i]) * field
        if delta <= 0.0 or uniforms[k] < np.exp(-delta / temp):
            z[i] = 1.0 - z[i]
            energy += delta
            if energy < best_e:
                best_z, best_e = z.copy(), energy
        temp *= ratio
    return best_z, best_e


def simulated_annealing(p: LoanPortfolio, q: Qubo, seed: int) -> SAResult:
    """Anneal the QUBO energy over all variables with seeded restarts.

    Args:
        p: Loan portfolio (for decoding, feasibility and objective).
        q: QUBO built from p.
        seed: RNG seed; same seed gives identical output.

    Returns:
        SAResult for the lowest-energy bit vector over all restarts.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    scale = float(np.abs(q.Q).max())
    t0, t1 = T_START_FRAC * scale, T_END_FRAC * scale
    sym = q.Q + q.Q.T
    best_z, best_e = None, np.inf
    for _ in range(RESTARTS):
        z, e = _anneal_once(sym, q, rng, t0, t1)
        if e < best_e:
            best_z, best_e = z, e
    bits = best_z.astype(int)
    x = decode_loans(q, bits)
    feasible = is_feasible(p, x)
    if not feasible:
        log.warning("SA best bitstring decodes to an infeasible loan vector")
    return SAResult(
        bits=bits,
        x=x,
        energy=float(q.energy(bits)),
        feasible=feasible,
        objective=objective(p, x, q.lam),
        runtime_s=time.perf_counter() - start,
        seed=seed,
    )
