"""Tests for sa_baseline.py."""

import numpy as np
import pytest

from baseline import brute_force
from data import make_portfolio
from qubo import build_qubo, is_feasible, objective
from sa_baseline import simulated_annealing


def _setup(seed: int, lam: float = 0.3):
    p = make_portfolio(8, seed)
    return p, build_qubo(p, lam)


def test_reproducible_with_seed() -> None:
    p, q = _setup(0)
    a = simulated_annealing(p, q, seed=3)
    b = simulated_annealing(p, q, seed=3)
    assert np.array_equal(a.bits, b.bits)
    assert a.energy == b.energy


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sane_and_never_beats_exact(seed: int) -> None:
    p, q = _setup(seed)
    res = simulated_annealing(p, q, seed=seed)
    bf = brute_force(p, 0.3)
    assert res.x.shape == (p.n,)
    assert res.feasible == is_feasible(p, res.x)
    assert res.objective == pytest.approx(objective(p, res.x, 0.3))
    if res.feasible:
        assert res.objective >= bf.objective - 1e-9
    # Energy bookkeeping must match a fresh evaluation.
    assert res.energy == pytest.approx(q.energy(res.bits))
    # Penalised energy should never be above the empty selection (E = 0 + slack
    # penalty can be zero), i.e. SA must find something better than +1.
    assert res.energy < 1.0


def test_runtime_positive() -> None:
    p, q = _setup(0)
    assert simulated_annealing(p, q, seed=0).runtime_s > 0.0
