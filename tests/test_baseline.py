"""Tests for baseline.py."""

import numpy as np
import pytest

from baseline import brute_force, greedy, random_sampling
from data import make_portfolio
from qubo import Qubo, build_qubo, decode_bitstring, is_feasible, objective

LAM = 0.3


@pytest.mark.parametrize("seed", range(5))
def test_brute_force_matches_qubo_minimum(seed: int) -> None:
    p = make_portfolio(8, seed)
    res = brute_force(p, LAM)
    assert res.objective == pytest.approx(build_qubo(p, LAM).all_energies().min())
    assert is_feasible(p, res.x)
    assert res.objective == pytest.approx(objective(p, res.x, LAM))
    assert res.runtime_s >= 0.0


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("seed", range(10))
def test_greedy_feasible_and_never_beats_brute_force(seed: int, lam: float) -> None:
    p = make_portfolio(8, seed)
    g, b = greedy(p, lam), brute_force(p, lam)
    assert is_feasible(p, g.x)
    assert g.objective >= b.objective - 1e-9
    assert g.objective == pytest.approx(objective(p, g.x, lam))


def test_random_sampling_reproducible_and_seed_sensitive() -> None:
    p = make_portfolio(8, 0)
    q = build_qubo(p, LAM)
    a, b = random_sampling(p, q, 2000, 5), random_sampling(p, q, 2000, 5)
    assert a.best_objective == b.best_objective
    assert (a.p_feasible_raw, a.p_feasible_decoded, a.p_optimal) == (
        b.p_feasible_raw,
        b.p_feasible_decoded,
        b.p_optimal,
    )
    c = random_sampling(p, q, 2000, 6)
    assert (c.p_feasible_raw, c.p_feasible_decoded) != (
        a.p_feasible_raw,
        a.p_feasible_decoded,
    )


def test_random_sampling_views_are_consistent() -> None:
    p = make_portfolio(8, 1)
    q = build_qubo(p, LAM)
    r = random_sampling(p, q, 5000, 0)
    assert 0.0 <= r.p_feasible_raw <= r.p_feasible_decoded <= 1.0
    assert r.p_optimal <= r.p_feasible_decoded
    assert r.best_objective is not None
    assert r.best_objective >= brute_force(p, LAM).objective - 1e-9


def test_decode_known_bitstring_qubit_order() -> None:
    q = Qubo(np.zeros((5, 5)), 0.0, 3, np.array([1, 2]), 1.0, LAM)
    # Qiskit key is little-endian: variable k is character [-1 - k].
    # "10110": v0='0', v1='1', v2='1', v3='0', v4='1' -> x = [0, 1, 1].
    assert decode_bitstring(q, "10110").tolist() == [0, 1, 1]
    assert decode_bitstring(q, "00001").tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
        decode_bitstring(q, "101")
