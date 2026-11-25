"""Tests for data.py."""

import numpy as np
import pytest

from data import RHO_CROSS, RHO_SAME, make_portfolio

SEEDS = range(10)


@pytest.mark.parametrize("seed", SEEDS)
def test_sigma_symmetric_and_psd(seed: int) -> None:
    s = make_portfolio(10, seed).sigma
    assert np.array_equal(s, s.T)
    assert np.linalg.eigvalsh(s).min() >= -1e-12


@pytest.mark.parametrize("seed", SEEDS)
def test_same_sector_more_correlated_than_cross(seed: int) -> None:
    p = make_portfolio(10, seed)
    vol = np.sqrt(np.diag(p.sigma))
    corr = p.sigma / np.outer(vol, vol)
    for i in range(p.n):
        for j in range(i):
            expected = RHO_SAME if p.sector[i] == p.sector[j] else RHO_CROSS
            assert corr[i, j] == pytest.approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_same_seed_is_identical(seed: int) -> None:
    a, b = make_portfolio(10, seed), make_portfolio(10, seed)
    for field in ("amount", "ret", "sector", "sigma"):
        assert np.array_equal(getattr(a, field), getattr(b, field))
    assert a.budget == b.budget


def test_different_seed_differs() -> None:
    assert not np.array_equal(make_portfolio(8, 1).sigma, make_portfolio(8, 2).sigma)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [8, 10])
def test_amounts_integer_in_range_and_sectors_used(seed: int, n: int) -> None:
    p = make_portfolio(n, seed)
    assert p.amount.dtype.kind == "i"
    assert p.amount.min() >= 1 and p.amount.max() <= 8
    assert set(p.sector) == {0, 1, 2}
    assert (p.ret > 0).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_budget_binds(seed: int) -> None:
    p = make_portfolio(10, seed)
    assert p.budget == max(1, round(0.4 * p.amount.sum()))
    assert p.amount.sum() > p.budget  # funding everything is infeasible
    assert p.amount.min() <= p.budget  # at least one loan fits


def test_portfolio_risk_scales_with_amount() -> None:
    p = make_portfolio(8, 0)
    x = np.zeros(p.n)
    x[0] = 1
    assert p.portfolio_risk(x) == pytest.approx(p.amount[0] ** 2 * p.sigma[0, 0])
    assert p.portfolio_risk(np.zeros(p.n)) == 0.0


@pytest.mark.parametrize("n", [1, 11])
def test_bad_n_rejected(n: int) -> None:
    with pytest.raises(ValueError):
        make_portfolio(n)
