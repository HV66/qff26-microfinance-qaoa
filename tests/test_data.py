"""Tests for data.py."""

import numpy as np
import pytest

from data import make_portfolio


def test_same_seed_is_identical() -> None:
    a, b = make_portfolio(8, 3), make_portfolio(8, 3)
    assert np.array_equal(a.amount, b.amount)
    assert np.array_equal(a.ret, b.ret)
    assert np.array_equal(a.risk, b.risk)
    assert a.budget == b.budget


def test_different_seed_differs() -> None:
    assert not np.array_equal(make_portfolio(8, 1).ret, make_portfolio(8, 2).ret)


@pytest.mark.parametrize("seed", range(5))
def test_values_valid_and_budget_binds(seed: int) -> None:
    p = make_portfolio(10, seed)
    assert (p.amount >= 1).all() and (p.ret > 0).all() and (p.risk > 0).all()
    assert 1 <= p.budget < p.amount.sum()


@pytest.mark.parametrize("n", [1, 11])
def test_bad_n_rejected(n: int) -> None:
    with pytest.raises(ValueError):
        make_portfolio(n)
