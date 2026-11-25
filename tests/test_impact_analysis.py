"""Tests for scripts/impact_analysis.py."""

import numpy as np
import pytest

from baseline import brute_force
from data import make_portfolio
from scripts import impact_analysis


def test_describe_matches_direct_computation() -> None:
    pf = make_portfolio(8, 0)
    bf = brute_force(pf, 0.3)
    d = impact_analysis.describe(pf, bf.x, 0.3)
    f = bf.x.astype(bool)
    assert d["amount_deployed"] == int(pf.amount[f].sum()) <= pf.budget
    assert d["n_funded"] == int(f.sum())
    assert d["expected_return"] == pytest.approx(float(pf.ret[f].sum()))
    assert d["portfolio_risk_xT_sigma_x"] == pytest.approx(pf.portfolio_risk(bf.x))
    assert d["objective"] == pytest.approx(bf.objective)
    shares = [v for v in d["amount_share_by_sector"].values() if v is not None]
    assert sum(shares) == pytest.approx(1.0)
    assert sum(d["loans_by_sector"].values()) == d["n_funded"]


def test_zero_funded_sector_is_reported_with_zero_share() -> None:
    pf = make_portfolio(8, 0)
    x = np.zeros(pf.n, dtype=int)
    first_sector = int(pf.sector[0])
    x[0] = 1  # fund one loan only: the other two sectors get nothing
    d = impact_analysis.describe(pf, x, 0.3)
    assert d["n_funded"] == 1
    assert len(d["unfunded_sectors"]) == 2
    assert first_sector not in d["unfunded_sectors"]
    other = next(s for s in d["unfunded_sectors"])
    assert d["amount_share_by_sector"][str(other)] == 0.0


def test_empty_selection_has_no_shares() -> None:
    pf = make_portfolio(8, 1)
    d = impact_analysis.describe(pf, np.zeros(pf.n, dtype=int), 0.3)
    assert d["n_funded"] == 0
    assert all(v is None for v in d["amount_share_by_sector"].values())
    assert d["amount_deployed"] == 0


def test_compute_covers_all_cases_and_respects_budget() -> None:
    out = impact_analysis.compute()
    assert len(out["rows"]) == 5 * 3 * 2
    assert all(r["amount_deployed"] <= r["budget"] for r in out["rows"])
    assert {r["solver"] for r in out["rows"]} == {"exact", "greedy"}
