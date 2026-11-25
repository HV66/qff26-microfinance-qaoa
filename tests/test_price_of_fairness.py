"""Tests for scripts/price_of_fairness.py."""

import numpy as np
import pytest

from baseline import brute_force
from data import LoanPortfolio, make_portfolio
from qubo import is_feasible
from scripts import price_of_fairness as pof


def _hand(
    amount: list[int], ret: list[float], sector: list[int], budget: int
) -> LoanPortfolio:
    n = len(amount)
    return LoanPortfolio(
        amount=np.array(amount),
        ret=np.array(ret),
        sector=np.array(sector),
        sigma=np.zeros((n, n)),
        budget=budget,
        seed=-1,
    )


def test_filter_correctness_on_hand_instance() -> None:
    pf = _hand([1, 1, 1, 1], [1.0, 0.9, 0.1, 0.05], [0, 0, 1, 1], 2)
    assert not pof.covers_all_sectors(pf, np.array([1, 1, 0, 0]))
    assert pof.covers_all_sectors(pf, np.array([1, 0, 1, 0]))
    unc, con = pof.enumerate_optima(pf, 0.3)
    assert unc is not None and con is not None
    assert unc[1].tolist() == [1, 1, 0, 0]
    assert unc[0] == pytest.approx(-1.9)
    assert con[1].tolist() == [1, 0, 1, 0]
    assert con[0] == pytest.approx(-1.1)


def test_infeasible_when_budget_cannot_cover_sectors() -> None:
    pf = _hand([5, 5], [1.0, 1.0], [0, 1], 5)
    unc, con = pof.enumerate_optima(pf, 0.3)
    assert unc is not None
    assert con is None


def test_zero_loss_when_unconstrained_already_covers() -> None:
    pf = _hand([1, 1, 1, 1], [1.0, 0.1, 0.9, 0.05], [0, 0, 1, 1], 2)
    unc, con = pof.enumerate_optima(pf, 0.3)
    assert unc is not None and con is not None
    assert pof.covers_all_sectors(pf, unc[1])
    assert con[0] == pytest.approx(unc[0])
    assert con[1].tolist() == unc[1].tolist()


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", pof.LAMS)
def test_real_instances(seed: int, lam: float) -> None:
    pf = make_portfolio(8, seed)
    row = pof.analyse(seed, lam)
    bf = brute_force(pf, lam)
    assert row["unconstrained_objective"] == pytest.approx(bf.objective)
    assert row["unconstrained_x"] == bf.x.tolist()
    if row["feasible"]:
        x = np.array(row["constrained_x"])
        assert row["loss_abs"] >= -1e-12
        assert is_feasible(pf, x)
        assert pof.covers_all_sectors(pf, x)
        assert row["constrained"]["amount_deployed"] <= pf.budget
        if row["unconstrained_covers_all_sectors"]:
            assert row["loss_abs"] == pytest.approx(0.0)
    else:
        assert row["loss_abs"] is None


def test_slack_qubit_arithmetic_hand_example() -> None:
    pf = _hand([1] * 6, [1.0] * 6, [0, 0, 0, 1, 2, 2], 3)
    # sector sizes 3, 1, 2 -> (2).bit_length()=2, (0).bit_length()=0, 1
    assert pof.extra_slack_qubits(pf) == 3


def test_summary_counts() -> None:
    rows = [
        {"lam": 0.3, "feasible": True, "loss_abs": 0.0, "loss_pct": 0.0},
        {"lam": 0.3, "feasible": True, "loss_abs": 0.5, "loss_pct": 10.0},
        {"lam": 0.3, "feasible": False, "loss_abs": None, "loss_pct": None},
    ]
    s = pof.summarise(rows)["0.3"]
    assert s["n_loss_positive"] == 1
    assert s["mean_loss_pct"] == pytest.approx(5.0)
    assert s["max_loss_pct"] == pytest.approx(10.0)
    assert s["n_infeasible"] == 1
