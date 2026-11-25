"""Who gets funded: impact and sector-concentration analysis of the solutions.

For each seed and each solver (exact optimum, greedy) and lambda in
{0.1, 0.3, 1.0}, report: loans funded, amount deployed vs budget, expected
return, portfolio risk x^T Sigma x, and sector shares including any sector
with zero funded loans. This is descriptive only: the QUBO has no fairness
constraint and none is added here.

Run:  python -m scripts.impact_analysis
Output: results/impact_analysis.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from baseline import brute_force, greedy
from data import N_SECTORS, LoanPortfolio, make_portfolio
from qubo import objective

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "impact_analysis.json"
SEEDS = (0, 1, 2, 3, 4)
LAMS = (0.1, 0.3, 1.0)
N_LOANS = 8


def describe(pf: LoanPortfolio, x: np.ndarray, lam: float) -> dict[str, Any]:
    """Return impact metrics for one selection x.

    Sector shares are the fraction of FUNDED AMOUNT and of funded loans in
    each sector; a sector with no funded loan has share 0.0 and is listed in
    ``unfunded_sectors``. Shares are None if nothing is funded.
    """
    xv = np.asarray(x, dtype=int)
    funded = xv.astype(bool)
    amount_deployed = int(pf.amount[funded].sum())
    n_funded = int(funded.sum())
    sector_ids = list(range(N_SECTORS))
    in_portfolio = sorted({int(s) for s in pf.sector})
    share_amount: dict[str, float | None] = {}
    share_loans: dict[str, float | None] = {}
    loans_by_sector: dict[str, int] = {}
    for s in sector_ids:
        in_s = funded & (pf.sector == s)
        loans_by_sector[str(s)] = int(in_s.sum())
        share_amount[str(s)] = (
            float(pf.amount[in_s].sum() / amount_deployed) if amount_deployed else None
        )
        share_loans[str(s)] = float(in_s.sum() / n_funded) if n_funded else None
    unfunded = [s for s in in_portfolio if loans_by_sector[str(s)] == 0]
    requested_by_sector = {
        str(s): int(pf.amount[pf.sector == s].sum()) for s in sector_ids
    }
    return {
        "x": xv.tolist(),
        "n_funded": n_funded,
        "n_requested": pf.n,
        "amount_deployed": amount_deployed,
        "budget": int(pf.budget),
        "budget_used_fraction": amount_deployed / pf.budget,
        "total_requested_amount": int(pf.amount.sum()),
        "expected_return": float(pf.ret[funded].sum()),
        "portfolio_risk_xT_sigma_x": pf.portfolio_risk(xv),
        "objective": objective(pf, xv, lam),
        "loans_by_sector": loans_by_sector,
        "requested_amount_by_sector": requested_by_sector,
        "amount_share_by_sector": share_amount,
        "loan_share_by_sector": share_loans,
        "unfunded_sectors": unfunded,
    }


def summarise_by_lambda(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean funded loans, return and risk of the exact optimum, per lambda.

    Also counts exact-optimum portfolios that leave a sector with zero funded
    loans and portfolios that leave part of the budget unlent.
    """
    out = []
    for lam in sorted({r["lam"] for r in rows}):
        sel = [r for r in rows if r["solver"] == "exact" and r["lam"] == lam]
        n = len(sel)
        out.append(
            {
                "lam": lam,
                "n_seeds": n,
                "mean_loans_funded": sum(r["n_funded"] for r in sel) / n,
                "mean_expected_return": sum(r["expected_return"] for r in sel) / n,
                "mean_portfolio_risk": sum(r["portfolio_risk_xT_sigma_x"] for r in sel)
                / n,
                "n_with_unfunded_sector": sum(1 for r in sel if r["unfunded_sectors"]),
                "n_budget_not_fully_used": sum(
                    1 for r in sel if r["amount_deployed"] < r["budget"]
                ),
            }
        )
    return out


def compute() -> dict[str, Any]:
    """Return the full analysis for all seeds, lambdas and both solvers."""
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        pf = make_portfolio(N_LOANS, seed)
        for lam in LAMS:
            for name, res in (
                ("exact", brute_force(pf, lam)),
                ("greedy", greedy(pf, lam)),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "lam": lam,
                        "solver": name,
                        **describe(pf, res.x, lam),
                    }
                )
    return {
        "n_loans": N_LOANS,
        "n_sectors": N_SECTORS,
        "note": (
            "Descriptive only. The QUBO has no fairness constraint; sectors are "
            "synthetic labels, not protected attributes."
        ),
        "exact_by_lambda": summarise_by_lambda(rows),
        "rows": rows,
    }


def main() -> None:
    """Write results/impact_analysis.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = compute()
    OUT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    n_unfunded = sum(1 for r in out["rows"] if r["unfunded_sectors"])
    log.info(
        "wrote %s (%d rows, %d with an unfunded sector)",
        OUT_FILE,
        len(out["rows"]),
        n_unfunded,
    )


if __name__ == "__main__":
    main()
