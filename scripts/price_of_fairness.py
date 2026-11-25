"""Price of fairness: cost of forcing every requested sector to get a loan.

Classical exhaustive enumeration of all 2^n loan vectors (no QAOA). Compares
the unconstrained exact optimum (budget only) with the optimum that also
requires every sector with at least one loan request to have at least one
funded loan. Also reports, by arithmetic only, how many extra slack qubits the
constraint would need in a QUBO (nothing is built or run).

Run:  python -m scripts.price_of_fairness
Output: results/price_of_fairness.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from data import LoanPortfolio, make_portfolio
from qubo import build_qubo, is_feasible, objective

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "price_of_fairness.json"
SEEDS = (0, 1, 2, 3, 4)
LAMS = (0.1, 0.3, 1.0)
N_LOANS = 8


def covers_all_sectors(pf: LoanPortfolio, x: np.ndarray) -> bool:
    """Return True if every requested sector has at least one funded loan."""
    funded = np.asarray(x, dtype=bool)
    requested = {int(s) for s in pf.sector}
    covered = {int(s) for s in pf.sector[funded]}
    return requested <= covered


def enumerate_optima(
    pf: LoanPortfolio, lam: float
) -> tuple[tuple[float, np.ndarray] | None, tuple[float, np.ndarray] | None]:
    """Enumerate all 2^n vectors; return (unconstrained, constrained) optima.

    Unconstrained: budget only. Constrained: budget plus the sector filter.
    Each is (objective, x) or None if no feasible vector exists. Ties go to
    the smallest bitstring index (strict improvement only).
    """
    best_u: tuple[float, np.ndarray] | None = None
    best_c: tuple[float, np.ndarray] | None = None
    for k in range(2**pf.n):
        x = np.array([(k >> i) & 1 for i in range(pf.n)], dtype=int)
        if not is_feasible(pf, x):
            continue
        val = objective(pf, x, lam)
        if best_u is None or val < best_u[0]:
            best_u = (val, x)
        if covers_all_sectors(pf, x) and (best_c is None or val < best_c[0]):
            best_c = (val, x)
    return best_u, best_c


def extra_slack_qubits(pf: LoanPortfolio) -> int:
    """Return sum over sectors of (n_s - 1).bit_length() (arithmetic only)."""
    total = 0
    for s in sorted({int(v) for v in pf.sector}):
        n_s = int((pf.sector == s).sum())
        total += (n_s - 1).bit_length()
    return total


def _describe(pf: LoanPortfolio, x: np.ndarray) -> dict[str, Any]:
    """Return portfolio metrics for one selection."""
    funded = np.asarray(x, dtype=bool)
    return {
        "x": np.asarray(x, dtype=int).tolist(),
        "loans_funded": int(funded.sum()),
        "amount_deployed": int(pf.amount[funded].sum()),
        "budget": int(pf.budget),
        "expected_return": float(pf.ret[funded].sum()),
        "portfolio_risk": pf.portfolio_risk(x),
    }


def analyse(seed: int, lam: float, n_loans: int = N_LOANS) -> dict[str, Any]:
    """Return one row of the price-of-fairness table."""
    pf = make_portfolio(n_loans, seed)
    unc, con = enumerate_optima(pf, lam)
    if unc is None:
        raise RuntimeError("x = 0 is always feasible, so this cannot happen")
    n_vars = build_qubo(pf, lam).n_vars
    extra = extra_slack_qubits(pf)
    row: dict[str, Any] = {
        "seed": seed,
        "lam": lam,
        "unconstrained_objective": unc[0],
        "unconstrained_x": unc[1].tolist(),
        "unconstrained_covers_all_sectors": covers_all_sectors(pf, unc[1]),
        "feasible": con is not None,
        "extra_slack_qubits": extra,
        "qubo_qubits_existing": n_vars,
        "qubo_qubits_with_constraint": n_vars + extra,
    }
    if con is None:
        row.update(
            constrained_objective=None,
            constrained_x=None,
            loss_abs=None,
            loss_pct=None,
            note="no portfolio satisfies the budget and the sector filter",
        )
        return row
    loss = con[0] - unc[0]
    row.update(
        constrained_objective=con[0],
        constrained_x=con[1].tolist(),
        loss_abs=loss,
        loss_pct=(100.0 * loss / abs(unc[0])) if abs(unc[0]) > 1e-12 else None,
        constrained=_describe(pf, con[1]),
        unconstrained=_describe(pf, unc[1]),
    )
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the per-lambda summary over all rows."""
    out: dict[str, dict[str, Any]] = {}
    for lam in sorted({r["lam"] for r in rows}):
        sub = [r for r in rows if r["lam"] == lam]
        ok = [r for r in sub if r["feasible"]]
        pcts = [r["loss_pct"] for r in ok if r["loss_pct"] is not None]
        out[str(lam)] = {
            "n_cases": len(sub),
            "n_loss_positive": sum(1 for r in ok if r["loss_abs"] > 1e-12),
            "mean_loss_pct": float(np.mean(pcts)) if pcts else None,
            "max_loss_pct": float(np.max(pcts)) if pcts else None,
            "n_infeasible": len(sub) - len(ok),
        }
    return out


def main() -> None:
    """Run all seeds and lambdas and write results/price_of_fairness.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = [analyse(s, lam) for s in SEEDS for lam in LAMS]
    result = {
        "constraint": "every sector with >= 1 loan request has >= 1 funded loan",
        "seeds": list(SEEDS),
        "lams": list(LAMS),
        "n_loans": N_LOANS,
        "rows": rows,
        "by_lambda": summarise(rows),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for r in rows:
        log.info(
            "seed=%d lam=%.1f feasible=%s loss_pct=%s extra_qubits=%d",
            r["seed"],
            r["lam"],
            r["feasible"],
            r["loss_pct"],
            r["extra_slack_qubits"],
        )
    log.info("by_lambda: %s", json.dumps(result["by_lambda"]))


if __name__ == "__main__":
    main()
