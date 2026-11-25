"""Solve a small loan-selection problem from a CSV file.

Research prototype on synthetic data. Not credit advice: a human credit
officer stays in the loop for every real lending decision.

Input CSV columns (exactly): amount, expected_return, sector.
  amount           positive integer (units)
  expected_return  positive float (return if the loan is funded)
  sector           any string label (mapped to integer ids in sorted order)

The CSV carries no covariance, so the loss covariance is built exactly as in
data.py (sector-block correlation RHO_SAME within a sector, RHO_CROSS across)
with the SAME per-loan volatility for every loan, the midpoint of
data.VOL_RANGE. At most 10 loans.

Run:  python solve_csv.py examples/loans_example.csv --budget 12 [--p 2]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from baseline import brute_force, greedy
from data import VOL_RANGE, LoanPortfolio, _build_sigma
from qaoa_run import optimise, sample_counts
from qubo import build_qubo, decode_bitstring, is_feasible, objective

log = logging.getLogger(__name__)

MAX_LOANS = 10
MAX_QUBITS = 20
REQUIRED_COLUMNS = ("amount", "expected_return", "sector")
DISCLAIMER = (
    "Research prototype on synthetic data. This is NOT credit advice. "
    "A human credit officer stays in the loop for every real decision."
)


class CsvInputError(ValueError):
    """Raised when the CSV file or the arguments are invalid."""


def _parse_amount(raw: str, row_no: int) -> int:
    """Parse a positive integer amount, rejecting floats and non-numbers."""
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise CsvInputError(f"row {row_no}: amount {raw!r} is not an integer") from exc
    if value <= 0:
        raise CsvInputError(f"row {row_no}: amount must be positive, got {value}")
    return value


def _parse_return(raw: str, row_no: int) -> float:
    """Parse a positive finite float expected return."""
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise CsvInputError(
            f"row {row_no}: expected_return {raw!r} is not a number"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise CsvInputError(
            f"row {row_no}: expected_return must be positive and finite, got {raw!r}"
        )
    return value


def read_loans(path: Path) -> tuple[list[int], list[float], list[str]]:
    """Read and validate the CSV; return (amounts, returns, sector labels).

    Raises:
        CsvInputError: On a missing file, bad header, empty file, too many
            loans, or any invalid value.
    """
    if not path.is_file():
        raise CsvInputError(f"file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        if sorted(header) != sorted(REQUIRED_COLUMNS):
            raise CsvInputError(
                f"columns must be exactly {list(REQUIRED_COLUMNS)}, got {header}"
            )
        amounts: list[int] = []
        returns: list[float] = []
        sectors: list[str] = []
        for row_no, row in enumerate(reader, start=2):
            clean = {k.strip(): (v or "") for k, v in row.items() if k is not None}
            if len(clean) != len(REQUIRED_COLUMNS):
                raise CsvInputError(f"row {row_no}: unexpected number of fields")
            sector = clean["sector"].strip()
            if not sector:
                raise CsvInputError(f"row {row_no}: sector is empty")
            amounts.append(_parse_amount(clean["amount"], row_no))
            returns.append(_parse_return(clean["expected_return"], row_no))
            sectors.append(sector)
    if not amounts:
        raise CsvInputError("the CSV file has no loan rows")
    if len(amounts) > MAX_LOANS:
        raise CsvInputError(
            f"{len(amounts)} loans given, but at most {MAX_LOANS} are supported "
            "(qubit budget of the simulator)"
        )
    return amounts, returns, sectors


def build_portfolio(
    amounts: list[int], returns: list[float], labels: list[str], budget: int
) -> tuple[LoanPortfolio, list[str]]:
    """Build a LoanPortfolio; return it with the sorted sector label list.

    Sigma is built with data._build_sigma and the same volatility for every
    loan (midpoint of data.VOL_RANGE).

    Raises:
        CsvInputError: If the budget is below the smallest loan amount.
    """
    if budget < min(amounts):
        raise CsvInputError(
            f"budget {budget} is below the smallest loan amount {min(amounts)}"
        )
    names = sorted(set(labels))
    sector = np.array([names.index(s) for s in labels], dtype=int)
    vol = np.full(len(amounts), (VOL_RANGE[0] + VOL_RANGE[1]) / 2.0)
    pf = LoanPortfolio(
        amount=np.array(amounts, dtype=int),
        ret=np.array(returns, dtype=float),
        sector=sector,
        sigma=_build_sigma(sector, vol),
        budget=int(budget),
        seed=0,
    )
    return pf, names


def qaoa_portfolio(
    pf: LoanPortfolio, lam: float, p: int, shots: int, seed: int
) -> np.ndarray | None:
    """Return the best budget-feasible sampled portfolio, or None if none."""
    q = build_qubo(pf, lam)
    if q.n_vars > MAX_QUBITS:
        raise CsvInputError(
            f"QUBO needs {q.n_vars} qubits (> {MAX_QUBITS}); lower the budget"
        )
    opt = optimise(q, p, seed)
    counts = sample_counts(q, p, np.array(opt["best_params"]), shots, seed)
    best_x: np.ndarray | None = None
    best_val = np.inf
    for key in sorted(counts):
        x = decode_bitstring(q, key)
        if not is_feasible(pf, x):
            continue
        val = objective(pf, x, lam)
        if val < best_val:
            best_val, best_x = val, x
    return best_x


def sector_audit(
    pf: LoanPortfolio, names: list[str], x: np.ndarray | None
) -> dict[str, Any]:
    """Return per-sector audit numbers for one selection (None = none found)."""
    if x is None:
        return {"found": False}
    funded = np.asarray(x, dtype=bool)
    deployed = int(pf.amount[funded].sum())
    rows = []
    for sid, name in enumerate(names):
        in_s = funded & (pf.sector == sid)
        amt = int(pf.amount[in_s].sum())
        rows.append(
            {
                "sector": name,
                "requested": int((pf.sector == sid).sum()),
                "funded": int(in_s.sum()),
                "amount": amt,
                "share": (amt / deployed) if deployed else 0.0,
            }
        )
    return {
        "found": True,
        "rows": rows,
        "deployed": deployed,
        "zero_funded_sectors": [r["sector"] for r in rows if r["funded"] == 0],
        "expected_return": float(pf.ret[funded].sum()),
        "risk": pf.portfolio_risk(np.asarray(x)),
    }


def print_audit(title: str, audit: dict[str, Any], obj: float | None) -> None:
    """Print one portfolio's sector audit table."""
    print(f"\n== {title} ==")
    if not audit["found"]:
        print("no budget-feasible sample found")
        return
    print(f"objective (lower is better): {obj:.4f}")
    print(f"{'sector':<14}{'requested':>10}{'funded':>8}{'amount':>8}{'share':>8}")
    for r in audit["rows"]:
        print(
            f"{r['sector']:<14}{r['requested']:>10}{r['funded']:>8}"
            f"{r['amount']:>8}{r['share']:>8.1%}"
        )
    zero = audit["zero_funded_sectors"] or "none"
    print(f"amount deployed: {audit['deployed']}")
    print(f"sectors with zero funded loans: {zero}")
    print(f"expected return: {audit['expected_return']:.4f}")
    print(f"risk x^T Sigma x: {audit['risk']:.4f}")


def _positive_int(text: str) -> int:
    """argparse type: a positive integer."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "csv_path", type=Path, help="CSV: amount,expected_return,sector"
    )
    parser.add_argument("--budget", type=_positive_int, required=True)
    parser.add_argument("--p", type=_positive_int, default=2, help="QAOA layers")
    parser.add_argument("--shots", type=_positive_int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=0.3)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lam < 0:
        parser.error("--lam must be >= 0")
    try:
        amounts, returns, labels = read_loans(args.csv_path)
        budget = min(args.budget, sum(amounts))
        pf, names = build_portfolio(amounts, returns, labels, budget)
        qaoa_x = qaoa_portfolio(pf, args.lam, args.p, args.shots, args.seed)
    except CsvInputError as exc:
        parser.error(str(exc))
    print(DISCLAIMER)
    print(
        f"\nLoans: {pf.n}, sectors: {names}, budget: {pf.budget} "
        f"(requested {args.budget}; capped at the total requested amount)"
    )
    print(
        "Assumption: the CSV has no covariance. Sigma uses sector-block "
        "correlation (0.6 same sector, 0.1 across) and the SAME volatility "
        f"{(VOL_RANGE[0] + VOL_RANGE[1]) / 2.0:.2f} for every loan "
        "(midpoint of the synthetic range)."
    )
    print(f"lambda={args.lam}, QAOA p={args.p}, shots={args.shots}, seed={args.seed}")
    exact = brute_force(pf, args.lam)
    greedy_res = greedy(pf, args.lam)
    q_obj = None if qaoa_x is None else objective(pf, qaoa_x, args.lam)
    print_audit(
        "Exact (brute force)", sector_audit(pf, names, exact.x), exact.objective
    )
    print_audit("Greedy", sector_audit(pf, names, greedy_res.x), greedy_res.objective)
    print_audit("QAOA (best feasible sample)", sector_audit(pf, names, qaoa_x), q_obj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
