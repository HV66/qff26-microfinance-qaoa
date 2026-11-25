"""Lambda sweep: how risk aversion changes the optimum and the greedy gap.

Run from the repo root:  python -m scripts.lambda_sweep
Writes results/lambda_sweep.csv and prints a per-lambda summary.
Does not change any default lambda.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from baseline import brute_force, greedy
from data import make_portfolio

LAMS = [0.0, 0.1, 0.3, 0.5, 1.0]
SEEDS = range(10)
N_LOANS = 8
OUT = Path("results") / "lambda_sweep.csv"
EPS = 1e-9

log = logging.getLogger(__name__)


def greedy_gap_pct(greedy_obj: float, best_obj: float) -> float:
    """Return greedy gap as a percent of |best objective| (0 if best is 0)."""
    if abs(best_obj) < EPS:
        return 0.0
    return 100.0 * (greedy_obj - best_obj) / abs(best_obj)


def run_sweep() -> list[dict[str, object]]:
    """Return one row per (seed, lam) with solver results."""
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        p = make_portfolio(N_LOANS, seed)
        x_lam0 = brute_force(p, 0.0).x
        for lam in LAMS:
            bf, gr = brute_force(p, lam), greedy(p, lam)
            rows.append(
                {
                    "seed": seed,
                    "lam": lam,
                    "bf_objective": round(bf.objective, 6),
                    "greedy_objective": round(gr.objective, 6),
                    "greedy_gap_pct": round(
                        greedy_gap_pct(gr.objective, bf.objective), 4
                    ),
                    "n_selected_bf": int(bf.x.sum()),
                    "bf_x": "".join(str(int(v)) for v in bf.x),
                    "optimum_differs_from_lam0": bool(not np.array_equal(bf.x, x_lam0)),
                }
            )
    return rows


def summarise(rows: list[dict[str, object]]) -> list[str]:
    """Return summary table lines, one per lambda."""
    header = (
        f"{'lam':>5} {'mean_gap%':>10} {'max_gap%':>9} "
        f"{'seeds_gap>0':>11} {'risk_changed_opt':>16} {'mean_n_sel':>10}"
    )
    lines = [header]
    for lam in LAMS:
        sub = [r for r in rows if r["lam"] == lam]
        gaps = [float(r["greedy_gap_pct"]) for r in sub]
        lines.append(
            f"{lam:>5.1f} {np.mean(gaps):>10.3f} {max(gaps):>9.3f} "
            f"{sum(g > EPS for g in gaps):>11d} "
            f"{sum(bool(r['optimum_differs_from_lam0']) for r in sub):>16d} "
            f"{np.mean([int(r['n_selected_bf']) for r in sub]):>10.2f}"
        )
    return lines


def main() -> None:
    """Run the sweep, save the CSV, print the summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = run_sweep()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("n=%d loans, seeds=%s, saved %s", N_LOANS, list(SEEDS), OUT)
    for line in summarise(rows):
        log.info(line)


if __name__ == "__main__":
    main()
