"""Record, per seed, whether the unbalanced QUBO's minimum decodes to the exact optimum.

Written for docs/preregistration.md Step 2. Nothing is tuned: the coefficient rule is
qubo_unbalanced.unbalanced_coefficients and is not changed here.

Run:  python -m scripts.unbalanced_min_decode
Output: results/unbalanced_min_decode.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from baseline import brute_force
from data import make_portfolio
from qubo import decode_loans, is_feasible, objective
from qubo_unbalanced import build_unbalanced_qubo, unbalanced_coefficients

log = logging.getLogger(__name__)

OUT_FILE = Path("results") / "unbalanced_min_decode.json"
LAM = 0.3
N_LOANS = 8
SEEDS = range(10)


def one_seed(seed: int) -> dict[str, Any]:
    """Return the minimum-energy decode and the exact optimum for one seed."""
    pf = make_portfolio(N_LOANS, seed)
    q = build_unbalanced_qubo(pf, LAM)
    best = int(np.argmin(q.all_energies()))
    bits = np.array([(best >> k) & 1 for k in range(q.n_vars)], dtype=int)
    decoded = decode_loans(q, bits)
    bf = brute_force(pf, LAM)
    l1, l2 = unbalanced_coefficients(pf)
    return {
        "seed": seed,
        "passes": bool(np.array_equal(decoded, bf.x)),
        "decoded_x": decoded.tolist(),
        "decoded_feasible": is_feasible(pf, decoded),
        "decoded_spend": int(pf.amount @ decoded),
        "decoded_objective": objective(pf, decoded, LAM),
        "exact_x": bf.x.tolist(),
        "exact_spend": int(pf.amount @ bf.x),
        "exact_objective": bf.objective,
        "budget": int(pf.budget),
        "l1": l1,
        "l2": l2,
        "penalty_one_unit_unspent": l2 - l1,
    }


def compute() -> dict[str, Any]:
    """Return outcomes for seeds 0 to 9."""
    rows = [one_seed(s) for s in SEEDS]
    return {
        "lam": LAM,
        "n_loans": N_LOANS,
        "pass_seeds": [r["seed"] for r in rows if r["passes"]],
        "fail_seeds": [r["seed"] for r in rows if not r["passes"]],
        "rows": rows,
    }


def main() -> None:
    """Write results/unbalanced_min_decode.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = compute()
    OUT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("pass %s | fail %s", out["pass_seeds"], out["fail_seeds"])


if __name__ == "__main__":
    main()
