"""Explain the size of the optimiser's final <H>: QUBO energy spectrum statistics.

For the seed-0 instance this script computes, for every bitstring of all
n + m variables, the QUBO energy, and splits the stored QAOA states
(results/qaoa_seed0_p*.json) into probability mass on bitstrings with a zero
budget residual ("raw-feasible", zero penalty) and on bitstrings that violate
the budget equation (nonzero residual, penalty P * residual^2).

Run:  python -m scripts.spectrum_stats
Output: results/spectrum_seed0.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from data import make_portfolio
from qaoa_run import energy_diagonal, fast_probabilities, scaled_cost_diagonal
from qubo import Qubo, build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT = RESULTS / "spectrum_seed0.json"
SEED = 0
N_LOANS = 8
LAM = 0.3
PS = (1, 2, 3, 4)


def residuals(q: Qubo, amount: np.ndarray, budget: int) -> np.ndarray:
    """Return spend + slack - budget for every bitstring (bit k = variable k)."""
    n = q.n_vars
    bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    weights = np.concatenate([amount, q.slack_coeffs])
    return bits @ weights - budget


def compute(results_dir: Path = RESULTS) -> dict[str, Any]:
    """Return spectrum statistics and the per-depth split of the stored <H>."""
    pf = make_portfolio(N_LOANS, SEED)
    q = build_qubo(pf, LAM)
    energies = energy_diagonal(q)
    res = residuals(q, pf.amount, pf.budget)
    ok = res == 0
    diag = scaled_cost_diagonal(q)
    per_p = []
    for p in PS:
        path = results_dir / f"qaoa_seed{SEED}_p{p}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        probs = fast_probabilities(
            diag, q.n_vars, np.array(run["optimiser"]["best_params"])
        )
        per_p.append(
            {
                "p": p,
                "final_H": float(probs @ energies),
                "stored_final_H": float(run["optimiser"]["best_value"]),
                "mass_on_zero_residual_states": float(probs[ok].sum()),
                "H_contribution_zero_residual_states": float(
                    (probs[ok] * energies[ok]).sum()
                ),
                "H_contribution_violating_states": float(
                    (probs[~ok] * energies[~ok]).sum()
                ),
            }
        )
    return {
        "instance": {
            "seed": SEED,
            "n_loans": N_LOANS,
            "lam": LAM,
            "n_qubits": q.n_vars,
        },
        "penalty": q.penalty,
        "offset": q.offset,
        "n_bitstrings": len(energies),
        "n_zero_residual_states": int(ok.sum()),
        "max_abs_residual": int(np.abs(res).max()),
        "energy_min": float(energies.min()),
        "energy_mean_uniform": float(energies.mean()),
        "energy_max": float(energies.max()),
        "energy_range_zero_residual": [
            float(energies[ok].min()),
            float(energies[ok].max()),
        ],
        "energy_min_violating": float(energies[~ok].min()),
        "per_p": per_p,
    }


def main() -> None:
    """Write results/spectrum_seed0.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = compute()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
