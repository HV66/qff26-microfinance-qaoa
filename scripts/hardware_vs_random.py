"""Compare the saved real-hardware counts with uniform random sampling.

Reads results/hardware_result.json (written by retrieve_hardware.py), draws the
uniform-random null with the SAME shot count on the same QUBO
(baseline.random_sampling), and runs the same exact one-sided binomial test as
compare.py. Writes results/hardware_vs_random.json.

Run:  python -m scripts.hardware_vs_random
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from baseline import random_sampling
from compare import binomial_verdict
from data import make_portfolio
from qubo import build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
IN_FILE = RESULTS / "hardware_result.json"
OUT_FILE = RESULTS / "hardware_vs_random.json"


def compare(result: dict[str, Any]) -> dict[str, Any]:
    """Return hardware metrics next to a same-shots uniform-random null."""
    inst, m = result["instance"], result["metrics"]
    shots = sum(result["counts"].values())
    pf = make_portfolio(inst["n_loans"], inst["seed"])
    q = build_qubo(pf, inst["lam"])
    null = random_sampling(pf, q, shots, inst["seed"])
    hits = round(m["p_optimal"] * shots)
    p_value, verdict = binomial_verdict(hits, shots, inst["n_loans"])
    return {
        "job_id": result["job_id"],
        "backend": result["backend"],
        "n_loans": inst["n_loans"],
        "shots": shots,
        "hardware": {
            "optimum_hits": hits,
            "p_optimum": m["p_optimal"],
            "p_feasible_decoded": m["p_feasible_decoded"],
            "p_feasible_raw": m["p_feasible_raw"],
        },
        "random_null": {
            "optimum_hits": round(null.p_optimal * shots),
            "p_optimum": null.p_optimal,
            "p_feasible_decoded": null.p_feasible_decoded,
            "p_feasible_raw": null.p_feasible_raw,
        },
        "uniform_expectation_p_optimum": 2.0 ** -inst["n_loans"],
        "expected_hits_uniform": shots * 2.0 ** -inst["n_loans"],
        "binomial_p_value_above_random": p_value,
        "verdict": verdict,
    }


def main() -> None:
    """Write results/hardware_vs_random.json from the saved hardware result."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = json.loads(IN_FILE.read_text(encoding="utf-8"))
    OUT_FILE.write_text(json.dumps(compare(result), indent=1), encoding="utf-8")
    log.info("wrote %s", OUT_FILE)


if __name__ == "__main__":
    main()
