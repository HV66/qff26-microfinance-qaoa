"""Compare every saved real-hardware result with uniform random sampling.

Reads each retrieved hardware result (results/hardware_result.json for the
first job and results/hardware_result_seed{S}_n{N}_p{P}.json for later jobs),
draws the uniform-random null with the SAME shot count on the same QUBO
(baseline.random_sampling), and runs the same exact one-sided binomial test as
compare.py. Writes results/hardware_vs_random.json with one entry per job; the
top-level fields of the first job are kept so older readers still work.

Run:  python -m scripts.hardware_vs_random
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from baseline import brute_force, random_sampling
from compare import binomial_verdict
from data import make_portfolio
from qaoa_run import fast_probabilities, scaled_cost_diagonal
from qubo import build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "hardware_vs_random.json"


def hardware_result_files(results_dir: Path = RESULTS) -> list[Path]:
    """Return all retrieved hardware result files, the first job's file first."""
    legacy = results_dir / "hardware_result.json"
    later = sorted(results_dir.glob("hardware_result_seed*_n*_p*.json"))
    return ([legacy] if legacy.exists() else []) + later


def ideal_p_optimum(result: dict[str, Any]) -> float:
    """Noiseless statevector P(optimum) of the circuit that was run on hardware.

    Uses the saved QAOA angles of the job and the verified numpy simulator
    (qaoa_run.fast_probabilities). The decoded loan vector is the low n bits of
    the basis-state index (variable k is bit k); slack bits are summed out.
    """
    inst = result["instance"]
    pf = make_portfolio(inst["n_loans"], inst["seed"])
    q = build_qubo(pf, inst["lam"])
    probs = fast_probabilities(
        scaled_cost_diagonal(q), q.n_vars, np.array(inst["params"], dtype=float)
    )
    opt = brute_force(pf, inst["lam"]).x
    opt_index = sum(int(b) << k for k, b in enumerate(opt))
    low = np.arange(2**q.n_vars) & (2**q.n_loans - 1)
    return float(probs[low == opt_index].sum())


def compare(result: dict[str, Any]) -> dict[str, Any]:
    """Return hardware metrics next to a same-shots uniform-random null."""
    inst, m = result["instance"], result["metrics"]
    shots = sum(result["counts"].values())
    pf = make_portfolio(inst["n_loans"], inst["seed"])
    q = build_qubo(pf, inst["lam"])
    null = random_sampling(pf, q, shots, inst["seed"])
    hits = round(m["p_optimal"] * shots)
    p_value, verdict = binomial_verdict(hits, shots, inst["n_loans"])
    p_below = float(
        binomtest(hits, shots, 2.0 ** -inst["n_loans"], alternative="less").pvalue
    )
    return {
        "job_id": result["job_id"],
        "backend": result["backend"],
        "seed": inst["seed"],
        "n_loans": inst["n_loans"],
        "p": inst.get("p", 1),
        "transpiled_depth": result.get("transpile", {}).get("depth"),
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
        "ideal_noiseless_p_optimum": ideal_p_optimum(result),
        "binomial_p_value_above_random": p_value,
        "binomial_p_value_below_random": p_below,
        "verdict": verdict,
    }


def compare_all(results_dir: Path = RESULTS) -> dict[str, Any]:
    """Return one comparison per hardware job plus the first job's fields on top."""
    jobs = [
        compare(json.loads(path.read_text(encoding="utf-8")))
        for path in hardware_result_files(results_dir)
    ]
    if not jobs:
        raise FileNotFoundError("no hardware result files found")
    return {**jobs[0], "jobs": jobs, "n_jobs": len(jobs)}


def main() -> None:
    """Write results/hardware_vs_random.json from all saved hardware results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = compare_all()
    OUT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s (%d hardware jobs)", OUT_FILE, out["n_jobs"])


if __name__ == "__main__":
    main()
