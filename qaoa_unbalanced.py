"""QAOA on the slack-free (unbalanced-penalty) QUBO, Step 2 of the preregistration.

Reuses qaoa_run's simulator, COBYLA settings, start-angle rule
(RNG seeded by seed * 100 + p), sampler path and metrics. The QUBO has n_loans
qubits, so "raw feasible" (zero slack residual) is not applicable and is not
reported; only decoded feasibility is.

Run:  python -m qaoa_unbalanced
Outputs: results/unbalanced_seed{S}_p{P}.json and results/unbalanced_vs_slack.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

import qaoa_run
from baseline import brute_force
from compare import binomial_verdict
from data import LoanPortfolio, make_portfolio
from qaoa_run import circuit_cost, energy_diagonal, sample_counts, sample_metrics
from qubo import Qubo
from qubo_unbalanced import build_unbalanced_qubo, unbalanced_coefficients

log = logging.getLogger(__name__)

RESULTS = Path("results")
SEEDS = (0, 1, 2, 3, 4)
PS = (1, 2, 3)


def uniform_null(pf: LoanPortfolio) -> dict[str, float]:
    """Return the analytic uniform-sampling null over the n_loans bits.

    P(optimum) = 2^-n. P(feasible) is the fraction of all 2^n loan vectors
    whose total amount fits the budget.
    """
    n = pf.n
    idx = np.arange(2**n)
    bits = (idx[:, None] >> np.arange(n)) & 1
    fits = (bits @ pf.amount) <= pf.budget
    return {"p_optimal": 2.0**-n, "p_feasible_decoded": float(fits.mean())}


def run_one(
    seed: int,
    p: int,
    lam: float = qaoa_run.LAM,
    n_loans: int = qaoa_run.N_LOANS,
    shots: int = qaoa_run.SHOTS,
    restarts: int = qaoa_run.RESTARTS,
    maxiter: int = qaoa_run.MAXITER,
) -> dict[str, Any]:
    """Run QAOA at depth p on the unbalanced QUBO of one instance."""
    pf = make_portfolio(n_loans, seed)
    q: Qubo = build_unbalanced_qubo(pf, lam)
    bf = brute_force(pf, lam)
    energies = energy_diagonal(q)
    l1, l2 = unbalanced_coefficients(pf)

    t0 = time.perf_counter()
    opt = qaoa_run.optimise(q, p, seed, restarts=restarts, maxiter=maxiter)
    opt_time = time.perf_counter() - t0
    params = np.array(opt["best_params"])
    counts = sample_counts(q, p, params, shots, seed)
    sampling = sample_metrics(pf, q, counts, bf.x, bf.objective)
    sampling.pop("p_feasible_raw")  # no slack bits: not applicable

    hits = round(sampling["p_optimal"] * shots)
    p_value, verdict = binomial_verdict(hits, shots, n_loans)
    best_idx = int(np.argmin(energies))
    min_bits = np.array([(best_idx >> k) & 1 for k in range(q.n_vars)])
    return {
        "config": {
            "seed": seed,
            "p": p,
            "lam": lam,
            "n_loans": n_loans,
            "shots": shots,
            "n_qubits": q.n_vars,
            "maxiter": maxiter,
            "restarts": restarts,
            "optimiser": "COBYLA",
            "penalty": q.penalty,
            "l1": l1,
            "l2": l2,
            "budget": pf.budget,
            "raw_feasible": "not applicable (no slack bits)",
        },
        "exact": {
            "objective": bf.objective,
            "x": bf.x.tolist(),
            "qubo_min_energy": float(energies.min()),
            "qubo_min_decodes_to_optimum": bool(np.array_equal(min_bits, bf.x)),
        },
        "optimiser": {
            "best_value": opt["best_value"],
            "best_params": opt["best_params"],
            "restart_values": opt["restart_values"],
            "n_evals": opt["n_evals"],
        },
        "sampling": {
            **sampling,
            "hits": hits,
            "binom_p_value_above_random": p_value,
            "verdict": verdict,
        },
        "circuit": circuit_cost(q, p, seed),
        "null": {**uniform_null(pf), "shots": shots, "kind": "analytic uniform"},
        "runtime_s": {"optimise": opt_time},
    }


def save(result: dict[str, Any], results_dir: Path = RESULTS) -> Path:
    """Write results/unbalanced_seed{S}_p{P}.json."""
    cfg = result["config"]
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"unbalanced_seed{cfg['seed']}_p{cfg['p']}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


def _slack_row(path: Path) -> dict[str, Any] | None:
    """Return the slack-version summary fields from a read-only result file."""
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    shots = int(r["config"]["shots"])
    hits = round(r["sampling"]["p_optimal"] * shots)
    _, verdict = binomial_verdict(hits, shots, int(r["config"]["n_loans"]))
    return {
        "qubits": r["config"]["n_qubits"],
        "depth": r["circuit"]["depth"],
        "two_qubit_gates": r["circuit"]["two_qubit_gates"],
        "p_optimum": r["sampling"]["p_optimal"],
        "p_feasible_decoded": r["sampling"]["p_feasible_decoded"],
        "verdict": verdict,
    }


def build_comparison(
    seeds: tuple[int, ...] = SEEDS,
    ps: tuple[int, ...] = PS,
    results_dir: Path = RESULTS,
) -> list[dict[str, Any]]:
    """Return side-by-side rows, unbalanced vs the saved slack runs."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for p in ps:
            upath = results_dir / f"unbalanced_seed{seed}_p{p}.json"
            if not upath.exists():
                continue
            u = json.loads(upath.read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": seed,
                    "p": p,
                    "unbalanced": {
                        "qubits": u["config"]["n_qubits"],
                        "depth": u["circuit"]["depth"],
                        "two_qubit_gates": u["circuit"]["two_qubit_gates"],
                        "p_optimum": u["sampling"]["p_optimal"],
                        "p_feasible_decoded": u["sampling"]["p_feasible_decoded"],
                        "verdict": u["sampling"]["verdict"],
                    },
                    "slack": _slack_row(results_dir / f"qaoa_seed{seed}_p{p}.json"),
                }
            )
    return rows


def main() -> None:
    """CLI: run all seeds and depths, then write the side-by-side file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--ps", type=int, nargs="+", default=list(PS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for seed in args.seeds:
        for p in args.ps:
            res = run_one(seed, p)
            save(res)
            s = res["sampling"]
            log.info(
                "seed=%d p=%d P(opt)=%.4f feas=%.3f verdict=%s depth=%d",
                seed,
                p,
                s["p_optimal"],
                s["p_feasible_decoded"],
                s["verdict"],
                res["circuit"]["depth"],
            )
    rows = build_comparison()
    (RESULTS / "unbalanced_vs_slack.json").write_text(
        json.dumps({"rows": rows}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
