"""CVaR ablation: QAOA trained on CVaR_alpha of the QUBO energy (alpha 0.1).

Preregistered in docs/preregistration.md (Step 3). The objective is the exact
CVaR of the QUBO energy under the QAOA output distribution: sort basis states
by energy ascending, take probability mass ``alpha`` from the lowest energies
(a fractional share of the boundary state) and divide by ``alpha``. Restarts,
maxiter, start angles, shots, sampler path and metrics match qaoa_run.

Run:  python qaoa_cvar.py
Output: results/cvar_seed{S}_p{P}.json and results/cvar_vs_expectation.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

import qaoa_run
from baseline import brute_force, random_sampling
from compare import binomial_verdict
from data import make_portfolio
from qubo import Qubo, build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
ALPHA = 0.1
SEEDS = (0, 1, 2, 3, 4)
PS = (1, 2, 3)
LAM = 0.3
N_LOANS = 8


def cvar_sorted(probs: np.ndarray, sorted_energies: np.ndarray, alpha: float) -> float:
    """CVaR_alpha when ``probs`` is already ordered like ``sorted_energies``.

    Args:
        probs: Probabilities in ascending-energy order.
        sorted_energies: Energies in ascending order.
        alpha: Tail mass in (0, 1].

    Returns:
        Sum of the lowest ``alpha`` probability mass times energy, over alpha.
    """
    cum = np.cumsum(probs)
    prev = cum - probs
    mass = np.clip(np.minimum(cum, alpha) - np.minimum(prev, alpha), 0.0, None)
    return float(mass @ sorted_energies / alpha)


def cvar(probs: np.ndarray, energies: np.ndarray, alpha: float = ALPHA) -> float:
    """Exact CVaR_alpha of an energy under a discrete distribution.

    Args:
        probs: Probability of each basis state (sums to 1).
        energies: Energy of each basis state.
        alpha: Tail mass in (0, 1]; alpha = 1 gives the plain expectation.

    Raises:
        ValueError: If alpha is outside (0, 1].
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    order = np.argsort(energies, kind="stable")
    return cvar_sorted(probs[order], energies[order], alpha)


def optimise_cvar(
    q: Qubo,
    p: int,
    seed: int,
    restarts: int = qaoa_run.RESTARTS,
    maxiter: int = qaoa_run.MAXITER,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """COBYLA on CVaR_alpha, with the same start angles as qaoa_run.optimise.

    RNG seeded by seed * 100 + p; beta ~ U[0, pi), gamma ~ U[0, 2 pi), drawn in
    the same order. Returns best_params, best_cvar, the expectation at those
    angles, per-restart finals, histories and n_evals.
    """
    cost_diag = qaoa_run.scaled_cost_diagonal(q)
    energies = qaoa_run.energy_diagonal(q)
    order = np.argsort(energies, kind="stable")
    sorted_e = energies[order]
    rng = np.random.default_rng(seed * 100 + p)
    best_value, best_params = np.inf, None
    histories: list[list[float]] = []
    finals: list[float] = []
    n_evals = 0
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, 2 * np.pi, p)])
        trace: list[float] = []

        def fun(params: np.ndarray, trace: list[float] = trace) -> float:
            probs = qaoa_run.fast_probabilities(cost_diag, q.n_vars, params)
            val = cvar_sorted(probs[order], sorted_e, alpha)
            trace.append(min(val, trace[-1]) if trace else val)
            return val

        res = minimize(fun, x0, method="COBYLA", options={"maxiter": maxiter})
        histories.append(trace)
        finals.append(float(res.fun))
        n_evals += len(trace)
        if res.fun < best_value:
            best_value, best_params = float(res.fun), res.x
    assert best_params is not None
    final_probs = qaoa_run.fast_probabilities(cost_diag, q.n_vars, best_params)
    return {
        "best_params": best_params.tolist(),
        "best_cvar": best_value,
        "expectation_at_best_params": float(final_probs @ energies),
        "restart_values": finals,
        "n_evals": n_evals,
        "histories": histories,
    }


def load_random_null(
    seed: int, p: int, results_dir: Path = RESULTS
) -> dict[str, Any] | None:
    """Read the random_null block of results/qaoa_seed{S}_p{P}.json (read-only)."""
    path = results_dir / f"qaoa_seed{seed}_p{p}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("random_null")


def run_one(
    seed: int,
    p: int,
    lam: float = LAM,
    n_loans: int = N_LOANS,
    shots: int = qaoa_run.SHOTS,
    restarts: int = qaoa_run.RESTARTS,
    maxiter: int = qaoa_run.MAXITER,
    results_dir: Path = RESULTS,
) -> dict[str, Any]:
    """Run the CVaR-trained QAOA on one instance and return all metrics."""
    pf = make_portfolio(n_loans, seed)
    q = build_qubo(pf, lam)
    bf = brute_force(pf, lam)
    t0 = time.perf_counter()
    opt = optimise_cvar(q, p, seed, restarts, maxiter)
    opt_time = time.perf_counter() - t0
    params = np.array(opt["best_params"])
    t1 = time.perf_counter()
    counts = qaoa_run.sample_counts(q, p, params, shots, seed)
    sample_time = time.perf_counter() - t1
    null = load_random_null(seed, p, results_dir)
    null_source = f"qaoa_seed{seed}_p{p}.json (read-only)"
    if null is None:
        rs = random_sampling(pf, q, shots, seed)
        null = {
            "p_optimal": rs.p_optimal,
            "p_feasible_decoded": rs.p_feasible_decoded,
            "best_objective": rs.best_objective,
            "shots": rs.shots,
        }
        null_source = "baseline.random_sampling (no qaoa_seed file for this instance)"
    energies = qaoa_run.energy_diagonal(q)
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
            "objective": "CVaR",
            "alpha": ALPHA,
            "penalty": q.penalty,
            "budget": pf.budget,
        },
        "exact": {
            "objective": bf.objective,
            "x": bf.x.tolist(),
            "qubo_min_energy": float(energies.min()),
        },
        "optimiser": opt,
        "sampling": qaoa_run.sample_metrics(pf, q, counts, bf.x, bf.objective),
        "circuit": qaoa_run.circuit_cost(q, p, seed),
        "random_null": {**null, "source": null_source},
        "runtime_s": {
            "optimise": opt_time,
            "sample": sample_time,
            "total": opt_time + sample_time,
        },
    }


def save(result: dict[str, Any], results_dir: Path = RESULTS) -> Path:
    """Write results/cvar_seed{S}_p{P}.json."""
    cfg = result["config"]
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"cvar_seed{cfg['seed']}_p{cfg['p']}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


def _side(res: dict[str, Any], expectation_key: str) -> dict[str, Any]:
    """Summarise one run (CVaR or expectation) for the comparison table."""
    s, cfg = res["sampling"], res["config"]
    shots = int(cfg["shots"])
    hits = round(s["p_optimal"] * shots)
    p_value, verdict = binomial_verdict(hits, shots, int(cfg["n_loans"]))
    return {
        "p_optimum": s["p_optimal"],
        "hits": hits,
        "p_feasible_decoded": s["p_feasible_decoded"],
        "best_gap_pct": s["best_feasible_gap_pct"],
        "expectation_at_angles": res["optimiser"][expectation_key],
        "binom_p_value": p_value,
        "verdict": verdict,
    }


def comparison_row(cvar_res: dict[str, Any], exp_res: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side row for one (seed, p): CVaR run versus expectation run."""
    cfg = cvar_res["config"]
    return {
        "seed": cfg["seed"],
        "p": cfg["p"],
        "cvar": _side(cvar_res, "expectation_at_best_params"),
        "expectation": _side(exp_res, "best_value"),
    }


def build_comparison(results_dir: Path = RESULTS) -> dict[str, Any]:
    """Join every cvar file with its existing expectation file (read-only)."""
    rows = []
    for seed in SEEDS:
        for p in PS:
            c_path = results_dir / f"cvar_seed{seed}_p{p}.json"
            e_path = results_dir / f"qaoa_seed{seed}_p{p}.json"
            if not (c_path.exists() and e_path.exists()):
                log.warning("skipping seed %d p=%d: missing file", seed, p)
                continue
            rows.append(
                comparison_row(
                    json.loads(c_path.read_text(encoding="utf-8")),
                    json.loads(e_path.read_text(encoding="utf-8")),
                )
            )
    return {"alpha": ALPHA, "rows": rows}


def main() -> None:
    """Run the 15 preregistered CVaR runs and write the comparison file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for seed in SEEDS:
        for p in PS:
            r = run_one(seed, p)
            save(r)
            log.info(
                "seed=%d p=%d cvar=%.4f P(opt)=%.4f t=%.1fs",
                seed,
                p,
                r["optimiser"]["best_cvar"],
                r["sampling"]["p_optimal"],
                r["runtime_s"]["total"],
            )
    out = build_comparison()
    (RESULTS / "cvar_vs_expectation.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
