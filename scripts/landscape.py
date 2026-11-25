"""Step 6a: QAOA landscape of the 5-loan, p=1 instance (seeds 0 and 1).

Two exact, noiseless surfaces over a (beta, gamma) grid:

* P(optimum): probability that the decoded loan vector (low n_loans bits of the
  basis index, slack bits summed out) equals the brute-force optimum.
* <H>: expectation of the QUBO energy (qaoa_run.energy_diagonal), the quantity
  the optimiser minimised.

Angle convention (qaoa_run.fast_probabilities, build_ansatz): params =
[beta_1..beta_p, gamma_1..gamma_p]; for p = 1 that is [beta, gamma].

Grid (preregistration, Step 6): beta in [0, pi) with 101 points, i.e. the
points k * pi / 101 for k = 0..100 (step pi/101, pi itself excluded);
gamma in [-pi, pi] with 201 points (np.linspace, step pi/100, both ends in).

The saved optimiser angles (results/hardware_params_seed{S}_n5_p1.json) are
compared with the grid optima. The optimiser's beta may lie outside [0, pi);
RX(2*beta) = exp(-i*beta*X) changes only by the global phase -1 when beta
grows by pi, so beta is reduced modulo pi. Gamma is NOT periodic in general
(the scaled cost has non-integer coefficients); a saved gamma outside
[-pi, pi] would only be wrapped for the grid lookup, and the exact values at
the saved angles are always computed directly, never read off the grid.

Run:  python -m scripts.landscape
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline import brute_force
from data import make_portfolio
from qaoa_run import (
    energy_diagonal,
    fast_probabilities,
    scaled_cost_diagonal,
)
from qubo import Qubo, build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
SEEDS = (0, 1)
N_LOANS = 5
P_LAYERS = 1
LAM = 0.3
N_BETA = 101
N_GAMMA = 201


def beta_grid(n_beta: int = N_BETA) -> np.ndarray:
    """Return n_beta evenly spaced betas on [0, pi), step pi / n_beta."""
    return np.arange(n_beta) * (np.pi / n_beta)


def gamma_grid(n_gamma: int = N_GAMMA) -> np.ndarray:
    """Return n_gamma evenly spaced gammas on [-pi, pi], both ends included."""
    return np.linspace(-np.pi, np.pi, n_gamma)


def optimum_mask(q: Qubo, optimum_x: np.ndarray) -> np.ndarray:
    """Return a boolean mask over basis states whose loan bits equal optimum_x.

    Variable k is bit k of the basis index; slack bits (index >= n_loans) are
    free, so the mask sums over them.
    """
    opt_index = sum(int(b) << k for k, b in enumerate(optimum_x))
    low = np.arange(2**q.n_vars) & (2**q.n_loans - 1)
    return low == opt_index


def evaluate(
    cost_diag: np.ndarray,
    energies: np.ndarray,
    mask: np.ndarray,
    n_qubits: int,
    beta: float,
    gamma: float,
) -> tuple[float, float]:
    """Return exact (P(optimum), <H>) for one p=1 angle pair [beta, gamma]."""
    probs = fast_probabilities(cost_diag, n_qubits, np.array([beta, gamma]))
    return float(probs[mask].sum()), float(probs @ energies)


def surfaces(
    q: Qubo,
    optimum_x: np.ndarray,
    n_beta: int = N_BETA,
    n_gamma: int = N_GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (P(optimum), <H>) arrays of shape (n_beta, n_gamma)."""
    cost_diag = scaled_cost_diagonal(q)
    energies = energy_diagonal(q)
    mask = optimum_mask(q, optimum_x)
    betas, gammas = beta_grid(n_beta), gamma_grid(n_gamma)
    p_opt = np.empty((n_beta, n_gamma))
    expect = np.empty((n_beta, n_gamma))
    for i, b in enumerate(betas):
        for j, g in enumerate(gammas):
            p_opt[i, j], expect[i, j] = evaluate(
                cost_diag, energies, mask, q.n_vars, float(b), float(g)
            )
    return p_opt, expect


def nearest_index(
    beta: float, gamma: float, n_beta: int = N_BETA, n_gamma: int = N_GAMMA
) -> tuple[int, int]:
    """Return the grid index (i, j) nearest to the angle pair.

    Beta is reduced modulo pi (exact symmetry) and the nearest point is taken
    on the circle, so a beta just below pi maps to index 0. Gamma is wrapped
    into [-pi, pi] before the lookup.
    """
    step_b = np.pi / n_beta
    i = round((beta % np.pi) / step_b) % n_beta
    g = (gamma + np.pi) % (2 * np.pi) - np.pi
    j = round((g + np.pi) / (2 * np.pi / (n_gamma - 1)))
    return i, min(max(j, 0), n_gamma - 1)


def grid_distance(
    a: tuple[int, int], b: tuple[int, int], n_beta: int = N_BETA
) -> dict[str, int]:
    """Return grid steps between two index pairs (beta circular, gamma linear)."""
    d_beta = abs(a[0] - b[0])
    return {"beta_steps": min(d_beta, n_beta - d_beta), "gamma_steps": abs(a[1] - b[1])}


def analyse_seed(
    seed: int,
    saved_params: list[float],
    n_beta: int = N_BETA,
    n_gamma: int = N_GAMMA,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compute surfaces and the saved-angle checks for one seed.

    Returns (summary dict, P(optimum) surface, <H> surface).
    """
    pf = make_portfolio(N_LOANS, seed)
    q = build_qubo(pf, LAM)
    bf = brute_force(pf, LAM)
    p_opt, expect = surfaces(q, bf.x, n_beta, n_gamma)
    betas, gammas = beta_grid(n_beta), gamma_grid(n_gamma)

    cost_diag = scaled_cost_diagonal(q)
    energies = energy_diagonal(q)
    mask = optimum_mask(q, bf.x)
    beta_s, gamma_s = float(saved_params[0]), float(saved_params[1])
    p_saved, h_saved = evaluate(cost_diag, energies, mask, q.n_vars, beta_s, gamma_s)
    p_saved_mod, h_saved_mod = evaluate(
        cost_diag, energies, mask, q.n_vars, beta_s % np.pi, gamma_s
    )

    i_p = np.unravel_index(int(np.argmax(p_opt)), p_opt.shape)
    i_h = np.unravel_index(int(np.argmin(expect)), expect.shape)
    idx_p, idx_h = (int(i_p[0]), int(i_p[1])), (int(i_h[0]), int(i_h[1]))
    idx_s = nearest_index(beta_s, gamma_s, n_beta, n_gamma)
    d_p = grid_distance(idx_s, idx_p, n_beta)
    d_h = grid_distance(idx_s, idx_h, n_beta)
    p_max = float(p_opt[idx_p])
    h_min = float(expect[idx_h])
    at_p = d_p["beta_steps"] <= 1 and d_p["gamma_steps"] <= 1
    at_h = d_h["beta_steps"] <= 1 and d_h["gamma_steps"] <= 1
    uniform = 1.0 / 2**N_LOANS

    summary: dict[str, Any] = {
        "seed": seed,
        "n_loans": N_LOANS,
        "n_qubits": q.n_vars,
        "p": P_LAYERS,
        "lam": LAM,
        "optimum_x": bf.x.tolist(),
        "uniform_p_optimum": uniform,
        "grid": {
            "n_beta": n_beta,
            "beta_range": "[0, pi), points k*pi/n_beta, step pi/n_beta",
            "beta_step": float(np.pi / n_beta),
            "n_gamma": n_gamma,
            "gamma_range": "[-pi, pi], np.linspace, both ends included",
            "gamma_step": float(gammas[1] - gammas[0]),
            "n_points": n_beta * n_gamma,
        },
        "saved_angles": {
            "beta": beta_s,
            "gamma": gamma_s,
            "beta_mod_pi": beta_s % np.pi,
            "p_optimum_exact": p_saved,
            "p_optimum_exact_beta_mod_pi": p_saved_mod,
            "expectation_exact": h_saved,
            "expectation_exact_beta_mod_pi": h_saved_mod,
            "nearest_grid_index": list(idx_s),
            "nearest_grid_beta": float(betas[idx_s[0]]),
            "nearest_grid_gamma": float(gammas[idx_s[1]]),
        },
        "grid_max_p_optimum": {
            "value": p_max,
            "beta": float(betas[idx_p[0]]),
            "gamma": float(gammas[idx_p[1]]),
            "index": list(idx_p),
            "expectation_there": float(expect[idx_p]),
        },
        "grid_min_expectation": {
            "value": h_min,
            "beta": float(betas[idx_h[0]]),
            "gamma": float(gammas[idx_h[1]]),
            "index": list(idx_h),
            "p_optimum_there": float(p_opt[idx_h]),
        },
        "ratio_saved_p_optimum_to_grid_max": p_saved / p_max,
        "saved_p_optimum_over_uniform": p_saved / uniform,
        "grid_max_p_optimum_over_uniform": p_max / uniform,
        "grid_distance_to_p_optimum_argmax": d_p,
        "grid_distance_to_expectation_argmin": d_h,
        "saved_angles_within_one_step_of_p_optimum_max": at_p,
        "saved_angles_within_one_step_of_expectation_min": at_h,
        "objective_disagreement": {
            "p_optimum_at_expectation_argmin": float(p_opt[idx_h]),
            "expectation_at_p_optimum_argmax": float(expect[idx_p]),
            "p_optimum_loss_from_following_expectation": p_max - float(p_opt[idx_h]),
            "expectation_gain_from_following_p_optimum": float(expect[idx_p]) - h_min,
        },
    }
    summary["statement"] = describe(summary)
    return summary, p_opt, expect


def describe(s: dict[str, Any]) -> str:
    """Return a plain sentence of what the numbers in a seed summary show."""
    at_p = s["saved_angles_within_one_step_of_p_optimum_max"]
    at_h = s["saved_angles_within_one_step_of_expectation_min"]
    dp = s["grid_distance_to_p_optimum_argmax"]
    dh = s["grid_distance_to_expectation_argmin"]
    return (
        f"Seed {s['seed']}: saved angles are "
        f"{'' if at_p else 'NOT '}within one grid step of the P(optimum) grid "
        f"maximum (distance {dp['beta_steps']} beta steps, {dp['gamma_steps']} "
        f"gamma steps) and {'' if at_h else 'NOT '}within one grid step of the "
        f"<H> grid minimum (distance {dh['beta_steps']} beta steps, "
        f"{dh['gamma_steps']} gamma steps)."
    )


def plot_seed(
    summary: dict[str, Any], p_opt: np.ndarray, expect: np.ndarray, path: Path
) -> None:
    """Save the 2-panel heat map (P(optimum) left, <H> right) for one seed."""
    n_beta, n_gamma = p_opt.shape
    betas, gammas = beta_grid(n_beta), gamma_grid(n_gamma)
    extent = (gammas[0], gammas[-1], betas[-1], betas[0])
    saved = summary["saved_angles"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    panels = (
        (
            axes[0],
            p_opt,
            "P(optimum), exact noiseless",
            "viridis",
            "grid_max_p_optimum",
        ),
        (
            axes[1],
            expect,
            "<H> (QUBO energy), exact noiseless",
            "magma_r",
            "grid_min_expectation",
        ),
    )
    for ax, surf, title, cmap, key in panels:
        im = ax.imshow(surf, aspect="auto", extent=extent, cmap=cmap)
        fig.colorbar(im, ax=ax, label=title.split(",")[0])
        ax.scatter(
            [saved["gamma"]],
            [saved["beta_mod_pi"]],
            marker="x",
            s=140,
            c="red",
            linewidths=2.5,
            label="saved optimiser angles (beta mod pi)",
        )
        opt = summary[key]
        ax.scatter(
            [opt["gamma"]],
            [opt["beta"]],
            marker="o",
            s=120,
            facecolors="none",
            edgecolors="white",
            linewidths=2.5,
            label="grid optimum of this panel",
        )
        ax.set_title(f"Seed {summary['seed']}: {title}", fontsize=13)
        ax.set_xlabel("gamma (rad)", fontsize=12)
        ax.set_ylabel("beta (rad)", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.suptitle(
        f"QAOA landscape, 5 loans, p=1, lambda={LAM} "
        f"(uniform P(optimum) = 1/32 = {summary['uniform_p_optimum']:.5f})",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_saved_params(seed: int, results_dir: Path = RESULTS) -> list[float]:
    """Return the saved optimiser angles [beta, gamma] for a 5-loan, p=1 seed."""
    path = results_dir / f"hardware_params_seed{seed}_n{N_LOANS}_p{P_LAYERS}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [float(v) for v in data["params"]]


def main() -> None:
    """Run both seeds, write the PNGs and JSON files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summaries = []
    for seed in SEEDS:
        summary, p_opt, expect = analyse_seed(seed, load_saved_params(seed))
        out = RESULTS / f"landscape_seed{seed}_n{N_LOANS}_p{P_LAYERS}.json"
        out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        plot_seed(summary, p_opt, expect, PLOTS / out.with_suffix(".png").name)
        log.info("wrote %s", out)
        summaries.append(summary)
    combined = {
        "step": "6a landscape",
        "seeds": list(SEEDS),
        "per_seed": summaries,
        "statements": [s["statement"] for s in summaries],
    }
    (RESULTS / "landscape.json").write_text(
        json.dumps(combined, indent=1), encoding="utf-8"
    )
    for s in summaries:
        print(json.dumps({k: v for k, v in s.items() if k != "grid"}, indent=1))


if __name__ == "__main__":
    main()
