"""Live end-to-end demo: microfinance loan selection, QAOA vs classical.

Run:  python demo.py            (default demo budget, well under 60 s)
      python demo.py --quick    (smaller budget, a few seconds)

The demo uses FEWER COBYLA restarts / iterations / shots than the full runs
(restarts=5, maxiter=200, shots=4096). Full-budget numbers over several seeds
live in results/results.md. This demo makes no quantum-advantage claim: it
prints a computed note comparing QAOA with the classical baselines.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from baseline import brute_force, greedy, random_sampling
from data import LoanPortfolio, make_portfolio
from qaoa_run import (
    circuit_cost,
    fast_probabilities,
    optimise,
    sample_metrics,
    scaled_cost_diagonal,
)
from qubo import (
    Qubo,
    bits_from_string,
    build_qubo,
    decode_loans,
    is_feasible,
    objective,
)

log = logging.getLogger(__name__)

SEED = 0
N_LOANS = 8
LAM = 0.3
DEMO_BUDGET = {"restarts": 2, "maxiter": 100, "shots": 4096}
QUICK_BUDGET = {"restarts": 1, "maxiter": 40, "shots": 1024}
EPS = 1e-9
OUT_DIR = Path("results")


def gap_pct(value: float | None, optimum: float) -> float | None:
    """Return the gap to the optimum as a percent of |optimum| (None if no value)."""
    if value is None:
        return None
    return 100.0 * (value - optimum) / abs(optimum) if abs(optimum) > EPS else 0.0


def show_instance(pf: LoanPortfolio) -> None:
    """Print the loans (sector, amount, return) and the budget."""
    print(f"\n[1] Instance: seed={pf.seed}, {pf.n} loans, budget={pf.budget}")
    print("    loan  sector  amount  return")
    for i in range(pf.n):
        print(
            f"    {i:>4}  {int(pf.sector[i]):>6}  {int(pf.amount[i]):>6}"
            f"  {pf.ret[i]:>6.3f}"
        )


def decoded_objective_counts(
    pf: LoanPortfolio, q: Qubo, counts: dict[str, int]
) -> Counter[float]:
    """Return shot counts keyed by the decoded objective of feasible samples."""
    out: Counter[float] = Counter()
    for key, c in counts.items():
        x = decode_loans(q, bits_from_string(q, key))
        if is_feasible(pf, x):
            out[round(objective(pf, x, q.lam), 6)] += c
    return out


def fast_sample_counts(
    q: Qubo, params: np.ndarray, shots: int, seed: int
) -> dict[str, int]:
    """Sample the optimised QAOA state with seeded multinomial draws.

    Uses qaoa_run.fast_probabilities (checked against the Qiskit circuit in
    tests/test_qaoa.py). The demo avoids qaoa_run.sample_counts here because
    the StatevectorSampler path is too slow for a live demo on 12 qubits.
    Bitstrings follow the Qiskit convention: qubit 0 is the rightmost character.
    """
    probs = fast_probabilities(scaled_cost_diagonal(q), q.n_vars, params)
    probs = probs / probs.sum()
    draws = np.random.default_rng(seed).multinomial(shots, probs)
    return {format(k, f"0{q.n_vars}b"): int(c) for k, c in enumerate(draws) if c}


def run_qaoa_row(
    pf: LoanPortfolio,
    q: Qubo,
    p: int,
    opt_x: np.ndarray,
    opt_obj: float,
    budget: dict[str, int],
) -> tuple[dict[str, Any], Counter[float]]:
    """Run QAOA at depth p with the demo budget; return (table row, histogram)."""
    t0 = time.perf_counter()
    opt = optimise(q, p, SEED, restarts=budget["restarts"], maxiter=budget["maxiter"])
    params = np.array(opt["best_params"])
    counts = fast_sample_counts(q, params, budget["shots"], SEED)
    runtime = time.perf_counter() - t0
    m = sample_metrics(pf, q, counts, opt_x, opt_obj)
    cost = circuit_cost(q, p, SEED)
    row = {
        "method": f"QAOA p={p}",
        "objective": m["best_feasible_objective"],
        "gap_pct": m["best_feasible_gap_pct"],
        "p_optimum": m["p_optimal"],
        "depth": cost["depth"],
        "runtime_s": runtime,
    }
    return row, decoded_objective_counts(pf, q, counts)


def build_rows(
    pf: LoanPortfolio, q: Qubo, budget: dict[str, int]
) -> tuple[list[dict[str, Any]], dict[int, Counter[float]]]:
    """Run every method and return (table rows, QAOA objective histograms)."""
    bf = brute_force(pf, q.lam)
    gr = greedy(pf, q.lam)
    t0 = time.perf_counter()
    rs = random_sampling(pf, q, budget["shots"], SEED)
    rs_time = time.perf_counter() - t0
    rows: list[dict[str, Any]] = [
        {
            "method": "Exact brute force",
            "objective": bf.objective,
            "gap_pct": 0.0,
            "p_optimum": None,
            "depth": None,
            "runtime_s": bf.runtime_s,
        },
        {
            "method": "Greedy",
            "objective": gr.objective,
            "gap_pct": gap_pct(gr.objective, bf.objective),
            "p_optimum": None,
            "depth": None,
            "runtime_s": gr.runtime_s,
        },
        {
            "method": f"Random ({budget['shots']} shots)",
            "objective": rs.best_objective,
            "gap_pct": gap_pct(rs.best_objective, bf.objective),
            "p_optimum": rs.p_optimal,
            "depth": None,
            "runtime_s": rs_time,
        },
    ]
    hists: dict[int, Counter[float]] = {}
    for p in (1, 2):
        row, hists[p] = run_qaoa_row(pf, q, p, bf.x, bf.objective, budget)
        rows.append(row)
    return rows, hists


def _fmt(value: float | None, spec: str) -> str:
    """Format an optional number; None prints as a dash."""
    return "-" if value is None else format(value, spec)


def format_table(rows: list[dict[str, Any]]) -> str:
    """Return the comparison table as text."""
    head = (
        f"{'method':<24}{'objective':>11}{'gap %':>9}{'P(optimum)':>12}"
        f"{'depth':>7}{'runtime s':>11}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['method']:<24}{_fmt(r['objective'], '.4f'):>11}"
            f"{_fmt(r['gap_pct'], '.2f'):>9}{_fmt(r['p_optimum'], '.4f'):>12}"
            f"{_fmt(r['depth'], 'd'):>7}{r['runtime_s']:>11.3f}"
        )
    return "\n".join(lines)


def honesty_note(rows: list[dict[str, Any]]) -> str:
    """Return a computed one-line note on QAOA versus the classical methods."""
    qaoa = [r for r in rows if r["method"].startswith("QAOA")]
    classical = [r for r in rows if r["method"] in ("Exact brute force", "Greedy")]
    best_q = min(r["gap_pct"] for r in qaoa if r["gap_pct"] is not None)
    best_c = min(r["gap_pct"] for r in classical)
    if best_c <= best_q + EPS:
        return (
            "Note: classical exact/greedy match or beat QAOA here "
            f"(best classical gap {best_c:.2f}% vs best QAOA gap {best_q:.2f}%). "
            "No quantum advantage is claimed."
        )
    return (
        f"Note: best QAOA gap {best_q:.2f}% is lower than the best classical gap "
        f"{best_c:.2f}% on this single instance; one small instance is not evidence "
        "of quantum advantage."
    )


def save_plot(
    rows: list[dict[str, Any]],
    hists: dict[int, Counter[float]],
    optimum: float,
    path: Path,
) -> None:
    """Save gap-by-method bars and QAOA sampled objective distributions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    pairs = [(r["method"], r["gap_pct"]) for r in rows if r["gap_pct"] is not None]
    ax1.bar([m for m, _ in pairs], [g for _, g in pairs], color="#4477aa")
    ax1.set_ylabel("gap to exact optimum (%)")
    ax1.set_title("Best feasible objective gap by method")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)
    for p, hist in hists.items():
        total = sum(hist.values())
        vals = sorted(hist)
        ax2.plot(
            vals,
            [hist[v] / total for v in vals],
            marker="o",
            ms=3,
            lw=0.8,
            label=f"QAOA p={p}",
        )
    ax2.axvline(optimum, color="k", ls="--", lw=1, label="exact optimum")
    ax2.set_xlabel("decoded objective (feasible samples, lower is better)")
    ax2.set_ylabel("fraction of feasible shots")
    ax2.set_title("Sampled objective distribution")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_demo(quick: bool = False, out_dir: Path = OUT_DIR) -> list[dict[str, Any]]:
    """Run the full demo, print each step, save the plot, return table rows."""
    budget = QUICK_BUDGET if quick else DEMO_BUDGET
    start = time.perf_counter()
    pf = make_portfolio(N_LOANS, SEED)
    show_instance(pf)

    q = build_qubo(pf, LAM)
    n_slack = int(q.slack_coeffs.shape[0])
    print(f"\n[2] QUBO: lam={LAM}, qubits={q.n_vars} ({pf.n} loan + {n_slack} slack)")

    print("\n[3] Classical baselines run (exact, greedy, random sampling).")
    print(
        f"[4] QAOA p=1,2 with a REDUCED budget: restarts={budget['restarts']}, "
        f"maxiter={budget['maxiter']}, shots={budget['shots']} "
        "(full runs: 5 / 200 / 4096; see results/results.md)."
    )
    rows, hists = build_rows(pf, q, budget)

    print("\n[5] Comparison (gap % = best feasible objective vs exact optimum)\n")
    print(format_table(rows))
    print("\n" + honesty_note(rows))

    plot_path = out_dir / "demo_plot.png"
    save_plot(rows, hists, float(rows[0]["objective"]), plot_path)
    print(f"[6] Plot saved to {plot_path}")
    print(f"Total demo time: {time.perf_counter() - start:.1f} s")
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller QAOA budget")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    run_demo(quick=args.quick, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
