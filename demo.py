"""Live end-to-end demo: microfinance loan selection, QAOA vs classical.

Run:  python demo.py            (default: reproduces the saved headline result)
      python demo.py --quick    (REDUCED BUDGET: re-optimises, results differ)

Default mode loads the optimised parameters saved by qaoa_run.py in
results/qaoa_seed0_p*.json, picks the depth p with the lowest stored
optimiser objective, and only samples that saved circuit with 4096 shots
through qaoa_run.sample_counts (the same primitives-V2 sampler path). With the
same seed it reproduces the stored sampling metrics exactly.

--quick re-runs the COBYLA optimisation with FEWER restarts / iterations /
shots than the full runs (5 / 200 / 4096), so its numbers differ from
results/results.md. This demo makes no quantum-advantage claim: it prints a
computed note comparing QAOA with the classical baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from baseline import brute_force, greedy, random_sampling
from data import LoanPortfolio, make_portfolio
from qaoa_run import circuit_cost, optimise, sample_counts, sample_metrics
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
FULL_SHOTS = 4096
QUICK_BUDGET = {"restarts": 1, "maxiter": 40, "shots": 1024}
EPS = 1e-9
OUT_DIR = Path("results")
RESULTS_DIR = Path("results")
DEPTH_FOOTNOTE = (
    "logical depth = generic gates {cx, rz, sx, x}, all-to-all connectivity, no routing. "
    "Transpiled depth on fake_fez (coupling map) is larger: see results/noise_run_p*.json."
)
BANNER = "*** REDUCED BUDGET, results will differ from results/results.md ***"


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


def load_saved_best(results_dir: Path) -> dict[str, Any] | None:
    """Load the saved seed-0 QAOA run with the lowest stored optimiser objective.

    Reads results/qaoa_seed{SEED}_p*.json written by qaoa_run.py and keeps
    files matching this demo's instance (seed, lam, n_loans).

    Returns:
        The parsed JSON of the best-p run, or None if no matching file exists.
    """
    best: dict[str, Any] | None = None
    for path in sorted(results_dir.glob(f"qaoa_seed{SEED}_p*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        cfg = run["config"]
        if (cfg["seed"], cfg["lam"], cfg["n_loans"]) != (SEED, LAM, N_LOANS):
            continue
        if (
            best is None
            or run["optimiser"]["best_value"] < best["optimiser"]["best_value"]
        ):
            best = run
    return best


def run_qaoa_row(
    pf: LoanPortfolio,
    q: Qubo,
    p: int,
    opt_x: np.ndarray,
    opt_obj: float,
    shots: int,
    params: np.ndarray,
) -> tuple[dict[str, Any], Counter[float]]:
    """Sample the circuit with fixed params; return (table row, histogram).

    Sampling goes through qaoa_run.sample_counts, the same primitives-V2 path
    as the full runs. The row runtime covers sampling only.
    """
    t0 = time.perf_counter()
    counts = sample_counts(q, p, params, shots, SEED)
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
    pf: LoanPortfolio,
    q: Qubo,
    shots: int,
    qaoa_runs: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[int, Counter[float]]]:
    """Run every method and return (table rows, QAOA objective histograms).

    Args:
        pf: Portfolio.
        q: QUBO built from pf.
        shots: Shot budget for QAOA sampling and for the random null.
        qaoa_runs: Mapping depth p to its optimised parameters.
    """
    bf = brute_force(pf, q.lam)
    gr = greedy(pf, q.lam)
    t0 = time.perf_counter()
    rs = random_sampling(pf, q, shots, SEED)
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
            "method": f"Random ({shots} shots)",
            "objective": rs.best_objective,
            "gap_pct": gap_pct(rs.best_objective, bf.objective),
            "p_optimum": rs.p_optimal,
            "depth": None,
            "runtime_s": rs_time,
        },
    ]
    hists: dict[int, Counter[float]] = {}
    for p, params in qaoa_runs.items():
        row, hists[p] = run_qaoa_row(pf, q, p, bf.x, bf.objective, shots, params)
        rows.append(row)
    return rows, hists


def select_qaoa_runs(
    q: Qubo, quick: bool, results_dir: Path
) -> tuple[dict[int, np.ndarray], int, str]:
    """Choose the QAOA circuits to sample and say where their parameters came from.

    Default mode loads the saved best-p parameters. Quick mode (and the fallback
    when no saved file matches) re-optimises p=1,2 with QUICK_BUDGET.

    Returns:
        (mapping p -> parameters, shot budget, human-readable source line).
    """
    saved = None if quick else load_saved_best(results_dir)
    if saved is not None:
        p = int(saved["config"]["p"])
        params = np.array(saved["optimiser"]["best_params"])
        src = (
            f"loaded saved parameters for p={p} (lowest stored objective "
            f"{saved['optimiser']['best_value']:.4f}) from "
            f"{results_dir / f'qaoa_seed{SEED}_p{p}.json'}"
        )
        return {p: params}, int(saved["config"]["shots"]), src
    if not quick:
        print("No matching saved QAOA result found; falling back to the reduced run.")
    print(BANNER)
    runs: dict[int, np.ndarray] = {}
    for p in (1, 2):
        opt = optimise(
            q,
            p,
            SEED,
            restarts=QUICK_BUDGET["restarts"],
            maxiter=QUICK_BUDGET["maxiter"],
        )
        runs[p] = np.array(opt["best_params"])
    src = (
        f"re-optimised p=1,2 with restarts={QUICK_BUDGET['restarts']}, "
        f"maxiter={QUICK_BUDGET['maxiter']} (full runs: 5 / 200)"
    )
    return runs, QUICK_BUDGET["shots"], src


def _fmt(value: float | None, spec: str) -> str:
    """Format an optional number; None prints as a dash."""
    return "-" if value is None else format(value, spec)


def format_table(rows: list[dict[str, Any]]) -> str:
    """Return the comparison table as text."""
    head = (
        f"{'method':<24}{'objective':>11}{'gap %':>9}{'P(optimum)':>12}"
        f"{'logical depth':>14}{'runtime s':>11}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['method']:<24}{_fmt(r['objective'], '.4f'):>11}"
            f"{_fmt(r['gap_pct'], '.2f'):>9}{_fmt(r['p_optimum'], '.4f'):>12}"
            f"{_fmt(r['depth'], 'd'):>14}{r['runtime_s']:>11.3f}"
        )
    lines.append(DEPTH_FOOTNOTE)
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
    n_loans: int = N_LOANS,
    shots: int = FULL_SHOTS,
) -> None:
    """Save P(optimum) per method and QAOA sampled objective distributions.

    Left panel: P(optimum) per sampling method (random, QAOA) with a dashed
    line at the uniform-random expectation 1/2^n_loans. The gap is not plotted
    because every method reaches a 0% best-sample gap, which is uninformative.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    pairs = [(r["method"], r["p_optimum"]) for r in rows if r["p_optimum"] is not None]
    ax1.bar([m for m, _ in pairs], [v for _, v in pairs], color="#4477aa", zorder=3)
    rate = 1 / 2**n_loans
    sd = (rate * (1 - rate) / shots) ** 0.5
    ax1.axhspan(
        max(rate - 2 * sd, 0.0),
        rate + 2 * sd,
        color="0.88",
        zorder=1,
        label=f"+/- 2 sd of a random count ({shots} shots)",
    )
    ax1.axhline(
        rate,
        color="k",
        ls="--",
        lw=1.2,
        zorder=4,
        label=f"uniform-random expectation 1/2^{n_loans}",
    )
    ax1.set_ylabel("P(optimum) per shot")
    ax1.set_title("P(optimum) by method vs uniform-random expectation")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)
    ax1.legend(fontsize=8)
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


def run_demo(
    quick: bool = False,
    out_dir: Path = OUT_DIR,
    results_dir: Path = RESULTS_DIR,
) -> list[dict[str, Any]]:
    """Run the full demo, print each step, save the plot, return table rows."""
    start = time.perf_counter()
    pf = make_portfolio(N_LOANS, SEED)
    show_instance(pf)

    q = build_qubo(pf, LAM)
    n_slack = int(q.slack_coeffs.shape[0])
    print(f"\n[2] QUBO: lam={LAM}, qubits={q.n_vars} ({pf.n} loan + {n_slack} slack)")

    print("\n[3] Classical baselines run (exact, greedy, random sampling).")
    runs, shots, source = select_qaoa_runs(q, quick, results_dir)
    print(f"[4] QAOA: {source}; sampling {shots} shots via qaoa_run.sample_counts.")
    rows, hists = build_rows(pf, q, shots, runs)

    print("\n[5] Comparison (gap % = best feasible objective vs exact optimum)\n")
    print(format_table(rows))
    print("\n" + honesty_note(rows))

    plot_path = out_dir / "demo_plot.png"
    save_plot(rows, hists, float(rows[0]["objective"]), plot_path, pf.n, shots)
    print(f"[6] Plot saved to {plot_path}")
    if quick:
        print(BANNER)
    print(f"Total demo time: {time.perf_counter() - start:.1f} s")
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="REDUCED BUDGET: re-optimise with a small budget, results differ",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    run_demo(quick=args.quick, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
