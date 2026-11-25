"""Head-to-head comparison: exact vs greedy vs SA vs random vs QAOA.

Reads QAOA result files (results/qaoa_seed{S}_p{P}.json written by qaoa_run.py),
runs the classical methods, and writes machine-readable tables
(results/compare_*.json / .csv) plus three plots (results/plots/compare_*.png).
Nothing is fabricated: a missing QAOA file produces no row.

Conventions: objective is MINIMISED (lower is better); the maximisation value
is its negative. Gap % = 100 * (objective - exact) / |exact|.

Run:  python compare.py            (uses existing QAOA files)
      python compare.py --run-qaoa-lam   (also runs QAOA at lam 0.1 and 1.0)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from baseline import brute_force, greedy, random_sampling
from data import make_portfolio
from qubo import build_qubo
from sa_baseline import simulated_annealing

log = logging.getLogger(__name__)

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
SHOTS = 4096
N_LOANS = 8
MAIN_LAM = 0.3
LAMBDAS = (0.1, 0.3, 1.0)
SEEDS = (0, 1, 2, 3, 4)
MAIN_PS = (1, 2, 3, 4)
SEED_SUMMARY_PS = (1, 2, 3)
# Okabe-Ito colourblind-safe palette.
COLOURS = {
    "exact": "#000000",
    "greedy": "#E69F00",
    "simulated annealing": "#56B4E9",
    "random sampling": "#999999",
    "qaoa": "#D55E00",
    "blue": "#0072B2",
    "green": "#009E73",
}
TABLE_FIELDS = [
    "method",
    "objective",
    "max_value",
    "gap_pct",
    "runtime_s",
    "circuit_depth",
    "p_optimum",
    "p_feasible_decoded",
    "note",
]


def gap_pct(value: float | None, exact: float) -> float | None:
    """Return the optimality gap in percent, or None if undefined.

    Args:
        value: Method objective (minimised) or None if no feasible answer.
        exact: Exact optimal objective.

    Returns:
        100 * (value - exact) / |exact|. When exact == 0 the relative gap is
        undefined, so return 0.0 for an exact match and None otherwise.
    """
    if value is None:
        return None
    if exact == 0.0:
        return 0.0 if value == 0.0 else None
    return 100.0 * (value - exact) / abs(exact)


def make_row(
    method: str,
    objective: float | None,
    exact: float,
    runtime_s: float | None = None,
    circuit_depth: int | None = None,
    p_optimum: float | None = None,
    p_feasible_decoded: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Build one comparison-table row (objective is minimised)."""
    return {
        "method": method,
        "objective": objective,
        "max_value": None if objective is None else -objective,
        "gap_pct": gap_pct(objective, exact),
        "runtime_s": runtime_s,
        "circuit_depth": circuit_depth,
        "p_optimum": p_optimum,
        "p_feasible_decoded": p_feasible_decoded,
        "note": note,
    }


def load_qaoa(seed: int, p: int, results_dir: Path = RESULTS) -> dict[str, Any] | None:
    """Load results/qaoa_seed{seed}_p{p}.json, or None if absent/unreadable."""
    path = results_dir / f"qaoa_seed{seed}_p{p}.json"
    if not path.exists():
        log.warning("missing QAOA result file: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("unreadable QAOA result file %s: %s", path, exc)
        return None


def qaoa_row(res: dict[str, Any], exact: float) -> dict[str, Any]:
    """Convert a qaoa_run result dict to a table row.

    The objective is the best decoded-feasible sample out of the shot budget.
    Runtime is optimisation + sampling on the noiseless simulator.
    """
    s = res["sampling"]
    note = ""
    if abs(res["exact"]["objective"] - exact) > 1e-9:
        note = "exact objective differs from this table's exact"
    return make_row(
        f"QAOA p={res['config']['p']}",
        s["best_feasible_objective"],
        exact,
        runtime_s=res["runtime_s"]["total"],
        circuit_depth=res["circuit"]["depth"],
        p_optimum=s["p_optimal"],
        p_feasible_decoded=s["p_feasible_decoded"],
        note=note,
    )


def classical_rows(
    seed: int, lam: float, n_loans: int = N_LOANS, shots: int = SHOTS
) -> tuple[float, list[dict[str, Any]]]:
    """Run exact, greedy, SA and random sampling on one instance.

    Returns:
        (exact objective, rows for exact/greedy/SA/random).
    """
    pf = make_portfolio(n_loans, seed)
    q = build_qubo(pf, lam)
    bf = brute_force(pf, lam)
    gr = greedy(pf, lam)
    sa = simulated_annealing(pf, q, seed)
    rs = random_sampling(pf, q, shots, seed)
    ex = bf.objective
    sa_note = "" if sa.feasible else "best QUBO state decodes infeasible"
    rows = [
        make_row("exact (brute force)", ex, ex, bf.runtime_s),
        make_row("greedy", gr.objective, ex, gr.runtime_s),
        make_row(
            "simulated annealing",
            sa.objective if sa.feasible else None,
            ex,
            sa.runtime_s,
            note=sa_note,
        ),
        make_row(
            f"random sampling ({shots} shots)",
            rs.best_objective,
            ex,
            None,
            p_optimum=rs.p_optimal,
            p_feasible_decoded=rs.p_feasible_decoded,
            note="" if rs.best_objective is not None else "no feasible sample",
        ),
    ]
    return ex, rows


def build_main_table(
    seed: int = 0,
    lam: float = MAIN_LAM,
    ps: tuple[int, ...] = MAIN_PS,
    results_dir: Path = RESULTS,
) -> dict[str, Any]:
    """Build the main table; QAOA rows appear only for files that exist."""
    ex, rows = classical_rows(seed, lam)
    missing = []
    for p in ps:
        res = load_qaoa(seed, p, results_dir)
        if res is None:
            missing.append(p)
        else:
            rows.append(qaoa_row(res, ex))
    return {
        "config": {"seed": seed, "lam": lam, "n_loans": N_LOANS, "shots": SHOTS},
        "exact_objective": ex,
        "rows": rows,
        "missing_qaoa_p": missing,
    }


def best_qaoa_p(
    seed: int, ps: tuple[int, ...], results_dir: Path = RESULTS
) -> int | None:
    """Pick the p with the smallest gap (ties: highest P(optimum)), or None.

    Selection uses only seed-0 lam=0.3 results already produced; it is
    documented, not tuned for lambda.
    """
    best: tuple[float, float, int] | None = None
    for p in ps:
        res = load_qaoa(seed, p, results_dir)
        if res is None:
            continue
        s = res["sampling"]
        gap = s["best_feasible_gap_pct"]
        key = (float("inf") if gap is None else gap, -s["p_optimal"], p)
        if best is None or key < best:
            best = key
    return None if best is None else best[2]


def lambda_table(
    lams: tuple[float, ...] = LAMBDAS,
    qaoa_p: int | None = None,
    run_qaoa: bool = False,
    results_dir: Path = RESULTS,
) -> list[dict[str, Any]]:
    """Lambda-sensitivity table (seed 0, n=8).

    QAOA is included at a single depth ``qaoa_p``. For lam=0.3 the main-table
    file is reused; other lambdas are run only if ``run_qaoa`` and cached in
    results/compare_qaoa_lam{lam}_p{p}.json.
    """
    out = []
    for lam in lams:
        ex, rows = classical_rows(0, lam)
        if qaoa_p is not None:
            res = None
            if lam == MAIN_LAM:
                res = load_qaoa(0, qaoa_p, results_dir)
            else:
                cache = results_dir / f"compare_qaoa_lam{lam}_p{qaoa_p}.json"
                if cache.exists():
                    res = json.loads(cache.read_text(encoding="utf-8"))
                elif run_qaoa:
                    import qaoa_run

                    res = qaoa_run.run_one(seed=0, p=qaoa_p, lam=lam)
                    cache.write_text(json.dumps(res, indent=1), encoding="utf-8")
            if res is not None:
                rows.append(qaoa_row(res, ex))
        for r in rows:
            out.append({"lam": lam, "exact_objective": ex, **r})
    return out


def seed_summary(
    seeds: tuple[int, ...] = SEEDS,
    ps: tuple[int, ...] = SEED_SUMMARY_PS,
    results_dir: Path = RESULTS,
) -> dict[str, Any]:
    """Per-seed gaps and mean/worst gap per method over seeds (lam=0.3)."""
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        ex, rows = classical_rows(seed, MAIN_LAM)
        for p in ps:
            res = load_qaoa(seed, p, results_dir)
            if res is not None:
                rows.append(qaoa_row(res, ex))
        for r in rows:
            per_seed.append(
                {"seed": seed, "method": r["method"], "gap_pct": r["gap_pct"]}
            )
    return {"per_seed": per_seed, "summary": summarise_gaps(per_seed)}


ALPHA = 0.05
VS_RANDOM_FIELDS = [
    "seed",
    "p",
    "shots",
    "qaoa_hits",
    "random_hits",
    "qaoa_p_optimum",
    "random_p_optimum",
    "binom_p_value",
    "qaoa_p_feasible",
    "random_p_feasible",
    "qaoa_gap_pct",
    "random_gap_pct",
    "verdict",
]


def binomial_verdict(hits: int, shots: int, n_loans: int) -> tuple[float, str]:
    """Test QAOA's P(optimum) against the uniform-random rate 2^-n_loans.

    Uniform random sampling draws every QUBO bit (loans and slack) uniformly,
    so the decoded loan vector equals the optimum with probability exactly
    2^-n_loans. The test is an exact binomial test of ``hits`` out of
    ``shots`` against that rate, at significance ALPHA.

    Returns:
        (one-sided p-value for "above random", verdict). Verdict is
        "above random" if hits are significantly more frequent than random,
        "below random" if significantly less frequent, otherwise
        "indistinguishable from random".
    """
    from scipy.stats import binomtest

    rate = 2.0**-n_loans
    p_above = float(binomtest(hits, shots, rate, alternative="greater").pvalue)
    p_below = float(binomtest(hits, shots, rate, alternative="less").pvalue)
    if p_above < ALPHA:
        return p_above, "above random"
    if p_below < ALPHA:
        return p_above, "below random"
    return p_above, "indistinguishable from random"


def vs_random_rows(
    seeds: tuple[int, ...] = SEEDS,
    ps: tuple[int, ...] = MAIN_PS,
    results_dir: Path = RESULTS,
) -> list[dict[str, Any]]:
    """One row per (seed, p): QAOA vs random sampling at the same shot budget.

    Every number comes from the saved qaoa_seed*_p*.json (the random null is
    stored in the same file, drawn with the same shots and seed). Missing
    files are skipped.
    """
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for p in ps:
            res = load_qaoa(seed, p, results_dir)
            if res is None:
                continue
            s, rn, cfg = res["sampling"], res["random_null"], res["config"]
            shots = int(cfg["shots"])
            hits = round(s["p_optimal"] * shots)
            p_value, verdict = binomial_verdict(hits, shots, int(cfg["n_loans"]))
            rows.append(
                {
                    "seed": seed,
                    "p": p,
                    "shots": shots,
                    "qaoa_hits": hits,
                    "random_hits": round(rn["p_optimal"] * shots),
                    "qaoa_p_optimum": s["p_optimal"],
                    "random_p_optimum": rn["p_optimal"],
                    "binom_p_value": p_value,
                    "qaoa_p_feasible": s["p_feasible_decoded"],
                    "random_p_feasible": rn["p_feasible_decoded"],
                    "qaoa_gap_pct": s["best_feasible_gap_pct"],
                    "random_gap_pct": rn["best_gap_pct"],
                    "verdict": verdict,
                }
            )
    return rows


def summarise_gaps(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed gaps into mean and worst (max) gap per method.

    Seeds where a method has no gap (None) are counted in ``n_missing`` and
    excluded from mean/worst; they are never silently treated as zero.
    """
    methods: dict[str, list[float | None]] = {}
    for r in per_seed:
        methods.setdefault(r["method"], []).append(r["gap_pct"])
    out = []
    for method, gaps in methods.items():
        vals = [g for g in gaps if g is not None]
        out.append(
            {
                "method": method,
                "n_seeds": len(vals),
                "n_missing": len(gaps) - len(vals),
                "mean_gap_pct": sum(vals) / len(vals) if vals else None,
                "worst_gap_pct": max(vals) if vals else None,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write rows to CSV with the given fields (None becomes empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fields})


def _plt():
    """Import matplotlib with a headless backend and readable defaults."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def plot_convergence(
    ps: tuple[int, ...], path: Path, results_dir: Path = RESULTS
) -> bool:
    """Plot best restart's best-so-far <H> per p vs evaluation (seed 0)."""
    plt = _plt()
    palette = [COLOURS["blue"], COLOURS["qaoa"], COLOURS["green"], COLOURS["greedy"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    exact = None
    drawn = False
    for i, p in enumerate(ps):
        res = load_qaoa(0, p, results_dir)
        if res is None:
            continue
        hist = min(res["optimiser"]["histories"], key=lambda h: h[-1])
        ax.plot(hist, color=palette[i % len(palette)], lw=1.5, label=f"p={p}")
        exact = res["exact"]["qubo_min_energy"]
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.axhline(exact, color="k", ls="--", lw=1.2, label="exact QUBO minimum")
    ax.set_yscale("symlog")
    ax.set_xlabel("objective evaluation")
    ax.set_ylabel("best-so-far <H> (symlog)")
    ax.set_title("QAOA energy convergence (best restart per p, seed 0)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_p_optimum(
    ps: tuple[int, ...], path: Path, results_dir: Path = RESULTS
) -> bool:
    """Plot P(optimum) vs p with the random-null level as a horizontal line."""
    plt = _plt()
    xs, ys, null = [], [], None
    for p in ps:
        res = load_qaoa(0, p, results_dir)
        if res is None:
            continue
        xs.append(p)
        ys.append(res["sampling"]["p_optimal"])
        null = res["random_null"]["p_optimal"]
    if not xs:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(xs, ys, "o-", color=COLOURS["qaoa"], lw=2, label="QAOA (sampled)")
    ax.axhline(
        null, color=COLOURS["random sampling"], ls="--", lw=1.8, label="random null"
    )
    ax.set_xticks(xs)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("P(optimum) over 4096 shots")
    ax.set_title("Probability of sampling the optimal portfolio (seed 0)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_gap_by_method(rows: list[dict[str, Any]], path: Path) -> bool:
    """Bar chart of optimality gap per method (rows with a gap only)."""
    plt = _plt()
    data = [(r["method"], r["gap_pct"]) for r in rows if r["gap_pct"] is not None]
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colours = []
    for name, _ in data:
        key = "qaoa" if name.startswith("QAOA") else name.split(" (")[0]
        colours.append(COLOURS.get(key, COLOURS["blue"]))
    ax.bar(range(len(data)), [g for _, g in data], color=colours)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([n for n, _ in data], rotation=30, ha="right")
    ax.set_ylabel("optimality gap vs exact (%)")
    ax.set_title("Optimality gap by method (seed 0, n=8, lam=0.3; 0 = optimal)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def run_all(run_qaoa_lam: bool = False, results_dir: Path = RESULTS) -> dict[str, Any]:
    """Build every table, write JSON/CSV/plots, and return the payload."""
    main = build_main_table(results_dir=results_dir)
    qp = best_qaoa_p(0, MAIN_PS, results_dir)
    lam_rows = lambda_table(qaoa_p=qp, run_qaoa=run_qaoa_lam, results_dir=results_dir)
    seeds = seed_summary(results_dir=results_dir)
    vs_random = vs_random_rows(results_dir=results_dir)
    payload = {
        "main": main,
        "lambda_sensitivity": {"qaoa_p": qp, "rows": lam_rows},
        "seed_summary": seeds,
        "vs_random": {"alpha": ALPHA, "rows": vs_random},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "compare_main.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
    write_csv(results_dir / "compare_main.csv", main["rows"], TABLE_FIELDS)
    write_csv(
        results_dir / "compare_lambda.csv",
        lam_rows,
        ["lam", "exact_objective", *TABLE_FIELDS],
    )
    write_csv(
        results_dir / "compare_seeds.csv",
        seeds["summary"],
        ["method", "n_seeds", "n_missing", "mean_gap_pct", "worst_gap_pct"],
    )
    write_csv(results_dir / "compare_vs_random.csv", vs_random, VS_RANDOM_FIELDS)
    plots = results_dir / "plots"
    plot_convergence(MAIN_PS, plots / "compare_energy_convergence.png", results_dir)
    plot_p_optimum(MAIN_PS, plots / "compare_p_optimum_vs_p.png", results_dir)
    plot_gap_by_method(main["rows"], plots / "compare_gap_by_method.png")
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-qaoa-lam", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = run_all(args.run_qaoa_lam)
    for r in payload["main"]["rows"]:
        log.info("%-28s obj=%s gap=%s", r["method"], r["objective"], r["gap_pct"])
    if payload["main"]["missing_qaoa_p"]:
        log.warning("missing QAOA p values: %s", payload["main"]["missing_qaoa_p"])


if __name__ == "__main__":
    main()
