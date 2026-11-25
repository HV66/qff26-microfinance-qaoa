"""Render results/results.md from results/*.json only (no recomputation).

Run:  python scripts/make_results_md.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RESULTS = Path("results")
PLOT_FILES = [
    ("compare_energy_convergence.png", "QAOA energy convergence"),
    ("compare_p_optimum_vs_p.png", "P(optimum) vs depth p"),
    ("compare_gap_by_method.png", "Optimality gap by method"),
]


def fmt(v: Any, nd: int = 4) -> str:
    """Format a value for a table cell ('-' for None)."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Return a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main_table_md(main: dict[str, Any]) -> str:
    """Markdown for the main comparison table."""
    heads = [
        "method",
        "objective (min)",
        "value (max = -obj)",
        "gap %",
        "runtime s",
        "depth",
        "P(optimum)",
        "P(feasible, decoded)",
        "note",
    ]
    keys = [
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
    out = md_table(heads, [[r[k] for k in keys] for r in main["rows"]])
    if main["missing_qaoa_p"]:
        out += f"\n\nQAOA result files missing for p = {main['missing_qaoa_p']}; no row shown."
    return out


def reading_md(main: dict[str, Any]) -> str:
    """Auto-generated plain statement comparing classical methods and QAOA."""
    rows = {r["method"]: r for r in main["rows"]}
    parts = []
    for name, r in rows.items():
        if name.startswith("QAOA"):
            continue
        parts.append(f"{name}: gap {fmt(r['gap_pct'], 2)}%")
    qaoa = [r for r in main["rows"] if r["method"].startswith("QAOA")]
    qaoa_ok = [r for r in qaoa if r["gap_pct"] is not None]
    text = "Seed {seed}, n={n_loans}, lam={lam}. ".format(**main["config"])
    text += "; ".join(parts) + ". "
    if not qaoa_ok:
        return text + "No QAOA result with a feasible sample is available."
    best = min(qaoa_ok, key=lambda r: r["gap_pct"])
    text += f"Best QAOA: gap {fmt(best['gap_pct'], 2)}% ({best['method']}). "
    classical = [
        r
        for r in main["rows"]
        if not r["method"].startswith(("QAOA", "exact")) and r["gap_pct"] is not None
    ]
    for r in classical:
        if r["gap_pct"] < best["gap_pct"]:
            verdict = "beats"
        elif r["gap_pct"] == best["gap_pct"]:
            verdict = "ties"
        else:
            verdict = "is worse than"
        text += f"{r['method']} {verdict} the best QAOA by gap. "
    null_p = next(
        (r["p_optimum"] for r in main["rows"] if r["method"].startswith("random")),
        None,
    )
    with_p = [r for r in qaoa if r["p_optimum"] is not None]
    if null_p is not None and with_p:
        top = max(with_p, key=lambda r: r["p_optimum"])
        n_loan_vectors = 2 ** main["config"]["n_loans"]
        text += (
            f"Highest QAOA P(optimum) is {fmt(top['p_optimum'])} ({top['method']}) "
            f"vs {fmt(null_p)} for uniform random sampling; picking one of the "
            f"{n_loan_vectors} loan vectors uniformly gives {fmt(1 / n_loan_vectors)}. "
            "With 4096 shots these P(optimum) values are only a few dozen hits, so "
            "differences of this size are close to sampling noise. "
        )
    text += (
        "The gap column is best-of-shots over a small search space, so every "
        "method that samples enough reaches 0%; P(optimum) is the more "
        "informative column. "
    )
    text += (
        "Classical solvers at this size are exact or near-exact in well under a "
        "second; no quantum advantage is claimed."
    )
    return text


def lambda_md(lam: dict[str, Any]) -> str:
    """Markdown for the lambda sensitivity table."""
    note = (
        f"QAOA at a single depth p={lam['qaoa_p']}."
        if lam["qaoa_p"] is not None
        else "No QAOA rows (no QAOA result files)."
    )
    rows = [
        [r["lam"], r["method"], r["objective"], r["gap_pct"], r["p_optimum"], r["note"]]
        for r in lam["rows"]
    ]
    heads = ["lam", "method", "objective (min)", "gap %", "P(optimum)", "note"]
    return note + "\n\n" + md_table(heads, rows)


def seeds_md(seeds: dict[str, Any]) -> str:
    """Markdown for the 5-seed summary."""
    rows = [
        [
            r["method"],
            r["n_seeds"],
            r["n_missing"],
            r["mean_gap_pct"],
            r["worst_gap_pct"],
        ]
        for r in seeds["summary"]
    ]
    heads = ["method", "seeds with gap", "seeds missing", "mean gap %", "worst gap %"]
    return md_table(heads, rows)


def vs_random_md(block: dict[str, Any]) -> str:
    """Markdown for the per-run QAOA vs random-sampling table with verdicts."""
    rows = block["rows"]
    if not rows:
        return "No QAOA result files found."
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    tally = ", ".join(f"{n} of {len(rows)} runs {v}" for v, n in sorted(counts.items()))
    shots = rows[0]["shots"]
    intro = (
        f"Each row compares one QAOA run with uniform random sampling at the same "
        f"{shots} shots. Random sampling draws every QUBO bit uniformly, so its "
        "P(optimum) is 1/256 in expectation for 8 loans. The verdict is an exact "
        f"one-sided binomial test of QAOA's optimum hits against that rate at "
        f"alpha={block['alpha']}; 'p-value' is the chance of at least that many "
        "hits under random sampling. With "
        f"{len(rows)} rows, about {len(rows) * block['alpha']:.1f} false 'above "
        "random' verdicts are expected by chance alone.\n\n"
        f"Tally: {tally}."
    )
    heads = [
        "seed",
        "p",
        "QAOA hits",
        "random hits",
        "QAOA P(opt)",
        "random P(opt)",
        "p-value",
        "QAOA P(feas)",
        "random P(feas)",
        "QAOA gap %",
        "random gap %",
        "verdict",
    ]
    keys = [
        "seed",
        "p",
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
    return intro + "\n\n" + md_table(heads, [[r[k] for k in keys] for r in rows])


def noise_md(paths: list[Path]) -> str:
    """Markdown table of noise-model runs (one block per saved file)."""
    blocks = ["## Noise-model simulation (fake_fez Aer noise model)"]
    for path in paths:
        d = json.loads(path.read_text(encoding="utf-8"))
        inst, tr = d["instance"], d["transpiled"]
        rows = [
            [
                name,
                d["mitigation_applied"][name],
                v["p_optimal"],
                v["p_feasible_decoded"],
                v["best_feasible_gap_pct"],
            ]
            for name, v in d["variants"].items()
        ]
        head = (
            f"Source: `{path.as_posix()}`. {inst['n_loans']} loans, "
            f"{inst['n_qubits']} qubits, p={inst['p']}, {d['shots']} shots, "
            f"parameters: {inst['params_source']}. Transpiled for "
            f"{d['backend_noise_model']}: depth {tr['depth']}, "
            f"{tr['two_qubit_gates']} two-qubit gates. Random sampling for "
            f"reference: P(optimum) = 1/{2 ** inst['n_loans']} in expectation."
        )
        table = md_table(
            ["variant", "mitigation", "P(optimum)", "P(feasible, decoded)", "gap %"],
            rows,
        )
        blocks.append(head + "\n\n" + table)
    return "\n\n".join(blocks)


def score_md(results_dir: Path) -> str:
    """Markdown for the optimiser's normalised score, seed 0, with its definition."""
    rows = []
    for path in sorted(results_dir.glob("qaoa_seed0_p*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        opt = run["optimiser"]
        rows.append(
            [
                run["config"]["p"],
                opt["best_value"],
                opt["normalised_score"],
                opt["normalised_score_uniform_state"],
            ]
        )
    if not rows:
        return "No seed-0 QAOA result files found."
    rows.sort(key=lambda r: r[0])
    intro = (
        "Normalised score = (Emax - E) / (Emax - Emin), where E is the optimiser's "
        "final exact <H> and Emin, Emax are the smallest and largest QUBO energies "
        "over all bitstrings. It is 1 at the QUBO minimum and 0 at the maximum. It "
        "is NOT a multiplicative approximation ratio. It is a measure of the "
        "optimiser's expectation value, not of sampled solution quality; the "
        "sampled metrics are P(optimum) and the best-feasible gap above."
    )
    table = md_table(
        ["p", "final <H>", "normalised score", "score of the uniform state"], rows
    )
    return intro + "\n\n" + table


def extra_md(title: str, path: Path) -> str:
    """Render top-level scalar fields of an optional result file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [[k, v] for k, v in data.items() if isinstance(v, (int, float, str, bool))]
    body = md_table(["field", "value"], rows) if rows else "(no scalar fields)"
    return f"## {title}\n\nSource: `{path.as_posix()}`\n\n{body}\n"


def build(results_dir: Path = RESULTS) -> str:
    """Assemble the full markdown document from results/compare_main.json."""
    src = results_dir / "compare_main.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    main = payload["main"]
    cfg = main["config"]
    sections = [
        (
            "# Results\n\nGenerated by `scripts/make_results_md.py` from "
            "`results/compare_main.json` and related result files. "
            "Objective is minimised (lower is better); "
            "gap % = 100*(obj-exact)/|exact|."
        ),
        f"## Main comparison (seed {cfg['seed']}, n={cfg['n_loans']}, "
        f"lam={cfg['lam']}, {cfg['shots']} shots)\n\n" + main_table_md(main),
        "## Reading the results\n\n" + reading_md(main),
        "## Lambda sensitivity (seed 0)\n\n" + lambda_md(payload["lambda_sensitivity"]),
        "## Five-seed summary (lam=0.3)\n\n" + seeds_md(payload["seed_summary"]),
    ]
    sections.append(
        "## Optimiser normalised score (seed 0)\n\n" + score_md(results_dir)
    )
    if "vs_random" in payload:
        sections.append(
            "## QAOA vs random sampling, every seed and depth\n\n"
            + vs_random_md(payload["vs_random"])
        )
    plots = [
        f"![{alt}](plots/{name})"
        for name, alt in PLOT_FILES
        if (results_dir / "plots" / name).exists()
    ]
    if plots:
        sections.append("## Plots\n\n" + "\n\n".join(plots))
    noise_files = sorted(results_dir.glob("noise_run*.json"))
    if noise_files:
        sections.append(noise_md(noise_files))
    if (results_dir / "hardware_result.json").exists():
        sections.append(
            extra_md("IBM Quantum hardware", results_dir / "hardware_result.json")
        )
    else:
        sections.append(
            "## IBM Quantum hardware\n\nNo real-hardware result exists: "
            "`results/hardware_result.json` is absent."
        )
    return "\n\n".join(sections) + "\n"


def main() -> None:
    """Write results/results.md."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = RESULTS / "results.md"
    out.write_text(build(), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
