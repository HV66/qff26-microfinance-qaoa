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
        "logical depth",
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


def holm_md(results_dir: Path) -> str:
    """Markdown for the Holm-Bonferroni correction over the whole test family."""
    path = results_dir / "holm_correction.json"
    if not path.exists():
        return (
            "No Holm correction file found (run `python -m scripts.holm_correction`)."
        )
    d = json.loads(path.read_text(encoding="utf-8"))
    summary = (
        f"Family of {d['family_size']} one-sided exact binomial tests "
        f"({sum(1 for r in d['rows'] if r['kind'].startswith('simulated'))} simulated "
        f"QAOA runs plus {sum(1 for r in d['rows'] if r['kind'].startswith('real'))} "
        "real-hardware jobs), each against its uniform-random "
        f"rate. {d['n_raw_below_alpha']} have a raw p-value below {d['alpha']}; "
        f"{d['n_survive_holm']} survive Holm-Bonferroni at {d['alpha']}"
        + (": " + "; ".join(d["survive_holm"]) if d["survive_holm"] else "")
        + ". Holm controls the family-wise error rate; the other raw hits are "
        "consistent with chance given the number of tests."
    )
    rows = [
        [
            r["label"],
            f"{r['p_raw']:.2e}" if r["p_raw"] < 1e-3 else f"{r['p_raw']:.4f}",
            f"{r['p_holm']:.2e}" if r["p_holm"] < 1e-3 else f"{r['p_holm']:.4f}",
            "yes" if r["survives"] else "no",
        ]
        for r in sorted(d["rows"], key=lambda r: r["p_raw"])
    ]
    table = md_table(
        ["test", "raw p (above random)", "Holm adjusted p", "survives Holm at 0.05"],
        rows,
    )
    return summary + "\n\n" + table


def impact_md(results_dir: Path) -> str:
    """Markdown for the impact and sector-concentration analysis."""
    path = results_dir / "impact_analysis.json"
    if not path.exists():
        return (
            "No impact analysis file found (run `python -m scripts.impact_analysis`)."
        )
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["rows"]
    intro = (
        d["note"] + " Amount shares are the share of the DEPLOYED amount per sector; "
        "loans per sector are counts of funded loans (sector 0 / 1 / 2). "
        "An unfunded sector had loan requests but received none."
    )
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r["seed"],
                r["lam"],
                r["solver"],
                r["n_funded"],
                f"{r['amount_deployed']} / {r['budget']}",
                r["expected_return"],
                r["portfolio_risk_xT_sigma_x"],
                " / ".join(str(r["loans_by_sector"][str(k)]) for k in range(3)),
                ", ".join(str(s) for s in r["unfunded_sectors"]) or "none",
            ]
        )
    table = md_table(
        [
            "seed",
            "lam",
            "solver",
            "loans funded",
            "amount deployed / budget",
            "expected return",
            "risk x^T Sigma x",
            "funded loans per sector (0 / 1 / 2)",
            "unfunded sector",
        ],
        table_rows,
    )
    exact = [r for r in rows if r["solver"] == "exact"]
    by_lam = "; ".join(
        f"lam {lam}: "
        f"{sum(1 for r in exact if r['lam'] == lam and r['unfunded_sectors'])} of "
        f"{sum(1 for r in exact if r['lam'] == lam)}"
        for lam in sorted({r["lam"] for r in exact})
    )
    summary = (
        f"Exact-optimum portfolios that leave at least one sector with zero funded "
        f"loans: {sum(1 for r in exact if r['unfunded_sectors'])} of {len(exact)} "
        f"({by_lam})."
    )
    by_lam_rows = [
        [
            r["lam"],
            r["mean_loans_funded"],
            r["mean_expected_return"],
            r["mean_portfolio_risk"],
            r["n_with_unfunded_sector"],
            r["n_budget_not_fully_used"],
        ]
        for r in d.get("exact_by_lambda", [])
    ]
    by_lam_table = md_table(
        [
            "lam",
            "mean loans funded",
            "mean expected return",
            "mean risk x^T Sigma x",
            "portfolios with an unfunded sector (of 5)",
            "portfolios not using the full budget (of 5)",
        ],
        by_lam_rows,
    )
    return intro + "\n\n" + summary + "\n\n" + by_lam_table + "\n\n" + table


def scaling_md(results_dir: Path) -> str:
    """Markdown for the scaling study: measured sizes plus a labelled extrapolation."""
    path = results_dir / "scaling_study.json"
    if not path.exists():
        return "No scaling study file found (run `python -m scripts.scaling_study`)."
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        [
            r["n_loans"],
            r["n_qubits"],
            f"{r['n_loans']} + {r['n_slack']}",
            f"{r['logical_depth_p1']} / {r['logical_two_qubit_p1']}",
            f"{r['logical_depth_p2']} / {r['logical_two_qubit_p2']}",
            f"{r['brute_force_runtime_s']:.4f}",
            (
                "skipped (n > 6)"
                if r["fake_fez_depth_p1"] is None
                else f"{r['fake_fez_depth_p1']} / {r['fake_fez_two_qubit_p1']}"
            ),
        ]
        for r in d["rows"]
    ]
    table = md_table(
        [
            "loans",
            "qubits",
            "loans + slack",
            "logical depth / 2q, p=1",
            "logical depth / 2q, p=2",
            "brute-force s",
            "fake_fez transpiled depth / 2q, p=1",
        ],
        rows,
    )
    e = d["extrapolation"]
    extra = (
        f"**{e['LABEL']}** (not run, not measured): at {e['n_loans']} loans with an "
        f"assumed budget of {e['assumed_budget']} ({e['assumed_budget_rule']}) the "
        f"formula gives {e['n_qubits']} qubits ({e['n_loans']} loans + "
        f"{e['n_slack']} slack), p * N * (N - 1) = "
        f"{e['logical_two_qubit_p1_extrapolated']} logical two-qubit gates at p=1 "
        f"and {e['logical_two_qubit_p2_extrapolated']} at p=2, and "
        f"{e['brute_force_states']} brute-force states (2^{e['n_loans']}). The "
        "two-qubit formula reproduces every measured size exactly; transpiled "
        "counts on a coupling map would be larger and are not extrapolated."
    )
    note = (
        d["note"]
        + " The squared budget penalty couples every pair of QUBO variables, so "
        "the logical two-qubit count is exactly p * N * (N - 1) for N qubits."
    )
    return note + "\n\n" + table + "\n\n" + extra


def hard_instance_md(results_dir: Path) -> str:
    """Join greedy gap, QAOA verdict vs random and enrichment, seeds 1-4.

    QAOA and random numbers come from results/qaoa_seed{1..4}_p*.json; the
    greedy gap from results/compare_main.json; Holm survival from
    results/holm_correction.json.
    """
    main_path = results_dir / "compare_main.json"
    holm_path = results_dir / "holm_correction.json"
    if not main_path.exists():
        return "No compare_main.json found."
    main = json.loads(main_path.read_text(encoding="utf-8"))
    greedy = {
        r["seed"]: r["gap_pct"]
        for r in main["seed_summary"]["per_seed"]
        if r["method"] == "greedy"
    }
    verdicts: dict[tuple[int, int], str] = {}
    csv_path = results_dir / "compare_vs_random.csv"
    if csv_path.exists():
        import csv

        with csv_path.open(encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                verdicts[(int(rec["seed"]), int(rec["p"]))] = rec["verdict"]
    holm = {}
    if holm_path.exists():
        for r in json.loads(holm_path.read_text(encoding="utf-8"))["rows"]:
            holm[r["label"]] = r["survives"]
    rows, flagged = [], []
    for seed in (1, 2, 3, 4):
        for path in sorted(results_dir.glob(f"qaoa_seed{seed}_p*.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            cfg, smp, rnd = run["config"], run["sampling"], run["random_null"]
            shots = int(cfg["shots"])
            hits = round(smp["p_optimal"] * shots)
            verdict = verdicts.get((seed, int(cfg["p"])), "not tested")
            enrich = smp["p_optimal"] / rnd["p_optimal"] if rnd["p_optimal"] else None
            survives = holm.get(f"simulated QAOA seed {seed} p={cfg['p']}")
            gap = greedy.get(seed)
            greedy_fails = gap is not None and gap > 0
            if greedy_fails:
                flagged.append((seed, cfg["p"], verdict, survives))
            rows.append(
                [
                    seed,
                    f"{gap:.4f}" if gap is not None else "-",
                    "yes" if greedy_fails else "no",
                    cfg["p"],
                    hits,
                    round(rnd["p_optimal"] * shots),
                    f"{enrich:.2f}" if enrich is not None else "-",
                    verdict,
                    "-" if survives is None else ("yes" if survives else "no"),
                ]
            )
    table = md_table(
        [
            "seed",
            "greedy gap %",
            "greedy fails",
            "p",
            "QAOA hits",
            "random hits",
            "enrichment (QAOA / random P(opt))",
            "verdict vs random (raw)",
            "survives Holm",
        ],
        rows,
    )
    if flagged:
        parts = ", ".join(
            f"seed {sd} p={p}: {v}"
            + (" (survives Holm)" if h else " (does not survive Holm)")
            for sd, p, v, h in flagged
        )
        n_above = sum(1 for _, _, v, _ in flagged if v == "above random")
        n_below = sum(1 for _, _, v, _ in flagged if v == "below random")
        verdict_line = (
            f"Where greedy fails (gap above 0), QAOA is above random in {n_above} of "
            f"{len(flagged)} runs and below random in {n_below}: {parts}. This is a "
            "mixed result, not a consistent advantage; it is one failing seed."
        )
    else:
        verdict_line = "Greedy does not fail on any of seeds 1 to 4."
    other_above = sum(1 for r in rows if r[2] == "no" and r[7] == "above random")
    other_total = sum(1 for r in rows if r[2] == "no")
    other_line = (
        f"Where greedy is exact, QAOA is above random in {other_above} of "
        f"{other_total} runs, so being above random does not track where greedy fails."
    )
    return verdict_line + " " + other_line + "\n\n" + table


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
            f"{d['backend_noise_model']}: transpiled depth {tr['depth']}, "
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


def hardware_md(path: Path) -> str:
    """Markdown for the real-hardware result (one job), taken from its JSON."""
    d = json.loads(path.read_text(encoding="utf-8"))
    inst, tr, m = d["instance"], d["transpile"], d["metrics"]
    shots = sum(d["counts"].values())
    hits = round(m["p_optimal"] * shots)
    rate = 2.0 ** -inst["n_loans"]
    instance_text = (
        f"seed {inst['seed']}, {inst['n_loans']} loans, "
        f"{inst['n_qubits']} qubits, p={inst['p']}"
    )
    rows = [
        ["job id", d["job_id"]],
        ["backend", d["backend"]],
        ["submitted (UTC)", d["submitted_utc"]],
        ["instance", instance_text],
        ["transpiled depth on this backend", tr["depth"]],
        ["two-qubit gates", tr["two_qubit_gates"]],
        ["shots (sum of saved counts)", shots],
        ["optimum hits", hits],
        ["P(optimum)", m["p_optimal"]],
        ["uniform-random expectation", rate],
        ["P(feasible, decoded)", m["p_feasible_decoded"]],
        ["P(raw feasible, zero penalty)", m["p_feasible_raw"]],
        ["best-sample gap %", m["best_feasible_gap_pct"]],
        ["distinct loan vectors sampled", m["distinct_loan_vectors"]],
    ]
    note = (
        "One real-hardware job. The 4096 shots cover every loan vector of this "
        "5-loan instance, so the best-sample gap is uninformative; P(optimum) "
        "against the uniform-random expectation is the metric. Transpiled depth "
        "here is on this backend; the noise-model section lists transpiled depth "
        "on the fake_fez target, and the main table lists logical depth."
    )
    return (
        "## IBM Quantum hardware\n\n"
        + note
        + "\n\n"
        + md_table(["field", "value"], rows)
    )


def hardware_jobs_md(results_dir: Path) -> str:
    """Markdown for ALL real-hardware jobs: one row per job, vs random, with Holm."""
    legacy = results_dir / "hardware_result.json"
    paths = ([legacy] if legacy.exists() else []) + sorted(
        results_dir.glob("hardware_result_seed*_n*_p*.json")
    )
    if not paths:
        return (
            "## IBM Quantum hardware\n\nNo real-hardware result exists: no "
            "`hardware_result*.json` file is present."
        )
    holm: dict[str, dict[str, Any]] = {}
    holm_path = results_dir / "holm_correction.json"
    if holm_path.exists():
        for r in json.loads(holm_path.read_text(encoding="utf-8"))["rows"]:
            holm[r["label"]] = r
    vs_path = results_dir / "hardware_vs_random.json"
    vs_by_job: dict[str, dict[str, Any]] = {}
    if vs_path.exists():
        for job in json.loads(vs_path.read_text(encoding="utf-8")).get("jobs", []):
            vs_by_job[job["job_id"]] = job
    rows = []
    for path in paths:
        d = json.loads(path.read_text(encoding="utf-8"))
        inst, tr, m = d["instance"], d["transpile"], d["metrics"]
        shots = sum(d["counts"].values())
        vs = vs_by_job.get(d["job_id"], {})
        label = (
            f"real {d['backend']} {inst['n_loans']} loans "
            f"seed {inst['seed']} p={inst['p']}"
        )
        h = holm.get(label)
        rows.append(
            [
                d["job_id"],
                f"{inst['seed']} / {inst['n_loans']} / {inst['p']}",
                f"{tr['depth']} / {tr['two_qubit_gates']}",
                shots,
                f"{round(m['p_optimal'] * shots)} (P {m['p_optimal']:.4f})",
                vs.get("random_null", {}).get("optimum_hits", "-"),
                f"{2.0 ** -inst['n_loans']:.4f}",
                m["p_feasible_decoded"],
                vs.get("verdict", "-"),
                "-" if h is None else f"{h['p_holm']:.4f}",
                "-" if h is None else ("yes" if h["survives"] else "no"),
            ]
        )
    table = md_table(
        [
            "job id",
            "seed / loans / p",
            "transpiled depth / 2q on this backend",
            "shots",
            "optimum hits",
            "random-null hits (same shots)",
            "uniform expectation P(opt)",
            "P(feasible, decoded)",
            "verdict vs random (raw)",
            "Holm adjusted p",
            "survives Holm",
        ],
        rows,
    )
    note = (
        f"{len(paths)} real-hardware job(s), each a separate SamplerV2 run with "
        "dynamical decoupling and Pauli twirling. With 4096 shots over a 5-loan "
        "instance every loan vector is sampled, so the best-sample gap is "
        "uninformative; P(optimum) against the uniform-random expectation is the "
        "metric. Depth is transpiled depth on the backend that ran the job."
    )
    return "## IBM Quantum hardware\n\n" + note + "\n\n" + table


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
    sections.append(
        "## Multiple-testing correction (Holm-Bonferroni)\n\n" + holm_md(results_dir)
    )
    sections.append(
        "## Who is funded: impact and sector concentration\n\n" + impact_md(results_dir)
    )
    sections.append("## Scaling and where it breaks\n\n" + scaling_md(results_dir))
    sections.append(
        "## Hard instances: where greedy fails, does QAOA beat random?\n\n"
        + hard_instance_md(results_dir)
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
    sections.append(hardware_jobs_md(results_dir))
    if (results_dir / "hardware_result.json").exists():
        sections.append(
            hardware_md(results_dir / "hardware_result.json").replace(
                "## IBM Quantum hardware", "### First hardware job: field detail", 1
            )
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
