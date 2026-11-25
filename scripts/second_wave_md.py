"""Markdown builders for the second-wave experiments (docs/preregistration.md).

Every number is read from a results/*.json file; nothing is typed by hand. Used by
scripts/make_results_md.py. Each builder returns a section body and degrades to a
one-line "not found" note if its input file is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(results_dir: Path, name: str) -> dict[str, Any] | None:
    path = results_dir / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _sci(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def fairness_md(results_dir: Path) -> str:
    """Price of fairness: sector-coverage constraint, classical, seeds 0-4."""
    d = _load(results_dir, "price_of_fairness.json")
    if d is None:
        return "No `price_of_fairness.json` found."
    rows = [
        [
            r["seed"],
            r["lam"],
            _cell(r["unconstrained_objective"]),
            (
                _cell(r.get("constrained_objective"))
                if r["feasible"]
                else "no feasible portfolio"
            ),
            _cell(r.get("loss_abs")),
            _cell(r.get("loss_pct"), 2),
            "yes" if r["unconstrained_covers_all_sectors"] else "no",
            (
                sum(r["constrained"]["x"])
                if r["feasible"] and "x" in r["constrained"]
                else "-"
            ),
            (
                f"{r['constrained'].get('amount_deployed', '-')} / {r['constrained'].get('budget', '-')}"
                if r["feasible"]
                else "-"
            ),
            f"{r['extra_slack_qubits']} ({r['qubo_qubits_existing']} -> {r['qubo_qubits_with_constraint']})",
        ]
        for r in d["rows"]
    ]
    summary_rows = [
        [
            lam,
            s["n_cases"],
            s["n_loss_positive"],
            s["mean_loss_pct"],
            s["max_loss_pct"],
            s["n_infeasible"],
        ]
        for lam, s in d["by_lambda"].items()
    ]
    intro = (
        f"Constraint: {d['constraint']}. Classical exhaustive search over all loan "
        "vectors, so no QAOA result is involved. Loss is the increase in the "
        "minimised objective against the unconstrained optimum; percent is relative "
        "to the absolute unconstrained objective, which is close to zero at lambda "
        "1.0, so percentages there are large and the absolute loss is the safer "
        "number. The extra qubits are arithmetic only (nothing was built): "
        "sum over sectors of bit_length(n_s - 1) slack bits to turn "
        "`sum_{i in s} x_i >= 1` into an equality."
    )
    t1 = _table(
        [
            "lam",
            "cases",
            "cases with loss > 0",
            "mean loss %",
            "max loss %",
            "infeasible",
        ],
        summary_rows,
    )
    t2 = _table(
        [
            "seed",
            "lam",
            "unconstrained objective",
            "constrained objective",
            "loss (abs)",
            "loss %",
            "unconstrained already covers all sectors",
            "loans funded (constrained)",
            "amount / budget (constrained)",
            "extra slack qubits (qubits before -> after)",
        ],
        rows,
    )
    return intro + "\n\n" + t1 + "\n\n" + t2


def unbalanced_md(results_dir: Path) -> str:
    """Slack-free (unbalanced) penalty versus the slack version."""
    vs = _load(results_dir, "unbalanced_vs_slack.json")
    dec = _load(results_dir, "unbalanced_min_decode.json")
    holm = _load(results_dir, "unbalanced_holm.json")
    if vs is None:
        return "No `unbalanced_vs_slack.json` found."
    technique = (
        "Technique: unbalanced penalization of an inequality constraint (credited in "
        "`qubo_unbalanced.py`; the citation is from memory and was not verified "
        "online). The coefficient rule is fixed in `docs/preregistration.md` and was "
        "not tuned."
    )
    parts = []
    if dec is not None:
        parts.append(
            f"**{len(dec['fail_seeds'])} of {len(dec['rows'])} seeds fail the "
            "brute-force decode check.** The minimum-energy bitstring decodes to the "
            f"exact optimum on {len(dec['pass_seeds'])} of "
            f"{len(dec['rows'])} seeds (lambda {dec['lam']}, {dec['n_loans']} loans): "
            f"pass {dec['pass_seeds']}, fail {dec['fail_seeds']}. On every failing "
            "seed the minimum is budget-feasible and spends the full budget but is a "
            "worse portfolio than the exact optimum: the penalty is almost an equality "
            "constraint at integer amounts (see the dated correction in the "
            "preregistration)."
        )
        fr = [
            [
                r["seed"],
                "pass" if r["passes"] else "FAIL",
                "".join(map(str, r["decoded_x"])),
                f"{r['decoded_spend']} / {r['budget']}",
                "".join(map(str, r["exact_x"])),
                r["exact_spend"],
                r["decoded_objective"],
                r["exact_objective"],
                r["penalty_one_unit_unspent"],
            ]
            for r in dec["rows"]
        ]
        parts.append(
            _table(
                [
                    "seed",
                    "min-energy decode",
                    "decoded x",
                    "spend / budget",
                    "exact x",
                    "exact spend",
                    "decoded objective",
                    "exact objective",
                    "penalty for 1 unspent unit",
                ],
                fr,
            )
        )
    rows = []
    for r in vs["rows"]:
        u, s = r["unbalanced"], r["slack"]
        rows.append(
            [
                r["seed"],
                r["p"],
                f"{u['qubits']} / {s['qubits']}",
                f"{u['depth']} / {s['depth']}",
                f"{u['two_qubit_gates']} / {s['two_qubit_gates']}",
                f"{u['p_optimum']:.4f} / {s['p_optimum']:.4f}",
                f"{u['p_feasible_decoded']:.4f} / {s['p_feasible_decoded']:.4f}",
                u["verdict"],
                s["verdict"],
            ]
        )
    parts.append(technique)
    parts.append(
        "QAOA on both versions (seeds 0-4, p = 1-3, same settings). Cells show "
        "unbalanced / slack. Verdicts are the raw one-sided binomial verdict against "
        "uniform random sampling."
    )
    parts.append(
        _table(
            [
                "seed",
                "p",
                "qubits",
                "logical depth",
                "two-qubit gates",
                "P(optimum)",
                "P(feasible, decoded)",
                "verdict, unbalanced",
                "verdict, slack",
            ],
            rows,
        )
    )
    if holm is not None:
        kept = [r["label"] for r in holm["rows"] if r["survives"]]
        parts.append(
            f"Holm family F2 (the {holm['family_size']} unbalanced runs only): "
            f"{holm['n_raw_below_alpha']} raw p-values below {holm['alpha']}, "
            f"{holm['n_survive_holm']} survive: {', '.join(kept) if kept else 'none'}. "
            "This is its own family and does not change the 18-test family above."
        )
    return "\n\n".join(parts)


def cvar_md(results_dir: Path) -> str:
    """CVaR (alpha 0.1) versus the expectation objective."""
    d = _load(results_dir, "cvar_vs_expectation.json")
    holm = _load(results_dir, "cvar_holm.json")
    if d is None:
        return "No `cvar_vs_expectation.json` found."
    rows = []
    tally = {"cvar": {}, "expectation": {}}
    for r in d["rows"]:
        c, e = r["cvar"], r["expectation"]
        for k, v in (("cvar", c), ("expectation", e)):
            tally[k][v["verdict"]] = tally[k].get(v["verdict"], 0) + 1
        rows.append(
            [
                r["seed"],
                r["p"],
                f"{c['hits']} / {e['hits']}",
                f"{c['p_optimum']:.4f} / {e['p_optimum']:.4f}",
                f"{c['p_feasible_decoded']:.4f} / {e['p_feasible_decoded']:.4f}",
                c["verdict"],
                e["verdict"],
            ]
        )
    fmt = lambda t: ", ".join(f"{n} {v}" for v, n in sorted(t.items()))
    intro = (
        f"CVaR with alpha {d['alpha']}: the optimiser minimises the mean energy of the "
        "lowest-energy 10% of probability mass instead of the mean energy. Same "
        "restarts, iterations, start angles and shots. Cells show CVaR / expectation. "
        f"Verdict counts over the 15 cells: CVaR: {fmt(tally['cvar'])}; "
        f"expectation: {fmt(tally['expectation'])}."
    )
    t = _table(
        [
            "seed",
            "p",
            "optimum hits",
            "P(optimum)",
            "P(feasible, decoded)",
            "verdict, CVaR",
            "verdict, expectation",
        ],
        rows,
    )
    out = intro + "\n\n" + t
    if holm is not None:
        kept = [r["label"] for r in holm["rows"] if r["survives"]]
        out += (
            f"\n\nHolm family F3 (the {holm['family_size']} CVaR runs only): "
            f"{holm['n_raw_below_alpha']} raw p-values below {holm['alpha']}, "
            f"{holm['n_survive_holm']} survive: {', '.join(kept) if kept else 'none'}. "
            "Its own family."
        )
    return out


def noise_ablation_md(results_dir: Path) -> str:
    """Noise ablation: noiseless, noisy, readout mitigation, global-folding ZNE."""
    d = _load(results_dir, "noise_ablation.json")
    if d is None:
        return "No `noise_ablation.json` found."
    rows = []
    for r in d["rows"]:
        v = r["variants"]
        z = r["zne"]
        rows.append(
            [
                r["seed"],
                r["ideal_statevector_p_optimum"],
                v["noiseless"]["p_optimal"],
                v["noisy"]["p_optimal"],
                v["noisy_readout_mitigated"]["p_optimal"],
                z["p_optimum_raw_extrapolated"],
                z["p_optimum_clipped"],
                r["deltas"]["zne_raw_vs_noisy"],
                r["uniform_expectation"],
            ]
        )
    deltas = [r["deltas"]["zne_raw_vs_noisy"] for r in d["rows"]]
    pos = sum(1 for x in deltas if x > 0)
    neg = sum(1 for x in deltas if x < 0)
    intro = (
        "5 loans, p = 1, 4096 shots, `fake_fez` noise model. ZNE is our own global "
        "unitary folding at scale factors 1, 3, 5 with a linear extrapolation to scale "
        "0 of P(optimum). Descriptive only: nothing was tuned. "
        f"ZNE minus noisy is positive on {pos} and negative on {neg} of "
        f"{len(deltas)} seeds, mean {sum(deltas) / len(deltas):.4f}. ZNE circuits are "
        "transpiled at optimization level 1 and folded, so scale 1 is not identical "
        "to the level-3 'noisy' circuit; the JSON records that difference. Per-scale "
        "values and circuit sizes are in `results/noise_ablation.json`."
    )
    t = _table(
        [
            "seed",
            "ideal (statevector)",
            "noiseless",
            "noisy",
            "noisy + readout mitigation",
            "ZNE (raw)",
            "ZNE (clipped)",
            "ZNE raw minus noisy",
            "uniform 1/32",
        ],
        rows,
    )
    sc = d["rows"][0]["zne"]["scales"]
    sizes = _table(
        ["scale", "seed 0 depth", "seed 0 two-qubit gates"],
        [[k, v["depth"], v["two_qubit_gates"]] for k, v in sc.items()],
    )
    return intro + "\n\n" + t + "\n\n" + sizes


def landscape_md(results_dir: Path) -> str:
    """p=1 angle landscape on 5 loans and the saved optimiser angles."""
    d = _load(results_dir, "landscape.json")
    if d is None:
        return "No `landscape.json` found."
    rows = []
    for s in d["per_seed"]:
        sa = s["saved_angles"]
        gm = s["grid_max_p_optimum"]
        hm = s["grid_min_expectation"]
        rows.append(
            [
                s["seed"],
                sa["p_optimum_exact"],
                gm.get("value", gm) if isinstance(gm, dict) else gm,
                s["ratio_saved_p_optimum_to_grid_max"],
                sa["expectation_exact"],
                hm.get("value", hm) if isinstance(hm, dict) else hm,
                s["uniform_p_optimum"],
            ]
        )
    stm = "\n".join(f"- {x}" for x in d["statements"])
    t = _table(
        [
            "seed",
            "P(optimum) at saved angles",
            "grid maximum P(optimum)",
            "ratio",
            "<H> at saved angles",
            "grid minimum <H>",
            "uniform P(optimum)",
        ],
        rows,
    )
    plots = "\n\n".join(
        f"![landscape seed {s['seed']}](plots/landscape_seed{s['seed']}_n5_p1.png)"
        for s in d["per_seed"]
        if (results_dir / "plots" / f"landscape_seed{s['seed']}_n5_p1.png").exists()
    )
    intro = (
        "Exact noiseless P(optimum) and exact <H> on a (beta, gamma) grid at p = 1, "
        "5 loans; the angles are the optimiser's saved angles for the hardware jobs. "
        "Left panel of each plot: P(optimum); right panel: <H>, the quantity the "
        "optimiser minimised."
    )
    return intro + "\n\n" + t + "\n\n" + stm + "\n\n" + plots


def calibration_md(results_dir: Path) -> str:
    """Read-only ibm_fez calibration context for the two saved jobs."""
    d = _load(results_dir, "ibm_calibration.json")
    if d is None:
        return "No `ibm_calibration.json` found."
    if d.get("status") != "ok":
        return f"Skipped: {d.get('reason', 'unknown reason')}."
    rows = [
        [
            j["seed"],
            j["job_id"],
            f"{j['job_record_stats']['depth']} / {j['job_record_stats']['two_qubit_gates']}",
            f"{j['fresh_transpile_stats']['depth']} / {j['fresh_transpile_stats']['two_qubit_gates']}",
            "yes" if j["fresh_matches_job_record_depth_and_2q"] else "no",
            j["median_two_qubit_error"],
            j["median_t1_us"],
            j["median_t2_us"],
            j.get("median_readout_error"),
        ]
        for j in d["jobs"]
    ]
    t = _table(
        [
            "seed",
            "job id",
            "job record depth / 2q",
            "fresh transpile depth / 2q",
            "fresh matches",
            "median 2q error",
            "median T1 (us)",
            "median T2 (us)",
            "median readout error",
        ],
        rows,
    )
    return f"Backend `{d['backend']}`. Caveats: {d['caveats']}\n\n" + t
