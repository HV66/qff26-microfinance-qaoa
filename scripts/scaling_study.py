"""Scaling study: how circuit size and classical cost grow with the loan count.

For n_loans = 4..10 on the seed-0 instance, report the qubit count (loans plus
slack bits), logical depth and two-qubit count at p=1 and p=2 (generic gates,
all-to-all connectivity), and the brute-force runtime. For n_loans <= 6 also
report the transpiled depth and two-qubit count on the fake_fez coupling map
at p=1; larger n are skipped for time and flagged in the output. No QAOA
optimisation is run: circuit size does not depend on the angle values, so
fixed placeholder angles are used.

A clearly labelled EXTRAPOLATION to 50 loans applies the same formula; it is
not a measurement.

Run:  python -m scripts.scaling_study
Output: results/scaling_study.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from baseline import brute_force
from data import MAX_UNITS, MIN_UNITS, make_portfolio
from hardware_run import circuit_stats, fallback_backend, transpile_for
from qaoa_run import BASIS, build_ansatz, circuit_cost
from qubo import build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "scaling_study.json"
SEED = 0
LAM = 0.3
N_RANGE = range(4, 11)
MAX_N_FAKE_FEZ = 6
BUDGET_FRACTION = 0.40
EXTRAPOLATE_N = 50
PLACEHOLDER = 0.3


def measure(n_loans: int, with_fake_fez: bool) -> dict[str, Any]:
    """Measure one instance size; no QAOA optimisation is run."""
    pf = make_portfolio(n_loans, SEED)
    q = build_qubo(pf, LAM)
    bf = brute_force(pf, LAM)
    row: dict[str, Any] = {
        "n_loans": n_loans,
        "n_slack": int(q.slack_coeffs.shape[0]),
        "n_qubits": q.n_vars,
        "budget": int(pf.budget),
        "brute_force_runtime_s": bf.runtime_s,
        "brute_force_states": 2**n_loans,
    }
    for p in (1, 2):
        cost = circuit_cost(q, p, SEED)
        row[f"logical_depth_p{p}"] = cost["depth"]
        row[f"logical_two_qubit_p{p}"] = cost["two_qubit_gates"]
    if with_fake_fez:
        backend = fallback_backend()
        circuit = build_ansatz(q, 1).assign_parameters(np.full(2, PLACEHOLDER))
        stats = circuit_stats(transpile_for(circuit, backend))
        row["fake_fez_depth_p1"] = stats["depth"]
        row["fake_fez_two_qubit_p1"] = stats["two_qubit_gates"]
    else:
        row["fake_fez_depth_p1"] = None
        row["fake_fez_two_qubit_p1"] = None
        row["fake_fez_skipped_reason"] = (
            f"skipped for n_loans > {MAX_N_FAKE_FEZ} (transpile time)"
        )
    return row


def expected_budget(n_loans: int) -> float:
    """Expected budget: 40% of the expected total requested amount."""
    mean_amount = (MIN_UNITS + MAX_UNITS) / 2
    return BUDGET_FRACTION * mean_amount * n_loans


def fit_two_qubit_growth(rows: list[dict[str, Any]], p: int) -> dict[str, float]:
    """Least-squares fit of the p-layer two-qubit count to a + b * n_qubits^2.

    The cost operator has one ZZ term per qubit pair present in the QUBO, so
    the two-qubit count grows with the square of the qubit count. The fit is
    used only for the labelled extrapolation.
    """
    x = np.array([r["n_qubits"] ** 2 for r in rows], dtype=float)
    y = np.array([r[f"logical_two_qubit_p{p}"] for r in rows], dtype=float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    return {
        "a": float(a),
        "b": float(b),
        "max_abs_residual": float(np.abs(resid).max()),
    }


def closed_form_two_qubit(n_qubits: int, p: int) -> int:
    """Logical two-qubit count p * 2 * C(N, 2) = p * N * (N - 1).

    The squared budget penalty couples every pair of QUBO variables, so the
    cost operator has one ZZ term per qubit pair, and each ZZ rotation costs
    two CX gates. The formula is checked against every measured size in
    tests/test_scaling_study.py.
    """
    return p * n_qubits * (n_qubits - 1)


def extrapolate(
    rows: list[dict[str, Any]], n_loans: int = EXTRAPOLATE_N
) -> dict[str, Any]:
    """Return an EXTRAPOLATION (not a measurement) to n_loans loans.

    Qubits: n_loans plus bit_length(expected budget) slack bits. Two-qubit
    count: the closed form p * N * (N - 1), which reproduces every measured
    size exactly; a least-squares fit in N^2 is kept as a cross-check. Depth
    is not extrapolated. The brute-force row is exactly 2^n states.
    """
    budget = round(expected_budget(n_loans))
    n_slack = int(budget).bit_length()
    n_qubits = n_loans + n_slack
    out: dict[str, Any] = {
        "LABEL": "EXTRAPOLATION, NOT A MEASUREMENT",
        "n_loans": n_loans,
        "assumed_budget": budget,
        "assumed_budget_rule": "40% of expected total amount (mean amount 4.5)",
        "n_slack": n_slack,
        "n_qubits": n_qubits,
        "brute_force_states": 2**n_loans,
        "brute_force_states_note": "exactly 2^n_loans, from the definition",
    }
    for p in (1, 2):
        fit = fit_two_qubit_growth(rows, p)
        out[f"two_qubit_p{p}_fit_crosscheck"] = fit
        out[f"logical_two_qubit_p{p}_extrapolated"] = closed_form_two_qubit(n_qubits, p)
        out[f"logical_two_qubit_p{p}_fit_value"] = fit["a"] + fit["b"] * n_qubits**2
    return out


def compute() -> dict[str, Any]:
    """Run the whole study and return the payload."""
    rows = [measure(n, with_fake_fez=n <= MAX_N_FAKE_FEZ) for n in N_RANGE]
    return {
        "seed": SEED,
        "lam": LAM,
        "note": (
            "No QAOA optimisation; circuits built with placeholder angles "
            f"({PLACEHOLDER}). Logical = generic gates {BASIS}, all-to-all "
            "connectivity. fake_fez transpiled counts only for "
            f"n_loans <= {MAX_N_FAKE_FEZ}."
        ),
        "rows": rows,
        "extrapolation": extrapolate(rows),
    }


def main() -> None:
    """Write results/scaling_study.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = compute()
    OUT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info(
        "wrote %s (%d measured sizes + 1 extrapolation)", OUT_FILE, len(out["rows"])
    )


if __name__ == "__main__":
    main()
