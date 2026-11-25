"""Step 6b: read-only calibration context for the two saved ibm_fez jobs.

For each saved job record (results/hardware_job.json and
results/hardware_job_seed1_n5_p1.json) the same circuit is re-transpiled exactly
as hardware_run.py does (same angles, optimization_level 3, seed_transpiler 0)
against the live ibm_fez target. From the target we report the median
two-qubit gate error over the two-qubit gate INSTANCES in the transpiled
circuit, and the median T1, T2 (microseconds) and readout error over the
physical qubits the circuit touches.

Two caveats, stored in the output file:

1. A fresh transpile can pick a different physical layout than the one used at
   submission time. The fresh depth and two-qubit count are reported next to
   the job record's so a reader can see whether they match.
2. The numbers are the CURRENT calibration of the backend, not the calibration
   at the time of the jobs (2026-11-25). They are context, not a measurement
   of the conditions of those jobs.

Read-only: no job is submitted. Credentials come only from the saved account
via hardware_run.connect_service(); nothing credential-like is printed or
stored. If there is no usable connection the step is skipped and says so.

Run:  python -m scripts.ibm_calibration
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests
from qiskit import QuantumCircuit
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit_ibm_runtime.exceptions import IBMError

import hardware_run
from data import make_portfolio
from qaoa_run import build_ansatz
from qubo import build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "ibm_calibration.json"
BACKEND_NAME = "ibm_fez"
JOB_FILES = (
    RESULTS / "hardware_job.json",
    RESULTS / "hardware_job_seed1_n5_p1.json",
)
CAVEATS = (
    "(1) The layout of a fresh transpile may differ from the layout used at "
    "submission time; compare the fresh depth and two-qubit count with the job "
    "record's. (2) The calibration data is the CURRENT calibration of the "
    "backend, not the calibration at the time of the jobs (2026-11-25), so this "
    "is context, not a measurement of the conditions of those jobs."
)
SKIP_ERRORS = (
    IBMError,
    QiskitBackendNotFoundError,
    requests.RequestException,
    OSError,
    ValueError,
)


def _median(values: list[float]) -> float | None:
    """Return the median of values, or None if the list is empty."""
    return float(np.median(values)) if values else None


def active_qubits(tqc: QuantumCircuit) -> list[int]:
    """Return sorted physical qubit indices touched by any gate or measurement."""
    used: set[int] = set()
    for inst in tqc.data:
        if inst.operation.name == "barrier":
            continue
        used.update(tqc.find_bit(q).index for q in inst.qubits)
    return sorted(used)


def two_qubit_errors(tqc: QuantumCircuit, target: Any) -> tuple[list[float], list[str]]:
    """Return (errors per two-qubit gate instance, notes about missing data)."""
    errors: list[float] = []
    notes: list[str] = []
    for inst in tqc.data:
        if inst.operation.num_qubits != 2 or inst.operation.name == "barrier":
            continue
        name = inst.operation.name
        pair = tuple(tqc.find_bit(q).index for q in inst.qubits)
        props = target[name].get(pair) if name in target.operation_names else None
        if props is None or props.error is None:
            notes.append(f"no error for {name} on {pair}")
            continue
        errors.append(float(props.error))
    return errors, notes


def calibration_summary(tqc: QuantumCircuit, target: Any) -> dict[str, Any]:
    """Return median 2q error, T1, T2 (us) and readout error for a circuit.

    Args:
        tqc: Transpiled circuit (physical qubit indices).
        target: A qiskit Target with error and qubit properties.
    """
    qubits = active_qubits(tqc)
    errs, notes = two_qubit_errors(tqc, target)
    t1: list[float] = []
    t2: list[float] = []
    ro: list[float] = []
    for q in qubits:
        qp = target.qubit_properties[q] if target.qubit_properties else None
        if qp is not None and qp.t1 is not None:
            t1.append(qp.t1 * 1e6)
        if qp is not None and qp.t2 is not None:
            t2.append(qp.t2 * 1e6)
        if "measure" in target.operation_names:
            props = target["measure"].get((q,))
            if props is not None and props.error is not None:
                ro.append(float(props.error))
    return {
        "physical_qubits_used": qubits,
        "n_qubits_used": len(qubits),
        "n_two_qubit_gate_instances": len(errs),
        "median_two_qubit_error": _median(errs),
        "median_t1_us": _median(t1),
        "median_t2_us": _median(t2),
        "median_readout_error": _median(ro),
        "notes": notes,
    }


def analyse_job(record_path: Path, backend: Any) -> dict[str, Any]:
    """Re-transpile one saved job's circuit on the backend and summarise it."""
    rec = hardware_run.load_job_record(record_path)
    inst = rec["instance"]
    pf = make_portfolio(inst["n_loans"], inst["seed"])
    q = build_qubo(pf, inst["lam"])
    p, params, provenance = hardware_run.get_params(
        q, inst["n_loans"], inst["seed"], p=inst["p"]
    )
    circuit = build_ansatz(q, p).assign_parameters(params)
    tqc = hardware_run.transpile_for(circuit, backend)
    fresh = hardware_run.circuit_stats(tqc)
    return {
        "job_record": record_path.as_posix(),
        "job_id": rec["job_id"],
        "seed": inst["seed"],
        "n_loans": inst["n_loans"],
        "p": p,
        "params_source": provenance,
        "transpile_settings": {
            "optimization_level": hardware_run.OPT_LEVEL,
            "seed_transpiler": hardware_run.TRANSPILE_SEED,
        },
        "job_record_stats": rec["transpile"],
        "fresh_transpile_stats": fresh,
        "fresh_matches_job_record_depth_and_2q": (
            fresh["depth"] == rec["transpile"]["depth"]
            and fresh["two_qubit_gates"] == rec["transpile"]["two_qubit_gates"]
        ),
        **calibration_summary(tqc, backend.target),
    }


def skipped(reason: str) -> dict[str, Any]:
    """Return the output record for a skipped step."""
    return {"status": "skipped", "reason": reason}


def run(
    job_files: tuple[Path, ...] = JOB_FILES, backend_name: str = BACKEND_NAME
) -> dict[str, Any]:
    """Connect read-only and build the calibration record, or a skip record."""
    try:
        service = hardware_run.connect_service()
        if service is None:
            return skipped("no usable saved IBM Quantum account (connect_service None)")
        backend = service.backend(backend_name)
        jobs = [analyse_job(path, backend) for path in job_files]
    except SKIP_ERRORS as exc:
        return skipped(f"connection or backend error: {type(exc).__name__}")
    return {
        "status": "ok",
        "backend": backend_name,
        "caveats": CAVEATS,
        "calibration_is_current_not_at_job_time": True,
        "jobs": jobs,
    }


def main() -> int:
    """Write results/ibm_calibration.json; always exit 0 (skip is not an error)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = run()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    if out["status"] == "skipped":
        print(f"Calibration step skipped: {out['reason']}")
    else:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
