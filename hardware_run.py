"""Submit the optimised QAOA circuit to a real IBM Quantum backend.

Pipeline: build QUBO (lam=0.3) -> QAOA ansatz -> assign optimised parameters
(from results/qaoa_seed{S}_p{P}.json, best p = highest sampling.p_optimal;
re-optimised with qaoa_run.optimise if the file is missing) -> transpile for
the backend -> SamplerV2 with dynamical decoupling and Pauli twirling ->
submit and RETURN IMMEDIATELY (results are fetched later by
retrieve_hardware.py). The job metadata goes to results/hardware_job.json.

Credentials: only the saved account (QiskitRuntimeService.save_account, done
by the user outside this repo) or the QISKIT_IBM_TOKEN environment variable.
This script never prints, stores or asks for a token. Without credentials it
prints the transpiled depth / 2-qubit gate count for a generic fallback target
(fake_fez, a Heron-style device) and exits with code 0.

Instance size: the full instance is 8 loans = 12 qubits. Depth after routing
is checked against MAX_TWO_QUBIT_GATES / MAX_DEPTH. If either is exceeded the
script falls back to the next smaller instance (5 loans = 9 qubits, then 4
loans = 8 qubits) and records why in the "reason" field of the saved JSON.
(Which size was actually used is decided by the real transpile result, see the
saved JSON; it is not asserted here.)

Verified against the installed qiskit-ibm-runtime 0.52.1: SamplerOptions has
dynamical_decoupling.{enable, sequence_type, ...} and twirling.{enable_gates,
enable_measure, num_randomizations, strategy}. See sampler_options().

Run:  python hardware_run.py --dry-run
      python hardware_run.py            (needs a saved IBM account / env token)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from baseline import brute_force
from data import LoanPortfolio, make_portfolio
from qaoa_run import build_ansatz, optimise
from qubo import Qubo, build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
JOB_FILE = RESULTS / "hardware_job.json"
LAM = 0.3
SEED = 0
SHOTS = 4096
OPT_LEVEL = 3
TRANSPILE_SEED = 0
MAX_TWO_QUBIT_GATES = 300
MAX_DEPTH = 1500
INSTANCE_LADDER = (8, 5, 4)
CANDIDATE_PS = (1, 2, 3, 4)


def sampler_options() -> dict[str, Any]:
    """Return the SamplerV2 options dict (names verified against 0.52.1)."""
    return {
        "default_shots": SHOTS,
        "dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"},
        "twirling": {
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": "auto",
            "shots_per_randomization": "auto",
            "strategy": "active-accum",
        },
    }


def validate_options(options: dict[str, Any]) -> None:
    """Raise if options contain a name SamplerOptions does not define."""
    from qiskit_ibm_runtime.options import SamplerOptions

    SamplerOptions(**options)


def best_p_from_results(seed: int, results_dir: Path = RESULTS) -> int | None:
    """Return the p with the highest sampling.p_optimal in seed results, if any."""
    best: tuple[float, int] | None = None
    for p in CANDIDATE_PS:
        path = results_dir / f"qaoa_seed{seed}_p{p}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        score = float(data["sampling"]["p_optimal"])
        if best is None or score > best[0]:
            best = (score, p)
    return None if best is None else best[1]


def get_params(
    q: Qubo, n_loans: int, seed: int, results_dir: Path = RESULTS
) -> tuple[int, np.ndarray, str]:
    """Return (p, optimised params, provenance string).

    Uses the saved seed-0 result for the full 8-loan instance when present.
    Otherwise (missing file or a reduced instance) re-optimises with
    qaoa_run.optimise at the best saved p, or p=1 if nothing is saved.
    """
    saved_p = best_p_from_results(seed, results_dir)
    p = 1 if saved_p is None else saved_p
    path = results_dir / f"qaoa_seed{seed}_p{p}.json"
    if saved_p is not None and n_loans == 8 and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return p, np.array(data["optimiser"]["best_params"]), f"saved {path.name}"
    cache = results_dir / f"hardware_params_seed{seed}_n{n_loans}_p{p}.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        return p, np.array(saved["params"]), f"cached {cache.name}"
    opt = optimise(q, p, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"params": opt["best_params"], "best_value": opt["best_value"]}),
        encoding="utf-8",
    )
    return p, np.array(opt["best_params"]), f"re-optimised p={p} n_loans={n_loans}"


def transpile_for(
    circuit: QuantumCircuit, backend: Any, seed: int = TRANSPILE_SEED
) -> QuantumCircuit:
    """Transpile for a backend (routing + native gates), measure_all included."""
    measured = circuit.copy()
    measured.measure_all()
    return transpile(
        measured, backend=backend, optimization_level=OPT_LEVEL, seed_transpiler=seed
    )


def circuit_stats(tqc: QuantumCircuit) -> dict[str, int]:
    """Return depth and two-qubit gate count of a transpiled circuit."""
    two_q = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
    return {"depth": int(tqc.depth()), "two_qubit_gates": int(two_q)}


def too_deep(stats: dict[str, int]) -> bool:
    """Return True if the circuit is unrealistic for current hardware."""
    return stats["two_qubit_gates"] > MAX_TWO_QUBIT_GATES or stats["depth"] > MAX_DEPTH


def prepare(
    backend: Any, seed: int = SEED, ladder: tuple[int, ...] = INSTANCE_LADDER
) -> dict[str, Any]:
    """Pick the largest instance that fits the depth limits on this backend.

    Returns a dict with the portfolio, qubo, p, params, transpiled circuit,
    stats, instance config and a human-readable "reason".
    """
    tried: list[str] = []
    for n_loans in ladder:
        pf = make_portfolio(n_loans, seed)
        q = build_qubo(pf, LAM)
        p, params, provenance = get_params(q, n_loans, seed)
        circuit = build_ansatz(q, p).assign_parameters(params)
        tqc = transpile_for(circuit, backend)
        stats = circuit_stats(tqc)
        log.info(
            "n_loans=%d qubits=%d p=%d depth=%d two_qubit=%d",
            n_loans,
            q.n_vars,
            p,
            stats["depth"],
            stats["two_qubit_gates"],
        )
        if not too_deep(stats) or n_loans == ladder[-1]:
            if tried:
                reason = (
                    "Smaller instance used because larger ones exceeded "
                    f"{MAX_TWO_QUBIT_GATES} 2q gates or depth {MAX_DEPTH} after "
                    "routing: " + "; ".join(tried)
                )
            elif too_deep(stats):
                reason = "Smallest instance; still above the depth limits."
            else:
                reason = "Full instance fits the depth limits after routing."
            return {
                "pf": pf,
                "q": q,
                "p": p,
                "params": params,
                "provenance": provenance,
                "tqc": tqc,
                "stats": stats,
                "n_loans": n_loans,
                "reason": reason,
            }
        tried.append(
            f"n_loans={n_loans} ({q.n_vars}q): depth={stats['depth']}, "
            f"2q={stats['two_qubit_gates']}"
        )
    raise RuntimeError("empty instance ladder")


def fallback_backend() -> Any:
    """Return a generic offline target: fake_fez (156-qubit Heron-style)."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    return FakeFez()


def connect_service() -> Any | None:
    """Return a QiskitRuntimeService, or None if no usable credentials.

    Reads only the saved account or QISKIT_IBM_TOKEN. Never touches the token.
    """
    import requests
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.exceptions import IBMError

    try:
        return QiskitRuntimeService()
    except (IBMError, requests.RequestException, OSError, ValueError) as exc:
        log.warning("No usable IBM Quantum credentials (%s).", type(exc).__name__)
        return None


def save_job_record(record: dict[str, Any], path: Path = JOB_FILE) -> Path:
    """Write the job record as JSON (creates the folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return path


def load_job_record(path: Path = JOB_FILE) -> dict[str, Any]:
    """Read a job record written by save_job_record."""
    return json.loads(path.read_text(encoding="utf-8"))


def instance_config(prep: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return the JSON-serialisable instance description for the job record."""
    pf: LoanPortfolio = prep["pf"]
    bf = brute_force(pf, LAM)
    return {
        "seed": seed,
        "n_loans": prep["n_loans"],
        "n_qubits": prep["q"].n_vars,
        "lam": LAM,
        "p": prep["p"],
        "params": prep["params"].tolist(),
        "params_source": prep["provenance"],
        "exact_objective": bf.objective,
        "exact_x": bf.x.tolist(),
    }


def submit(backend: Any, prep: dict[str, Any], seed: int) -> dict[str, Any]:
    """Submit the transpiled circuit with SamplerV2 and return the job record."""
    from qiskit_ibm_runtime import SamplerV2

    options = sampler_options()
    validate_options(options)
    sampler = SamplerV2(mode=backend, options=options)
    job = sampler.run([prep["tqc"]], shots=SHOTS)
    return {
        "job_id": job.job_id(),
        "backend": backend.name,
        "submitted_utc": dt.datetime.now(dt.UTC).isoformat(),
        "shots": SHOTS,
        "options": options,
        "transpile": {
            "optimization_level": OPT_LEVEL,
            "seed_transpiler": TRANSPILE_SEED,
            **prep["stats"],
        },
        "instance": instance_config(prep, seed),
        "reason": prep["reason"],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="do not submit")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    service = connect_service()
    if service is None:
        backend: Any = fallback_backend()
        prep = prepare(backend, args.seed)
        print(
            f"No IBM credentials found: nothing submitted. Generic target "
            f"{backend.name}: n_loans={prep['n_loans']} depth={prep['stats']['depth']} "
            f"two_qubit_gates={prep['stats']['two_qubit_gates']}"
        )
        print(prep["reason"])
        return 0

    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=12)
    prep = prepare(backend, args.seed)
    print(
        f"Backend {backend.name}: n_loans={prep['n_loans']} "
        f"depth={prep['stats']['depth']} "
        f"two_qubit_gates={prep['stats']['two_qubit_gates']}"
    )
    print(prep["reason"])
    if args.dry_run:
        validate_options(sampler_options())
        print("Dry run: options valid, job NOT submitted.")
        return 0
    record = submit(backend, prep, args.seed)
    path = save_job_record(record)
    print(f"Submitted job {record['job_id']} to {record['backend']}; saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
