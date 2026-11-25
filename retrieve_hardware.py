"""Fetch the result of the job submitted by hardware_run.py and score it.

Reads results/hardware_job.json, retrieves the job from IBM Quantum (saved
account or QISKIT_IBM_TOKEN only), decodes the counts with the shared
qubo.decode_loans / bits_from_string helpers, computes the same metrics as the
simulations (qaoa_run.sample_metrics) and writes the result file for that
instance: results/hardware_result.json for the first job (seed 0, 5 loans,
p=1), results/hardware_result_seed{S}_n{N}_p{P}.json for any other. Select a
job with --seed, --n-loans and --p.
If the job is not finished yet it says so and exits with code 0.

Run:  python retrieve_hardware.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from baseline import brute_force
from data import make_portfolio
from hardware_run import (
    connect_service,
    job_file_for,
    load_job_record,
    result_file_for,
)
from qaoa_run import sample_metrics
from qubo import build_qubo

log = logging.getLogger(__name__)

DONE_STATUSES = {"DONE"}
FAILED_STATUSES = {"ERROR", "CANCELLED"}


def score_counts(instance: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    """Compute sample metrics for counts on the recorded instance."""
    pf = make_portfolio(instance["n_loans"], instance["seed"])
    q = build_qubo(pf, instance["lam"])
    bf = brute_force(pf, instance["lam"])
    return sample_metrics(pf, q, counts, np.array(bf.x), bf.objective)


def extract_counts(result: Any) -> dict[str, int]:
    """Return counts from a SamplerV2 PrimitiveResult (classical register meas)."""
    return dict(result[0].data.meas.get_counts())


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-loans", type=int, default=5)
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    job_file = args.job_file or job_file_for(args.seed, args.n_loans, args.p)
    if not job_file.exists():
        print(f"{job_file} not found: no hardware job was submitted.")
        return 0
    record = load_job_record(job_file)
    inst = record["instance"]
    result_file = result_file_for(inst["seed"], inst["n_loans"], inst["p"])
    service = connect_service()
    if service is None:
        print("No IBM credentials: cannot fetch the job.")
        return 0
    job = service.job(record["job_id"])
    status = str(job.status())
    if status in FAILED_STATUSES:
        print(f"Job {record['job_id']} ended with status {status}.")
        return 1
    if status not in DONE_STATUSES:
        print(
            f"Job {record['job_id']} not finished yet (status {status}). Retry later."
        )
        return 0
    counts = extract_counts(job.result())
    out = {
        "job_id": record["job_id"],
        "backend": record["backend"],
        "submitted_utc": record["submitted_utc"],
        "instance": record["instance"],
        "transpile": record["transpile"],
        "options": record["options"],
        "metrics": score_counts(record["instance"], counts),
        "counts": counts,
    }
    result_file.write_text(json.dumps(out, indent=1), encoding="utf-8")
    m = out["metrics"]
    print(
        f"P(optimum)={m['p_optimal']:.4f} P(feasible decoded)="
        f"{m['p_feasible_decoded']:.4f} gap%={m['best_feasible_gap_pct']} "
        f"saved {result_file}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
