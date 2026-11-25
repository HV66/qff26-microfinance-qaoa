"""Holm-Bonferroni correction over the whole family of QAOA-vs-random tests.

The family is every one-sided "above random" exact binomial test in the repo:
one per saved QAOA run (results/compare_vs_random.csv) plus one per real-hardware
job (results/hardware_vs_random.json, all jobs). Existing result files are only read.

Run:  python -m scripts.holm_correction
Output: results/holm_correction.json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from compare import ALPHA, holm_bonferroni

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "holm_correction.json"


def build_family(results_dir: Path = RESULTS) -> list[dict[str, Any]]:
    """Return the labelled list of tests (label, raw one-sided p-value)."""
    family: list[dict[str, Any]] = []
    with (results_dir / "compare_vs_random.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            family.append(
                {
                    "label": f"simulated QAOA seed {row['seed']} p={row['p']}",
                    "kind": "simulated QAOA (8 loans)",
                    "p_raw": float(row["binom_p_value"]),
                }
            )
    hw_path = results_dir / "hardware_vs_random.json"
    if hw_path.exists():
        hw = json.loads(hw_path.read_text(encoding="utf-8"))
        # One entry per hardware job; fall back to the single-job layout.
        for job in hw.get("jobs", [hw]):
            family.append(
                {
                    "label": (
                        f"real {job['backend']} {job['n_loans']} loans "
                        f"seed {job.get('seed', 0)} p={job.get('p', 1)}"
                    ),
                    "kind": f"real hardware ({job['n_loans']} loans)",
                    "p_raw": float(job["binomial_p_value_above_random"]),
                }
            )
    return family


def compute(results_dir: Path = RESULTS, alpha: float = ALPHA) -> dict[str, Any]:
    """Return the Holm-adjusted family and a short summary."""
    family = build_family(results_dir)
    adjusted = holm_bonferroni([t["p_raw"] for t in family], alpha)
    rows = [
        {**t, **a, "raw_below_alpha": t["p_raw"] < alpha}
        for t, a in zip(family, adjusted)
    ]
    raw_hits = [r["label"] for r in rows if r["raw_below_alpha"]]
    kept = [r["label"] for r in rows if r["survives"]]
    return {
        "alpha": alpha,
        "family_size": len(rows),
        "n_raw_below_alpha": len(raw_hits),
        "n_survive_holm": len(kept),
        "raw_below_alpha": raw_hits,
        "survive_holm": kept,
        "rows": rows,
    }


def main() -> None:
    """Write results/holm_correction.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = compute()
    OUT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info(
        "family %d, raw p<%.2f: %d, survive Holm: %d",
        out["family_size"],
        out["alpha"],
        out["n_raw_below_alpha"],
        out["n_survive_holm"],
    )


if __name__ == "__main__":
    main()
