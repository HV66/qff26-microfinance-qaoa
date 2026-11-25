"""Holm-Bonferroni correction over the 15 CVaR QAOA-vs-random tests (family F3).

Own family, separate from results/holm_correction.json. Reads
results/cvar_seed{S}_p{P}.json for seeds 0..4 and p = 1, 2, 3.

Run:  python -m scripts.cvar_holm
Output: results/cvar_holm.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

from compare import ALPHA, holm_bonferroni

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "cvar_holm.json"
SEEDS = (0, 1, 2, 3, 4)
PS = (1, 2, 3)


def build_family(results_dir: Path = RESULTS) -> list[dict[str, Any]]:
    """Return the 15 labelled one-sided 'above random' tests.

    Raises:
        FileNotFoundError: If any of the 15 CVaR result files is missing.
    """
    family: list[dict[str, Any]] = []
    for seed in SEEDS:
        for p in PS:
            path = results_dir / f"cvar_seed{seed}_p{p}.json"
            res = json.loads(path.read_text(encoding="utf-8"))
            cfg = res["config"]
            shots = int(cfg["shots"])
            hits = round(res["sampling"]["p_optimal"] * shots)
            rate = 2.0 ** -int(cfg["n_loans"])
            p_raw = float(binomtest(hits, shots, rate, alternative="greater").pvalue)
            family.append(
                {
                    "label": f"CVaR QAOA seed {seed} p={p}",
                    "seed": seed,
                    "p": p,
                    "hits": hits,
                    "shots": shots,
                    "p_raw": p_raw,
                }
            )
    return family


def compute(results_dir: Path = RESULTS, alpha: float = ALPHA) -> dict[str, Any]:
    """Return the Holm-adjusted family and a summary."""
    family = build_family(results_dir)
    adjusted = holm_bonferroni([t["p_raw"] for t in family], alpha)
    rows = [
        {**t, **a, "raw_below_alpha": t["p_raw"] < alpha}
        for t, a in zip(family, adjusted)
    ]
    return {
        "alpha": alpha,
        "family": "F3: CVaR QAOA vs uniform random, one-sided exact binomial",
        "family_size": len(rows),
        "n_raw_below_alpha": sum(r["raw_below_alpha"] for r in rows),
        "n_survive_holm": sum(r["survives"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    """Write results/cvar_holm.json."""
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
