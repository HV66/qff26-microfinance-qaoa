"""Holm-Bonferroni correction over the unbalanced-QAOA family (F2).

The family is the 15 one-sided "above random" exact binomial tests of
results/unbalanced_seed{S}_p{P}.json (seeds 0-4, p = 1-3). It is its own
family and is not mixed with results/holm_correction.json.

Run:  python -m scripts.unbalanced_holm
Output: results/unbalanced_holm.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from compare import ALPHA, holm_bonferroni

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "unbalanced_holm.json"
SEEDS = (0, 1, 2, 3, 4)
PS = (1, 2, 3)


def build_family(results_dir: Path = RESULTS) -> list[dict[str, Any]]:
    """Return labelled tests (label, raw one-sided p-value) from saved runs."""
    family: list[dict[str, Any]] = []
    for seed in SEEDS:
        for p in PS:
            path = results_dir / f"unbalanced_seed{seed}_p{p}.json"
            res = json.loads(path.read_text(encoding="utf-8"))
            family.append(
                {
                    "label": f"unbalanced QAOA seed {seed} p={p}",
                    "kind": "unbalanced QAOA (8 loans, 8 qubits)",
                    "p_raw": float(res["sampling"]["binom_p_value_above_random"]),
                }
            )
    return family


def compute(
    results_dir: Path = RESULTS,
    alpha: float = ALPHA,
    family: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the Holm-adjusted family and a short summary."""
    fam = build_family(results_dir) if family is None else family
    adjusted = holm_bonferroni([t["p_raw"] for t in fam], alpha)
    rows = [
        {**t, **a, "raw_below_alpha": t["p_raw"] < alpha} for t, a in zip(fam, adjusted)
    ]
    return {
        "alpha": alpha,
        "family": "F2: unbalanced QAOA vs uniform random",
        "family_size": len(rows),
        "n_raw_below_alpha": sum(r["raw_below_alpha"] for r in rows),
        "n_survive_holm": sum(r["survives"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    """Write results/unbalanced_holm.json."""
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
