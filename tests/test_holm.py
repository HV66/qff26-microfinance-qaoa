"""Tests for the Holm-Bonferroni correction."""

import csv
from pathlib import Path

import pytest

from compare import holm_bonferroni
from scripts import holm_correction


def test_holm_matches_hand_computation() -> None:
    # m = 4, sorted p: 0.01, 0.02, 0.03, 0.04
    # adjusted: 4*0.01=0.04, 3*0.02=0.06, 2*0.03=0.06, 1*0.04=0.04 -> max-so-far
    out = holm_bonferroni([0.03, 0.01, 0.04, 0.02], alpha=0.05)
    assert [round(o["p_holm"], 10) for o in out] == [0.06, 0.04, 0.06, 0.06]
    assert [o["survives"] for o in out] == [False, True, False, False]
    assert [o["rank"] for o in out] == [3, 1, 4, 2]


def test_holm_is_monotone_capped_and_order_preserving() -> None:
    ps = [0.5, 0.0001, 0.2, 0.9, 0.04]
    out = holm_bonferroni(ps)
    assert [o["p_raw"] for o in out] == ps
    by_rank = sorted(out, key=lambda o: o["rank"])
    adj = [o["p_holm"] for o in by_rank]
    assert adj == sorted(adj)
    assert all(a <= 1.0 for a in adj)
    assert all(o["p_holm"] >= o["p_raw"] for o in out)


def test_holm_never_rejects_more_than_raw() -> None:
    ps = [0.001, 0.01, 0.02, 0.03, 0.04, 0.2]
    out = holm_bonferroni(ps)
    assert sum(o["survives"] for o in out) <= sum(p < 0.05 for p in ps)


def test_single_test_unchanged() -> None:
    out = holm_bonferroni([0.03])
    assert out[0]["p_holm"] == pytest.approx(0.03) and out[0]["survives"]


def test_family_includes_hardware_and_counts(tmp_path: Path) -> None:
    with (tmp_path / "compare_vs_random.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "p", "binom_p_value"])
        w.writeheader()
        w.writerow({"seed": 0, "p": 1, "binom_p_value": 0.4})
        w.writerow({"seed": 1, "p": 1, "binom_p_value": 0.001})
    (tmp_path / "hardware_vs_random.json").write_text(
        '{"backend": "x", "n_loans": 5, "binomial_p_value_above_random": 0.01}',
        encoding="utf-8",
    )
    out = holm_correction.compute(tmp_path)
    assert out["family_size"] == 3
    assert out["n_raw_below_alpha"] == 2
    # sorted p: 0.001*3 = 0.003, 0.01*2 = 0.02, 0.4*1 = 0.4
    assert out["n_survive_holm"] == 2
    assert out["rows"][-1]["kind"].startswith("real hardware")
