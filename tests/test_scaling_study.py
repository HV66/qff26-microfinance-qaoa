"""Tests for scripts/scaling_study.py."""

import pytest

from data import make_portfolio
from qubo import build_qubo
from scripts import scaling_study


def test_measure_without_fake_fez_is_fast_and_consistent() -> None:
    row = scaling_study.measure(4, with_fake_fez=False)
    q = build_qubo(make_portfolio(4, 0), 0.3)
    assert row["n_qubits"] == q.n_vars == row["n_loans"] + row["n_slack"]
    assert (
        row["fake_fez_depth_p1"] is None and "skipped" in row["fake_fez_skipped_reason"]
    )
    assert row["logical_depth_p2"] > row["logical_depth_p1"]
    assert row["logical_two_qubit_p2"] == pytest.approx(2 * row["logical_two_qubit_p1"])
    assert row["brute_force_states"] == 16


def test_fake_fez_counts_are_larger_than_logical() -> None:
    row = scaling_study.measure(4, with_fake_fez=True)
    assert row["fake_fez_depth_p1"] >= row["logical_depth_p1"]
    assert row["fake_fez_two_qubit_p1"] >= 1


def test_extrapolation_is_labelled_and_uses_formula() -> None:
    rows = [scaling_study.measure(n, with_fake_fez=False) for n in (4, 5, 6, 7)]
    ex = scaling_study.extrapolate(rows, 50)
    assert ex["LABEL"] == "EXTRAPOLATION, NOT A MEASUREMENT"
    assert ex["n_loans"] == 50 and ex["brute_force_states"] == 2**50
    assert ex["n_qubits"] == 50 + ex["n_slack"]
    # The fit reproduces a measured size closely.
    fit = ex["two_qubit_p1_fit_crosscheck"]
    r = rows[-1]
    pred = fit["a"] + fit["b"] * r["n_qubits"] ** 2
    assert pred == pytest.approx(r["logical_two_qubit_p1"], rel=0.1)
    assert ex["logical_two_qubit_p1_extrapolated"] > r["logical_two_qubit_p1"]


def test_closed_form_matches_every_measured_size() -> None:
    for n in (4, 5, 6, 8, 10):
        row = scaling_study.measure(n, with_fake_fez=False)
        for p in (1, 2):
            expected = scaling_study.closed_form_two_qubit(row["n_qubits"], p)
            assert row[f"logical_two_qubit_p{p}"] == expected
