"""Tests for the CVaR objective, its runner and the CVaR Holm family."""

import json
from pathlib import Path

import numpy as np
import pytest

import qaoa_cvar
import qaoa_run
from data import make_portfolio
from qubo import build_qubo
from scripts import cvar_holm


def test_cvar_alpha_one_is_expectation() -> None:
    rng = np.random.default_rng(1)
    probs = rng.random(16)
    probs /= probs.sum()
    energies = rng.normal(size=16)
    assert qaoa_cvar.cvar(probs, energies, 1.0) == pytest.approx(probs @ energies)


def test_cvar_hand_built_distribution() -> None:
    # Energies sorted: -3 (0.05), -1 (0.15), 2 (0.8). alpha 0.1 -> only -3 (0.05)
    # plus 0.05 of the -1 state: (0.05*-3 + 0.05*-1) / 0.1 = -2.
    probs = np.array([0.8, 0.05, 0.15])
    energies = np.array([2.0, -3.0, -1.0])
    assert qaoa_cvar.cvar(probs, energies, 0.1) == pytest.approx(-2.0)


def test_cvar_boundary_exact_state_mass() -> None:
    # Boundary falls exactly at the end of the lowest state: result is that energy.
    probs = np.array([0.9, 0.1])
    energies = np.array([5.0, -4.0])
    assert qaoa_cvar.cvar(probs, energies, 0.1) == pytest.approx(-4.0)


def test_cvar_fractional_share_of_boundary_state() -> None:
    probs = np.array([0.5, 0.5])
    energies = np.array([1.0, 3.0])
    # alpha 0.25 lies entirely inside the lowest state.
    assert qaoa_cvar.cvar(probs, energies, 0.25) == pytest.approx(1.0)
    # alpha 0.75: 0.5 of energy 1 and 0.25 of energy 3 -> (0.5 + 0.75) / 0.75.
    assert qaoa_cvar.cvar(probs, energies, 0.75) == pytest.approx(1.25 / 0.75)


def test_cvar_is_monotone_below_expectation() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        probs = rng.random(32)
        probs /= probs.sum()
        energies = rng.normal(size=32)
        exp = probs @ energies
        values = [qaoa_cvar.cvar(probs, energies, a) for a in (0.05, 0.1, 0.5, 1.0)]
        assert all(v <= exp + 1e-12 for v in values)
        assert values == sorted(values)


def test_cvar_point_mass_is_that_energy() -> None:
    probs = np.zeros(8)
    probs[5] = 1.0
    energies = np.arange(8, dtype=float)
    assert qaoa_cvar.cvar(probs, energies, 0.1) == pytest.approx(5.0)


def test_cvar_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        qaoa_cvar.cvar(np.array([1.0]), np.array([0.0]), 0.0)


def test_cvar_matches_qaoa_distribution_at_alpha_one() -> None:
    q = build_qubo(make_portfolio(4, 0), 0.3)
    params = np.array([0.4, 0.7])
    probs = qaoa_run.fast_probabilities(
        qaoa_run.scaled_cost_diagonal(q), q.n_vars, params
    )
    energies = qaoa_run.energy_diagonal(q)
    assert qaoa_cvar.cvar(probs, energies, 1.0) == pytest.approx(probs @ energies)


def test_end_to_end_small(tmp_path: Path) -> None:
    res = qaoa_cvar.run_one(
        seed=0, p=1, n_loans=4, shots=512, restarts=1, maxiter=20, results_dir=tmp_path
    )
    assert res["config"]["objective"] == "CVaR"
    assert res["config"]["alpha"] == 0.1
    assert res["config"]["shots"] == 512
    opt = res["optimiser"]
    assert opt["best_cvar"] <= opt["expectation_at_best_params"] + 1e-9
    assert 0.0 <= res["sampling"]["p_optimal"] <= 1.0
    path = qaoa_cvar.save(res, tmp_path)
    assert path.name == "cvar_seed0_p1.json"
    assert json.loads(path.read_text(encoding="utf-8"))["config"]["seed"] == 0


def test_comparison_row_verdicts() -> None:
    cvar_res = {
        "config": {"seed": 0, "p": 1, "shots": 4096, "n_loans": 8},
        "sampling": {
            "p_optimal": 0.5,
            "p_feasible_decoded": 0.9,
            "best_feasible_gap_pct": 0.0,
        },
        "optimiser": {"expectation_at_best_params": -1.0},
    }
    exp_res = {
        "config": {"seed": 0, "p": 1, "shots": 4096, "n_loans": 8},
        "sampling": {
            "p_optimal": 2**-8,
            "p_feasible_decoded": 0.3,
            "best_feasible_gap_pct": 5.0,
        },
        "optimiser": {"best_value": 2.0},
    }
    row = qaoa_cvar.comparison_row(cvar_res, exp_res)
    assert row["cvar"]["verdict"] == "above random"
    assert row["expectation"]["verdict"] == "indistinguishable from random"
    assert row["cvar"]["hits"] == 2048
    assert row["expectation"]["expectation_at_angles"] == 2.0


def test_cvar_holm_on_synthetic_files(tmp_path: Path) -> None:
    for seed in range(5):
        for p in (1, 2, 3):
            hits = 2000 if (seed + p) % 2 == 0 else 16
            payload = {
                "config": {"seed": seed, "p": p, "shots": 4096, "n_loans": 8},
                "sampling": {"p_optimal": hits / 4096},
            }
            (tmp_path / f"cvar_seed{seed}_p{p}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    out = cvar_holm.compute(tmp_path)
    assert out["family_size"] == 15
    assert out["n_raw_below_alpha"] == 7
    assert out["n_survive_holm"] == 7
    assert len(out["rows"]) == 15
    assert all("p_holm" in r for r in out["rows"])


def test_cvar_holm_requires_all_15(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cvar_holm.compute(tmp_path)
