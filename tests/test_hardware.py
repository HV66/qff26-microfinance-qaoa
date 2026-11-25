"""Tests for hardware_run, retrieve_hardware and noise_run (offline only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import hardware_run
import noise_run
import retrieve_hardware
from data import make_portfolio
from qubo import build_qubo


def test_sampler_option_names_valid() -> None:
    """Every option name we use exists in the installed SamplerOptions."""
    hardware_run.validate_options(hardware_run.sampler_options())


def test_bad_option_name_rejected() -> None:
    """A made-up option name must be rejected (validation is real)."""
    with pytest.raises((TypeError, ValueError)):
        hardware_run.validate_options({"dynamical_decoupling": {"enabled": True}})


def test_no_credentials_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without credentials: message, exit code 0, nothing submitted."""
    monkeypatch.setattr(hardware_run, "connect_service", lambda: None)
    monkeypatch.setattr(hardware_run, "INSTANCE_LADDER", (4,))
    monkeypatch.setattr(
        hardware_run,
        "get_params",
        lambda q, n, s, d=hardware_run.RESULTS: (1, np.array([0.3, 0.4]), "test"),
    )
    assert hardware_run.main(["--dry-run"]) == 0
    assert "No IBM credentials" in capsys.readouterr().out


def test_job_record_round_trip(tmp_path: Path) -> None:
    """Saving and loading the job record preserves it."""
    record = {"job_id": "abc", "backend": "x", "options": {"a": {"b": 1}}}
    path = hardware_run.save_job_record(record, tmp_path / "sub" / "job.json")
    assert hardware_run.load_job_record(path) == record


def test_retrieve_without_job_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No job file: clean message, exit code 0."""
    code = retrieve_hardware.main(["--job-file", str(tmp_path / "none.json")])
    assert code == 0
    assert "not found" in capsys.readouterr().out


def test_score_counts_perfect_optimum() -> None:
    """All shots on the optimal loan vector give P(optimum) = 1."""
    from baseline import brute_force

    pf = make_portfolio(4, 0)
    q = build_qubo(pf, 0.3)
    bf = brute_force(pf, 0.3)
    bits = [str(int(v)) for v in bf.x][::-1]
    key = "0" * (q.n_vars - q.n_loans) + "".join(bits)
    inst = {"n_loans": 4, "seed": 0, "lam": 0.3}
    metrics = retrieve_hardware.score_counts(inst, {key: 100})
    assert metrics["p_optimal"] == 1.0
    assert metrics["p_feasible_decoded"] == 1.0


def test_mitigate_counts_inverts_known_confusion() -> None:
    """Readout mitigation recovers a state distorted by a known confusion."""
    a = np.array([[0.9, 0.1], [0.1, 0.9]])
    shots = 100000
    counts = {"0": int(0.9 * shots), "1": int(0.1 * shots)}
    out = noise_run.mitigate_counts(counts, [a], shots, seed=0)
    assert out.get("0", 0) / shots > 0.99


def test_noise_run_tiny_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end on 4 loans (8 qubits); JSON has all three variants."""
    monkeypatch.setattr(
        noise_run,
        "get_params",
        lambda q, n, s, d=None: (1, np.array([0.4, 0.6]), "test"),
    )
    out_file = tmp_path / "noise.json"
    assert (
        noise_run.main(["--n-loans", "4", "--shots", "512", "--out", str(out_file)])
        == 0
    )
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert set(data["variants"]) == {
        "noiseless",
        "noisy",
        "noisy_readout_mitigated",
    }
    assert data["transpiled"]["two_qubit_gates"] > 0
    for m in data["variants"].values():
        assert 0.0 <= m["p_optimal"] <= 1.0
