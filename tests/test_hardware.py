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
        lambda q, n, s, d=hardware_run.RESULTS, p=None: (
            1,
            np.array([0.3, 0.4]),
            "test",
        ),
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
        lambda q, n, s, d=None, p=None: (1, np.array([0.4, 0.6]), "test"),
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


def test_small_instance_mode_prints_size_before_any_submit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--n-loans 5: qubits, depth, 2q count are printed; nothing is submitted."""
    monkeypatch.setattr(hardware_run, "connect_service", lambda: None)
    monkeypatch.setattr(
        hardware_run,
        "get_params",
        lambda q, n, s, d=hardware_run.RESULTS, p=None: (1, np.array([0.3, 0.4]), "t"),
    )
    submitted: list[int] = []
    monkeypatch.setattr(hardware_run, "submit", lambda *a, **k: submitted.append(1))
    assert hardware_run.main(["--n-loans", "5", "--p", "1"]) == 0
    out = capsys.readouterr().out
    assert "n_loans=5" in out and "qubits=9" in out
    assert "depth=" in out and "two_qubit_gates=" in out
    assert "Small-instance mode" in out
    assert "python hardware_run.py --n-loans 5 --p 1" in out
    assert submitted == []


def test_min_qubits_for_matches_qubo() -> None:
    assert hardware_run.min_qubits_for(5) == 9
    assert hardware_run.min_qubits_for(4) == 8


def test_job_and_result_files_are_keyed_by_instance(tmp_path: Path) -> None:
    """The first job keeps its legacy file names; every other instance gets its own."""
    jf, rf = hardware_run.job_file_for, hardware_run.result_file_for
    assert jf(0, 5, 1, tmp_path) == tmp_path / "hardware_job.json"
    assert rf(0, 5, 1, tmp_path) == tmp_path / "hardware_result.json"
    assert jf(1, 5, 1, tmp_path) == tmp_path / "hardware_job_seed1_n5_p1.json"
    assert rf(1, 5, 1, tmp_path) == tmp_path / "hardware_result_seed1_n5_p1.json"
    assert jf(0, 4, 1, tmp_path) != jf(0, 5, 1, tmp_path)
    assert jf(0, 5, 2, tmp_path) != jf(0, 5, 1, tmp_path)
    paths = {jf(s, n, p, tmp_path) for s in (0, 1, 2) for n in (4, 5) for p in (1, 2)}
    assert len(paths) == 12


def test_params_source_description_is_exact(tmp_path: Path) -> None:
    d = hardware_run.describe_params_source
    # Nothing saved: must say it re-optimises on that exact instance.
    assert "RE-OPTIMISED now" in d(5, 1, 1, tmp_path)
    assert "seed 1, 5 loans, p=1" in d(5, 1, 1, tmp_path)
    (tmp_path / "hardware_params_seed1_n5_p1.json").write_text(
        '{"params": [0.1, 0.2]}', encoding="utf-8"
    )
    assert "CACHED parameters" in d(5, 1, 1, tmp_path)
    # A saved 8-loan run is used only for 8 loans.
    (tmp_path / "qaoa_seed2_p1.json").write_text("{}", encoding="utf-8")
    assert "SAVED parameters" in d(8, 2, 1, tmp_path)
    assert "SAVED" not in d(5, 2, 1, tmp_path)


def _patch_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[int]:
    """Make hardware_run.main run offline against a temp results folder."""
    monkeypatch.setattr(hardware_run, "RESULTS", tmp_path)
    monkeypatch.setattr(
        hardware_run,
        "job_file_for",
        lambda s, n, p, results_dir=tmp_path: hardware_run.RESULTS
        / f"job_seed{s}_n{n}_p{p}.json",
    )
    monkeypatch.setattr(
        hardware_run,
        "get_params",
        lambda q, n, s, d=None, p=None: (1, np.array([0.3, 0.4]), f"test seed {s}"),
    )
    submitted: list[int] = []

    class FakeBackend:
        name = "fake_backend"

        def __init__(self) -> None:
            self.n = 0

    class FakeService:
        def least_busy(self, **kwargs: object) -> FakeBackend:
            return FakeBackend()

    monkeypatch.setattr(hardware_run, "connect_service", lambda: FakeService())
    real_prepare = hardware_run.prepare
    real_fallback = hardware_run.fallback_backend

    def fake_prepare(backend, seed, ladder, p):
        return real_prepare(real_fallback(), seed, ladder, p)

    monkeypatch.setattr(hardware_run, "prepare", fake_prepare)

    def fake_submit(backend, prep, seed):
        submitted.append(seed)
        return {"job_id": f"job{seed}", "backend": "fake_backend", "seed": seed}

    monkeypatch.setattr(hardware_run, "submit", fake_submit)
    return submitted


def test_second_job_does_not_overwrite_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two seeds write two files; resubmitting the same instance is refused."""
    submitted = _patch_offline(monkeypatch, tmp_path)
    assert hardware_run.main(["--n-loans", "4", "--p", "1", "--seed", "0"]) == 0
    first = tmp_path / "job_seed0_n4_p1.json"
    first_bytes = first.read_bytes()
    assert hardware_run.main(["--n-loans", "4", "--p", "1", "--seed", "1"]) == 0
    second = tmp_path / "job_seed1_n4_p1.json"
    assert first.read_bytes() == first_bytes  # untouched
    assert second.exists() and second.read_bytes() != first_bytes
    assert submitted == [0, 1]
    # The same instance again must not overwrite and must not submit.
    assert hardware_run.main(["--n-loans", "4", "--p", "1", "--seed", "0"]) == 1
    assert first.read_bytes() == first_bytes
    assert submitted == [0, 1]
    assert "Refusing to submit" in capsys.readouterr().out


def test_dry_run_prints_parameter_source_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    submitted = _patch_offline(monkeypatch, tmp_path)
    assert (
        hardware_run.main(["--n-loans", "4", "--p", "1", "--seed", "1", "--dry-run"])
        == 0
    )
    out = capsys.readouterr().out
    assert "QAOA parameters for seed 1, 4 loans, p=1" in out
    assert "test seed 1" in out
    assert "Job record path:" in out and "job_seed1_n4_p1.json" in out
    assert submitted == [] and list(tmp_path.iterdir()) == []


def test_no_credentials_path_names_the_parameter_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(hardware_run, "connect_service", lambda: None)
    monkeypatch.setattr(
        hardware_run,
        "get_params",
        lambda q, n, s, d=None, p=None: (1, np.array([0.3, 0.4]), "cached x (seed 1)"),
    )
    assert hardware_run.main(["--n-loans", "4", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "QAOA parameters for seed 1, 4 loans, p=1" in out
    assert "--seed 1" in out and "Job record would be written to:" in out


def test_retrieve_selects_job_by_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        retrieve_hardware, "job_file_for", lambda s, n, p: tmp_path / f"j{s}.json"
    )
    assert retrieve_hardware.main(["--seed", "7"]) == 0
    assert "j7.json" in capsys.readouterr().out
