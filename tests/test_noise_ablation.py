"""Offline tests for the noise ablation (own global-folding ZNE)."""

import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scripts import noise_ablation as na


def _small_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.7, 1)
    qc.sx(2)
    qc.cx(1, 2)
    qc.rx(0.3, 0)
    return qc


def test_fold_scale_one_is_identity_copy() -> None:
    qc = _small_circuit()
    folded = na.fold_circuit(qc, 1)
    assert folded.count_ops() == qc.count_ops()


def test_folding_preserves_ideal_distribution() -> None:
    qc = _small_circuit()
    base = Statevector(qc).probabilities()
    for scale in (3, 5):
        folded = na.fold_circuit(qc, scale)
        assert Statevector(folded).probabilities() == pytest.approx(base, abs=1e-12)


def test_fold_gate_counts_scale_linearly() -> None:
    qc = _small_circuit()
    n_cx = qc.count_ops()["cx"]
    for scale in (1, 3, 5):
        assert na.fold_circuit(qc, scale).count_ops()["cx"] == scale * n_cx


def test_fold_rejects_even_scale() -> None:
    with pytest.raises(ValueError):
        na.fold_circuit(_small_circuit(), 2)


def test_linear_extrapolation_hand_example() -> None:
    # Points (1, 0.5), (3, 0.3), (5, 0.1): slope -0.1, intercept 0.6.
    raw, clipped = na.extrapolate_to_zero((1, 3, 5), (0.5, 0.3, 0.1))
    assert raw == pytest.approx(0.6)
    assert clipped == pytest.approx(0.6)


def test_extrapolation_clipping_marks_both() -> None:
    raw, clipped = na.extrapolate_to_zero((1, 3, 5), (0.5, 0.3, 0.1))
    assert raw == clipped
    raw, clipped = na.extrapolate_to_zero((1, 3, 5), (0.9, 0.6, 0.3))
    assert raw == pytest.approx(1.05)
    assert clipped == 1.0
    raw, clipped = na.extrapolate_to_zero((1, 3, 5), (0.01, 0.1, 0.19))
    assert raw < 0.0
    assert clipped == 0.0


def test_loan_index_and_ideal_probability_sums_slack() -> None:
    probs = np.zeros(2**5)
    probs[0b10110] = 0.25  # loans (low 3 bits) = 0b110
    probs[0b00110] = 0.25  # same loans, different slack bits
    probs[0b00111] = 0.5
    assert na.loan_index(np.array([0, 1, 1])) == 0b110
    assert na.p_optimum_from_probs(probs, 3, 0b110) == pytest.approx(0.5)


def test_end_to_end_four_loans(tmp_path: Path) -> None:
    res = na.run_seed(seed=2, n_loans=4, shots=256, results_dir=tmp_path)
    assert set(res["zne"]["scales"]) == {"1", "3", "5"}
    depths = [res["zne"]["scales"][s]["depth"] for s in ("1", "3", "5")]
    assert depths == sorted(depths)
    two_q = [res["zne"]["scales"][s]["two_qubit_gates"] for s in ("1", "3", "5")]
    assert two_q[1] == 3 * two_q[0] and two_q[2] == 5 * two_q[0]
    assert 0.0 <= res["zne"]["p_optimum_clipped"] <= 1.0
    assert res["uniform_expectation"] == pytest.approx(1 / 2**4)
    for key in ("noiseless", "noisy", "noisy_readout_mitigated"):
        assert 0.0 <= res["variants"][key]["p_optimal"] <= 1.0
    cache = tmp_path / "noise_ablation_params_seed2_n4_p1.json"
    assert json.loads(cache.read_text(encoding="utf-8"))["params"]
