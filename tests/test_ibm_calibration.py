"""Offline tests for scripts.ibm_calibration (no network)."""

import json
from pathlib import Path

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library import CZGate
from qiskit.transpiler import InstructionProperties, QubitProperties, Target

from scripts import ibm_calibration


def _stub_target() -> Target:
    target = Target(
        num_qubits=4,
        qubit_properties=[
            QubitProperties(t1=10e-6, t2=20e-6),
            QubitProperties(t1=30e-6, t2=40e-6),
            QubitProperties(t1=50e-6, t2=60e-6),
            QubitProperties(t1=999e-6, t2=999e-6),
        ],
    )
    target.add_instruction(
        CZGate(),
        {
            (0, 1): InstructionProperties(error=0.01),
            (1, 2): InstructionProperties(error=0.03),
        },
    )
    target.add_instruction(
        Measure(),
        {
            (0,): InstructionProperties(error=0.1),
            (1,): InstructionProperties(error=0.2),
            (2,): InstructionProperties(error=0.4),
            (3,): InstructionProperties(error=0.9),
        },
    )
    return target


def _stub_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4, 3)
    qc.cz(0, 1)
    qc.cz(0, 1)
    qc.cz(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def test_median_computation_on_synthetic_target() -> None:
    s = ibm_calibration.calibration_summary(_stub_circuit(), _stub_target())
    assert s["physical_qubits_used"] == [0, 1, 2]  # qubit 3 is untouched
    assert s["n_two_qubit_gate_instances"] == 3
    # instances 0.01, 0.01, 0.03 -> median 0.01 (couplings weighted by use)
    assert s["median_two_qubit_error"] == pytest.approx(0.01)
    assert s["median_t1_us"] == pytest.approx(30.0)
    assert s["median_t2_us"] == pytest.approx(40.0)
    assert s["median_readout_error"] == pytest.approx(0.2)
    assert s["notes"] == []


def test_median_of_empty_is_none() -> None:
    assert ibm_calibration._median([]) is None


def test_skip_path_when_no_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(ibm_calibration.hardware_run, "connect_service", lambda: None)
    out_file = tmp_path / "ibm_calibration.json"
    monkeypatch.setattr(ibm_calibration, "OUT_FILE", out_file)
    assert ibm_calibration.main() == 0
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["status"] == "skipped" and saved["reason"]
    assert "skipped" in capsys.readouterr().out


def test_skip_path_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Service:
        def backend(self, name: str) -> None:
            raise OSError("offline")

    monkeypatch.setattr(
        ibm_calibration.hardware_run, "connect_service", lambda: Service()
    )
    out = ibm_calibration.run()
    assert out["status"] == "skipped" and "OSError" in out["reason"]
