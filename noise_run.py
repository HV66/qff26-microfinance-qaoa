"""Noise-model simulation of the QAOA circuit (offline fallback for hardware).

Runs the SAME optimised circuit three ways, all with fixed seeds:
  (a) noiseless   : ideal Aer simulation of the transpiled circuit
  (b) noisy       : Aer noise model built from fake_fez (156-qubit Heron-style
                    fake backend) via AerSimulator.from_backend
  (c) noisy + readout mitigation : the noisy counts corrected with a tensored
                    (per-qubit) readout calibration matrix that is measured in
                    the same noisy simulator on the same physical qubits.

What was actually applied in (c): ONLY readout-error mitigation (per-qubit
2x2 confusion matrices, inverted, negative quasi-probabilities clipped to 0
and renormalised, then resampled with a fixed seed to counts). No dynamical
decoupling, twirling or ZNE is applied here; the Aer noise model contains no
idle-time dephasing for DD to cancel, so a DD pass would be meaningless in
this simulator. The tensored model ignores correlated readout errors.

Sizes: the full instance is 8 loans = 12 qubits. If it does not finish within
the time budget, run with --n-loans 5 (9 qubits); the choice used is stored in
the saved JSON ("instance").

Run:  python noise_run.py [--n-loans 8] [--p 1] [--shots 4096] [--seed 0]
Output: results/noise_run_p{P}.json (8 loans) or noise_run_n{N}_p{P}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

from baseline import brute_force
from data import make_portfolio
from hardware_run import (
    LAM,
    SHOTS,
    circuit_stats,
    fallback_backend,
    get_params,
    transpile_for,
)
from qaoa_run import build_ansatz, sample_metrics
from qubo import build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")


def run_sampler(
    sim: AerSimulator, circuit: QuantumCircuit, shots: int, seed: int
) -> dict[str, int]:
    """Sample a measured circuit with the Aer primitives-V2 sampler."""
    sampler = AerSamplerV2.from_backend(sim, default_shots=shots, seed=seed)
    result = sampler.run([circuit]).result()[0]
    return dict(result.data.meas.get_counts())


def measured_physical_qubits(tqc: QuantumCircuit) -> list[int]:
    """Return, for each classical bit k, the physical qubit measured into it."""
    mapping: dict[int, int] = {}
    for inst in tqc.data:
        if inst.operation.name == "measure":
            q_idx = tqc.find_bit(inst.qubits[0]).index
            c_idx = tqc.find_bit(inst.clbits[0]).index
            mapping[c_idx] = q_idx
    return [mapping[k] for k in range(len(mapping))]


def readout_calibration(
    sim: AerSimulator, phys: list[int], n_phys: int, shots: int, seed: int
) -> list[np.ndarray]:
    """Measure a 2x2 confusion matrix A_k[meas, prep] per measured qubit.

    Two circuits (all prepared 0, all prepared 1) on the same physical qubits
    are enough for a tensored model. Returns one matrix per classical bit.
    """
    n = len(phys)
    p_flip = np.zeros((2, n))  # p_flip[prep, k] = P(measure 1 | prepared prep)
    for prep in (0, 1):
        qc = QuantumCircuit(n_phys, n)
        if prep:
            for q in phys:
                qc.x(q)
        for k, q in enumerate(phys):
            qc.measure(q, k)
        sampler = AerSamplerV2.from_backend(sim, default_shots=shots, seed=seed + prep)
        counts = sampler.run([qc]).result()[0].data.c.get_counts()
        for key, c in counts.items():
            for k in range(n):
                p_flip[prep, k] += c * int(key[-1 - k])
        p_flip[prep] /= shots
    mats = []
    for k in range(n):
        p01, p11 = p_flip[0, k], p_flip[1, k]
        mats.append(np.array([[1 - p01, 1 - p11], [p01, p11]]))
    return mats


def mitigate_counts(
    counts: dict[str, int], mats: list[np.ndarray], shots: int, seed: int
) -> dict[str, int]:
    """Apply the inverse tensored confusion matrix and resample to counts.

    Quasi-probabilities below zero are clipped and the vector renormalised.
    """
    n = len(mats)
    probs = np.zeros(2**n)
    for key, c in counts.items():
        probs[int(key, 2)] += c
    probs /= probs.sum()
    tensor = probs.reshape((2,) * n)
    for k, a in enumerate(mats):
        axis = n - 1 - k  # array axis 0 is the most significant bit
        tensor = np.moveaxis(
            np.tensordot(np.linalg.inv(a), tensor, axes=([1], [axis])), 0, axis
        )
    quasi = np.clip(tensor.reshape(-1), 0.0, None)
    quasi /= quasi.sum()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(shots, quasi)
    return {format(i, f"0{n}b"): int(c) for i, c in enumerate(draws) if c}


def run_variants(
    n_loans: int, shots: int, seed: int, p: int | None = None
) -> dict[str, Any]:
    """Run noiseless, noisy and noisy+readout-mitigated variants."""
    pf = make_portfolio(n_loans, seed)
    q = build_qubo(pf, LAM)
    bf = brute_force(pf, LAM)
    p, params, provenance = get_params(q, n_loans, seed, p=p)
    circuit = build_ansatz(q, p).assign_parameters(params)
    fake = fallback_backend()
    tqc = transpile_for(circuit, fake)
    stats = circuit_stats(tqc)
    phys = measured_physical_qubits(tqc)

    ideal = AerSimulator(seed_simulator=seed)
    noisy = AerSimulator.from_backend(fake, seed_simulator=seed)

    def score(counts: dict[str, int]) -> dict[str, Any]:
        return sample_metrics(pf, q, counts, np.array(bf.x), bf.objective)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    c_ideal = run_sampler(ideal, tqc, shots, seed)
    timings["noiseless_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    c_noisy = run_sampler(noisy, tqc, shots, seed)
    timings["noisy_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mats = readout_calibration(noisy, phys, tqc.num_qubits, shots, seed)
    c_mit = mitigate_counts(c_noisy, mats, shots, seed)
    timings["readout_mitigation_s"] = time.perf_counter() - t0

    return {
        "instance": {
            "seed": seed,
            "n_loans": n_loans,
            "n_qubits": q.n_vars,
            "lam": LAM,
            "p": p,
            "params_source": provenance,
            "exact_objective": bf.objective,
            "exact_x": bf.x.tolist(),
        },
        "backend_noise_model": fake.name,
        "shots": shots,
        "transpiled": stats,
        "mitigation_applied": {
            "noiseless": "none",
            "noisy": "none",
            "noisy_readout_mitigated": "tensored readout calibration only",
        },
        "variants": {
            "noiseless": score(c_ideal),
            "noisy": score(c_noisy),
            "noisy_readout_mitigated": score(c_mit),
        },
        "readout_confusion_p01_p10": [[float(a[1, 0]), float(a[0, 1])] for a in mats],
        "timings_s": timings,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; writes results/noise_run_p{P}.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-loans", type=int, default=8)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=None, help="QAOA depth")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = run_variants(args.n_loans, args.shots, args.seed, args.p)
    inst = out["instance"]
    default_name = (
        f"noise_run_p{inst['p']}.json"
        if inst["n_loans"] == 8
        else f"noise_run_n{inst['n_loans']}_p{inst['p']}.json"
    )
    out_path = args.out if args.out is not None else RESULTS / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved {out_path}")
    print(
        f"n_loans={args.n_loans} qubits={out['instance']['n_qubits']} "
        f"p={out['instance']['p']} depth={out['transpiled']['depth']} "
        f"2q={out['transpiled']['two_qubit_gates']}"
    )
    for name, m in out["variants"].items():
        print(
            f"{name:26s} P(opt)={m['p_optimal']:.4f} "
            f"P(feas dec)={m['p_feasible_decoded']:.4f} "
            f"gap%={m['best_feasible_gap_pct']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
