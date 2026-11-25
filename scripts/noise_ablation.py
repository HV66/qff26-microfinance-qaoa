"""Noise ablation (preregistered Step 5): noiseless, noisy, readout-mitigated, ZNE.

5 loans, seeds 0..4, p = 1, lambda 0.3, 4096 shots, fake_fez noise model
(AerSimulator.from_backend(fallback_backend()), as in noise_run.py). Angles come
from results/hardware_params_seed{S}_n5_p1.json when present, otherwise they are
optimised with qaoa_run.optimise (5 restarts, maxiter 200) and cached in
results/noise_ablation_params_seed{S}_n5_p1.json.

Zero-noise extrapolation is our OWN global unitary folding: the transpiled
(unmeasured) circuit U is replaced by U (U^dagger U)^k with k = (scale - 1) / 2
at scale factors 1, 3, 5, then measured on the noisy simulator. P(optimum) at
the three scales is fitted with a least-squares line and evaluated at scale 0.
The transpile for the ZNE circuits uses optimisation_level 1 (see ZNE_OPT_LEVEL)
so the folded gates are not optimised away; the folded circuit is built after
transpilation, so no later pass can cancel U^dagger U. This differs from the
level-3 circuit used for the noisy and readout-mitigated variants (as in
noise_run.py), which is recorded in the output.

Descriptive only: whatever it shows is reported, nothing is tuned.

Run:  python -m scripts.noise_ablation
Output: results/noise_ablation.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import qaoa_run
from baseline import brute_force
from data import make_portfolio
from hardware_run import (
    LAM,
    SHOTS,
    circuit_stats,
    fallback_backend,
    transpile_for,
)
from noise_run import (
    measured_physical_qubits,
    mitigate_counts,
    readout_calibration,
    run_sampler,
)
from qubo import Qubo, build_qubo

log = logging.getLogger(__name__)

RESULTS = Path("results")
OUT_FILE = RESULTS / "noise_ablation.json"
SEEDS = (0, 1, 2, 3, 4)
N_LOANS = 5
P = 1
SCALES = (1, 3, 5)
ZNE_OPT_LEVEL = 1
ZNE_TRANSPILE_SEED = 0


def fold_circuit(
    circuit: QuantumCircuit, scale: int, basis_gates: list[str] | None = None
) -> QuantumCircuit:
    """Global unitary folding: U -> U (U^dagger U)^((scale - 1) / 2).

    Args:
        circuit: Unitary (measurement-free) circuit U.
        scale: Odd positive noise scale factor.
        basis_gates: If given, U^dagger is translated into this gate set with
            optimisation_level 0 (no gate cancellation), e.g. sx^dagger has
            to become native gates for the Aer noise model.

    Raises:
        ValueError: If scale is not an odd positive integer.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd positive integer")
    folded = circuit.copy()
    inverse = circuit.inverse()
    if basis_gates is not None:
        inverse = transpile(inverse, basis_gates=basis_gates, optimization_level=0)
    for _ in range((scale - 1) // 2):
        folded.barrier()
        folded.compose(inverse, inplace=True)
        folded.barrier()
        folded.compose(circuit, inplace=True)
    return folded


def extrapolate_to_zero(
    scales: tuple[int, ...], values: tuple[float, ...]
) -> tuple[float, float]:
    """Least-squares line through (scale, value), evaluated at scale 0.

    Returns:
        (raw intercept, intercept clipped to [0, 1]).
    """
    slope, intercept = np.polyfit(np.asarray(scales, float), np.asarray(values), 1)
    del slope
    raw = float(intercept)
    return raw, float(min(1.0, max(0.0, raw)))


def loan_index(x: np.ndarray) -> int:
    """Basis-index bits of a loan vector (loan k is bit k)."""
    return int(sum(int(v) << k for k, v in enumerate(x)))


def p_optimum_from_probs(probs: np.ndarray, n_loans: int, opt_index: int) -> float:
    """Sum probability over all slack-bit values of the optimum loan vector."""
    idx = np.arange(len(probs))
    return float(probs[(idx & (2**n_loans - 1)) == opt_index].sum())


def load_angles(
    q: Qubo,
    n_loans: int,
    seed: int,
    results_dir: Path = RESULTS,
    hardware_dir: Path = RESULTS,
) -> tuple[np.ndarray, str]:
    """Return (angles, provenance): hardware cache, own cache, or optimise+cache."""
    hw = hardware_dir / f"hardware_params_seed{seed}_n{n_loans}_p{P}.json"
    own = results_dir / f"noise_ablation_params_seed{seed}_n{n_loans}_p{P}.json"
    for path in (hw, own):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return np.array(data["params"]), f"cached {path.name}"
    opt = qaoa_run.optimise(q, P, seed)
    results_dir.mkdir(parents=True, exist_ok=True)
    own.write_text(
        json.dumps({"params": opt["best_params"], "best_value": opt["best_value"]}),
        encoding="utf-8",
    )
    return np.array(opt["best_params"]), f"optimised now, cached {own.name}"


def transpile_unitary(
    circuit: QuantumCircuit, backend: Any
) -> tuple[QuantumCircuit, list[int]]:
    """Transpile without measurement; return (circuit, physical qubit per variable)."""
    tqc = transpile(
        circuit,
        backend=backend,
        optimization_level=ZNE_OPT_LEVEL,
        seed_transpiler=ZNE_TRANSPILE_SEED,
    )
    phys = [int(i) for i in tqc.layout.final_index_layout()]
    return tqc, phys


def measure_on(circuit: QuantumCircuit, phys: list[int]) -> QuantumCircuit:
    """Append a 'meas' register and measure variable k's qubit into bit k."""
    out = circuit.copy()
    out.add_register(ClassicalRegister(len(phys), "meas"))
    out.barrier()
    for k, qubit in enumerate(phys):
        out.measure(qubit, k)
    return out


def run_seed(
    seed: int,
    n_loans: int = N_LOANS,
    shots: int = SHOTS,
    results_dir: Path = RESULTS,
    hardware_dir: Path = RESULTS,
) -> dict[str, Any]:
    """Run all variants for one seed and return a JSON-ready dict."""
    pf = make_portfolio(n_loans, seed)
    q = build_qubo(pf, LAM)
    bf = brute_force(pf, LAM)
    angles, provenance = load_angles(q, n_loans, seed, results_dir, hardware_dir)
    circuit = qaoa_run.build_ansatz(q, P).assign_parameters(angles)
    fake = fallback_backend()
    ideal = AerSimulator(seed_simulator=seed)
    noisy = AerSimulator.from_backend(fake, seed_simulator=seed)

    def p_opt(counts: dict[str, int]) -> dict[str, Any]:
        return qaoa_run.sample_metrics(pf, q, counts, np.array(bf.x), bf.objective)

    t0 = time.perf_counter()
    tqc = transpile_for(circuit, fake)
    phys = measured_physical_qubits(tqc)
    c_ideal = run_sampler(ideal, tqc, shots, seed)
    c_noisy = run_sampler(noisy, tqc, shots, seed)
    mats = readout_calibration(noisy, phys, tqc.num_qubits, shots, seed)
    c_mit = mitigate_counts(c_noisy, mats, shots, seed)
    variants = {
        "noiseless": p_opt(c_ideal),
        "noisy": p_opt(c_noisy),
        "noisy_readout_mitigated": p_opt(c_mit),
    }

    unitary, uphys = transpile_unitary(circuit, fake)
    basis = sorted(unitary.count_ops().keys() - {"barrier"})
    scale_out: dict[str, Any] = {}
    for scale in SCALES:
        folded = fold_circuit(unitary, scale, basis)
        measured = measure_on(folded, uphys)
        counts = run_sampler(noisy, measured, shots, seed)
        stats = circuit_stats(measured)
        scale_out[str(scale)] = {
            "p_optimal": p_opt(counts)["p_optimal"],
            **stats,
        }
    values = tuple(scale_out[str(s)]["p_optimal"] for s in SCALES)
    raw, clipped = extrapolate_to_zero(SCALES, values)

    probs = qaoa_run.fast_probabilities(
        qaoa_run.scaled_cost_diagonal(q), q.n_vars, angles
    )
    ideal_p = p_optimum_from_probs(probs, n_loans, loan_index(bf.x))
    noiseless = variants["noiseless"]["p_optimal"]
    noisy_p = variants["noisy"]["p_optimal"]
    return {
        "seed": seed,
        "n_loans": n_loans,
        "n_qubits": q.n_vars,
        "p": P,
        "lam": LAM,
        "shots": shots,
        "angles": angles.tolist(),
        "angles_source": provenance,
        "exact_x": bf.x.tolist(),
        "ideal_statevector_p_optimum": ideal_p,
        "uniform_expectation": 1.0 / 2**n_loans,
        "variants": variants,
        "variants_transpile": {
            "optimization_level": 3,
            **circuit_stats(tqc),
        },
        "zne": {
            "method": "global unitary folding, U (U^dagger U)^k, k=(scale-1)/2",
            "transpile_optimization_level": ZNE_OPT_LEVEL,
            "transpile_note": (
                "ZNE circuits are transpiled WITHOUT measurement at level "
                f"{ZNE_OPT_LEVEL}, folded afterwards, then measured; the noisy "
                "and mitigated variants use the level-3 circuit of noise_run.py"
            ),
            "scales": scale_out,
            "p_optimum_raw_extrapolated": raw,
            "p_optimum_clipped": clipped,
            "raw_outside_unit_interval": not 0.0 <= raw <= 1.0,
        },
        "deltas": {
            "zne_raw_vs_noiseless": raw - noiseless,
            "zne_raw_vs_noisy": raw - noisy_p,
            "zne_clipped_vs_noiseless": clipped - noiseless,
            "zne_clipped_vs_noisy": clipped - noisy_p,
            "zne_scale1_vs_noisy_level3": values[0] - noisy_p,
        },
        "runtime_s": time.perf_counter() - t0,
    }


def main() -> None:
    """Run seeds 0..4 and write results/noise_ablation.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for seed in SEEDS:
        r = run_seed(seed)
        rows.append(r)
        log.info(
            "seed=%d ideal=%.4f noiseless=%.4f noisy=%.4f mit=%.4f "
            "zne_raw=%.4f zne_clip=%.4f t=%.0fs",
            seed,
            r["ideal_statevector_p_optimum"],
            r["variants"]["noiseless"]["p_optimal"],
            r["variants"]["noisy"]["p_optimal"],
            r["variants"]["noisy_readout_mitigated"]["p_optimal"],
            r["zne"]["p_optimum_raw_extrapolated"],
            r["zne"]["p_optimum_clipped"],
            r["runtime_s"],
        )
    out = {
        "descriptive_only": True,
        "backend_noise_model": fallback_backend().name,
        "scales": list(SCALES),
        "rows": rows,
    }
    OUT_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
