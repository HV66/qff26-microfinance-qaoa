"""QAOA for the microfinance QUBO: ansatz, optimisation, sampling, metrics.

Pipeline: QUBO -> Ising (own conversion) -> QAOAAnsatz (qiskit core) ->
COBYLA on the exact noiseless statevector expectation -> final evaluation by
sampling the optimised circuit with a primitives-V2 sampler -> metrics.

Why an exact statevector expectation for the optimiser: the cost operator is
diagonal, so <H> is exact and noise-free, with no shot noise in COBYLA's
objective. This isolates the QAOA algorithm from sampling noise. It is a
simulation-only shortcut: on hardware the expectation would be estimated from
shots. The final numbers (P(optimum), feasibility) come from SAMPLING, not
from the statevector.

Run:  python qaoa_run.py --seeds 0 --ps 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from baseline import brute_force, random_sampling
from data import LoanPortfolio, make_portfolio
from qubo import (
    Qubo,
    bits_from_string,
    build_qubo,
    decode_loans,
    is_feasible,
    objective,
    residual,
    to_ising,
)

log = logging.getLogger(__name__)

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
SHOTS = 4096
MAXITER = 200
RESTARTS = 5
LAM = 0.3
N_LOANS = 8
BASIS = ["cx", "rz", "sx", "x"]


def qubo_to_ising(q: Qubo) -> tuple[SparsePauliOp, float]:
    """Return (Ising operator without identity term, constant offset).

    Uses our own conversion (qubo.to_ising, z = (1 - Z) / 2) and moves the
    identity coefficient out as a separate constant, so that for every
    bitstring: <b|op|b> + offset == QUBO energy of b.
    """
    full = to_ising(q)
    has_z = full.paulis.z.any(axis=1)
    offset = float(full.coeffs[~has_z].real.sum())
    return SparsePauliOp(full.paulis[has_z], full.coeffs[has_z]), offset


def z_diagonal(op: SparsePauliOp) -> np.ndarray:
    """Return the diagonal of a Z/I-only operator without building matrices.

    A Z-string has eigenvalue (-1)^(sum of the bits it acts on) on bitstring b.
    This is exact and fast; op.to_matrix is slow and memory-heavy at 12+ qubits.
    tests/test_qaoa.py checks it against op.to_matrix on a small instance.

    Args:
        op: SparsePauliOp whose Paulis contain only I and Z.
    """
    n = op.num_qubits
    idx = np.arange(2**n)
    diag = np.zeros(2**n)
    for pauli, coeff in zip(op.paulis, op.coeffs):
        parity = np.zeros(2**n, dtype=np.int64)
        for k in np.flatnonzero(pauli.z):
            parity ^= (idx >> int(k)) & 1
        diag += coeff.real * (1 - 2 * parity)
    return diag


def energy_diagonal(q: Qubo) -> np.ndarray:
    """Return the Ising energy (op diagonal + offset) of every bitstring.

    Index bit k is variable k, which is qubit k.
    """
    op, offset = qubo_to_ising(q)
    return z_diagonal(op) + offset


def scaled_cost(q: Qubo) -> tuple[SparsePauliOp, float]:
    """Return the cost operator divided by its largest |coefficient|.

    A fixed, a-priori normalisation (not tuned) so that the angle gamma has a
    comparable meaning across instances. Returns (scaled operator, scale).
    """
    op, _ = qubo_to_ising(q)
    scale = float(np.abs(op.coeffs).max())
    return op / scale, scale


def build_ansatz(q: Qubo, p: int) -> QAOAAnsatz:
    """Build the depth-p QAOA ansatz (qiskit QAOAAnsatz, |+>^N start).

    It has 2p parameters: p mixer angles beta, then p cost angles gamma.
    """
    cost, _ = scaled_cost(q)
    return QAOAAnsatz(cost_operator=cost, reps=p)


def fast_probabilities(
    cost_diag: np.ndarray, n_qubits: int, params: np.ndarray
) -> np.ndarray:
    """Return QAOA output probabilities by direct numpy simulation.

    Same circuit as build_ansatz (|+>^N, then per layer exp(-i gamma H_c) and
    the RX(2 beta) mixer on every qubit), but without Qiskit's slow
    PauliEvolutionGate matrix exponential. tests/test_qaoa.py checks it
    against the Qiskit circuit. Parameter order: p betas, then p gammas.

    Args:
        cost_diag: Diagonal of the SCALED cost operator, shape (2^N,).
        n_qubits: Number of qubits N.
        params: Array of length 2p.
    """
    p = len(params) // 2
    beta, gamma = params[:p], params[p:]
    psi = np.full(2**n_qubits, 2 ** (-n_qubits / 2), dtype=complex)
    for layer in range(p):
        psi = psi * np.exp(-1j * gamma[layer] * cost_diag)
        tensor = psi.reshape([2] * n_qubits)
        cos_b, sin_b = np.cos(beta[layer]), -1j * np.sin(beta[layer])
        for axis in range(n_qubits):
            a, b = np.take(tensor, 0, axis=axis), np.take(tensor, 1, axis=axis)
            tensor = np.stack([cos_b * a + sin_b * b, sin_b * a + cos_b * b], axis=axis)
        psi = tensor.reshape(-1)
    return np.abs(psi) ** 2


def scaled_cost_diagonal(q: Qubo) -> np.ndarray:
    """Return the diagonal of the scaled cost operator used by build_ansatz."""
    cost, _ = scaled_cost(q)
    return z_diagonal(cost)


def optimise(
    q: Qubo, p: int, seed: int, restarts: int = RESTARTS, maxiter: int = MAXITER
) -> dict[str, Any]:
    """Run COBYLA from several seeded random starts; keep the best.

    Starts: beta ~ U[0, pi), gamma ~ U[0, 2 pi). The start RNG depends only on
    (seed, p), so reruns are identical.

    Returns:
        dict with best_params, best_value, histories (best-so-far per
        restart), n_evals and per-restart final values.
    """
    cost_diag = scaled_cost_diagonal(q)
    energies = energy_diagonal(q)
    rng = np.random.default_rng(seed * 100 + p)
    best: dict[str, Any] = {"best_value": np.inf}
    histories: list[list[float]] = []
    finals: list[float] = []
    n_evals = 0
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, 2 * np.pi, p)])
        trace: list[float] = []

        def fun(params: np.ndarray, trace: list[float] = trace) -> float:
            val = float(fast_probabilities(cost_diag, q.n_vars, params) @ energies)
            trace.append(min(val, trace[-1]) if trace else val)
            return val

        res = minimize(fun, x0, method="COBYLA", options={"maxiter": maxiter})
        histories.append(trace)
        finals.append(float(res.fun))
        n_evals += len(trace)
        if res.fun < best["best_value"]:
            best = {"best_params": res.x.tolist(), "best_value": float(res.fun)}
    return {
        **best,
        "histories": histories,
        "restart_values": finals,
        "n_evals": n_evals,
    }


def sample_counts(
    q: Qubo, p: int, params: np.ndarray, shots: int, seed: int
) -> dict[str, int]:
    """Sample the optimised circuit with a primitives-V2 sampler.

    Uses qiskit.primitives.StatevectorSampler (exact simulation, seeded
    sampling noise only). The circuit is first transpiled to basic gates
    because simulating the PauliEvolutionGate directly is very slow at 12
    qubits. Keys are Qiskit bitstrings (qubit 0 rightmost).
    """
    circuit = transpile(
        build_ansatz(q, p).assign_parameters(params),
        basis_gates=BASIS,
        optimization_level=1,
        seed_transpiler=seed,
    )
    circuit.measure_all()
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    result = sampler.run([circuit]).result()[0]
    return dict(result.data.meas.get_counts())


def circuit_cost(q: Qubo, p: int, seed: int) -> dict[str, int]:
    """Return transpiled depth and 2-qubit gate count on a generic basis.

    Basis {cx, rz, sx, x}, all-to-all connectivity (no coupling map),
    optimization_level 1. A real backend adds routing, so this is a lower
    bound on hardware depth.
    """
    tqc = transpile(
        build_ansatz(q, p),
        basis_gates=BASIS,
        optimization_level=1,
        seed_transpiler=seed,
    )
    ops = tqc.count_ops()
    return {"depth": int(tqc.depth()), "two_qubit_gates": int(ops.get("cx", 0))}


def gap_percent(value: float, optimum: float) -> float | None:
    """Return 100 * (value - optimum) / |optimum|, or None if undefined.

    When the optimum is exactly 0 the percentage is undefined, so a match
    returns 0.0 and anything else returns None. A suboptimal value is never
    reported as a 0% gap.
    """
    if abs(optimum) < 1e-12:
        return 0.0 if abs(value - optimum) < 1e-9 else None
    return 100.0 * (value - optimum) / abs(optimum)


def sample_metrics(
    pf: LoanPortfolio,
    q: Qubo,
    counts: dict[str, int],
    opt_x: np.ndarray,
    opt_obj: float,
) -> dict[str, Any]:
    """Compute sampling metrics with the SHARED decode helpers.

    P(optimum): decoded loan bits equal the brute-force optimal x.
    Feasible (raw): zero penalty on the full bitstring. Feasible (decoded):
    loan bits satisfy the budget. Gap: best feasible decoded sample vs the
    exact optimum, in percent of |optimum|.
    """
    total = sum(counts.values())
    p_opt = p_raw = p_dec = 0
    best_obj, best_x = None, None
    decoded: Counter[tuple[int, ...]] = Counter()
    for key, c in counts.items():
        bits = bits_from_string(q, key)
        x = decode_loans(q, bits)
        decoded[tuple(int(v) for v in x)] += c
        p_raw += c * int(residual(pf, q, bits) == 0)
        if is_feasible(pf, x):
            p_dec += c
            val = objective(pf, x, q.lam)
            if best_obj is None or val < best_obj:
                best_obj, best_x = val, x
        p_opt += c * int(np.array_equal(x, opt_x))
    gap = None
    if best_obj is not None:
        gap = gap_percent(best_obj, opt_obj)
    return {
        "p_optimal": p_opt / total,
        "p_feasible_raw": p_raw / total,
        "p_feasible_decoded": p_dec / total,
        "best_feasible_objective": best_obj,
        "best_feasible_x": None if best_x is None else best_x.tolist(),
        "best_feasible_gap_pct": gap,
        "distinct_loan_vectors": len(decoded),
        "shots": total,
    }


def run_one(
    seed: int, p: int, lam: float = LAM, n_loans: int = N_LOANS, shots: int = SHOTS
) -> dict[str, Any]:
    """Run QAOA at depth p on one instance and return all metrics."""
    pf = make_portfolio(n_loans, seed)
    q = build_qubo(pf, lam)
    bf = brute_force(pf, lam)
    energies = energy_diagonal(q)
    e_min, e_max = float(energies.min()), float(energies.max())

    t0 = time.perf_counter()
    opt = optimise(q, p, seed)
    opt_time = time.perf_counter() - t0
    params = np.array(opt["best_params"])
    t1 = time.perf_counter()
    counts = sample_counts(q, p, params, shots, seed)
    sample_time = time.perf_counter() - t1

    def ratio(e: float) -> float:
        return (e_max - e) / (e_max - e_min)

    null = random_sampling(pf, q, shots, seed)
    return {
        "config": {
            "seed": seed,
            "p": p,
            "lam": lam,
            "n_loans": n_loans,
            "shots": shots,
            "n_qubits": q.n_vars,
            "maxiter": MAXITER,
            "restarts": RESTARTS,
            "optimiser": "COBYLA",
            "penalty": q.penalty,
            "budget": pf.budget,
        },
        "exact": {
            "objective": bf.objective,
            "x": bf.x.tolist(),
            "qubo_min_energy": e_min,
        },
        "optimiser": {
            "best_value": opt["best_value"],
            "best_params": opt["best_params"],
            "restart_values": opt["restart_values"],
            "n_evals": opt["n_evals"],
            "approx_ratio": ratio(opt["best_value"]),
            "approx_ratio_uniform_state": ratio(float(energies.mean())),
            "histories": opt["histories"],
        },
        "sampling": sample_metrics(pf, q, counts, bf.x, bf.objective),
        "random_null": {
            "best_objective": null.best_objective,
            "best_gap_pct": (
                None
                if null.best_objective is None
                else 100.0 * (null.best_objective - bf.objective) / abs(bf.objective)
            ),
            "p_feasible_raw": null.p_feasible_raw,
            "p_feasible_decoded": null.p_feasible_decoded,
            "p_optimal": null.p_optimal,
            "shots": null.shots,
        },
        "circuit": circuit_cost(q, p, seed),
        "runtime_s": {
            "optimise": opt_time,
            "sample": sample_time,
            "total": opt_time + sample_time,
        },
    }


def plot_convergence(result: dict[str, Any], path: Path) -> None:
    """Save best-so-far <H> per COBYLA restart, with the exact optimum line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = result["config"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, h in enumerate(result["optimiser"]["histories"]):
        ax.plot(h, label=f"restart {i}", lw=1.2)
    ax.axhline(
        result["exact"]["qubo_min_energy"],
        color="k",
        ls="--",
        lw=1,
        label="exact QUBO minimum",
    )
    ax.set_yscale("symlog")
    ax.set_xlabel("objective evaluation")
    ax.set_ylabel("best-so-far <H> (symlog)")
    ax.set_title(f"QAOA convergence: seed {cfg['seed']}, p={cfg['p']}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save(result: dict[str, Any]) -> Path:
    """Write results/qaoa_seed{S}_p{P}.json and the convergence plot."""
    cfg = result["config"]
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"qaoa_seed{cfg['seed']}_p{cfg['p']}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    plot_convergence(
        result, PLOTS / f"qaoa_seed{cfg['seed']}_p{cfg['p']}_convergence.png"
    )
    return path


def main() -> None:
    """CLI: run QAOA for the given seeds and depths and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--ps", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for seed in args.seeds:
        for p in args.ps:
            r = save(run_one(seed, p))
            s = json.loads(r.read_text(encoding="utf-8"))
            log.info(
                "seed=%d p=%d ratio=%.4f P(opt)=%.4f feas_dec=%.3f gap=%s depth=%d t=%.1fs",
                seed,
                p,
                s["optimiser"]["approx_ratio"],
                s["sampling"]["p_optimal"],
                s["sampling"]["p_feasible_decoded"],
                s["sampling"]["best_feasible_gap_pct"],
                s["circuit"]["depth"],
                s["runtime_s"]["total"],
            )


if __name__ == "__main__":
    main()
