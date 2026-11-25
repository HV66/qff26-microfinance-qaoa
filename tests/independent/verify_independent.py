"""Independent re-verification of the microfinance QUBO / Ising math.

Reference values are computed here from the problem definition only, with
numpy. The project's objects (Q matrix, Ising operators, baselines) are only
the things under test. Run from anywhere:

    .venv\\Scripts\\python.exe tests/independent/verify_independent.py

Exit code 1 if any check fails.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline import brute_force, greedy
from data import LoanPortfolio, make_portfolio
from qaoa_run import qubo_to_ising
from qubo import Qubo, build_qubo, to_ising

LAMS = (0.0, 0.3, 1.0)
TOL = 1e-8
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    """Store one check result."""
    RESULTS.append((name, bool(ok), detail))


# ---------------------------------------------------------------- reference
def ref_slack(budget: int) -> np.ndarray:
    """Own slack weights 1,2,4,..., remainder, summing to budget."""
    m = int(budget).bit_length()
    pw = [1 << k for k in range(m - 1)]
    return np.array(pw + [budget - sum(pw)], dtype=np.int64)


def all_bits(n_vars: int) -> np.ndarray:
    """All 2^N bit rows; bit k of the row index is column k."""
    idx = np.arange(1 << n_vars, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n_vars)) & 1).astype(np.int64)


def ref_g(p: LoanPortfolio, x: np.ndarray, lam: float) -> np.ndarray:
    """g(x) = -ret.x + lam * w^T Sigma w, rows of x are selections."""
    w = x * p.amount[None, :]
    risk = np.einsum("ri,ij,rj->r", w, p.sigma, w)
    return -(x @ p.ret) + lam * risk


def ref_opt(p: LoanPortfolio, lam: float) -> tuple[float, np.ndarray]:
    """Constrained optimum over x in {0,1}^n, spend <= budget."""
    x = all_bits(p.n)
    feas = (x @ p.amount) <= p.budget
    g = np.where(feas, ref_g(p, x, lam), np.inf)
    return float(g.min()), g


def ref_energy(p: LoanPortfolio, lam: float):
    """Penalised energy for every (x, s) bitstring plus helper arrays."""
    c = ref_slack(p.budget)
    pen = float(p.ret.sum() + 1.0)
    bits = all_bits(p.n + len(c))
    x, s = bits[:, : p.n], bits[:, p.n :]
    res = x @ p.amount + s @ c - p.budget
    e = ref_g(p, x, lam) + pen * res.astype(float) ** 2
    return e, x, res, c, pen


def test_qubo_energy(q: Qubo) -> np.ndarray:
    """z^T Q z + offset evaluated over all bitstrings (matrix product)."""
    b = all_bits(q.n_vars).astype(float)
    return ((b @ q.Q) * b).sum(axis=1) + q.offset


def diag_of(op, offset: float = 0.0) -> np.ndarray:
    """Diagonal of a (diagonal) SparsePauliOp plus a constant."""
    return np.asarray(op.to_matrix(sparse=True).diagonal()).real + offset


# ------------------------------------------------------------------- checks
def check_core() -> None:
    """Checks 1 and 2: brute force, full QUBO spectrum, decoding, margin."""
    for lam in LAMS:
        fails: list[str] = []
        worst_diff = 0.0
        min_margin = np.inf
        n_inf_total = 0
        for seed in range(10):
            p = make_portfolio(8, seed)
            opt, _ = ref_opt(p, lam)
            e, x, res, c, pen = ref_energy(p, lam)
            q = build_qubo(p, lam=lam)
            # object under test vs mine
            if not np.array_equal(q.slack_coeffs, c):
                fails.append(f"s{seed}:slack")
            if abs(q.penalty - pen) > 1e-12:
                fails.append(f"s{seed}:penalty {q.penalty} vs {pen}")
            diff = float(np.abs(test_qubo_energy(q) - e).max())
            worst_diff = max(worst_diff, diff)
            if diff > TOL:
                fails.append(f"s{seed}:energy diff {diff:.2e}")
            # (b) minimum equals constrained optimum
            if abs(e.min() - opt) > TOL:
                fails.append(f"s{seed}:min {e.min()} != opt {opt}")
            # (c) all argmin states decode feasible with optimal objective
            am = np.flatnonzero(e <= e.min() + 1e-9)
            for k in am:
                xs = x[k : k + 1]
                feas = (xs @ p.amount)[0] <= p.budget
                g = ref_g(p, xs, lam)[0]
                if not feas or res[k] != 0 or abs(g - opt) > TOL:
                    fails.append(f"s{seed}:argmin {k} bad feas={feas} g={g}")
            # (d) infeasible never <= optimum
            infeas = ((x @ p.amount) > p.budget) | (res != 0)
            n_inf_total += int(infeas.sum())
            margin = float(e[infeas].min() - opt)
            min_margin = min(min_margin, margin)
            if margin <= 0 or np.any(e[infeas] <= opt):
                fails.append(f"s{seed}:infeasible reaches optimum, margin {margin}")
            # a feasible x with the right slack must exist for the optimum
            if int(((x @ p.amount) <= p.budget).sum()) == 0:
                fails.append(f"s{seed}:no feasible x")
        tag = f"lam={lam}"
        record(f"1 own brute-force opt seeds0-9 n=8 {tag}", True, "computed")
        record(
            f"2a QUBO energy == own energy {tag}",
            not fails
            or all(
                "energy" not in f and "slack" not in f and "penalty" not in f
                for f in fails
            ),
            f"max|diff|={worst_diff:.2e}",
        )
        record(
            f"2b/2c min==opt, argmin feasible+optimal {tag}",
            not any(("min" in f or "argmin" in f) for f in fails),
            "",
        )
        record(
            f"2d infeasible > opt {tag}",
            not any("infeasible" in f or "no feasible" in f for f in fails),
            f"min margin={min_margin:.4f} over {n_inf_total} infeasible states",
        )
        if fails:
            record(f"  details {tag}", False, "; ".join(fails[:5]))


def check_ising() -> None:
    """Check 3: both Ising conversions reproduce the independent energies."""
    for lam in (0.3, 1.0, 0.0):
        worst_a = worst_b = 0.0
        worst_rev = np.inf
        for seed in range(5):
            p = make_portfolio(8, seed)
            e, *_ = ref_energy(p, lam)
            q = build_qubo(p, lam=lam)
            da = diag_of(to_ising(q))
            op, off = qubo_to_ising(q)
            db = diag_of(op, off)
            worst_a = max(worst_a, float(np.abs(da - e).max()))
            worst_b = max(worst_b, float(np.abs(db - e).max()))
            # discriminating power: reversed qubit order must differ clearly
            nv = q.n_vars
            rev = np.array([int(format(i, f"0{nv}b")[::-1], 2) for i in range(1 << nv)])
            worst_rev = min(worst_rev, float(np.abs(e[rev] - e).max()))
        record(
            f"3 qubo.to_ising diag == own energy lam={lam}",
            worst_a < TOL,
            f"max|diff|={worst_a:.2e}",
        )
        record(
            f"3 qaoa_run.qubo_to_ising+off == own energy lam={lam}",
            worst_b < TOL,
            f"max|diff|={worst_b:.2e}",
        )
        record(
            f"3 reversed order would be detected lam={lam}",
            worst_rev > 1e-3,
            f"min over seeds of max|E-E_rev|={worst_rev:.3f}",
        )
    # hand-built asymmetric QUBO: E = 3 z0 + 5 z1 z2 - 2 z2 (N=3) + 0.5
    qm = np.zeros((3, 3))
    qm[0, 0], qm[1, 2], qm[2, 2] = 3.0, 5.0, -2.0
    hq = Qubo(
        Q=qm, offset=0.5, n_loans=1, slack_coeffs=np.array([1, 2]), penalty=1.0, lam=0.0
    )
    b = all_bits(3)
    expect = 3 * b[:, 0] + 5 * b[:, 1] * b[:, 2] - 2 * b[:, 2] + 0.5
    d1 = diag_of(to_ising(hq))
    o2, off2 = qubo_to_ising(hq)
    d2 = diag_of(o2, off2)
    ok = np.abs(d1 - expect).max() < TOL and np.abs(d2 - expect).max() < TOL
    rev = np.array([int(format(i, "03b")[::-1], 2) for i in range(8)])
    discr = np.abs(expect[rev] - expect).max() > 1.0
    record(
        "3 asymmetric hand QUBO, bit k == qubit k",
        ok and discr,
        f"d1={np.abs(d1 - expect).max():.1e} d2={np.abs(d2 - expect).max():.1e}",
    )


def check_data() -> None:
    """Check 4: instance properties."""
    bad: list[str] = []
    for n in range(2, 11):
        for seed in list(range(10)) + [123, 999]:
            p = make_portfolio(n, seed)
            p2 = make_portfolio(n, seed)
            tag = f"n{n}s{seed}"
            if not (
                np.array_equal(p.amount, p2.amount)
                and np.array_equal(p.ret, p2.ret)
                and np.array_equal(p.sigma, p2.sigma)
                and np.array_equal(p.sector, p2.sector)
                and p.budget == p2.budget
            ):
                bad.append(f"{tag}:nondeterministic")
            if not np.allclose(p.sigma, p.sigma.T, atol=1e-15):
                bad.append(f"{tag}:asym")
            if np.linalg.eigvalsh((p.sigma + p.sigma.T) / 2).min() < -1e-12:
                bad.append(f"{tag}:notPSD")
            if not (
                np.issubdtype(p.amount.dtype, np.integer)
                and p.amount.min() >= 1
                and p.amount.max() <= 8
            ):
                bad.append(f"{tag}:amount range")
            if p.budget != max(1, (2 * int(p.amount.sum()) + 2) // 5):
                bad.append(f"{tag}:budget {p.budget} vs sum {p.amount.sum()}")
            if not p.amount.sum() > p.budget:
                bad.append(f"{tag}:budget does not bind")
            vol = np.sqrt(np.diag(p.sigma))
            corr = p.sigma / np.outer(vol, vol)
            same = p.sector[:, None] == p.sector[None, :]
            off = ~np.eye(n, dtype=bool)
            if not np.allclose(corr[same & off], 0.6, atol=1e-9):
                bad.append(f"{tag}:same-sector corr")
            if not np.allclose(corr[~same], 0.1, atol=1e-9):
                bad.append(f"{tag}:cross-sector corr")
            if not (vol.min() >= 0.1 - 1e-12 and vol.max() <= 0.4 + 1e-12):
                bad.append(f"{tag}:vol range")
            rate = p.ret / p.amount
            if (
                not (rate.min() >= 0.10 - 1e-4 and rate.max() <= 0.35 + 1e-4)
                or p.ret.min() <= 0
            ):
                bad.append(f"{tag}:ret/amount range")
            if set(np.unique(p.sector)) - {0, 1, 2}:
                bad.append(f"{tag}:sector ids")
            if n >= 3 and len(np.unique(p.sector)) != 3:
                bad.append(f"{tag}:sector coverage")
    record(
        "4 data properties n=2..10, 12 seeds each",
        not bad,
        "; ".join(bad[:6]) or "108 instances ok",
    )
    try:
        make_portfolio(1, 0)
        make_portfolio(11, 0)
        record("4 n out of range rejected", False, "no ValueError")
    except ValueError:
        record("4 n out of range rejected", True, "")


def check_slack() -> None:
    """Check 5: slack encoding reachability, including edge budgets."""
    from qubo import slack_coefficients  # object under test

    bad: list[str] = []
    for b in list(range(1, 80)) + [127, 128, 129, 255, 256, 1000]:
        c = slack_coefficients(b)
        if not np.array_equal(c, ref_slack(b)):
            bad.append(f"B={b}:coeffs {c.tolist()} vs {ref_slack(b).tolist()}")
        if int(c.sum()) != b or len(c) != b.bit_length() or c.min() < 1:
            bad.append(f"B={b}:sum/len/positive")
        sums = all_bits(len(c)) @ c
        if set(sums.tolist()) != set(range(b + 1)):
            bad.append(f"B={b}:reachable set != 0..B")
    record(
        "5 slack reaches exactly 0..B (B=1..79,127..1000)",
        not bad,
        "; ".join(bad[:4]) or "ok",
    )
    # build_qubo with edge budgets via hand-built portfolios
    rng = np.random.default_rng(7)
    edge_bad: list[str] = []
    for budget in (1, 2, 3, 4, 7, 8, 15, 16, 17):
        for lam in (0.0, 0.5):
            n = 6
            amount = rng.integers(1, 9, size=n)
            ret = np.round(rng.uniform(0.1, 3.0, size=n), 4)
            v = rng.uniform(0.1, 0.4, size=n)
            sec = rng.integers(0, 3, size=n)
            same = (sec[:, None] == sec[None, :]).astype(float)
            corr = 0.5 * same + 0.1 + 0.4 * np.eye(n)
            p = LoanPortfolio(amount, ret, sec, np.outer(v, v) * corr, int(budget), 0)
            e, x, res, _c, _pen = ref_energy(p, lam)
            q = build_qubo(p, lam=lam)
            opt, _ = ref_opt(p, lam)
            d = np.abs(test_qubo_energy(q) - e).max()
            infeas = ((x @ p.amount) > p.budget) | (res != 0)
            if d > TOL or abs(e.min() - opt) > TOL or e[infeas].min() <= opt:
                edge_bad.append(
                    f"B={budget},lam={lam}: d={d:.1e} min={e.min()} opt={opt}"
                )
    record(
        "5 edge budgets 1..17 hand portfolios: energy/opt/penalty",
        not edge_bad,
        "; ".join(edge_bad[:3]) or "ok",
    )


def check_baselines() -> None:
    """Check 6: baseline.brute_force and baseline.greedy vs own solver."""
    bad: list[str] = []
    max_gap = 0.0
    for lam in LAMS:
        for seed in range(10):
            p = make_portfolio(8, seed)
            opt, _ = ref_opt(p, lam)
            bf = brute_force(p, lam)
            xb = np.asarray(bf.x)[None, :]
            if (
                abs(bf.objective - opt) > TOL
                or abs(ref_g(p, xb, lam)[0] - opt) > TOL
                or (xb @ p.amount)[0] > p.budget
            ):
                bad.append(f"bf lam={lam} s{seed}: {bf.objective} vs {opt}")
            gr = greedy(p, lam)
            xg = np.asarray(gr.x)[None, :]
            gval = ref_g(p, xg, lam)[0]
            if (xg @ p.amount)[0] > p.budget:
                bad.append(f"greedy infeasible lam={lam} s{seed}")
            if gval < opt - TOL:
                bad.append(f"greedy beats optimum lam={lam} s{seed}: {gval} < {opt}")
            if abs(gr.objective - gval) > TOL:
                bad.append(f"greedy reported obj mismatch lam={lam} s{seed}")
            max_gap = max(max_gap, gval - opt)
    record(
        "6 baseline.brute_force == own opt; greedy feasible & >= opt",
        not bad,
        f"max greedy gap={max_gap:.4f}; " + ("; ".join(bad[:4]) or "ok"),
    )


def check_stress() -> None:
    """Extra adversarial sweep: other n, seeds and lambdas (incl. large)."""
    bad: list[str] = []
    cnt = 0
    for n in (2, 3, 5, 9, 10):
        for seed in (100, 101, 102):
            for lam in (0.0, 0.3, 1.0, 5.0, 50.0):
                p = make_portfolio(n, seed)
                e, x, res, _c, _pen = ref_energy(p, lam)
                q = build_qubo(p, lam=lam)
                opt, _ = ref_opt(p, lam)
                d = float(np.abs(test_qubo_energy(q) - e).max())
                infeas = ((x @ p.amount) > p.budget) | (res != 0)
                cnt += 1
                if d > 1e-7 or abs(e.min() - opt) > 1e-7 or e[infeas].min() <= opt:
                    bad.append(
                        f"n{n}s{seed}lam{lam}: d={d:.1e} min-opt={e.min() - opt:.1e}"
                    )
    record(
        f"X stress n in {{2,3,5,9,10}}, lam up to 50 ({cnt} inst)",
        not bad,
        "; ".join(bad[:4]) or "ok",
    )


def main() -> int:
    """Run all checks, print table, return exit code."""
    t0 = time.perf_counter()
    for fn in (
        check_core,
        check_ising,
        check_data,
        check_slack,
        check_baselines,
        check_stress,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report as failure, keep going
            record(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc}")
    width = max(len(n) for n, _, _ in RESULTS)
    print(f"{'CHECK'.ljust(width)}  RESULT  DETAIL")
    for name, ok, detail in RESULTS:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}    {detail}")
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(
        f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed, {n_fail} failed, {time.perf_counter() - t0:.1f}s"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
