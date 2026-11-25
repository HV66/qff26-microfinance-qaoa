"""QUBO formulation of microfinance loan selection.

Problem (maximise return minus risk, within budget):

    max  sum_i ret_i x_i  -  lam * w^T Sigma w      with w_i = amount_i x_i
    s.t. sum_i amount_i x_i <= budget,   x_i in {0, 1}

We minimise the negative objective. The inequality becomes an equality with
binary slack bits and a quadratic penalty:

    E(x, s) = -sum_i ret_i x_i
              + lam * sum_ij a_i a_j Sigma_ij x_i x_j
              + P * (sum_i a_i x_i + sum_k c_k s_k - budget)^2

Variable order is [x_0 .. x_{n-1}, s_0 .. s_{m-1}]. Bit k of a bitstring
index is variable k, and variable k maps to qubit k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from data import LoanPortfolio


@dataclass(frozen=True)
class Qubo:
    """Upper-triangular QUBO: E(z) = z^T Q z + offset, z in {0,1}^N.

    Attributes:
        Q: Upper-triangular matrix, shape (N, N). The diagonal holds linear
            terms (valid because z_i^2 = z_i for a bit).
        offset: Constant energy shift.
        n_loans: Number of loan variables n.
        slack_coeffs: Integer weight c_k of each slack bit, length m.
        penalty: Penalty weight P used.
        lam: Risk-aversion weight used.
    """

    Q: np.ndarray
    offset: float
    n_loans: int
    slack_coeffs: np.ndarray
    penalty: float
    lam: float

    @property
    def n_vars(self) -> int:
        """Total variables = loans + slack bits = qubits."""
        return int(self.Q.shape[0])

    def energy(self, z: np.ndarray) -> float:
        """Return E(z) for one bit vector z of length n_vars."""
        zf = np.asarray(z, dtype=float)
        return float(zf @ self.Q @ zf + self.offset)

    def all_energies(self) -> np.ndarray:
        """Return E for all 2^N bitstrings; bit k of the index is variable k."""
        idx = np.arange(2**self.n_vars)
        bits = ((idx[:, None] >> np.arange(self.n_vars)) & 1).astype(float)
        return np.einsum("bi,ij,bj->b", bits, self.Q, bits) + self.offset


def slack_coefficients(budget: int) -> np.ndarray:
    """Return bounded-binary slack weights that represent exactly 0..budget.

    Bit count m = ceil(log2(budget + 1)) = budget.bit_length(). Weights are
    1, 2, 4, ..., 2^(m-2), then a remainder so the weights sum to budget.
    Every integer 0..budget is reachable and nothing above budget is.

    Args:
        budget: Positive integer budget.

    Returns:
        Integer array of m weights.

    Raises:
        ValueError: If budget < 1.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    m = budget.bit_length()
    powers = [2**k for k in range(m - 1)]
    return np.array([*powers, budget - sum(powers)], dtype=int)


def auto_penalty(p: LoanPortfolio) -> float:
    """Choose penalty weight P = sum(ret) + 1 (requires lam >= 0).

    Objective g(x) = -ret.x + lam * risk(x). Proof that no penalised state
    can beat the best feasible state:

    1. risk(x) = w^T Sigma w >= 0 because Sigma is PSD, and lam >= 0, so
       g(x) >= -ret.x >= -sum(ret) for every bitstring x (ret > 0).
       This does not use the signs of the Sigma entries.
    2. x = 0 (no loans, slack = budget) is feasible with objective 0, so the
       feasible optimum is <= 0.
    3. Any constraint or slack violation has an integer residual
       r = spend + slack - budget != 0, because amounts, slack weights and
       budget are integers. Its penalty is P * r^2 >= P.
    4. So a penalised state has energy >= -sum(ret) + P = 1 > 0 >= the
       feasible optimum.

    P is not made larger than needed, because a very large P flattens the
    useful part of the energy landscape and hurts QAOA.
    """
    return float(p.ret.sum() + 1.0)


def build_qubo(
    p: LoanPortfolio, lam: float = 0.3, penalty: float | None = None
) -> Qubo:
    """Build the penalised QUBO for a loan portfolio.

    Args:
        p: Loan portfolio.
        lam: Risk-aversion weight (lambda).
        penalty: Penalty weight P. If None, chosen by auto_penalty.

    Returns:
        A Qubo with n + m variables, where m = bit_length(budget).
    """
    n = p.n
    c = slack_coefficients(p.budget)
    big_n = n + len(c)
    pen = auto_penalty(p) if penalty is None else float(penalty)

    # E = z^T M z + lin . z + offset, with M symmetric.
    m_sym = np.zeros((big_n, big_n))
    lin = np.zeros(big_n)

    # Term 1: -sum_i ret_i x_i. Reward for lending; linear in x.
    lin[:n] += -p.ret

    # Term 2: lam * sum_ij a_i a_j Sigma_ij x_i x_j. Sector-correlated risk of
    # the amount-weighted selection; the i == j part is linear for bits.
    m_sym[:n, :n] += lam * np.outer(p.amount, p.amount) * p.sigma

    # Term 3: P * (u.z - B)^2, with u = [a, c] (loan and slack weights) and
    # B = budget. Expands to P * (z^T u u^T z - 2 B u.z + B^2).
    u = np.concatenate([p.amount, c]).astype(float)
    m_sym += pen * np.outer(u, u)  # 3a: quadratic part of the penalty
    lin += -2.0 * pen * p.budget * u  # 3b: linear part of the penalty
    offset = pen * float(p.budget) ** 2  # 3c: constant part of the penalty

    # Fold into upper-triangular Q. Each off-diagonal pair M_ij + M_ji = 2 M_ij.
    # The diagonal absorbs the linear terms since z_i^2 = z_i.
    q = 2.0 * np.triu(m_sym, k=1)
    q[np.diag_indices(big_n)] = np.diag(m_sym) + lin
    return Qubo(Q=q, offset=offset, n_loans=n, slack_coeffs=c, penalty=pen, lam=lam)


def to_ising(q: Qubo) -> SparsePauliOp:
    """Convert a QUBO to an Ising Hamiltonian (own implementation).

    Substitute z_i = (1 - Z_i) / 2, where Z_i is the Pauli-Z on qubit i
    (eigenvalue +1 for bit 0, -1 for bit 1). Then

        z_i       = 1/2 - Z_i / 2
        z_i z_j   = 1/4 - Z_i / 4 - Z_j / 4 + Z_i Z_j / 4

    so for E = sum_i Q_ii z_i + sum_{i<j} Q_ij z_i z_j + offset:

        coeff(Z_i)     = -Q_ii / 2 - (1/4) * sum_{j != i} Q_ij
        coeff(Z_i Z_j) = Q_ij / 4
        coeff(I)       = offset + sum_i Q_ii / 2 + sum_{i<j} Q_ij / 4

    The constant is kept as an identity term, so the operator's diagonal
    equals the QUBO energy of every bitstring exactly.

    Args:
        q: The QUBO. Q must be upper triangular.

    Returns:
        SparsePauliOp on q.n_vars qubits. Qubit k is variable k, which is
        character k from the right in a Qiskit Pauli label.
    """
    big_n = q.n_vars
    upper = np.triu(q.Q, k=1)
    diag = np.diag(q.Q)
    z_coeff = -diag / 2.0 - (upper.sum(axis=0) + upper.sum(axis=1)) / 4.0
    identity = q.offset + diag.sum() / 2.0 + upper.sum() / 4.0

    terms: list[tuple[str, float]] = [("I" * big_n, float(identity))]
    for i in range(big_n):
        terms.append((_pauli_label(big_n, (i,)), float(z_coeff[i])))
    for i in range(big_n):
        for j in range(i + 1, big_n):
            if upper[i, j] != 0.0:
                terms.append((_pauli_label(big_n, (i, j)), float(upper[i, j] / 4.0)))
    return SparsePauliOp.from_list(terms).simplify()


def _pauli_label(n_qubits: int, z_qubits: tuple[int, ...]) -> str:
    """Return a Pauli label with Z on the given qubits and I elsewhere.

    Qiskit labels are little-endian: qubit k is character n_qubits - 1 - k.
    """
    chars = ["I"] * n_qubits
    for k in z_qubits:
        chars[n_qubits - 1 - k] = "Z"
    return "".join(chars)
