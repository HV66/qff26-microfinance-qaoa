"""Slack-free ("unbalanced") budget penalty for the microfinance loan QUBO.

The budget constraint sum_i a_i x_i <= B is handled without slack bits by a
penalty that is linear plus quadratic in the remaining budget
g = B - sum_i a_i x_i:

    E(x) = -ret.x + lam * w^T Sigma w + (-l1 * g + l2 * g^2),   w_i = a_i x_i

so the QUBO has n_loans variables (qubits) instead of n_loans + bit_length(B).

Credit: the technique "unbalanced penalization" is from Montanez-Barrera,
Willsch, Maldonado-Romo and Michielsen, arXiv:2211.13914, Quantum Science and
Technology 2024. This citation is from memory and was not verified online in
this session. The coefficient rule below (l1, l2) is our own preregistered rule
(docs/preregistration.md, Step 2), not the paper's.

Coefficient rule (fixed in advance, never tuned): with P = auto_penalty and
r = min_i ret_i,

    l1 = 2 * (sqrt(r^2 + r * P) - r),     l2 = P - l1.

Then l1 + l2 = P and l1^2 / (4 * l2) = r.
"""

from __future__ import annotations

import logging

import numpy as np

from data import LoanPortfolio
from qubo import Qubo, auto_penalty

log = logging.getLogger(__name__)


def unbalanced_coefficients(p: LoanPortfolio) -> tuple[float, float]:
    """Return the preregistered (l1, l2) for a portfolio.

    Args:
        p: Loan portfolio (all returns must be positive).

    Returns:
        (l1, l2) with l1 + l2 == auto_penalty(p) and l1^2 / (4 l2) == min(ret).

    Raises:
        ValueError: If any return is not positive.
    """
    r = float(p.ret.min())
    if r <= 0.0:
        raise ValueError(f"returns must be positive, got min {r}")
    big_p = auto_penalty(p)
    l1 = 2.0 * (float(np.sqrt(r * r + r * big_p)) - r)
    return l1, big_p - l1


def build_unbalanced_qubo(p: LoanPortfolio, lam: float = 0.3) -> Qubo:
    """Build the slack-free QUBO with the unbalanced budget penalty.

    Args:
        p: Loan portfolio.
        lam: Risk-aversion weight.

    Returns:
        A qubo.Qubo with n_vars == n_loans, an empty slack_coeffs array,
        penalty P = auto_penalty(p) and the given lam. Q is upper triangular.
    """
    n = p.n
    l1, l2 = unbalanced_coefficients(p)
    a = p.amount.astype(float)
    budget = float(p.budget)

    m_sym = (lam * p.sigma + l2) * np.outer(a, a)
    lin = -p.ret + l1 * a - 2.0 * l2 * budget * a
    offset = -l1 * budget + l2 * budget**2

    q = 2.0 * np.triu(m_sym, k=1)
    q[np.diag_indices(n)] = np.diag(m_sym) + lin
    log.debug("unbalanced QUBO: n=%d l1=%.4f l2=%.4f", n, l1, l2)
    return Qubo(
        Q=q,
        offset=float(offset),
        n_loans=n,
        slack_coeffs=np.zeros(0, dtype=int),
        penalty=auto_penalty(p),
        lam=lam,
    )
