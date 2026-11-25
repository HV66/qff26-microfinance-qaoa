"""Seeded synthetic microfinance loan data with sector-correlated risk.

All data is synthetic. No real borrower, customer or confidential data is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Loan amounts are small integers ("units", e.g. 1 unit = 100 currency units).
# Integer amounts keep the budget-slack encoding in qubo.py exact and small.
MIN_UNITS = 1
MAX_UNITS = 8
# Interest rate (return per unit lent) range.
RATE_RANGE = (0.10, 0.35)
# Per-unit loss volatility range (std-dev of loss per unit lent).
VOL_RANGE = (0.10, 0.40)
# Budget as a fraction of the total requested amount.
BUDGET_FRACTION = 0.40
# Number of economic sectors (e.g. farming, retail, services).
N_SECTORS = 3
# Correlation of loan losses: same sector vs different sector.
RHO_SAME = 0.60
RHO_CROSS = 0.10


@dataclass(frozen=True)
class LoanPortfolio:
    """A set of candidate loans plus a lending budget.

    Attributes:
        amount: Requested amount per loan, integer units, shape (n,).
        ret: Expected return per loan if funded, shape (n,).
        sector: Sector id (0..2) per loan, shape (n,).
        sigma: Covariance of loss per UNIT lent, shape (n, n), symmetric PSD.
        budget: Total amount available, integer units.
        seed: Seed that generated this portfolio.
    """

    amount: np.ndarray
    ret: np.ndarray
    sector: np.ndarray
    sigma: np.ndarray
    budget: int
    seed: int

    @property
    def n(self) -> int:
        """Number of candidate loans."""
        return int(self.amount.shape[0])

    def portfolio_risk(self, x: np.ndarray) -> float:
        """Return x^T Sigma x with x scaled by amount.

        Args:
            x: Binary selection vector, shape (n,).

        Returns:
            w^T Sigma w where w_i = amount_i * x_i, so a loan's risk
            contribution grows with its amount (quadratically).
        """
        w = self.amount * np.asarray(x)
        return float(w @ self.sigma @ w)


def _build_sigma(sector: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Build a symmetric PSD covariance with sector-block correlation.

    Correlation C = (RHO_SAME - RHO_CROSS) * B + RHO_CROSS * J
    + (1 - RHO_SAME) * I, where B[i, j] = 1 if same sector (diagonal
    included), J is all-ones and I is identity. B, J and I are PSD and the
    coefficients are positive, so C is PSD by construction (diagonal is 1).
    Sigma = diag(vol) C diag(vol) stays PSD.
    """
    same = (sector[:, None] == sector[None, :]).astype(float)
    corr = (
        (RHO_SAME - RHO_CROSS) * same
        + RHO_CROSS * np.ones_like(same)
        + (1.0 - RHO_SAME) * np.eye(len(sector))
    )
    sigma = np.outer(vol, vol) * corr
    return (sigma + sigma.T) / 2.0


def make_portfolio(n_loans: int = 8, seed: int = 0) -> LoanPortfolio:
    """Generate a reproducible synthetic loan portfolio.

    Args:
        n_loans: Number of candidate loans, must be 2..10 (qubit budget).
        seed: RNG seed. Same seed gives identical output.

    Returns:
        A LoanPortfolio. ret = amount * rate. Sectors are assigned so every
        sector appears when n_loans >= 3.

    Raises:
        ValueError: If n_loans is outside 2..10.
    """
    if not 2 <= n_loans <= 10:
        raise ValueError(f"n_loans must be in 2..10, got {n_loans}")
    rng = np.random.default_rng(seed)
    amount = rng.integers(MIN_UNITS, MAX_UNITS + 1, size=n_loans).astype(int)
    rate = rng.uniform(*RATE_RANGE, size=n_loans)
    vol = rng.uniform(*VOL_RANGE, size=n_loans)
    sector = rng.permutation(np.arange(n_loans) % N_SECTORS).astype(int)
    budget = max(1, round(BUDGET_FRACTION * amount.sum()))
    return LoanPortfolio(
        amount=amount,
        ret=np.round(amount * rate, 4),
        sector=sector,
        sigma=_build_sigma(sector, vol),
        budget=budget,
        seed=seed,
    )


if __name__ == "__main__":
    p = make_portfolio()
    print(f"seed={p.seed} n={p.n} budget={p.budget} total_requested={p.amount.sum()}")
    for i in range(p.n):
        print(
            f"loan {i}: sector={p.sector[i]} amount={p.amount[i]} "
            f"ret={p.ret[i]:.4f} vol={np.sqrt(p.sigma[i, i]):.3f}"
        )
    print(f"risk if all funded: {p.portfolio_risk(np.ones(p.n)):.3f}")
