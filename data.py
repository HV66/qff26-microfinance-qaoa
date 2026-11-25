"""Seeded synthetic microfinance loan data.

All data is synthetic. No real borrower, customer or confidential data is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Loan amounts are small integers ("units", e.g. 1 unit = 100 currency units).
# Integer amounts keep the budget-slack encoding in qubo.py exact and small.
MIN_UNITS = 1
MAX_UNITS = 8
# Interest rate (return per unit lent) and default probability ranges.
RATE_RANGE = (0.10, 0.35)
PD_RANGE = (0.02, 0.30)
# Budget as a fraction of the total requested amount.
BUDGET_FRACTION = 0.40


@dataclass(frozen=True)
class LoanPortfolio:
    """A set of candidate loans plus a lending budget.

    Attributes:
        amount: Requested amount per loan, integer units, shape (n,).
        ret: Expected return per loan if funded, shape (n,).
        risk: Risk penalty per loan if funded (expected loss), shape (n,).
        budget: Total amount available, integer units.
        seed: Seed that generated this portfolio.
    """

    amount: np.ndarray
    ret: np.ndarray
    risk: np.ndarray
    budget: int
    seed: int

    @property
    def n(self) -> int:
        """Number of candidate loans."""
        return int(self.amount.shape[0])


def make_portfolio(n_loans: int = 8, seed: int = 0) -> LoanPortfolio:
    """Generate a reproducible synthetic loan portfolio.

    Args:
        n_loans: Number of candidate loans, must be 2..10 (qubit budget).
        seed: RNG seed. Same seed gives identical output.

    Returns:
        A LoanPortfolio. ret = amount * rate; risk = amount * default
        probability (expected loss). Both scale with amount, so the budget
        constraint genuinely binds.

    Raises:
        ValueError: If n_loans is outside 2..10.
    """
    if not 2 <= n_loans <= 10:
        raise ValueError(f"n_loans must be in 2..10, got {n_loans}")
    rng = np.random.default_rng(seed)
    amount = rng.integers(MIN_UNITS, MAX_UNITS + 1, size=n_loans)
    rate = rng.uniform(*RATE_RANGE, size=n_loans)
    pd_ = rng.uniform(*PD_RANGE, size=n_loans)
    budget = max(1, round(BUDGET_FRACTION * amount.sum()))
    return LoanPortfolio(
        amount=amount.astype(int),
        ret=np.round(amount * rate, 4),
        risk=np.round(amount * pd_, 4),
        budget=budget,
        seed=seed,
    )


if __name__ == "__main__":
    p = make_portfolio()
    print(f"seed={p.seed} n={p.n} budget={p.budget} total_requested={p.amount.sum()}")
    for i in range(p.n):
        print(f"loan {i}: amount={p.amount[i]} ret={p.ret[i]:.4f} risk={p.risk[i]:.4f}")
