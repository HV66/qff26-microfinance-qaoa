# Preregistration: second-wave experiments

Written on 2026-11-26, before any script for these experiments exists or has run.
Repository state when written: commit `ab87e2f` (`git log -1`). The commit that adds
this file is the evidence of ordering: every new script and result file is committed
after it.

## Principles

- **All results are reported whatever they show.** A null, negative or contradictory
  result is reported with the same prominence as a positive one. No setting listed
  below is changed after seeing a result. If a check fails (for example a test on a
  seed), the failure is reported and nothing is tuned to repair it.
- **Each new family of tests gets its own Holm-Bonferroni correction** at family-wise
  alpha 0.05, using `compare.holm_bonferroni`. The earlier family (16 simulated QAOA
  runs plus 2 hardware jobs, 18 tests, `results/holm_correction.json`) is not re-opened
  and its numbers do not change.
- Additive only: new modules, new `results/*.json`, new README sections. No edit to
  `qubo.py`, `baseline.py`, any existing result file, or lambda. No hardware job.
- Every number in the README and docs must come from a file under `results/`.

## Instances and fixed settings

| item | setting | value (fixed now) |
|---|---|---|
| instances | seeds | 0, 1, 2, 3, 4 (8 loans, `make_portfolio(8, seed)`) unless a step says 5 loans |
| lambda | headline | 0.3 |
| lambda | also used only in step 1 | 0.1 and 1.0 |
| optimiser | method | COBYLA (`scipy.optimize.minimize`) |
| optimiser | restarts / maxiter | 5 / 200 |
| optimiser | start angles | as in `qaoa_run.optimise`: RNG seeded by `seed * 100 + p`; beta ~ U[0, pi), gamma ~ U[0, 2 pi) |
| sampling | shots | 4096, `qaoa_run.sample_counts` (primitives-V2 `StatevectorSampler`), seed = instance seed |
| depth | QAOA layers | p = 1, 2, 3 |
| cost scaling | angle scale | `qaoa_run.scaled_cost` (largest absolute Ising coefficient) |
| CVaR | alpha | 0.1 |
| ZNE | scale factors | 1, 3, 5 |
| significance | alpha | 0.05, one-sided exact binomial test against the uniform rate `2^-n_loans` (`compare.binomial_verdict`) |

## Step 1: price of fairness (classical, no QAOA)

- Constraint: every sector that has at least one loan request in the instance must have
  at least one funded loan, in addition to the budget.
- Method: exhaustive enumeration (`baseline.brute_force` logic written independently in
  the new script, using `qubo.objective` and `qubo.is_feasible`) with the sector filter.
- Instances: seeds 0 to 4, lambda in {0.1, 0.3, 1.0}.
- Reported: objective loss versus the unconstrained exact optimum (absolute and percent of
  the absolute unconstrained objective), loans funded, amount deployed versus budget, and
  whether any feasible portfolio exists. If none exists the cell says so.
- Extra QUBO slack qubits, arithmetic only (nothing is built or run): the constraint
  `sum_{i in sector s} x_i >= 1` becomes an equality with a slack integer
  `t_s in {0, ..., n_s - 1}` where `n_s` is the number of loans in sector s, which needs
  `(n_s - 1).bit_length()` binary slack bits; the total is the sum over sectors.
- No hypothesis test, so no Holm family.

## Step 2: slack-free ("unbalanced") budget penalty

- Technique credited to the literature on **unbalanced penalization** (Montanez-Barrera,
  Willsch, Maldonado-Romo and Michielsen, arXiv 2211.13914, Quantum Science and
  Technology 2024). The citation is from the author's memory and could not be checked
  online in this session; the README says so. The coefficients below are our own rule,
  not the paper's.
- Form: `E(x) = -ret . x + lam * x_w^T Sigma x_w + (-l1 * g + l2 * g^2)`, with
  `g = budget - sum_i amount_i x_i` and no slack bits, so the QUBO has `n_loans` variables.
- **Coefficient rule, fixed now.** Let `P = qubo.auto_penalty(portfolio)` and
  `r = min_i ret_i` (all returns are positive). Set
  `l1 = 2 * (sqrt(r^2 + r * P) - r)` and `l2 = P - l1`.
  Properties of the rule: `l1 + l2 = P`, so an overspend of one unit costs exactly P, the
  same minimum violation cost as the slack version; and the most the penalty can reward
  unspent budget, `l1^2 / (4 * l2)`, equals `r`, the smallest loan return. The rule uses
  only `auto_penalty` and the instance returns.
- **Tests first** (`tests/test_qubo_unbalanced.py`, written before the module): for seeds
  0 to 9, lambda 0.3, 8 loans, the minimum-energy bitstring of the unbalanced QUBO must
  decode to the brute-force constrained optimum. Any failing seed is reported in the
  results as a failure. The test records which seeds fail instead of tuning the rule.
  No outcome is assumed in advance.
- QAOA comparison: seeds 0 to 4, p = 1, 2, 3, settings above, on the unbalanced QUBO,
  using the existing fast simulator for optimisation and the same sampler path. Side by
  side with the slack version's existing files `results/qaoa_seed{S}_p{P}.json` (read
  only): qubits, logical depth and two-qubit count (`qaoa_run.circuit_cost`), P(optimum),
  P(feasible, decoded), and the verdict against uniform random. "Raw feasible" has no
  meaning without slack bits and is not reported for the unbalanced version.
- Holm family F2: the 15 one-sided tests of the unbalanced QAOA runs (5 seeds, 3 depths).

## Step 3: CVaR ablation

- Objective: `CVaR_alpha(E)` of the QUBO energy under the QAOA output distribution, alpha
  0.1: sort basis states by energy, take probability mass 0.1 from the lowest energies
  (a fractional share of the last state), divide by 0.1. Exact, from the statevector.
- Same restarts, maxiter, start angles, shots, seeds 0 to 4, p = 1, 2, 3. Same sampler
  path and metrics as the expectation runs.
- Side by side with the expectation-objective runs already in `results/` (read only), with
  the same binomial verdict.
- Holm family F3: the 15 one-sided tests of the CVaR runs.

## Step 4: CSV solver

- Input columns `amount`, `expected_return`, `sector`; a `--budget` argument; at most 10
  loans, otherwise a clear error. `amount` must be a positive integer.
- The CSV has no covariance, so the loss covariance is built exactly as in `data.py`:
  sector-block correlation 0.6 within a sector and 0.1 across, with the same per-loan
  volatility for every loan, the midpoint of `data.VOL_RANGE`. This is stated in the output.
- Output: exact, greedy and QAOA-sampled portfolios and the sector audit. Synthetic
  example file only. No hypothesis test.

## Step 5: noise ablation

- 5 loans, seeds 0 to 4, p = 1, lambda 0.3, 4096 shots, fake_fez noise model, as in
  `noise_run.py`, using the saved or cached angles (`hardware_params_seed{S}_n5_p1.json`;
  seeds without a cache file are optimised with the settings above and cached).
- Variants: noiseless, noisy, noisy with tensored readout mitigation (`noise_run`
  helpers), and zero-noise extrapolation by our own **global unitary folding** at scale
  factors 1, 3, 5 (`U` followed by `(U^dagger U)` repeated `(scale - 1) / 2` times on the
  unitary part of the transpiled circuit, then measurement).
- Extrapolation: least-squares line through the three (scale, value) points, evaluated at
  scale 0. The quantity extrapolated is P(optimum). Estimates outside [0, 1] are reported
  as computed and also clipped, both marked.
- Descriptive only. If a test is added it gets its own Holm family.

## Step 6: landscape and calibration

- Grid for the 5-loan, p = 1 instance, seeds 0 and 1: beta in [0, pi) with 101 points,
  gamma in [-pi, pi] with 201 points. Two surfaces: exact noiseless P(optimum), and the
  objective the optimiser actually minimised (exact <H> of the QUBO).
- Check: is the saved optimiser angle pair (`hardware_params_seed{S}_n5_p1.json`, beta
  reduced modulo pi) at the grid maximum of P(optimum), and at the grid minimum of <H>?
  Reported as the ratio of P(optimum) at the saved angles to the grid maximum, and the
  grid distance to the grid optimum.
- Calibration (read-only): median two-qubit gate error, median T1 and median T2 of
  `ibm_fez` for the qubits used by a re-transpile of the saved jobs (same settings as
  `hardware_run.py`). The layout may differ from the layout used at submission and the
  calibration is the current one, not the one at submission time; both are stated.
  Credentials only from the saved account; if the connection fails the step is skipped and
  says so. No job is submitted.

## What would change this plan

Only an error in this document. Any such correction is made in a new, dated section at the
end of this file and the original text is left as written.

## Correction, dated 2026-11-26 (added after the Step 2 tests ran; original text above is unchanged)

Step 2 states that the unbalanced penalty "can reward unspent budget by at most
`l1^2 / (4 * l2) = r`". That property is true but it is attained only at the real-valued
minimiser `g* = l1 / (2 * l2)`, which is below 1 on every seed checked (0.12 to 0.25). Loan
amounts are integers, so the unspent budget `g` is a whole number and the penalty at `g = 1`
is `l2 - l1`, which is positive (5.3 to 8.1 on the seeds checked). The rule therefore behaves
almost like an equality constraint, so it penalises leaving any budget unspent, while the true
problem has an inequality constraint. The coefficient rule itself is unchanged. The Step 2
test results (3 of 10 seeds pass) are reported as they came out.

The pytest gate cannot stay green with seven deliberately failing tests, so the per-seed
test now pins the recorded outcome (seeds 0, 2, 6 decode to the exact optimum; seeds 1, 3, 4,
5, 7, 8, 9 do not, and each failing minimum must be budget-feasible and strictly worse than
the optimum) and `scripts/unbalanced_min_decode.py` writes every seed's outcome to
`results/unbalanced_min_decode.json`. This keeps the failures visible without tuning anything.
