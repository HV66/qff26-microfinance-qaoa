Note: written before the metric rename approx_ratio -> normalised_score.

# Independent review: `qaoa_run.py` and `tests/test_qaoa.py`

Reviewed read-only. No source or test files were changed. Numerical checks used a temporary scratch script outside the repository.

## Findings (ranked by severity)

### [P2] `P(optimum)` excludes tied optimal loan vectors

**`qaoa_run.py:266`** — `p_opt` is incremented only when the decoded loan vector equals the single `opt_x` supplied by the caller. `brute_force` returns one argmin, so if another distinct feasible loan vector has the same minimum objective, its shots are omitted from `p_optimal`. This makes the metric lower than the probability of sampling *any* optimum (the usual meaning of `P(optimum)`). The documented portfolio seeds 0–4 each have a unique optimal loan vector, so I found no effect on those runs. Either count against the full set of optimal vectors or label the metric as probability of the selected/reference optimum.

### [P3] README describes a demo sampling path that the code does not use

**`README.md:173`** — The claim that the live demo samples with a NumPy helper rather than `StatevectorSampler` does not match the implementation. `demo.py:32,119–124` calls `qaoa_run.sample_counts`; `qaoa_run.py:216–218` samples with `StatevectorSampler`. The reduced optimizer/shots apply to `--quick`; default mode loads saved parameters and uses the saved run's shot count (`demo.py:201–228`). Update this README statement to distinguish the modes and remove the NumPy-sampler claim.

### [P3] Zero-optimum best-feasible gap is reported as zero even when the sample is suboptimal

**`qaoa_run.py:267–269`** — When `opt_obj == 0`, `best_feasible_gap_pct` is forced to `0.0`, irrespective of `best_obj`. For a valid instance whose feasible optimum is zero but whose best sampled feasible objective is positive, this falsely reports a zero gap (the percentage is undefined with the current denominator). The checked default seeds have negative optima, so this is an edge case rather than an observed result defect; consider returning `None` or defining a separate zero-safe normalization.

## Checks / no findings

- **QUBO → Ising:** The `z=(1-Z)/2` signs and identity offset extraction in `qubo.py:156–194` / `qaoa_run.py:59–69` are consistent. An independent scratch enumeration checked all 256 bitstrings of a 4-loan instance: `op` diagonal + returned offset matched QUBO energies with maximum absolute error `2.73e-12`. The existing test also checks all bitstrings for five 8-loan instances (`tests/test_qaoa.py:24–31`).
- **Fast simulator vs Qiskit:** For p=1,2,3,4, direct statevector probabilities from `build_ansatz(...).assign_parameters(params)` matched `fast_probabilities` for nontrivial random parameters; maximum absolute probability errors were respectively `5.72e-17`, `6.01e-16`, `1.62e-15`, and `1.56e-15`. This validates the positional parameter order (betas then gammas), little-endian diagonal indexing, default `|+>` start, global-X mixer `RX(2 beta)`, and scaled cost evolution. The Ising identity is removed from the scaled cost, which changes only a global phase. Note: the committed direct-equivalence regression test only parameterizes p=1,2 (`tests/test_qaoa.py:102–111`); extending it to p=3,4 would preserve this coverage.
- **Metrics:** `p_feasible_raw` tests zero full-bitstring residual; `p_feasible_decoded` tests the budget on decoded loans; both divide counts by total shots. The best-feasible gap sign is correct for a minimization problem when the exact optimum is the denominator reference: sampled feasible objective minus exact optimum is nonnegative. `approx_ratio` at `qaoa_run.py:300–301` is a spectrum-normalized score `(Emax - E)/(Emax - Emin)` (1 at minimum, 0 at maximum), not the conventional multiplicative approximation ratio; documentation/consumers should interpret it as a normalized score.
- **Qiskit 2.x APIs:** No use was found in the reviewed files of an API marked deprecated in the pinned Qiskit 2.6.2 dependency. The imports/usages (`QAOAAnsatz`, `StatevectorSampler`, `SparsePauliOp`, `Statevector`, and `transpile`) are current APIs.
- Apart from the demo-sampling statement above, I found no README claim in the reviewed QAOA flow that is unsupported by the inspected code.
