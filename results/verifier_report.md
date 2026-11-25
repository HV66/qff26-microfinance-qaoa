# Independent verifier report (Subagent C)

Command (from repo root): `.venv\Scripts\python.exe tests/independent/verify_independent.py`
Date: Wed 2026-11-25 (IST), run started 14:33. Exit code: 0. Wall time: about 7 s including imports (script reports 1.2 s of checks).

Method: reference values (constrained optimum, penalised energy of all 2^(n+m) bitstrings, slack weights, penalty P = sum(ret)+1) are recomputed in the script with numpy from the problem definition. Only `make_portfolio` and the objects under test (`build_qubo` output, `to_ising`, `qaoa_run.qubo_to_ising`, `slack_coefficients`, `baseline.brute_force`, `baseline.greedy`) are imported. `objective`, `is_feasible`, decode helpers and `Qubo.all_energies` are not used.

## Real output

```
CHECK                                                        RESULT  DETAIL
1 own brute-force opt seeds0-9 n=8 lam=0.0                   PASS    computed
2a QUBO energy == own energy lam=0.0                         PASS    max|diff|=7.28e-12
2b/2c min==opt, argmin feasible+optimal lam=0.0              PASS    
2d infeasible > opt lam=0.0                                  PASS    min margin=6.4301 over 60226 infeasible states
1 own brute-force opt seeds0-9 n=8 lam=0.3                   PASS    computed
2a QUBO energy == own energy lam=0.3                         PASS    max|diff|=1.09e-11
2b/2c min==opt, argmin feasible+optimal lam=0.3              PASS    
2d infeasible > opt lam=0.3                                  PASS    min margin=6.7891 over 60226 infeasible states
1 own brute-force opt seeds0-9 n=8 lam=1.0                   PASS    computed
2a QUBO energy == own energy lam=1.0                         PASS    max|diff|=1.09e-11
2b/2c min==opt, argmin feasible+optimal lam=1.0              PASS    
2d infeasible > opt lam=1.0                                  PASS    min margin=6.7891 over 60226 infeasible states
3 qubo.to_ising diag == own energy lam=0.3                   PASS    max|diff|=4.37e-11
3 qaoa_run.qubo_to_ising+off == own energy lam=0.3           PASS    max|diff|=4.37e-11
3 reversed order would be detected lam=0.3                   PASS    min over seeds of max|E-E_rev|=2420.806
3 qubo.to_ising diag == own energy lam=1.0                   PASS    max|diff|=3.64e-11
3 qaoa_run.qubo_to_ising+off == own energy lam=1.0           PASS    max|diff|=3.64e-11
3 reversed order would be detected lam=1.0                   PASS    min over seeds of max|E-E_rev|=2427.470
3 qubo.to_ising diag == own energy lam=0.0                   PASS    max|diff|=4.73e-11
3 qaoa_run.qubo_to_ising+off == own energy lam=0.0           PASS    max|diff|=4.37e-11
3 reversed order would be detected lam=0.0                   PASS    min over seeds of max|E-E_rev|=2415.456
3 asymmetric hand QUBO, bit k == qubit k                     PASS    d1=0.0e+00 d2=0.0e+00
4 data properties n=2..10, 12 seeds each                     PASS    108 instances ok
4 n out of range rejected                                    PASS    
5 slack reaches exactly 0..B (B=1..79,127..1000)             PASS    ok
5 edge budgets 1..17 hand portfolios: energy/opt/penalty     PASS    ok
6 baseline.brute_force == own opt; greedy feasible & >= opt  PASS    max greedy gap=0.4989; ok
X stress n in {2,3,5,9,10}, lam up to 50 (75 inst)           PASS    ok

28/28 passed, 0 failed, 1.2s
```

## Notes and caveats (no failures)

- The "1 own brute-force" rows only record that the reference optimum was computed; they assert nothing by themselves. The optimum is asserted in 2b/2c and in check 6.
- Check 2 covers seeds 0-9, n=8 (N = 8 + budget.bit_length() qubits, all 2^N states). Smallest gap between any infeasible state (loan budget violated or nonzero residual) and the optimum is 6.43 (lam=0), 6.79 (lam=0.3 and 1.0) in seeds 0-9. Always positive, so the P = sum(ret)+1 penalty holds with margin in these instances. The extra stress sweep (n in {2,3,5,9,10}, lam up to 50, 75 instances) and hand-built portfolios with budgets 1..17 also held.
- Qubit-order check is discriminating: reversing bit order changes energies by about 2400, so a reversed mapping would have failed. An asymmetric hand-built QUBO also confirms bit k = qubit k for both Ising paths.
- Data: Sigma symmetric/PSD (min eigenvalue >= -1e-12), same-sector corr 0.6, cross-sector 0.1, vol in [0.1, 0.4], rate in [0.10, 0.35] (ret is rounded to 4 dp, tolerance 1e-4 used), amounts integers 1..8, budget = round(0.4*sum), budget binds, deterministic across calls, all 3 sectors present for n>=3, n outside 2..10 raises ValueError. All 108 instances OK.
- lam=0 ties: argmin check iterates over all tied argmin states and all were feasible and optimal.
- Greedy is feasible and never beats the optimum; largest observed suboptimality gap is 0.4989 (objective units) over seeds 0-9 and lam in {0, 0.3, 1.0}.
- Not covered: the QAOA circuit, sampling, and metric code in qaoa_run.py beyond the Ising conversion. `random_sampling` was not checked.

## Verdict

PASS. No discrepancy found. No bug in data.py, qubo.py, baseline.py or qaoa_run.py (conversion functions) at the tolerance of 1e-8. No files other than the two owned ones were touched.
