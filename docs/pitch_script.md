# Pitch script: microfinance loan selection with QUBO + QAOA

Honest framing throughout: we built and verified the full QAOA pipeline. At 8 loans classical methods solve the problem exactly and QAOA does not beat them. We make no quantum-advantage claim. Two real-hardware jobs exist (`ibm_fez`, 5 loans, p=1, 4096 shots each, seeds 0 and 1; `results/hardware_result.json` and `results/hardware_result_seed1_n5_p1.json`) and they disagree; no other size was run on hardware. All numbers come from `results/` unless marked derived.

## A. 3-minute pitch (target 3:00)

| Time | Segment | Say | Show |
|---|---|---|---|
| 0:00-0:25 (25 s) | Who benefits | "A microfinance lender has a fixed pool of money and more borrowers than it can fund. Every loan it declines is a family without working capital; every default it absorbs shrinks the pool for the next borrower. Better selection means more low-income borrowers funded per unit of risk." | Title slide |
| 0:25-0:55 (30 s) | The problem | "Which loans do we fund under a budget? Returns are easy to add up. Risk is not: defaults are correlated within a sector, so ten loans in one sector are riskier than ten spread across three. That makes it a quadratic binary optimisation problem." | Architecture slide, point at data.py -> qubo.py |
| 0:55-1:25 (30 s) | Approach | "We encode it as a QUBO: expected return minus lambda times correlated risk, plus a budget penalty with binary slack bits. 8 loans become 12 qubits. We convert to an Ising Hamiltonian, run QAOA at depth 1 to 4, and sample 4096 shots. Next to it sits an exact brute force, greedy, simulated annealing and random sampling, on identical instances." | Architecture slide, quantum branch then classical branch |
| 1:25-2:05 (40 s) | Live demo | "This is one command." Run `python demo.py` (loads the saved seed-0 parameters and samples 4096 shots). Point at: (1) the loan table with sectors and budget; (2) the penalty/feasibility line; (3) the final table where every method reports the same objective; (4) the computed note stating no advantage. | Terminal, then `results/demo_plot.png` |
| 2:05-2:35 (30 s) | Honest result | "At 8 loans, brute force finds the optimum in 0.0061 seconds. QAOA reaches the same optimum, 0.00 percent gap, but so does random sampling, because the search space is only 256 loan vectors. The informative number is the chance a shot hits the optimum. On our headline instance QAOA is 0.0051 at best against 0.0029 for random sampling, and a binomial test says that is indistinguishable from random. Across all 16 runs, 5 are above random, 10 indistinguishable and 1 below. On a fake_fez noise model P(optimum) falls to 0.0029." | results.md table |
| 2:35-2:50 (15 s) | Quantum claim | "We built and verified the full QAOA pipeline, 28 of 28 independent checks pass. At 8 loans classical methods solve it exactly and QAOA does not beat them. We make no advantage claim. We ran two small 5-loan jobs on ibm_fez: one came out above the random rate (166 hits against 128 expected) and one below it (82 hits), so we draw no conclusion and make no advantage claim." | Limitations slide |
| 2:50-3:00 (10 s) | Close | "Next: larger instances where brute force is infeasible, more hardware runs and seeds with and without our dynamical decoupling and twirling, and warm-started QAOA. Thank you." | Next steps slide |

Segment seconds: 25+30+30+40+30+15+10 = 180 s = 3:00.

Honest claim, slide by slide:

- Results slide: "A 0 percent gap is weak evidence, because 4096 shots touch 201 to 256 of the 256 loan vectors. P(optimum) against random sampling is the metric."
- Circuit slide: "The feasibility finding is size: transpiled depth on fake_fez 334 with 223 two-qubit gates at p=1 and 655 with 519 at p=2. Our own logical depth, with all-to-all gates, is 72 and 114."
- Limitations slide (the honest claim): "No quantum advantage at this size. We contribute a verified pipeline and noise measurements, plus two small 5-loan ibm_fez jobs that disagree; we have not run the 12-qubit instance on hardware."

## B. 5-minute finals version (target 5:00; then 3 min Q&A)

| Time | Segment | Content |
|---|---|---|
| 0:00-0:30 (30 s) | Beneficiary | Same opening; add that lenders and borrowers both lose when selection ignores correlated sector risk. |
| 0:30-1:10 (40 s) | Problem | Budget knapsack plus sector-correlated covariance; show the synthetic data design (3 sectors, same-sector correlation 0.6, cross-sector 0.1). Synthetic only, no real borrower data. |
| 1:10-2:00 (50 s) | Approach | Architecture slide in full: QUBO with penalty P = sum(returns) + 1 and slack bits, Ising conversion, QAOAAnsatz, COBYLA on exact statevector, Sampler 4096 shots, shared decode helpers. |
| 2:00-2:40 (40 s) | Verification | Independent verifier (`results/verifier_report.md`): 28/28 checks pass; every infeasible state sits at least 6.43 above the optimum (lam=0) and 6.79 (lam=0.3, 1.0) over seeds 0-9, so the penalty holds with margin. Qubit-order check is discriminating (reversal changes energies by about 2400). |
| 2:40-3:30 (50 s) | Live demo | `python demo.py`; point at the loan table, feasibility, per-method objective and the computed no-advantage note. Fallback if it hangs: show `results/demo_plot.png`. |
| 3:30-4:15 (45 s) | Results | Main table (seed 0): every method 0.00 percent gap; brute force 0.0061 s, greedy 0.0006 s, SA 0.1558 s, QAOA 3 to 6 s. Five-seed summary: greedy mean gap 3.08 percent, worst 15.40 percent; SA, random and QAOA p=1..3 all 0.00 percent. Lambda sweep: greedy suboptimal in 7 of 50 seed/lambda cases, worst 30.83 percent (derived from `lambda_sweep.csv`, see note below). Noise model fake_fez: transpiled depth on fake_fez 334, 223 two-qubit gates (logical depth 72), 0.00 percent gap, P(optimum) 0.0029 noisy, 0.0022 with readout mitigation. |
| 4:15-4:40 (25 s) | Honest claim | No advantage. Per-run binomial test vs random (`results/results.md`): 5 of 16 QAOA runs above random, 10 indistinguishable, 1 below; the seed-0 headline runs are all indistinguishable. Two 5-loan ibm_fez jobs: seed 0 above random (166 hits vs 128 expected, survives Holm), seed 1 below random (82 hits); mixed, no advantage claim. |
| 4:40-5:00 (20 s) | Next steps | Larger n beyond brute force, hardware run via `hardware_run.py` then `retrieve_hardware.py`, warm start, CVaR objective. |

Segment seconds: 30+40+50+40+50+45+25+20 = 300 s = 5:00.

<!-- Derived figures, computed with a one-off Python snippet over results/lambda_sweep.csv:
     50 (seed, lam) rows; 7 rows with greedy_gap_pct > 0; max greedy_gap_pct 30.8328;
     28 rows where optimum differs from lam=0.
     P(optimum) hit counts: 0.0051*4096 = 20.9, 0.0029*4096 = 11.9, 1/256*4096 = 16.0 -->

## C. Likely judge Q&A

1. **Why not just use greedy or brute force?**
   At 8 loans, you should. Brute force solves it in 0.0061 s. Greedy is not safe though: over 10 seeds and 5 lambda values greedy missed the optimum in 7 of 50 cases, up to a 30.83 percent gap (`results/lambda_sweep.csv`; seed 2 is worst), and averaged 3.08 percent gap over our 5 main seeds. Brute force is exponential: 2^8 loan vectors here, 2^n in general. Our classical baselines are the honest reference, and they win at this size.

2. **Is the 0 percent gap meaningful?**
   Mostly not. The gap column is best-of-shots on a 256-vector search space, so even uniform random sampling with 4096 shots reaches 0.00 percent (`results/results.md`). We say so in the results file. The informative column is P(optimum).

3. **Why is P(optimum) near random?**
   For the headline instance (seed 0) it is not distinguishable from random: QAOA at p=1..4 gave 0.0042, 0.0051, 0.0029, 0.0024 against 0.0029 for random sampling at the same 4096 shots, and the binomial test calls all four "indistinguishable from random" (`results/results.md`). Across all 16 runs, 5 are above random, 10 are indistinguishable and 1 is below. So there is a small real effect on some seeds, not on the headline one, and it does not make QAOA competitive with exact or simulated annealing. The penalty terms and slack qubits spread amplitude over 4096 bitstrings (12 qubits), most of them infeasible. The depth-4 run also decoded only 0.3960 feasible vs 0.8833 at p=2, so more layers did not help with our COBYLA budget. We report it rather than tune it.

4. **How was the penalty chosen and proven?**
   P = sum(returns) + 1, set in `qubo.auto_penalty`. Any budget violation or nonzero slack residual then costs at least P, more than any possible return gain. The independent verifier (`tests/independent/verify_independent.py`, `results/verifier_report.md`) recomputed the energy of all 2^12 bitstrings for seeds 0-9: the minimum is the true constrained optimum and the smallest infeasible margin is 6.43 (lam=0) and 6.79 (lam=0.3, 1.0). It also ran 75 stress instances and budgets 1..17. This is verified on those instances, not a proof for all inputs, beyond the argument in the code docstring.

5. **What about noise and hardware?**
   We ran `fake_fez` noise-model simulations with the saved seed-0 parameters (`results/noise_run_p1.json`, `noise_run_p2.json`, `noise_run_n5_p1.json`, 4096 shots). Transpiled depth on fake_fez at 12 qubits, p=1: 334 with 223 two-qubit gates, P(optimum) 0.0054 noiseless, 0.0029 noisy, 0.0022 with tensored readout mitigation. At p=2: depth 655, 519 two-qubit gates, P(optimum) 0.0042, 0.0034 and 0.0037. At 5 loans (9 qubits), p=1: depth 237, 130 two-qubit gates, P(optimum) 0.0500, 0.0420 and 0.0464. Readout mitigation lowered P(optimum) in one row and raised it in two; these are tens of hits, so we claim no benefit. We also ran two real jobs on `ibm_fez`: 5 loans, p=1, 4096 shots each, SamplerV2 with dynamical decoupling and Pauli twirling, for seeds 0 and 1. Seed 0 gave P(optimum) 0.0405 (166 hits; 128 expected for a uniform guess), above random and surviving the Holm correction over all 18 tests; transpiled depth 238 and 130 two-qubit gates on that backend (237 on the fake_fez target, same circuit, different backend). Seed 1 gave 0.0200 (82 hits), below random (exact test p = 7e-06); its angles give only 0.0134 even with no noise, so that low value is mostly the angles, not the hardware. The two runs disagree, we draw no general conclusion, and neither is an advantage claim; the 12-qubit circuit was not run on hardware (519 two-qubit gates at p=2 is above our 300-gate limit).

6. **How does this scale to realistic portfolios?**
   Qubits = n loans + slack bits (12 at n=8), and the data generator is limited to n of 2 to 10. Realistic portfolios are hundreds to thousands of loans, far beyond today's noisy hardware at this circuit depth (depth 334 and 223 two-qubit gates at 12 qubits, p=1, already) and beyond our simulator. Classical solvers such as MIP or SA also scale much further. A credible next step is a mid-size instance (tens of loans) where exact brute force stops being feasible, compared to SA and a MIP solver.

7. **Data ethics: where does the data come from?**
   Fully synthetic and deterministic by seed (`data.py`): 3 sectors, integer loan amounts 1 to 8, rates 0.10 to 0.35, volatilities 0.1 to 0.4, same-sector correlation 0.6 and cross-sector 0.1. No real borrower data. A real deployment would need fairness auditing, because optimising return minus risk can exclude the poorest borrowers, which conflicts with the microfinance mission. We have not built that.

8. **So what is the quantum value?**
   At this size, none: classical exact, greedy and simulated annealing solve the 8-loan instance, and QAOA is indistinguishable from random sampling on the headline seed (`results/results.md`). We make no advantage claim. Our contribution is the verified pipeline and the measurements: an own QUBO-to-Ising conversion checked on every bitstring, a numpy simulator matched to Qiskit at p=1 to 4, a per-run statistical test against random sampling, and noise-model measurements of what the circuit costs on a real coupling map (depth 334 and 223 two-qubit gates at p=1, depth 655 and 519 at p=2). That is a tested starting point for larger instances and real hardware, not a speedup.

9. **Why QAOA at all?**
   Because portfolio selection with a budget is a QUBO, QAOA's natural input, and a verified, reproducible pipeline is a prerequisite for testing larger instances and hardware. We claim a working testbed with an honest baseline, not a speedup.

## C2. Honest-metrics paragraph (say or paste on the results slide)

With 12 qubits there are 4096 basis states, and we sample QAOA 4096 times. Each run therefore touches 201 to 256 of the 256 loan vectors (`results/qaoa_seed*_p*.json`, field `distinct_loan_vectors`), so the best-sample gap is close to an exhaustive search and a 0 percent gap is weak evidence. The meaningful metric is P(optimum) against random sampling at the same shots: 5 of 16 QAOA runs are above random, 10 are indistinguishable and 1 is below (`results/results.md`). The feasibility finding is circuit size on a real coupling map: depth 334 with 223 two-qubit gates at p=1, and depth 655 with 519 two-qubit gates at p=2 (`results/noise_run_p1.json`, `results/noise_run_p2.json`).

## D. Do not say

- Do not say "quantum advantage", "speedup", "outperforms", or "better than classical". The results show no such thing.
- Hardware: say exactly two 5-loan, p=1 jobs ran on `ibm_fez` (4096 shots each, seeds 0 and 1) and that they disagree. Do not quote only the seed-0 result. Do not say the 12-qubit instance ran on hardware, and do not present any hardware P(optimum) as advantage or as a mitigation benefit.
- Depth: always say whether a depth is logical (generic gates, all-to-all) or transpiled on a named backend.
- Do not present the 0.00 percent QAOA gap as evidence of quality; random sampling also reaches it.
- Do not claim the penalty is proven for all inputs; say verified on seeds 0-9 plus stress cases.
- Do not quote P(optimum) differences between QAOA depths as real; they are sampling noise.
- Do not describe the data as real or representative of any lender.
- Do not quote any number not in `results/`; re-check against the files before the finals.
