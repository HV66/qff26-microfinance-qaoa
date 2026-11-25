# STATUS

Updated: 2026-11-25. Code freeze: Thu 26 Nov 23:59 IST.

## Done (committed)
- data.py, qubo.py, baseline.py, scripts/lambda_sweep.py (212 tests passing at 701f6aa)

## In progress
- Phase 1: qaoa_run.py + tests/test_qaoa.py written, NOT yet gated or committed.
  First pytest run of tests/test_qaoa.py exceeded 300 s (running in background). Likely cause: run_one test and optimise at 12 qubits. Check before committing.
- Phase 2 subagents launched in parallel BEFORE the Phase 1 commit (user asked to fan out):
  - A hardware: hardware_run.py, retrieve_hardware.py, noise_run.py, tests/test_hardware.py
  - B compare: compare.py, sa_baseline.py, scripts/make_results_md.py, results/results.md
  - C verifier: tests/independent/verify_independent.py, results/verifier_report.md
  - D demo/packaging: demo.py, LICENSE (Apache-2.0), requirements.txt, .gitignore, scripts/run_all.py

## Blocked
- Real IBM hardware run needs the user's saved account or QISKIT_IBM_TOKEN.

## Next
- Finish Phase 1 gate and commit, then run seed 0 p=1..4 and seeds 1-4 p=1..3.
- Merge Phase 2, run full gate, read verifier_report.md, commit.
- Phase 3 docs (README, pitch), Phase 4 fresh-clone QA.
