# STATUS

Updated: 2026-11-25. Code freeze: Thu 26 Nov 23:59 IST.

## Done (committed, 269 tests, ruff and black clean)
- data, QUBO (own Ising conversion, PSD penalty proof), baselines (exact, greedy, SA, random)
- QAOA p=1..4 seed 0, p=1..3 seeds 1-4; per-run QAOA vs random binomial table
- Noise-model runs with real saved parameters: 12 qubits p=1, p=2; 9 qubits p=1
- Independent verifier 28/28; demo (default = saved best-p, --quick = reduced budget)
- README, architecture.mmd, pitch_script.md, submission_checklist.md
- Fresh-clone `scripts/run_all.py`: all 9 steps OK, exit 0, 372 s

## Blocked
- Real IBM hardware: no saved account or QISKIT_IBM_TOKEN here. Nothing submitted.
  `results/hardware_job.json` does not exist. Submit path untested.

## Left for the user
- See final report: hardware run, video, public repo push, team form.
