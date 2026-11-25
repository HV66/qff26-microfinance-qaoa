# Submission checklist (due Thu 26 Nov 2026, 23:59 IST; submit early)

Status values: DONE (path verified to exist in this repo), PARTIAL, TODO-USER (only the user can do it). Paths were checked on 2026-11-25. DONE means the file exists, not that its content was re-audited here.

## Required

| Item | Satisfied by | Status |
|---|---|---|
| Public GitHub repo, MIT/Apache-2.0 licence | `LICENSE` (Apache-2.0) exists. Repo still local: create a public GitHub repo, push, and confirm the LICENSE shows on the page. Suggested: `git remote add origin <url>` then `git push -u origin main` | PARTIAL (licence done; public repo TODO-USER) |
| README: problem, approach, how to run, results | `README.md` (has run, error-mitigation, limitations and credits sections) | DONE |
| Runnable Qiskit code or notebook | `qaoa_run.py`, `demo.py`; one command: `python demo.py`; full pipeline: `python scripts/run_all.py` | DONE |
| Classical baseline and comparison table | `baseline.py`, `sa_baseline.py`, `compare.py`; table in `results/results.md`, data in `results/compare_main.csv`, `results/compare_seeds.csv` | DONE |
| 3-minute pitch video link | Script ready in `docs/pitch_script.md` (section A). Record at 3:00 or less, host it, put the link in README and the form | TODO-USER |
| Team name, members, track in submission form | Track: Finance (Option B, microfinance QUBO + QAOA). Form URL is not in the deck: ask organisers on Discord (#event-ops) | TODO-USER |

## Strongly recommended

| Item | Satisfied by | Status |
|---|---|---|
| Architecture diagram (1 slide) | `docs/architecture.mmd` (Mermaid source). Render to PNG/SVG for the slide, e.g. `npx @mermaid-js/mermaid-cli -i docs/architecture.mmd -o docs/architecture.png`, or paste into mermaid.live | PARTIAL (source done; image TODO-USER) |
| Real hardware results or noise-model simulation | Noise model done: `noise_run.py` -> `results/noise_run_p1.json`, `noise_run_p2.json`, `noise_run_n5_p1.json` (fake_fez). Real hardware NOT run: no `results/hardware_result.json`. After setting the IBM token: `python hardware_run.py`, then later `python retrieve_hardware.py` | PARTIAL (noise model DONE; hardware TODO-USER) |
| Error-mitigation notes and circuit-depth stats | `README.md` section "Error mitigation and circuit depth"; depth per p in `results/results.md`; transpiled depth 334, 223 two-qubit gates in `results/noise_run_p1.json` | DONE |
| Honest limitations and next steps | `README.md` section "Honest limitations"; `docs/pitch_script.md` section D | DONE |
| Pinned Qiskit and package versions | `requirements.txt` (qiskit==2.6.2, qiskit-aer==0.18.4, qiskit-ibm-runtime==0.52.1, and others pinned with ==) | DONE |
| Submit early, not at 23:59 | Target: submit well before the deadline, ideally by Thu 26 Nov evening. Re-run before submitting: `python -m pytest -q` and `python scripts/run_all.py` | TODO-USER |

## Finals extras (Fri 27 Nov, 09:30 IST)

| Item | Satisfied by | Status |
|---|---|---|
| 5-minute live pitch | `docs/pitch_script.md` section B (timings sum to 5:00) | PARTIAL (script DONE; rehearsal TODO-USER) |
| Rehearsed live demo | `python demo.py`; fallback `results/demo_plot.png`. Rehearse twice on the presenting machine | TODO-USER |
| 3-minute Q&A prep | `docs/pitch_script.md` section C (8 answers with numbers from `results/`) | PARTIAL (answers DONE; rehearsal TODO-USER) |
| Update claims if hardware gets run | Edit hardware lines in `docs/pitch_script.md` and `README.md` only after `results/hardware_result.json` exists | TODO-USER |
