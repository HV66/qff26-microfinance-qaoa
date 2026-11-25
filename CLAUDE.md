# CLAUDE.md: QubitQuest, IBM Qiskit Fall Fest 2026 (QFF26) Hackathon

Source of truth: `QFF26_Hackathon_Playbook_Case_Studies_and_Judging.pptx` (17 slides). Re-read it if anything here seems off. The track/idea is **not chosen yet**. Wait for the user to pick one, then apply the track section below.

## 1. Event facts
- Name: 48-Hour Quantum for Humanity Hackathon (IBM Qiskit Fall Fest 2026). Hybrid/online. Tags: #QiskitFallFest2026 #QuantumForHumanity.
- Links: ibm.biz/qiskitfallfest2026, Discord discord.gg/u2MKXRT4n.
- 4 tracks: Healthcare & Life Sciences, Climate/Energy/Agriculture, Education, Financial Inclusion & Resilient Markets. One track per team.
- Teams of 3-5 (recommended; a team of 3 should cover all four roles). One team per person. Team lead and team name required.
- Every registered team gets free IBM Quantum credits (real hardware access).

## 2. Timeline (all IST)
| When | What |
|---|---|
| Tue 24 Nov 18:00-19:00 | Kickoff, team formation on Discord (#hackathon-teams) |
| Tue 24 Nov 19:00 | Teams locked, hacking starts. Hardware deep dive (Eagle/Heron) |
| Tue 24 Nov midnight | Git repo created and shared with mentor |
| Thu 26 Nov 16:30 | Midpoint review. Top 20 teams get a 15-min 1-on-1 mentor sprint (feedback only, no score) |
| **Thu 26 Nov 23:59** | **CODE FREEZE. Submit GitHub repo + 3-min pitch video link. Late submissions are NOT scored** |
| Fri 27 Nov 08:00 | Judges score repos and videos. Top 10 announced on Discord and email |
| Fri 27 Nov 09:30 | Grand finals: 5-min live pitch + 3-min Q&A before Executive Judges (rubric applied again) |
| Fri 27 Nov 12:00 | Awards: IBM Quantum Prize, credits, certificates |

Today is 2026-11-25, so the code freeze is about a day away. Prioritise ruthlessly and tell the user the time left whenever planning.

Tie-break: higher Technical score wins, then higher Impact score. The tie-break and late-submission rules are marked "proposed" in the deck notes.

## 3. Judging (100 pts, each judge scores 1-10 per criterion, weighted)
1. **Technical rigor & Qiskit code, 30%**: clean, reproducible code; efficient circuit depth; correct Runtime primitives (Estimator/Sampler); error mitigation; tests and README.
2. **Humanitarian/Business impact, 25%**: real problem and real beneficiary; who gains and by how much; realistic path to deployment; ethical data use.
3. **Innovation & quantum feasibility, 25%**: original idea; honest justification of quantum advantage vs a classical baseline; realistic about today's hardware.
4. **Pitch & presentation, 20%**: clear 3-min story; architecture diagram; working live demo; confident Q&A.

Organisers explicitly reward an honest classical comparison over inflated quantum-advantage claims.

## 4. Submission checklist (due Thu 26 Nov 23:59 IST)
Required:
- [ ] Public GitHub repo with an open-source licence (MIT/Apache-2.0)
- [ ] README: problem, approach, how to run, results
- [ ] Runnable Qiskit code or notebook
- [ ] Classical baseline and comparison table
- [ ] 3-minute pitch video link
- [ ] Team name, members and track in the submission form (form link not in the deck, so ask the user or the organisers)

Strongly recommended (these influence scores):
- [ ] Architecture diagram (1 slide)
- [ ] Results from real IBM Quantum hardware, or a noise-model simulation
- [ ] Error-mitigation notes and circuit-depth stats
- [ ] Honest limitations and next steps
- [ ] Pinned Qiskit and package versions (`requirements.txt`/`pyproject.toml`)
- [ ] Submit early, not at 23:59

Finals extras: a 5-min live pitch and a 3-min Q&A. Prepare a rehearsed live demo and Q&A prep.

## 5. Rules and code of conduct
- All code written during the 48 hours. Open-source libraries are allowed and must be credited (README credits section).
- Data: synthetic or public, anonymised/de-identified only. Never real patient, customer or confidential data.
- No plagiarism, no inflated claims. Judges' decisions are final.
- Respect everyone, help other teams, report concerns in #event-ops.
- Use IBM Quantum credits responsibly: **simulator first, then queue hardware jobs**.
- Several rules are flagged in the deck notes as "proposed defaults, confirm with organisers". Treat them as binding, but mention that to the user if it matters.

## 6. Tracks and what each demands
Every track pitch is 3 minutes. Full deliverables are listed per track.

**Healthcare (TB/malaria drug discovery)**: VQE or Hamiltonian simulation of a small molecule fragment, using Runtime Estimator, a compact ansatz and ZNE mitigation. Deliver: energy vs bond-distance curve, circuit depth and shot budget, accuracy and runtime vs a Hartree-Fock baseline. Stretch: active-space reduction. Tools: Qiskit SDK, SQD addons, Qiskit Nature, Qiskit ML. Mentor: Pradeep Lamba. Channel #challenge-healthcare.

**Climate (microgrids / carbon capture)**: Load balancing on 6-12 microgrid nodes as QUBO solved with QAOA (Sampler, vary layers p, classical optimiser). Alternative: CO2 binding on a MOF fragment. Deliver: QUBO formulation, convergence vs graph density, comparison with a greedy or simulated-annealing baseline, deployment metrics (% demand met, cost saved). Stretch: one run on real hardware and noise impact. Mentors: Bharadwaj Mummaneni, Phalak Bhatnagar. #challenge-climate.

**Education**: Interactive multilingual tool teaching one idea (superposition, entanglement or Grover), real Qiskit circuits, runs on a laptop simulator, with live histogram updates on gate change. Deliver: working demo in 2+ languages, 3 guided lessons with quizzes, open-source repo, evidence from a quick learner test (even 3 people), 3-min pitch. Stretch: accessibility (screen reader, low-bandwidth). #challenge-education.

**Finance (fraud / microfinance)**: Option A: QSVM with a quantum kernel for fraud classification (PCA to fit qubit budget). Option B: microfinance portfolio via QUBO and QAOA. Deliver: quantum vs classical RBF-kernel comparison (accuracy, precision/recall, runtime), fair reproducible dataset split, honest limits note, 3-min pitch. Mentors: Archit Chadalawada, Guncha Malik. #challenge-finance.

## 7. Working rules for Claude on this project
1. **Do not start building until the user names the idea/track.** Then restate the track's deliverables as a checklist and a time-boxed plan.
2. Every design decision should map to the rubric. Check the 30/25/25/20 weights before choosing scope.
3. Always build a **classical baseline** next to the quantum solution, with a comparison table. Never claim quantum advantage that is not measured. Report limits plainly.
4. Use current Qiskit idioms: Runtime primitives (`Estimator`/`Sampler`, V2), transpile for the target backend, report circuit depth and shot counts. Pin versions. Check the installed Qiskit version before writing code, since APIs change between releases.
5. Develop on simulators (and a noise model from a real backend) first. Run hardware jobs sparingly to conserve credits. Log backend, date and job IDs for reproducibility.
6. Add error mitigation where relevant (e.g. ZNE, dynamical decoupling, readout mitigation) and document it.
7. Reproducibility: fixed seeds, deterministic splits, `requirements.txt`, a one-command run path, a small test suite, and a README with run instructions.
8. Data: synthetic or public anonymised only. Flag anything that looks like real personal data.
9. Credit all third-party libraries and datasets in the README. Include an open-source LICENSE file. Code must be original to the 48-hour window, so do not paste in outside project code.
10. Keep scope narrow: "small and working beats big and broken." Prefer a working end-to-end slice early, then improve it. Freeze features at about hour 40 and shift to README, video and slides.
11. Produce the supporting deliverables too, not just code: architecture diagram (1 slide), pitch script (3 min, plus 5 min finals version), demo script, Q&A prep, limitations and next steps.
12. Aim to have everything submitted well before Thu 26 Nov 23:59 IST. Warn the user about time at each milestone.
13. Unknowns to ask the user or organisers: submission form URL, exact repo naming, video hosting format, which hardware backends the team's credits cover.

## 8. Suggested repo layout
```
README.md  LICENSE  requirements.txt
src/ (quantum/, classical_baseline/, data/, utils/)
notebooks/  tests/  results/ (tables, plots, job logs)
docs/ (architecture.png, pitch_script.md, limitations.md)
```

## 9. Pace plan from the deck
Hours 0-6: pick one narrow problem, repo, hello-world circuit on a simulator. Hours 6-30: build quantum core and classical baseline in parallel, commit often. Hours 30-40: hardware runs, error mitigation, midpoint sprint. Hours 40-48: freeze features, README, 3-min video, submit early.
