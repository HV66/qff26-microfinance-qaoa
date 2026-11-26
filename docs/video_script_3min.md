# 3-minute video script (about 410 words, about 2:55 at a calm pace)

Deck: `docs/QubitQuest_Mentor_Deck.pptx` (slides 1-7 are the video; slide 8 is hidden, for finals Q&A).

**[Slide 1, 0:00-0:15]**
"Hi, I'm Harsha from Team Q-Spark, Finance track, Option B. A microfinance lender has a fixed budget and more borrowers than it can fund. Which loans should it pick? We tested whether QAOA can help, and we tested it honestly."

**[Slide 2, 0:15-0:38]**
"Returns add up easily, but risk doesn't. Borrowers in the same sector default together after a shared shock, like a bad harvest. So we maximise return, penalise correlated risk, and respect the budget. All data is synthetic and seeded, so this is a test of the method, not a credit-scoring system."

**[Slide 3, 0:38-1:08]**
"Each loan is a yes-or-no bit. The QUBO scores return, subtracts risk for same-sector pairs, and penalises going over budget. Eight loans become twelve qubits, because four slack bits encode the budget. We convert it to an Ising Hamiltonian with our own code and run QAOA at depths one to four. Four classical baselines run on the same instances: exact search, greedy, simulated annealing and random sampling. An independent checker re-derived every energy and passed 28 of 28 checks."

**[Slide 4, 1:08-1:43]** (optional: 8-second sped-up clip of `python demo.py` over the first sentence)
"Here's the surprise: every method finds the optimum, even random sampling. With only 256 possible portfolios and 4096 shots, we nearly sample the whole space, so a zero percent gap proves little. The honest metric is the chance that one shot hits the best answer. QAOA scores between 0.0024 and 0.0051. Random scores 0.0029. Statistically, that's indistinguishable. Greedy is the only method that misses the optimum, by 3 percent on average and 15 percent in the worst case."

**[Slide 5, 1:43-2:25]**
"We also ran two real jobs on ibm_fez: five loans, nine qubits, depth one, 4096 shots, with dynamical decoupling and Pauli twirling. [cut to the cropped IBM Quantum Workloads page for 4 to 6 seconds, then back to the slide] The first job got 0.0405, above the random rate. The second got 0.0200, below it, and its angles only give 0.0134 even with no noise. Two jobs are too few to conclude anything, so we report both. Our twelve-qubit instance needs over five hundred two-qubit gates at depth two, so we did not run it on hardware."

**[Slide 6, 2:25-2:48]**
"Our verdict: zero quantum-advantage claims. Of sixteen QAOA runs, five were above random, ten indistinguishable, and one below. What we offer is a verified, reproducible testbed, with 407 tests, fixed seeds and pinned versions. And a fairness warning: in six of fifteen optimal portfolios, one whole sector gets no funding."

**[Slide 7, 2:48-3:00]**
"Everything is open source under Apache-2.0 at github.com/HV66/qff26-microfinance-qaoa. Thank you."

## Before recording
- Test count confirmed: `.venv\Scripts\python.exe -m pytest -q` printed "407 passed in 79.77s" on 2026-11-27. Re-run if the code changes.
- Time it. If over 3:00, cut the "independent checker" sentence on slide 3 first, then the demo clip.
- Crop the IBM Workloads screenshot to the job rows. Redact API key, instance CRN, account name and email. Job IDs are fine.
- Say 0.0405 and 0.0200 exactly as shown on slide 5.
- Upload unlisted, then test the link while logged out.
