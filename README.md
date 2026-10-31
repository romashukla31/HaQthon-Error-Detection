# Quantum Checkpoint

Local Streamlit game for IBM Qiskit Fall Fest Track 1. Spend one Paulice
spacetime checkpoint, filter 100–2,000 noisy Aer shots by helper q2, and compare
the two-data-qubit state before and after filtering.

## Run it on Windows

```powershell
cd C:\Users\Roma Shukla\Desktop\quantum-checkpoint
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local address Streamlit prints, normally `http://localhost:8501`.

## What is implemented

- Data q0 or q1 can be selected, with verified Short and Long Paulice intervals.
  The q0 pair starts at the same actual layer (L2) and ends at L5 or L9.
- Helper q2 is the only check qubit. Its syndrome is direct: `0 = clean / keep`,
  `1 = flagged / discard`.
- Level 1 has six Clifford S gates after the check. Level 2 moves one existing
  `T(q1)` immediately before the fixed payload `CX(q0, q1)` and leaves five T
  gates after the check. The fixed payload and local Aer noise model do not
  change when a placement changes.
- The q0-only **Try verified inside-T pair** action is a separate fair comparison:
  it uses one matched S/T gate inside the interval where the propagated Paulice
  check is Z. It has the same endpoints, gate count, and noise in both cases.
- Exact scores are density-matrix **data-state fidelity** (the data is unencoded).
  A shot sample uses the selected seed; exact values deliberately do not change
  when only shot count or seed changes.

## Noise and acceptance prediction

Aer uses 0.3% 1-qubit depolarizing noise, 2.5% 2-qubit depolarizing noise, and
1.5% independent readout flip. The exact predicted acceptance includes the same
q2 readout flip used by sampled shots. This fixes the earlier mismatch where
pre-readout helper projection was compared with post-readout samples.

Run the checks:

```powershell
.\.venv\Scripts\python.exe -c "from simulation import verification_report; print(verification_report())"
.\.venv\Scripts\python.exe -c "from simulation import sampling_consistency_report; print(sampling_consistency_report(repeats=120, shots=1000))"
```

The first validates every offered checkpoint with noise off, both prediction-game
faults, the inside-T pair, and exact-value shot-count independence. The second
pools independent seeds and reports binomial uncertainty for sampled versus
predicted acceptance.

## Demo limitations

- This is local Aer simulation, not IBM hardware data, and q0/q1 are not logically
  encoded.
- Paulice automatically builds Clifford checks. The regular moved-T construction
  and the q0-only inside-T example are offered only after their commutation and
  noiseless preservation were verified.
- A checkpoint can lower data-state fidelity; the app reports the actual result.
