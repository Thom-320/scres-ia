# David sandbox reconciliation

## Reviewed artifacts

- Fable: `notebooks/david_sandbox_program_o_ret.ipynb` at `1a4097c`.
- Codex: `notebooks/program_o_david_model_lab_FULL.ipynb` at `df8645b`.
- Governing evaluator amendment: `2ade04f`.

## What was retained from Fable

- exact enumeration of the 65,536 open-loop calendars;
- all ten frozen state-rich classical configurations;
- modal, phase-only, and frequency-matched placebos;
- the compact `H_learned` / `H_neural` presentation;
- explicit warning that continuous-action SB3 SAC is incompatible with `Discrete(4)`.

## Corrections required

1. Fable's sandbox took the maximum classical score separately on every tape.
   That creates a privileged selector that is not the O-R estimand. The final
   notebook selects one classical configuration by maximum mean across the
   common tapes, exactly as the amended evaluator does.
2. Its generic `predict(obs)` rollout did not retain recurrent hidden state or
   `episode_start`. RecurrentPPO therefore became memoryless during evaluation.
   The final notebook carries LSTM state through all eight decisions.
3. It evaluated only `rho90_share90`. The final notebook evaluates all three
   connected O-R cells.
4. Its editable PPO cell did not actually provide a DMPLA extractor or a
   discrete SAC implementation. The final notebook provides both, plus history
   and three positional-encoding modes.
5. The original Codex C/H perturbation swapped belief, phase, and previous-action
   fields as though they were product pairs. The final helper now performs the
   semantic involution: product pairs swap, beliefs map to `1-p`, action `k`
   maps to `3-k`, and public time fields stay unchanged.

## Final routing

The canonical development laboratory is:

`notebooks/program_o_david_model_lab_FULL.ipynb`

It remains development-only. It cannot change the historical Program O verdict,
promote O-R, or open the 748-series scientific tapes.
