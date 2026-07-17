# Program O-R — David model laboratory

## Purpose

`notebooks/program_o_david_model_lab_FULL.ipynb` is a development-only laboratory for David. It keeps the Program O-R environment and evaluation contract fixed while allowing unrestricted edits to the neural surface.

The notebook includes four runnable paths:

1. `RECURRENT_PPO`: the frozen O-R learner family (`MlpLstmPolicy`, LSTM 64, MLP 64-64). This is the reference, not a new scientific rerun.
2. `PPO_DMPLA`: PPO with an editable causal-history DMPLA encoder and learned, sinusoidal, or no positional encoding.
3. `DISCRETE_SAC_DMPLA`: categorical SAC over the actual `Discrete(4)` action. Stable-Baselines3 SAC is not used because it requires a continuous action space.
4. `CUSTOM`: an immediately runnable template whose encoder and training functions David may replace.

## What David may change

- history length and history encoder;
- positional encoding;
- DMPLA/DMLPA layers, heads, widths, normalization, and pooling;
- recurrent, attention, convolutional, or hybrid neural architecture;
- PPO or categorical-SAC hyperparameters;
- the custom training algorithm in the marked cell.

## What remains locked

- Program O-R physical replay and three cells;
- four weekly production-mix actions and eight decisions;
- the 21 allowed state variables before any causal history stacking;
- no latent regime, future demand, tape identifier, true cell parameters, or oracle information;
- terminal `ret_excel_request_snapshot_v2` reward;
- development-only `949...` tapes;
- the common same-tape evaluator and resource/conservation checks.

## Recommended workflow

1. Run `smoke` with `RECURRENT_PPO` once.
2. Run `smoke` with `PPO_DMPLA` and `DISCRETE_SAC_DMPLA`.
3. Edit only the marked architecture cell until the desired model builds and trains.
4. Switch to `development` and compare several optimizer seeds on the same evaluation tapes.
5. Judge models by canonical ReT and the guardrail table, not by training reward.
6. Reject a candidate that collapses to one fixed calendar or violates conservation/resources.
7. If one architecture survives, freeze it in a new preregistration before using any new holdout namespace.

The notebook cannot promote Paper 2, reopen Program O, or use the reserved `748...` scientific tapes.
