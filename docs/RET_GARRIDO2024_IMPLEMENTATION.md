# Garrido 2024 Paper-Faithful Reward Family

## Status

The repository now includes a paper-faithful Garrido et al. (2024) Cobb-Douglas reward/index family implemented in [env_experimental_shifts.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/env_experimental_shifts.py).

This was added as a **separate family** and does **not** replace the frozen mainline `ReT_seq_v1` contract.

## Implemented Modes

Two explicit modes now exist:

- `ReT_garrido2024_raw`
  Raw Cobb-Douglas product from Garrido 2024 Eq. 3.
  This is documented in code as a `training reward candidate`.

- `ReT_garrido2024`
  Sigmoid-bounded Garrido 2024 index from Eq. 6.
  This is documented in code as the `evaluation/audit index`.

The intended split is:

- training reward candidate: `ReT_garrido2024_raw`
- paper-facing evaluation index: `ReT_garrido2024`

## DES-Compatible Variable Semantics

The five Garrido-2024 variables are now computed with explicit MFSC DES semantics:

- `ζ`
  Average finished-goods ration inventory since warmup end.

- `ε`
  Average pending backorder quantity since warmup end.

- `φ`
  Average spare assembly capacity since warmup end.

- `τ`
  Average net-requirement coverage-time proxy since warmup end.

- `κ̇`
  Average operational cost normalized by a Monte-Carlo reference cost.

### Concrete MFSC mapping

- `ζ`
  Computed from downstream ration buffers:
  `rations_al + rations_sb + rations_sb_dispatch + rations_cssu + rations_theatre`

- `ε`
  Computed from `pending_backorder_qty`

- `φ`
  Computed as:
  `max(available assembly capacity in the step - produced quantity in the step, 0)`

- `τ`
  Computed from a DES-compatible proxy:
  `NR_t = max(GR_t - I_(t-1) + B_(t-1), 0)`
  with denominator `min(GR_t, Θ_t)` stabilized with a physical floor of `1.0`

- `κ̇`
  Computed from average step cost divided by `kappa_ref`

## Added DES Signals

To support the faithful transformation, [supply_chain.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/supply_chain.py) now exposes these step-level signals:

- `new_produced`
- `new_available_assembly_hours`
- `new_available_assembly_capacity`

These are required to compute `φ` and `κ̇` from the simulator rather than from RL-side proxies.

## Calibration Procedure

The calibration script was rewritten in [calibrate_cd_exponents.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/scripts/calibrate_cd_exponents.py).

It now follows the Garrido 2024 maxima procedure:

1. Run Monte-Carlo episodes.
2. Compute episode-level averages for `ζ, ε, φ, τ, κ̇`.
3. Find the maxima across the Monte-Carlo sample.
4. Set each exponent by:

```text
exponent * ln(max_value) = 0.20
```

The repo default calibration file is:

- [ret_garrido2024_calibration.json](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/data/ret_garrido2024_calibration.json)

This file is automatically loaded by the environment unless another calibration path is passed through CLI.

## Why Raw vs Sigmoid Are Split

The split is intentional and should be preserved:

- `ReT_garrido2024_raw`
  Best interpreted as the trainable continuous objective.
  It preserves the multiplicative Cobb-Douglas structure without adding saturation.

- `ReT_garrido2024`
  Best interpreted as the paper-facing bounded resilience index.
  It corresponds to Garrido 2024 Eq. 6 and is the correct audit/evaluation form.

This keeps the code aligned with the paper while avoiding unnecessary compression of the RL learning signal.

## Operational Rule for the Repo

The Garrido-2024 family should be treated as follows:

- `ReT_seq_v1`
  Mainline repo reward contract for current PPO benchmarking

- `ReT_garrido2024_raw`
  Paper-faithful training-reward candidate

- `ReT_garrido2024`
  Paper-faithful evaluation/audit index

- `ret_thesis_corrected`
  Thesis-aligned audit signal

## CLI Integration

The Garrido-2024 family is wired through:

- [train_agent.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/train_agent.py)
- [benchmark_control_reward.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/scripts/benchmark_control_reward.py)
- [run_paper_benchmark.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/scripts/run_paper_benchmark.py)

The optional override flag is:

```bash
--ret-g24-calibration /path/to/calibration.json
```

## Implementation Summary

The key facts to preserve are:

- The Garrido 2024 transformation is now explicitly implemented, not just approximated.
- The five variables are DES-grounded, not ad hoc RL proxies.
- Exponents are calibrated with the paper-style maxima rule.
- The raw product and sigmoid index are intentionally separate.
- The raw form is for training candidacy; the sigmoid form is for evaluation and audit.
