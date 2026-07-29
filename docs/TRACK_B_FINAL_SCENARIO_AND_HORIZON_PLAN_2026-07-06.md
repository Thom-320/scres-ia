# Track B final scenario and horizon plan - 2026-07-06

## Objective

Move from diagnostic screens to final, reviewer-facing confirmations for:

1. Case A: Garrido current risk level, all thesis risks active.
2. Case B: Garrido current risk level, downstream risks only (`R22/R23/R24`).
3. Case C: selected per-risk headroom cell, after the current Case C grid lands.

Each final confirmation should run both:

- PPO+MLP, the efficient spine;
- Real-KAN, the interpretable/novel architecture sidecar.

Primary metric remains Garrido Excel ReT: `order_ret_excel_mean`.

Tail metric: `order_ret_excel_cvar05_mean`.

Mechanism metrics:

- `order_ret_excel_risk_conditional_mean_mean`;
- Excel branch mix (`fill_rate`, `recovery`, `risk_no_recovery`, `unfulfilled`);
- resource cost.

## Horizon question

The current Track B horizon is `104` weekly decisions, i.e. about 2 years.
Garrido's thesis uses much longer strategic horizons in parts of the analysis,
but training PPO/Real-KAN directly at 10-20 year episodes would be expensive and
may reduce the number of learning episodes per compute budget.

Therefore, horizon is now treated as an experimental design parameter, not a
default.

## Stage 0: horizon screen

Runner:

`scripts/run_track_b_horizon_screen.py`

Protocol:

- scenarios: Case A and Case B;
- horizons: `52,104,156,260` weeks (1, 2, 3, 5 years);
- observation: `v7_no_forecast`;
- reward: `control_v1`;
- seeds: `1,2,3`;
- train timesteps: `20000`;
- eval episodes: `6`;
- batch size: `64`;
- architecture: PPO+MLP only.

Decision:

Choose the shortest horizon that:

1. has enough risk exposure for the scenario;
2. keeps PPO's advantage over static visible on `order_ret_excel_mean`;
3. does not collapse CVaR/risk-conditional metrics into noise;
4. is computationally feasible for PPO+MLP and Real-KAN final confirmation.

## Stage 1: final confirmation

Runner:

`scripts/run_track_b_final_scenario_confirm.py`

Protocol after horizon selection:

- architectures: `ppo,real_kan`;
- scenarios: Case A and Case B immediately, Case C after selection;
- observation: `v7_no_forecast`;
- reward: `control_v1`;
- horizon: selected by Stage 0;
- default final scale: at least `5 seeds x 60k x 12 eval episodes`;
- Real-KAN uses batch size `256`;
- PPO+MLP uses batch size `64` unless a completed batch-size verdict overrides it.

## Claim boundaries

Case A is the most thesis-faithful risk roster.

Case B is a controlled downstream mechanism scenario. It can support a branch
shift / exposure-reduction mechanism claim, but its absolute ReT scale must not
be compared directly against Case A because the Excel branch composition differs.

Case C is a headroom/stress design. It can be used to test whether targeted
frequency/impact changes create more adaptive or preventive headroom, but it is
not the default thesis-faithful scenario.
