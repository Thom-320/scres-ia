# Track B Case C per-risk headroom preregistration - 2026-07-06

## Question

Can we create a controlled Garrido-native Track B stress setting with enough
adaptive/preventive headroom by changing frequency and impact per risk ID, rather
than multiplying all enabled downstream risks together?

Primary metric: Garrido Excel ReT, `order_ret_excel_mean`.

Important caveat: this lane enables only `R22,R23,R24`, so absolute ReT values
are not comparable to the all-risk Track B headline. Compare cells by relative
margin against the best static baseline within the same scenario, branch mix, and
resource cost.

## Protocol

- Runner: `scripts/run_track_b_case_c_per_risk_headroom_grid.py`
- Output: `outputs/experiments/track_b_case_c_per_risk_headroom_grid_2026-07-06/`
- Observation: `v7_no_forecast`
- Risk level: Garrido `current`
- Enabled risks: `R22,R23,R24`
- Reward: `control_v1`
- Action contract: `track_b_v1`
- Seeds: `1,2,3`
- Train timesteps: `30000`
- Eval episodes: `8`
- Horizon: `104`
- Batch size: `64`

## Cells

| Cell | Frequency by risk | Impact by risk | Purpose |
|---|---|---|---|
| `base` | none | none | downstream-only reference |
| `r24_freq2_impact1p25` | R24=2.0 | R24=1.25 | frequent learnable demand surge |
| `r24_freq3_impact1p0` | R24=3.0 | none | high R24 frequency headroom |
| `r22_impact1p5` | none | R22=1.5 | LOC severity only |
| `r23_impact1p5` | none | R23=1.5 | advanced-unit severity only |
| `r22r23_impact1p5` | none | R22=1.5, R23=1.5 | downstream disruption severity |
| `r24_freq2_r22r23_impact1p5` | R24=2.0 | R22=1.5, R23=1.5 | mixed prediction and severity |
| `r24_freq3_r22r23_impact1p5` | R24=3.0 | R22=1.5, R23=1.5 | maximum headroom screen |

## Decision Rule

Promote a cell to a heavier confirmatory/prevention gate only if it shows:

1. Clear relative margin over the best static baseline within the same scenario.
2. A branch mix that remains interpretable, reported via Excel branch percentages.
3. Resource cost that does not simply collapse to always-max capacity unless the
   explicit objective is stress/adaptation rather than operational efficiency.

If a promoted cell exists, the next gate is a 5-10 seed 60k confirmation followed
by the fixed-RNG event-aligned prevention audit on the target risks.
