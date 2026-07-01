# Garrido Controlled-Risk Probe (2026-06-29)

## Purpose

Test the user's thesis-faithful idea: Garrido controls risk families/scenarios, so we should not only train/evaluate on all-risk war stress. This probe activates controlled risk families or CF-specific risk subsets and compares a cheap learned Track-A continuous policy against a dense static buffer/shift frontier.

This is a signal-finding screen, not a confirmatory run.

## Runner

`scripts/run_garrido_controlled_risk_probe.py`

Default cheap screen:

```bash
.venv/bin/python scripts/run_garrido_controlled_risk_probe.py \
  --cases R1,R2,R3 \
  --seeds 1 \
  --n-envs 2 \
  --timesteps 4000 \
  --eval-episodes 2 \
  --max-steps 52 \
  --fracs 0,0.05,0.10,0.125,0.15,0.25,0.50 \
  --reward-mode ReT_excel_plus_cvar \
  --cvar-alpha 0.2
```

Metrics:

- Primary: Excel ReT.
- Secondary: CVaR95 service loss, flow fill, lost rate, resource composite.
- Learning audit: action variation (`frac_std`) and correlations between action and realized-risk/hazard observation fields.

## Results

### R1/R2/R3 family screen

Artifact: `outputs/experiments/garrido_controlled_risk_probe_r1r2r3_1seed_2026-06-29`

| case | dynamic Excel | best static Excel | dynamic resource | action variability | verdict |
|---|---:|---:|---:|---:|---|
| R1 | 0.005617 | `f0_S2` 0.005733 | 0.549 | `frac_std=0.034` | no promote |
| R2 | 0.350189 | `f0_S2` 0.350189 | 0.250 | `frac_std=0.000` | no promote |
| R3 | 0.486757 | `f0_S2` 0.486757 | 0.250 | `frac_std=0.000` | no promote |

R1 shows weak action variation, but the top correlations are largely time/hazard proxy fields and it still loses to a simple static. R2/R3 collapse exactly to the best constant.

### CF-specific R2 screen

Artifact: `outputs/experiments/garrido_controlled_risk_probe_cf13_cf20_1seed_2026-06-29`

| case | risk overrides | dynamic Excel | best static Excel | action variability | verdict |
|---|---|---:|---:|---:|---|
| CF13 | R21/R22/R24 increased, R23 current | 0.286115 | `f0_S2` 0.286115 | `frac_std=0.000` | no promote |
| CF20 | R21/R22/R23/R24 increased | 0.216960 | `f0.05_S1` 0.261393 | `frac_std=0.000` | no promote |

## Interpretation

The controlled-risk idea is scientifically useful and thesis-faithful, but the first cheap probes do not produce a win. They reinforce the same structural boundary seen elsewhere: with Track-A buffer/shift variables, the best policy is usually a simple constant base-stock/shift choice.

This does **not** prove the lane is exhausted. It says:

1. Do not scale broad R1/R2/R3 training yet.
2. If continuing this lane, focus on the hard cases where the static frontier is not trivially `f0_S2`, especially CF20-like R2 severe subsets, but switch to either per-op buffer or Track B if the goal is raw Excel ReT.
3. For learning evidence, do not rely on win/loss alone. Use:
   - action variability and risk/hazard correlation,
   - checkpoint improvement,
   - static imitation gate,
   - retained-vs-reset `Delta memory`,
   - ablation/shuffle of risk observations.

## Next Suggested Step

Keep this runner as a cheap gate. The higher-potential scientific work now is:

1. R2 endogenous fidelity audit: why R2 endogeneous ReT is still high versus Raw_data2.
2. H4 retained-vs-reset on the current best continuous lane.
3. Track B / downstream dispatch, if the objective is a positive raw Excel ReT result.
