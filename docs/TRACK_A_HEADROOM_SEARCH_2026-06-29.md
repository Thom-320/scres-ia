# Track A Headroom Search - 2026-06-29

## Purpose

This audit implements the "activate headroom without self-deception" gate:
before training PPO, test whether Garrido's Track A decision families
(inventory buffer and shift) actually expose a time/regime-varying optimum under
controlled risk families and stress multipliers.

The rule is strict: PPO is not promoted unless a dense static-only gate shows
that a regime oracle beats the best single constant policy and that the best
action changes across regimes.

## Implementation

New runner:

```bash
scripts/run_track_a_headroom_search.py
```

Outputs:

- `static_runs.csv`
- `best_static_by_regime.csv`
- `gate_summary.json`

Supported controls:

- `--mode continuous|per_op`
- `--families R1,R2,R3,R24,mixed`
- `--phis ...`
- `--psis ...`
- `--fracs ...`
- `--shifts ...`
- `--seeds ...`
- `--max-steps ...`

The runner compares the dense common static policy against a regime oracle and
reports `oracle_minus_best_static`, a bootstrap CI, whether the best action
changes across regimes, and a pass/fail promotion flag.

## Runs

### Quick smoke

Continuous:

```bash
.venv/bin/python scripts/run_track_a_headroom_search.py \
  --quick \
  --mode continuous \
  --output outputs/experiments/track_a_headroom_search_quick_continuous_2026-06-29
```

Per-op:

```bash
.venv/bin/python scripts/run_track_a_headroom_search.py \
  --quick \
  --mode per_op \
  --output outputs/experiments/track_a_headroom_search_quick_per_op_2026-06-29
```

Both quick smokes produced no meaningful headroom.

### Broad continuous subset

```bash
.venv/bin/python scripts/run_track_a_headroom_search.py \
  --mode continuous \
  --families R1,R2,R3,R24,mixed \
  --phis 1,4,8 \
  --psis 1.0,2.0 \
  --fracs 0,0.05,0.10,0.15,0.25,0.50 \
  --shifts 1,2,3 \
  --seeds 7000,7001,7002 \
  --max-steps 52 \
  --output outputs/experiments/track_a_headroom_search_broad3_continuous_2026-06-29
```

Result:

- Best single constant: `f0.10_S1`, Excel `0.198775`
- Oracle: Excel `0.198951`
- Oracle minus best static: `+0.000176`
- CI95: `[+0.000024, +0.001611]`
- Best action changes: yes
- Gate verdict: pass, but very small effect

### Full continuous grid

```bash
.venv/bin/python scripts/run_track_a_headroom_search.py \
  --mode continuous \
  --families R1,R2,R3,R24,mixed \
  --phis 1,2,4,6,8 \
  --psis 1.0,1.5,2.0 \
  --fracs 0,0.05,0.10,0.15,0.25,0.50 \
  --shifts 1,2,3 \
  --seeds 7000,7001,7002 \
  --max-steps 52 \
  --output outputs/experiments/track_a_headroom_search_full3_continuous_2026-06-29
```

Result:

- Best single constant: `f0.10_S1`, Excel `0.199841`
- Oracle: Excel `0.200137`
- Oracle minus best static: `+0.000296`
- CI95: `[+0.000296, +0.002372]`
- Best action changes: yes
- Gate verdict: pass, but still tiny

Most regimes are captured by either `f0.10_S1` or `f0_S2`; R3 and R24 are
large plateaus, while R1 and mixed stress create the few action switches.

## Interpretation

The gate found a real but weak structural opening. This is not enough for a
claim and not enough for a confirmatory PPO run. It is enough to justify one
cheap PPO smoke on a deliberately constructed campaign tape that alternates the
regimes where the best static action changes.

The claim boundary is:

- Do not claim Track A is solved or that PPO can win yet.
- Do not train PPO on stationary war cells from this result.
- Do not use a coarse frontier.
- If the PPO smoke cannot beat `f0.10_S1` under dense CRN, close this Track A
  headroom path and move weight to H4 retained-vs-reset and Track B.

## Next Step

Build a small campaign from the regimes that actually change the best action:

- `R1_phi1/4/8`
- selected `R2_phi4/8`
- selected `mixed_phi4/8`

Then run a cheap PPO smoke:

- `continuous_its`
- `risk_obs + hazard`
- reward: `ReT_excel_plus_cvar alpha=0.2` and `ReT_excel_delta`
- `h52`
- 1-2 seeds
- `20k-30k` timesteps
- dense CRN frontier from the same campaign

Promote only if the learned policy beats `f0.10_S1` or the dense campaign
frontier on Excel or CVaR at equal/lower resource, and if the action trace shows
non-trivial switching rather than a disguised constant.

## PPO Campaign Smoke

Implemented:

```bash
scripts/run_track_a_headroom_campaign.py
```

The campaign uses the regimes where the static argmax moved:

```text
R1_phi1_psi1
R1_phi4_psi1
R1_phi8_psi1
R2_phi4_psi1
R2_phi8_psi2
mixed_phi4_psi1
mixed_phi8_psi2
```

Static campaign gate:

```text
robust static: f0.05_S3
robust Excel: 0.036423
oracle Excel: 0.039223
oracle - robust: +0.002800
oracle diversity: 7 actions
```

This is a much stronger opening than the full averaged grid, but it is still
an oracle opening: the learner must discover the switching policy.

### ReT_excel_plus_cvar alpha=0.2 smoke

Artifact:

```text
outputs/experiments/track_a_headroom_campaign_plus_cvar_smoke_2026-06-29
```

Result:

```text
dynamic Excel: 0.029998
dynamic CVaR: 4.606e9
dynamic resource: 0.378
frac_std: 0.039
Excel Pareto: false
CVaR Pareto: false
```

Verdict: fails. The policy is nearly non-adaptive and is dominated by dense
static policies.

### ReT_excel_delta smoke

Artifact:

```text
outputs/experiments/track_a_headroom_campaign_excel_delta_smoke_2026-06-29
```

Result:

```text
dynamic Excel: 0.036308
robust static Excel: 0.036423
dynamic CVaR: 4.378e9
best static CVaR: 4.236e9
dynamic resource: 0.514
frac_std: 0.234
Excel Pareto: false
CVaR Pareto: false
```

Mechanism:

```text
top frac driver: active_R22, corr = 0.855
second: recent_R22, corr = 0.637
```

Verdict: useful learning signal but no win. The agent does condition on the R2
risk signal and nearly matches the robust static, but it spends too much
resource and remains dominated by `f0.05_S2` / `f0.10_S1` style dense statics.

## Updated Conclusion

There is a real Track A oracle opening if we explicitly construct a campaign
from regimes with changing optima, but PPO from scratch did not capture enough
of it. The strongest learned policy so far is **adaptive but not superior**.

The only remaining Track A repair worth testing is not more blind PPO. It is:

1. behavior-clone / warm-start from the static oracle actions for this campaign;
2. then PPO fine-tune;
3. compare against the same dense campaign frontier.

If warm-start maintains but does not improve the oracle/static frontier, Track A
is closed as "static/oracle policy search, not sequential RL headroom." If it
improves, that is the first credible Track A dynamic lane.

## Behavior Cloning / Warm-Start Repair

Implemented BC warm-start in:

```bash
scripts/run_track_a_headroom_campaign.py
```

The warm-start builds an oracle action table from the dense static frontier,
collects observations while replaying that oracle, then minimizes MSE between
the PPO Gaussian policy mean and the oracle action.

Plumbing smoke:

```text
outputs/experiments/track_a_headroom_campaign_bc_smoke_plumbing_2026-06-29
```

The BC path executed and emitted `behavior_cloning` stats.

### BC + PPO fine-tune

Command shape:

```bash
.venv/bin/python scripts/run_track_a_headroom_campaign.py \
  --mode train \
  --reward-mode ReT_excel_delta \
  --bc-epochs 20 \
  --timesteps 20000 \
  --output outputs/experiments/track_a_headroom_campaign_excel_delta_bc20_smoke_2026-06-29
```

Result:

```text
BC loss: 0.2901 -> 0.0594
dynamic Excel: 0.033348
dynamic CVaR: 4.291e9
dynamic resource: 0.370
frac_std: 0.047
Excel Pareto: false
CVaR Pareto: false
```

PPO fine-tuning after BC degraded the stronger from-scratch Excel result
(`0.036308`) and remained dominated by dense static policies.

### BC only

Command shape:

```bash
.venv/bin/python scripts/run_track_a_headroom_campaign.py \
  --mode train \
  --reward-mode ReT_excel_delta \
  --bc-epochs 100 \
  --timesteps 0 \
  --output outputs/experiments/track_a_headroom_campaign_excel_delta_bc100_only_2026-06-29
```

Result:

```text
dynamic Excel: 0.035128
dynamic CVaR: 4.394e9
dynamic resource: 0.284
frac_std: 0.076
Excel Pareto: false
CVaR Pareto: false
```

BC alone learned a lower-resource smoothed approximation to the oracle, but did
not beat the dense static frontier. The best equal-resource static remained
`f0.05_S2` on Excel, and `f0.10_S1` remained the cheap CVaR comparator.

## Warm-Start Verdict

This separates the hypotheses:

- Exploration from scratch was part of the problem, but not the whole problem.
- BC successfully reduces action MSE, so the implementation is not broken.
- The oracle is effectively a discrete action table by regime; the continuous
  PPO policy smooths that table and loses enough sharpness/resource efficiency
  to be dominated.

Track A has a real oracle headroom, but the learnable continuous PPO policy has
not converted it into a win. The next Track A-only repair, if any, should be a
categorical/factorized action policy over the oracle action set, not more
continuous PPO tuning.
