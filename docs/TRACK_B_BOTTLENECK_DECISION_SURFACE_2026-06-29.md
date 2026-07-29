# Track B Bottleneck / Decision-Surface Audit

Date: 2026-06-29

## Verdict

The strongest corrected claim is:

> The binding resilience lever is downstream dispatch capacity and recovery, not Track A manufacturing shifts alone. Op12 is the final service valve and is highly sensitive, but the bottleneck is the downstream chain as a coupled Op9 -> Op10 -> Op12 system, not Op12 in isolation.

This supports opening Track B as the next high-probability raw-ReT lane. It does not yet justify deleting shifts entirely; shifts should be kept as a secondary/costed support lever until a dense Track B frontier proves they are redundant under the chosen campaign.

## What Is Confirmed

### Thesis and code

Garrido's thesis specifies downstream distribution as fixed daily dispatch:

- Op9 Supply Battalion dispatches approximately 2,400-2,600 rations daily.
- Op10 LOC SB-to-CSSU transport uses approximately 2,400-2,600 rations daily.
- Op11 CSSU delivery/distribution uses approximately 2,400-2,600 rations daily.
- Op12 LOC CSSU-to-theatre final delivery uses approximately 2,400-2,600 rations daily.
- Demand is also approximately 2,400-2,600 rations/day.

The repo mirrors this in `supply_chain/config.py`: Op9/Op10/Op11/Op12 have `q=(2400,2600)` and daily cadence. S1 manufacturing capacity is approximately balanced with demand/distribution; S2/S3 scale upstream production but do not automatically scale last-mile distribution.

### Existing Track B evidence

Historical Track B briefs already show the downstream lever has authority:

- Full Track B PPO reached near-perfect flow/ReT in prior benchmark settings.
- Downstream-only control, with shifts fixed, also reached near-perfect flow/ReT.
- Shift-only control lost badly versus static.
- Learned Track B policies did not simply max everything; they used moderate downstream capacity and mostly low shifts.

Those results are strong but should be re-run on the current branch before any paper claim.

### Quick sensitivity probe from 2026-06-29

Output: `outputs/experiments/track_b_downstream_sensitivity_2026-06-29/`

Cell: Track B environment, increased risk, h52, 2 seeds, direct action probes.

Key rows:

| policy | Excel ReT | flow_fill | lost_rate | backlog | comment |
|---|---:|---:|---:|---:|---|
| S1/base downstream | 0.002231 | 0.729 | 0.142 | 151k | baseline |
| Op12 x1.1 | 0.002409 | 0.755 | 0.119 | 133k | Op12 +10% gives about +8% ReT in this cell |
| Op10 x2.0 | 0.002399 | 0.746 | 0.128 | 140k | also positive |
| all downstream x1.5 | 0.002399 | 0.745 | 0.128 | 141k | positive but not additive |
| S2/base downstream | 0.002744 | 0.880 | 0.001 | 111k | manufacturing can still matter in this cell |
| S3/base downstream | 0.002696 | 0.840 | 0.035 | 125k | S3 worse than S2 |

Interpretation:

- Op12 is highly sensitive and directly tied to served orders.
- Op12 alone is not proven to be the only bottleneck.
- Downstream stages interact; Op9/Op10/Op12 should be controlled together.
- Shifts are not the primary Track B lever, but removing them outright is premature.

## Corrected Track B Variables From Scratch

If the goal is to maximize raw Excel ReT with the least thesis distortion, the candidate control variables are:

1. Op12 dispatch multiplier/cadence: final delivery to theatre; most direct service valve.
2. Op10 dispatch multiplier/cadence: upstream LOC link feeding CSSU; vulnerable under R22-style transport disruption.
3. Op9 dispatch multiplier/cadence: feeds the downstream pipeline from Supply Battalion.
4. Backorder service policy: queue priority, expedite fraction, or max queue depth before forced recovery. This directly targets lost orders and recovery time.
5. Op9/theatre-adjacent safety stock: buffer at or just before the binding downstream chain.

Keep assembly shift as an optional sixth/costed support variable for now. If forced to use exactly five variables, fix shift to S2 or allow it only as an exogenous scenario factor. Do not assume S3 is always useful.

## How To Amplify Headroom

The best way to create honest headroom is not more Track A PPO. It is to design a Track B gate where the static optimum changes because the downstream bottleneck changes:

- Campaigns: R22/R23/R24-heavy and R13+R24 mixed stress.
- Intensities: phi in {1, 2, 4, 8, 16}; include demand-surge and LOC-disruption regimes.
- Static gate first: dense grid over Op9/Op10/Op12 multipliers, queue policy, Op9 buffer, and shift.
- Promote to PPO only if oracle-by-regime beats best constant by a positive CI.

Candidate amplification levers:

1. Controlled R13+R24 or R22+R24 campaign: likely largest headroom because it creates competing upstream/downstream pressure.
2. Dense Track B static gate: avoids another coarse-frontier artifact.
3. Longer training only after the static gate passes; more timesteps cannot create headroom if a constant policy already dominates.

## Claim Boundary

Do not claim yet:

- "Op12 is the only bottleneck."
- "10% Op12 increase always gives 10% ReT."
- "No shifts should ever be used."

Claim safely:

- The thesis-faithful Track A variables are frontier-limited.
- Downstream dispatch is the next natural decision frontier because it touches the binding service path.
- Track B should control Op9/Op10/Op12 jointly, with Op12 as the most direct service valve.

## Live Follow-Up Runs

Started on 2026-06-29 PM:

- Static Track B DOE, common downstream multiplier, h104, 3 seeds:
  `outputs/experiments/track_b_gain_2026-06-29/doe_static_h104/`
- Track B ablation, h104, seed 1, 50k timesteps:
  `outputs/experiments/track_b_gain_2026-06-29/ablation_50k_seed1_h104/`
- Queued Track B joint confirmation, h104, 3 seeds, 100k timesteps:
  `outputs/experiments/track_b_gain_2026-06-29/joint_confirm_100k_3seed_h104/`

Important implementation correction: `track_b_v1` is currently 8D
(`Track A 6D + op10 + op12`). The old ablation wrapper still used 7D indices;
it was corrected before running the 2026-06-29 ablation:

- `shift_only` now freezes dims 6-7 (Op10/Op12).
- `downstream_only` now freezes dim 5 (shift).

First partial result from `joint` Track B, seed 1, 50k:

| policy | Excel/ReT | flow_fill | assembly hours | comment |
|---|---:|---:|---:|---|
| best static in smoke (`s3_d1.50`) | 0.00508 | not best | 17,472 | coarse static frontier |
| S2 baseline (`s2_d1.00`) | 0.00505 | 0.669 | 11,648 | baseline |
| PPO joint | 0.00551 | 0.922 | 13,440 | raw ReT win vs coarse static |

This is a live positive signal, not a confirmed claim. It still needs:

1. `downstream_only` and `shift_only` ablation completion.
2. 3-seed / 100k joint confirmation.
3. A denser independent Track B static frontier over Op9/Op10/Op12 before any paper claim.
