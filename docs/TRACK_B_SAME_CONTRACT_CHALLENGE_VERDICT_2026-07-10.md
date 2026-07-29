# Track B same-contract challenge — final verdict (2026-07-10)

## Executive verdict

**The stop rule fails. Retire the claim that PPO has a Track B adaptive
advantage over a strong same-contract static policy, and do not lead with the
downstream-bottleneck mechanism at IJPR. Pivot Paper 1 to a benchmark and
decision-contract-design contribution, with C&IE primary and SMPT backup.**

The failure is decisive on the primary comparison. On 60 untouched tapes
(`400001–400060`), the calibration-only optimized constant full-contract
policy attains Excel ReT `0.005906366`, versus `0.005888317` for the ten
canonical PPO checkpoints. PPO minus static is `−0.000018049`, with two-way
checkpoint × tape CI95 `[−0.000028615, −0.000008087]`; only 2/10 checkpoint
means and 2/60 tape means are positive. CVaR05 agrees: `−0.000101921`, CI95
`[−0.000163278, −0.000042639]`.

The second contrast is positive but cannot rescue the headline. Fresh
factorial joint PPO exceeds the learned upstream/shift arm anchored at fixed
Op10 `2.0x`, Op12 `1.5x` by `+0.000021106`, CI95
`[+0.000007015, +0.000034963]`; 4/5 seed means and 58/60 tape means are
positive. Thus dynamic dispatch access has a small within-learner increment,
but it does not establish adaptive superiority over the stronger constant
full-contract policy.

## Frozen design

- Protocol: `docs/TRACK_B_SAME_CONTRACT_CHALLENGE_PROTOCOL_2026-07-10.md`.
- Global search: 128 scrambled Sobol candidates plus the prespecified old
  frontier winner, evaluated on calibration tapes `300001–300012`.
- One refinement only: 136 deduplicated candidates around the eight screen
  leaders, evaluated on `300001–300024`.
- Frozen test: 60 new tapes, `400001–400060`.
- Primary metric: Garrido/Excel order-mean ReT (`ret_excel`).
- Inference: two-way bootstrap resampling checkpoint/training-seed and tape.

The selected constant action signals are
`(-0.231787, 0.419829, -0.701838, -0.823369, -0.787802, 0, 0.552683,
0.450964)`, with S2. Under the Track B decoder these correspond approximately
to Op3 quantity `1.076x`, Op9 quantity `1.565x`, Op3 ROP `0.724x`, Op9 ROP
`0.632x`, Op5 buffer `0.606x`, Op10 dispatch `1.665x`, and Op12 dispatch
`1.588x`. Its calibration Excel ReT was `0.005944627`; it was frozen before
the final tapes were opened.

## Primary results

| Contrast (Excel ReT) | Mean | Two-way CI95 | Seed direction | Tape direction |
|---|---:|---:|---:|---:|
| Canonical joint PPO − best full-contract static | −0.000018049 | [−0.000028615, −0.000008087] | 2/10 positive | 2/60 positive |
| Factorial joint PPO − learned upstream/shift with best fixed dispatch | +0.000021106 | [+0.000007015, +0.000034963] | 4/5 positive | 58/60 positive |

Absolute Excel ReT means: full-contract static `0.005906366`; canonical PPO
`0.005888317`; fresh factorial joint PPO `0.005915956`; anchored learned arm
`0.005894850`.

## Claim disposition

1. **Retracted as a general Track B advantage:** PPO beats strong static
   control on the full Track B decision contract.
2. **Bounded historical result:** PPO decisively beats the 147-cell static
   family that varies shift and dispatch while fixing upstream controls. This
   demonstrates comparator-family sensitivity, not adaptive superiority.
3. **Retained secondary finding:** within freshly trained learned policies,
   allowing dispatch to vary dynamically adds a small positive increment over
   fixing dispatch at `2.0x/1.5x`. Do not call this proof of bottleneck value
   over static control.
4. **Retained audit contribution:** the sequence from restricted frontier to
   same-contract challenge shows why RL studies must align comparator decision
   rights with learner decision rights before interpreting a gain as learning.

## Publication consequence

The IJPR-first ladder is no longer justified by the planned adaptive-
bottleneck mechanism. The defensible primary story is an adversarial benchmark
audit: a large and highly stable apparent PPO gain disappears—and reverses—
when the static comparator is optimized over the same decision contract. Lead
with **C&IE**, then **SMPT**. Paper 2 remains paused until this reframing is
submitted.

## Artifacts

- Search: `outputs/experiments/track_b_static_contract_search_2026-07-10/`
- Anchored training: `outputs/experiments/track_b_factorial_upstream_shift_best_dispatch_2026-07-10/`
- Final 1,260-row crossed matrix and summary:
  `outputs/experiments/track_b_same_contract_challenge_2026-07-10/`
- Runners: `scripts/run_track_b_static_contract_search.py`,
  `scripts/run_track_b_contract_factorial.py`, and
  `scripts/evaluate_track_b_same_contract_challenge.py`.
