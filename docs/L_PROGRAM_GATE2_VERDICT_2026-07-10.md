# Program L(e-1) Gate 2 verdict — terminal headroom stop

**Verdict:** `STOP_NO_DEPLOYABLE_ADAPTIVE_HEADROOM`  
**PPO trained:** no scientific PPO run; only bounded software smokes in `/tmp`  
**Virgin tapes opened:** no

## Corrected evidence package

- Gate 0: `outputs/preflight/l_program/gate0_crn_v2.json` — PASS.
- Gate 1: `results/headroom/l_program_gate1_crn_v2_2026-07-10/` — 60
  materialized calibration tapes, 18 static configurations and six observable
  heuristics.
- Gate 2: `results/headroom/l_program_gate2_crn_v2_2026-07-10/` — 100 states
  per buffer, three actions per state, four-week branches, GroupKFold by tape
  and a depth-four observable tree.

The earlier unsuffixed Gate 1/2 artifacts are retracted. Equal seeds had been
used while risks were active during policy-dependent warm-up, so the relative
shock calendars were not identical. The corrected lane first generates every
risk calendar from a risk-free B0/S1 warm-up, expresses events relative to the
treatment boundary, hashes the tape, and replays it after each policy's own
warm-up with endogenous risk generators disabled.

## Gate 1

Integrity checks:

- 1,440 rows = 60 tapes x (18 statics + 6 heuristics);
- 60 unique materialized tape hashes;
- every row belongs to `calibration`;
- no empty risk calendar;
- panel SHA256:
  `78644352dfcfe27ab24f90a0e4e3c1a68271d3131e583c9aa8ef73b26032bd27`.

The unconstrained best static was `I1344/S2`:

- mean Excel ReT: `0.1100055698`;
- mean service-loss AUC: `66,890,946.94` ration-hours;
- mean extra shift-hours: `17,472`.

Within non-extreme buffers I168-I672, S2 was also the best-ReT fixed shift.
B0 selected S3; I1344 selected S2. The fact that all positive buffers have the
same two headline means is itself a boundary signal: at this horizon, the
strategic reserves are not differentially engaged often enough to change the
aggregate comparator.

## Gate 2

Prefix observations were asserted exactly equal before all 1,800 branches.
Across the 600 sampled states:

- states with an action effect on four-week late-backlog AUC: `0/600`;
- maximum loss spread across S1/S2/S3: `0`;
- states with an action effect on final Excel ReT: `0/600`;
- maximum Excel-ReT spread: `0`.

The actions did alter accounted capacity. In a direct falsification check, the
same branch used 1,176, 1,344, and 1,512 shift-hours under S1/S2/S3 requests,
respectively, while production, service loss and ReT remained identical. Thus
the zero is not caused by a disconnected action. The requested capacity does
not reach the service outcomes within the preregistered four-week window; the
tested states are upstream/material constrained.

The depth-four tree consequently learned only a tie-break toward S1. No
non-extreme buffer met any meaningful action-variation or service-improvement
criterion, and none met the joint promotion bar.

Branch-row SHA256:
`47d04e30bb8835cda311417f9d7e58ae7500d39c11b2cc464b3998c05005b1f3`.

## Scientific decision

Gate 2 is terminal under the preregistration. Do not run Gate 3, tune PPO, widen
the action space, lengthen the branch horizon, or replace the null with an
architecture comparison inside Program L v1. The permitted conclusion is:

> In the Garrido-grounded fixed-buffer/weekly-shift contract, the observable
> four-week branching test finds no deployable adaptive headroom. Therefore the
> environment cannot identify retained learning on this decision surface, and
> powered PPO is not scientifically authorized.

This is a boundary result about the tested contract. It does not retract the
separate Track B evidence, whose downstream action surface is different.
