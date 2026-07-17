# Program O relevant-risk sensitivity v1.1 — preflight verdict

## Verdict

`STOP_V1_1_BEFORE_G2_RARE_RISK_FIXTURE_NOT_POPULATED`

G0 passed on all 12 burned tapes. The reconstructed `belief_mpc__3`
calendars exactly matched the corrective-validation record and direct replays
of both policy and frozen static comparator matched every custodied metric to a
maximum absolute difference of `2.22e-16`.

G1 was then executed at the preregistered thesis frequency `phi=1`, with risks
activated only after the common neutral Program O warm-up:

| Risk | Events | Observed operations | Gate |
|---|---:|---|---|
| R11 | 192 | 5, 6 | PASS |
| R14 | 672 | 7 | PASS |
| R21 | 0 | none | FAIL |
| R22 | 5 | 8, 10 | FAIL; frozen map also contains 4, 12 |
| R24 | 48 | 13 | PASS |
| R3 | 0 | none | PASS frozen |

R24 created contingent demand under both already-frozen product labels
(`P_C=13`, `P_H=11`). No foreign risk fired and no populated event used an
operation outside its frozen map.

## Interpretation

The failed G1 is a preflight-coverage failure, not evidence that R21/R22 are
physically irrelevant. An eight-week treatment and 12 tapes have little power
to populate every target of rare uniform-window risks. That limitation was
knowable from the occurrence windows, but v1.1 nevertheless froze exact
coverage at `phi=1` and explicitly prohibited an escalated liveness fixture
from rescuing a failed G1.

Accordingly, G2 was not executed or read. Changing the coverage rule after
observing these counts would be post-result gate repair. A future contract may
separate deterministic implementation liveness from thesis-rate incidence,
but it must be prospective and cannot select or modify Program O-R.

The historical Program O STOP, the Program O-R learner run, and all other
program verdicts remain unchanged.

