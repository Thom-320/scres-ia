# Program I branching verdict — terminal pre-policy STOP

Program I executed 4,320 exact replay branches: 60 fresh tapes, four
prefix-balanced states per tape, three actions, three families, and 4/8-week
horizons. Every branch passed raw-state and flow-ledger identity before action;
the maximum mass residual was zero. Seeds 1110001+ were not opened and no
observable policy or learner was trained.

## Results at eight weeks

| Family | Best constant | Action support | Oracle minus constant ReT | Service-loss reduction | Equal resource | 4/8 winner agreement |
|---|---|---|---:|---:|---|---:|
| Production | S1 | S1 42.5%, S2 34.2%, S3 23.3% | 0.0000395, CI95 [0.0000325, 0.0000467] | 0.398% | No | 67.5% |
| Op9 dispatch | SLOW | FAST 11.7%, THESIS 17.1%, SLOW 71.3% | 0.0000113, CI95 [0.0000083, 0.0000147] | 1.212% | No | 80.8% |
| Op10/12 transport | OP10_PRIORITY | BALANCED 10.8%, OP10 71.3%, OP12 17.9% | 0.0000105, CI95 [0.0000074, 0.0000142] | 0.482% | Yes | 80.8% |

The preregistered practical gates were ReT headroom at least 0.01, service-loss
reduction at least 5%, and horizon agreement at least 90%. All three families
failed magnitude and service. Production and dispatch additionally lack equal
resources. The resource-equal transport family also failed horizon stability.

## Interpretation

Program I found local ranking reversals but not deployable adaptive headroom.
The reversals are too small to justify observable-policy fitting and are not a
license to enlarge risk multipliers, change metrics, or tune a learner. This is
a useful distinction: global sensitivity identified controls that move the
response, while exact branching demonstrated that their state-contingent value
is practically negligible.

Verdict:

```text
STOP_PROGRAM_I_NO_RESOURCE_ADJUSTED_ADAPTIVE_HEADROOM
observable_policy_authorized = false
ppo_authorized = false
confirmation_tapes_opened = false
```
