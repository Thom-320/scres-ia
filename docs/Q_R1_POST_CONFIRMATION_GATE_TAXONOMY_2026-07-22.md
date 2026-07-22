# Q-R1 post-confirmation gate taxonomy (2026-07-22)

## Immutable historical result

The frozen successor contract remains adjudicated as
`STOP_REPAIRED_Q_R1_NO_RETAINED_INFORMATION_PASS`. No threshold, arm, seed,
metric, or verdict is changed after opening roots 7572001--7572032.

The prospective result nevertheless separates three scientifically different
questions that the composite verdict had combined:

| Claim | Evidence in the frozen run | Current status |
|---|---|---|
| Mean canonical cold-start ReT | retained-reset = +0.06563; clustered LCB95 = +0.05244 | strong prospective signal, not a contract PASS |
| History mechanism | dose .90 > .75 > iid; iid exactly zero; shuffled and wrong non-positive; delayed +0.00591 | unresolved under the frozen composite gate |
| Joint service safety | worst-product LCB95 = -0.03645 vs frozen margin -0.02 | failed |

## Why delayed is not a null placebo

`delayed_posterior` uses the retained prior from the preceding campaign. Under
a campaign process with persistence 0.90, that prior remains causally
predictive of the current campaign. Its expected effect is therefore not zero.
It must be treated prospectively as an age-of-memory ablation, with a monotone
decay estimand, rather than as a shuffled-history placebo. This semantic defect
does not retroactively alter the frozen STOP.

Valid null controls for a successor contract must break the relationship
between history and the current sequence: cross-root block shuffle, iid
campaigns, and wrong/permuted history. Their construction and expected signs
must be frozen before new roots are assigned.

## Garrido ReT versus deployment safety

Canonical ReT is the domain construct and remains unmodified. Worst-product,
unresolved demand, and ledger completeness remain mandatory disclosures.
Whether worst-product is a gate for the domain claim or only for a `safe`
deployment label requires written, neutral validation from Garrido.

Until that response exists, future adjudication must report two verdicts:

1. `RETAINED_RET_EFFECT`: canonical mean ReT effect with causal controls.
2. `RETAINED_SAFE_EFFECT`: the ReT effect plus the frozen service constraints.

Failure of the second must never erase the first numerically; success of the
first must never be described as operational safety.

The written question to Garrido must remain neutral:

> In the original ReT construct, was operational acceptance based on mean
> canonical ReT alone? Independently, did the intended doctrine require a
> minimum service floor for every product or campaign, or a limit on unresolved
> demand? If so, how should that floor or limit be defined before a policy is
> evaluated?

No candidate numerical margin is included in the question.  A domain answer
may govern a new contract, but cannot change the frozen adjudication.

## Immediate successor route

1. Build comparator challenge v2 on burned development histories.
2. Estimate the ReT--service Pareto frontier without learner returns.
3. Freeze the strongest tested deployable structured controller, never call it
   an optimal MPC.
4. Quantify how much of the +0.06563 signal remains under service constraints.
5. Only a positive, powered residual against that controller may authorize
   learned terminal value or belief correction.
