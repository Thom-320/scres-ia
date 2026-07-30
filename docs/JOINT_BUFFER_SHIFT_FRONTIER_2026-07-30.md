# The joint buffers × shifts frontier — all 648 postures

**Status:** `DEVELOPMENT_SCREEN_NO_CLAIM`. Runner
`scripts/enumerate_joint_buffer_shift_frontier.py`, artifact
`results/joint_frontier/buffer_shift_648_v1/result.json`. Development roots
1,900,001–1,900,012, disjoint from every previous block. Step cadence 672 h,
post-RPj-fix so `ret_excel` is cadence-invariant here.

Closes the gap neither existing run covered: the v2 comparator enumerates all 216 buffer
vectors but pins `shifts = 1`; the five-column panel crosses buffers with shifts over five
hand-picked postures on four tapes. Garrido's expanded contract is buffers **and** shifts,
so the frontier is 6³ × 3 = **648**. Service floors and the comparison set were declared
before evaluation — the set *is* the complete domain, which matters because `κ̇` is
set-relative.

## Did pinning `shifts = 1` cost anything?

**In R1r, no. In R2r, yes — for exactly the two endpoints that matter most.**

| family | metric | joint winner (648) | winner with shifts pinned to 1 | moved? |
|---|---|---|---|---|
| R1r | `R_cobb_douglas` | `168/0/168` \| S1 | `168/0/168` \| S1 | no |
| R1r | `ret_excel` | `0/0/336` \| S1 | `0/0/336` \| S1 | no |
| R1r | `ret_excel_full_ledger` | `0/0/336` \| S1 | `0/0/336` \| S1 | no |
| R1r | `ret_excel_cvar10` | `0/0/336` \| S1 | `0/0/336` \| S1 | no |
| R2r | `R_cobb_douglas` | `0/0/168` \| S1 | `0/0/168` \| S1 | no |
| R2r | `ret_excel` | `0/0/336` \| S1 | `0/0/336` \| S1 | no |
| R2r | **`ret_excel_cvar10`** | **`168/168/168` \| S2** | `0/0/168` \| S1 | **yes** |
| R2r | **`ret_excel_full_ledger`** | **`0/672/168` \| S2** | `0/672/168` \| S1 | **yes** |

So v2's shift pinning was harmless for the primary endpoint everywhere, and harmless
throughout R1r — but in R2r it hid a shift-2 optimum for both the tail metric and the
uncensored ledger. Those are the two endpoints that disagreed with the primary in the
prospective confirmation, so the omission was not in a corner.

**Independent cross-validation:** `0/0/336` is exactly the incumbent v2 selected for R1r
under `ret_excel`, recovered here on a disjoint root block. Two enumerations, different
tapes, same answer.

## What survives about shifts, and what does not

The panel reported that at posture `168/0/168` all three ReT variants preferred shift 3
while `R_cobb_douglas` preferred shift 1, at identical fill. Over the full domain:

**Survives, and much more strongly than before.** `R_cobb_douglas` prefers **shift 1 in
216 of 216 buffer postures, in both families** — unanimous, no exceptions. And the
mechanism holds: at `168/0/168` the fill is identical across all three shift levels
(0.99635 in R1r; 0.92074/0.92108/0.92108 in R2r) while three shifts cost **2.35×** in R1r
and **2.40×** in R2r. Extra capacity buys no aggregate service and is not free.

**Does not survive — retracted.** "All three ReT variants prefer shift 3" is false. At
`168/0/168`:

| metric | R1r prefers | R2r prefers |
|---|---|---|
| `ret_excel` | **S2** | **S2** |
| `ret_excel_full_ledger` | **S2** | **S2** |
| `ret_excel_cvar10` | S3 | S2 |

And across all 216 postures the ReT variants have **no stable shift preference at all**:

| metric | R1r: best shift count | R2r: best shift count |
|---|---|---|
| `ret_excel` | S1 138 · S2 32 · S3 46 | S1 85 · S2 28 · S3 103 |
| `ret_excel_full_ledger` | S1 138 · S2 32 · S3 46 | S1 50 · S2 133 · S3 33 |
| `ret_excel_cvar10` | S1 64 · S2 48 · S3 104 | S1 163 · S2 30 · S3 23 |
| `R_cobb_douglas` | **S1 216** · S2 0 · S3 0 | **S1 216** · S2 0 · S3 0 |

This is a **stronger** indictment of ReT than the claim it replaces. The original reading
was "ReT prefers the expensive option". The correct one is worse: **ReT's ranking over the
capacity dimension is unstable across the buffer domain** — `ret_excel` picks shift 1 in
138 postures and shift 3 in 46 of the same family. A metric whose capacity preference
flips with the buffer vector is not measuring the value of capacity.

Caveat on the comparison: the panel's numbers were taken at 24 h cadence, before the RPj
fix, on four tapes. These are 672 h, post-fix, twelve tapes. The difference could be any of
the three, and this screen does not separate them. What it establishes is what holds now.

## Service

**R1r: 612 of 648 postures clear the declared floors**, and the floor sweep is stable —
identical winner vectors at every cap from 20,000 to unbounded
(`sweep_winner_vectors_stable: true`). Under the constraint the ReT trio moves to
`0/0/1344|S1`, so the constraint does bind at the top.

**R2r: 0 of 648 clear them.** Not a subset effect — the complete joint domain fails. But
the sweep is **not** stable, and the honest statement is the threshold-specific one:

| unresolved-backorder cap | postures passing | distinct winners |
|---|---:|---:|
| 20,000 | **0** | — |
| 40,000 | **0** | — |
| 50,000 | **0** | — |
| 60,000 | 463 | 3 |
| 100,000 | 542 | 4 |
| unbounded | 542 | 4 |

So: **no posture in the complete joint domain meets an unresolved-backorder cap of 50,000
under R2r at full escalation**, and above 60,000 a majority do. 50,000 remains an
exploratory cap, not a validated operational threshold, so this bounds nothing about
deployability — it says the R2r service conclusion is a statement about where the cap is
drawn, and the cliff sits between 50,000 and 60,000.

## Pareto front — (κ↓, fill↑, lost↓) only

Seven non-dominated cells in R1r, six in R2r. Excluded axes: unresolved backorder,
delivered rations, strategic injection, τ. Notable that the R1r front contains
`0/1344/0|S2` — the only shift-2 entry, and a posture that carries buffer *only* at the
assembly-line raw-material node, which no hand-picked set would have included.

## What this establishes, and what it does not

**Establishes:** the joint frontier exists and is enumerated; pinning shifts cost nothing
for the primary endpoint but hid a shift-2 optimum for the tail and uncensored ledger in
R2r; `R_cobb_douglas` has a unanimous shift preference and the ReT variants have none;
no posture in the complete domain clears a 50,000 unresolved-backorder cap under R2r.

**Does not establish:** anything about controllers — this is a static enumeration, and
adaptive value over the joint domain remains unmeasured. Nothing about `H_obs`. Nothing
about a learner. Twelve development roots per family, one risk corner each, means over
roots without paired intervals. `κ̇` is normalised over this 648-member set and no number
here is comparable to any table with a different set. Unit costs are `c = 1`, Garrido's
own §3.1 assumption, not costs calibrated for this DES. Everything is conditioned on
`GARRIDO_FULFILLMENT_DELAY_HOURS = 54`.
