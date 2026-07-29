# Step 3 of Garrido's design: structured control on the expanded contract

**Status:** `DEVELOPMENT_SCREEN_NO_CLAIM`. No confirmation universe opened, no learner
trained. Runner `scripts/run_expanded_contract_comparators.py`, result
`results/expanded_contract_comparators/result.json`.

Garrido, 2026-07-28: *"Baseline, modelo de Garrido. Luego tu corres el MPC con los datos
originales del modelo. Luego corres el MPC con mas variables. Y vemos como se comporta."*
Steps one and two existed. This is step three, and it was the missing link: the neural
residual is defined against the best structured controller, so until a structured
controller ran on the expanded contract there was no denominator for any architectural
claim.

He also named the incumbent by name — *"demand driven material requirement planning ...
quizas uno puede crackear esa vaina y meter una red neuronal"* — so DDMRP is a required
comparator, implemented as the actual method rather than a fixed level wearing its name.

## Design

Three arms with **identical decision rights** (the three strategic buffer targets),
**identical information** (the simulator's own state at each epoch) and an **identical
admissible set** (the Table 6.16 ladder plus a zero floor). Monthly decision cadence over
52 weeks, 6 tapes, both risk families at their fully-increased corner so the contrast is
between controllers rather than between risk patterns.

- `static_I*` — one rung held throughout. The incumbent.
- `ddmrp_dynamic` — decoupled lead time, rolling average daily usage, red/yellow/green
  zones with lead-time and variability factors, net flow position, dynamic buffer
  adjustment. Writes top-of-green.
- `mpc_receding_horizon` — replans each epoch against the **real DES**. The Program Q
  transducer is invalid here: `extract_full_des_skeleton` freezes "only action-independent
  events" and buffer targets change exactly those, so every candidate is evaluated by
  re-simulating the committed prefix under common random numbers.

The static incumbent sits inside the MPC's candidate set at every epoch. An MPC that
loses to it has failed to find a solution it could reach, which is a search failure and is
reported as one.

## Result

| family | arm | ReT | full ledger | ret_thesis | fill | strategic injected | Δ vs best static | tapes > 0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1r | **static_I168** | **0.004402** | 0.004386 | 0.004386 | 0.99652 | 400,834 | — | — |
| R1r | static_I336 | 0.004380 | 0.004365 | 0.004365 | 0.99658 | 533,767 | −0.000022 | 3/6 |
| R1r | static_I672 | 0.004344 | 0.004330 | 0.004330 | 0.99664 | 809,342 | −0.000058 | 3/6 |
| R1r | mpc_receding_horizon | 0.004339 | 0.004321 | 0.004321 | 0.99599 | 532,514 | −0.000063 | 2/6 |
| R1r | static_I1344 | 0.004316 | 0.004302 | 0.004302 | 0.99664 | 1,118,622 | −0.000085 | 2/6 |
| R1r | ddmrp_dynamic | 0.004237 | 0.004223 | 0.004223 | **0.99664** | **10,288,359** | −0.000165 | 1/6 |
| R1r | static_I0 | 0.003518 | 0.002740 | 0.002740 | 0.77190 | 0 | −0.000884 | 0/6 |
| R2r | **static_I168** | **0.411032** | 0.243687 | 0.034969 | 0.90576 | 299,797 | — | — |
| R2r | static_I336 | 0.408450 | 0.246340 | **0.035360** | 0.91087 | 420,424 | −0.002582 | 1/6 |
| R2r | mpc_receding_horizon | 0.407826 | 0.244325 | 0.035139 | 0.90700 | 310,892 | −0.003206 | 1/6 |
| R2r | ddmrp_dynamic | 0.402659 | 0.248228 | 0.035075 | 0.91064 | **9,645,799** | −0.008373 | 1/6 |
| R2r | static_I0 | 0.275743 | 0.165846 | 0.031245 | 0.76812 | 0 | −0.135289 | 0/6 |

**Neither structured controller beats the best fixed posture, in either family.** The best
comparator on this contract is a constant, and it is the *smallest* non-zero rung.

## What this answers, and what it does not

**The expanded contract does not saturate the structured controller. It saturates the
decision.** Having a buffer at all is worth a great deal — `static_I0` loses 0.000884 in
R1r and 0.135289 in R2r — which is consistent with the H2/H3 result that the buffer right
moves ReT by +11% to +25%. But *choosing when to change the level* is worth nothing. The
entire value is in the level, and one fixed level captures it.

This is not merely an MPC search failure, and the admissible set shows why. Above I168 the
static ladder is flat: 0.004402, 0.004380, 0.004241, 0.004344, 0.004316 — a spread of
1.6e-04. **The spread of the admissible set bounds what any adaptive policy over that set
can buy**, and the MPC's shortfall (6.3e-05) sits inside that band. There is nothing to
adapt toward. A larger scenario budget would tighten the MPC's selection noise; it cannot
manufacture headroom that the level ladder does not contain.

**Consequence for the neural question.** The residual for a learner is defined against the
best structured controller. Here that controller is a constant, and no richer controller
reaches it. So this contract offers no residual to capture — independently reaching the
same conclusion as the Q-R1 architecture gate (`neural_premium` −0.021, 0/5 seeds, negative
in all three kappa cells, `NO_GO_NEURAL_BAKEOFF`).

## Two findings that are about the instrument, not the controllers

**DDMRP is dominated on ReT while delivering better service.** It carries the *highest*
fill rate in R1r (0.99664 against the incumbent's 0.99652) and still scores worst among the
non-zero arms. This is the same pattern already documented for shift postures in
`AUTHORITY_LADDER_SCREEN_FINDINGS_2026-07-28.md` §3: ReT does not reward service.
Reporting DDMRP as "beaten" without that caveat would misrepresent it.

**It also burns 26x the strategic material** — 10,288,359 units against the incumbent's
400,834 in R1r, 32x in R2r — for worse ReT. Since the objective prices none of that
material, DDMRP is being rewarded by nothing for the extra stock and still loses, which
means excess buffer is not merely wasteful here but actively harmful to the endpoint. The
MPC is far more restrained (532,514) because it optimises the endpoint directly.

**The metrics disagree about the optimum.** In R2r `ret_excel` prefers I168 (0.411032)
while `ret_thesis` prefers I336/I504 (0.035360 against I168's 0.034969). The arms are not
ranked identically by the two metrics, which is why the panel is reported in full rather
than a single column.

## Limits

Six tapes at one risk corner per family, 52-week horizon, monthly cadence, two MPC
scenarios per candidate. This is a development screen and the per-tape sign counts are
thin. What it establishes is directional and mechanical — the ladder is flat above I168, so
adaptation over that ladder has almost no room — not a confirmed estimate of any delta.
