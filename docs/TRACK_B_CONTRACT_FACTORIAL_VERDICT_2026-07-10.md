# Corrected Decision-Contract Factorial — Verdict (2026-07-10)

> **Superseded as the Paper 1 promotion gate.** The later same-contract
> challenge (`docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md`)
> shows that a calibration-only constant policy spanning the full Track B
> contract exceeds the canonical ten-checkpoint PPO mean on 60 new tapes.
> The factorial below remains valid for its stated restricted comparator and
> nested learned-policy contrasts, but it no longer supports a general PPO
> advantage or an IJPR-leading bottleneck claim.

**Headline: the pre-registered dispatch-access gate PASSES — adding Op10/Op12 dispatch
authority to an otherwise identical contract yields an identified positive held-out
contrast (+0.000092, two-way CI95 [+0.000077, +0.000108], 5/5 seeds). But the identified
decomposition re-proportions the mechanism story: most of the Track B gain over the dense
static frontier comes from closed-loop control of the upstream/shift dims that the static
family holds fixed (~82%), with dispatch access contributing a real but modest increment
(~18%), and dispatch authority alone adding nothing over the dense static frontier that
already optimizes exactly that subspace.**

## Design (fixes the two verified E4 defects)

Runs `track_b_factorial_{joint,upstream_shift,dispatch_only}_2026-07-09`
(`scripts/run_track_b_contract_factorial.py`). Unlike E4:
- arms are true nested contracts: `upstream_shift` freezes dims 6-7 (no dispatch);
  `dispatch_only` freezes dims 0-5 (S2, neutral upstream multipliers); `joint` = full 8D;
- one COMMON comparator (the prespecified S2/2.00/1.50 static) and one COMMON held-out
  tape battery (eval seeds 200001-200024, fresh);
- canonical lr 3e-4 (E4 used 1e-4), 5 seeds x 60k, canonical protocol.

## Results (Excel ReT; two-way = checkpoint x tape cluster bootstrap)

| Arm | mean ReT | vs static (two-way CI95) |
|---|---|---|
| joint (8D) | 0.005925 | +0.000517 [+0.000461, +0.000583] |
| upstream_shift (no dispatch) | 0.005833 | +0.000425 [+0.000372, +0.000488] |
| dispatch_only | 0.005397 | −0.000012 [−0.000028, +0.000004] |
| static (same tapes) | 0.005408 | — |

Nested contrasts (paired per seed x tape):
- **PRIMARY joint − upstream_shift = +0.000092, two-way CI95 [+0.000077, +0.000108],
  seed t-CI [+0.000075, +0.000109], 5/5 seeds positive** → gate PASSES.
- joint − dispatch_only = +0.000529 [+0.000474, +0.000593], 5/5.

Independent replication note: the joint arm (5 freshly trained seeds) reproduces the
canonical headline on fresh tapes (+0.000517 here vs +0.000486 in the 10-checkpoint
crossed evaluation) — a full end-to-end replication of the Track B result.

## Interpretation (what changes in the paper)

1. **The mechanism claim survives in identified form**: dispatch authority has causal,
   positive, seed-consistent value on top of an otherwise identical contract. The
   pre-registered promote rule is met; the title's alignment framing is retained.
2. **The proportions must be corrected everywhere**: E4's pattern reading ("downstream
   dispatch access is the strongest single lever") is inverted by the identified design.
   The dominant share of the advantage over the dense static frontier comes from adaptive
   control of the upstream qty/ROP/Op5/shift dims — precisely the dims the static family
   holds at canonical values. Dispatch-only control cannot beat the frontier because the
   147-cell grid already densely optimizes the shift x dispatch subspace; this is a
   coherence check, not a paradox.
3. **Refined design principle** (paper language): adaptive value concentrates where the
   comparator family is static — closed-loop authority over levers that fixed policies
   cannot move — and extending authority to the downstream bottleneck adds a further
   identified increment. "The learner wins by moving what statics cannot, and wins more
   when it can also reach the bottleneck."
4. Not a contradiction of C15 (Track A family retrained on this regime loses): the
   upstream_shift arm uses track_b_v1's upstream parameterization (qty AND ROP
   multipliers + Op5 + shift), a richer upstream contract than Garrido's buffer-fraction
   family. The Track A boundary result stands for the thesis-grounded contract.

## Addendum 2026-07-10 (verified reviewer critique) — anchor caveat + open challenge

Two comparator caveats were verified against the decoder
(`env_experimental_shifts.py`: dims 0-3 and 6-7 map `1.25 + 0.75x`; dim 4 maps
`1.0 + 0.5x`):

1. **The `upstream_shift` arm freezes dispatch at 1.25x/1.25x (signal 0.0), not
   at the best known fixed dispatch (2.0x/1.5x).** The PRIMARY +0.000092 therefore
   identifies "adding adaptive dispatch vs a NEUTRAL fixed dispatch", which
   conflates adaptive-dispatch value with the value of a better fixed dispatch
   level. Fix arm `upstream_shift_best_dispatch` ({6:1.0, 7:1/3}) added to the
   script.
2. **The `dispatch_only` arm freezes upstream qty/ROP at 1.25x (signal 0.0),
   while the 147 statics use base 1.0x** (op5 at 1.0x matches). Its null vs
   static is a near-subspace comparison, not an exact same-subspace one.

**Open pre-registered check (running 2026-07-10):** the same-contract static
challenge (`docs/TRACK_B_SAME_CONTRACT_CHALLENGE_PROTOCOL_2026-07-10.md`):
(i) calibration-only full-contract constant-policy search (tapes 300001-300024),
(ii) `upstream_shift_best_dispatch` 5x60k, (iii) held-out verdict on virgin
tapes 400001-400060 with a frozen stop rule. Until it lands, do not present the
+0.000092 increment as "adaptive dispatch beats the best fixed dispatch" — only
as "dispatch access over an otherwise identical contract with neutral fixed
dispatch". Consolidated reproducible contrasts:
`scripts/aggregate_track_b_factorial.py` →
`outputs/experiments/track_b_factorial_aggregate_2026-07-09/factorial_aggregate_summary.json`
(reproduces every number above exactly).

## Manuscript actions
- §4.4: report this factorial as the identified decomposition (E4 remains as a
  supplementary pattern screen with its disclosed defects).
- Abstract/intro/§5: re-proportion "tied to exposing the downstream recovery bottleneck"
  claims → identified-increment language; remove any "strongest single lever" phrasing.
- Registry C7: Supported (identified) with corrected proportions.
