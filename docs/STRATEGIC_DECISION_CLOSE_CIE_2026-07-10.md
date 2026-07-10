# Strategic decision — close Paper 1 for C&IE, shelve adaptive-advantage hunt

Date: 2026-07-10
Status: **decision recorded, pending final editorial compression**

## Decision

**Option (a): compress and close the pivoted Paper 1 for C&IE → SMPT.**
Do not open Paso 1 (reward alignment / CD metric / env changes) now.

This could be **our paper** (solo-authored).

## Rationale

1. **Two reversions detected by our own protocol form a solid methodological
   contribution:**
   - Canonical PPO loses to the same-contract static constant (−18e-6, CI
     entirely negative, 2/60 tapes positive).
   - The fresh joint bundle (52-dim v7, clean protocol) does *not* replicate
     its post-hoc +9.6e-6 advantage on virgin tapes: it shrinks to +6.2e-6
     with CI crossing zero (47/60 tapes, not ≥50).

   Framed contribution: *"an adversarial benchmark detects apparent RL gains
   that do not survive a same-contract comparator nor a held-out
   replication."*

2. **Opening Paso 1 now is scientifically premature and strategically
   risky.** After two failed gates, hunting for a configuration that makes
   RL win is p-hacking — even if pre-registered, because the decision of
   *what* to pre-register is already contaminated by knowing we lost.
   A reviewer would call this metric shopping.

3. **The pivoted manuscript is already publishable.** Title: *"When Apparent
   Reinforcement-Learning Gains Depend on the Static Frontier."* It does
   not depend on RL winning anything. The benchmark-audit + comparator-
   design story stands on the two detected reversions alone.

## Future work (explicitly deferred, not started)

If the question *"under what problem structure does adaptive control add
value?"* is revisited later, it would be:

- A **separate Paper 2** with its own pre-registration.
- A **modified environment** (e.g., dispatch cost, surge inertia,
  perishability, partially-predictable risks) — changing the question, not
  rescuing the current one.
- **Same-contract re-optimization** of the static comparator under the
  changed conditions (so we do not repeat the misaligned-comparator error).
- Run **in parallel or after** the C&IE/SMPT submit, never before.

## Evidence trail

- Same-contract challenge verdict:
  `docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md`
- Clean-joint replication protocol + result:
  `docs/TRACK_B_CLEAN_REPLICATION_PROTOCOL_2026-07-10.md`,
  `outputs/experiments/track_b_clean_replication_2026-07-10/summary.json`
- Both test batteries (400001–400060, 500061–500120) are now burned for
  any future confirmatory use.

## State of the science (final, honest)

| Claim | Verdict |
|---|---|
| PPO canonical beats static full-contract | **Refuted** (−18e-6, CI negative) |
| PPO joint clean beats static full-contract | **No detected difference** (+6.2e-6, CI crosses zero) |
| Dispatch increment (within-learner) | Retained but tiny (+21e-6), not promotable as adaptive advantage |
| Historical headline (+486e-6) | Real vs restricted frontier; artifact as adaptive claim |

Never write "PPO retains a small advantage" — it does not, at current
precision.
