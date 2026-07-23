# Q-R1 comparator v2 — corrective freeze amendment v1 (separate internal adversarial audit)

**Date:** 2026-07-22
**Branch:** `codex/q-r1-comparator-reconciliation`
**Parent tip audited:** `41977bfa` ("Adjudicate high-budget comparator convergence")
**Auditor:** root/PI agent. **This is a separate internal adversarial audit, not an
independent one.** The auditor did not write `program_t_*`, the comparator v2 scripts, or
the convergence receipts, and adversarially re-derived them — but it operates inside the
same project and the same automation as the executor. In the manuscript this must be
described as a separate internal adversarial audit; reserve "independent" for a genuinely
external person or team.
**Claim status:** `BURNED_DEVELOPMENT_NO_CLAIM`. Nothing here is a result.

---

## 0. What this document is — and what it is not

This is a **versioned corrective amendment to an already-registered decision**. It is
**not** a new selection rule, and it must not be cited as one.

The decision to escalate to c256/c1024 with the pure-ReT selector, to keep the original
outcome gate unchanged, to freeze on pass, and to declare convergence unresolved on
fail, was recorded in commit `51957969` **before** the c256/c1024 result existed. The
only thing that went wrong afterwards is **instrumental**: the freeze utility committed
earlier still pointed at the service tie-break candidate that `51957969` itself had just
rejected. This amendment fixes the instrument and records the fix. It changes no
threshold, no objective, no gate, and no candidate family.

Anyone reviewing this file should check §1 first. If §1's quotation does not predate the
result, the rest of this document is void.

---

## 1. Pre-result authority (the load-bearing check)

Commit `51957969` — *"Reject unstable service tie-break comparator"* — is the **immediate
parent** of `41977bfa` and is dated **2026-07-22 13:43:09 -0500**. The c256/c1024
convergence receipt it authorizes was produced at **2026-07-22T20:14:48Z** and committed
in `41977bfa` at **15:16:00 -0500**. The authority therefore precedes the result by ~1h33m
of wall clock and by exactly one commit.

`51957969` added, verbatim, to `docs/Q_R1_COMPARATOR_V2_BURNED_PREFLIGHT_2026-07-22.md`:

> 5. Return to the pure-ReT selector and run the already specified global
>    c256/c1024 convergence check.  This is a budget escalation, not a changed
>    outcome gate.  If it fails, comparator convergence remains unresolved.
> 6. If and only if the original convergence thresholds pass, freeze that
>    universal comparator, expand its raw retained/reset rows, and run the
>    clustered prospective power audit with SESOI +0.01.
> 7. M1/M2/M4 require a deployable residual against that frozen comparator;
>    oracle knowledge remains a clairvoyant upper bound, never learned value.

All four elements of the freeze route are present pre-result:
(i) pure-ReT selector, (ii) gates unchanged, (iii) freeze iff pass, (iv) unresolved iff fail.

The same commit recorded *why* the service tie-break was abandoned — instability, not
performance:

> This audit failed: c64/c256 agreed on 80/96 first actions (83.33%), while
> mean and q95 value errors remained only 0.000383 and 0.001014.  The
> service tie-breaker therefore increases numerical ranking instability and
> is not eligible for freeze.

Verification:

```bash
git merge-base --is-ancestor 51957969 41977bfa && echo PRE-RESULT
git show 51957969 -- docs/Q_R1_COMPARATOR_V2_BURNED_PREFLIGHT_2026-07-22.md
```

---

## 2. The instrument defect

`scripts/freeze_q_r1_comparator_v2.py` (committed before `51957969`) encodes the
**rejected** candidate. It is **preserved unmodified** as evidence of the error; it is not
rewritten and must not be deleted.

| Line | Constant | Encodes | Status after `51957969` |
|---|---|---|---|
| 15 | `EXPECTED_TOLERANCE = 0.002` | service indifference band | rejected candidate |
| 16 | `EXPECTED_TIE_BREAKER = "service"` | service tie-break | rejected candidate |
| 54 | `"conditional_paths": 64` | **hardcoded** c64 | failed convergence (§3) |
| 14 | `EXPECTED_SIGNATURE = [4,"scenario",0.0,"expected"]` | no particle count | see below |

Two distinct failure modes:

1. **It cannot execute the authorized freeze.** Fed the c256/c1024 receipt
   (`tolerance = 0.0`, `tie_breaker = "legacy"`), it raises `ValueError` at lines 30–33.
2. **Worse: if those two guards were naively relaxed, it emits a mislabelled contract.**
   The receipt's `signature` field is `[4, "scenario", 0.0, "expected"]` and carries **no
   particle count**, so the c256 row matches `EXPECTED_SIGNATURE`, while line 54 stamps
   `"conditional_paths": 64` into the emitted config.

   This was **executed, not assumed**. Relaxing only `EXPECTED_TOLERANCE → 0.0` and
   `EXPECTED_TIE_BREAKER → "legacy"` and feeding the genuine c256/c1024 receipt produces:

   ```
   config_id emitted : qr1_v2_scenario_h4_c256_wf0.00_unone_expected_tol0.0000_legacy
   conditional_paths : 64          <-- receipt is c256
   ```

   an internally inconsistent contract whose id says c256 and whose config says c64. This
   is a silent-mislabelling hazard, not merely a wrong pointer, and is the reason the
   amended instrument matches on the **full config id** and **derives** the particle count
   rather than hardcoding it.

The blocker was correctly self-reported in
`results/q_r1/comparator_v2_c256_c1024_v1/convergence_adjudication.json`:

> `"freeze_blocker": "prewritten freeze utility targets failed c64 service-tie candidate rather than predeclared pure-ReT c256 candidate"`

---

## 3. Independent audit findings

### Finding A — the c64 rejection is a **coverage** result, and must be stated as one

This is the one place where a reader can be honestly misled, because two committed
artifacts show the *same config pair* with opposite verdicts:

| Artifact | pair | roots | N states | agreement | pass |
|---|---|---|---|---|---|
| `preflight_p64_p256_convergence_v1` | c64 vs c256 | 7570801–04 | 16 | 1.00000 | **True** |
| `pareto_h4_c64_v1` | c64 vs c256 | 7570801–04 | 16 | 1.00000 | **True** |
| `expansion_v1` | c64 vs c256 | 7570801–24 | **96** | **0.90625** | **False** |
| `c256_c1024_v1` | **c256 vs c1024** | 7570801–24 | **96** | **0.96875** | **True** |

There is no contradiction: c64's pass was measured on **16 decision states over 4 roots**
and did not survive expansion to **96 states over 24 roots** (87/96 agreement; 9
disagreements). The decisive point for this amendment is that the **c256 pass is measured
on the identical expanded coverage** — 96 states, same 24 roots, same four root blocks.
The comparison is apples-to-apples; the c64 pass is superseded by strictly more evidence,
not overruled by a different standard.

Any citation of `pareto_h4_c64_v1: convergence_pass = true` without this table is
misleading and should be treated as an error.

### Finding B — the full tested ladder, and the honest scope of "smallest"

| config | vs | roots | N | agreement | pass |
|---|---|---|---|---|---|
| `scenario_h4_c4_wf0.00` | c16 | 4 | 16 | 0.81250 | False |
| `scenario_h4_c16_wf0.00` | c64 | 4 | 16 | 0.93750 | False |
| `scenario_h4_c64_wf0.00` | c256 | 24 | 96 | 0.90625 | False |
| **`scenario_h4_c256_wf0.00`** | **c1024** | **24** | **96** | **0.96875** | **True** |

**c128 was never tested.** The frozen object is therefore the smallest **tested** budget in
the ladder {4, 16, 64, 256} that passes under expanded coverage — not the smallest
convergent budget in an absolute sense. This is a disclosure, not a defect: nothing
downstream depends on minimality, and a lower convergent budget would only reduce compute.

### Finding C — the service/constraint-aware family has **no** convergent budget at all

| config | vs | roots | N | agreement | pass |
|---|---|---|---|---|---|
| `constraint_aware_h4_c16_wf0.70` | c64 | 4 | 16 | 0.93750 | False |
| `constraint_aware_h4_c64_wf0.70` | c256 | 24 | 96 | 0.90625 | False |
| `scenario_h4_c64_wf0.00_tol0.0020_service` | c256 | 24 | 96 | 0.83333 | False |
| `constraint_aware_h3/h4_c4_wf0.80` | c16 | 4 | 16 | 0.00000 | False (`inf` value error) |
| `constraint_aware_h4_*_wf0.70` | **c256 vs c1024** | — | — | **never run** | — |

Consequence: **`RETAINED_SAFE_EFFECT` cannot be frozen today**, and the reason is *missing
convergence evidence*, not a performance comparison. That family would need its own
c256/c1024 escalation before any deployment-safety claim. Recording this now prevents a
later reading in which the safe-effect family looks quietly dropped.

Note also that the `wf0.80` floor produces `inf` value error and 0.0 agreement — the
constraint is *over*-binding there, i.e. infeasible. This is separate from the standing
observation that the `wf0.70` floor was **inactive** in the burned Pareto
(`constraint_aware` output identical to `scenario`).

### Finding D — zero performance-based selection, verified field-by-field

Provenance flags on the c256/c1024 merged receipt and its adjudication:

| field | value |
|---|---|
| `selection_performed` | `false` |
| `learner_return_used` | `false` |
| `retained_minus_reset_used_for_selection` | `false` |
| `freeze_executed` (pre-amendment) | `false` |
| `pareto_executed` (pre-amendment) | `false` |
| `power_audit_executed` | `false` |
| `learner_authorized` | `false` |
| `claim_status` | `BURNED_DEVELOPMENT_NO_CLAIM` |

The selection criterion at every step of the ladder was **numerical convergence of the
planner against a higher-budget reference** — never realized ReT, never retained−reset,
never a learner return. The rejected candidate (service tie-break) was rejected at
83.33% agreement while its *value* errors were small (mean 0.000383, q95 0.001014):
i.e. it was discarded despite being numerically close in value, purely for ranking
instability. That is the signature of a validity criterion, not a performance criterion.

---

## 4. What is frozen

Exactly the object `51957969` step 6 authorizes, with every field derived from the
receipt rather than asserted:

| field | value |
|---|---|
| horizon | **4** |
| mode | **scenario** (pure ReT) |
| conditional paths | **256** |
| worst-product floor | **0.0** |
| max unresolved orders | `null` |
| tail alpha | 0.10 |
| service statistic | `expected` |
| value-indifference tolerance | **0.0** |
| tie-breaker | **legacy** |
| config id | `qr1_v2_scenario_h4_c256_wf0.00_unone_expected_tol0.0000_legacy` |
| primary objective | `early_ret_complete_cohort` |
| scientific role | strongest **tested** universal structured comparator — **not** an optimality claim |

**Convergence receipt (hashed):**

- path: `results/q_r1/comparator_v2_c256_c1024_v1/convergence_merged/result.json`
- sha256: `15d8d00b9fd4b9647724657ba5ff08207eac4a9f822b49bed80ebb916b7b9941`
- roots: `7570801–7570824` (blocks 01–06, 07–12, 13–18, 19–24), 96 comparable arm states

**Gate — unchanged from the original, all four pass:**

| criterion | threshold | observed | pass |
|---|---|---|---|
| first-action agreement | ≥ 0.95 | **0.96875** | ✅ |
| mean abs planning value error | ≤ 0.005 | **0.00010953** | ✅ |
| q95 abs planning value error | ≤ 0.01 | **0.00024667** | ✅ |
| abstentions (low / high) | 0 | **0 / 0** | ✅ |

**Secondary disclosures — corrected in v1_1.** The v1 list was inherited verbatim from the
superseded utility and carried two defects:

- `ret_total` **does not exist**. `evaluate_calendar` emits `ret_full`; the only `ret_total`
  in the tree is an unrelated accumulator in `scripts/benchmark_ret_ablation_static.py`. A
  disclosure named after a nonexistent key is silently dropped by any table builder.
- Renaming alone would have been worse than the bug. On this evaluation path **both**
  `ret_full` and `lost_orders` are structurally degenerate: `full_order_values` is gated on
  `completed` at score_time (`program_o_full_des_transducer.py:731-742`), which is empty for
  the early cohort, so `ret_full ≡ 0.0`; and `lost_orders` is hardcoded `np.zeros`
  (`program_o_full_des_transducer.py:795`). Measured on the calibration probe: `ret_full`
  0.0 and `lost_orders` 0.0 in **4/4** arm evaluations while `ret_visible` varied normally.
  Listing either as a live guardrail puts a dead column in every downstream table — the same
  class of defect the five external audits flagged.

Corrected list (`contracts/q_r1_comparator_v2_frozen_c256_v1_1.json`):
`early_ret_visible`, `ret_visible`, `worst_product_fill`, `unresolved_orders`, `resources`,
with `ret_full` and `lost_orders` moved to a `degenerate_disclosures` block that states why
each is dead. **The correction is schema-only and versioned**: `config`, `config_id`, the
convergence receipt and hash, the gate, and the execution authority are byte-identical, and
**v1 itself was not mutated** because the burned Pareto was evaluated against it and records
its hash. Consequence to expect: v1 is byte-regenerable only from the freeze instrument as
of `712a3724`; the current instrument regenerates the corrected list. That divergence is the
fix, not a defect.

---

## 5. Authorized and not authorized

**Authorized by this amendment:**

1. Execute the freeze with the amended instrument
   (`scripts/freeze_q_r1_comparator_v2_amended.py`) against the hashed receipt above.
2. Expand the **burned** Pareto with raw paired retained/reset rows against the frozen
   c256 comparator.
3. Run the clustered prospective **power audit** with SESOI **+0.01**.

**Not authorized:**

- Fresh roots. `fresh_roots_assigned` stays `null`.
- Any learner. M1/M2/M4 remain gated on a deployable residual measured against this
  frozen comparator (`51957969` step 7).
- Freezing any constraint-aware / service candidate (Finding C).
- Any claim from burned material. Everything above is
  `BURNED_DEVELOPMENT_NO_CLAIM`.
- Re-running the ladder to look for a better comparator. If the residual against this
  frozen c256 comparator is null, that is the finding.

---

## 5b. Instrument acceptance tests (executed)

The amended instrument was exercised against the real receipts and against synthetic
receipts degraded one field at a time:

| test | input | result |
|---|---|---|
| T1 | genuine c256/c1024 receipt | **accepts**; derives `conditional_paths = 256`; all four gate criteria recomputed `true` |
| T2 | `expansion_v1` receipt (c64 failed) | refuses |
| T3 | `tieaware_v1` receipt (rejected service tie-break) | refuses |
| T4 | old instrument + genuine receipt | refuses — `ValueError: unexpected ReT indifference tolerance` (cannot execute the authorized freeze) |
| T5 | old instrument, two guards relaxed, genuine receipt | **emits mislabelled contract** (id c256 / config c64) — §2 hazard demonstrated |
| T6a | synthetic: `low_config` rewritten to c64 | refuses — *predeclared pure-ReT c256 comparator is missing or duplicated* |
| T6b | synthetic: agreement degraded to 0.90625 | refuses — *failed convergence gate: ['first_action_agreement']* |
| T6c | synthetic: `selection_performed = true` | refuses — *invalid selection provenance* |

T6b matters specifically: the instrument does not trust the receipt's `convergence_pass`
boolean, so a receipt whose boolean and whose summary numbers disagree is rejected rather
than frozen.

**T1–T6 are now permanent**: `tests/test_q_r1_comparator_v2_freeze.py`, 16 tests, all
passing. Documented-but-unversioned checks protect nothing; T5 in particular pins the
mislabelling hazard so it cannot silently reappear.

---

## 5c. Gate re-derived from the raw rows (closes a real gap in §5b)

§5b's wording was too strong. The freeze instrument applies the thresholds to the
receipt's **summary** fields and refuses to trust its `convergence_pass` boolean — but it
does **not** rebuild those summaries from the underlying rows. A receipt whose summaries
were internally correct while its raw rows disagreed with them would have passed. That
gap is real and is now closed by a separate fail-closed instrument,
`scripts/audit_q_r1_comparator_v2_freeze.py`, which must run before any downstream
evaluation.

It re-derives everything from the **96 raw `convergence_pairs` rows**, using the exact
conventions of `merge_q_r1_comparator_v2_shards.py::merge_convergence` (agreement =
`mean(low_action == high_action)`; `np.quantile(errors, 0.95)`; strict `<` on both error
bounds), and additionally validates coverage.

**Result — `results/q_r1/comparator_v2_c256_c1024_v1/freeze_audit_recomputed.json`:**

| quantity | re-derived from 96 rows | receipt summary | \|Δ\| |
|---|---|---|---|
| first-action agreement | 0.96875 (93/96, **3** disagreements) | 0.96875 | **0.0** |
| mean abs planning value error | 0.00010953411184428986 | same | **0.0** |
| q95 abs planning value error | 0.00024666925036373466 | same | **0.0** |

Coverage checks all pass: 96 unique row identities; exactly 96 rows; a single signature;
the four exact root blocks `[801-806][807-812][813-818][819-824]`; exact root coverage
(24 roots, none missing, none extra); both prior arms present; `conditional_path_budgets
== [256, 1024]`; `states == 48`; `comparable_arm_states == 96`; all three
selection-provenance flags `false`; `claim_status` burned.

**Verdict: `FREEZE_ENTITLED_RECOMPUTED_FROM_RAW_ROWS`, 21/21 checks.** The re-derivation
reproduces the summaries **bit-for-bit** (Δ exactly 0.0, not merely within tolerance), so
the freeze was entitled to happen and the burned Pareto evaluated against it is not
affected.

One boundary convention was aligned in passing: the merger uses strict `<` on the mean and
q95 bounds while the freeze instrument used `<=`. No observed value is anywhere near those
bounds, so nothing changes numerically; the audit uses the merger's convention.

---

## 6. Scope limits of this audit

The auditor verified: commit ancestry and timestamps; the verbatim pre-result text; the
full convergence ladder including the superseded c64 pass; the gate arithmetic; the
receipt hash; the provenance flags; and the two failure modes of the old instrument.

The auditor did **not** re-derive the planner's value estimates, re-run any convergence
shard, or verify that the exact 6-state enumeration is itself correct. §5c re-derives the
convergence *statistics* from the recorded per-row actions and value errors; it does not
recompute those per-row quantities from the planner, so it certifies the receipt's
internal consistency, not the planner's correctness. Those rest on the
comparator v2 construction and its earlier verification, not on this document. First-action
agreement of 0.96875 against c1024 is evidence of **numerical stability at the chosen
budget**, not evidence of planner optimality — the "strongest tested" wording in §4 is
deliberate and must not be upgraded.

---

## 7. Reproduction

```bash
git merge-base --is-ancestor 51957969 41977bfa && echo PRE-RESULT
git show 51957969 -- docs/Q_R1_COMPARATOR_V2_BURNED_PREFLIGHT_2026-07-22.md
shasum -a 256 results/q_r1/comparator_v2_c256_c1024_v1/convergence_merged/result.json
python scripts/freeze_q_r1_comparator_v2_amended.py \
  --receipt results/q_r1/comparator_v2_c256_c1024_v1/convergence_merged/result.json \
  --output contracts/q_r1_comparator_v2_frozen_c256_v1.json
```
