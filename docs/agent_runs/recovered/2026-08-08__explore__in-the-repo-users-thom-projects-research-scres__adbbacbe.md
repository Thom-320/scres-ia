# Explore — In the repo <HOME>/Projects/research/scres-ia, establish EXACTLY how many artifacts c

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `adbbacbe4aa935994`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T01:25:16.699Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-adbbacbe4aa935994.jsonl`
- **turns**: 70 · **assistant messages**: 4

## Task given

```text
In the repo <HOME>/Projects/research/scres-ia, establish EXACTLY how many artifacts carry CONFIRMATION grade, and pull their numbers. Four external audits all assert "there are two confirmations, not three" — I need to verify that against the repo itself.

Do this:
1. Find the authoritative index (likely `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md`) and the canonical claims table (`docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`) and its amendments (`docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA*.md`). Report what each says about evidence grades.
2. Read `research/evidence_registry.jsonl` if present — count rows by `evidence_grade`. List EVERY artifact_path whose grade is CONFIRMATION (or CONFIRMACIÓN), with its claim_status, seed_block and endpoint.
3. For `results/grid_transfer_confirmation_v2/result.json` (find the real path): report the exact contrasts — UCB1 vs cold start, UCB1 vs state-blind marginal replay, the neuron vs marginal replay, GP and OFAT vs replay. Give point estimates and CIs verbatim, plus seed block, n, primary endpoint, and self_sha256.
4. Same for `garrido_h2_h3_confirmation_v1`: the six panels, Holm p-values, seed roots, self_sha256.
5. Check whether any THIRD artifact could plausibly be called a confirmation (e.g. a corrective instrument validation for the ReT repair). Name it and its grade.

Report concrete values and file paths. Do not summarise vaguely — I need citable numbers. Read-only.
```

## Final answer

## Bottom line

**The "two confirmations" claim is stale, not wrong-in-spirit.** The repo's own machine-readable registry carries **4 rows** at `evidence_grade: CONFIRMATORY`, which resolve to **3 distinct artifacts** — the fourth is `garrido_h2_h3_confirmation_v1` counted twice (once on `HEAD`, once on `codex/paper-b-retained-v5`, identical `content_sha256`).

The four audits are reading `<HOME>/Projects/research/scres-ia/docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md`, which was committed `a9b878d` at **Fri Aug 7 12:35:39 2026**. The third confirmation (`gsa_confirmation`) was created **2026-08-07T18:52:30Z** and its corrective landed in `c4162b9` at **13:55:18 local**. The index predates it. The index is not lying; it is 6 hours behind.

---

## 1. What each authority document says

`<HOME>/Projects/research/scres-ia/docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md` (line 16, 20):

> `| **CONFIRMACIÓN** | prospectivo, bloque de semillas virgen, contrato preregistrado | **2** |`
> `### Las dos confirmaciones, y son sólo dos`

Lists exactly `grid_transfer_confirmation_v2` and `garrido_h2_h3_confirmation_v1`. It *does* mention GSA at line 66–67, but only as `gsa_resilience_only/` — a development artifact — and at line 92 flags `g3a_v2_development` / `7700001–7700120` as "the one virgin block left in the whole project." That block is precisely the one `gsa_confirmation` later burned.

`<HOME>/Projects/research/scres-ia/docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` line 63 is stricter still:

> `## 3. La única confirmación prospectiva no es neuronal`

i.e. the canonical claims table counts **one** (scoped to the learning/transfer line). Line 61 explicitly demotes the `search_ladder_v5` numbers: `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION`, `run_role = CACHE_ANALYSIS`, seeds `5300001–012` reused, "**No es confirmación prospectiva.**"

`<HOME>/Projects/research/scres-ia/docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` line 17 registers A2 = `grid_transfer_confirmation_v2` with file hash `7bc33823ccd90b5e` and `run_role: CONFIRMATION` / `CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK_NO_RL_NO_NEURAL_LEARNER`.

`..._ENMIENDA_1.md` adds A10 (`garrido_h2_h3_confirmation_v1`, file hash `bc375d3021b64d10`) and A11 (`q_r1/successor_confirmation_v1`), and states the rule the audits are missing (line 182–186):

> `run_role: CONFIRMATION` **NO implica claim confirmado**

A11 is the witness: `PROSPECTIVE_EXECUTION_COMPLETED · COMPOUND_VERDICT_STOP · NO_POSITIVE_CONFIRMATORY_CLAIM`.

`..._ENMIENDA_2.md` line 9 is worth knowing before you cite any hash: `self_sha256` is the digest of the sealed payload *before* the seal is inserted; `file_sha256` is the bytes on disk; **they never coincide**, and a prior design review mislabelled 16 file digests as `self_sha256`.

---

## 2. Registry counts — `<HOME>/Projects/research/scres-ia/research/evidence_registry.jsonl` (216 rows)

| evidence_grade | rows |
|---|---:|
| DEVELOPMENT | 88 |
| UNCONTRACTED | 57 |
| REPLAY | 42 |
| NEGATIVE_OR_HALTED | 20 |
| DIAGNOSTIC | 5 |
| **CONFIRMATORY** | **4** |

The grade string is `CONFIRMATORY` — there is no `CONFIRMATION`/`CONFIRMACIÓN` value in the registry. All four CONFIRMATORY rows:

| artifact_path | claim_status_as_authored | seed_block | branch | self_sha256 |
|---|---|---|---|---|
| `results/grid_transfer_confirmation_v2/result.json` | `GRID_TRANSFER_CONFIRMED__UCB1` | `8200001-8200060` | HEAD | `eceb9ee97613e172130b7175a79cf6c7d5ddfb54f0ec5435b1174842c20b0f5a` |
| `results/garrido_h2_h3_confirmation_v1/result.json` | `CONFIRM_H2_H3_ALL_SIX_PANELS` | `96111336-97836128` | HEAD | `null` |
| `results/garrido_h2_h3_confirmation_v1/result.json` | `CONFIRM_H2_H3_ALL_SIX_PANELS` | `96111336-97836128` | `codex/paper-b-retained-v5` | `null` |
| `results/gsa_confirmation/result.json` | `GSA_CONFIRMED_ON_VIRGIN_BLOCK` | `7700001-7700120` | HEAD | `f1181706cf591edb669a1735eab6d6c1b9f116caed3e4175bfb1bafe1ab8adc9` |

`endpoint` is `null` in the `dedup_key` of **all four**, and `dedup_key_complete: false` throughout. `grade_rationale` for each is `"confirmation role over custody block <range>"` — so the grade is assigned from custody role, *not* from whether the claim held.

The HEAD row for `garrido_h2_h3` carries `duplicate_of: "results/garrido_h2_h3_confirmation_v1/result.json"` (self-referential), which is how the double-count shows up.

---

## 3. `results/grid_transfer_confirmation_v2/result.json`

Path is exactly `<HOME>/Projects/research/scres-ia/results/grid_transfer_confirmation_v2/result.json`.

- `self_sha256` = `eceb9ee97613e172130b7175a79cf6c7d5ddfb54f0ec5435b1174842c20b0f5a`
- file sha256 (bytes on disk) = `7bc33823ccd90b5e83068287ddffa387427734c098900aa9710e72890ca0cd4a`
- `seed_block` = `"garrido_grid_transfer_v2_confirmation"`, `seeds` = 8200001…8200060 (60 seeds)
- **n = 60 on every contrast**
- `run_role` = `CONFIRMATION`; `scope` = `CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK_NO_RL_NO_NEURAL_LEARNER`
- `contract_path` = `docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md`, `contract_sha256` = `1abbb2df6526f82b887e1ba07cb66f5fa06bb2fbf1f847e79a63b036938c5c9d`
- `n_base_configs` = 288 → `n_ext_configs` = 4608; `budget` = 24
- `created_at` = `2026-08-06T09:39:38.294409+00:00`; `elapsed_seconds` = 3802.395
- `reference_path` = `results/grid_transfer_ordered_v1/result.json`, `reference_sha256` = `5fa117707b148cf1b981b76d317bb1527f8256e6e9f3eb3fc9d5577dba59040d`
- `transfers` = `{"gp": false, "neuron": false, "ofat": false, "ucb1": true}`

**Primary endpoint:** there is no `endpoint` key in the artifact. The contract `docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md:56` defines it: `E[δ_M]` with `δ_M(s) = AUC_marginal(s) − AUC_UCB1_transfer(s)`. Lower AUC is better; contrasts are signed so positive = transfer wins.

### Contrasts, verbatim (all n = 60)

| arm | vs cold start (mean [lcb95, ucb95]) | vs state-blind marginal replay |
|---|---|---|
| **ucb1** | 0.05743819277092614 [0.049888584439231054, 0.06480569983808135] | **0.03073311127302739 [0.019896866431745156, 0.042561504429982196]** ✅ |
| **neuron** | 0.054393629143446906 [0.04289557823009438, 0.06687098572184685] | **−0.011782983814352867 [−0.01848914143725492, −0.004835955987826106]** ❌ |
| **gp** | 0.01432635783389502 [0.008787735338250295, 0.02056083569699176] | −0.021595031378833847 [−0.03050640445741258, −0.012267301800787132] ❌ |
| **ofat** | 0.014218869453841312 [0.008002602681115948, 0.019318535198958688] | −0.024672161316045497 [−0.032580438401241286, −0.01666244793292766] ❌ |

The neuron-vs-marginal figure the index quotes (−0,01178 [−0,01849, −0,00484]) matches the artifact exactly.

### Supporting `mean_auc` (lower is better)

`ucb1_transfer` 0.07268348890154588 · `ucb1_marginal` 0.10341660017457327 · `ucb1_cold` 0.130121681672472
`neuron_transfer` 0.07980427883032978 · `neuron_marginal` 0.06802129501597692 · `neuron_cold` 0.1341979079737767
`gp_transfer` 0.09341379718626547 · `gp_marginal` 0.0718187658074316 · `gp_cold` 0.10774015502016045
`ofat_transfer` 0.09404105138601099 · `ofat_marginal` 0.06936889006996547 · `ofat_cold` 0.10825992083985228

Note the mechanism directly: `neuron_marginal` (0.0680) is *lower* than `neuron_transfer` (0.0798). The neuron's own visit marginals, replayed state-blind, beat the neuron.

### Falsifiers — `all_passed: true` (6/6)

`f1_the_null_subgrid_reproduces_the_288_cache` (103,680 cells checked, `max_abs_delta` = 0.0, 0 mismatches) · `f2_the_new_factors_move_the_endpoint` · `f3_transfer_beats_its_marginal_replay` · `f4_seed_custody` · `f_budgets_are_matched` (24 per search, 0 failures) · `f_source_manifest_is_identical`.

**One caveat you should carry:** `f4_seed_custody` passed but self-declares `registry_status: "BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED"` with `registry_is_complete: false` and the note — "the central inventory declares itself incomplete, so this is NO_KNOWN_COLLISION and not a proof of virginity." Block virginity here is *not-known-to-collide*, not proven.

---

## 4. `results/garrido_h2_h3_confirmation_v1/result.json`

Path: `<HOME>/Projects/research/scres-ia/results/garrido_h2_h3_confirmation_v1/result.json`.

- **`self_sha256` does not exist in this artifact** — registry has `self_sha256: null`, and the top-level key set has no such field. The citable digests are `content_sha256` = `19e2c1ec4aab387e` (registry) and file sha256 = `bc375d3021b64d1069f4111d5350294db77317c04e81ec222ac1ebdd17e8b195` (matches the `bc375d3021b64d10` in ENMIENDA_1 A10). If an audit hands you a `self_sha256` for this artifact, it is a mislabelled file digest — exactly the error ENMIENDA_2 §"Los dos hashes" documents.
- `status` = `CONFIRM_H2_H3_ALL_SIX_PANELS`; `global_confirmation_pass` = `true`
- `confirmation_roots_opened` = `true`; `development_roots_opened` = `false`
- **`confirmation_tape_roots` (12 virgin roots):** `96111336, 96453941, 96549564, 96643766, 96876420, 96985702, 97004217, 97185027, 97327421, 97613612, 97646500, 97836128`
- `n_tapes` = **12 per panel**; `row_count` = 1080; `neutral_shift_checks` = `{all_equal: true, count: 108}`
- `contract_sha256` = `1d3c80bd48feac4c71065ad3e432accfcc99cfe7c0cbf20a8166ea95103cfe98`; `contract_path` in registry is `null` (the contract lives at `contracts/garrido_h2_h3_confirmation_v1.json` per ENMIENDA_1)
- `freeze_receipt_sha256` = `352a4dcaa4635c4aeeafa582c783588af2e156a58ee73f3c45ec75f7ebae1a0f`
- `code_commit` = `9829084de18d0e1bf57d0da31d54beca56dd6997`; `created_at` = `2026-07-29T16:53:36.360329+00:00`
- **Claim boundary, verbatim:** `"Confirmation applies only to H2/H3 resource interventions in the frozen thesis-grounded reconstructed DES; it does not establish learner, feedback, or architectural value."`

**Primary endpoint = `ret_excel`** (that is the metric whose `one_sided_positive_p` feeds the Holm ladder — each panel's `ret_excel.one_sided_positive_p` equals its `holm.p_value` exactly).

### Six panels — Holm ladder and primary endpoint

| panel | Holm rank | Holm threshold | p_value | ret_excel mean [lcb95, ucb95] | n |
|---|---:|---|---|---|---:|
| R3:H2_buffer | 1 | 0.008333333333333333 | 4.7554109189133694e-17 | 0.03623302776727328 [0.03527370395484829, 0.037192351579698274] | 12 |
| R1r:H2_buffer | 2 | 0.01 | 6.618352850350777e-17 | 0.0005354201229473343 [0.0005208109332238583, 0.0005500293126708102] | 12 |
| R2r:H3_shift | 3 | 0.0125 | 4.80521239899934e-16 | 0.07582819033358885 [0.07334974955636112, 0.0783066311108166] | 12 |
| R3:H3_shift | 4 | 0.016666666666666666 | 1.2176835466651612e-15 | 0.02359898015235028 [0.02275943950123066, 0.0244385208034699] | 12 |
| R2r:H2_buffer | 5 | 0.025 | 1.297614470374752e-15 | 0.1284415546074628 [0.12384565157279213, 0.13303745764213346] | 12 |
| R1r:H3_shift | 6 | 0.05 | 1.895791510834798e-15 | 0.00036415833276666384 [0.00035066979556362535, 0.0003776468699697023] | 12 |

All six `holm.pass = true`. Index's "Holm p 4,8e−17…1,9e−15" is slightly off at the low end: the minimum is **4.7554e−17** (R3:H2_buffer), not 4.8e−17 — the 4.805e−16 value belongs to R2r:H3_shift. Max 1.8958e−15 is correct.

Every one of the six `panel_gates` has all seven sub-gates `true`: `confirmed`, `primary_holm`, `fill_lcb_positive`, `delivered_lcb_positive`, `full_ledger_lcb_positive`, `generated_orders_exact_zero`, `unresolved_ucb_negative`.

Each panel also carries a full secondary ledger (`ret_thesis`, `ret_excel_full_ledger`, `ret_continuous`, `flow_fill_rate`, `delivered_rations`, `lost_orders`, `omitted_rows`, `scored_rows`, `served_orders`, `generated_orders`). E.g. R1r:H2_buffer `flow_fill_rate` mean 0.0875118332520174 [0.08700638248972684, 0.08801728401430797], `delivered_rations` mean 614350.5083333333, `generated_orders` exactly 0.0 across all 12 tapes.

---

## 5. The third artifact — and it is GSA, not the ReT repair

### The actual third confirmation: `results/gsa_confirmation/result.json`

`<HOME>/Projects/research/scres-ia/results/gsa_confirmation/result.json` — grade **CONFIRMATORY**, `run_role: CONFIRMATION`, `scope: CONFIRMATION_ON_REPURPOSED_VIRGIN_BLOCK_NO_TRAINING_AUTHORISED`.

- `self_sha256` = `f1181706cf591edb669a1735eab6d6c1b9f116caed3e4175bfb1bafe1ab8adc9`; file sha256 = `1f487d91900e2ea42e792b0b1873e2230ee47388df4d4a602bbc9a932c5969dd`
- `seed_block` = `g3a_v2_development`, 7700001–7700120, `opens_once: true`, `repurposed_from: "G3a asymmetric-claimant development"`, gate lifted per `docs/AUTORIZACION_PI_REPROPOSITO_BLOQUE_7700001_2026-08-07.md`
- `n_tapes` = 120; `claim_status` = `GSA_CONFIRMED_ON_VIRGIN_BLOCK`
- `H_obs` mean 0.012286775167294721 [0.00867686813365007, 0.01615513451498119], n=120; `H_PI` = 0.012926135742150043; `eta` = 0.9505373773253464
- `obs_minus_placebo` mean 0.0715180378293156 [0.0504619478342132, 0.09429398762433945]
- contract `docs/PREREGISTRO_CONFIRMACION_GSA_2026-08-07.md`, sha `302c2e21499ad5cbfaa54589a2946d395db273811fffde8a7dfd6b0002b286ce`

**Critical: `falsifiers.all_passed = false`.** `f4_the_placebo_is_uninformed` = **failed** and `f6_the_result_can_be_negative` = **failed**, yet `claim_status` reads `GSA_CONFIRMED_ON_VIRGIN_BLOCK`. This is the strongest reason not to simply append it to the "two" and call it three. `docs/CORRECCION_FALSADORES_CONFIRMACION_GSA_2026-08-07.md` owns both as specification errors: f4 demanded a property "insatisfacible por construcción" (the belief policy emits only 2 distinct sequences across 120 tapes), and f6 "comprueba **el dato, no el estimador**."

### Its corrective — graded DEVELOPMENT, deliberately

`<HOME>/Projects/research/scres-ia/results/gsa_confirmation_corrective/result.json`

- `evidence_grade` = **DEVELOPMENT**, rationale `"contracted, but not a confirmation over a virgin block"`
- `run_role` = `CORRECTIVE_REANALYSIS_OF_A_CONFIRMATION`; `scope` = `SAME_BURNED_BLOCK_SAME_THETA_ONLY_THE_FALSIFIERS_AND_PLACEBO_ARE_CORRECTED`
- `self_sha256` = `e26b2b16107274ca08549b5c2bfa44c593246557f9105a647310222b60d774fe`
- `claim_status` = `GSA_CONFIRMED_ON_VIRGIN_BLOCK_AS_A_ONE_BIT_CALENDAR_CHOICE`
- `falsifiers.all_passed` = **true**
- `supersedes` → `results/gsa_confirmation/result.json` seal `f1181706cf591edb…`
- `obs_minus_permuted_placebo` mean 0.07210657135008629 [0.049801679249144644, 0.0959939565344071]
- `negative_control_obs_minus_oracle` mean −0.0006393605748553234 [−0.001447611268211855, −7.048799550112284e-05] — the corrected f6
- `policy_sequences` = `{"ABAB": 89, "AAAA": 31}`, `placebo_ties` = 74, `best_static_calendar` = `["A","B","A","B"]`, both sequences in the static comparator set

So the *fixed* GSA analysis is graded DEVELOPMENT and the *broken* one is graded CONFIRMATORY. The registry grades on custody role, not on falsifier outcome.

### The ReT repair instrument you asked about — grade DEVELOPMENT, and half of it is NOT_CONFIRMED

`<HOME>/Projects/research/scres-ia/results/metric_audit/ret_metric_repair_confirmation_v1/result.json`

- `evidence_grade` = **DEVELOPMENT** (both `result.json` and `adjudication.json`), rationale `"contracted, but not a confirmation over a virgin block"`
- `claim_status` = `PROSPECTIVE_CORRECTIVE_CONFIRMATION` — the *name* says confirmation, the grade does not
- `self_sha256` = `bde02309f72c9ee11222704a63a4c38dc27ca74cb9c15743dc42bd8b104c0ff4`
- contract `contracts/ret_metric_repair_confirmation_v1.json`, sha `c1efdc20fc9d75743a5789ab3ddf90108d7f18cf1230dcd62fb5a85815dc441e`
- `source_result_sha256` = `["418f669a44cadf9987a5c195f5d67f8356baf149799897a661da2d9e9a3ca19f", "49f3218485acd958f44b7fb1c1ad6864966bed356533520e19f0b3d03d134125"]`
- n = 16 tapes per family; `neural_authorization` = `false`; `historical_endpoint_unchanged` = `true`; `quantity_time_causal_status` = `DISCLOSED_PROXY_NOT_EXACT_ATTRIBUTION`
- **`families.R1r.verdict` = `NOT_CONFIRMED`** · `families.R2r.verdict` = `PASS_MATERIAL_REPAIRED_MPC`
- R2r `ret_excel` delta_mean 0.012515645095362442, ci95 [0.009003888449569987, 0.015954997659645993], 15/16 positive — but R2r `ret_excel_full_ledger` delta_mean **−0.004483477888646593** ci95 [−0.006600459182423437, −0.0023880153128589383], and R2r `ret_thesis` delta_mean 0.0003699960639464311 ci95 [−0.0010844538318445054, 0.0017306137353799477] (straddles zero). The sign flips depending on which ReT variant you read.
- R1r `ret_excel` delta_mean −1.954003976329104e-05 ci95 [−4.955658717723581e-05, −1.6796812196908317e-07]

**This one cannot plausibly be called a third confirmation.** It is DEVELOPMENT-graded, one of its two families is `NOT_CONFIRMED`, and its own primary metric disagrees with its full-ledger variant on sign.

### Also checked and excluded

- `results/q_r1/successor_confirmation_v1/merged/result.json` — grade DEVELOPMENT, `claim_status: PROSPECTIVE_CONFIRMATION`, on branch `codex/paper-b-cf1-cf20-replication`. ENMIENDA_1 A11 labels it `PROSPECTIVE_EXECUTION_COMPLETED · COMPOUND_VERDICT_STOP · NO_POSITIVE_CONFIRMATORY_CLAIM`.
- `results/gsa_resilience_only/result.json` — DEVELOPMENT, `GSA_QUALIFIES_UNDER_RESILIENCE_ONLY`, seal `759c2955cccf40623dde2b786bc09557a27f9d24c85d5c3d6c3432361668e52e`.
- `results/determinism_repair_control/result.json` — **DIAGNOSTIC**, `DETERMINISM_REPAIRED_SEED_IS_A_REPLICATION_UNIT_AGAIN`, rationale "instrument property, no comparative claim".
- `results/k3/confirmation.json` — index line 81: self-retracted, `effective_verdict: RETRACT_...`.

---

## How I'd answer the auditors

Three numbers are all defensible, for three different questions:

- **2** — confirmations with `all_passed: true` falsifiers *and* a positive claim that survives. This is what the index means and what the audits are echoing.
- **3** — distinct artifacts carrying `evidence_grade: CONFIRMATORY` in the registry today. GSA is the third; it burned the last virgin block (7700001–7700120) at 18:52Z on 7 Aug, after the index was written at 12:35 local.
- **4** — raw CONFIRMATORY row count, inflated by the cross-branch duplicate of `garrido_h2_h3_confirmation_v1`.

The auditors are not wrong so much as reading a document that was already superseded when they read it. But the honest version has a sharper edge than "there are three": **the third confirmation shipped with two failing falsifiers**, and the repo's own correction (`CORRECCION_FALSADORES_CONFIRMACION_GSA_2026-08-07.md`) says so in the first sentence. The corrected re-analysis passes all falsifiers but is graded DEVELOPMENT because the block was already burned. So GSA is a confirmation by custody and a development result by evidence — which is exactly the distinction ENMIENDA_1 line 182 warns about: `run_role: CONFIRMATION` does not imply a confirmed claim.

Two things worth flagging for whoever maintains this:

1. `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md` line 16 and heading line 20 assert "2" and "sólo dos" while the registry says 3 distinct. Line 92's "Queda UN bloque virgen en todo el proyecto" is also now false — that block was opened the same day.
2. `dedup_key.endpoint` is `null` on all four CONFIRMATORY rows and `dedup_key_complete` is `false` across the entire registry, so the registry cannot currently be used to dedupe on endpoint — which is probably why the h2_h3 cross-branch duplicate survives.
