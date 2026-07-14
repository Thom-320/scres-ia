# Source reconstruction for the Paper 2 environment search

Date: 2026-07-13
Status: Phase 0 source-of-truth extraction; this document does not authorize new physics.

## Source identity

| Source | SHA-256 | Role in this search |
|---|---|---|
| `WRAP_Theses_Garrido_Rios_2017.pdf` | `de9192d233b0c728ece6156b754fc64543146868121358b8a95c73b3edaa55cf` | Primary physical and metric authority for the MFSC DES. |
| `garrido2024 scres+AI.pdf` | `3e3bc8f82e20b891ee163fb8a035dd37be4312fa11f58dde77452dc1bb903ae6` | Conceptual authority for the learning question, not validation of an adaptive decision contract. |
| `garrido et al 2024 factory resilience.pdf` | `1260863dc295232faf24b820e1f67d53f25f81ffa2d221f7ef02a02310519c43` | Primary source for the factory/APP Cobb-Douglas construct; not the order-level MFSC endpoint. |
| `v.0_neuralNet-scres.pdf` | `521b12770e94f3e70c4c88ce1e38613f4e0aad3e1dab114632c9c89dbfad182d` | In-progress manuscript reconstruction; useful for tracing intended implementation, not physical ground truth. |
| `v.0_neuralNet-scres.docx` | `b111070a05c8f4d1afa058454138bed9b4b74900ab87eaaf6eb5186b6e8293f2` | Editable form of the same draft. Visual rendering confirms placeholders and incomplete result sections. |
| `Raw_data1+Re.xlsx` | `30b88c9b9fe68ef527dbfcc70d8e653ea7bd152ab891b3fc0ecf53cb6f043486` | Primary raw workbook for `CF1`-`CF10`; operational ReT formula in column `U`. |
| `Raw_data2+Re.xlsx` | `4bd462771fefff16fc5666a851256b3780198d474832dec1423c0b6f94be86b0` | Primary raw workbook for `CF11`-`CF20`; operational ReT formula in column `AA`. |
| `Rsult_1.xlsx` | `1901f683f6014cf75237c17233b8eba04f541b956f2d19dcecf2edc00e83b00a` | Secondary transformed/discretized workbook; not a consolidation of all 20 raw `CF` sheets. |

The canonical PDF texts were extracted with Poppler 26.06.0
`pdftotext -layout`; the DOCX text was extracted with Pandoc 3.10
`-t plain`. The DOCX was separately rendered to 21 page images and visually
sampled. The draft contains placeholder abstract/results text, internal
drafting comments, a proposed Cobb-Douglas training reward, and incomplete
claims. It is not an audited result. Workbook identities and hashes above refer
to the local files audited by
[`docs/audits/garrido_excel_des_2026-06-25/README.md`](../../docs/audits/garrido_excel_des_2026-06-25/README.md);
the raw books are source evidence, while the README and its JSON output are
audit evidence.

### Deterministic searchable-text reconstruction

Full copyrighted source text is not committed. A clean clone can reconstruct
and verify every canonical text when supplied the source files whose hashes are
listed above:

```bash
python scripts/verify_source_extractions.py \
  --output-dir tmp/paper2_source_extracts \
  --report tmp/paper2_source_extraction_verification.json
```

The verifier is fail-closed on source SHA-256, extractor version, PDF page
count, output SHA-256, bytes, lines and words. `--source ID=/path` permits a
source to live at another path without relaxing its identity check. Exact
commands, output names and counts are machine-readable in
[`source_extraction_index.json`](source_extraction_index.json).

The five historical full-text files under ignored `tmp/` paths were audited
before removing them from the evidence dependency: their hashes equal the five
canonical output hashes in the v3 index. They are disposable local caches, not
required inputs. The tracked verifier plus the hash-identified original sources
are the reconstruction authority.

The tracked root [`thesis.txt`](../../thesis.txt) is not the canonical indexed
extract. It is a legacy extraction of the same thesis PDF made with default
flow (`pdftotext` without `-layout`): 443,582 bytes, 10,008 lines, 65,778 words,
SHA-256 `c0b1671c...72da1c7`. The canonical layout-preserving extract is 533,903
bytes, 7,869 lines, 65,856 words, SHA-256 `5c6101b9...a75d05c`. The verifier
regenerates both byte-for-byte and proves that this hash difference is an
extractor-mode difference, not a second or changed thesis source. Search notes
and indexed page references use the layout-preserving form.

## Thesis chapter-by-chapter reading map

The complete 187-page PDF was text-extracted. The map below records what each chapter contributes and, equally importantly, what it does not authorize.

| Thesis block | Printed pages | Extracted content relevant to this search | Claim boundary |
|---|---:|---|---|
| Chapter 1 - Introduction | 16-20 | Military-logistics problem, objectives, scope, research problem, contributions and thesis structure. | Establishes the research setting; it does not define a sequential adaptive decision contract. |
| Chapter 2 - Systematic literature review | 22-38 | Contemporary-to-2017 SCRES measurement approaches, review protocol, gaps and classification of methods. | Historical source for the thesis gap, not a current literature review for Paper 2. |
| Chapter 3 - Hypotheses and conceptual framework | 40-49 | Fixed-scenario hypotheses for risk frequency, on-hand buffers and short-term manufacturing capacity. | Tests moderation across configurations; it does not show adaptive-over-open-loop value. |
| Chapter 4 - Research methods | 51-58 | Simulation as the quantitative method, questionnaire/case-study component and non-parametric analysis. | Methodological authority for the original study only; new physics still requires fresh verification and validation. |
| Chapter 5 - Operationalization of resilience | 60-77 | Military-SC characteristics; AP/RP/DP definitions and algorithms; Equations 5.1-5.5; accumulated fill-rate case. | Primary conceptual source for thesis ReT. The raw Excel population/formula is a separate operational artifact reconciled below. |
| Chapter 6 - DES, experiment and questionnaire | 79-120 | Op1-Op13, capacities, demand, risks, assumptions, verification/validation, 90 configurations, run length, warm-up, output matrix and questionnaire. | Primary MFSC-physics authority, subject to its explicit simplifications and proxy/validation limits. |
| Chapter 7 - Results | 123-139 | Association/causality analysis for risk frequency and moderation by buffers or shifts. | Results concern fixed configurations and the thesis ReT categorization; they do not establish feedback, learning or neural value. |
| Chapter 8 - Conclusions and future research | 141-149 | Findings, practical implications, stationarity/cost/single-case limitations, lead-time and optimal-resilience questions. | Identifies questions and limitations; it does not supply numerical parameters for new mechanisms. |
| Bibliography and Annexes A-E | 150-186 | Source bibliography, Simulink code, questionnaire and R algorithms for risk/ReT categorization and association rules. | Useful provenance and implementation evidence; not proof that the Python DES is distributionally identical to Simulink. |

## Thesis-native MFSC semantics

The operation numbers below follow thesis Chapter 6, especially Sections 6.2-6.5 and Figure 6.2.

| Operation | Thesis-native role | Native cadence/capacity facts relevant to decisions |
|---|---|---|
| Op1 | Military logistics agency contracts 12 suppliers. | Processing time is 672 h; 12 contracts are renewed every 4,032 h. Op1 and Op2 remain outside the adaptive scope. |
| Op2 | Suppliers prepare and ship raw materials. | Processing time is 24 h; monthly delivery is 190,000 units of each `rm1`-`rm12`, with 672 h ROP. |
| Op3 | WDC receives, verifies, and stores raw materials. | Processing time is 24 h; weekly release is 15,500 units of each `rm1`-`rm12`, with 168 h ROP and zero initial raw-material inventory. |
| Op4 | WDC-to-assembly line of communication. | 24 h transport of 15,500 units of each raw material, weekly/168 h. |
| Op5 | Pre-assembly of energetic products. | Balanced line rate 320.5 rations/h; PT 0.003125 h/ration, Q=1 and immediate ROP; 2,564 units per 8 h shift. |
| Op6 | Assembly. | Same balanced PT, Q=1 and immediate cadence as Op5. |
| Op7 | Quality control, final packaging and shipping release. | PT 0.003125 h/ration; boxes of 10; release batch 5,000 every 48 h. |
| Op8 | Assembly-to-supply-battalion transport. | 24 h transport; batch 5,000 every 48 h. |
| Op9 | Supply battalion receipt/classification/storage. | Processing time 24 h; daily downstream quantity 2,400-2,600; zero initial finished inventory. |
| Op10 | Supply battalion-to-CSSU transport. | 24 h; daily quantity 2,400-2,600. |
| Op11 | Two CSSUs receive and issue rations. | Actual handling is under 1 h and is modeled as PT=0; daily quantity 2,400-2,600. |
| Op12 | CSSU-to-theatre transport. | 24 h; daily quantity 2,400-2,600. |
| Op13 | Theatre demand. | Regular demand is 2,400-2,600 rations/day, six days/week; `R24` adds 2,400-2,600 contingent rations per 672 h and has priority. |

The thesis describes a dual policy: assemble-to-stock from Op1 through Op9 and assemble-to-order from Op9 through Op13, using `(ROP,Q)` logic. Once the first 5,000-ration batch reaches Op9, the promised downstream order lead time is fixed at 48 h. The deterministic warm-up calculation is 838.8 h; the implemented warm-up is triggered by the first Op9 arrival because stochastic risks can lengthen it. The main non-terminating runs use up to 20 years (161,280 h), 90 fixed configurations and paired seed reuse across matched configuration families. These are source facts, not a license to reuse opened seeds for Paper 2.

The thesis uses one product, Cold Weather Combat Ration #1, as a simplifying representation. It also states that the real MFSC produces 21 ration types selected for nutritional and climatic requirements. This is evidence that product mix is a real omitted dimension; it is not evidence for any particular product shares, bill of materials, substitution relationship, setup time, weight, volume, or forecast accuracy.

The thesis treats the ration as non-perishable for the modeled horizon: properly stored combat rations can remain usable for approximately three years. A two-week shelf-life contract is physically incompatible with this product.

The thesis also states that one ration pack covers 24 hours, must not exceed
1.4 kg, and that a soldier typically carries about five packs, with the number
depending on mission length and resupply frequency. Op13 discussion says that
mission planning determines ration requirements before battlefield uncertainty
materializes. These facts anchor a mission-loadout/carried-autonomy question,
but the current DES ends at theatre delivery and supplies no cohort carried-
inventory state, loadout authority, mass constraint, sealed consumption-order
ledger or unused-pack return rule.

## Thesis-native risk semantics

- `R11`: Op5/Op6 workstation breakdowns, with approximately two-hour mean repair.
- `R14`: non-conforming Op7 products are returned for reprocessing.
- `R21`: a natural-disaster event affects Op3, Op5-Op7, and Op9, with approximately 120 h mean recovery.
- `R22`: line-of-communication disruption affecting Op4, Op8, Op10, or Op12, with approximately 24 h mean rehabilitation.
- `R23`: CSSU outage/reactivation, with approximately 120 h mean recovery.
- `R24`: 2,400-2,600 contingent rations per 672 h, prioritized ahead of ordinary demand; this cadence is not the same as regular daily demand.
- `R3`: black-swan interruption affecting multiple upstream operations for 672 h.

**Source claim.** The thesis defines risk-on/risk-off configurations, specified occurrence and recovery distributions, and experimental increases in risk frequency. It does not grant unrestricted permission to tune disruption impact. Any broader authorization to vary frequency or impact is separate author/project evidence and must not be attributed to the thesis itself. The thesis does not natively provide scarce vehicle fleets, capacity-reservation pipelines, CSSU-to-CSSU lanes, reporting errors, finite storage, multiple product ledgers, substitution, mission expiry, or active information-acquisition resources.

Sections 5.2 (pp. 60-61) and 6.2 (p. 80) state that military distribution can use land, sea/water, and air modes and that CSSUs select a mode according to urgency and equipment availability. This is an operational anchor for asking about multimodal choice, not a numerical contract: the thesis does not give mode-specific fleet counts, payloads, cycle times, R22 exposure, eligibility, or intertemporal commitment.

## Binding assumptions and their consequence for candidate mechanisms

Thesis Section 6.5 assumes:

- one homogeneous ration product;
- proactive buffers and fixed scenario policies;
- late orders remain backorders; the pending list is capped at 60, with SPT service and contingent priority;
- vehicle availability and route planning are taken for granted;
- storage capacity is unlimited;
- no partial deliveries or last-minute production changes;
- scheduled maintenance is represented as a fixed weekly 24 h interruption;
- stationary parameters/demand over a long non-terminating experiment.

These are disclosed modeling assumptions, not universal operational laws. Modifying them is a researcher extension and requires an intervention ledger, null regime, unit/conservation tests, plausible envelope fixed independently of learner results, and an exact Garrido sign-off question.

Table 6.16 reports inventory levels for buffering scenarios. It does not report physical storage capacities and must not be used to invent finite-space limits.

## Thesis openings that are real but incomplete

Chapter 8 explicitly identifies the first three limitations/future directions below; Chapter 6 supplies the fourth model simplification:

1. Long-term demand-pattern changes should be incorporated quickly into buffering and require advance cross-functional forecasting.
2. Stationary parameters and demand are a major limitation.
3. Lead time is an explicit future-research dimension.
4. Sections 6.3.1 and 6.5.3 state that the real 21-product assortment was compressed to one homogeneous product.

These passages justify asking whether regime forecasts, product mix, or lead-time commitments exist operationally. They do not validate numerical signal quality, persistence, scarce capacity, or neural superiority.

## Canonical endpoint reconciliation

Thesis Section 5.6.2--5.6.3 (printed pp. 72--76) defines four mutually
conditional order/time cases. With the thesis weights
`Re_max=1`, `Re=0.5`, and `Re_min=0`, they are:

\[
Re(AP_j)=Re_{max}\,AP_j/LT_j,\qquad
Re(RP_j)=Re\,(1/RP_j),
\]

\[
Re(DP_j,RP_j)=Re_{min}(DP_j-RP_j)/CT_j,\qquad
Re(FR_t)=1-(B_t+U_t)/D_t.
\]

Equation 5.5 selects the applicable dynamic case when a risk is present and
the fill-rate case otherwise. The workbook formula reconstructed in
`supply_chain.ret_thesis.compute_ret_per_order_excel_formula` is the operational
authority for this search:

```text
IF(risk_active,
   IF(APj > 0, APj / LTj, 0.5 * (1 / RPj)),
   1 - ((Bt + Ut) / j))
```

The workbook branch does not clip its result, and its visible sheets omit many
lost or horizon-unresolved rows while retaining their cumulative `Bt`/`Ut`
effects. Consequently, the binding repository implementation is
`ret_excel_visible_v1`, evaluated through the canonical episode aggregator:

- completed, non-lost order rows are visible in the mean;
- lost and unresolved orders are not assigned a favorable synthetic ReT value;
- their demand/backlog effects remain in the cumulative ledger;
- lost orders are therefore a simultaneous non-inferiority guardrail, not an optional reporting field.

`program_g.ret_order_metrics` uses a different full-ledger implementation and must not be substituted for `ret_excel_visible_v1`. Service loss, quantity-weighted ReT, worst-CSSU fill, backlog, and tail outcomes remain simultaneous diagnostics; none may replace the primary endpoint after policy results are observed.

### Workbook traceability and fidelity boundary

| Workbook | Raw scope | Formula evidence | Role and limit |
|---|---|---|---|
| `Raw_data1+Re.xlsx` | `CF1`-`CF10`; risk columns `R11_1`, `R11_2`, `R12`, `R13`, `R14`; ReT column `U` | Rowwise recalculation matches cached Excel values exactly. | Primary raw workbook for the operational-risk family. |
| `Raw_data2+Re.xlsx` | `CF11`-`CF20`; risk columns `R21_1`-`R21_5`, `R22_1`-`R22_4`, `R23`, `R24`; ReT column `AA` | Rowwise recalculation matches cached Excel values exactly. | Primary raw workbook for natural-disaster/attack/contingent-demand cases. |
| `Rsult_1.xlsx` | `Cf1`-`Cf12` plus aggregate `APj`, `RPj`, `DPj`, `Re` sheets | Secondary normalization/discretization analysis. | Must not be treated as a direct consolidation of the 20 raw `CF` sheets. |

The frozen [workbook audit](../../docs/audits/garrido_excel_des_2026-06-25/README.md) and its [machine-readable result](../../outputs/audits/garrido_replication_2026-06-25/replication_audit.json) report:

- 20 raw `CF` sheets and 47,546 formula-bearing rows audited;
- zero formula mismatches and maximum absolute difference `0.0`;
- exact branch-share agreement only in the forensic `excel_order_tape + excel_risk_tape` lane;
- mean absolute ReT gap `0.0037525451` for that forensic replication.

These facts establish four distinct levels that must not be collapsed:

1. **Formula fidelity:** the conditional Excel formula is transcribed exactly.
2. **Row-population fidelity:** `ret_excel_visible_v1` reconstructs the sparse visible-order population and cumulative `Bt`/`Ut` ledger.
3. **Forensic numerical replication:** replaying workbook `Q/OPTj` and visible risk/AP/RP/DP tapes reproduces the workbook closely.
4. **Endogenous DES fidelity:** the repository's own demand/risk generators do **not** thereby inherit the workbook trajectories or distributional validity.

Only levels 1-3 are established by this workbook audit. This section makes no claim that endogenous Python risk dynamics exactly reproduce the thesis Simulink model.

### Fresh 2026-07-13 metric lock

The workbook audit was rerun read-only against the three current local files.
It again covered 20 raw `CF` sheets and 47,546 formula-bearing rows, with zero
mismatches and maximum absolute difference `0.0`. Seventeen focused formula,
visible-population and packaging tests passed. The result locks
`ret_excel_visible_v1` as the current Paper 2 primary endpoint; it does not
authorize a metric switch. The machine-readable reconciliation is
[`metric_governance_audit.json`](metric_governance_audit.json).

## Garrido 2024 factory-resilience Cobb-Douglas boundary

The separate IJPR paper *Zero-inventory plans, constant workforce, or hybrid
approach?* studies a hypothetical make-to-stock factory over a 36-week APP
horizon. Demand variability is the sole uncertainty, and seven fixed pure APP
substrategies are compared. Its factory-resilience construct uses average
inventory `zeta`, backorders `epsilon`, spare capacity `phi`, net-requirement
fulfillment time `tau` and normalized total cost `kappa_dot`:

```text
0.024 ln(zeta) - 0.026 ln(epsilon) + 0.040 ln(phi)
- 0.060 ln(tau) - 0.1771 ln(kappa_dot)
```

Equation 6 applies a sigmoid to this score. The coefficients equate the five
terms at maxima observed in 10,000 simulations of that factory and strategy
ensemble. They are not universal MFSC preference weights. The paper reports a
dominant fixed match-chase strategy, `S12`; it does not demonstrate sequential
feedback or learning value.

The repository has two related but non-equivalent adaptations. Program G uses
the published exponents as an explicitly inspired distribution construct with
constant `phi` and `kappa`. The shift environment maps five APP-like variables
into the MFSC DES, but its default calibration replaces the published
coefficient calculation with standardized logs under `variance_log` balancing;
the file also stores DES-rebalanced exponent fields that are not used by that
active score. Neither
may be called the exact published factory index without qualification. See the
[full audit](../../docs/COBB_DOUGLAS_FACTORY_METRIC_AUDIT_2026-07-13.md).

For the current Paper 2 definition, Cobb-Douglas is a frozen secondary
construct-sensitivity outcome. A separate factory/APP study could preregister
it as primary only with a validated APP variable/cost ledger and complete
same-contract comparator frontier; that would be a distinct claim and would
not retroactively rescue a null on canonical ReT.

A proposed standalone Stage-A runner was independently audited and rejected. A
later provenance correction found an untracked JSON showing that the rejected
runner nevertheless consumed seeds `9,600,001–9,600,400` without authorization,
an immutable commit receipt, a tape manifest or per-tape ledgers. It used a
non-source demand generator, materially changed Algorithms 1/2, exposed a
latent full-path variability parameter to the policy, and omitted the complete
resource-matched frontier. The machine record is
[`factory_app_cd_gate_v1_rejection.json`](factory_app_cd_gate_v1_rejection.json);
the raw JSON is quarantined, all 400 seeds are burned, and the replacement
protocol remains blocked before preregistration. Its reported numbers are not a
scientific result or a valid null.

## Garrido 2024 claim boundary

**Source claims.** The 2024 paper is explicitly a preliminary, theoretical, exploratory analysis. It:

- gives a 12-step archetype for building a SCRES DES, including material/information-flow data, risk identification, assumptions, verification, validation, experimental design, warm-up/run length and metric construction;
- depicts conventional repeated DES as an open-loop process that does not retain lessons across runs and proposes a learning mechanism between decision inputs and simulation outcomes;
- uses AP, RP, DP and fill rate as example SCRES drivers, while stating that other drivers could be chosen;
- proposes a neuron-style combination of driver signals, weights and an activation function, and introduces a learning-centered definition of SCRES;
- identifies backpropagation networks, Kolmogorov-Arnold networks and simulation-optimization/reinforcement learning as candidate families.

Those are conceptual proposals, not evaluated policies or parameter values.

It does not:

- validate an MFSC observation or action contract;
- demonstrate adaptive-over-open-loop value;
- demonstrate neural-over-classical value;
- specify retained-learning controls or equal-compute estimands;
- license privileged simulator state;
- license replacing the thesis endpoint with a training reward.

It also states that a hypothetical supply chain may be used when real information is incomplete, while emphasizing material and information flows as essential input data and verification/validation as essential steps.

**Project inference.** This supports studying a disclosed hypothetical extension, but it does not validate that extension as Garrido/MFSC physics. A thesis-product claim still requires units, conservation, an independently fixed plausible envelope and domain validation.

## v0 draft claim boundary

**Draft contents.** The v0 manuscript is a working draft, not a source of confirmed results. It contains:

- placeholder abstract, results, and figures;
- proposed hypotheses about recovery time, volatility, and learning;
- a proposed SimPy/Gymnasium reconstruction of the 13-operation DES with hourly internal dynamics and weekly (168 h) decisions;
- a stated 15-dimensional baseline observation, a five-dimensional Track A action with inventory/reorder/shift controls, and mutable downstream-dispatch extensions;
- a proposed Cobb-Douglas-style sequential reward `SC^0.60 * BC^0.25 * AE^0.15`, which is not the canonical thesis endpoint;
- PPO/RecurrentPPO, HSN and DKANA/KAN architecture prose, including unresolved citations and drafting comments;
- numerical validation/results prose whose provenance must be checked against repository outputs before use;
- no audited adaptive-headroom or retained-learning result.

**Project claim-routing rule.** Any manuscript statement that conflicts with a terminal verdict JSON, canonical metric contract, or corrective audit is superseded by those machine artifacts. A smooth RL reward may be used for optimization only if frozen in advance; the scientific verdict must still be computed with canonical order-level ReT and all guardrails. This is project governance, not a claim made by the draft itself.

## Current source-derived candidate boundary

| Family | Source status before simulation |
|---|---|
| Multi-ration product mix/substitution | Materially new and anchored by the documented 21-product reality, but blocked on product/BOM/substitution/signal facts. |
| Lead-time-dependent capacity reservation | Lead time is an acknowledged future direction; scarce advance reservation and its signal are introduced and blocked on operational facts. |
| Lateral CSSU transshipment | No thesis-native lateral lane; blocked unless Garrido confirms it and it is not dominated by central resupply. |
| Multimodal Op10/Op12 choice | Land/air/water choice is mentioned, but every mode-specific physical/resource parameter is absent; blocked on domain facts. |
| Censored demand/active reporting | Current DES records orders; hidden demand and active sensing are introduced and blocked on system-of-record facts. |
| Admission/abandonment | Current orders backorder rather than physically expire; blocked absent mission deadlines and explicit rejection authority. |
| Reporting lag alone | A strict information subset of Program G/H and cannot create value without a new irreversible multi-stage authority. |
| Mission loadout/carried autonomy | Pack duration, mass and typical personal carry are thesis facts, but cohort allocation, signal timing, demand accounting and return/transfer physics are absent; current-kernel action liveness and headroom are exactly zero. |
| Op7 inspection effort | The thesis already specifies fixed quality control, R14 `p=.03/.08` defects and Op6 rework. Variable inspection effort, effort-to-sensitivity/specificity, a persistent lot state, leading signal and escaped-defect consequence are absent; current-kernel incremental headroom is exactly zero. |
| Component-specific R13 / Op4 kit balancing | Twelve raw materials and 12 R13 deliveries are thesis facts, but the executable model aggregates them and provides neither component pipelines nor a finite mixed-load Op4 action. Current-kernel headroom is exactly zero; implementation is blocked on BOM/load/expedition facts. |
| R14 detected-lot disposition | Return to Op6 is thesis-native and fixed. Rework-versus-replacement authority, time/yield/material differences and persistent risk attribution are absent; current-kernel headroom is exactly zero and current rework weakly dominates discard. |

No row in this table authorizes PPO. A candidate first needs a preregistered, resource-restricted clairvoyant ceiling, observable classical conversion, null cell, guardrails, and a complete action-trajectory/static-frontier audit.
