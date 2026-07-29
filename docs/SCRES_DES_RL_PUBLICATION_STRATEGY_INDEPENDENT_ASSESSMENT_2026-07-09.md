# SCRES-DES-RL: independent scientific and publication strategy

**Assessment date:** 9 July 2026  
**Decision horizon:** 16 months  
**Repository/branch audited:** `scres-ia`, `codex/garrido-replication-experiments`, HEAD `ba65772`  
**Primary endpoint:** Garrido Excel ReT (`order_ret_excel_mean` / `ret_excel`)  
**Recommendation:** Strategy C — close and submit Paper 1 first; pursue Paper 2 only after its scientific gate survives.

This assessment starts from the primary artifacts: the Garrido-Ríos thesis, the original Excel workbooks, Garrido et al. (2024), the Garrido proposal/draft, the repository history and raw CSV/JSON evidence, and a page-by-page visual inspection of the current 51-page manuscript. Journal information was checked against current publisher pages and, where stated, 2024 JCR/CiteScore/SCImago data on 9 July 2026. Publisher turnaround figures are descriptive averages, not promises; “first decision” often includes desk rejection.

Epistemic labels used below:

- **Verified fact:** directly reproduced from a primary workbook, CSV/JSON, source code, PDF, or publisher page.
- **Source-supported conclusion:** an interpretation closely entailed by those facts.
- **Committee inference:** the publication-strategy judgment made from those facts.
- **Recommendation:** an action proposed under the 16-month objective.

## 1. Executive verdict

The project demonstrates one publishable scientific contribution now: in a thesis-grounded discrete-event supply-chain benchmark, a learned state-feedback controller improves Garrido Excel ReT and a concordant operational metric panel when the decision contract reaches downstream dispatch. The confirmed headline is PPO `0.0058977` versus the prespecified static comparator `0.0054596`, a difference of `+0.0004382`, with all ten training-seed summaries positive. Fill, backlog, service-loss area, CVaR, and completion/recovery/delivery tails point in the same operational direction. This is evidence of **adaptive recovery under an expanded decision contract**. It is not evidence of prediction, causal prevention, advance anticipation, organizational learning, cross-campaign retention, KAN superiority, or general RL superiority.

The Garrido draft tells a broader story: resilience as accumulated learning, represented by `R_t=f(S_t,D_t,L_{t-1})`, with ANN/RNN/RL, predictive accuracy, and hypotheses H1–H4. The primary sources do not support that story. Garrido et al. (2024) is explicitly exploratory and implements no policy, predictor, retention mechanism, or held-out test. The thesis DES is historically state-dependent, but state dependence is not learning. Existing retained-versus-reset evidence is too small for a central path-dependency theory. Paper 1 must therefore replace `L_{t-1}` rhetoric with a conventional state-feedback control formulation and keep policy parameters frozen during held-out evaluation.

The best route for Thomas is **Strategy C**: run two targeted Paper 1 closure tests, correct provenance and claim language, compress the manuscript, obtain Garrido’s domain/permissions sign-off, and submit to **International Journal of Production Research**. Cascade rapidly to **Computers & Industrial Engineering**, then **Simulation Modelling Practice and Theory** if necessary. Do not let Paper 2 delay this submission.

Paper 2 is scientifically promising but conditional. Its current result says that, in an engineered extreme R21 cell, a frozen heterogeneous reserve posture plus adaptive 8D PPO matches the observed 11D result at current precision and strongly outperforms no-buffer 8D. It does not show that weekly dynamic buffers, anticipation, or RL posture optimization are needed. Paper 2 proceeds only if a held-out classical per-operation reserve frontier, route-aware replenishment, captured order targets, actual-inventory holding exposure, and a compact regime/horizon matrix preserve a distinct slow–fast design contribution.

Recommended authorship is Thomas first and corresponding author, Garrido second, conditional on substantive domain validation, theory revision, permissions clarification, manuscript review, and final accountability. A solo Thomas paper is a contingency only if Garrido formally declines authorship and all rights are independently resolved.

## 2. What is actually proven

### 2.1 Primary-source lineage

1. **Garrido-Ríos (2017):** a 13-operation military food supply-chain DES, nine risk processes, static strategic-inventory and shift interventions, and the piecewise Excel ReT metric.
2. **Garrido et al. (2024):** an exploratory conceptual motivation for closing simulation with AI; no implemented controller, predictor, online update, cross-episode memory test, or performance result.
3. **Current project:** a reconstructed SimPy/Gymnasium benchmark, dense static and heuristic comparators, learned state-feedback policies, common-random-number experiments, forensic workbook replay, and extensive falsification/claim-boundary work.

The thesis and workbooks support a **thesis-grounded reconstruction with forensic workbook replay and throughput checks**. They do not support calling the environment a fully validated empirical digital twin. The thesis reports 42 verification checks and eight annual throughput comparisons, with individual gaps of approximately `-21.6%` to `+14.1%`; it does not define or accept the `±15%` criterion currently stated in the draft.

### 2.2 Claim-status table

| Claim | Status | Verified basis | Publication boundary |
|---|---|---|---|
| PPO Track B improves Excel ReT against the frozen dense-frontier winner | **Supported, with an inference-design repair required** | PPO `0.005897733`; static `0.005459568`; delta `+0.000438164`; 10/10 training-seed means positive in `docs/track_b_q1_stats_2026-07-02_final_10seed/` | Keep the descriptive result. Replace the training-seed-only CI with a fully crossed, dependence-aware held-out evaluation before submission. |
| The direction is operationally coherent | **Supported** | Order-level ReT `+0.000426`; flow fill about `+0.295`; lower backlog/service-loss AUC and materially lower CTj/RPj/DPj tails | Report a prespecified metric panel; do not reduce the paper to one scalar. |
| The result is not driven by privileged forecasting | **Supported, narrowly** | No-forecast and masking tests retain the main direction; a true-regime lookup does not reproduce PPO | Say the demonstrated gain does not require the tested forecast/regime inputs. Do not say prediction is generally useless. |
| PPO is Pareto non-dominated under the tested resource-price assumptions | **Bounded** | ReT/resource panels and post hoc dispatch-price sensitivity | No unconditional cost victory; unpriced dispatch/inventory/capacity remains a boundary. |
| Current and increased regimes preserve a positive sign | **Supported after provenance repair** | Same-sign h104 evidence exists, but one manuscript bundle mixes PPO/static random tapes | Replace the table, prose, and heatmap with one coherent same-tape source. |
| Severe disruption is a service-floor boundary | **Supported boundary/null** | Severe-regime results compress policy differences | Use as a boundary condition, not as failure to generalize that should be hidden. |
| Track A establishes that upstream-only authority is insufficient | **Bounded** | Track A is negative/weak across several screens | Retain as a boundary result, but correct protocol labels, duplicated rows, and 3- versus 5-seed descriptions. |
| Downstream dispatch access is the causal mechanism | **Not yet identified** | Existing E4 wrappers are misnamed: “shift_only” controls dimensions 0–5; “downstream_only” freezes only shift and retains other controls; arms use different comparators | Run a corrected common-comparator factorial or downgrade to “the extended Track B contract permits a benchmark win.” |
| The 147-cell frontier is a full 8D static frontier | **Unsupported wording** | It is a dense downstream `shift × Op10 × Op12` frontier; upstream checks are local | Call it a dense downstream static frontier, never a full 8D exhaustive frontier. |
| SAC and TD3 establish algorithm invariance | **Preliminary screen only** | Small-scale screens replicate the direction | Do not say “regardless of algorithm.” No confirmatory tournament is needed for Paper 1. |
| KAN is superior or scientifically necessary | **Unsupported** | Real-KAN 11D roughly matches PPO 11D; KAN 8D is absent | Architecture is not the novelty. Keep KAN as a sidecar or omit it. |
| The model predicts real SCRES outcomes more accurately | **Unsupported** | No separate predictor, real adaptive-decision dataset, calibrated forecast target, or out-of-sample real-world validation | Delete “predictive accuracy” from RQ, objective, title, hypotheses, and abstract. |
| PPO demonstrates accumulated learning/path dependency across disruptions | **Unsupported as a central theory** | Retained-versus-reset effect is small; current held-out headline freezes policy parameters | Use adaptive state feedback. Cross-campaign learning remains future work. |
| Ruta B demonstrates causal prevention/anticipation | **Retracted** | Splice gate failed negative controls; timing and event-anticipation controls were null | Preserve in the audit registry, not as a positive paper result. |
| Dynamic 11D weekly reserve scheduling adds value | **Unsupported/current null** | Dynamic 11D `0.340164`; globally frozen posture `0.340605`; difference `+0.000440`, CI spanning zero | Say “no detected dynamic advantage at current precision,” not “equivalent,” unless a margin and TOST are prespecified. |
| Strategic reserve headroom exists under natural/moderate R21 | **Unsupported/null** | Natural and moderate regimes are effectively null | Paper 2 must present a regime boundary, not general prevention. |
| Strategic reserve headroom exists in the engineered extreme R21 cell | **Supported but physically conditional** | No-buffer 8D `0.311676`; fixed-posture 8D `0.340605`; delta `+0.028928`, CI95 `[+0.016283,+0.041574]`; 5/5 seeds and 117/120 episodes positive | Re-test after route-aware replenishment and actual-inventory cost accounting. |
| RL is required to optimize the reserve posture | **Unsupported** | The fixed posture was distilled from 11D policies; no calibration-only classical per-operation frontier exists | Mandatory Paper 2 gate. If classical search matches it, claim reserve-design value, not RL necessity. |
| The Excel ReT metric is bounded in `[0,1]` | **False as a universal statement** | Exact workbook replay yields values above one, including `Raw_data1 CF1!U2835=1.000155` and `Raw_data2 CF12!AA2086=160.2564` | Preserve exact Excel ReT as primary, add bounded sensitivity and outlier counts. Canonical five-seed Track B order ledgers contain no out-of-range values and are unchanged by clipping. |

### 2.3 What the Garrido draft says versus what the evidence supports

| Draft proposition | Evidence verdict | Required disposition |
|---|---|---|
| `R_t=f(S_t,D_t,L_{t-1})` makes resilience a learning-dependent property | `L` is not operationalized or isolated by the headline experiment | Replace in Paper 1 with `X_{k+1}=T(X_k,A_k,D_k,ξ_k)`, `O_k=φ(X_k)`, `A_k~πθ(.|O_k)`, with `θ` frozen in held-out evaluation. |
| ANN, RNN, KAN, and RL are alternative “neural learning algorithms” | Category error: architectures/approximators, a learning paradigm, and an optimizer are mixed | State explicitly: MLP/KAN/RNN are function approximators; PPO/RL is the control-learning procedure; backpropagation is optimization. |
| H1: neural models shorten recovery | Too broad and confounds policy flexibility with learning | Test PPO versus strong static/rule policies under identical information, actions, tapes, and resource accounting. |
| H2: successive exposure creates a learning curve | Not established | Remove from Paper 1. Reintroduce only after a preregistered persistent-versus-reset held-out campaign test. |
| H3: neural models reduce volatility | Directionally compatible but not attributable to “neural learning” as a class | Test tail/dispersion outcomes for the named policy and contract, not all neural models. |
| H4: accumulated knowledge causes path dependency | Not established; DES state history already creates path dependence without learning | Future work only. |
| Predictive accuracy improves | No predictor or real target validation | Delete rather than build a predictor solely to rescue wording. |
| DKA/KAN is the contribution | No symmetric or confirmatory architecture evidence | Remove from the main contribution. |

### 2.4 Where the novelty actually resides

**Committee inference:** novelty ranks as follows.

1. **Primary:** a decision-contract/benchmark mechanism — the value of learning depends on whether the controller can act at the operational bottleneck.
2. **Secondary:** benchmark depth and falsification discipline — thesis/workbook lineage, dense static competition, CRN evaluation, full operational panels, forecast/lookup controls, severe boundaries, and explicit retractions.
3. **Secondary:** a reproducible audit methodology for separating metric fidelity, model verification, calibration, and claim status.
4. **Conditional Paper 2 novelty:** a slow–fast design separating strategic reserve posture from fast adaptive recovery, if classical and physical gates survive.
5. **Not novel here:** PPO, MLP, DES–RL integration, backpropagation, KAN, generic “AI-enhanced SCRES,” or a universal learning theory.

Direct prior art is decisive: IJPR has already published a system-dynamics/RL supply-chain recovery paper ([Bussieweke et al., DOI 10.1080/00207543.2024.2383293](https://doi.org/10.1080/00207543.2024.2383293)); a 2025–2026 Journal of Simulation review concludes that DES–RL is already established in supply-chain analysis ([DOI 10.1080/17477778.2025.2500393](https://doi.org/10.1080/17477778.2025.2500393)). Therefore, “first DES–RL SCRES model” and equivalent novelty language are prohibited.

## 3. One-paper versus two-paper decision

| Strategy | Clarity | Defensible novelty | Remaining burden | Contradiction risk | Salami-slicing risk | Plausible venues | 16-month outlook |
|---|---|---|---|---|---|---|---|
| **A. One omnibus paper:** Track A, Track B, prevention boundary, Track B-P, retention | **Low.** Four different questions and evidence standards compete for the spine. | Diluted; the decision-contract result disappears inside “learning SCRES.” | **Very high:** corrected mechanism and inference plus reserve frontier, physics, costs, theory reconciliation | **Very high:** strong recovery, retracted prevention, weak retention, and fixed posture matching dynamic 11D coexist awkwardly | Low formal slicing risk, but high “several incomplete papers stapled together” risk | No clean fit; too diffuse/long for IJPR and not methodologically novel enough for EJOR | **Lowest** probability of acceptance or advanced revision |
| **B. Two papers pursued now:** recovery/action contract + reserve/two-stage design | Medium-high if strictly separated | Strong for P1; conditional for P2 | **Very high concurrently**; P2 is not independently closed | Medium; P2 can undermine a generic learning narrative | **High** until P2 has its own classical frontier, corrected physics, costs, RQ, and evidence | P1: IJPR/C&IE; P2: C&IE/SMPT | Plausible, but likely to delay the closest-to-submission paper |
| **C. Submit P1 after targeted closure; gate P2** | **High** | Concentrates the demonstrated action-contract/benchmark novelty | Moderate for P1; P2 effort is spent only after a go/no-go result | **Lowest**; unsupported theories remain bounded | Manageable because a later P2 must rest on slow–fast reserve design, not reused PPO results | P1: IJPR → C&IE → SMPT; conditional P2: C&IE → SMPT → JOS | **Highest** chance that P1 is accepted or in advanced revision within 16 months |

### Decision

**Choose Strategy C.** “Submit Paper 1 now” means a short, targeted closure phase followed by submission; it does not mean sending the current PDF. The current 51-page manuscript has scientific provenance defects, incomplete authorship/funding/acknowledgment fields, a spillover abstract, broken equation placement, small/crowded exhibits, an appendix figure interrupting the references, and `[TO COMPLETE]`/`[TO UPDATE]` placeholders. Paper 2 work begins only after Paper 1 is submitted.

If the corrected factorial fails to identify a downstream-access contrast, Paper 1 remains publishable but the title and contribution must be downgraded to a bounded extended-contract benchmark, and **Computers & Industrial Engineering or SMPT becomes more realistic than IJPR**.

## 4. Recommended Paper 1 architecture

### 4.1 Title, claim, research question, and objective

**Provisional title, conditional on the corrected factorial:**

> **Decision-Contract Alignment for Adaptive Supply-Chain Recovery: A Thesis-Grounded DES–RL Benchmark**

**Fallback title if the downstream-access gate fails:**

> **Adaptive Supply-Chain Recovery under an Extended Downstream Decision Contract: A Thesis-Grounded DES Benchmark**

**One-sentence claim:**

> Under identical held-out disruption streams, an eight-dimensional PPO policy improves Garrido Excel ReT and a concordant operational panel relative to a prespecified static comparator when the controller has downstream dispatch authority; the gain may be attributed to decision-contract alignment only if a corrected factorial isolates a positive downstream-access contrast.

**Research question:**

> Under what decision-contract conditions does an adaptive controller improve out-of-sample supply-chain resilience relative to strong fixed and rule-based policies in a thesis-grounded discrete-event simulation?

**Objective:**

> To reconstruct and audit the Garrido-Ríos supply-chain model, define nested operational decision contracts, and determine whether a learned controller produces robust recovery gains—and through which actionable decision rights—under common held-out disruption streams.

### 4.2 Falsifiable hypotheses and stop rules

**H1 — Adaptive recovery advantage.** On a fully crossed held-out tape set, PPO under the expanded downstream contract achieves higher mean Garrido Excel ReT than the prespecified static comparator, with a positive lower confidence bound under checkpoint- and tape-aware inference.

- **Promote:** two-way/random-effects CI is wholly above zero, all or nearly all checkpoint means retain the sign, and no single tape cluster drives the result.
- **Stop/downgrade:** if the interval includes zero or the sign is unstable, the Q1 headline win does not survive.

**H2 — Downstream decision-contract effect.** Adding Op10/Op12 dispatch authority to an otherwise identical upstream-and-shift contract produces a positive held-out Excel ReT contrast under a common comparator.

- **Promote:** the nested dispatch-access contrast has a dependence-aware CI wholly above zero and consistent seed direction.
- **Stop/downgrade:** remove causal/mechanism language and use the fallback title if this contrast is null or negative.

**H3 — Operational concordance.** The ReT gain is accompanied by prespecified improvement in fill rate, backlog, and service-loss AUC without an offsetting deterioration in principal CTj/RPj/DPj tails.

- **Promote:** directional concordance survives multiplicity-aware reporting and effect-size intervals.
- **Stop/downgrade:** a ReT-only gain is treated as metric-specific rather than operationally general.

**H4 — Information robustness.** A no-forecast controller remains superior to the prespecified static comparator on held-out Excel ReT.

- This tests independence from the tested privileged input. It does not establish equivalence to the full-information controller unless a smallest effect margin is prespecified and a valid equivalence test passes.

**Secondary boundary test.** The sign should remain positive in the thesis-defined current and increased regimes. Severe disruption is reported as a service-floor boundary, not forced into a universal robustness hypothesis.

The original H2/H4 concerning cumulative exposure and path dependency are removed. They are different scientific questions and are not necessary for Paper 1.

### 4.3 Numbered contributions

1. **Auditable thesis-grounded benchmark.** The study links topology, disruption logic, decision variables, and the primary ReT formula to the thesis and original workbooks while distinguishing software verification, formula replay, historical calibration, and external validation.
2. **Decision-contract test of adaptive value.** The paper asks whether adaptive control creates value only when the controller can affect the binding operational bottleneck; PPO itself is not presented as the contribution.
3. **Strong comparison and inference design.** PPO is evaluated against a dense downstream static frontier, rule-based and privileged-information controls, shared exogenous tapes, and checkpoint-by-tape crossed inference.
4. **Multi-metric operational interpretation.** Excel ReT is accompanied by fill, backlog, service-loss AUC, CVaR, CTj/RPj/DPj tails, and explicit resource-price sensitivities.
5. **Claim-boundary contribution.** Negative and boundary evidence separates adaptive recovery from prediction, causal prevention, anticipation, algorithmic novelty, and cross-campaign organizational learning.

### 4.4 Risk engine: risks to use and how to use them

**Recommendation:** preserve the thesis’s nine named risks for continuity, but encode them in a factorized, auditable exogenous event ledger. The policy must mitigate consequences; it must not turn exogenous risks “on” or “off.” Every event record should contain `risk_id`, family, sampled start/end, affected operation/link, magnitude and units, occurrence parameter, recovery distribution, observability, scenario/tape ID, random-stream ID, source regime, and any project-specific frequency or impact multiplier.

| Risk | Thesis mechanism and current occurrence | Impact and deactivation | What the controller may observe | Paper 1 use |
|---|---|---|---|---|
| **R11** | Breakdown at one of Op5/Op6; one sampled event in a discrete `1–168 h` window (increased window `42 h`) | Affected operation unavailable; exponential repair, mean `2 h`; deactivate when repair completes | Realized outage flag, elapsed downtime, resulting queue/WIP—not future repair draw | Frequent operational capacity shock; useful for recovery behavior, not prevention |
| **R12** | Delayed supplier contracts at Op1; 12 Bernoulli trials, `p=1/11` current, `4/11` increased | Each delayed contract receives `+168 h`; “deactivation” is completion of the delayed procurement item | Outstanding procurement/pipeline delays once realized | Slow upstream delay; preserve for thesis continuity and long-horizon exposure |
| **R13** | Delayed planned deliveries at Op2; 12 trials, `p=.10` current, `.40` increased | Each affected delivery receives `+24 h`; clears when the delivery arrives | Realized pipeline delay and available ETA, if operationally knowable | Upstream logistics delay; do not leak which future deliveries will be delayed |
| **R14** | Defective Op7 output; Binomial rework, `p=.03` current, `.08` increased | Defective units enter rework; effect ends when rework completes | Realized defect/rework queue and downstream shortfall | High-frequency yield/rework shock; model as flow loss/rework, not facility outage |
| **R21** | Correlated natural-disaster outage at Op3,5,6,7,9; one event per `1–16,128 h` window (increased `4,032 h`) | Each affected operation has an exponential repair, mean `120 h`; restore each independently and reference-count overlaps | Realized node outages, route availability, queues and inventory—not regime or future onset | Multi-node infrastructure shock; also the specific mechanism for conditional Paper 2 |
| **R22** | Attack/outage at one selected link/operation among Op4,8,10,12; current window `4,032 h`, increased `1,344 h` | Selected link unavailable; exponential repair, mean `24 h` | Realized route outage and downstream pressure | Direct downstream logistics interruption; central to the actionability question |
| **R23** | Op11 forward-support outage; current window `8,064 h`, increased `1,344 h` | Op11 unavailable; exponential repair, mean `120 h` | Realized outage and backlog | Downstream node outage; important to test Op10/Op12 decision rights |
| **R24** | Priority demand pulse of `2,400–2,600` units; current window `672 h`, increased `336 h` | Pulse enters demand; effect clears through service/loss rules rather than a repair clock | Realized demand and backlog; no future pulse schedule | Demand shock; keep distinct from physical outages |
| **R3** | Black swan disables Op5,6,7,9; current window `161,280 h`, increased `80,640 h` | Fixed `672 h` downtime; restore after the fixed duration with overlap-safe logic | Realized multi-node outage | Rare/OOD stress component; report separately and do not oversample in the headline evaluation |

Implementation rules:

1. Sample an offset within each occurrence window and then advance through the remainder of that window, preserving one-event-per-window thesis semantics.
2. Use **reference-counted downtime** so overlapping risks cannot reactivate a facility or route prematurely.
3. Separate occurrence frequency from impact. Frequency multipliers alter windows/Binomial probabilities; impact multipliers alter repair duration, delay, rework amount, or demand-pulse size. Never use an unlabeled generic “severity” multiplier.
4. Freeze a common exogenous event tape and keep policy stochasticity on a separate stream.
5. The headline observation excludes true regime, future onset, future duration, and future demand pulses. Privileged information is diagnostic only.
6. Current and increased thesis regimes alter frequency/probability, not impact. The severe cells used by this project are extensions and must be labeled as synthetic stress tests.
7. Freeze and disclose source inconsistencies rather than silently harmonizing them: Op2/R13 cadence, alternative `2,000–2,500` versus `2,400–2,600` demand descriptions, and inconsistent annual-capacity tables.

### 4.5 Decisions, observations, constraints, reward, and architecture

#### Decision epoch and internal time

- DES resolution: event/hour level.
- Policy epoch: every `168 h`, matching the weekly cycle already used by the benchmark.
- Garrido should validate the physical meaning of a weekly Op10/Op12 target. If downstream releases occur continuously, the weekly action is a release-policy parameter, not instantaneous material creation.

#### Eight-dimensional Track B decision contract

1. Op3 shipment/order quantity control.
2. Op9 shipment/order quantity control.
3. Op3 reorder-point control.
4. Op9 reorder-point control.
5. Op5 quantity/control.
6. Assembly shift level (`S1`, `S2`, or `S3`).
7. Op10 downstream dispatch/release quantity.
8. Op12 downstream dispatch/release quantity.

Every action must obey nonnegativity, available stock, active facility/route status, process lead time, frozen thesis-grounded bounds, and any minimum dwell/ramp rule. An action changes a feasible operating policy; it does not teleport stock or cancel a disruption.

#### Main no-forecast observation

- inventories and pipeline stock at relevant nodes;
- WIP and queue pressure;
- backlog amount, age, and recent growth;
- rolling/cumulative fill and service-loss measures;
- Op10/Op12 downstream pressure;
- current shift level and utilization;
- realized facility/link outage flags and elapsed downtime;
- recent disruption indicators, recency, and exponentially weighted history;
- calendar/order-cycle phase;
- outstanding orders and operationally available ETA information;
- previous action.

Exclude the true regime label, future disruption ledger, hidden repair draws, and privileged forecasts from the headline contract.

#### Architecture

- **Primary controller:** PPO with a small MLP. Treat the MLP as a function approximator, PPO as the control-learning procedure, and backpropagation as the optimizer.
- **Evaluation contract:** freeze `θ` for the held-out campaign. The paper tests state-feedback adaptation, not online organizational learning.
- **Recurrence:** not required unless an observability/POMDP audit shows that retained history contains predictive information absent from the frozen observation.
- **KAN/SAC/TD3:** optional sidecars, not a contribution or submission blocker.

#### Training objective and evaluation

Keep the frozen `control_v1` service-loss-plus-shift-charge reward for the confirmatory Paper 1 experiments; changing it now would move the target. Evaluate with:

- exact Garrido Excel ReT as primary;
- clipped `[0,1]` ReT sensitivity and branch/outlier counts;
- fill rate, backlog, lost orders, service-loss AUC;
- CTj, RPj, and DPj tails;
- CVaR and dispersion;
- shift, dispatch, inventory, and other resource exposures.

Do not use ReT simultaneously as the sole reward and sole validation criterion. Do not claim a cost victory unless all material resource use is priced or constrained equivalently.

### 4.6 Recommended section structure

1. Introduction: practical problem, decision-contract gap, RQ, bounded contributions.
2. Related research and framing: DES resilience, adaptive control, decision rights/actionability, distinction from prediction/prevention/retained learning.
3. Case foundation and provenance: topology, risks, workbook ReT, reconstruction/calibration/validation ladder, case boundary.
4. DES environment and decision contracts: transition/event logic, observations/actions/constraints, nested contracts, PPO/reward.
5. Experimental design and inference: comparator selection, CRN tapes, corrected factorial, crossed evaluation, promotion/stop rules.
6. Results: primary panel, factorial, information controls, regime boundary, resource/Pareto analysis.
7. Discussion: mechanism, production/managerial implications, boundaries, what is not established.
8. Conclusion.

### 4.7 Essential exhibits and body/supplement allocation

**Main text: maximum eight core exhibits.**

1. Figure 1 — topology, risk locations, observations, and decision rights by contract.
2. Table 1 — provenance and validation ladder.
3. Table 2 — hypotheses, contrasts, endpoints, inference units, promotion/stop rules.
4. Figure 2 — crossed held-out effects by checkpoint and evaluation tape.
5. Table 3 — primary and operational metric panel with uncertainty.
6. Figure 3 — corrected action-contract factorial in absolute performance and nested contrasts.
7. Figure 4 — regime/service-floor/resource-Pareto boundary panels.
8. Table 4 — falsification and claim-boundary matrix.

**Supplement:** exact workbook formula and outlier audit; all 147 downstream static cells; comparator-selection protocol; crossed inference details; Track A; no-forecast/masking; true-regime lookup; training curves/hyperparameters/checkpoint rules; SAC/TD3 screen results; full risk and tape manifest; tail distributions; artifact hashes; prevention retraction provenance.

Target a main body of roughly `9,000–10,000` words. The present 51-page PDF is an evidence archive, not the submission form.

### 4.8 Prohibited or unsafe language

Do not write: “first DES–RL SCRES model”; “RL causes resilience”; “the supply chain learns from prior disruptions”; “organizational learning”; “path-dependent learning”; “predictive accuracy”; “prevention”; “anticipation”; “independent of algorithm choice”; “validated digital twin”; “empirically validated military supply chain”; “worst case” for p99; “equivalent” for an interval spanning zero; “cost saving” without a full cost/resource model; “downstream dispatch is the causal mechanism” before H2 passes; or “full 8D static frontier” for the downstream enumeration.

### 4.9 Three abstract options

The conservative abstract can survive a failed H2 gate after minor edits. The ambitious and IJPR-tailored abstracts may be used only if the corrected factorial and crossed evaluation pass. All numbers must be regenerated from the frozen final bundle.

#### Conservative abstract — 179 words

Adaptive controllers are frequently introduced into supply-chain simulations without establishing whether their gains arise from learning, additional decision authority, or favorable evaluation. We reconstruct the Garrido-Ríos military food supply-chain discrete-event model, audit its workbook-derived order resilience metric, and compare fixed and learned policies under common disruption streams. The primary comparison uses Garrido Excel ReT and an operational panel covering fill rate, backlog, service-loss area, and process-time tails. In the current ten-training-seed evidence bundle, an eight-dimensional PPO policy attains ReT 0.005898 versus 0.005460 for the prespecified static comparator, a difference of 0.000438, with positive seed-level differences in all ten seeds. The operational panel points in the same direction, and a no-forecast controller retains a positive advantage. We treat these results as evidence of adaptive recovery within the simulated decision contract, not as evidence of prediction, prevention, or organizational learning. A corrected factorial experiment and a fully crossed checkpoint-by-evaluation-tape analysis are specified as confirmatory gates for attributing the gain to downstream dispatch authority. The study offers an auditable benchmark and a bounded account of when learned control improves simulated supply-chain recovery.

#### Ambitious but defensible abstract — 192 words

When does reinforcement learning improve supply-chain resilience rather than merely add a more flexible policy class? We address this question in a thesis-grounded discrete-event benchmark whose order flows, disruptions, and resilience calculation are traced to the Garrido-Ríos model and original workbooks. Policies are evaluated on identical held-out disruption streams against a dense downstream static frontier, rule-based controls, privileged-information falsification tests, and restricted-action ablations. The current evidence shows that an eight-dimensional PPO controller increases the primary Garrido Excel ReT from 0.005460 to 0.005898 across ten training seeds, while improving fill, backlog, service-loss area, and process-time tails. The gain survives removal of privileged forecasts; a true-regime lookup table does not reproduce it; and upstream-only control remains a boundary result. We therefore locate the contribution in decision-contract alignment: adaptive value emerges only when the controller can act on the operational bottleneck that determines recovery. This mechanism claim is promoted only if a corrected common-comparator factorial isolates downstream dispatch access and crossed inference over checkpoints and evaluation tapes remains positive. The paper advances resilience modeling by linking learning value to actionable decision rights, while explicitly excluding unsupported claims about prevention, anticipation, prediction, and cross-campaign organizational learning.

#### IJPR-tailored abstract — 199 words

Production systems do not benefit from adaptive analytics merely because a learning algorithm is embedded in a simulator; the controller must possess decision rights that can alter the binding operational constraint. This study examines that proposition in a reproducible reconstruction of the Garrido-Ríos military food supply chain. We audit the original Excel resilience calculation, define upstream and downstream action contracts, and evaluate PPO policies against a dense static frontier using common disruption streams and seed-aware inference. In the current evidence package, the downstream-capable policy improves Garrido Excel ReT from 0.005460 to 0.005898, with positive differences for all ten training seeds. Fill rate, backlog, service-loss area, and process-time tails move in the same operational direction. Forecast masking and a privileged true-regime lookup control indicate that the result is not explained by explicit disturbance forecasts or simple regime-conditioned rules. A corrected factorial design is the promotion gate for the proposed mechanism: adding downstream dispatch authority to an otherwise identical contract must yield a positive held-out contrast under a common comparator. The resulting contribution is not a new reinforcement-learning algorithm, but production-research evidence that actionability and decision-contract design govern whether adaptive control creates resilience value. Severe disruption remains an explicit service-floor boundary.

### 4.10 Manuscript-ready framing passages

**Theory/positioning paragraph:**

> This study does not treat reinforcement learning as a synonym for supply-chain learning. We define adaptation operationally: a state-feedback policy maps currently observable inventory, backlog, capacity, and disruption consequences to feasible recovery actions. Policy parameters are frozen during held-out evaluation. The scientific question is therefore not whether a neural model can fit simulated outcomes, but whether a learned decision rule improves recovery relative to strong fixed and rule-based policies under identical disruption streams, and whether that value depends on the decision rights available to the controller. Cross-campaign retention, prediction, prevention, and organizational learning are separate constructs and are not inferred from within-episode state feedback.

**Contribution paragraph:**

> The contribution is a decision-contract result within an auditable, thesis-grounded benchmark. We show how the apparent value of adaptive control changes when authority is restricted to upstream inventory and capacity decisions or extended to downstream dispatch. The design combines a dense downstream static frontier, common exogenous disruption tapes, checkpoint-by-tape inference, workbook-faithful ReT, and an operational metric panel. This structure allows the analysis to distinguish a genuine recovery advantage from favorable comparator selection, privileged information, metric singularities, and unsupported claims about anticipation or retained learning.

## 5. Recommended Paper 2 architecture — conditional

Paper 2 should not enter full manuscript production until the classical-frontier and physical-accounting gates pass. The following is the architecture to use if they do.

### 5.1 Title, claim, research question, and objective

**Provisional title:**

> **Strategic Reserve Postures and Adaptive Recovery under Recurrent Disruption: A Slow–Fast Design in a Discrete-Event Supply-Chain Model**

**One-sentence claim:**

> Conditional on route-aware replenishment, actual-inventory holding exposure, and a held-out classical per-operation reserve frontier, a fixed heterogeneous strategic reserve posture combined with adaptive recovery captures the observed buffer benefit under an engineered extreme R21 regime, while weekly state-contingent reserve modulation provides no detected incremental value.

**Research question:**

> Under what disruption conditions do strategic reserves create measurable resilience headroom, and can a slow fixed reserve design combined with fast adaptive recovery match or outperform weekly dynamic buffer control?

**Objective:**

> To identify the regime boundary at which strategic inventory becomes valuable, optimize physically feasible per-operation reserve postures using calibration scenarios only, and compare no-buffer, fixed-reserve, and dynamically modulated reserve contracts under common held-out disruption streams and explicit resource accounting.

### 5.2 Falsifiable hypotheses

**H1 — Reserve–regime interaction.** The held-out benefit of reserves is materially larger under a prespecified extreme R21 regime than under natural and moderate intensities.

- Fail if reserve gains do not exhibit a reproducible intensity interaction or exist only after post hoc cell selection.

**H2 — Physically feasible fixed-posture benefit.** After route-aware replenishment and actual time-weighted inventory accounting, the selected fixed posture plus adaptive recovery outperforms no-buffer adaptive recovery on held-out Excel ReT and remains non-dominated in the resilience–resource plane.

- Fail if route correction removes the gain or actual holdings make the posture dominated across the prespecified cost/resource range.

**H3 — Limited value of weekly modulation.** Dynamic 11D control does not exceed the fixed two-stage design by more than a prespecified smallest operationally meaningful difference.

- Use “equivalent” only if the margin is selected before final evaluation and a valid TOST/non-inferiority test passes. Otherwise report “no detected difference at current precision.”

**H4 — Classical-frontier gate.** The interpretation survives a classical per-operation reserve optimizer selected exclusively on calibration scenarios.

- If the classical solution matches the distilled posture, the contribution is the slow–fast reserve-design principle, not RL as the posture optimizer.
- If the classical solution dominates and no distinct mechanism remains, stop Paper 2.

**H5 — Local robustness.** The fixed-posture benefit survives at least one adjacent positive intensity or horizon and the principal physical/cost sensitivities.

- If it survives only in the original exact engineered cell, report a narrow stress-cell boundary result or stop; do not generalize.

### 5.3 Numbered contributions

1. **Reserve-headroom regime map:** identifies where strategic inventory is effectively null and where it becomes operationally material.
2. **Physically compliant reserve-design benchmark:** per-operation reserves are selected under route availability, lead times, captured order commitments, and actual inventory exposure.
3. **Slow–fast resilience architecture:** strategic reserve sizing is separated from online recovery control rather than embedded in an unnecessarily large weekly action space.
4. **Fair test of temporal flexibility:** no-buffer, fixed-posture, and dynamically modulated policies share held-out tapes, resource accounting, and margin-based inference.
5. **Publishable negative result:** absence of weekly state-contingent modulation value becomes a design insight, not a hidden failed architecture.
6. **Explicit mechanism boundary:** the paper concerns strategic headroom under an engineered infrastructure-disruption regime, not generic prevention or event anticipation.

### 5.4 R21 risk design and activation semantics

Paper 2 should use a focused R21 factorial, not a free mixture of all risks.

| Cell | Occurrence window | Repair mean per affected operation | Interpretation |
|---|---:|---:|---|
| Thesis natural/current | `16,128 h` | `120 h` | Expected null/low-headroom anchor |
| Thesis increased-frequency | `4,032 h` | `120 h` | Thesis-defined frequency change |
| Project moderate | Prespecified between the thesis and extreme cells | Prespecified, labeled separately | Adjacent boundary cell; never chosen after viewing final outcomes |
| Engineered extreme | `2,016 h` (`frequency ×8` from current) | `480 h` (`impact ×4`) | Synthetic compounding-starvation stress, not an empirical frequency claim |
| Adjacent positive/horizon check | One locked neighbor or a longer/shorter horizon | Same physical semantics | Tests whether the result is a local regime transition rather than one exact cell |

R21 simultaneously disables Op3, Op5, Op6, Op7, and Op9. Each affected operation receives an independent repair draw; restoration is operation-specific and overlap-safe. Reserve replenishment must respect inbound route and facility availability. If other thesis risks remain active as background, their regime and tapes must be identical across all reserve treatments and declared in advance. For the cleanest mechanism test, first isolate R21, then add one locked thesis-current background sensitivity.

The controller sees only realized outages and operational consequences. It does not see the hazard clock, future onset, future repair draws, or the “extreme” label. Thus a reserve posture is **preparedness**, not prediction; and an agent that reacts after onset is **recovery**, not prevention.

### 5.5 Decisions, observations, physics, objective, and architecture

#### Slow design variables

Choose before the held-out campaign:

\[
(b_3,b_5,b_9)\in[0,1]^3,
\]

where each `b_i` is a fraction of the operation-specific thesis `I_1344` reserve. The currently distilled posture is approximately `(0.1531, 0.2480, 0.2068)` and is a candidate—not an optimum.

The classical frontier must include:

- no buffer;
- thesis levels `I168`, `I336`, `I504`, `I672`, `I1344`;
- homogeneous/common-scalar reserves;
- heterogeneous per-operation search near and beyond the distilled posture;
- reserve-budget-constrained Pareto frontiers.

Selection occurs on calibration seeds only; the posture is frozen before held-out evaluation.

#### Fast decisions and observations

Use the same eight-dimensional recovery contract and no-forecast observation as Paper 1. The fixed posture requires no deployment observation. The 11D comparator adds weekly Op3/Op5/Op9 target outputs but receives no privileged event timing.

#### Non-negotiable physical rules

1. A replenishment request captures its requested quantity/target and lead time when placed; a later target change cannot rewrite an order already in transit.
2. Stock cannot arrive through an unavailable inbound route or facility. Delayed orders wait or follow an explicitly modeled alternative route.
3. Inventory is never injected merely because a lead-time timer expired.
4. Holding exposure is the time integral of actual on-hand stock by operation, not the nominal target fraction.
5. Capacity, switching, shortage, lost-order, and replenishment exposures are logged separately.
6. Legacy versus corrected physics are reported transparently; corrected physics controls the claim.

#### Architecture and optimization

- **Stage 1:** a classical deterministic/derivative-free, Bayesian, or grid-assisted robust optimizer sizes `(b3,b5,b9)` on calibration scenarios under a reserve budget or multiobjective criterion.
- **Stage 2:** the frozen Paper 1 PPO policy controls fast recovery with the chosen posture installed.
- **Comparators:** no-buffer 8D, fixed-posture 8D, classical-best fixed posture + 8D, and dynamic 11D.
- **Inference:** common held-out tapes; checkpoint/tape dependence; formal margin test only if equivalence/non-inferiority wording is essential.
- **RL posture optimization:** not a claim unless the distilled/learned procedure beats the classical calibration-only frontier out of sample.

### 5.6 Recommended section structure

1. Introduction: strategic versus operational resilience decisions and conditional contribution.
2. Related research and slow–fast logic: strategic inventory, adaptive recovery, temporal decomposition, resource commitment.
3. Case and revised reserve physics: topology, R21, routes, captured orders, holding exposure.
4. Two-stage architecture: classical reserve search, fixed selection, 8D recovery, 11D comparator.
5. Experimental design: regime map, calibration/test separation, CRN, cost/resource rules, equivalence margin, stop criteria.
6. Results: headroom map, classical frontier, policy comparison, physical/cost/horizon sensitivities.
7. Discussion: slow–fast implications, null weekly modulation, extreme-regime and topology boundaries.
8. Conclusion.

### 5.7 Essential exhibits and body/supplement allocation

**Main text:** Figure 1 slow–fast architecture and replenishment physics; Table 1 regimes/costs/hypotheses; Figure 2 reserve-headroom map; Table 2 held-out classical frontier; Figure 3 common-tape no-buffer/fixed/dynamic comparison; Table 3 ReT/operational/resource panel; Figure 4 physical, cost, horizon, and adjacent-cell sensitivities; Table 4 margin test and claim boundary.

**Supplement:** complete 3D frontier; optimizer budget/convergence; cross-fitted and within-checkpoint audits; captured-target and disabled-route unit tests; event traces; pre/post-event and replay controls; dynamic-buffer action distributions; natural/moderate nulls; KAN 11D sidecar; inference specification; hashes and compute manifest.

### 5.8 Prohibited or unsafe language

Do not write: “RL prevents disruptions”; “the policy anticipates R21”; “dynamic buffers improve resilience”; “RL is required to optimize the posture” before H4; “fixed and dynamic are equivalent” without a margin test; “reserves improve SCRES generally”; “optimal posture” outside the searched space; “cost-effective” before actual holding/capacity costs; “route-feasible” before correction; “organizational memory”; “path-dependent learning”; “KAN improves performance/interpretability” from the current sidecar; or “robust across regimes” if the effect remains confined to extreme R21.

### 5.9 Three post-gate abstract options

These are templates, not submission-ready factual records. Refresh every number after the classical, route-aware, and actual-inventory reruns.

#### Conservative abstract — 201 words

Strategic inventory can improve disruption performance, but flexible weekly reserve decisions may add complexity without creating additional value. We study this issue in the Garrido-Ríos discrete-event supply-chain model by separating slow reserve design from fast adaptive recovery. Current screening evidence identifies reserve headroom only under an engineered extreme R21 disruption regime; natural and moderate intensities are effectively null. In five-seed confirmatory runs under that extreme regime, an adaptive eight-dimensional policy without buffers attains Garrido Excel ReT 0.311676, whereas the same recovery contract combined with a fixed heterogeneous reserve posture attains 0.340605. The fixed posture also slightly exceeds the dynamic eleven-dimensional policy, 0.340164, although the difference is not statistically resolved and is not claimed as equivalence. These findings support a conditional two-stage interpretation: reserve placement may be a slow design decision, while adaptive control is reserved for recovery operations. Publication of that interpretation requires three additional gates: held-out classical optimization of per-operation reserves, route-aware replenishment that prevents infeasible stock injection, and cost measurement based on actual time-weighted inventory. Until those gates pass, the evidence establishes an extreme-regime headroom result and a null result for weekly modulation, not a general prevention claim or a demonstration that reinforcement learning is needed to size reserves.

#### Ambitious but defensible abstract — 198 words

Supply-chain resilience decisions operate on different time scales: strategic reserves are positioned slowly, whereas disruption recovery requires rapid operational adaptation. We develop a two-stage discrete-event framework that separates these decisions and ask whether weekly state-contingent reserve modulation adds value beyond a fixed heterogeneous posture. The study uses the Garrido-Ríos supply-chain topology and maps reserve headroom across disruption intensity before comparing no-buffer, fixed-buffer, and dynamically controlled buffer contracts on common held-out streams. Existing five-seed evidence reveals a sharp regime interaction. Under an engineered extreme R21 regime, fixed reserves at Op3, Op5, and Op9 increase Garrido Excel ReT from 0.311676 to 0.340605 relative to adaptive recovery without buffers, with positive differences in 117 of 120 episodes. Dynamic eleven-dimensional control reaches 0.340164 and does not show a resolved advantage over the fixed posture. The proposed contribution is therefore a slow-fast design principle, not a claim of event anticipation: strategically calibrated reserves absorb a structural exposure, while the learned policy manages downstream recovery. This interpretation is conditional on a held-out classical per-operation reserve frontier, route-aware lead-time physics, actual-inventory holding cost, and a prespecified non-inferiority or equivalence margin. If these gates fail, the paper stops rather than recasting a simulator artifact as prevention.

#### Computers & Industrial Engineering-tailored abstract — 203 words

Computerized resilience models often enlarge an agent’s action space without testing whether the added temporal flexibility is operationally useful. This study proposes a slow-fast design for reserve positioning and adaptive recovery in a thesis-grounded discrete-event supply-chain environment. A classical optimizer selects fixed reserve fractions at three operations before evaluation, while a learned eight-dimensional controller adjusts recovery decisions during disruption. The design is compared with no-buffer recovery and an eleven-dimensional controller that can modulate buffers weekly, using common random streams, held-out disruption tapes, route-aware replenishment, and time-weighted inventory costs. The motivating evidence is an engineered extreme R21 regime in which a distilled fixed posture raises Garrido Excel ReT from 0.311676 to 0.340605 relative to no-buffer adaptive recovery; the dynamic policy reaches 0.340164 and shows no resolved incremental benefit. Natural and moderate regimes remain boundary-null cases. The paper’s publishable claim is conditional: the fixed two-stage design must remain non-dominated after classical per-operation reserve search and physical and economic sensitivity tests, and any equivalence statement must pass a prespecified margin-based test. The contribution is a computational design and falsification framework for deciding when resilience flexibility belongs in strategic configuration and when it belongs in online control, rather than a claim that reinforcement learning universally prevents disruptions.

### 5.10 Manuscript-ready framing passages

**Slow–fast contribution paragraph:**

> Resilience interventions operate on different clocks. Strategic reserves are committed before a disruption campaign and incur exposure even when no event occurs; dispatch and capacity decisions respond to realized operational conditions. We therefore separate reserve sizing from online recovery control. The first stage selects a physically feasible, heterogeneous reserve posture using calibration scenarios only. The second stage freezes that posture and applies an adaptive recovery policy on held-out disruption streams. This decomposition tests whether weekly reserve modulation contributes information-dependent value or merely approximates a slow design choice that can be made once.

**Boundary paragraph:**

> The reserve result is a regime-interaction finding, not a universal prevention result. Natural and moderate R21 intensities provide null anchors, while the engineered high-frequency, long-duration cell creates compounding starvation in which reserves may have headroom. The policy does not predict or deactivate R21 events. A reserve posture is installed ex ante, and the operational controller reacts to realized system states. Any claim is therefore conditional on the stress regime, replenishment physics, actual inventory exposure, and comparison with a classical fixed-reserve frontier.

## 6. Journal strategy matrix

### 6.1 How to read the journal evidence

All publisher pages below were accessed **9 July 2026**. Impact factors and CiteScores displayed by publishers are labeled by their source year where the publisher supplies it; SCImago quartiles cited here are the latest open 2024 SJR classification, not a prediction of a 2026 JCR category. A publisher’s “submission to first decision” includes desk decisions and therefore substantially understates the likely time for a reviewed paper. Where a page/word rule or acceptance rate could not be verified from an official source, it is reported as **not verified**.

### 6.2 Current metrics, publication rules, and turnaround

| Journal | Current verified metrics/quartile | Scope and contribution bar | Length, supplement, code/data | OA/APC | Official speed; cautious total estimate |
|---|---|---|---|---|---|
| **International Journal of Production Research (IJPR)** | 2024 JIF `7.3`, JCR best quartile **Q1**; 2024 CiteScore `17.3`, best quartile **Q1**; SJR `2.242`; official acceptance `12%` | Novel production/logistics decision aid, exhaustive related work, state-of-art comparison, real-life relevance, and managerial insight are explicit requirements | Hard word/page cap **not verified**; supplementary material supported. A compact `9–10k`-word paper plus replication supplement is strategically appropriate | Hybrid; optional OA APC varies; **no APC** on non-OA route | `3 d` first decision; `61 d` first post-review decision; `21 d` acceptance-to-online. Estimate `7–12 months` to acceptance/advanced revision if reviewed. [Official IJPR page](https://www.tandfonline.com/journals/tprs20/about-this-journal) |
| **Computers & Industrial Engineering (C&IE)** | CiteScore `13.2`; JIF `6.5`; 2024 SCImago SJR about `1.628`, best quartile **Q1** | Computerized methodologies and significant IE applications; originality may reside in the problem, tool use, general approach, and result—not necessarily a new RL algorithm | Hard cap **not verified**; Elsevier supplement/data statement available. Provide code, tapes, ledgers, and one-command tables | OA `USD 3,880`; subscription route **no publication fee** | `5 d` first; `92 d` after review; `206 d` to acceptance; `5 d` online. Estimate `7–12 months`. [Official C&IE page](https://www.sciencedirect.com/journal/computers-and-industrial-engineering) |
| **Simulation Modelling Practice and Theory (SMPT)** | CiteScore `9.8`; JIF `4.6`; 2024 SCImago **Q1** in Modeling and Simulation | Original simulation/modeling methods or applications that significantly inform simulation practice; validation, experimental design, comparison, and hybrid AI are in scope | Hard cap **not verified**; online supplement supported; simulation artifacts and V&V evidence should be central | OA `USD 3,110`; subscription route **no publication fee** | `17 d` first; `96 d` after review; `135 d` acceptance; `11 d` online. Estimate `5–9 months`. [Official SMPT page](https://www.sciencedirect.com/journal/simulation-modelling-practice-and-theory), [SCImago 2024](https://www.scimagojr.com/journalsearch.php?q=12189&tip=sid) |
| **Journal of Simulation (JOS)** | 2024 JIF `1.7`; CiteScore `3.6`, best quartile **Q2**; SJR `0.356`; official acceptance `8%` | DES, agent-based/system dynamics, hybrids, simulation methods, practical applications, and defence are explicitly in scope | Hard cap **not verified**; technical notes and supplementary artifacts available. Code/tape package would materially help | Hybrid; optional OA price **not verified**; no APC on non-OA route | `5 d` first; `87 d` post-review; `21 d` online. Estimate `6–12 months`. [Official JOS page](https://www.tandfonline.com/journals/tjsm20/about-this-journal) |
| **Journal of the Operational Research Society (JORS)** | 2024 JIF `2.7`, JCR best quartile **Q2**; CiteScore `7.4`, best quartile **Q1**; SJR `0.917`; official acceptance `12%` | Applied OR and practical case studies are welcomed; simulation, inventory, production, reliability, and decision support are explicit domains | Hard cap **not verified**. Code/data deposit is encouraged and editors may request artifacts during review | Hybrid; optional OA price **not verified**; no APC on non-OA route | `15 d` first; `88 d` post-review; `15 d` online. Estimate `7–12 months`. [Official JORS page](https://www.tandfonline.com/journals/tjor20/about-this-journal) |
| **European Journal of Operational Research (EJOR)** | CiteScore `13.2`; JIF `6.0`; 2024 SJR `2.239`, **Q1**; publisher acceptance `13%` | Major new OR finding or genuinely novel application; current standard PPO benchmark is below the natural bar unless H2 becomes a general decision-contract principle | **30 manuscript pages**, all inclusive, single column, 11pt, 1.5 spacing; supplement separate; individual results/transparency expected and code invited | OA `USD 3,290`; subscription route no fee | Desk/reviewer invitation usually `<1 week`; reviewed first round about `3 months`; `326 d` displayed to acceptance / just under one year to publication. Estimate `10–16+ months`. [Official guide](https://www.sciencedirect.com/journal/european-journal-of-operational-research/publish/guide-for-authors), [journal page](https://www.sciencedirect.com/journal/european-journal-of-operational-research) |
| **International Journal of Production Economics (IJPE)** | CiteScore `20.2`; JIF `10.0`; 2024 SJR `2.833`, **Q1** | Engineering–management interface with a strong theoretical base and economic/financial consequences. A stress-cell computational result without real cost economics is insufficient | Hard cap **not verified**; supplement/data statement supported | OA `USD 4,290`; subscription route no fee | `2 d` first; `70 d` after review; `254 d` acceptance; `4 d` online. Estimate `9–15 months`. [Official IJPE page](https://www.sciencedirect.com/journal/international-journal-of-production-economics) |
| **Transportation Research Part E (TRE)** | CiteScore `15.0`; JIF `8.8`; 2024 SJR `2.513`, **Q1** | Logistics, inventory, warehousing, risk/disruption, SCM, and AI are in scope; the paper must contribute to logistics rather than use logistics as a setting | Hard cap **not verified**; double-anonymous; supplement and a data-availability statement supported | OA `USD 4,030`; subscription route no fee | `3 d` first; `57 d` after review; `205 d` acceptance; `13 d` online. Estimate `8–13 months`. [Official TRE page](https://www.sciencedirect.com/journal/transportation-research-part-e-logistics-and-transportation-review), [guide](https://www.sciencedirect.com/journal/transportation-research-part-e-logistics-and-transportation-review/publish/guide-for-authors) |
| **Decision Support Systems (DSS)** | CiteScore `13.5`; JIF `6.8`; 2024 SJR `2.366`, **Q1** | Requires a contribution to the theory, design, implementation, impact, or evaluation of decision support—not merely an OR/ML model | `34` double-spaced pages reported in current guide; supplement/data statement supported; decision-support artifact/evaluation would be expected | OA `USD 3,900`; subscription route no fee | `3 d` first; `302 d` to acceptance; post-review first decision **not verified**. Estimate `10–16+ months`. [Official DSS page](https://www.sciencedirect.com/journal/decision-support-systems), [guide](https://www.sciencedirect.com/journal/decision-support-systems/publish/guide-for-authors) |
| **Engineering Applications of Artificial Intelligence (EAAI)** | Current metric/quartile not needed for the decision; **not verified in this audit** | Requires a novel AI contribution in a real engineering application and explicitly asks for validation on public datasets for replicability | **50-page maximum**; single column; public-data/repository statement required (Option C) | APC/turnaround not verified here; subscription option available | Low fit: standard PPO/MLP and a controlled military benchmark do not satisfy the natural AI-novelty/public-data story. [Official guide](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors) |
| **Expert Systems with Applications (ESWA)** | Metrics not material to this decision | The journal explicitly states that it **no longer considers military/defence applications** | Not assessed further | Not assessed | **Do not submit.** [Official journal page](https://www.sciencedirect.com/journal/expert-systems-with-applications) |

### 6.3 Audience, recent comparable work, one-topology tolerance, and desk risks

| Journal | Audience and recent comparable evidence | Tolerance for a deep one-topology benchmark | Algorithmic novelty expected | Most likely desk-rejection reason | Qualitative 16-month prospect after required fixes |
|---|---|---|---|---|---|
| **IJPR** | Production/logistics/SCM scholars and managers. Direct prior art combines supply-chain recovery and RL ([DOI 10.1080/00207543.2024.2383293](https://doi.org/10.1080/00207543.2024.2383293)); therefore the contribution must be decision-contract/actionability, not DES–RL | **Moderate** if the case yields a transferable production decision principle, strong comparators, and managerial implications | Method novelty not mandatory, but a novel decision aid and state-of-art comparison are | Standard PPO, one military case, causal ablation defect, weak validation language, 51-page diffusion | **Medium** if H1/H2 and audit gates pass; low for the current PDF |
| **C&IE** | Industrial engineering/computerized methods. A recent DRL digital-twin resilience paper used realistic semiconductor data ([DOI 10.1016/j.cie.2025.111389](https://doi.org/10.1016/j.cie.2025.111389)) | **Moderate-high** if reproducibility, benchmark depth, and generalizable design logic are strong | A new RL algorithm is not essential; originality and significance are | Single-case transferability, incomplete resource accounting, or benchmark seen as incremental | **Medium-high** after P1 closure; **medium** for conditional P2 after all gates |
| **SMPT** | Simulation methodologists and hybrid-model practitioners. DES–DRL integration is already published in dynamic scheduling ([DOI 10.1016/j.simpat.2024.102948](https://doi.org/10.1016/j.simpat.2024.102948)) | **High** if V&V, event semantics, CRN design, and reproducibility are central | Simulation-method/experimental novelty can substitute for algorithm novelty | Paper reads as SCM theory with routine simulation rather than a simulation contribution | **Medium-high** |
| **JOS** | Simulation researchers/practitioners; defence applications are explicitly accepted. Its current DES–ML supply-chain review maps the mature field ([DOI 10.1080/17477778.2025.2500393](https://doi.org/10.1080/17477778.2025.2500393)) | **High** for a deeply audited practical model | Low; methodological/practical insight suffices | Insufficient novelty after removing broad AI claims, or excessive paper length | **Medium-high** as specialist final destination despite the low official acceptance rate |
| **JORS** | Applied OR and case-study audience; direct recent comparator specific to this mechanism **not verified** | **Moderate-high** if framed as an applied OR decision study with reproducible evidence | Low-to-moderate; practical OR contribution is acceptable | Standard method and insufficient transferable OR insight | **Medium** |
| **EJOR** | Top OR methods/applications audience. Recent RL work in multiobjective multi-echelon control demonstrates a high technical bar ([DOI 10.1016/j.ejor.2026.02.002](https://doi.org/10.1016/j.ejor.2026.02.002)) | **Low-moderate** unless the case establishes a major general OR principle | High application novelty or method novelty | No major OR finding; PPO benchmark and one topology; page cap | **Low**; do not lead here unless the corrected factorial materially elevates theory |
| **IJPE** | Production economics/operations management audience. Recent RL work emphasizes theoretically meaningful information-sharing decisions ([DOI 10.1016/j.ijpe.2026.110079](https://doi.org/10.1016/j.ijpe.2026.110079)) | **Low** without economic generalization | Algorithm novelty secondary to theory/economic consequence | Missing cost/economic theory and one synthetic stress cell | **Low** now; possible aspirational P2 only after a strong economics result |
| **TRE** | Logistics and transportation scholars. A directly comparable benchmark-depth DES–RL paper was **not verified** in this audit | **Moderate** only if logistics mechanisms—not the military case—drive the contribution | Method novelty not always necessary; logistics contribution is | Weak transportation/logistics theory, production emphasis, or military specificity | **Low-medium**, not a lead target |
| **DSS** | Information-systems and decision-support scholars; comparable SCRES decision-support artifact **not verified** | **Low** without user/organizational DSS design or evaluation | DSS theory/artifact contribution matters more than RL novelty | Direct OR/ML contribution submitted to the wrong outlet | **Low** |
| **EAAI** | AI engineering audience; public/reproducible engineering validation expected | **Low** for a non-public military benchmark | **High** relative to the current paper | No novel AI method and public-data constraint | **Low** |
| **ESWA** | Not applicable | None for this case | Not applicable | Military/defence exclusion | **Zero by scope** |

### 6.4 Submission ladder

| Paper | Journal | Tier | Topical fit | Method fit | Strongest selling point | Primary desk risk | Extra work required | Timeline/confidence | Cascade after rejection |
|---|---|---|---|---|---|---|---|---|---|
| P1 | **EJOR** | Aspirational only | Medium | Low-medium | A general decision-contract principle if H2 is unusually strong | Standard PPO and one topology do not meet major-OR-finding bar | All P1 blockers plus a genuinely general formal/actionability result and 30-page compression | Low; likely `10–16+ months` | Skip unless H2 changes the theory; otherwise start at IJPR |
| P1 | **IJPR** | **Primary realistic** | **High** | High | Decision rights at a production bottleneck, exhaustive comparison, and managerial boundary | Direct IJPR RL-recovery precedent; current causal/inference defects and length | H1/H2 gates, provenance repairs, validation rewrite, compact exhibits, managerial synthesis | **Medium** after fixes; post-review decision around 2 months officially | Desk reject: C&IE within `10 days`; reviewed reject: within `21 days` |
| P1 | **C&IE** | Backup 1 | High | **High** | Deep computerized IE benchmark and falsification design | Transferability and incomplete resource pricing | Same frozen evidence; reframe toward IE methodology and reproducibility | **Medium-high**, `7–12 months` | SMPT in `10–14 days` |
| P1 | **SMPT** | Backup 2 | Medium | **Very high** | Audited DES–RL environment, decision-contract experiment, and CRN inference | SCM theory overwhelms simulation contribution | Reframe title/introduction around simulation design, V&V, and experimental method | **Medium-high**, `5–9 months` | JORS, then JOS |
| P1 | **JORS/JOS** | Specialist fallback | Medium / medium | High / very high | Practical OR case or simulation-audit lesson | Incremental method or weak generalization | Concise specialist rewrite | **Medium** | Choose JORS for OR story, JOS for DES story |
| P2 | **IJPE** | Aspirational only | Medium-high | Medium | General slow–fast reserve economics, if real costs and regime interaction survive | No economic theory; engineered cell | All P2 gates plus economically interpretable cost/resource model | **Low** now | Do not attempt unless the gates change the paper materially |
| P2 | **C&IE** | **Primary realistic, conditional** | High | **High** | Classical reserve design + corrected physical DES + adaptive recovery | Classical optimizer may erase RL-posture novelty; legacy physics | Every P2 mandatory gate | **Medium** only after gate closure | SMPT |
| P2 | **SMPT** | Backup 1 | Medium | **Very high** | Slow–fast simulation architecture, physical sensitivities, and informative null | Single extreme cell overgeneralized as prevention | Same gates; emphasize simulation method and boundary map | **Medium-high** if clean | JOS |
| P2 | **JOS** | Backup 2 | Medium | Very high | Defence-compatible DES application and negative dynamic-control result | Lower SCM reach; need a concise simulation lesson | Compact specialist version | **Medium** | Final credible destination or stop |

### 6.5 Unambiguous journal recommendation

- **Paper 1:** submit to **IJPR** after the targeted closure package. Prepare C&IE and SMPT formats before submission so a desk rejection costs days, not months.
- **Paper 2:** do not choose a journal yet. If all gates pass, submit first to **C&IE**, then SMPT, then JOS.
- Do not lead with EJOR, IJPE, TRE, DSS, EAAI, or ESWA under the current contribution.

## 7. Minimum publishable package

### 7.1 Paper 1 mandatory blockers

| Priority | Required artifact and contrast | Minimum scale | Primary endpoint | Promotion rule | Stop/downgrade rule |
|---|---|---|---|---|---|
| **1** | **Fully crossed held-out evaluation:** all 10 frozen PPO checkpoints and the ex ante selected static comparator on the same unseen tapes; preserve raw order ledgers and checkpoint/scenario identifiers | `10 checkpoints × ≥24` common held-out tapes; use `60` if inexpensive or if intervals are unstable | Exact Garrido Excel ReT; secondary operational panel | Checkpoint/tape-aware two-way or random-effects CI for PPO–static wholly positive; no single checkpoint/tape cluster dominates | If CI includes zero or sign is unstable, do not retain the Q1 headline; diagnose before any new training |
| **2** | **Corrected decision-contract factorial:** fixed; upstream+shift without dispatch (`dims 0–5`); dispatch-only (`dims 6–7`, freeze `0–5`); joint 8D. Same observation, budget, tapes, and one common comparator | Minimum `5 training seeds × 60k`; use 10 if compute is modest; evaluate on the common held-out battery | Excel ReT. Primary nested contrast: joint minus otherwise-matched no-dispatch arm | Dispatch-access contrast positive with dependence-aware CI above zero and consistent seeds; absolute arm means also reported | If null/negative, remove causal action-alignment claim and use fallback title/venue framing |
| **3** | **Evidence-provenance repair:** replace mixed-CRN h104 with one coherent bundle; correct Track A arm definitions/seed counts/duplicate row; label every 5- versus 10-seed result; remove invented `±15%` validation | No new training; regenerate directly from source CSV/JSON | Exact table–figure–prose agreement | Zero unresolved numerical or protocol contradictions | Do not submit while any conflict remains |
| **4** | **Excel ReT audit:** executable formula lineage, branch unit tests, min/max/out-of-range counts, order IDs, exact and clipped sensitivity on the final ledgers | Entire crossed evaluation ledger | Exact Excel ReT plus count/magnitude outside `[0,1]` | Headline sign and material effect survive; pathology is transparent | If singularities/outliers materially create the gain, bound or replace the headline endpoint before submission |
| **5** | **Submission artifact:** shortened manuscript, frozen supplement, one-command replication bundle, data/code/permissions statement, author/affiliation/CRediT/funding fields, and visual QA | Main text about `9–10k` words and `≤8` core exhibits | Editorial coherence and reproducibility | Every exhibit supports one claim; zero placeholders, clipped floats, broken equations, or inaccessible evidence | Do not submit the current 51-page PDF |

Why blocker 1 matters: the current 120 PPO rows contain only **21 unique evaluation tapes**, because evaluation seeds overlap across checkpoint/episode combinations. Clustering only by training seed treats those repeated tapes inadequately and makes the reported interval too optimistic. This does not erase the large descriptive effect—all ten checkpoint summaries are positive—but it prevents calling the current CI definitive.

Why blocker 2 matters: the present source code defines `shift_only` by freezing only dimensions 6–7, so the policy still controls all upstream dimensions and shifts; `downstream_only` freezes only dimension 5, so it still controls upstream inventory and downstream dispatch. The names and manuscript interpretation do not match the actual interventions. This is an identification defect, not a cosmetic label.

### 7.2 Paper 1 strongly recommended

| Work | Required artifact | Decision rule |
|---|---|---|
| **Resource/economic sensitivity** | Prespecified dispatch/resource-price or matched-resource grid with Pareto effects; do not tune the price on test outcomes | Keep only bounded cost/resource language. ReT superiority does not imply cheaper operation |
| **No-forecast headline** | Make no-forecast the principal deployable information contract; treat full forecast/regime inputs as diagnostics | Strengthens recovery/actionability without creating a prediction claim |
| **Generalization cleanup** | One same-tape current/increased h104 bundle and the severe service-floor boundary | No universal “robust to disruption severity” language |
| **External reproducibility** | One command that regenerates the headline table, figure, and claim-registry row from a frozen release | Complete before preprint/submission or a reviewer response |
| **Independent pre-submission read** | One simulation/OR reader not involved in the build checks claim, arm, seed, and figure labels | Any unresolved mismatch returns to the evidence freeze; it does not trigger new algorithm experiments |

### 7.3 Paper 2 mandatory gate

| Priority | Required artifact and contrast | Minimum scale | Primary endpoint | Promotion rule | Stop/reframe rule |
|---|---|---|---|---|---|
| **1** | **Classical per-operation fixed-reserve frontier:** calibration-only search over Op3/Op5/Op9; lock and test no-buffer, thesis levels, homogeneous, heterogeneous classical best, distilled posture, and dynamic 11D | At least `5` calibration seeds plus `10` held-out seeds or a common `≥24`-tape battery; expand only if intervals are unstable | Excel ReT plus actual resource exposure/cost | Slow–fast fixed design remains non-dominated; either distilled posture adds value beyond classical search or a distinct classical reserve-design mechanism emerges | If classical search matches/dominates and no general mechanism remains, drop RL-optimizer wording and consider stopping P2 |
| **2** | **Physical replenishment correction:** route-aware deliveries, captured-at-order request/target, and unit tests showing zero injection through disabled routes | Re-run no-buffer 8D, fixed 8D, dynamic 11D with `≥5` seeds on the common battery | Excel ReT, service-loss AUC, actual inventories, route violations | Fixed–no-buffer gain remains positive with CI above zero and zero physical violations | If gain disappears, do not publish the legacy reserve result |
| **3** | **Actual-inventory holding exposure/cost:** time-weighted stored inventory, capacity, replenishment, shortage, lost-order, and switching ledgers—not target-fraction proxies | Same common battery and prespecified plausible cost/resource grid | Resource-adjusted ReT/Pareto position and total cost | Two-stage design remains non-dominated over a defensible range | If dominated once actual holdings are priced, restrict to a headroom diagnostic or stop |
| **4** | **Compact regime/horizon matrix:** natural, moderate, engineered R21, one locked adjacent cell, and at least two defensible horizons | Minimum `5` seeds per cell on common tapes | Posture-minus-no-buffer Excel ReT and service loss | Extreme cell replicates and null-to-positive transition is interpretable | If only one exact cell survives, report a narrow boundary at most |
| **5** | **Formal equivalence/non-inferiority test, only if that word is worth retaining** | Prespecified practical margin before held-out analysis; preferably `10` held-out seeds | Dynamic 11D minus fixed posture | TOST/CI lies wholly inside the margin | Otherwise say “no detected difference at current precision” |

### 7.4 Optional/future work

- A second civilian topology or external case **after Paper 1 submission**.
- A recurrent controller only after an observability audit demonstrates information in history that the current state omits.
- Composite shocks after the thesis current/increased benchmark is frozen.
- Additional Paper 2 horizons only if the compact matrix leaves the editorial interpretation genuinely ambiguous.
- A genuine persistent-versus-reset research program as a separate future theory paper, with matched histories, held-out future streams, and online update semantics.

### 7.5 Experiments that should not be run now

- KAN 8D.
- A definitive SAC/TD3/PPO algorithm tournament.
- Stronger retained-versus-reset H4 merely to preserve the Garrido draft.
- A separate predictor merely to preserve “predictive accuracy.”
- More Ruta B, splice, anticipation, or event-timing searches.
- Larger networks, transformers, recurrent policies, attention, or DKA without a demonstrated observability or reviewer need.
- Broad reward shaping and hyperparameter sweeps without a named threat to validity.
- Hundreds of dynamic-buffer actions before route and cost physics are fixed.
- A second topology before the closest paper is submitted.

### 7.6 Explicit answers to the requested experiment questions

| Proposed work | Paper 1 | Conditional Paper 2 | Decision |
|---|---|---|---|
| Classical per-operation reserve frontier | Not relevant | **Mandatory** | Run only after P1 submission; it determines whether P2 exists and whether RL posture language survives |
| Route-aware replenishment | Not required for the no-buffer P1 spine; disclose where applicable | **Mandatory** | Physical-validity gate |
| Holding cost based on actual inventory | Bounded resource sensitivity is strongly recommended | **Mandatory** | Target-fraction prices are not enough for P2 |
| Horizon/regime generalization | Repair current/increased same-tape evidence; no broad new grid | **Compact mandatory matrix** | Editorially valuable only at the specific scale above |
| KAN 8D | **No** | **No before all gates; probably no** | Does not change the journal decision |
| Confirmatory SAC/TD3 | **No** | **No** | Screen-scale evidence is enough for a sidecar; algorithm invariance is not the claim |
| Stronger H4 retained/reset | **No** | **No** | Delete the theory claim rather than fund it accidentally |
| Separate predictor | **No** | **No** | Delete predictive-accuracy language |
| Formal equivalence test | Not needed | Conditional | Run only if “equivalent” materially improves P2; otherwise use neutral wording |

## 8. Authorship and Garrido collaboration plan

### 8.1 Recommended authorship

**Default for Paper 1 and, if it survives, Paper 2:**

1. **Thomas — first author and corresponding author**
2. **Garrido — second author**

Thomas has led the research execution: DES reconstruction, thesis/workbook audit, software, Gymnasium/action contracts, experimental and CRN design, training, static frontiers, statistical analysis, visualization, artifact documentation, adversarial controls, retractions, and technical writing. First authorship is therefore well supported.

Thomas should also be corresponding author if he accepts responsibility for submission, artifact/data questions, reviewer responses, deadlines, revisions, and long-term preservation. Corresponding authorship is an accountability and communication role, not a seniority prize. Garrido may be corresponding author if an institution requires an established contact or if he explicitly assumes and can reliably discharge the full editorial responsibility; that should not change Thomas’s first authorship.

Garrido should be a substantive coauthor because the project uses his thesis case, original workbooks, SCRES research agenda, domain knowledge, and conceptual proposal. His pre-submission contribution must be made explicit through domain validation, theory/managerial revision, permissions work, full-manuscript review, and final accountability.

**Should Thomas publish something alone?** Not this principal paper, and not now. Solo authorship would understate Garrido’s intellectual/resource contribution and create avoidable questions about permission and case ownership. It becomes reasonable only if Garrido formally declines authorship in writing, all reuse/release rights are independently resolved, and the submitted framing is demonstrably Thomas’s. A solo software/method note is also not recommended during the 16-month window because it would distract from Paper 1 and create salami-slicing risk.

Do not use equal-contribution language unless later work becomes genuinely comparable. Guidance, seniority, or supplying a draft does not determine first-author order; neither does Thomas’s computational dominance erase Garrido’s case and conceptual contribution.

### 8.2 Preliminary CRediT matrix

CRediT records actual contribution; it does not itself decide authorship eligibility.

| CRediT role | Thomas | Garrido | Required before submission |
|---|---|---|---|
| Conceptualization | **Lead:** experimental contribution, action-contract framing, falsification strategy | **Substantial:** original SCRES problem, thesis case, conceptual agenda | Joint approval of narrowed claim and theoretical boundary |
| Methodology | **Lead:** DES reconstruction, RL environment, CRN, baselines, statistics | Supporting/domain methodology | Garrido validates operational meaning, feasibility, risks, and cadence |
| Software | **Lead** | None currently | No additional Garrido software contribution required |
| Validation | **Lead:** computation, workbook, source/artifact checks | **Required co-lead:** domain and case interpretation | Written memo separating verification, calibration, and empirical validation |
| Formal analysis | **Lead** | None currently | Garrido reviews interpretation; he need not duplicate computation |
| Investigation | **Lead** | Supporting through sources/domain questions | Resolve open cadence, demand, risk, and ReT provenance questions |
| Resources | Supporting: organization/preservation | **Lead:** thesis, workbooks, case foundation, domain access | Confirm ownership, permissions, and release limits |
| Data curation | **Lead** | Supporting | Approve release, redaction, aggregation, or justified restriction |
| Writing — original draft | **Lead** | Existing conceptual draft; further contribution required | Garrido substantively rewrites/revises case, theory, and managerial discussion |
| Writing — review and editing | **Lead** | **Required substantial review** | Full manuscript and response-letter review, not a title/abstract approval |
| Visualization | **Lead** | Review | Confirm topology, risk, and domain labels |
| Supervision | Not applicable | Lead/advisory | Continue guidance without displacing first authorship |
| Project administration | **Lead** | Supporting | Agree deadlines, freezes, and sign-off process |
| Funding acquisition | Not verified | Not verified | Assign only if documented |

For conditional Paper 2, Garrido’s validation role must expand to reserve feasibility, replenishment routes/lead times, actual-inventory cost interpretation, and the plausibility and labeling of extreme R21.

### 8.3 Decisions Thomas can close independently

Thomas can:

- remove unsupported, retracted, or misleading claims;
- run the crossed evaluation and corrected factorial under a frozen analysis plan;
- define statistical promotion/stop rules;
- repair result provenance and freeze evidence bundles;
- rewrite, shorten, and visually redesign the manuscript and supplement;
- select tables, figures, appendices, and repository structure;
- prepare journal formatting, cover letter, preprint package, and reviewer-response templates;
- stop KAN 8D, predictor, retained/reset expansion, algorithm tournaments, and prevention searches;
- recommend Strategy C and prepare Paper 1 for sign-off.

Thomas should not unilaterally:

- describe the case as empirically validated or make undocumented claims about military operations;
- release original workbooks, restricted topology information, or third-party material;
- reproduce thesis figures/tables without rights confirmation;
- post a coauthored preprint, submit, or resubmit without Garrido’s explicit approval.

### 8.4 Exact inputs and approvals required from Garrido

Before Paper 1 submission, obtain:

1. **Case-validation memo:** topology, operation meanings, disruption semantics, feasible decisions, implementation delays, and managerial meaning of Op10/Op12 dispatch.
2. **ReT provenance confirmation:** whether the workbook formula is definitive, how values above one were historically interpreted, and whether undocumented corrections existed.
3. **Validation clarification:** the proper interpretation of the eight throughput comparisons and expert inspection, with no invented `±15%` threshold.
4. **Risk-generator review:** which mechanisms are thesis-faithful, which are deliberate project extensions, and which frequencies/intensities are synthetic stress tests.
5. **Resource interpretation:** defensible meaning of inventory, shifts, switching, and dispatch exposure; exact monetary calibration is not required for Paper 1.
6. **Permissions statement:** ownership and permissible use/release of thesis text, workbooks, topology diagrams, military descriptions, source data, and third-party materials.
7. **Theory contribution:** substantive revision of SCRES and managerial framing that accepts the distinction among adaptive recovery, prediction, prevention, and cross-campaign learning.
8. **Authorship details:** affiliation, ORCID, conflicts, funding, CRediT roles, corresponding-author agreement.
9. **Accountability:** review of full manuscript and responses; written approval of every submitted version.

Additional Paper 2 inputs: reserve implementation feasibility, route/lead-time semantics, replenishment order commitments, actual-stock cost meaning, and the operational plausibility of the extreme R21 stress design.

### 8.5 Proposed 90-minute meeting agenda

1. `10 min` — Supported, bounded, unsupported, and retracted evidence.
2. `10 min` — Decide Strategy C: Paper 1 first; Paper 2 conditional.
3. `15 min` — Topology, validation language, workbook provenance, ReT above one.
4. `15 min` — Physical meaning/timing of inventory, shift, dispatch, and replenishment actions.
5. `10 min` — Thesis-faithful risks versus engineered stress tests.
6. `10 min` — Paper 1 boundary: adaptive recovery; no prediction, prevention, anticipation, or organizational learning.
7. `10 min` — Author order, corresponding author, CRediT, review deadlines.
8. `5 min` — Permissions, military sensitivity, affiliations, funding, conflicts, AI disclosure.
9. `5 min` — Submission target, experiment stop date, and signed deliverables.

The meeting should end with written decisions, not general approval.

### 8.6 Short Spanish email

**Asunto: Estrategia de publicación y contribuciones necesarias para cerrar el artículo SCRES–DES–RL**

> Profesor Garrido:
>
> Después de auditar la tesis, los Excel originales, el artículo de 2024, el manuscrito y los resultados computacionales, mi recomendación es presentar primero un artículo centrado en control adaptativo y alineación del espacio de decisiones. El resultado principal es defendible, pero no debemos presentarlo como evidencia de predicción, prevención o aprendizaje organizacional entre campañas. El estudio de reservas estratégicas quedaría como un segundo artículo condicionado a cerrar primero la frontera clásica y las sensibilidades físicas.
>
> Propongo que yo sea primer autor y autor de correspondencia, dada mi responsabilidad sobre el desarrollo, los experimentos, el análisis y la escritura, y que usted sea coautor por la base conceptual, la tesis, los datos y la validación del dominio.
>
> Para cerrar el artículo necesito su participación directa en: validar la interpretación del caso y de los riesgos; aclarar el alcance de la validación original y de ReT; revisar la viabilidad de las decisiones; confirmar permisos sobre tesis y archivos; fortalecer la discusión teórica y gerencial; y aprobar el manuscrito final, la autoría y la declaración CRediT.
>
> ¿Podemos reunirnos 90 minutos para tomar estas decisiones y fijar responsabilidades y fechas? Le enviaré antes una síntesis de dos páginas y la matriz de claims.
>
> Un abrazo,  
> Thomas

### 8.7 Versioning and approval process

1. Preserve the thesis and original workbooks read-only; record hashes and provenance.
2. Create an immutable Paper 1 evidence release containing code commit, environments, seeds, tapes, raw ledgers, analyses, tables, and figures.
3. Maintain one claim registry linking every numerical/theoretical statement to an artifact and status.
4. Generate numbers and exhibits from code; never edit result values manually in LaTeX.
5. Use two freezes:
   - **Evidence freeze:** Thomas confirms reproducibility; Garrido confirms domain interpretation.
   - **Submission freeze:** both approve exact PDF, supplement, code/data statement, CRediT, cover letter, and target journal.
6. Obtain written approval from both authors before preprint, initial submission, and each revision. Silence is not approval.
7. Link every reviewer-driven numerical change to the replacement artifact in a response matrix.
8. Allow five business days for coauthor review at each freeze; use an explicit deadline and approval checkbox.

### 8.8 Permissions, acknowledgements, and institutional issues

Confirm with the authors’ institutions and the selected journal; this is not legal advice:

- copyright/reuse rights for thesis text, workbooks, diagrams, tables, software, and third-party material;
- whether military topology, risk assumptions, or operational details are sensitive/restricted;
- whether data must be anonymized, aggregated, controlled-access, or withheld with a justified availability statement;
- ethics, security, export-control, defence-research, or institutional review requirements;
- university/funder ownership of original workbooks and derivative software;
- an open-source code license separate from a data/workbook license;
- permission to reproduce figures versus redrawing and citing them;
- exact affiliations, funding, conflicts, acknowledgements, and contribution statements;
- current journal policy on generative-AI assistance and the required disclosure;
- preprint policy and approval by all authors/responsible institutional authorities.

If original workbooks cannot be public, release a legally permitted synthetic/derived test fixture, the exact formula implementation, hashes/provenance, and a clear controlled-access or non-sharing reason. Reproducibility does not authorize disclosure.

## 9. Sixteen-month roadmap

The target is not merely “submitted.” By October 2027, Paper 1 should be accepted, in advanced revision, or at least in a mature reviewed cycle at a credible journal. Paper 2 must never compromise that outcome.

| Month | Aggressive route | Conservative route |
|---|---|---|
| **M1 — Jul 2026** | Freeze claims; run corrected factorial and start 10-checkpoint crossed evaluation; reconcile h104, Track A, 5/10-seed labels, and ReT audit; hold Garrido decision meeting | Freeze protocol and claims; finish source/artifact audit; prepare scripts and obtain Garrido’s case/permissions inputs before costly runs |
| **M2 — Aug 2026** | Complete crossed inference; lock Paper 1 experiments by **31 Aug**; rewrite to `9–10k` words; reduce exhibits; build supplement and replication bundle | Run corrected factorial/crossed evaluation; investigate a discrepancy once; lock analysis plan; no architecture experiments |
| **M3 — Sep 2026** | Numerical/visual/adversarial audit; Garrido theory/domain revision and sign-off; submit to **IJPR by 30 Sep**; post preprint only after permissions and journal-policy check | Complete only protocol-defect reruns; reconcile provenance; absolute experiment cutoff **15 Oct** except confirmed implementation error |
| **M4 — Oct 2026** | If desk-rejected, reframe and submit to **C&IE within 10 days**; if reviewed, open only Paper 2’s classical/physics gate | Rewrite/shorten P1; rebuild figures/supplement; obtain formal validation and permissions decisions |
| **M5 — Nov 2026** | P2 only: route-aware replenishment, captured order targets, actual inventory exposure, held-out classical frontier; **go/no-go by 30 Nov** | Submit P1 to IJPR by **30 Nov**; desk cascade within 10 days; start P2 only after submission |
| **M6 — Dec 2026** | Complete P1 revision within `21–30 days` if reviews arrive; if P2 passes, run compact regime/horizon matrix, otherwise close it | Stabilize replication/response templates; P2 receives only physical corrections and classical frontier |
| **M7 — Jan 2027** | Resubmit P1 revision; draft P2 slow–fast paper only after gate; no KAN/predictor/horse race | Respond to P1 reviews within 30 days; reviewed rejection cascades to C&IE/SMPT within **21 days** |
| **M8 — Feb 2027** | Independent/Garrido audit of P2; submit to **C&IE** only if every gate survives | P2 **go/no-go by 28 Feb**; stop if classical design dominates without a distinct mechanism or physics removes gain |
| **M9 — Mar 2027** | Handle P1 second round: minor in `≤14 days`, major in `≤30`; P2 begins review/cascade | Complete P1 revision/resubmission; if P2 passed, run only compact confirmatory matrix |
| **M10 — Apr 2027** | If P1 reviewed-reject, submit to **SMPT within 21 days** with a simulation-method fit rewrite; P2 desk reject goes to SMPT within 10 days | Draft P2 only after full artifact/visual audit; keep P1 primary |
| **M11 — May 2027** | Address P2 reviews; freeze exploratory branches; prepare archival P1 release if accepted | P1 second-round response or rapid SMPT cascade; Garrido reviews P2 theory/case |
| **M12 — Jun 2027** | Target P1 acceptance/advanced revision; return P2 revision within 30 days | Submit P2 to C&IE only if all gates passed; otherwise formally close and preserve null/boundary evidence |
| **M13 — Jul 2027** | P1 proofs, repository DOI, data/code statement, and disclosures | P1 accepted/advanced revision; P2 desk cascade to SMPT within 10 days if needed |
| **M14 — Aug 2027** | Final P2 revision; reviewed reject cascades to **JOS within 21 days** | P1 revision/proofs; P2 under review with frozen scope |
| **M15 — Sep 2027** | Consolidate replication docs and present seminars/conference without expanding claims | Respond to P2 reviews within 30 days; reviewed rejection leads to JOS or closure based on scientific substance |
| **M16 — Oct 2027** | Required endpoint: P1 accepted/online/advanced revision; P2 accepted/in review only if gated | Required endpoint: P1 accepted/advanced revision; P2 status cannot compromise P1 |

### Hard stop, preprint, and cascade rules

- **Paper 1 secondary-experiment stop:** `31 Aug 2026` aggressive; `15 Oct 2026` conservative.
- After the stop, run only work required to fix a demonstrated implementation defect or answer a decisive reviewer objection.
- **Paper 2 gate:** `30 Nov 2026` aggressive; `28 Feb 2027` conservative.
- Post a preprint only after the evidence freeze, permissions/security review, Garrido approval, and target-journal preprint-policy confirmation. A preprint is recommended once those conditions hold because it establishes a dated, citable artifact while review proceeds.
- Desk rejection: resubmit within **10 calendar days**.
- Rejection after review: resubmit within **21 calendar days**.
- P1 cascade: **IJPR → C&IE → SMPT → JORS or JOS**.
- P2 conditional cascade: **C&IE → SMPT → JOS**.
- Prepare title page, anonymized version, highlights, cover letters, and formatting variants in advance.
- Do not answer rejection by adding KAN, a predictor, more algorithms, or a new topology unless reviewers identify that exact point as decisive and an editor offers a credible revision/resubmission route.

## 10. Reviewer simulation

### 10.1 Plausible IJPR desk-rejection note

> Thank you for submitting your manuscript on reinforcement-learning control in a thesis-derived supply-chain simulation. The study contains extensive computational work, but I am unable to send it for review in its present form. The use of a standard PPO controller within a discrete-event supply-chain model does not itself establish sufficient methodological novelty, particularly given recent simulation–RL studies in supply-chain recovery. The manuscript’s proposed novelty—action-space alignment—is not convincingly isolated because the reported ablation arms do not correspond to the causal contrasts described in the text.
>
> The paper also relies on one historical military topology whose empirical validation and broader production-management relevance are not adequately established. Several theoretical claims concerning learning, anticipation, and path dependence appear stronger than the experiments support. In addition, the manuscript is substantially longer than necessary, contains multiple screen-scale side studies, and does not provide a sufficiently concise managerial contribution for IJPR’s audience.
>
> A narrower paper with a corrected experimental design, clearly bounded benchmark contribution, accurate validation language, and explicit connection between decision authority and operational bottlenecks may be suitable as a substantially revised submission.

### 10.2 Hostile but competent Reviewer #2

**Recommendation: reject and resubmit only after major redesign.**

1. **Novelty is overstated.** DES–RL supply-chain control is not new, and PPO is standard. The manuscript alternates among benchmark, decision contract, empirical mechanism, learning theory, and software audit without selecting one contribution.
2. **The central mechanism is unidentified.** “Shift-only” still controls upstream dimensions, while “downstream-only” retains other actions. Each treatment is also compared with a different in-arm comparator. These experiments cannot show that downstream dispatch causes the gain.
3. **The inferential unit is pseudoreplicated.** Ten training seeds create 120 model–scenario rows but only 21 unique evaluation tapes. A training-seed-only CI ignores overlapping scenario dependence.
4. **Validation is overstated.** The thesis contains eight annual throughput comparisons and expert inspection, but no prespecified `±15%` acceptance criterion. “Empirically validated digital twin” is indefensible.
5. **The primary metric has unresolved mathematical behavior.** ReT is described as normalized, yet its workbook formula can exceed one—dramatically when recovery duration approaches zero. The paper must show that singularities, clipping, or branch frequencies do not create the headline.
6. **Evidence levels are conflated.** The ten-seed package contains PPO and one frozen static comparator, while the 147-cell frontier/Pareto/sidecar claims largely rely on five seeds.
7. **Resource comparison is incomplete.** The controller may consume unpriced dispatch, inventory, or capacity. Arbitrary post hoc prices do not establish equal resources or managerial cost advantage.
8. **Generalization is narrow.** Current/increased regimes within one military topology are not external validation; severe disruption imposes a service floor.
9. **Falsification language exceeds the controls.** No-forecast does not prove prediction has no value; a true-regime lookup failure does not prove privileged information is useless; a CI spanning zero is not equivalence.
10. **Learning theory is not demonstrated.** A small retained/reset effect does not establish organizational learning or cross-campaign path dependency. State dependence in DES is not learning.
11. **Result provenance is inconsistent.** The h104 table appears to mix random tapes while claiming CRN; Track A seed counts and repeated rows are inconsistent.
12. **Reproducibility and permissions are unresolved.** Military source materials and original workbooks are central, but release rights, ownership, and replication access are not defined.
13. **Managerial theory is underdeveloped.** Even if PPO wins, the manuscript must explain why an organization should redesign decision authority, which bottleneck is binding, and what resource commitments are implied.
14. **The manuscript is not submission-ready.** Its length, placeholder fields, crowded exhibits, and diffuse side studies obstruct the primary claim.

### 10.3 Strongest evidence-based response

The strongest response is to concede valid objections and present a narrower, corrected paper. The following response is usable **only after** the factorial and crossed evaluation actually exist:

> We thank the reviewer for identifying places where the manuscript conflated a benchmark result with broader claims about learning and resilience. We have revised the paper around one proposition: a learned state-feedback policy improves the Garrido Excel ReT benchmark when the decision contract reaches the downstream operational bottleneck. We no longer claim algorithmic novelty, predictive accuracy, prevention, anticipation, organizational learning, or universal RL superiority.
>
> We replaced the misdescribed ablation with a preregistered common-comparator factorial: upstream-and-shift control without dispatch, dispatch-only control, joint control, and a fixed baseline. We retain decision-contract mechanism language only because the downstream-access nested contrast is positive under the prespecified held-out test; had that gate failed, the manuscript protocol required a descriptive extended-contract interpretation.
>
> We also replaced the overlapping evaluation design with a fully crossed evaluation of all ten frozen checkpoints on the same held-out scenario battery and report checkpoint- and tape-aware inference. The original descriptive result—PPO ReT 0.005898 versus 0.005460, delta 0.000438, with all ten checkpoint summaries positive—now appears alongside the corrected interval and complete seed/tape map.
>
> We removed the undocumented `±15%` criterion and describe the environment as a thesis-grounded, workbook-audited computational case rather than an empirically validated digital twin. The validation section now separates software verification, exact workbook formula replay, historical throughput calibration, domain review, and external-validity limits.
>
> We reproduce the piecewise Excel ReT formula and acknowledge that it is not globally bounded despite the thesis description. We report branch frequencies, extrema, out-of-range order IDs, exact and clipped results, and the full operational panel. In the original canonical five-seed Track B ledger, no PPO or static order value lay outside `[0,1]`, and clipping did not change the means; the revised crossed ledger is audited identically.
>
> The revised manuscript clearly distinguishes ten-seed confirmatory evidence from five-seed frontier and sidecar analyses. It replaces the incompatible h104 analysis with one common-tape source and corrects Track A arm definitions, seed counts, and the duplicated entry.
>
> We no longer claim unconditional cost savings. We report resilience–resource trade-offs under explicit dispatch, shift, and inventory assumptions. Likewise, no-forecast and true-regime controls are described narrowly as tests of dependence on those tested inputs, not as evidence that forecasting or information is generally valueless.
>
> Finally, we reduce the main paper to the prespecified hypotheses and eight principal exhibits. Training diagnostics, exploratory algorithms, null/retracted prevention work, and detailed provenance move to a versioned supplement and replication archive containing exogenous tapes, seed maps, raw ledgers, analysis scripts, and a permissions-qualified data statement.

### 10.4 Objections that cannot yet be answered

Even after Paper 1’s minimum fixes, the authors cannot honestly claim to have resolved:

- external validity beyond one thesis-derived military food-supply topology;
- empirical effectiveness in a deployed supply chain;
- organizational learning or cross-campaign knowledge retention;
- causal prevention, advance anticipation, or disruption detection;
- real-world predictive accuracy;
- superiority of PPO, KAN, SAC, TD3, or RL as an algorithm class;
- fully calibrated monetary welfare or optimal real-world resource allocation;
- formal equivalence where no margin/test was prespecified;
- Paper 2’s reserve mechanism before classical, route-aware, captured-order, actual-inventory, and regime gates;
- whether the reserve advantage survives realistic route failure and cost accounting;
- unrestricted public release of all workbooks and case details before rights/security confirmation.

State these boundaries directly. Honest limitations are less damaging than claims a reviewer can falsify from the repository.

## 11. Final action list — ordered by impact divided by time

1. **Very high impact / low–medium time:** run the fully crossed 10-checkpoint common-tape evaluation. This is the cheapest decisive audit of the headline inference.
2. **Very high / medium:** run the corrected decision-contract factorial. This determines whether Paper 1 can claim a mechanism and target IJPR, or only a bounded benchmark result better suited to C&IE/SMPT.
3. **Very high / low:** repair h104 provenance, Track A definitions/seed labels, 5/10-seed distinctions, unsupported validation language, and false equivalence/causal wording.
4. **High / low:** execute the final-ledger ReT branch/outlier/clipping audit and publish it in the supplement.
5. **High / medium:** rewrite Paper 1 around adaptive recovery and decision authority; remove prediction, prevention, anticipation, organizational learning, KAN superiority, and algorithm invariance.
6. **High / low:** hold the Garrido meeting and obtain the domain memo, permissions decision, theory revision, CRediT details, and sign-off schedule.
7. **High / medium:** reduce the current 51-page evidence archive to a clean `9–10k`-word paper and visually re-audit every page/table/figure; complete authors, affiliations, funding, acknowledgements, repository pin, and data/code statements.
8. **High / low:** freeze a reproducible release and submit to IJPR; cascade within 10 days after a desk rejection and 21 days after reviewed rejection.
9. **High / medium, only after P1 submission:** run Paper 2’s held-out classical per-operation reserve frontier.
10. **High / medium-high, only if that gate survives:** correct replenishment physics and actual-inventory accounting, then rerun the three central policies.
11. **Medium / medium:** run the compact P2 regime/horizon matrix and make a write/stop decision.
12. **Low / high:** defer architecture comparisons, prediction, retained learning, and new topologies.

### Primary artifact register

- Thesis: `WRAP_Theses_Garrido_Rios_2017.pdf`, especially the ReT definition, R11–R3 tables, current/increased regimes, buffer/capacity interventions, limitations, and future agenda.
- Workbooks: `Raw_data1+Re.xlsx`, `Raw_data2+Re.xlsx`, and `Rsult_1.xlsx`; the raw CF1–CF20 formulas were replayed exactly, while `Rsult_1.xlsx` is a derived categorization/data-mining artifact and should not be treated as raw ground truth without a provenance manifest.
- Garrido et al. (2024): conceptual/exploratory bridge only.
- Garrido draft: retire as the submission vehicle; preserve its conceptual contribution and replace its unsupported theory/validation/architecture passages.
- Paper 1 core: `docs/track_b_q1_stats_2026-07-02_final_10seed/`, plus the source ledgers, h104 coherent bundle, corrected factorial, and crossed evaluation to be created.
- Paper 2 core: `outputs/experiments/track_bp_frozen_posture_audit_5seed_2026-07-09/`, `docs/TRACK_BP_GATE2_SCREEN_VERDICT_2026-07-09.md`, and the new classical/physical gate artifacts.
- Claim/audit history: `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`, `docs/REVIEWER_DEFENSE_MATRIX_2026-07.md`, `docs/PROMISING_LANES_REGISTRY.md`, `docs/PREVENTION_GATE_AUTOPSY_AND_CLOSURE_2026-07-07.md`, and later Track B-P verdicts.

### `SUBMIT NOW AFTER THESE FIXES`

- Fully crossed 10-checkpoint common-tape evaluation with dependence-aware inference.
- Corrected action-contract factorial—or explicit removal of the causal alignment claim.
- Coherent h104 artifact, accurate Track A definitions, and correct 5/10-seed labels.
- Final-ledger Excel ReT formula/outlier/clipping audit.
- Narrowed adaptive-recovery claim and `9–10k`-word manuscript.
- Eight core exhibits, clean supplement, one-command replication, and page-by-page visual QA.
- Completed author/affiliation/CRediT/funding/acknowledgement/data/code/permissions fields.
- Garrido domain-validation memo, substantive theory review, and written final approval.

### `RUN ONLY IF IT CHANGES THE JOURNAL DECISION`

- More held-out tapes if the crossed interval is unstable.
- Resource-price/matched-budget sensitivity needed to sustain an IJPR managerial claim.
- A second topology only if IJPR review makes transferability the sole decisive objection.
- Formal TOST only if Paper 2 needs the word “equivalent.”
- Recurrence only if an observability audit demonstrates a genuine memory problem.
- Adjacent R21 cells/horizons only after Paper 2’s classical and physical gates survive.
- Any reviewer-requested experiment only with a written objection and a prespecified promotion/stop rule.

### `DO NOT SPEND TIME ON THIS`

- KAN 8D.
- Confirmatory SAC/TD3 algorithm horse races.
- Stronger retained/reset H4 evidence for this paper.
- A separate predictor to rescue “predictive accuracy.”
- More Ruta B prevention or anticipation splices.
- Bigger networks, attention, transformers, DKA, or broad reward sweeps without a demonstrated need.
- The omnibus Paper A.
- Writing Paper 2 before the classical-frontier and physical-validity gates.
- A new comprehensive topology before Paper 1 is submitted.

**Decisive recommendation:** choose **Strategy C**. Close the two Paper 1 blockers, rewrite around decision-contract alignment and adaptive recovery, obtain Garrido’s active coauthor sign-off, and submit to **IJPR**. Paper 2 exists only if the classical reserve frontier and corrected physical/cost model leave a distinct slow–fast result. Do not submit one long paper, do not publish the principal paper alone by default, and do not spend the next months defending accumulated-learning, prediction, prevention, or architecture claims that the evidence does not support.
