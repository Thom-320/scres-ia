# External Agent Prompt: SCRES-DES-RL Scientific and Publication Strategy

Act as a combined senior committee consisting of:

1. an associate editor at a Q1 journal in operations research, production, or supply-chain management;
2. a methodological reviewer specializing in discrete-event simulation, reinforcement learning, and computational experimental design;
3. an academic publication and authorship strategist.

Conduct an independent assessment from first principles. Do not assume that our current framing, the proposed split into two papers, or our preferred journals are correct. Review the primary artifacts, investigate the current publication landscape, and recommend the strategy with the highest probability of producing a high-impact publication that is accepted, in advanced revision, or close to publication within the next 16 months.

Write the report in Spanish. Write proposed titles, research questions, hypotheses, contribution statements, abstracts, and manuscript-ready passages in polished academic English as well.

## Epistemic rules

- Do not accept a number merely because it appears in a narrative document. Verify it against CSV/JSON artifacts whenever possible.
- Distinguish rigorously among confirmatory evidence, preliminary screens, secondary findings, boundary/null results, unsupported claims, and retracted claims.
- The project's primary metric is Garrido Excel ReT (`order_ret_excel_mean` or `ret_excel`, depending on the runner), not `order_level_ret_mean`.
- Do not confuse architecture (MLP, RNN, KAN), learning paradigm (RL), and optimization procedure (backpropagation).
- Keep within-episode adaptive control, cross-campaign retention, prediction, prevention, and anticipation conceptually separate.
- Do not use “first” or “novel” without a literature search supporting it.
- Do not recommend experiments reflexively. Every proposed experiment must close a specific reviewer objection and have an explicit promotion or stop rule.
- Clearly label facts, source-supported conclusions, and your own inferences.

## Materials to inspect

### Garrido draft and theoretical foundation

- `/Users/thom/Downloads/v.0_neuralNet-scres.pdf`
- `/Users/thom/Downloads/v.0_neuralNet-scres.docx`
- `/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Supernote/Document/20_RESEARCH/PhD-Papers/garrido2024 scres+AI.pdf`
- The Garrido-Rios thesis and original workbooks referenced or stored in the repository.

Garrido's draft proposes `R_t=f(S_t,D_t,L_{t-1})`, accumulated learning, ANN/RNN/RL, hypotheses H1-H4, and predictive accuracy. Decide which elements remain defensible in light of the actual evidence, which must be reformulated, and which belong only in future work.

### Repository, manuscript, and evidence

Local repository:

- `/Users/thom/Projects/research/scres-ia`

GitHub branch:

- `codex/garrido-replication-experiments`

Inspect at minimum:

- `docs/manuscript_current/submission/elsevier/main.pdf`
- `docs/manuscript_current/submission/elsevier/sections/`
- `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`
- `docs/REVIEWER_DEFENSE_MATRIX_2026-07.md`
- `docs/PROMISING_LANES_REGISTRY.md`
- `docs/PAPER2_TRACK_BP_RESEARCH_PROGRAM_2026-07-09.md`
- `docs/TRACK_BP_GATE2_SCREEN_VERDICT_2026-07-09.md`
- `docs/TRACK_B_PREVENTION_HEADROOM_GENERALIZED_VERDICT_2026-07-08.md`
- `docs/PREVENTION_GATE_AUTOPSY_AND_CLOSURE_2026-07-07.md`
- `docs/TRACK_B_FINAL_AUDIT_PACKAGE_2026-07-06.md`
- the primary CSV/JSON artifacts linked from those documents;
- `git log`, to understand the actual research sequence, corrections, retractions, and contributions.

Visually inspect the compiled PDF, every table, and every figure. A successful LaTeX build is not evidence of submission quality.

## Current claims to audit, not assume

### Paper 1 evidence package

1. Track B improves Garrido Excel ReT against a dense downstream static frontier: PPO `0.005898` versus static `0.005460`; delta `+0.000438`; seed-clustered CI95 `[+0.000421,+0.000458]`; 10/10 seeds positive.
2. CVaR05, fill, backlog, service-loss AUC, and CTj/RPj/DPj tails support the same operational direction.
3. The defensible mechanism is action-space alignment with the downstream bottleneck: Track A is a boundary/null result; Track B wins when the contract reaches downstream dispatch.
4. The result does not depend on privileged forecasting: no-forecast is statistically equivalent to full-v7 over 15 seeds, and masking regime plus forecast retains most of the gain.
5. A non-learning lookup table with access to the true regime does not reproduce PPO.
6. Generalization is positive in current and increased regimes; severe disruption is the service-floor boundary.
7. PPO is Pareto non-dominated, but there is no unconditional cost victory. Positive dispatch charges can make PPO cheaper than the aggressive best-ReT static.
8. SAC and TD3 replicate the pattern only at screen scale. Do not present this as a definitive algorithm comparison.
9. Under `track_b_v1`, the evidence supports adaptive recovery, not causal prevention or anticipation.
10. The Ruta B preventive claim was retracted because the splice gate failed negative controls.
11. Retained-versus-reset learning has a small positive effect, but it does not currently support a central theory of path dependency or organizational learning.

### Paper 2 evidence package

1. Strategic-buffer headroom emerges under an engineered extreme R21 regime (frequency x8, impact x4), while natural and moderate intensities are effectively null.
2. Confirmatory 5-seed x 60k results:
   - 8D without buffers: `0.311676`;
   - 11D with dynamic buffer outputs: `0.340164`;
   - 8D with a frozen heterogeneous reserve posture: `0.340605`.
3. Frozen posture minus dynamic 11D: `+0.000440`, CI95 `[−0.000799,+0.001680]`; the two are statistically equivalent.
4. Frozen posture minus no-buffer 8D: `+0.028928`, CI95 `[+0.016283,+0.041574]`; 5/5 seeds and 117/120 episodes positive.
5. The calibrated posture is approximately `(Op3=0.1531, Op5=0.2480, Op9=0.2068)` as fractions of I_1344.
6. Within-checkpoint, cross-fitted, replay, and pre/post-event controls find no value from weekly scheduling, state-contingent buffer modulation, or event anticipation.
7. The current interpretation is a two-stage design: optimize a slow/fixed strategic reserve posture, then use PPO 8D for adaptive recovery.
8. The posture was distilled from 11D policies. A calibration-only classical per-operation reserve frontier is still required before claiming that RL is needed as the posture optimizer.
9. The current top-up implementation can inject stock after the lead time even when the supply route is down. Route-aware replenishment and holding cost based on actual stored inventory remain necessary physical sensitivities.
10. Real-KAN 11D approximately matches PPO 11D as a sidecar, but KAN 8D has not been run, so the architecture decomposition is not symmetric.

## Authorship and collaboration context

Thomas performed virtually all computational and empirical work in the repository: DES reconstruction, thesis/workbook audit, software, Gymnasium wrappers, action contracts, training, experiments, CRN design, static frontiers, statistical analysis, figures, documentation, adversarial controls, retractions, and technical writing.

Garrido contributed the conceptual draft, thesis/case foundation, domain knowledge, academic guidance, and discussion of the research agenda. Propose an honest CRediT allocation and authorship strategy. Do not assume that providing a draft or supervision automatically determines author order, but do not ignore Garrido's conceptual contribution, source materials, domain validation, and intellectual ownership.

Analyze separately:

- what Thomas can prepare and lead independently;
- what requires Garrido's validation, approval, information, or active contribution;
- what additional contributions Garrido should make before submission;
- reasonable author order and alternatives conditional on future contributions;
- corresponding-author responsibilities;
- preliminary CRediT roles for each contributor;
- permissions, acknowledgements, code/data availability, and institutional issues that must be resolved.

Do not provide definitive legal advice. Identify issues that require institutional or journal-policy confirmation.

## Mandatory journal research

Use current information, not remembered rankings. Prefer official journal and publisher sources and supplement them with JCR, Scopus/CiteScore, or SCImago where appropriate. State the source and access date and distinguish among metrics.

Evaluate journals from at least these families:

- supply-chain management and resilience;
- production and operations management;
- operations research and decision sciences;
- simulation and digital twins;
- applied AI/ML in industrial engineering.

Potential candidates may include, but are not limited to: International Journal of Production Research, International Journal of Production Economics, European Journal of Operational Research, Computers & Industrial Engineering, Decision Support Systems, Transportation Research Part E, Journal of Simulation, and Simulation Modelling Practice and Theory. Do not recommend a venue merely because it is Q1; assess fit with the actual contribution type.

For every candidate, verify:

- scope and accepted contribution types;
- current quartile and metrics, with sources;
- audience and recent comparable papers;
- tolerance for a benchmark-depth study on one validated topology;
- expected algorithmic novelty versus methodological or empirical novelty;
- word/page limits, appendix policy, and code/data expectations;
- open-access/APC requirements and non-APC options;
- available official turnaround information and a cautious estimate of first decision, revision cycles, and total timeline;
- likely desk-rejection reasons;
- qualitative probability of acceptance or advanced revision within 16 months.

Do not invent acceptance rates or review times. If reliable information is unavailable, write “not verified.”

## Questions you must answer

### 1. Independent diagnosis

- What scientific contribution is actually demonstrated?
- What story does Garrido's draft attempt to tell, and what story does the evidence support?
- Can the theoretical framing be reconciled with the evidence, or must it be replaced?
- Does the current evidence support one Q1 paper, two papers, or only one paper now plus a conditional second paper later?
- Where is the novelty: algorithm, action contract, benchmark, mechanism, theory, or audit methodology?

### 2. One-paper versus two-paper decision

Compare three strategies explicitly:

A. One long paper combining Track A, Track B, the prevention boundary, Track B-P, and path dependency.

B. Two papers:
   - Paper 1: action-space alignment and adaptive recovery;
   - Paper 2: strategic reserve posture, contract-regime interaction, and two-stage design.

C. Submit Paper 1 now and pursue Paper 2 only after completing the classical fixed frontier and physical sensitivities.

For each strategy report clarity, novelty, remaining experimental burden, internal contradiction risk, salami-slicing risk, target journals, editorial timeline, and probability of success. End with one unambiguous recommendation.

### 3. Architecture of each recommended paper

For each proposed paper provide:

- provisional title;
- one-sentence claim;
- research question and objective;
- falsifiable hypotheses;
- numbered contributions;
- section structure;
- essential tables and figures;
- body-versus-appendix allocation;
- prohibited/unsafe language;
- three 150–220-word abstracts: conservative, ambitious, and tailored to the primary target journal.

### 4. Minimum publishable package

Separate the remaining work into:

- mandatory blockers;
- strongly recommended work;
- optional/future work;
- experiments that should not be run.

For each blocker specify the required artifact, contrast, scale/seeds, primary endpoint, and promotion or stop criterion.

Explicitly assess whether we need:

- a classical per-operation reserve frontier for Paper 2;
- route-aware replenishment;
- actual-inventory holding cost;
- horizon/regime generalization;
- KAN 8D;
- confirmatory SAC/TD3;
- stronger H4 retained/reset evidence;
- a separate predictor for the “predictive accuracy” claim.

Do not declare everything necessary. Prioritize according to editorial value and the 16-month constraint.

### 5. Journal and submission ladder

Provide a table containing:

- paper;
- journal;
- tier: aspirational / primary realistic / backup;
- topical fit;
- methodological fit;
- strongest selling point;
- primary desk-rejection risk;
- additional work required;
- expected timeline and confidence;
- cascade strategy after rejection.

Recommend one primary journal for each paper and a concrete sequence of backups. Balance aiming high with the requirement that the work be close to publication within 16 months.

### 6. Authorship and Garrido collaboration plan

Provide:

- a Thomas/Garrido CRediT matrix;
- recommended first author and corresponding author, with rationale;
- decisions Thomas can close independently;
- exact inputs and approvals required from Garrido;
- additional contributions Garrido should undertake;
- a proposed meeting agenda;
- a short Spanish email presenting the strategy and requesting those contributions;
- a versioning and approval process before submission.

### 7. Sixteen-month roadmap

Build an aggressive and a conservative month-by-month schedule covering:

- scientific closure;
- writing;
- numerical and visual audit;
- Garrido review;
- preprint, if recommended;
- initial submission;
- revision/response windows;
- deadline for stopping secondary experiments;
- rapid resubmission strategy.

### 8. Reviewer simulation

Write:

- a plausible editorial desk-rejection note;
- a hostile but competent Reviewer #2 report;
- the strongest evidence-based response;
- objections that cannot yet be answered.

## Required output structure

1. **Executive verdict**: maximum 500 words and an unambiguous recommendation.
2. **What is actually proven**: claim table using supported / bounded / unsupported / retracted.
3. **One-paper versus two-paper decision matrix**.
4. **Recommended Paper 1 architecture**.
5. **Recommended Paper 2 architecture**, if applicable.
6. **Journal strategy matrix**, with sources and access dates.
7. **Minimum publishable package**.
8. **Authorship and Garrido collaboration plan**.
9. **Sixteen-month roadmap**.
10. **Reviewer simulation**.
11. **Final action list**, ordered by impact divided by time.

End with three explicit lists:

- `SUBMIT NOW AFTER THESE FIXES`
- `RUN ONLY IF IT CHANGES THE JOURNAL DECISION`
- `DO NOT SPEND TIME ON THIS`

Finish with a decisive recommendation, not “it depends.” If two routes remain plausible, choose one as the primary recommendation and present the other only as a contingency.
