# Mechanism-first primary-source literature review

Date frozen for the development search: 2026-07-13
Purpose: derive necessary mechanisms, bounds, comparator obligations, and falsifiable MFSC hypotheses. A cited result in another objective or domain is not evidence of a Garrido MFSC effect.

## Search and evidence protocol

This is a **targeted, mechanism-first primary-source review**, not a systematic review and not a claim that every 2026 publication was captured.

- **Frozen scope:** value of information/VSS, stochastic and robust control, POMDP/belief sufficiency, inventory/lead times, multi-echelon allocation, maintenance/repair, capacity, routing, queueing, military readiness, sensitivity/simulation optimization, RL environment design and continual learning.
- **Source surfaces:** publisher or DOI landing pages for peer-reviewed original research; PMLR/arXiv for original ML papers; official GAO reports for military-logistics readiness; the Garrido thesis/workbooks for MFSC-native units.
- **Query families:** `perfect information stochastic programming bound`; `censored lost sales inventory learning lead time`; `belief state Markov-modulated demand`; `multi-echelon allocation information delay`; `condition based maintenance finite repair crew imperfect repair`; `adaptive production inventory routing`; `queue admission abandonment`; `continual reinforcement learning transfer retained weights`; and each family combined with `primary research`, `Operations Research`, `Management Science`, `Transportation Science` or the relevant publisher.
- **Include:** a source only when its model exposes a mechanism, structural result, bound, comparator or experimental-design implication relevant to the required chain. Government audits are included only for operational observability/readiness semantics.
- **Exclude:** reviews used only as authority, vendor claims, uncited “RL beats X” headlines, effect sizes from non-comparable objectives and any numerical range not verified in the primary source or MFSC thesis.
- **Range rule:** if a transferable numerical range was not verified, the evidence table says `unavailable / domain-blocked`; absence is not filled from intuition.
- **Currentness limit:** the original discovery/query log was not retained. A reconstructive query-and-inclusion audit was executed on 2026-07-13 and is preserved in [`literature_discovery_inclusion_inventory_20260713.json`](literature_discovery_inclusion_inventory_20260713.json). It verifies the cited source identities and mechanism coverage, but it does not turn this targeted review into a systematic or exhaustive review. A later systematic refresh could add papers without changing any current gate automatically.

Evidence labels used below:

- **SOURCE:** paraphrase or structure supported by the cited primary source.
- **PROJECT INFERENCE/GATE:** a Paper 2 rule or MFSC embedding test derived from the sources; it is not a finding of the cited paper.
- **DOMAIN-BLOCKED:** the MFSC parameter or physical fact is unavailable and must not be invented.

## Mechanism-to-evidence matrix

| Topic/mechanism | Equation or state required | Verified range and unit | Primary source(s) | Falsifiable MFSC hypothesis / status |
|---|---|---|---|---|
| Perfect and partial information | `H_PI^B=E[Y(pi_PI^B)-Y(C_B)]`; information relaxation adds a non-anticipativity penalty | Outcome unit: canonical ReT delta. Practical `0.01` threshold is a PROJECT GATE, not literature-derived. | Avriel & Williams (1970); Brown, Smith & Sun (2010); Balseiro & Brown (2019) | PROJECT GATE: close a frozen cell only with a valid resource-restricted upper confidence bound below `0.01`; otherwise compute a tighter bound. |
| Value of the stochastic solution / robust mean-value policy | `VSS=E[Y(x_stoch)]-E[Y(x_mean)]` under the same uncertainty model | Transferable MFSC range unavailable / domain-blocked. Delage et al.'s reported setting is two-stage mixed-integer with objective uncertainty, not the MFSC. | Delage, Arroyo & Ye (2014) | Compare the best robust/mean-value calendar with the stochastic policy before attributing value to feedback. |
| Censored lost-sales information and POMDP belief | `S_t=min(D_t,I_t)` and posterior `b_{t+1}(theta) proportional to P(S_t|theta,I_t)b_t(theta)` | Thesis demand anchor: 2,400-2,600 rations/day, six days/week. True censoring, reporting cost and signal delay are unavailable / domain-blocked. | Smallwood & Sondik (1973); Chen (2010); Huh et al. (2009); Zhang, Chao & Shi (2020); Chen et al. (2024) | SOURCE-to-domain test: if every requested order is logged, censoring is false. A sensing action must reveal otherwise unavailable information before an irreversible decision and consume a matched resource. |
| Lost-sales inventory with positive lead time | Inventory position/pipeline is part of the control state; compare base-stock, `(s,S)`, constant-order and belief policies | Thesis promised downstream `LT=48 h`; any new replenishment lead-time distribution is unavailable / domain-blocked. | Zhang, Chao & Shi (2020); Goldberg et al. (2016) | Observable headroom must survive optimized constant-order/base-stock policies; a large perfect-information gap alone is insufficient. |
| Multi-product substitution | `x_i,t+1=x_i,t+u_i,t-L-sum_j y_ij,t`, with `y_ij=0` if compatibility `M_ij=0` | Thesis anchor: 21 ration types; 12-component BOM is known only for Cold Weather #1. Product shares, other BOMs and `M_ij` are unavailable / domain-blocked. | Nagarajan & Rajagopalan (2008); Chen & Chao (2020) | Null: symmetric BOM/compatibility removes product-specific observable value. Positive test requires approved suitability and a binding shared ingredient/assembly/lift constraint. |
| Advance information, regime switching and capacity reservation | `sum_i r_i,t<=C`; activation after `d`; commitment locked for `m`; belief over regime `Z_t` | Thesis anchor: S1-S3 are 8 h shifts; line rate 320.5 rations/h. Reservation capacity, `d`, `m`, cancellation and transition probabilities are unavailable / domain-blocked. | Ozer & Wei (2004); Boada-Collado, Chopra & Smilowitz (2022); Avci, Gokbayrak & Nadar (2020) | Signal must arrive before commitment and change the preferred scarce allocation. Null: immediate costless cancellation/retargeting removes the asserted commitment mechanism. |
| Dynamic shift/capacity allocation | Capacity pipeline/ramping state must persist across decision epochs | Native range: S=1,2,3; 8 h/shift; effective weekly hours 48, 96, 144 after maintenance. Activation delay, overtime cost and dwell are unavailable / domain-blocked. | Lorca & Sun (2016); Garrido-Rios thesis Sec. 6.3.2 | A shift extension is new only if validated lead/dwell makes actions incompatible across time; instantaneous free shifts reduce to a strong reactive/static capacity comparator. |
| Multi-echelon allocation and information lag | `O_t^lag subset O_t^current`; pipeline and local inventory/backlog must be included in state | Native downstream transport legs are 24 h. Separate decision authorities, reporting delay distribution and local irrevocability are unavailable / domain-blocked. | Zipkin (1984); Moinzadeh (2002) | Information lag alone cannot improve the optimum with unchanged actions/physics. A new family requires a distinct earlier irreversible allocation authority. |
| Lateral transshipment and inventory routing | Domination screen: `tau_AB>=tau_SB,B`, same lift and positive central stock imply direct central dispatch is no later | Native Op10/Op12 travel is 24 h. Lateral lanes, local lift, payload and turnaround are unavailable / domain-blocked. | Paterson, Teunter & Glazebrook (2012); Olsson (2015); Kleywegt, Nori & Savelsbergh (2004) | Lateral transfer must be faster, use distinct local lift or become informed after central dispatch locks; otherwise test central-dispatch domination. |
| Condition-based maintenance and finite repair crews | Controlled degradation state `P(X_t+1|X_t,a_t)` plus crew location/commitment; imperfect repair needs a post-repair state distribution | Thesis anchors: R11 mean repair about 2 h; R21/R23 mean recovery 120 h. Crew count/travel and imperfect-repair factor are unavailable / domain-blocked. | Abbou & Makis (2019); Sanoubar et al. (2023) | Whittle/index, threshold and exact small MDP comparators are mandatory. Current review has no verified primary-source MFSC range for imperfect repair; do not add one. |
| Queue admission, priority and abandonment | Queue state by class/deadline; rejection must enter the lost-demand ledger | Native pending-list cap is 60 with SPT and contingent priority. Mission expiry/patience and rejection authority are unavailable / domain-blocked. | Ata (2006); Long et al. (2020) | Null: without physical expiry or rejection authority, an admission action is not live. Priority value remains a separate comparator question and is not closed by that null. |
| Adaptive/robust stochastic optimization | Non-anticipative scenario-tree or partition/affine decision rule under the same uncertainty set and resources | Uncertainty-set/rule parameters unavailable until a candidate contract is frozen. | Postek & den Hertog (2016); Lorca & Sun (2016); Cui et al. (2023) | Any learner must beat the strongest resource-matched robust/adaptive rule or a certified bound on it. |
| Integrated production/inventory routing | Conservation over production, setup, inventory, vehicle capacity and route/visit decisions | Native product flow and 24 h LOC times exist; fleet count, payload, setup and turnaround are unavailable / domain-blocked. | Adulyasak, Cordeau & Jans (2015); Zhang et al. (2021); Chitsaz, Cordeau & Jans (2019) | A joint learner cannot be compared only with separately tuned production and routing heuristics; the comparator must jointly optimize the conserved resources. |
| Global sensitivity of policy value | Morris elementary effects; Sobol `S_i=Var(E[H|X_i])/Var(H)` with `H` equal to paired headroom/guardrails | Every input distribution/range must be frozen; no generic literature range transfers to the MFSC. | Morris (1991); Sobol (2001) | Screen `H_PI`, `H_obs`, resource-adjusted `H_obs` and guardrails, not raw policy ReT; report every frozen cell including nulls. |
| Simulation optimization / selection bias | Prespecified indifference zone and probability of correct/near-best selection | PROJECT practical difference is ReT `0.01`; sample counts require a frozen power/selection design. | Boesel, Nelson & Kim (2003) | Development winners require independent validation; a selected sample maximum is not confirmatory evidence. |
| RL environment generalization and trajectory audit | Disjoint generator sets; `a_t` dependence on state after controlling for phase; modal/calendar replacement | Seed ranges are project contract data, not supplied by Procgen. No unburned block is named here. | Cobbe et al. (2019); Justesen et al. (2018) | If a calibration-only fixed sequence reproduces the learner under matched resources, classify the result as open-loop schedule discovery. |
| Military-logistics readiness/observability | End-to-end wait, final destination, cost/resource and mission-service ledgers | GAO supplies semantic requirements, not MFSC numerical ranges; communications availability and signal ownership are unavailable / domain-blocked. | GAO-15-226; GAO-21-125; GAO-21-313 | A route/threat signal must specify observer, time, communications mode and decision echelon; mean ReT alone cannot establish readiness. |
| Perishable/age-sensitive inventory | Age-bucket conservation and expiry rule | Thesis product life is approximately 3 years at <=27 C; no short-horizon expiry within the modeled product contract. | Garrido-Rios thesis Sec. 6.3 | DOMAIN EXCLUSION: do not use a two-week shelf life. A perishable family requires a different validated product, not retuning this ration. |
| Factory/APP Cobb-Douglas construct | `R=sigma(.024 ln zeta-.026 ln epsilon+.040 ln phi-.060 ln tau-.1771 ln kappa_dot)` | Published exponents are dimensionless and calibrated for a factory APP construct; they are not MFSC order-level ReT weights. | Garrido, Ponguta & Garcia-Reyes (2025; online 2024) | Construct-sensitivity only for current Paper 2. It cannot rescue quarantined visible-v1 results or substitute for `ret_excel_request_snapshot_v2`, and cannot be transferred to spatial allocation without a new validation study. |
| Retained/continual learning | `H_retained=E[Y(persistent)-Y(reset)]` under matched campaigns, information and compute | Campaign persistence, ordering and compute budgets unavailable until Paper 2 learned value exists. | Powers et al. (2022); Kaplanis et al. (2018, 2019); Caccia et al. (2023); Knoblauch et al. (2020) | Paper 3 remains conditional: persistent must beat reset, scratch, frozen and structured belief controls on virgin campaign sequences. |

## 1. Information value is an upper bound, not a deployable result

Avriel and Williams define the expected value of perfect information (EVPI) as the expected improvement available when uncertain outcomes are revealed before planning; by construction it is an upper bound on what any information-acquisition intervention is worth ([Operations Research, 1970](https://doi.org/10.1287/opre.18.5.947)). Brown, Smith, and Sun develop information-relaxation bounds for dynamic programs: reveal future uncertainty, then subtract a dual penalty for violating non-anticipativity ([Operations Research, 2010](https://doi.org/10.1287/opre.1090.0796)). Balseiro and Brown show why penalized information relaxations can be much tighter than a raw perfect-information relaxation ([Operations Research, 2019](https://doi.org/10.1287/opre.2018.1782)).

For this search:

\[
H_{PI}^{B}=E[Y(\pi_{PI}^{B})-Y(C_B)]
\]

is a resource-restricted diagnostic ceiling. It does not establish `H_obs`. When raw perfect information is too valuable, compute a dual-penalized ceiling before declaring a family unresolved.

The cheap pre-simulation relaxation is

\[
UB_{\mathrm{affected}}
=E\left[\frac{1}{N}\sum_{j=1}^{N}
\max_a\{ReT_j(a)-ReT_j(C_B),0\}\right].
\]

It drops shared-resource coupling and is therefore optimistic. **PROJECT GATE:** a bootstrap UCB95 below 0.01 is sufficient to close the screened cell under this project's frozen practical threshold. The threshold is not a result of the information-value papers. A value above 0.01 only licenses the exact resource-restricted calculation.

Do not assume per-row ReT is capped at one. The exact workbook formula can exceed one for short recovery periods, and policy actions can change which rows are visible. The shortcut `affected rows / total rows` is valid only for an explicitly clipped sensitivity with a fixed denominator. The primary bound must optimize the canonical numerator and visible-row denominator directly through the repository aggregator.

**PROJECT ARTIFACT; M/T/R state unchanged here.** The M/T/R locked-tape audit obtains a valid per-tape row bound from integer-hour delivery timing, yet its mean upper gap is `138.185` because the smallest event-to-delivery gap can be tiny and almost every order lies after the first sensitive event. A physically simultaneous M+T+R relaxation is also not a canonical upper bound: changing completion and visibility can reverse the workbook metric. This review records that metric warning only; it does not alter the family's registry/verdict state. Exact quotient/optimization remains necessary.

The value of the stochastic solution (VSS) similarly asks whether explicitly modeling uncertainty improves on the expected-value solution. Delage, Arroyo, and Ye derive tractable bounds showing that a mean-value solution can be robust in specific two-stage mixed-integer settings ([Operations Research, 2014](https://doi.org/10.1287/opre.2014.1318)). This is a warning against assuming that a stochastic or learned controller must beat a robust calendar.

## 2. Censored lost-sales inventory has real learning value, but only under true censoring

Smallwood and Sondik establish the finite-state POMDP result relevant here: the current probability distribution over latent states is a sufficient control state, and the finite-horizon value function is piecewise-linear and convex in that belief ([Operations Research, 1973](https://doi.org/10.1287/opre.21.5.1071)). This does not imply that the repository observation is sufficient; the transition and observation kernels still have to be specified and tested.

Chen derives Bayesian inventory control when demand beyond available inventory is not observed, making the posterior a control state ([Operations Research, 2010](https://doi.org/10.1287/opre.1090.0726)). Huh et al. give an adaptive policy for lost-sales inventory with censored demand ([Mathematics of Operations Research, 2009](https://doi.org/10.1287/moor.1080.0367)). Zhang, Chao, and Shi obtain regret guarantees for lost-sales systems with positive lead times, where actions have lasting consequences ([Management Science, 2020](https://doi.org/10.1287/mnsc.2019.3288)). Chen et al. treat simultaneous censored demand and uncertain supply ([Management Science, 2024](https://doi.org/10.1287/mnsc.2022.02476)).

Necessary MFSC mechanism:

\[
S_t=\min(D_t,I_t),\qquad
b_{t+1}(\theta)\propto P(S_t\mid\theta,I_t)b_t(\theta),
\]

with true demand sealed from the policy but retained by the evaluator. **PROJECT EMBEDDING TEST:** an active reporting action must consume an operational resource and arrive before replenishment commitment. If the MFSC records every requested order even under a stockout, censoring is false. Without an active information action the proposed mechanism does not add censored-demand learning value beyond the existing replenishment family.

Comparator obligation: exact belief DP in the smallest screen, posterior base-stock/`(s,S)` policies, optimized constants, all full-horizon order/report calendars, and MPC. Goldberg et al. show that constant-order policies can be asymptotically optimal for lost-sales inventory with large lead times ([Mathematics of Operations Research, 2016](https://doi.org/10.1287/moor.2015.0760)); a constant is not a weak comparator here.

## 3. Multi-product substitution can create ranking reversals, but structured controls may solve it

The thesis explicitly compresses 21 real ration types into one product, so product mix is a domain-anchored omission. A conservative multi-product ledger is

\[
x_{i,t+1}=x_{it}+u_{i,t-L}-\sum_j y_{ijt},
\qquad y_{ijt}=0\ \text{if}\ M_{ij}=0,
\]

with shared ingredient, assembly, transport-weight, and volume constraints. The requested SKU remains fixed in a sealed evaluator ledger; substitution counts as fulfillment only if its mission/climate/nutritional compatibility is approved before results.

Nagarajan and Rajagopalan establish optimal or near-structured policies in substitution systems, so product labels do not automatically require a neural controller ([Management Science, 2008](https://doi.org/10.1287/mnsc.1080.0871)). Chen and Chao study learning with stockout-based substitution when primary demand and substitution probabilities are unknown, explicitly using structured exploration ([Management Science, 2020](https://doi.org/10.1287/mnsc.2019.3474)).

Necessary ranking reversal:

\[
E[D_{i,t+L:t+L+m}-IP_i\mid O_t]
\gtrless
E[D_{k,t+L:t+L+m}-IP_k\mid O_t]
\]

under a binding shared production/ingredient resource, product-specific commitment, and persistent deployable signal. The null cell makes BOMs and suitability symmetric; product labels must then have zero observable value.

Comparator obligation: every full-horizon product-mix calendar, robust mix, approved substitution doctrine, product-specific base-stock, min-cost/network-flow allocation, rolling deterministic/stochastic MILP, and exact two-product DP.

## 4. Advance information has value only before an irreversible capacity commitment

Özer and Wei show that advance demand information interacts with limited capacity and commitment decisions ([Operations Research, 2004](https://doi.org/10.1287/opre.1040.0126)). Boada-Collado, Chopra, and Smilowitz analyze complementarity between information and flexible capacity under timing/commitment ([Manufacturing & Service Operations Management, 2022](https://doi.org/10.1287/msom.2022.1090)). Avci, Gokbayrak, and Nadar derive belief-dependent structure for inventory under Markov-modulated demand ([Production and Operations Management, 2020](https://doi.org/10.1111/poms.13088)).

The minimal reservation mechanism is

\[
\sum_i r_{i,t}\le C,\quad
r_{i,t}\text{ activates after }d,\quad
r_{i,t}\text{ is locked for }m,
\]

with regime-dependent lead time `L_i(Z_it)` and a signal observable at least `d` before activation. **PROJECT FALSIFICATION CONDITION:** if reservation is immediately and costlessly cancelable, unused capacity automatically retargets, or the signal arrives after commitment, the proposed intertemporal-commitment mechanism is absent; the cited papers do not prove that every such variant is dominated. This candidate is not thesis-native until Garrido confirms finite shared lift, activation lead, dwell, cancellation, and signal semantics.

Comparator obligation: all full-horizon reservation calendars, fixed/periodic route shares, robust reservation, belief DP, stochastic MPC with pipeline state, and convexified policies under equal reserved-capacity-hours.

## 5. Spatial allocation and lateral transshipment face a domination test

Dynamic transshipment can improve inventory pooling in systems with stock imbalance and transfer lead time ([Paterson, Teunter, and Glazebrook, European Journal of Operational Research, 2012](https://doi.org/10.1016/j.ejor.2012.03.005); [Olsson, European Journal of Operational Research, 2015](https://doi.org/10.1016/j.ejor.2014.10.015)). Dynamic inventory-routing models likewise use ADP or strong routing controls rather than a static-only baseline ([Kleywegt, Nori, and Savelsbergh, Transportation Science, 2004](https://doi.org/10.1287/trsc.1030.0041)).

For the MFSC, lateral A-to-B transfer is weakly dominated when all hold:

\[
\tau_{AB}\ge\tau_{SB,B},\quad
\text{same lift is consumed},\quad
I_{SB}>0.
\]

Direct SB-to-B delivery arrives no later and does not deplete A. Lateral transshipment is genuinely new only if it is faster, uses distinct local lift, becomes informed only after central dispatch is locked, or central stock is unavailable. Otherwise it reduces to Program G/DRA-1.

## 6. Finite maintenance crews are adaptive problems, but strong indices and MDPs are required

Group maintenance with multiple degrading facilities and limited repairers has a restless-bandit formulation and a Whittle-index comparator ([INFORMS Journal on Computing, 2019](https://doi.org/10.1287/ijoc.2018.0863)). Sanoubar et al. formulate mobile condition-based maintenance as an MDP jointly deciding travel, idle, and repair actions ([Transportation Science, 2023](https://doi.org/10.1287/trsc.2021.0302)).

Program J already showed that causal maintenance allocation in the serial Op5-Op7 line has negligible canonical ReT ceiling. A materially new alarm family must add a domain-valid precursor that predicts a failure whose downstream inventory cannot cover, plus a scarce crew with travel/repair commitment. It must face a Whittle/index rule, condition-based thresholds, exact small-state MDP, and all maintenance calendars. Merely improving signal accuracy or breakdown probability reparameterizes J/I and is blocked.

The closest thesis-anchored scarcity-by-concurrency question is different from Program J: one R21/R3 event disables several operations concurrently and the thesis names the Maintenance Battalion. Current executable recovery remains independent and parallel, so it has no sequencing action and exact current-physics headroom is zero. A scarce mobile-team extension cannot be called falsified from restoration-order invariance without a machine proof; it is `blocked_domain_fact` until team count/skills, travel/activation, repair-work distributions, preemption, and pre-dispatch damage-assessment timing/error are supplied. If licensed, it must face every fixed restoration permutation, critical-path/WSPT rules, exact DP and resource-constrained MPC before a learner.

## 6a. Inspection value requires a real quality state, scarce test capacity and qualified-output consequences

Primary inspection-allocation models make the mechanism explicit: limited inspection machines constrain production, while the decision chooses inspection location or fraction ([Bai and Yun, *Computers & Industrial Engineering*, 1996](<https://doi.org/10.1016/0360-8352(96)00008-3>)); integrated production-system design chooses inspection and production capacity jointly ([Tang, *European Journal of Operational Research*, 1991](<https://doi.org/10.1016/0377-2217(91)90334-R>)). A two-state in-control/out-of-control model with imperfect classification supports an exact dynamic inspection/disposition policy ([European Journal of Operational Research, 2009](https://doi.org/10.1016/j.ejor.2008.01.026)). These are mechanism papers, not evidence that the MFSC has the required state or authority.

The thesis-native MFSC instead has fixed Op7 quality control and independent Binomial R14 defects at `p=.03/.08`; it says a detected item returns to Op6 but does not establish 100% detection. The current DES treats every generated defect as detected and has no inspection action, so incremental current-kernel `H_PI=H_obs=0`. A valid extension must CRN-couple exogenous per-unit/lot latent draws, account separately for inspected true/false classifications and uninspected defective/conforming units, complete held/rework/scrap flows, prevent unqualified output from fulfilling orders, and keep R14 metric activation independent of the detection action. It must face all full-horizon inspection calendars, fixed AQL plans, Shewhart/EWMA/CUSUM/SPRT rules, Bayesian belief thresholds, and DP/POMDP/MPC under equal inspector-hours. Merely increasing R14 frequency reopens only the closed buffer/shift mechanism.

## 6b. Component-kit control is an assemble-to-order problem only if components and lift are real

Joint replenishment and allocation in a multicomponent assemble-to-order system is formulated as a two-stage stochastic integer program; the component-allocation subproblem is a multidimensional knapsack problem ([Akçay and Xu, *Management Science*, 2004](https://doi.org/10.1287/mnsc.1030.0167)). With capacitated component production, transport lead and fixed shipment cost, dynamic component production, expedition/shipping and order sequencing face threshold or Brownian-control comparators ([Plambeck, *Operations Research*, 2008](https://doi.org/10.1287/opre.1070.0497)). More recent asymptotic control uses multistage stochastic programming and component-allocation policies that depend on BOMs, component availability, backlog costs and outstanding lead times ([Reiman, Wan and Wang, *Stochastic Systems*, 2023](https://doi.org/10.1287/stsy.2022.0099)).

For MFSC embedding, those ingredients cannot be inferred from the thesis's statement that 12 raw materials exist. The current DES aggregates them, applies R13 as a common Op2 delay and supplies no finite component-load Op4 action, hence current-kernel `H_PI=H_obs=0`. The distinct kit-balancing contract must freeze one finished-ration type and remove output-product choice/substitution. Before simulation Garrido must confirm named component delays, BOMs, component on-hand/pipeline observability, finite mixed-load lift and any expedition/substitution authority. The candidate must face a certified full-horizon MILP/open-loop bound, proportional-kit and component-base-stock doctrines, kit-bottleneck heuristics, robust mix and DP/MPC under identical component and vehicle ledgers. If a real component identity exists but is hidden before dispatch, only `H_obs=0` follows; perfect-information value may remain positive.

## 7. Queue admission requires real mission expiry, not demand shedding

The thesis backlogs late orders and only uses a cap-60 pending list. It does not establish operational abandonment. Queueing work provides strong dynamic/static-index comparators for abandonment and admission, including due-date admission/sequencing and multiclass priority rules ([Ata, Operations Research, 2006](https://doi.org/10.1287/opre.1060.0308); [Long et al., Operations Research, 2020](https://doi.org/10.1287/opre.2019.1908)).

An admissible extension needs physical mission deadlines, observed classes, explicit rejection authority, and a ledger in which rejection is lost demand. **PROJECT EMBEDDING TEST:** without physical expiry or rejection authority, an admission action is not live. This does not prove that queue priority/sequencing has no value; that is a separate same-contract comparator question. Under the current thesis facts, abandonment remains domain-blocked.

## 8. Multi-echelon information lag cannot create value by itself

For unchanged physics/actions,

\[
O_t^{lag}\subset O_t^{current}
\Rightarrow J^*(O^{lag})\le J^*(O^{current}).
\]

Adding reporting lag to Program G/H therefore cannot rescue it. A genuinely new multi-echelon problem needs separate central and local authorities, an irrevocable 24 h pipeline decision, local rationing authority, a binding shared resource, and persistent state. It must face echelon base-stock, delayed-information policies, decentralized rules, belief DP, and stochastic-program bounds; see the dynamic allocation structure in [Zipkin, Mathematics of Operations Research, 1984](https://doi.org/10.1287/moor.9.3.402) and information exchange in [Moinzadeh, Management Science, 2002](https://doi.org/10.1287/mnsc.48.3.414.7730).

## 9. Global sensitivity must screen adaptive value, not raw outcome

Morris screening is appropriate for sparse qualitative factor effects ([Technometrics, 1991](https://doi.org/10.1080/00401706.1991.10484804)); Sobol indices decompose variance under explicit input distributions ([Mathematics and Computers in Simulation, 2001](<https://doi.org/10.1016/S0378-4754(00)00270-6>)). In this project the response is not `ReT(policy)` but the paired CRN contrast `H_PI`, `H_obs`, resource-adjusted `H_obs`, and guardrail deltas. Program I already found a scarcity-by-concurrency interaction while signal quality and risk magnitude were inert in its contract.

The cell-selection rule must precede results:

1. screen all frozen cells with common tapes;
2. reject cells whose affected-order UCB or resource-restricted `H_PI` UCB is below 0.01;
3. require adjacent connected cells, action support, and null-cell collapse;
4. select the least-favorable plausible connected survivor, not the maximizer;
5. open fresh development-validation tapes only after the rule is frozen.

### 9.1 Adaptive and robust stochastic optimization

Multistage robust optimization distinguishes a static decision from a decision rule that can react only to uncertainty revealed so far. Postek and den Hertog construct integer and continuous decision rules by splitting the uncertainty set, retaining non-anticipativity while allowing later actions to differ by revealed scenarios ([INFORMS Journal on Computing, 2016](https://doi.org/10.1287/ijoc.2016.0696)). Lorca and Sun's multistage unit-commitment model shows the operative mechanism: time causality matters when ramping capacity is limited and uncertain loads vary enough for earlier commitments to bind ([Operations Research, 2016](https://doi.org/10.1287/opre.2015.1456)). Cui et al. formulate adaptive inventory-routing decisions under distributional ambiguity and solve their finite-horizon model exactly ([Operations Research, 2023](https://doi.org/10.1287/opre.2022.2407)).

MFSC implication: “robust” and “adaptive” are comparator classes, not labels of quality. A candidate needs a state variable revealed before a binding action, an explicit uncertainty set or probability law, and a robust/open-loop decision rule using the same resources. Affine/partition rules and exact scenario-tree bounds must precede a neural claim.

### 9.2 Production and inventory routing

Adulyasak, Cordeau, and Jans jointly optimize production setup, customer visits, production quantities and delivery quantities in two-stage and multistage stochastic production routing, using Benders decomposition and rollout ([Operations Research, 2015](https://doi.org/10.1287/opre.2015.1401)). Zhang et al. provide an exact Benders method for a multivehicle production-routing problem with order-up-to replenishment ([Transportation Science, 2021](https://doi.org/10.1287/trsc.2019.0964)). Chitsaz, Cordeau, and Jans integrate assembly, inventory and routing with explicit setup and vehicle subproblems ([INFORMS Journal on Computing, 2019](https://doi.org/10.1287/ijoc.2018.0817)).

MFSC implication: an integrated production-routing extension is credible only when vehicle capacity, travel/turnaround, setup, product inventory and commitment are all conserved. The comparator must jointly optimize them; comparing a sequential learner with separately tuned production and routing heuristics would be structurally weak.

### 9.3 Military-logistics readiness and observability

Primary government audits emphasize endpoint and information semantics rather than an abstract “AI advantage.” GAO's materiel-distribution audit requires reliable end-to-end performance data, customer wait time, cost and final-destination outcomes ([GAO-15-226](https://www.gao.gov/products/gao-15-226)). Its contested-mobility audit documents that transport under opposition is a readiness problem requiring tracked implementation and training, not simply a larger stochastic delay ([GAO-21-125](https://www.gao.gov/products/gao-21-125)). Its Army logistics-system audit reports that asset visibility and reporting can improve operations but may fail in disconnected combat settings ([GAO-21-313](https://www.gao.gov/products/gao-21-313)).

MFSC implication: a proposed route signal must specify who observes it, when, under what communications availability, and at what decision echelon. Readiness claims must retain customer wait, final destination, cost/resource and mission-service guardrails; mean ReT alone is not a military-readiness construct.

### 9.4 Simulation optimization and selection bias

Boesel, Nelson, and Kim show that the best sample mean found during a large simulation search is not necessarily the best system; ranking-and-selection procedures require additional samples and a prespecified indifference zone to identify a best or near-best design with controlled probability ([Operations Research, 2003](https://doi.org/10.1287/opre.51.5.814.16751)).

MFSC implication: Morris, Sobol, Bayesian optimization or heuristic search may locate a region, but they do not make the selected cell confirmatory. Cell selection, practical indifference zone, adjacent-region rule and seed blocks must be frozen before new tapes. The confirmatory comparison uses paired CRN inference on a once-opened universe; the development winner's sample mean is not reused.

### 9.5 RL-environment and equilibrium-policy tests

Cobbe et al.'s Procgen benchmark separates training and unseen generated environments and shows that environment diversity is central to measuring generalization ([arXiv:1912.01588](https://arxiv.org/abs/1912.01588)). Justesen et al. similarly find that generalization depends on the level-generator distribution, not just the learning algorithm ([arXiv:1806.10729](https://arxiv.org/abs/1806.10729)). For this DES, random seeds are therefore part of the contract: development, calibration, validation and virgin generators must be disjoint and their distributions frozen.

The stronger equilibrium-policy test is domain-specific: enumerate complete action trajectories, condition action choice on observable state after controlling for episode phase, and replace the learner with its modal sequence, calibration-only calendar and phase controller. If the replacement reproduces outcomes, the learner found an open-loop equilibrium/calendar. This is a scientific classification test, not a generic RL benchmark score.

### 9.6 Garrido factory-resilience Cobb-Douglas construct

Garrido, Ponguta, and Garcia-Reyes combine Monte Carlo demand variability, pure aggregate-production-planning strategies, and a calibrated factory-resilience Cobb-Douglas index ([International Journal of Production Research, 2025; published online 2024](https://doi.org/10.1080/00207543.2024.2425771)). In the paper's log form,

\[
\operatorname{logit}(\mathcal R)
=0.024\ln\zeta-0.026\ln\epsilon+0.040\ln\phi
-0.060\ln\tau-0.1771\ln\dot\kappa,
\]

where the inputs are factory APP inventory, backorders, spare capacity, fulfillment time, and cost. The paper assumes demand variability as the uncertainty and reports a fixed zero-inventory APP strategy as the preferred strategy in its tested design. These coefficients are therefore primary evidence for that factory construct, not transferable weights for order-level MFSC distribution resilience and not evidence that adaptive feedback beats a full-horizon calendar.

MFSC implication: retain the published coefficients exactly when reporting this construct as a frozen secondary sensitivity. A Cobb-Douglas ranking cannot replace `ret_excel_request_snapshot_v2`, rescue quarantined visible-v1 results, or waive lost-order and worst-node guardrails. A factory-APP decision contract could elevate the construct only through a separate preregistration and validation that preserves the canonical Paper 2 success definition or explicitly changes the paper's scientific question before results.

## 10. Learned-policy and retained-learning evidence

Powers et al. define continual RL as performance under changing tasks/environments and provide baselines in the CORA framework ([PMLR 199, 2022](https://proceedings.mlr.press/v199/powers22b.html)). Kaplanis et al. use policy consolidation to retain knowledge across tasks ([PMLR 97, 2019](https://proceedings.mlr.press/v97/kaplanis19a.html)) and complex synapses to reduce forgetting at multiple timescales ([PMLR 80, 2018](https://proceedings.mlr.press/v80/kaplanis18a.html)). Caccia et al. explicitly study task-agnostic continual RL and faster adaptation under limited data/compute ([PMLR 232, 2023](https://proceedings.mlr.press/v232/caccia23a.html)). Knoblauch, Husain, and Diethe show that optimal continual learning generally needs perfect memory and is NP-hard, helping explain why replay/memory controls can dominate regularization-only methods ([PMLR 119, 2020](https://proceedings.mlr.press/v119/knoblauch20a.html)). These works establish that retention, interference, plasticity and memory access require explicit controls; they do not make persistent weights evidence of organizational learning.

For Paper 2, a learner must beat the maximum over open-loop, interpretable, belief, DP/ADP, and MPC comparators under identical resources. Full action trajectories are data. The decisive replacement tests are its modal sequence, calibration-optimized tape-independent calendar, and phase-only controller.

For Paper 3, after and only after learned Paper 2 value:

\[
H_{retained}=E[Y(\text{persistent})-Y(\text{reset})]
\]

on virgin campaign sequences, with the same campaign multiset, information, interactions, compute, resources, replay memory and ordering controls. Frozen-previous-policy and scratch-matched-compute controls separate retention from curriculum luck or extra training. Report forward transfer, backward transfer/forgetting and current-campaign performance separately; an average that hides degradation of prior campaigns is not retained value.

## 11. Literature-derived decision, not a claim of success

The cited sources contain problem classes with structured adaptive solutions, learning/regret results or adaptive-vs-benchmark comparisons. Their objectives, units and parameter regimes do not transfer an effect size to the MFSC. They simultaneously raise the comparator bar: belief DPs, base-stock policies, Whittle indices, network-flow/MILP controls, and stochastic MPC may capture all observable value. None of these sources implies neural incremental value in canonical Garrido ReT.

Current theory-first ranking:

1. Multi-ration product mix: best domain anchor, blocked on product and signal facts.
2. Advance capacity reservation with regime-dependent lead time: coherent mechanism, blocked on transport facts.
3. Censored demand with active reporting: coherent problem class, blocked because current MFSC records demand and no sensing action is documented.
4. Lateral transshipment: blocked by central-dispatch domination absent distinct facts.
5. Alarm maintenance: blocked as a reparameterization of J unless a new precursor/travel commitment is validated.
6. Admission/abandonment and information lag alone: formally reduce to closed or weaker contracts under current physics.
