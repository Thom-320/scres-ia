# Learning-Augmented Event-Triggered MPC for Supply-Chain Resilience under Nonstationary Disruptions

## Exact static benchmarks and full discrete-event validation

**Target journal:** Computers & Industrial Engineering  
**Document status:** pre-results manuscript, frozen scientific framing; no hybrid result is asserted  
**Model status:** Garrido-informed, thesis-grounded reconstruction and researcher extension

## Abstract

Discrete-event simulation can reproduce disruption propagation and recovery in
complex supply chains, yet its decision logic commonly remains open loop: model
outputs are inspected after a run rather than converted into state-contingent
interventions during the run. This study develops a learning-augmented,
event-triggered model predictive controller for a two-product extension of a
thesis-grounded military food supply-chain DES. The controller combines a causal
recurrent belief representation, a distributional continuation-value model,
constrained scenario MPC, and a calibrated fallback to robust feasible MPC. The
method is evaluated only after three pre-learner conditions are established:
residual observable headroom beyond strengthened MPC, value from endogenous
review timing under a matched attention budget, and a decision-relevant
nonstationary risk envelope. The experimental design compares the proposed
method with the complete set of 65,536 open-loop product-mix calendars, pure
recurrent reinforcement learning, exact dynamic programming on a reduced
benchmark, and preregistered nominal, particle, scenario, robust,
constraint-aware, and event-triggered MPC controllers. Information, production,
transport, review rights, and online computation are matched. The primary
endpoint is canonical order-level ReT under `ret_excel_request_snapshot_v2`,
with frozen robustness to both plausible simultaneous-event orderings. The
study is designed to identify when learned belief, continuation value, and
review timing add value—and when strengthened structured control makes them
unnecessary. Results for the integrated hybrid remain gated and will be added
only from virgin confirmation artifacts.

**Keywords:** supply-chain resilience; discrete-event simulation; model
predictive control; reinforcement learning; event-triggered control; partial
observability; simulation optimization

## 1. Introduction

Supply-chain resilience models must represent both the physical consequences of
disruption and the decisions through which organizations respond. Discrete-event
simulation (DES) is well suited to the first task because it preserves queues,
lead times, resource contention, inventory propagation, and event timing. It is
less often used as a closed-loop decision system. A simulated chain may undergo
thousands of disruptions while its policy remains fixed, so simulation
experience informs the analyst but not the next in-model decision.

Garrido, Pongutá, and Adarme (2024) identify this separation as a limitation in
the operationalization of simulation-based supply-chain resilience and propose
AI algorithms as a bridge between DES outputs and decision variables. Their
contribution is exploratory: it motivates neural and reinforcement-learning
mechanisms but does not implement a closed-loop controller or establish that a
learned method improves resilience. The present study operationalizes the
within-campaign part of that agenda. It does not treat the mere presence of a
neural network as evidence of learning, and it does not yet study knowledge
retained across campaigns.

The empirical setting is a reconstructed version of the 13-operation Military
Food Supply Chain described by Garrido-Ríos (2017). Because the original
Simulink implementation and all endogenous workbook distributions are
unavailable, the Python model is described as a Garrido-informed,
thesis-grounded reconstruction and researcher extension—not as an exact
reproduction. The extension introduces two nonfungible product classes sharing
fixed production and transport rights. Every policy selects only the allocation
of those rights; no policy creates inventory, capacity, or transport.

Earlier experiments in this environment establish an unusually demanding
baseline. A complete frontier of (4^8=65{,}536) open-loop calendars is
available. Program Q independently confirmed that a recurrent learned policy
outperformed this complete open-loop frontier and was statistically equivalent
in canonical ReT to the strongest frozen belief-state controller. It did not
establish an incremental neural premium, and its compound worst-product service
gate failed. The integrated study therefore asks a narrower and harder
question: can learned belief, continuation value, and endogenous review timing
improve a strengthened, constraint-aware MPC under equal information, physical
resources, attention, and online computation?

The contributions are fourfold. First, the study separates efficient static
policy discovery, observable feedback value, timing value, and hybrid
incremental value. Second, it provides exact static and reduced-POMDP benchmarks
before evaluating a full DES. Third, it introduces a confidence-gated
learning-augmented scenario-MPC architecture with a feasible robust fallback.
Fourth, it uses full action-trajectory, information-placebo, service,
resource, and direct-DES replay audits to distinguish genuine feedback from
schedule discovery or hidden resource use.

## 2. Research questions and hypotheses

The central question is:

> Can a simulator-budgeted learning-augmented planner outperform the complete
> open-loop frontier, pure recurrent RL, and strong belief-, robust-, and
> event-triggered MPC under matched information, resources, intervention
> rights, review budgets, and online computation?

Four prospectively ordered hypotheses follow.

**H1 — Static search efficiency.** Under an equal calendar-by-tape simulator
budget, simulation-based optimizers approach the exact static optimum without
reading the enumerated score matrix during search.

**H2 — Observable feedback value.** Observable weekly feedback achieves higher
canonical ReT than the complete static frontier under identical physical
rights.

**H3 — Endogenous review value.** A variable-review adaptive controller with
four review rights outperforms the strongest fixed-cadence adaptive controller
with the same four rights. The unconstrained weekly perfect-information oracle
is an upper ceiling, not a comparator that the restricted policy can logically
dominate.

**H4 — Hybrid incremental value.** The complete learning-augmented controller
outperforms strengthened MPC and pure recurrent RL while satisfying frozen
worst-product, lost-demand, resource, review, and computation guardrails.

Successive-disruption learning curves, volatility reduction, path dependency,
and retained (L_{e-1}) are reserved for Paper 3. They are not inferred from
within-episode recurrence.

## 3. System and decision problem

### 3.1 DES provenance and claim boundary

The model preserves the thesis-grounded operation sequence, order flow,
production and transport timing, disruption interfaces, and canonical ReT
continuity metric. The two-product layer is a disclosed researcher extension
motivated by the aggregation of multiple ration classes into one homogeneous
product in the thesis. Products are nonfungible in the primary cell and share
the same aggregate processing and transport rights.

### 3.2 Decisions and information

Each eight-week episode contains 24 fixed production slots. At a weekly
commitment, action k_t ∈ {0, 1, 2, 3} assigns k_t slots to product C and
3-k_t to product H. The event-triggered action is

a_t = (k_t, d_t), with d_t ∈ {1, 2, 4} weeks,

with four review rights. Between reviews the selected mix is held; hidden
weekly changes are prohibited.

Observable history includes product-specific inventory, WIP, locked pipeline,
in-transit quantities, backlog quantities and ages, realized demand history,
previous actions, current disruption indicators, and remaining review rights.
True regimes, transition parameters, future demand or risk, tape identity,
random seeds, and oracle actions are excluded.

### 3.3 Outcomes and resources

The primary endpoint is order-level ReT computed with
`ret_excel_request_snapshot_v2`. Because the exact ordering of simultaneous
request and delivery callbacks remains historically ambiguous, both plausible
event orderings are frozen robustness analyses; neither is selected after
results. Secondary outcomes include quantity-weighted ReT, full-ledger ReT,
service-loss AUC, backlog, fill, cycle time, recovery time, CVaR10, lost demand,
and worst-product fill.

Production slots, production quantity, dispatch rights, vehicle-hours, demand,
and review rights are ledgered. Online inference and optimization time are
measured separately from offline training and simulator-data generation.

## 4. Learning-augmented event-triggered MPC

The proposed controller has four components.

1. A causal recurrent encoder maps observable history to a deployable belief
   representation. It is trained primarily for downstream planning regret,
   with belief likelihood as an auxiliary objective.
2. A distributional terminal model predicts continuation ReT quantiles beyond
   the finite MPC horizon and is trained with a proper quantile loss.
3. Constrained scenario MPC enumerates or optimizes feasible product-mix and
   review-time commitments under production, transport, lost-demand, and
   worst-product constraints.
4. A calibrated confidence gate accepts learned corrections only within
   calibration support and when predicted improvement and service-risk bounds
   pass. Otherwise it falls back to feasible robust MPC.

The gate is an abstention mechanism, not a deployment-safety certificate.
Episode-level service and tail behavior remain confirmatory outcomes.

## 5. Experimental design

### 5.1 U0: simulator-budgeted static search

Random search, cross-entropy search, Gaussian-process Bayesian optimization,
discrete-embedding CMA-ES, autoregressive policy gradient, and PPO receive the
same physical calendar-by-tape call budget. They query the simulator directly.
The 65,536-score matrix is inaccessible until every candidate and trace is
frozen, after which it supplies exact rank, regret, best-so-far area, and calls
to 5%, 1%, and 0.5% gaps. U0 is non-blocking and belongs in a compact main-text
result or supplement.

### 5.2 T0: strengthened comparator gate

Before fitting learned components, ReT-aligned MPC is evaluated at horizons 1,
3, 4, 6, and 8 with nominal, scenario, robust, and constraint-aware objectives,
and analytic or particle beliefs. A quality–p95-online-time frontier is frozen,
followed by one primary online budget. The reduced exact POMDP is used to expose
truncation effects: in the current benchmark, horizon-3 MPC already reaches the
exact belief-DP optimum. Therefore no neural premium can be claimed there.

Hybrid fitting requires a paired 95% lower confidence bound of at least 0.015
for the best observable policy over strengthened MPC, together with resource,
lost-order, and worst-product guardrails. Failure routes directly to the
Program-Q fallback manuscript.

### 5.3 U1/S1b: nonstationary envelope

The risk cascade is used only if strengthened MPC absorbs the current
opportunity. Garrido-native risks are screened first. A disclosed
researcher-defined envelope may add latent normal/surge/recovery regimes,
product–demand–risk interaction, and mean-preserving processing-time
stochasticity. Selection uses safe perfect-information headroom, classical
observable conversion, action-ranking reversals, and guardrail feasibility;
learner return is forbidden.

Action-dependent R14 production/quality physics runs in direct SimPy until its
exact multiplier and extreme-calendar family receive an independent
transducer certificate. Every candidate region has a stationary control with
the same total risk mass.

### 5.4 U2: timing oracle

The primary timing estimand compares variable and fixed review schedules under
the same four-review budget. The complete weekly calendar class is reported as
an upper bound. Promotion requires a paired lower confidence bound of 0.015.
Failure produces `STOP_NO_ENDOGENOUS_REVIEW_VALUE` and prohibits training an
event-triggered learner.

### 5.5 U3/U4: component attribution

Nested arms are strengthened MPC; learned belief plus MPC; learned terminal
value plus MPC; learned timing plus MPC; belief plus value; the complete hybrid;
and pure recurrent RL. Real history must beat shuffled, delayed, wrong-channel,
phase-only, modal, and frequency-matched replacements.

### 5.6 U5: virgin confirmation

Architecture, loss weights, online budget, margins, comparator set, simulator
tapes, and optimizer seeds are frozen before confirmation. The hierarchical
primary tests are hybrid minus best static (LCB at least 0.01), hybrid minus
best classical (LCB at least 0.01), and hybrid minus pure RL (LCB at least
zero). Additional requirements are at least 70% favorable tapes, at least 8 of
10 positive optimizer seeds, worst-product-fill LCB no lower than -0.02, no
increase in lost demand, exact resource and review equality, noncontradictory
CVaR10 direction, and successful trajectory and direct-DES audits.

## 6. Current evidence and prospective boundary

Completed evidence establishes the physical and methodological foundation, not
the hybrid claim. Program O found material perfect-information headroom. Program
Q confirmed that recurrent feedback exceeded every open-loop calendar and was
equivalent in ReT to the strongest frozen belief-state controller, but failed
its worst-product compound gate and showed no neural premium. U0, T0, S1b, and
U2 now have executable, fail-closed surfaces; no new scientific seeds and no
hybrid confirmation are authorized by those implementation artifacts.

The integrated hybrid results will be reported only if every preceding gate
passes. Otherwise the paper routes to the already replicated boundary result:
learned feedback can outperform exhaustive open-loop scheduling while matching,
but not improving upon, structured belief-MPC; worst-product safety remains
uncertified.

## 7. Managerial implications to be tested

The final decision aid will report: which product mix to commit under observable
inventory and backlog imbalance; when replanning is worth scarce managerial
attention; when a robust MPC is sufficient; when learned corrections are
accepted or rejected; and the amortization point at which offline learning is
repaid by lower online planning cost. A null hybrid premium is managerially
meaningful if structured control captures all deployable value.

## 8. Limitations

The study uses one thesis-grounded system plus controlled benchmark families;
the product extension is not an estimated description of every ration class.
ReT timestamp semantics retain a frozen robustness ambiguity. Scenario models
cannot certify all possible nonstationarity. The controller is centralized,
and “best” means best among preregistered tested methods—not globally optimal in
the full DES POMDP. Organizational learning and cross-campaign retention are
outside this paper.

## 9. Conclusion

This study treats learning as an incremental decision mechanism, not a label
attached to a DES. Exact static benchmarks, strengthened structured control,
matched review rights, and fail-closed pre-learner gates determine whether a
hybrid is warranted. The design can therefore produce either a positive hybrid
result or a scientifically useful boundary showing that model-based feedback is
already sufficient.

## References (working)

- Garrido-Ríos, J. G. (2017). A Mixed-Method Study on the Effectiveness of a Buffering Strategy in the Relationship between Risks and Resilience. Doctoral thesis.
- Garrido, J. G., Pongutá, W., and Adarme, W. (2024). Enhancing the Operationalization of SCRES-Based Simulation Models with AI Algorithms.
- The final submission bibliography will include primary sources for belief-state control, scenario MPC, distributional RL, event-triggered control, simulation optimization, and DES–ML integration. References will be frozen before journal submission rather than inferred from repository notes.
