# 2. Methods

## 2.1 Outer-loop state

For configuration (x_k), context (c_k), and exogenous DES realization (omega_k),

\[
Y_k=DES(x_k,c_k,\omega_k),\qquad
L_k=U(L_{k-1},x_k,Y_k),\qquad
x_{k+1}=\pi(L_k,c_{k+1}).
\]

The DES state is reset between replications. (L_k) is the retained state of the search procedure;
it is not inventory, disruption memory, or a policy state acting inside an episode. A memoryless twin
reinitializes (L) under the same family and seed vector.

## 2.2 Estimands and comparisons

The primary endpoint is normalized AUC regret, where lower is better. RQ1 uses positional paired
replicates within each family and the 12-seed replay block declared in the retention artifact. The
six contrasts are reported as development/replay evidence, not as a new confirmation.

RQ2 uses the 288-to-4,608 expansion and compares retained transfer, cold start, and state-blind
replay of each method's own marginal visit distribution. The state-blind comparator preserves
marginal sampling frequencies while removing retained ordering structure. The confirmation is
limited to the reserved block, frozen caches, inherited demand process, and the declared source
manifests.

RQ3 has three separate estimands: the algebraic Figure 5 identity, matched-parameter KAN/MLP search,
and the five-seed KAN latent contract. Fit numbers from other contracts are not combined with the
matched search bakeoff.

## 2.3 Physical panels and source boundary

The six thesis-derived panels are reported as a prospective reproduction of comparative directional
effects. They are not complete source-model validation, Simulink reproduction, or order-level
behavioural replication.
The source's unresolved `sumBt` interpretation remains a limitation. Non-anticipative normalization
is an integrity control: it is fitted on the declared prefix and never sees the unrun surface.

## 2.4 Demand-process sensitivity

The inherited process is the primary contract. The source-faithful seasonal mode treats Garrido's
`GR` as a trajectory generator/input. The researcher-defined Holt-Winters mode adds an observable
seasonal signal and is to be tested against next-period realised demand, naive and seasonal-naive
baselines, RMSE, bias, lagged cross-correlation and a shuffled placebo. This sensitivity is
`DEVELOPMENT_SENSITIVITY`, has a 72-hour cap, cannot regrade RQ2, and does not authorize retuning or
new confirmation seeds.

## 2.5 Custody and release

Every cited result resolves through `papers/paper2/claim_lock.json` to its artifact, hashes, claim
status, evidence grade, scope, endpoint and contract. The v2 provenance certificate separately
adjudicates historical source recovery and current-head behavioural equivalence. The independent
stdlib verifier does not import the simulation module whose drift it checks.
