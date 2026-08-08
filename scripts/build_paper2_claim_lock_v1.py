#!/usr/bin/env python3
"""One table the manuscript cites, instead of four documents nobody can hold in their head.

WHY THIS EXISTS. Authority over what may be written currently lives in a base claims table plus four
amendments, and resolving any single number means remembering which later document invalidated which
earlier sentence. That is exactly the mechanism that produced the `ofat_transfer` fossil: two sealed
artifacts give opposite-signed bounds on byte-identical replicates, a reconciliation measured the
bound positive in only 65% of 40 resampling seeds, Amendment 1 forbade the phrase "excludes zero" --
and the phrase kept circulating anyway, because forbidding it lived in a fifth place.

So: one row per claim the manuscript actually cites. Ten or fifteen rows, not the 216 of the
registry. The figure builder and the prose reference `claim_id`, never a loose path.

BOTH HASHES, ALWAYS. `self_sha256` is the digest of the sealed payload computed BEFORE its own seal
is inserted; `file_sha256` is the digest of the bytes on disk. They answer different questions --
did the payload change? is this the file I was handed? -- and they never coincide. A design review
of this repository once cited file digests under the name `self_sha256` for sixteen artifacts: every
value right, every label wrong. Hence both, each under its own name.

GRADE IS READ, NOT ASSUMED. Amendment 3 recorded that `garrido_h2_h3_confirmation_v1/result.json`
carries no `run_role`, `scope`, `claim_status` or `self_sha256` at top level -- its confirmation
status lives in a sibling receipt. A census that greps `run_role` loses it; one working from memory
loses the GSA row instead. So a missing field is reported as GRADE_NOT_MACHINE_DISCOVERABLE with
the sibling that carries it, never silently dropped.

Development tooling. Reads artifacts, writes one JSON. No seeds, no science.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("papers/paper2/claim_lock.json")

#: Ownership and status ONLY. Renamed from PORTFOLIO_CLAIM_LOCK because the old name invited exactly
#: what `numbers_in()` below now forbids: a second file where a figure can live. Two copies of a
#: number stay equal only until one of them is edited.
PORTFOLIO_MAP = Path("papers/PORTFOLIO_MAP.json")

# One row per claim the manuscript cites. `allowed` and `forbidden` are the wording contract; every
# forbidden phrase here was actually written somewhere and had to be retracted.
CLAIMS: list[dict] = [
    # RQ2 IS THREE ROWS, NOT ONE. The preregistration names factorized UCB1 as the sole confirmatory
    # arm (lines 15, 40, 104: the other carriers "son secundarios y exploratorios ... no pueden
    # seleccionar otro ganador despues de abrir las semillas"). A single row asserting both the UCB1
    # result and the neural result at grade CONFIRMATORY over-graded the second one, which four
    # external audits caught. And a row stating only the within-family contrasts concealed the
    # absolute ranking, which the twelve-arm re-read caught. Both defects are fixed by separating
    # the claim from its grade: RQ2a is confirmatory, RQ2b is prespecified-secondary, RQ2c is
    # post-hoc on prospective data. RQ2a may not be cited without RQ2c on the same page.
    {
        "claim_id": "RQ2A_UCB1_TRANSFERS_BEYOND_ITS_OWN_MARGINAL_REPLAY",
        "artifact": "results/grid_transfer_confirmation_v2/result.json",
        "section": "RQ2a (the preregistered confirmation, leads Results)",
        "endpoint": "auc_regret_norm",
        "estimand": "UCB1 transfer vs cold start and vs a state-blind replay of its own marginals",
        "allowed": ("In a prospective expansion from 288 to 4,608 configurations, the preregistered "
                    "confirmatory arm -- factorized UCB search -- outperformed both cold start and a "
                    "state-blind replay of its own search marginals (+0.03073, LCB95 +0.01990, "
                    "n=60)."),
        "forbidden": ["only UCB1 learns", "UCB1 is universally superior", "factorized UCB policy",
                      "UCB1 was the best arm", "only UCB1 transferred"],
        "why_forbidden": ("'policy' would conflate the outer loop with within-episode control; and "
                          "'best arm' is false -- RQ2C ranks ucb1_transfer fourth of twelve and "
                          "indistinguishable from the three above it"),
        "must_be_cited_with": ["RQ2C_THE_TOP_FOUR_ARMS_ARE_INDISTINGUISHABLE",
                               "COMPARATOR_IS_NOT_FIXED_DURING_THE_RUN"],
    },
    {
        "claim_id": "RQ2B_SECONDARY_CARRIERS_SHOW_NO_SUCH_ADVANTAGE",
        "artifact": "results/grid_transfer_confirmation_v2/result.json",
        "section": "RQ2b (prespecified secondary, prospective)",
        "secondary_estimand_declared_in": (
            "docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md lines 15 and 39-41: "
            "UCB1 factor-wise is named the confirmatory arm, and neuron, OFAT and GP-EI are "
            "'secundarios y exploratorios en esta confirmacion'"),
        "endpoint": "auc_regret_norm",
        "estimand": "each secondary carrier's transfer vs its own state-blind marginal replay",
        "allowed": ("Prespecified secondary analyses found no corresponding advantage for the "
                    "evaluated neural, GP-EI or OFAT carriers over the full run; the neural "
                    "carrier's contrast averaged -0.01178 [-0.01849, -0.00484]. This average is not "
                    "stationary against that comparator: over the first twenty seeds the neural "
                    "contrast was +0.00032 and it reaches -0.02146 over the last twenty as the "
                    "histogram accumulates (rho = -0.378, permutation p = 0.003). The negative is "
                    "not an artifact of that drift: against a frozen ex-ante level prior, which "
                    "does not drift, the same carrier loses by MORE (-0.02204 [-0.02599, "
                    "-0.01829])."),
        "forbidden": ["the neuron has no memory", "neural carriers cannot transfer",
                      "this confirms the neural carrier fails", "a confirmatory negative",
                      "the neural carrier did not transfer"],
        "why_forbidden": ("the preregistration declares these arms secondary and exploratory, so a "
                          "prospective negative at secondary grade is not a confirmed negative -- "
                          "even though the repaired comparator makes the negative larger rather "
                          "than smaller"),
        "must_be_cited_with": ["COMPARATOR_IS_NOT_FIXED_DURING_THE_RUN",
                               "RQ2D_A_TRANSPORTABLE_PRIOR_BEATS_THREE_OF_FOUR_CARRIERS"],
    },
    {
        "claim_id": "RQ2C_THE_TOP_FOUR_ARMS_ARE_INDISTINGUISHABLE",
        "artifact": "results/best_arm_reanalysis/result.json",
        "section": "RQ2c (post-hoc re-read of the same prospective artifact)",
        "endpoint": "auc_regret_norm",
        "estimand": "paired per-seed difference of every arm against the lowest-mean arm, n=60",
        "allowed": ("Re-read as twelve arms rather than four within-family contrasts, the four "
                    "cold-start arms occupy ranks 9-12 without exception, the four lowest-regret "
                    "arms are mutually indistinguishable -- all six pairwise contrasts among them "
                    "straddle zero and none is rejected under Holm -- and three of those four are "
                    "frequency-matched replays of a carrier's visit marginals; retention beat a "
                    "carrier's own marginal replay in one of four families and lost distinguishably "
                    "in three."),
        "forbidden": ["the state-blind control wins", "marginal replay is the best procedure",
                      "retention does not help", "this was preregistered",
                      "the ranking selects a new winner",
                      "the top four are indistinguishable from the best arm",
                      "three of the four discard the carrier entirely",
                      "the transferable object is a level-frequency prior",
                      "a visit histogram is enough", "state-blind marginal replay"],
        "must_be_cited_with": ["COMPARATOR_IS_NOT_FIXED_DURING_THE_RUN"],
        "why_forbidden": ("the three paired contrasts against the incumbent all cross zero, so "
                          "reading the mean ranking as a verdict is the same defect ENMIENDA_1 "
                          "forbids for ofat; and the preregistration forbids selecting a different "
                          "winner after the seeds were opened -- this row reports a tie, not a "
                          "winner"),
    },
    {
        "claim_id": "RQ2D_A_TRANSPORTABLE_PRIOR_BEATS_THREE_OF_FOUR_CARRIERS",
        "artifact": "results/comparator_repair/result.json",
        "section": "RQ2d (preregistered comparator repair, replay on the burned block)",
        "endpoint": "auc_regret_norm",
        "estimand": "each carrier's transferred state against a frozen ex-ante level prior",
        "allowed": ("Against a level-frequency prior frozen during base-grid training -- "
                    "transportable, and deployable without running the carrier on the target case "
                    "-- factorized UCB search retains an advantage of +0.04179 [+0.03221, +0.05188], "
                    "larger than its advantage over the online comparator. The other three carriers "
                    "lose to that prior: the neural carrier by -0.02204 [-0.02599, -0.01829], GP-EI "
                    "by -0.02097 [-0.02872, -0.01294] and OFAT by -0.04284 [-0.04963, -0.03720]. So "
                    "a transportable visit prior is sufficient to replace the transferred state of "
                    "three of four carriers, and insufficient only for factorized UCB."),
        "forbidden": ["this confirms UCB1", "the frozen prior is the best comparator overall",
                      "a level-frequency prior is enough", "retained state is unnecessary",
                      "this raises the grade of RQ2a"],
        "why_forbidden": ("the seeds are burned and the run is a replay, so no grade improves; and "
                          "the sufficiency of a prior is carrier-specific -- true for three "
                          "carriers, false for the one the paper recommends"),
        "must_be_cited_with": ["RQ2A_UCB1_TRANSFERS_BEYOND_ITS_OWN_MARGINAL_REPLAY"],
    },
    {
        "claim_id": "RQ1_RETENTION_SIX_FAMILIES",
        "artifact": "results/retention_contrasts/result.json",
        "section": "RQ1 (development/replay reanalysis)",
        "endpoint": "auc_regret_norm",
        "estimand": "memoryless twin minus state-retaining twin, paired by seed within family",
        # The six-family sentence was point-estimate-only until simultaneous inference closed. It
        # closed: `retention_simultaneous` reports max-T intervals from ONE shared bootstrap index
        # matrix, all six above zero, Holm rejecting all six, and the bound's sign stable in 40 of 40
        # resampling seeds -- so the strong wording is earned rather than assumed. The per-family
        # numbers below stay in this row; anything JOINT is cited from RQ1_SIMULTANEOUS.
        "allowed": ("In a development reanalysis of a sealed replay, state retention lowered AUC "
                    "regret in all six matched families: neuron +0.06070 "
                    "[+0.04568, +0.07953], UCB1 +0.05153 [+0.03583, +0.06593], OFAT +0.03750 "
                    "[+0.02920, +0.04675], KG +0.03461 [+0.02610, +0.04315], GP-EI +0.02271 "
                    "[+0.01276, +0.03410], and Thompson +0.01985 [+0.01022, +0.02956], "
                    "with n=12 paired seeds per family."),
        "forbidden": ["retention was prospectively confirmed", "six families prove causality",
                      "excludes zero", "the six leading ranks establish retention"],
        "must_be_cited_with": ["RQ1_SIMULTANEOUS_AND_THE_DEPLOYED_ENDPOINT"],
        "why_forbidden": ("the analysis is a sealed-tape reanalysis with no new seeds or "
                          "adjudication; its estimand is within-family AUC, not a prospective "
                          "confirmation or a context-specific effect"),
    },
    {
        "claim_id": "COMPARATOR_IS_NOT_FIXED_DURING_THE_RUN",
        "artifact": "results/comparator_drift/result.json",
        "section": "RQ2 limitation, and Methods where the comparator is defined",
        "endpoint": "auc_regret_norm",
        "estimand": "per-seed contrast and per-arm regret regressed on run order",
        "allow_failed_falsifiers": True,
        "allowed": ("The marginal-replay comparator is built from a visit histogram that is created "
                    "once and updated throughout the run, including with the transferred arm's "
                    "visits on the case being scored. It is therefore not carrier-independent and "
                    "not fixed: at the first evaluation it holds 24 real visits against 4,608 "
                    "pseudocounts and samples almost uniformly, and by the last it holds 8,640, 65% "
                    "of the mass. The four marginal arms are the four most strongly "
                    "negatively-correlated with run order of all twelve, ahead of every cold and "
                    "transfer arm, resolved at p < 0.05 in three of four; no cold arm drifts in any "
                    "family. The current case contributes between 0.52% and 0.18% of the histogram "
                    "mass. Declared falsifier f2, which asked the same question of the contrast "
                    "rather than of the comparator, required resolution in at least three of four "
                    "families and returned two; it is reported failed and its bar was not moved. "
                    "The preregistered repair separates the two defects: removing the current case "
                    "leaves the drift intact (the causal-prefix comparator still drifts in 4 of 4 "
                    "families, rho -0.167 to -0.366) and moves each contrast by about 0.001, while "
                    "a prior frozen before the target grid is touched does not drift at all (rho "
                    "-0.070 to +0.148). Accumulation across cases was the whole effect; current-case "
                    "contamination was bounded and negligible, with observed total variation "
                    "0.005160 against a derived ceiling of 0.005181 over 1,440 measurements."),
        "forbidden": ["state-blind marginal replay", "the comparator is a fixed control",
                      "an ex-ante transportable prior", "the histogram can be deployed alone",
                      "this invalidates the confirmation"],
        "why_forbidden": ("the accurate name is a carrier-state-blind, sequence-blind ONLINE "
                          "frequency replay; and the UCB1 contrast stays positive across the whole "
                          "run including its last window, so the confirmation is qualified rather "
                          "than withdrawn"),
    },
    {
        "claim_id": "RQ1_SIMULTANEOUS_AND_THE_DEPLOYED_ENDPOINT",
        "artifact": "results/retention_simultaneous/result.json",
        "section": "RQ1 (joint inference) and Limitations (endpoint sensitivity)",
        "endpoint": "auc_regret_norm (primary) and final simple regret at budget 24 (secondary)",
        "estimand": "the six paired within-family contrasts treated as one inferential family",
        "allowed": ("Treated as one inferential family with a shared bootstrap index matrix, all six "
                    "contrasts survive max-T simultaneous intervals and Holm on the preregistered "
                    "AUC endpoint (simultaneous critical value 2.591 against a 1.906 marginal "
                    "reference), and the sign of each lower bound is stable across 40 resampling "
                    "seeds. On the simple regret of the recommendation actually deployed at budget "
                    "24, all six point estimates keep the same sign but the joint resolution "
                    "collapses: one of six retains a simultaneous lower bound above zero, and the "
                    "family ordering is not preserved between the two endpoints."),
        "forbidden": ["retention was prospectively confirmed", "the endpoints agree",
                      "AUC and final regret give the same ranking",
                      "retention fails on the deployed endpoint"],
        "why_forbidden": ("the direction holds in all six under BOTH endpoints -- what changes is "
                          "resolution, not sign, and reporting either half alone misleads in "
                          "opposite directions"),
    },
    {
        "claim_id": "H_REGIME_MUST_BE_LABELLED_BY_METRIC",
        "artifact": "results/h_regime_crosswalk/result.json",
        "section": "Results 3.4 -- H_regime cited with its metric, or not at all",
        "endpoint": "H_regime and its transform-invariant ordinal companions",
        "allow_failed_falsifiers": True,
        "estimand": "the same statistic on two metrics, two grids, and under monotone rescaling",
        "allowed": ("H_regime names one statistic computed on two different metrics, and each "
                    "citation must say which. On the ret_excel_risk_conditional surface of the 288 "
                    "grid at twelve seeds it is 0.003802, reproduced here to the last digit against "
                    "surface_gates_v2; on the Cobb-Douglas index reconstructed from aggregates.json "
                    "at six seeds it is 0.0 with a universal argmax. The ret_excel figure is not "
                    "scale-invariant: a strictly increasing rescaling that leaves every ordering "
                    "untouched moves it to 0.010776, and 0.028294 to 0.067539 on the extended grid. "
                    "What is invariant is ordinal -- mean pairwise rank correlation +0.844 and "
                    "+0.909, top-25 overlap 91.7% and 23.5%."),
        "forbidden": ["H_regime = 0.0038", "no context-conditioned architecture can pay",
                      "the 288 zero is transform-proof",
                      "a single configuration is optimal in every context",
                      "H_regime is 0.0038 against a bar of 0.05"],
        "why_forbidden": ("an unlabelled figure conflates two metrics; the transform-proof zero and "
                          "the universal argmax belong to the Cobb-Douglas surface and may not be "
                          "transferred to the one the manuscript cites; and no bar comparison "
                          "survives a statistic that moves under a rescaling that preserves every "
                          "ordering"),
    },
    {
        "claim_id": "EXPANSION_ADDS_REACHABLE_OPTIMA",
        "artifact": "results/expansion_difficulty/result.json",
        "section": "RQ2 methods (why the expanded grid is the right test)",
        "endpoint": "expected simple regret of a uniform 24-draw, by order statistics",
        "estimand": "E[best of a uniform n-subset] on 288 and on 4,608 of the same sealed surfaces",
        "allowed": ("The expansion from 288 to 4,608 configurations adds reachable improvement "
                    "rather than diluting a uniform starting policy: the extended optimum is "
                    "strictly better in 136 of 360 cells, 0.97% of the 4,320 added configurations "
                    "exceed the base optimum, and the expected simple regret of a uniform 24-draw "
                    "falls from 0.07429 to 0.06755 against each grid's own optimum and from 0.10284 "
                    "to 0.06755 against a common reference. Computed exactly by order statistics "
                    "and checked against a 20,000-draw Monte Carlo control."),
        "forbidden": ["the expansion dilutes cold start",
                      "the new factors move the endpoint 18x less than the contrasts",
                      "the expansion hardens the problem",
                      "the expanded grid is a harder benchmark"],
        "why_forbidden": ("the dilution claim is measured false under both normalisers; the 18x "
                          "ratio has no single value (7.0x, 30.5x and 67.5x for the three canonical "
                          "choices, and 18x is none of them); and 'harder' is not established "
                          "either -- uniform search does BETTER on the larger grid"),
    },
    {
        "claim_id": "V0_HYPOTHESES_ADJUDICATED",
        "artifact": "results/v0_adjudication_matrix/result.json",
        "section": "Appendix A (reconciliation with the v0 hypotheses)",
        "endpoint": "not applicable -- adjudicates other artifacts' endpoints",
        "estimand": "the distance between each v0 sentence and the estimand actually measured",
        "allowed": ("Of the four v0 hypotheses, three are supported on development evidence and "
                    "none is confirmatory: H1 on a redefined endpoint (restricted TTR, +125.99 h "
                    "[+98.35, +154.54]) whose companion recovery gate returned no learning "
                    "headroom, H2 on a context ordinal (+0.04220 [+0.03466, +0.04992]) rather than "
                    "successive within-run disruptions, H4 for search cost rather than delivered "
                    "resilience. H3 is not supported with a live estimand of the wrong sign "
                    "(-1.109e15, p = 0.821). Of the two research questions, dynamic "
                    "operationalization is answered only for an outer loop and predictive accuracy "
                    "is not answered at all."),
        "forbidden": ["we answered the v0 questions", "three of four hypotheses hold",
                      "resilience is learning-dependent", "R_t = f(S_t, D_t, L_{t-1})",
                      "H1 is unanswerable as written", "the hypotheses were confirmed"],
        "why_forbidden": ("each bare summary is true only with qualifiers that travel separately "
                          "from the sentence; and 'unanswerable as written' is now false -- "
                          "h1_h3_originales_v3 is a preregistered recovery-time comparison"),
    },
    {
        "claim_id": "VALIDATION_SIX_THESIS_PANELS",
        "artifact": "results/garrido_h2_h3_confirmation_v1/result.json",
        "sibling_receipt": "results/garrido_h2_h3_confirmation_v1/completion_receipt.json",
        "section": "Methods / targeted validation",
        "endpoint": "flow_fill_rate + delivered + full_ledger + unresolved (concordant)",
        "estimand": "directional buffer and shift response, six preregistered panels",
        "allowed": ("The reconstruction prospectively reproduced six thesis-derived comparative "
                    "panels, 12/12 tapes each under Holm correction, with generated_orders exactly "
                    "zero in every tape of every panel and lost orders falling in all six."),
        "forbidden": ["validation of the DES", "the DES is validated",
                      "order-level behavioural replication", "reproduces the Simulink model"],
        "why_forbidden": ("the artifact's own claim_boundary says it establishes no learner, "
                          "feedback or architectural value; and sumBt is unreconstructed above "
                          "1.09% of 47,780 rows"),
    },
    {
        "claim_id": "FIG5_IS_AN_IDENTITY",
        "artifact": "results/garrido_fig5_surrogate/result.json",
        "section": "Methods, as an algebraic proposition",
        "endpoint": "R2 of the drawn network on its own drivers", "estimand": "identity check",
        "allowed": ("Under the literal reading of Fig. 5, ReT is exactly the sum of the driver "
                    "contributions supplied as inputs (R2 = 1.0, maximum identity error "
                    "3.22e-15), so reconstructing ReT from them is an identity rather than a "
                    "non-trivial predictive task."),
        "forbidden": ["Garrido's neuron is absurd", "the proposal is wrong",
                      "the neuron cannot learn"],
        "why_forbidden": "it relocates where the learning is; it does not disparage the proposal",
    },
    {
        "claim_id": "LADDER_STATEFUL_ARMS_LEAD",
        "artifact": "results/search_ladder_v5/result.json",
        "section": "RQ1", "endpoint": "auc_regret_norm",
        "estimand": "ranking of 15 deployable methods at budget 24",
        "allowed": ("In a development replay, the six state-retaining arms occupied the six leading "
                    "positions of fifteen, and the same approximator fell from rank 2 to rank 12 "
                    "when its memory was removed."),
        "forbidden": ["retention was prospectively confirmed", "retention is confirmed across families",
                      "the top-six ranking shows retention causes"],
        "why_forbidden": ("burned tapes, no adjudication; and rank position is not a causal "
                          "estimand -- the within-family paired contrasts are"),
    },
    {
        "claim_id": "RETENTION_NEURON_VS_RESET",
        "artifact": "results/garrido_normaliser_audit_v3/result.json",
        "section": "RQ1", "endpoint": "auc_regret_norm (prefix normaliser)",
        "estimand": "memory minus reset, within the neural family",
        "allowed": ("Retaining state improved the same approximator by +0.06070 AUC "
                    "[LCB95 +0.04556] under a prefix normaliser blind to the unrun surface."),
        "forbidden": ["7.90 runs", "7.24 runs", "13.54", "12.42", "5.83 runs saved"],
        "why_forbidden": ("the runs-to-threshold figures are censored at 0.056/0.153/0.222/0.611 "
                          "per arm and the +7.90 variant used the oracle normaliser"),
    },
    {
        "claim_id": "OFAT_CONTRAST_IS_RESAMPLING_UNSTABLE",
        "artifact": "results/ofat_lcb_reconciliation/result.json",
        "section": "RQ1, stated as a limit",
        "endpoint": "auc_regret_norm", "estimand": "neuron_memory vs ofat_transfer",
        "allowed": ("Two sealed artifacts place the bound on opposite sides of zero over "
                    "byte-identical replicates, and the bound is positive in 65% of 40 resampling "
                    "seeds; the contrast is not distinguishable under the resampling procedure."),
        "forbidden": ["excludes zero", "excluye el cero", "the neuron beats OFAT"],
        "why_forbidden": "Amendment 1 section 3 forbids it; the fossil survived four documents",
    },
    {
        "claim_id": "KAN_FIT_DOES_NOT_CONVERT_TO_SEARCH",
        "artifact": "results/surrogate_architecture_bakeoff/result.json",
        "section": "RQ3b", "endpoint": "auc_regret_norm (lower is better)",
        "estimand": "kan minus parameter-matched mlp",
        "allowed": ("At matched parameter budgets (532 vs 529), the KAN search arm had higher AUC "
                    "regret than its matched MLP by +0.01037, CI95 [+0.00302, +0.01893], "
                    "p = 0.0012; this bakeoff adjudicates search, not supervised fit. The best "
                    "searcher of the seven-architecture bake-off was a five-parameter neuron."),
        "forbidden": ["KAN is worse", "KANs do not work", "neural networks are unnecessary"],
        "why_forbidden": "the claim is about fit-versus-search on this task, not about KANs at large",
    },
    {
        "claim_id": "KAN_LATENT_UNDERPERFORMS_UNDER_MATCHED_CONTRACT",
        "artifact": "results/dmlpa_kan_latent/result.json",
        "section": "Supplement (different contract from RQ3b)",
        "endpoint": "ret_mean_track_b_v1 (HIGHER is better)",
        "estimand": "kan latent minus mlp latent, paired over 5 seeds",
        "allowed": ("Under the parameter-matched latent contract tested, the KAN arm underperformed "
                    "its MLP counterpart: 97.58 against 98.44, a paired -0.862 [-1.605, -0.119] "
                    "over five seeds with four of five negative -- -0.88% relative, smaller than "
                    "the 0.76-1.06 within-seed evaluation SD, and at matched parameter count the "
                    "KAN affords hidden_dim=10 against the MLP's 152."),
        "forbidden": ["KAN hurts", "KAN is worse", "-0.86225"],
        "why_forbidden": ("the bare difference is uninterpretable without endpoint, orientation, "
                          "absolute means and the 15x width confound"),
    },
    {
        "claim_id": "NEURAL_PREMIUM_NEEDS_CURVATURE_ABOVE_NOISE",
        "artifact": "results/headroom/buffer_prediction_premium/result.json",
        "section": "Discussion", "endpoint": "held-out R2, seed-grouped CV",
        "estimand": "MLP and KAN minus linear",
        "allowed": ("On a deliberately curved surface the backpropagation MLP scored below a "
                    "straight line (-0.128 [-0.316, +0.060]) and neither approximator reached the "
                    "preregistered SESOI of 0.05, with in-situ curvature 0.0763 against 0.3174 of "
                    "unexplained episode-level variance."),
        "forbidden": ["networks never help", "curvature is always below noise"],
        "why_forbidden": "it is a condition, measured on one surface, not a law",
    },
    {
        "claim_id": "SURFACE_IS_MATERIALLY_NONLINEAR",
        "artifact": "results/functional_form_diagnostics/result.json",
        "section": "Supplement + RQ3 discussion",
        "endpoint": "held-out R2, folds cut on seed", "estimand": "quadratic+interactions minus linear",
        "allowed": ("Ramsey RESET rejects linearity in all six contexts (F 384-2463), and both AIC "
                    "and seed-grouped held-out R2 select quadratic-with-interactions in all six; "
                    "the non-linearity buys +0.19 to +0.23 of held-out R2 in the R1r family. The "
                    "architecture tie is therefore not explained by a linear surface."),
        "forbidden": ["the surface is linear", "AIC proves linearity",
                      "RESET shows the variables are linear"],
        "why_forbidden": ("AIC is a relative criterion and a non-significant RESET would have been "
                          "failure to reject, not evidence; here it rejects"),
    },
    {
        "claim_id": "DEMAND_PROCESS_SCOPE",
        "artifact": "results/demand_process/result.json",
        "section": "Results 3.1, scope stated up front",
        "endpoint": "weekly CV and lag-1 autocorrelation", "estimand": "realised demand process",
        "allowed": ("Within the thesis-inherited U(2400,2600) demand process: realised weekly CV "
                    "7.1%, 24.8% of weeks already exceed single-shift capacity, and lag-1 "
                    "autocorrelation -0.228 against an iid band of +/-0.065."),
        "forbidden": ["demand is static", "almost deterministic demand", "weekly CV 0.94%",
                      "demand is iid"],
        "why_forbidden": ("0.94% was hand-derived and wrong; and the series is not iid, which "
                          "retracts the 'no demand state to condition on' argument"),
        "allow_failed_falsifiers": True,
    },
    {
        "claim_id": "DEMAND_PROCESS_LITERAL_GENERATOR",
        "artifact": "results/demand_seasonal_engine/result.json",
        "section": "Results 3.1, bounded demand-process sensitivity",
        "endpoint": "weekly CV, seasonal-lag ACF, sampler moments, forecast skill at t+1",
        "estimand": "structure of the researcher-implemented Garrido-style trajectory generator",
        "allow_failed_falsifiers": True,
        "allowed": ("The source-faithful generator produces weekly CV 0.1775 and lag-12 ACF 0.839 "
                    "against 0.0713 and 0.015 for the thesis-uniform control, and the alpha/gamma "
                    "sampler covers the unit interval over 2,000 instrument draws (0.504 +/- 0.290 "
                    "and 0.497 +/- 0.288). The researcher-defined Holt-Winters observable signal is "
                    "NOT established as informative: it correlates 0.826 with realised demand at "
                    "t+1 against 0.006 for a shuffled placebo and reaches MASE 0.367 against a "
                    "naive baseline, but MASE 4.344 against a SEASONAL-naive one -- so it is "
                    "roughly four times worse than repeating the previous season. The engine "
                    "remains ENGINE_PARTIAL and no forecast-information claim is made."),
        "forbidden": ["Garrido's equation is a forecast validated against realised demand",
                      "forecast skill is established", "the forecast is informative",
                      "the observable signal beats its baselines",
                      "alpha and gamma are validated by 12 episodes",
                      "seasonal demand is confirmed"],
        "why_forbidden": ("a 0.826 correlation is nearly free when the series carries a strong "
                          "seasonal component -- the trivial seasonal-naive baseline beats this "
                          "instrument by 4.3x, which is exactly what the amended g5 exists to "
                          "catch and what the previous correlation-only gate would have passed"),
    },
    {
        "claim_id": "RISK_PROFILE_TAILORING_BUYS_NOTHING",
        "artifact": "results/garrido_risk_headroom_sensitivity_v1/result.json",
        "section": "Discussion / answer to the randomised-R2 request",
        "endpoint": "H_profile_safe", "estimand": "value of tailoring the posture to the risk profile",
        "allowed": ("Across 45 risk profiles escalating the thesis's own recurrent risks one at a "
                    "time (4,860 evaluations), no door passes: max H_profile_safe 6.93e-05 "
                    "[0, 2.08e-04] against a 0.01 bar. The optimum varies across at most three "
                    "postures; following it buys nothing."),
        "forbidden": ["the optimal posture is invariant across all 45 profiles",
                      "the optimum does not move", "randomised R2 is answered"],
        "why_forbidden": ("unique_profile_optima is 1, 2 or 3 by row; and the screen varied "
                          "PROFILES, not the within-episode realisation"),
    },
]


def digests(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "file_sha256": None, "self_sha256": None}
    payload = json.loads(path.read_text())
    return {
        "exists": True,
        "file_sha256": sha256(path.read_bytes()).hexdigest(),
        "self_sha256": payload.get("self_sha256"),
        "claim_status": payload.get("claim_status"),
        "scope": payload.get("scope"),
        "run_role": payload.get("run_role"),
        "registration_status": payload.get("registration_status"),
        "contract_sha256": payload.get("contract_sha256"),
        "falsifiers_all_passed": payload.get("falsifiers", {}).get("all_passed"),
    }


def grade(meta: dict, row: dict) -> tuple[str, list[str]]:
    """Read the grade; never assume it. Missing fields are named, not filled in.

    WHY `registration_status` IS CONSULTED FIRST. `scope` is free text that describes what the run
    DID, and a post-hoc re-read of a confirmation legitimately says so: `best_arm_reanalysis` carries
    `REREAD_OF_ONE_SEALED_CONFIRMATION_NO_SEEDS_NO_NEW_RUN`. A substring test on that string returns
    CONFIRMATORY for an analysis that was never preregistered -- the exact over-grading four external
    audits flagged in the manuscript, manufactured here by the instrument that is supposed to prevent
    it. So an artifact that declares itself unregistered can never be graded confirmatory, whatever
    else its prose contains, and `run_role` (a controlled field) outranks `scope` (a descriptive one).
    """
    missing = [k for k in ("run_role", "scope", "claim_status", "self_sha256") if not meta.get(k)]
    if missing and row.get("sibling_receipt"):
        return "GRADE_IN_SIBLING_RECEIPT", missing
    if missing:
        return "GRADE_NOT_MACHINE_DISCOVERABLE", missing
    rr, sc = str(meta["run_role"]), str(meta["scope"])
    reg = str(meta.get("registration_status") or "")
    if "NOT_PREREGISTERED" in reg or "POST_HOC" in reg or "POST_HOC" in rr:
        # The tapes may be prospective; the analysis is not. Both halves go in the grade.
        return ("POST_HOC_ON_PROSPECTIVE_DATA" if "CONFIRMATION" in sc
                else "POST_HOC_ON_DEVELOPMENT_DATA"), []
    if "CONFIRMATION" in rr:
        # A confirmatory RUN can carry estimands the preregistration declared secondary. The grade
        # belongs to the (artifact, estimand) pair, and the artifact cannot express which of its
        # estimands is which -- so the row declares it, and must cite the lines that say so. The
        # field name keeps the distinction visible: this grade was declared, not discovered.
        if row.get("secondary_estimand_declared_in"):
            return "PRESPECIFIED_SECONDARY_IN_A_CONFIRMATORY_RUN", []
        return "CONFIRMATORY", []
    if "REPLAY" in rr or "REPLAY" in sc:
        return "REPLAY", []
    if "DIAGNOSTIC" in rr:
        return "DIAGNOSTIC", []
    if "CONFIRMATION" in sc:
        return "CONFIRMATORY", []
    return "DEVELOPMENT", []


def numbers_in(node, path: str = "") -> list[str]:
    """Every numeric leaf in the portfolio map, by path. Booleans are not numbers here.

    The rule this enforces: ownership lives in one file, figures live in another, and the build
    fails rather than letting a number appear in both. `page_budget` is the sole allowance, because
    it is a policy limit and not a result.
    """
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            hits += numbers_in(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits += numbers_in(v, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if not path.endswith("page_budget"):
            hits.append(path)
    return hits


def validate_grader() -> dict:
    """Run the grader against the case that fooled it, and against a control that must still pass.

    A guard nobody watches fail is not a guard. The first case is the real string from
    `best_arm_reanalysis`; before the fix it returned CONFIRMATORY. The second is the real string
    from `grid_transfer_confirmation_v2`, which must keep grading CONFIRMATORY -- otherwise the fix
    would have bought correctness by refusing to grade anything, which is not correctness.
    """
    full = {"claim_status": "X", "self_sha256": "X", "scope": "", "run_role": ""}
    trap = grade({**full, "scope": "REREAD_OF_ONE_SEALED_CONFIRMATION_NO_SEEDS_NO_NEW_RUN",
                  "run_role": "REREAD", "registration_status":
                  "POST_HOC_REREAD_PROMPTED_BY_EXTERNAL_REVIEW_NOT_PREREGISTERED"}, {})[0]
    ctrl = grade({**full, "scope": "CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK_NO_RL_NO_NEURAL_LEARNER",
                  "run_role": "CONFIRMATION"}, {})[0]
    return {
        "post_hoc_reread_is_not_graded_confirmatory": {
            "passed": trap != "CONFIRMATORY", "returned": trap},
        "a_real_confirmation_still_grades_confirmatory": {
            "passed": ctrl == "CONFIRMATORY", "returned": ctrl},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows, problems = [], []
    if PORTFOLIO_MAP.exists():
        stray = numbers_in(json.loads(PORTFOLIO_MAP.read_text()))
        if stray:
            problems.append(f"PORTFOLIO_MAP carries numeric values at {stray} -- ownership only")
    else:
        problems.append(f"{PORTFOLIO_MAP} is missing; claim ownership is undeclared")
    grader = validate_grader()
    for name, chk in grader.items():
        if not chk["passed"]:
            problems.append(f"GRADER: {name} returned {chk['returned']!r}")
    for c in CLAIMS:
        p = Path(c["artifact"])
        meta = digests(p)
        g, missing = grade(meta, c)
        if not meta["exists"]:
            problems.append(f"{c['claim_id']}: artifact missing at {p}")
        if meta.get("falsifiers_all_passed") is False and not c.get("allow_failed_falsifiers"):
            problems.append(f"{c['claim_id']}: falsifiers did not all pass -- cite WITH the failure")
        row = dict(c)
        row.update({"paper_id": "P2", "evidence_grade": g,
                    "missing_top_level_fields": missing, **meta})
        if c.get("sibling_receipt"):
            row["sibling_receipt_file_sha256"] = digests(Path(c["sibling_receipt"]))["file_sha256"]
        rows.append(row)

    payload = {
        "schema_version": "paper2_claim_lock_v1",
        "portfolio_map": str(PORTFOLIO_MAP),
        "portfolio_map_sha256": sha256(PORTFOLIO_MAP.read_bytes()).hexdigest()
        if PORTFOLIO_MAP.exists() else None,
        "generated_at_commit": commit,
        "n_claims": len(rows),
        "supersedes_for_citation": [
            "docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md",
            "docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md",
            "docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_1.md",
            "docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_2.md",
            "docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_3.md",
            "docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_4.md",
        ],
        "supersession_note": ("those documents remain the adjudication record and are NOT retired; "
                              "this file is the single place the manuscript and figure builder "
                              "resolve a citation, so authority no longer depends on remembering "
                              "which amendment came last"),
        "problems": problems,
        "grader_self_test": grader,
        "claims": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print(f"{'claim_id':48}{'grado':26}{'fals':6} artefacto")
    for r in rows:
        f = {True: "ok", False: "RED", None: "-"}[r.get("falsifiers_all_passed")]
        print(f"{r['claim_id']:48}{r['evidence_grade']:26}{f:6} {'OK' if r['exists'] else 'FALTA'}")
    if problems:
        print("\nPROBLEMAS:")
        for p in problems:
            print("  -", p)
    print(f"\n  -> {args.output}  ({len(rows)} claims)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
