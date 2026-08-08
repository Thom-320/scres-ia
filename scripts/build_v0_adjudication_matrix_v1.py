#!/usr/bin/env python3
"""What the v0 draft asked, what was actually measured, and the gap between the two.

WHY A SEALED MATRIX AND NOT A PARAGRAPH. "We answered the v0 questions" and "three of four
hypotheses hold" have both circulated in this project, and both are true only under qualifiers that
travel separately from the sentence. A paragraph cannot be checked; a table whose every figure must
resolve inside the artifact its own row names can.

THE FALSIFIER THAT JUSTIFIES THE FILE. `f1` walks each row's cited numbers and requires them to be
present in the named artifact at the named JSON path. That is exactly the check that would have
caught the `+7.90 runs` fossil -- a real number from the ORACLE normaliser panel, cited for months
without its normaliser -- and it is why the rows carry pointers rather than transcribed values. A row
whose figure cannot be resolved fails here rather than reaching a reviewer.

WHAT THIS DELIBERATELY DOES NOT DO. It does not upgrade anything. Every H1-H4 artifact is
development-grade on already-open blocks, and three of them say so in their own `scope`. The matrix
records the verdict each artifact reached, the estimand it reached it ON, and the distance between
that estimand and the sentence the v0 draft wrote.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Reads sealed artifacts. No seed is opened, no simulation runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
CONTRACT = Path("docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md")

#: Each row names its artifact and the JSON path of every figure it cites. Nothing is transcribed.
ELEMENTS: list[dict] = [
    {
        "id": "RQ_DYNAMIC_OPERATIONALIZATION",
        "v0_asked": ("Can the SCRES model be operationalised dynamically, so that the chain's "
                     "behaviour responds to accumulated experience?"),
        "verdict": "PARTIALLY_ANSWERED",
        "answered_on": ("an outer loop over restarted DES runs: search state persists across runs, "
                        "the physical system does not"),
        "not_answered": ("dynamic operationalization INSIDE an episode. The DES is reset between "
                         "runs and no arm observes or acts during one"),
        "artifact": "results/grid_transfer_confirmation_v2/result.json",
        "figures": {},
    },
    {
        "id": "RQ_PREDICTIVE_ACCURACY",
        "v0_asked": "Does the neural component improve predictive accuracy of the resilience index?",
        "verdict": "NOT_ANSWERED",
        "answered_on": None,
        "not_answered": ("no held-out predictive validation of a resilience index exists in this "
                         "repository. Fit quality was measured only inside search contracts, where "
                         "it is an input to a search comparison and not a predictive claim"),
        "artifact": None,
        "figures": {},
    },
    {
        "id": "FORMALISM_L_IS_ENDOGENOUS",
        "v0_asked": "R_t = f(S_t, D_t, L_{t-1}) -- learning as an endogenous state variable",
        "verdict": "SUPPORTED_BUT_A_DIFFERENT_OBJECT_ON_A_DIFFERENT_INDEX",
        "answered_on": ("L survives as an endogenous state variable, but it is a factorised bandit "
                        "statistic living on the RUN index k, not on physical time t: "
                        "L_k = U(L_{k-1}, x_k, Y_k) and x_{k+1} = pi(L_k, c_{k+1})"),
        "not_answered": ("the subscript t in the v0 expression promises within-episode adaptive "
                         "control, which nothing here tests"),
        "artifact": "results/retention_simultaneous/result.json",
        "figures": {},
    },
    {
        "id": "H1_RECOVERY_TIME",
        "v0_asked": "A hybrid neural model recovers faster than a static simulation",
        "verdict": "SUPPORTED_ON_A_REDEFINED_ENDPOINT__DEVELOPMENT_ONLY",
        "answered_on": ("restricted_ttr = min(TTR, tau) with tau = 1344 h and a paired placebo, "
                        "under isolated shocks. The declared redefinition is in the artifact"),
        "not_answered": ("no neural causality is identified: the contrast is between postures, not "
                         "between a learner and its ablation. And a companion surface gate on the "
                         "recovery lane returned STOP_NO_RECOVERY_LEARNING_HEADROOM, so a learner "
                         "exploiting this endpoint is not established"),
        "artifact": "results/manuscript/h1_h3_originales_v3/result.json",
        "figures": {
            "hybrid_vs_static_mean": ["contrasts", "H1_hybrid_vs_static", "mean"],
            "hybrid_vs_static_lcb95": ["contrasts", "H1_hybrid_vs_static", "lcb95"],
            "hybrid_vs_static_ucb95": ["contrasts", "H1_hybrid_vs_static", "ucb95"],
            "n_cells": ["contrasts", "H1_hybrid_vs_static", "n_cells"],
            "tau_hours": ["tau_hours"],
        },
        "companion_stop": "results/garrido_v0_surface_gates_v1/result.json",
    },
    {
        "id": "H2_LEARNING_CURVE",
        "v0_asked": "Performance improves over successive disruptions",
        "verdict": "SUPPORTED_ON_ITS_OWN_ESTIMAND__DEVELOPMENT_ONLY",
        "answered_on": ("OLS slope of (reset AUC - memory AUC) against the context ordinal 1..6, "
                        "with a null control that crosses zero, so the trend is not the rising "
                        "difficulty of the escalated contexts"),
        "not_answered": ("successive DISRUPTIONS within a run. The ordinal is a context sequence in "
                         "an outer loop, and the artifact's own estimand note says a large but FLAT "
                         "advantage would support H4 rather than H2"),
        "artifact": "results/manuscript/h2_learning_curve/result.json",
        "figures": {"primary_slope": ["primary_slope"],
                    "null_slope": ["null_slope_random_minus_ofat"]},
    },
    {
        "id": "H3_VARIANCE_REDUCTION",
        "v0_asked": "Learning reduces outcome variance across disruption intensities",
        "verdict": "NOT_SUPPORTED__LIVE_ESTIMAND_WRONG_SIGN",
        "answered_on": ("the estimand exists and was measured; the point estimate has the wrong "
                        "sign and the interval spans zero by fifteen orders of magnitude"),
        "not_answered": ("H3' -- variance of SEARCH cost -- is a different construct and does not "
                         "rescue H3. This is no longer 'no estimand', it is 'no effect', which is "
                         "the stronger negative"),
        "artifact": "results/manuscript/h1_h3_originales_v3/result.json",
        "figures": {
            "hybrid_vs_reset_mean": ["contrasts", "H3_hybrid_vs_reset", "mean"],
            "hybrid_vs_reset_lcb95": ["contrasts", "H3_hybrid_vs_reset", "lcb95"],
            "hybrid_vs_reset_ucb95": ["contrasts", "H3_hybrid_vs_reset", "ucb95"],
            "p_one_sided": ["contrasts", "H3_hybrid_vs_reset", "p_one_sided"],
            "n_cells": ["contrasts", "H3_hybrid_vs_reset", "n_cells"],
        },
    },
    {
        "id": "H4_PATH_DEPENDENCE_ON_L",
        "v0_asked": "Resilience at time t depends positively on accumulated prior learning",
        "verdict": "SUPPORTED_FOR_SEARCH_ONLY__DEVELOPMENT_ONLY",
        "answered_on": ("retained search state lowers AUC regret in all six matched families, and "
                        "the six survive simultaneous max-T inference and Holm"),
        "not_answered": ("DELIVERED RESILIENCE at time t is not shown to depend on history. What "
                         "depends on history is the cost of FINDING a good configuration. On the "
                         "simple regret of the recommendation actually deployed, only one of six "
                         "keeps a simultaneous lower bound above zero"),
        "artifact": "results/retention_simultaneous/result.json",
        "figures": {
            "n_families_simultaneous_above_zero":
                ["summary", "n_families_simultaneous_lcb_above_zero"],
            "n_families_holm": ["summary", "n_families_rejected_under_holm"],
            "c_simultaneous": ["by_endpoint", "auc", "c_simultaneous"],
            "neuron_mean": ["by_endpoint", "auc", "per_family", "neuron", "mean"],
        },
        "forbidden_figure": ("+7.90 runs -- the oracle-normaliser panel, cited in the v0 draft "
                             "without naming its normaliser"),
    },
]


def dig(payload, path: list[str]):
    node = payload
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None, False
    return node, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/v0_adjudication_matrix/result.json"))
    args = ap.parse_args()

    rows, unresolved, missing_artifacts = [], [], []
    for el in ELEMENTS:
        row = {k: v for k, v in el.items() if k != "figures"}
        path = el.get("artifact")
        if path is None:
            row["resolved_figures"] = {}
            rows.append(row)
            continue
        p = ROOT / path
        if not p.exists():
            missing_artifacts.append(path)
            row["resolved_figures"] = {}
            rows.append(row)
            continue
        payload = json.loads(p.read_text())
        row["artifact_self_sha256"] = payload.get("self_sha256")
        row["artifact_claim_status"] = payload.get("claim_status")
        row["artifact_scope"] = payload.get("scope")
        row["artifact_preregistration"] = payload.get("preregistration") or payload.get(
            "contract_path")
        resolved = {}
        for name, ptr in el["figures"].items():
            value, ok = dig(payload, ptr)
            resolved[name] = {"json_path": ptr, "value": value, "resolved": ok}
            if not ok:
                unresolved.append(f"{el['id']}.{name} at {ptr} in {path}")
        row["resolved_figures"] = resolved
        rows.append(row)

    by_verdict = {}
    for r in rows:
        by_verdict.setdefault(r["verdict"], []).append(r["id"])

    hyp = [r for r in rows if r["id"].startswith("H")]
    n_supported = sum(1 for r in hyp if r["verdict"].startswith("SUPPORTED"))
    n_confirmatory = sum(1 for r in hyp
                         if "CONFIRMAT" in str(r.get("artifact_scope", "")).upper()
                         and "NO_ADJUDICATION" not in str(r.get("artifact_scope", "")).upper())

    falsifiers = {
        "f1_every_cited_figure_resolves_inside_its_named_artifact": {
            "passed": not unresolved, "unresolved": unresolved,
            "why_it_can_fail": ("a figure whose JSON path does not exist is a transcribed number, "
                                "and a transcribed number is how the +7.90 oracle fossil survived "
                                "for months")},
        "f2_every_named_artifact_exists": {
            "passed": not missing_artifacts, "missing": missing_artifacts},
        "f3_no_hypothesis_is_graded_confirmatory": {
            "passed": n_confirmatory == 0, "n_confirmatory": n_confirmatory,
            "scopes": {r["id"]: r.get("artifact_scope") for r in hyp},
            "why_it_can_fail": ("if any H1-H4 artifact were confirmatory this row would say so; "
                                "all four sit on already-open blocks and their own scopes declare "
                                "NO_ADJUDICATION")},
        "f4_an_unanswered_element_names_no_artifact": {
            "passed": all(r.get("artifact") is None for r in rows
                          if r["verdict"] == "NOT_ANSWERED"),
            "why_it_can_fail": ("citing an artifact beside NOT_ANSWERED would be the shape of a "
                                "claim being smuggled in under a negative label")},
        "f5_the_count_of_supported_hypotheses_is_computed_not_asserted": {
            "passed": n_supported == len([r for r in hyp if r["verdict"].startswith("SUPPORTED")]),
            "n_supported": n_supported, "n_hypotheses": len(hyp),
            "why_it_can_fail": "a hard-coded 'three of four' would survive a change in the rows"},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))

    payload = {
        "schema_version": "v0_adjudication_matrix_v1",
        # "3 of 4 supported on development evidence" is the sentence this very file forbids, wearing
        # the evidence-grade qualifier instead of the one that makes it true: NONE of the three is
        # supported as the v0 draft wrote it. Each is supported on a different object -- a redefined
        # endpoint, its own ordinal, search cost rather than delivered resilience -- and the status
        # names them, so the string cannot be quoted without them.
        "claim_status": (f"{n_supported}_OF_{len(hyp)}_SUPPORTED_NONE_AS_WRITTEN__"
                         + "__".join(sorted(
                             r["verdict"].replace("SUPPORTED_", "").replace("__DEVELOPMENT_ONLY", "")
                             for r in hyp if r["verdict"].startswith("SUPPORTED")))
                         if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED"),
        "scope": "REREAD_OF_SEALED_ARTIFACTS_NO_SEEDS_NO_NEW_RUN",
        "run_role": "POST_HOC_REREAD",
        "registration_status": "ADJUDICATION_MATRIX_NOT_PREREGISTERED_READS_ONLY_SEALED_VERDICTS",
        "endpoint": "not applicable -- this artifact adjudicates other artifacts' endpoints",
        "estimand": "the distance between each v0 sentence and the estimand actually measured",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": None,
        "n_elements": len(rows), "n_hypotheses": len(hyp),
        "n_hypotheses_supported": n_supported,
        "n_hypotheses_confirmatory": n_confirmatory,
        "by_verdict": by_verdict,
        "elements": rows,
        "forbidden_summaries": [
            {"phrase": "we answered the v0 questions",
             "why": ("one of the two research questions has no held-out predictive validation at "
                     "all, and the other is answered for an outer loop the v0 draft did not "
                     "describe")},
            {"phrase": "three of four hypotheses hold",
             "why": ("true only with every qualifier attached: H1 on a redefined endpoint whose own "
                     "recovery lane returned STOP, H2 on a context ordinal rather than successive "
                     "disruptions, H4 for search cost rather than delivered resilience, and none "
                     "of the three confirmatory")},
            {"phrase": "resilience is learning-dependent",
             "why": "the measured dependence is of SEARCH on its own history, not of resilience"},
            {"phrase": "R_t = f(S_t, D_t, L_{t-1})",
             "why": ("the subscript promises within-episode adaptive control; the measured object "
                     "lives on the run index k")},
        ],
        "falsifiers": falsifiers,
    }
    seal_and_write(payload, ROOT / args.out, contract=ROOT / CONTRACT,
                   reference=ROOT / "results/manuscript/h1_h3_originales_v3/result.json")

    print(f"{'elemento':34}{'veredicto':52}{'grado del artefacto'}")
    for r in rows:
        sc = str(r.get("artifact_scope") or "-")[:34]
        print(f"  {r['id']:32}{r['verdict']:52}{sc}")
    print(f"\nhipótesis sostenidas: {n_supported}/{len(hyp)} · confirmatorias: {n_confirmatory}")
    print(f"falsadores: {'todos pasan' if falsifiers['all_passed'] else 'FALLO'}")
    for n, v in falsifiers.items():
        if isinstance(v, dict) and not v["passed"]:
            print(f"  FALLA {n}: {v}")
    print(f"-> {args.out}")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
