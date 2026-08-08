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

# One row per claim the manuscript cites. `allowed` and `forbidden` are the wording contract; every
# forbidden phrase here was actually written somewhere and had to be retracted.
CLAIMS: list[dict] = [
    {
        "claim_id": "RQ2_UCB1_TRANSFERS_BEYOND_MARGINAL_REPLAY",
        "artifact": "results/grid_transfer_confirmation_v2/result.json",
        "section": "RQ2 (confirmation, leads Results)",
        "endpoint": "auc_regret_norm", "estimand": "transfer vs cold start and vs state-blind marginal replay",
        "allowed": ("In a prospective expansion from 288 to 4,608 configurations, only a factorized "
                    "UCB search strategy outperformed both cold start and a state-blind replay of "
                    "its own search marginals (+0.03073, LCB95 +0.01990, n=60); the neural carrier "
                    "did not (-0.01178, [-0.01849, -0.00484])."),
        "forbidden": ["only UCB1 learns", "the neuron has no memory",
                      "UCB1 is universally superior", "factorized UCB policy"],
        "why_forbidden": "'policy' would conflate the outer loop with within-episode control",
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
        "allowed": ("At matched parameter budgets (532 vs 529) the KAN attained better supervised "
                    "fit yet searched worse (+0.01037, CI95 [+0.00302, +0.01893], p = 0.0012), and "
                    "the best searcher of the seven-architecture bake-off was a five-parameter "
                    "neuron."),
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
        "contract_sha256": payload.get("contract_sha256"),
        "falsifiers_all_passed": payload.get("falsifiers", {}).get("all_passed"),
    }


def grade(meta: dict, row: dict) -> tuple[str, list[str]]:
    """Read the grade; never assume it. Missing fields are named, not filled in."""
    missing = [k for k in ("run_role", "scope", "claim_status", "self_sha256") if not meta.get(k)]
    if missing and row.get("sibling_receipt"):
        return "GRADE_IN_SIBLING_RECEIPT", missing
    if missing:
        return "GRADE_NOT_MACHINE_DISCOVERABLE", missing
    rr, sc = str(meta["run_role"]), str(meta["scope"])
    if "CONFIRMATION" in rr or "CONFIRMATION" in sc:
        return "CONFIRMATORY", []
    if "REPLAY" in rr or "REPLAY" in sc:
        return "REPLAY", []
    if "DIAGNOSTIC" in rr:
        return "DIAGNOSTIC", []
    return "DEVELOPMENT", []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows, problems = [], []
    for c in CLAIMS:
        p = Path(c["artifact"])
        meta = digests(p)
        g, missing = grade(meta, c)
        if not meta["exists"]:
            problems.append(f"{c['claim_id']}: artifact missing at {p}")
        if meta.get("falsifiers_all_passed") is False:
            problems.append(f"{c['claim_id']}: falsifiers did not all pass -- cite WITH the failure")
        row = dict(c)
        row.update({"evidence_grade": g, "missing_top_level_fields": missing, **meta})
        if c.get("sibling_receipt"):
            row["sibling_receipt_file_sha256"] = digests(Path(c["sibling_receipt"]))["file_sha256"]
        rows.append(row)

    payload = {
        "schema_version": "paper2_claim_lock_v1",
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
