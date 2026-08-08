#!/usr/bin/env python3
"""AIC and Ramsey RESET on the sealed design surface, because Garrido asked for them.

WHY, AND WHY THEY ARE SECONDARY. At the 2026-08-07 meeting Garrido asked for AIC and a Ramsey RESET
to test whether the variables behave linearly, reasoning that if they do it would explain why a KAN
does not beat an MLP. The ask is legitimate and these run in minutes on already-sealed data, so
there is no reason not to answer it in the language he asked for.

They are NOT arbiters, and the artifact says so in its own payload:

  * AIC is a RELATIVE selection criterion between models fitted to the same response. A lower AIC
    for a linear specification does not establish that the surface is linear, only that the extra
    parameters of the alternative did not pay for themselves on THIS sample.
  * Ramsey RESET detects certain omitted non-linearity in a given specification. A non-significant
    RESET is not evidence of linearity; it is failure to reject, at whatever power the design has.
  * Neither speaks to the question the paper actually asks, which is whether a more expressive
    approximator SEARCHES better. Predictive fit and sequential-search efficiency are distinct
    quantities, and this project has already measured a case where they point opposite ways.

So the primary evidence stays what it was: out-of-sample performance under seed-grouped CV, which
this script reports alongside, and search regret, which lives elsewhere.

DATA. The sealed base surface of the transfer confirmation: 288 configurations x 6 contexts x 60
seeds. Response is the cached `value` (`ret_excel_risk_conditional`). Predictors are the four base
factors. Contexts are fitted SEPARATELY -- pooling six risk regimes into one regression would
manufacture curvature out of regime shifts and then attribute it to the factors.

Design authorised by the approved plan of 2026-08-08 (Phase 3). Development diagnostic on sealed
data. No seeds opened, no adjudication, no learner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import linear_reset  # noqa: E402

from supply_chain.arm_runner import run_falsifiers, seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

from build_transfer_confirmation_cache_v1 import BASE_CONFIGS, BASE_NAMES  # noqa: E402

CACHE = Path("results/surface_cache/garrido_transfer_confirmation_v2_base")
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def load_surface() -> dict[str, dict]:
    """One design matrix per context. Seed is carried so CV can group on it."""
    by_ctx: dict[str, dict[str, list]] = {}
    for path in sorted(CACHE.rglob("*.json")):
        p = json.loads(path.read_text())
        ctx, seed = p["context"], int(path.stem)
        d = by_ctx.setdefault(ctx, {"X": [], "y": [], "seed": []})
        for idx, cell in enumerate(p["cells"]):
            cfg = BASE_CONFIGS[idx]
            d["X"].append([float(cfg[n]) for n in BASE_NAMES])
            d["y"].append(float(cell["value"]))
            d["seed"].append(seed)
    return {c: {k: np.asarray(v, dtype=float) for k, v in d.items()} for c, d in by_ctx.items()}


def designs(X: np.ndarray) -> dict[str, np.ndarray]:
    """Four nested-ish specifications on the same rows. `linear` is the RESET reference."""
    n, p = X.shape
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)  # scale so squares stay conditioned
    inter = np.column_stack([Z[:, i] * Z[:, j] for i in range(p) for j in range(i + 1, p)])
    return {
        "linear": Z,
        "linear_interactions": np.column_stack([Z, inter]),
        "quadratic": np.column_stack([Z, Z ** 2]),
        "quadratic_interactions": np.column_stack([Z, Z ** 2, inter]),
    }


def grouped_cv_r2(Xd: np.ndarray, y: np.ndarray, seed: np.ndarray, folds: int = 5) -> float:
    """Held-out R2 with folds cut on SEED, never on rows: two rows from one seed share a tape, so a
    row-wise split would leak the tape across the split and inflate every model equally."""
    uniq = np.unique(seed)
    parts = np.array_split(uniq, folds)
    num, den = 0.0, 0.0
    for held in parts:
        te = np.isin(seed, held)
        tr = ~te
        if tr.sum() < Xd.shape[1] + 2 or te.sum() == 0:
            continue
        A = sm.add_constant(Xd[tr], has_constant="add")
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred = sm.add_constant(Xd[te], has_constant="add") @ beta
        num += float(((y[te] - pred) ** 2).sum())
        den += float(((y[te] - y[tr].mean()) ** 2).sum())
    return float("nan") if den <= 0 else 1.0 - num / den


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=Path("results/functional_form_diagnostics/result.json"))
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_3.md"))
    args = ap.parse_args()
    t0 = time.time()

    surface = load_surface()
    out: dict[str, dict] = {}
    for ctx, d in sorted(surface.items()):
        X, y, seed = d["X"], d["y"], d["seed"].astype(int)
        specs = designs(X)
        rows = {}
        for name, Xd in specs.items():
            fit = sm.OLS(y, sm.add_constant(Xd, has_constant="add")).fit()
            rows[name] = {
                "aic": float(fit.aic), "bic": float(fit.bic),
                "r2_in_sample": float(fit.rsquared),
                "r2_heldout_seed_grouped": grouped_cv_r2(Xd, y, seed),
                "k_params": int(fit.df_model) + 1, "n": int(fit.nobs),
            }
        lin = sm.OLS(y, sm.add_constant(specs["linear"], has_constant="add")).fit()
        reset = {}
        for power in (2, 3):
            try:
                r = linear_reset(lin, power=power, test_type="fitted", use_f=True)
                reset[f"power_{power}"] = {"statistic": float(r.statistic),
                                           "p_value": float(r.pvalue)}
            except Exception as exc:  # pragma: no cover - reported, not swallowed
                reset[f"power_{power}"] = {"error": f"{type(exc).__name__}: {exc}"}
        best_aic = min(rows, key=lambda k: rows[k]["aic"])
        best_oos = max(rows, key=lambda k: (rows[k]["r2_heldout_seed_grouped"]
                                            if np.isfinite(rows[k]["r2_heldout_seed_grouped"])
                                            else -np.inf))
        out[ctx] = {
            "models": rows, "ramsey_reset_on_linear": reset,
            "aic_selects": best_aic, "heldout_r2_selects": best_oos,
            "response_sd": float(y.std()), "n_seeds": int(len(np.unique(seed))),
            "delta_aic_linear_minus_best": float(rows["linear"]["aic"] - rows[best_aic]["aic"]),
        }

    def h1():
        """The factors move the response at all.

        CAN FAIL: this project has measured 4.56M units of raw material moving exactly zero ReT, so
        an inert design is a live possibility and would make every diagnostic below meaningless."""
        r2 = {c: v["models"]["linear"]["r2_in_sample"] for c, v in out.items()}
        return bool(all(v > 0.01 for v in r2.values())), r2

    def h2():
        """AIC is compared only across models fitted to identical rows.

        CAN FAIL: a specification that dropped rows would make its AIC incomparable, and AIC has no
        internal guard against that."""
        bad = {c: {m: v["models"][m]["n"] for m in v["models"]}
               for c, v in out.items() if len({v["models"][m]["n"] for m in v["models"]}) != 1}
        return not bad, {"contexts_with_unequal_n": bad,
                         "n_per_context": {c: v["models"]["linear"]["n"] for c, v in out.items()}}

    def h3():
        """RESET returned a usable statistic in every context.

        CAN FAIL: a singular or near-singular design matrix makes the auxiliary regression
        degenerate, and a swallowed exception would look like a clean non-rejection."""
        errs = {c: v["ramsey_reset_on_linear"] for c, v in out.items()
                if any("error" in x for x in v["ramsey_reset_on_linear"].values())}
        return not errs, {"contexts_with_reset_errors": errs}

    def h4():
        """AIC and held-out R2 are allowed to disagree, and the artifact records when they do.

        This is not a pass/fail of the surface -- it is the check that we LOOKED. It fails only if
        the two selectors were never compared, which would mean the script silently reported one."""
        agree = {c: (v["aic_selects"] == v["heldout_r2_selects"]) for c, v in out.items()}
        return len(agree) == len(out), {"aic_agrees_with_heldout": agree,
                                        "n_contexts_compared": len(agree)}

    fals = run_falsifiers({"h1_the_factors_move_the_response": h1,
                           "h2_aic_compared_on_identical_rows": h2,
                           "h3_reset_is_computable": h3,
                           "h4_both_selectors_were_compared": h4})

    payload = {
        "schema_version": "functional_form_diagnostics_v1",
        "claim_status": "FUNCTIONAL_FORM_DIAGNOSTICS_REQUESTED_BY_DOMAIN_EXPERT",
        "scope": "DEVELOPMENT_DIAGNOSTIC_ON_SEALED_SURFACE_NO_SEEDS_OPENED_NO_ADJUDICATION",
        "run_role": "DIAGNOSTIC",
        "primary_metric": "heldout_r2_seed_grouped",
        "secondary_metrics": ["aic", "ramsey_reset_p_value"],
        "requested_by": "Garrido, meeting 2026-08-07",
        "data_source": str(CACHE),
        "predictors": list(BASE_NAMES),
        "response": "ret_excel_risk_conditional (cached `value`)",
        "by_context": out,
        "falsifiers": fals,
        "elapsed_seconds": time.time() - t0,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "what_these_do_not_establish": [
            "AIC is a relative criterion between models on the same response; a lower AIC for the "
            "linear specification does not establish that the surface is linear",
            "a non-significant Ramsey RESET is failure to reject, not evidence of linearity, and "
            "its power here is not characterised",
            "neither addresses whether a more expressive approximator SEARCHES better, which is the "
            "question the manuscript asks; fit and sequential-search efficiency are distinct",
            "contexts are fitted separately on purpose; any pooled statement would confound factor "
            "curvature with regime shifts",
        ],
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/demand_process/result.json"))

    print(f"\n{'contexto':14}{'AIC elige':>24}{'R2 oos elige':>24}{'RESET p (2)':>13}{'R2 lin oos':>12}")
    for c, v in out.items():
        p2 = v["ramsey_reset_on_linear"].get("power_2", {}).get("p_value", float("nan"))
        print(f"{c:14}{v['aic_selects']:>24}{v['heldout_r2_selects']:>24}{p2:13.3g}"
              f"{v['models']['linear']['r2_heldout_seed_grouped']:12.4f}")
    for k, v in fals.items():
        if k != "all_passed":
            print(f"  {k:40} {'PASA' if v['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if fals["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
