#!/usr/bin/env python3
"""CVaR gate instrument audit — contract cvar_gate_instrument_audit_v1 (frozen 2026-07-17).

Was the zero-margin simultaneous non-inferiority gate on ret_visible_cvar10 a calibrated
instrument, or a de-facto superiority test? BURNED corrective-validation data only (7430001-48).

Machinery: replicates scripts/screen_program_o_fixed_clock_hobs_validation.joint_bootstrap
line-for-line on the per-tape estimand-delta panel (linear `counts @ D` bootstrap, centered
studentized max-t over the full family incl. placebos, 95th-quantile critical, LCB = point - c*SE).
Gate M0 self-verifies identity by reproducing the published LCBs with the original seed and
10,000 resamples before any analysis is read.

Analyses (all frozen in the contract):
  A1 power curve: worlds with TRUE cvar10 effect s (observed cvar columns recentred + s in every
     cell), outer tape-resampled worlds, inner max-t bootstrap; power = P(joint LCB95 >= margin).
  A2 positive controls: per-tape mean-oracle (argmax ret_visible) and tail-oracle (argmax cvar10)
     over the full 65,536-calendar frontier, metric-family-only (fewer estimands -> SMALLER
     critical value -> a-fortiori direction disclosed).
  A3 trivial control = A1 at s=0 (true effect exactly zero, real noise). The degenerate
     self-comparison (delta identically 0 -> SE=0 -> inactive -> LCB=point=0 -> passes by the
     zero-variance exact bound) is reported analytically.
  A4 margins {-0.01, -0.02} evaluated on the same world LCBs (report-only; frozen grid).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.screen_program_o_state_rich_fit import HIGHER_KEYS, LOWER_KEYS  # noqa: E402

RUN = ROOT / "results/program_o/fixed_clock_hobs_corrective_validation_v1/remote_run/artifacts/validation"
OUT_DIR = ROOT / "results/program_o/cvar_gate_instrument_audit_v1"
CELLS = ["rho75_share90", "rho90_share75", "rho90_share90"]
N_TAPES = 48
SEED_TEXT = b"program-o-fixed-clock-hobs-validation-v1"   # original machinery seed (identity)
AUDIT_SEED = 20260717
SHIFT_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
MARGINS = [0.0, -0.01, -0.02]
OUTER_WORLDS = 600
INNER_RESAMPLES = 2000       # reduced vs 10,000 (disclosed); M0 identity check uses 10,000
CVAR_KEY = "ret_visible_cvar10"


def load_everything():
    result = json.load(open(RUN / "result.json"))
    panels, ctrl_idx, static_idx, placebo_rows = {}, {}, {}, {}
    keys_needed = ["ret_visible", CVAR_KEY] + list(HIGHER_KEYS) + list(LOWER_KEYS)
    keys_needed = list(dict.fromkeys(keys_needed))
    for cell in CELLS:
        mats = {k: np.zeros((N_TAPES, 65536)) for k in keys_needed}
        for t in range(N_TAPES):
            z = np.load(RUN / f"raw_calendar_matrix/{cell}/tape_{7430001 + t}.npz")
            for k in keys_needed:
                mats[k][t] = z[k]
        panels[cell] = mats
        ctrl_idx[cell] = np.asarray(result["cells"][cell]["calendar_indices"], dtype=np.int64)
        static_idx[cell] = int(result["cells"][cell]["static_index"])
        placebo_rows[cell] = result["placebos"][cell]
    return result, panels, ctrl_idx, static_idx, placebo_rows


def build_delta_panel(panels, ctrl_idx, static_idx, placebo_rows, *, metric_only=False):
    """Per-tape signed delta columns, EXACTLY as joint_bootstrap constructs them."""
    names, cols, kinds = [], [], []
    rows = np.arange(N_TAPES)
    for cell in CELLS:
        p = panels[cell]
        pol = {k: p[k][rows, ctrl_idx[cell]] for k in p}
        s = static_idx[cell]
        signed = [("ret_visible", 1.0, "primary")]
        signed += [(k, 1.0, "guardrail") for k in HIGHER_KEYS]
        signed += [(k, -1.0, "guardrail") for k in LOWER_KEYS]
        for key, sign, kind in signed:
            names.append(f"{cell}::{kind}::{key}")
            kinds.append(kind)
            cols.append(sign * (pol[key] - p[key][:, s]))
        if not metric_only:
            real = p["ret_visible"][rows, ctrl_idx[cell]]
            for family, frow in placebo_rows[cell].items():
                for mode, plc in frow["placebos"].items():
                    idx = np.asarray(plc["calendar_indices"], dtype=np.int64)
                    names.append(f"{cell}::placebo::{family}::{mode}")
                    kinds.append("placebo")
                    cols.append(real - p["ret_visible"][rows, idx])
    return names, np.column_stack(cols)


def maxt_lcbs(D, *, resamples, rng):
    """Line-faithful replica of joint_bootstrap's inference on a (48, K) delta panel."""
    bidx = rng.integers(0, N_TAPES, size=(resamples, N_TAPES))
    counts = np.zeros((resamples, N_TAPES))
    for i, sample in enumerate(bidx):
        counts[i] = np.bincount(sample, minlength=N_TAPES)
    counts /= float(N_TAPES)
    point = D.mean(axis=0)
    boot = counts @ D
    se = boot.std(axis=0, ddof=1)
    active = se > 1e-15
    max_t = np.zeros(resamples)
    if np.any(active):
        std = (point[None, active] - boot[:, active]) / se[active]
        max_t = np.max(std, axis=1)
    crit = float(np.quantile(max_t, 0.95))
    lcb = point.copy()
    lcb[active] = point[active] - crit * se[active]
    return point, lcb, crit


def cvar_positions(names):
    return {n: i for i, n in enumerate(names) if n.endswith(f"guardrail::{CVAR_KEY}")}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result, panels, ctrl_idx, static_idx, placebo_rows = load_everything()
    names, D = build_delta_panel(panels, ctrl_idx, static_idx, placebo_rows)
    cpos = cvar_positions(names)

    # ---- M0: machinery identity (original seed, 10,000 resamples) ------------------------------
    seed = int.from_bytes(hashlib.sha256(SEED_TEXT).digest()[:8], "big")
    _, lcb0, crit0 = maxt_lcbs(D, resamples=10_000, rng=np.random.default_rng(seed))
    published = {n: result["inference"]["estimates"][n]["simultaneous_lcb95"] for n in names
                 if n in result["inference"]["estimates"]}
    m0 = {}
    for n, i in cpos.items():
        m0[n] = {"reproduced_lcb95": float(lcb0[i]), "published_lcb95": published.get(n)}
    ident_ok = all(v["published_lcb95"] is not None and abs(v["reproduced_lcb95"] - v["published_lcb95"]) < 5e-4
                   for v in m0.values())
    out = {"schema_version": "cvar_gate_instrument_audit_result_v1",
           "contract": "contracts/cvar_gate_instrument_audit_v1.json",
           "burned_data_only": True, "sealed_or_virgin_opened": False,
           "M0_machinery_identity": {"per_cell": m0, "reproduced_critical": crit0,
                                     "published_critical": result["inference"]["simultaneous_critical"],
                                     "tolerance": 5e-4, "pass": bool(ident_ok)},
           "reductions_disclosed": {"outer_worlds": OUTER_WORLDS, "inner_resamples": INNER_RESAMPLES,
                                    "identity_check_resamples": 10_000}}
    if not ident_ok:
        out["status"] = "STOP_MACHINERY_MISMATCH"
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out["M0_machinery_identity"], indent=1))
        return 1

    # ---- A1/A3/A4: power over true-effect worlds ----------------------------------------------
    ci = sorted(cpos.values())
    Dz = D.copy()
    Dz[:, ci] -= Dz[:, ci].mean(axis=0, keepdims=True)   # true cvar effect exactly 0, real noise
    rng = np.random.default_rng(AUDIT_SEED)
    world_tapes = rng.integers(0, N_TAPES, size=(OUTER_WORLDS, N_TAPES))
    power = {f"{s:+.2f}": {f"{m:+.2f}": {"per_cell_pass": [0] * len(ci), "all_cells_pass": 0}
                           for m in MARGINS} for s in SHIFT_GRID}
    for w in range(OUTER_WORLDS):
        Dw_base = Dz[world_tapes[w]]
        wrng = np.random.default_rng(np.random.SeedSequence([AUDIT_SEED, 7, w]))
        inner_rng_state = wrng
        for s in SHIFT_GRID:
            Dw = Dw_base.copy()
            Dw[:, ci] += s
            _, lcb, _ = maxt_lcbs(Dw, resamples=INNER_RESAMPLES,
                                  rng=np.random.default_rng(np.random.SeedSequence([AUDIT_SEED, 11, w, int(s * 1000)])))
            for m in MARGINS:
                rec = power[f"{s:+.2f}"][f"{m:+.2f}"]
                passes = [bool(lcb[i] >= m) for i in ci]
                for j, ok in enumerate(passes):
                    rec["per_cell_pass"][j] += int(ok)
                rec["all_cells_pass"] += int(all(passes))
    for s in power:
        for m in power[s]:
            rec = power[s][m]
            rec["per_cell_power"] = [c / OUTER_WORLDS for c in rec["per_cell_pass"]]
            rec["all_cells_power"] = rec["all_cells_pass"] / OUTER_WORLDS
    out["A1_A3_A4_power"] = {"shift_grid": SHIFT_GRID, "margins": MARGINS,
                             "cell_order": [n for n, _ in sorted(cpos.items(), key=lambda kv: kv[1])],
                             "power": power,
                             "A3_trivial_note": "s=+0.00 row IS the truly-equivalent policy (real noise, zero true effect); the degenerate literal self-comparison has delta==0 -> SE=0 -> inactive -> LCB=point=0 -> PASSES via the zero-variance exact bound (analytic)"}
    # minimum shift for 80% power at margin 0 (all cells jointly)
    xs = [float(s) for s in SHIFT_GRID]
    ys = [power[f"{s:+.2f}"]["+0.00"]["all_cells_power"] for s in xs]
    min80 = None
    for a, b, pa, pb in zip(xs, xs[1:], ys, ys[1:]):
        if pa < 0.8 <= pb:
            min80 = a + (b - a) * (0.8 - pa) / (pb - pa)
            break
    out["A1_min_true_effect_for_80pct_power_margin0"] = min80 if min80 is not None else (
        "above +0.08" if ys[-1] < 0.8 else "at or below grid minimum")

    # ---- A2: positive controls (oracles), metric-family only (a fortiori) ---------------------
    a2 = {}
    for oracle_name, key in (("mean_oracle_argmax_ret_visible", "ret_visible"),
                             ("tail_oracle_argmax_cvar10", CVAR_KEY)):
        oidx = {cell: panels[cell][key].argmax(axis=1).astype(np.int64) for cell in CELLS}
        onames, OD = build_delta_panel(panels, oidx, static_idx, placebo_rows, metric_only=True)
        _, olcb, ocrit = maxt_lcbs(OD, resamples=10_000, rng=np.random.default_rng(seed))
        opos = cvar_positions(onames)
        a2[oracle_name] = {
            "family": "metric estimands only (no placebos) -> smaller critical -> EASIER gate; a failure here holds a fortiori for the full family",
            "critical": ocrit,
            "per_cell": {n: {"point": float(OD[:, i].mean()), "lcb95": float(olcb[i]),
                             "passes_margin0": bool(olcb[i] >= 0.0)} for n, i in opos.items()},
        }
    out["A2_positive_controls"] = a2

    # ---- verdict per frozen decision rule ------------------------------------------------------
    trivial_pass_prob = power["+0.00"]["+0.00"]["all_cells_power"]
    mean_oracle_fails = any(not v["passes_margin0"] for v in a2["mean_oracle_argmax_ret_visible"]["per_cell"].values())
    defect = bool(trivial_pass_prob < 0.5 or mean_oracle_fails)
    out["decision"] = {
        "trivial_equivalent_pass_probability_margin0": trivial_pass_prob,
        "mean_oracle_fails_margin0": mean_oracle_fails,
        "instrument_defect_certified": defect,
        "rule": "defect iff trivial-equivalent pass prob < 0.5 OR the mean-oracle fails margin-0 (frozen in contract)",
        "consequence": ("documented instrument mis-specification -> hand to INDEPENDENT AUDITOR for adjudication together with Garrido M2; STOP unchanged"
                        if defect else "gate strict but valid; STOP armored; route depends entirely on M2"),
    }
    out["status"] = ("INSTRUMENT_DEFECT_CERTIFIED_PENDING_INDEPENDENT_ADJUDICATION" if defect
                     else "INSTRUMENT_VALID_STOP_ARMORED")
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=1))
    print("M0 identity:", out["M0_machinery_identity"]["pass"],
          "| crit repro/pub:", round(crit0, 4), "/", round(result["inference"]["simultaneous_critical"], 4))
    print("A3 trivial pass prob @margin0:", trivial_pass_prob)
    print("A1 min effect for 80% power @margin0:", out["A1_min_true_effect_for_80pct_power_margin0"])
    for k, v in a2.items():
        print(k, {n.split("::")[0]: p["passes_margin0"] for n, p in v["per_cell"].items()})
    print("STATUS:", out["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
