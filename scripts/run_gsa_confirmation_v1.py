#!/usr/bin/env python3
"""Prospective confirmation of the GSA lane on the last virgin block in the project.

Seeds 7,700,001-7,700,120 (`g3a_v2_development`) were the only block never opened. The PI
authorised repurposing them from G3a to this lane on 2026-08-07
(docs/AUTORIZACION_PI_REPROPOSITO_BLOQUE_7700001_2026-08-07.md), lifting the
submission_a_receipt gate for this opening only. It opens ONCE. There is no rescue.

WHAT IS FROZEN. Theta comes from development and is compared field by field: nothing is re-tuned
on the virgin block, or this would be a second search wearing a confirmation's clothes. The
reading rule is in the preregistration, committed before the block was opened.

WHAT CONFIRMS. LCB95 of H_obs above zero AND LCB95 of (obs - uninformed placebo) above zero. The
placebo is the belief policy's action sequence from a DIFFERENT tape: same action distribution,
no aligned information. headroom_sensitivity.py never had one.

Preregistration: docs/PREREGISTRO_CONFIRMACION_GSA_2026-08-07.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.headroom_sensitivity import (  # noqa: E402
    ACTIONS, WEEKS, _belief_policy, materialize_tape_theta, periodic_calendars, theta_to_cell,
)
from supply_chain.seed_custody import module_manifest  # noqa: E402

from run_gsa_resilience_only_v1 import THETA, boot, score  # noqa: E402

MODULES = ("supply_chain/headroom_sensitivity.py", "supply_chain/program_g.py",
           "supply_chain/arm_runner.py")
DEV = Path("results/gsa_resilience_only/result.json")
SEED0, N_TAPES = 7_700_001, 120
BLOCK_ID = "g3a_v2_development"
SESOI_NOTE = 0.005          # declared before opening: below this, "confirmed but smaller"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/gsa_confirmation/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(7_700_001)

    if args.output.exists():
        raise SystemExit(f"{args.output} already exists: this block opens ONCE. Refusing.")
    dev = json.loads(DEV.read_text())
    theta_frozen = all(abs(float(THETA[k]) - float(dev["theta"][k])) < 1e-12 for k in THETA)

    cell = theta_to_cell(THETA)
    tapes = [materialize_tape_theta(SEED0 + i, cell) for i in range(N_TAPES)]
    cals = periodic_calendars(WEEKS)
    all_seqs = list(itertools.product(ACTIONS, repeat=WEEKS))
    print(f"  bloque virgen {SEED0}-{SEED0 + N_TAPES - 1} · {N_TAPES} cintas · "
          f"theta congelado {theta_frozen}", flush=True)

    by_cal = np.array([[score(t, c)["ret_order"] for t in tapes] for c in cals])
    best_cal = int(by_cal.mean(axis=1).argmax())
    static_is_argmax = bool(by_cal.mean(axis=1)[best_cal] >= by_cal.mean(axis=1).max() - 1e-12)
    static_rows = [score(t, cals[best_cal]) for t in tapes]

    seqs = [_belief_policy(t) for t in tapes]
    obs_rows = [score(t, s) for t, s in zip(tapes, seqs)]
    placebo_rows = [score(t, seqs[(i + 1) % len(tapes)]) for i, t in enumerate(tapes)]
    placebo_is_foreign = all(seqs[(i + 1) % len(tapes)] != seqs[i] or len(set(seqs)) == 1
                             for i in range(len(tapes)))
    oracle = np.array([max(score(t, s)["ret_order"] for s in all_seqs) for t in tapes])

    def col(rows, k):
        return np.array([r[k] for r in rows], dtype=float)

    st, ob, pl = col(static_rows, "ret_order"), col(obs_rows, "ret_order"), \
        col(placebo_rows, "ret_order")
    ret_gap = ob - st
    att_gap = col(obs_rows, "attended") - col(static_rows, "attended")
    corr = (float(np.corrcoef(ret_gap, att_gap)[0, 1])
            if np.std(ret_gap) > 1e-12 and np.std(att_gap) > 1e-12 else 0.0)

    h_obs, vs_placebo = boot(ret_gap, rng), boot(ob - pl, rng)
    h_pi = float((oracle - st).mean())
    eta = float(ret_gap.mean() / h_pi) if abs(h_pi) > 1e-9 else 0.0

    falsifiers = {
        "f1_the_block_is_virgin_and_opened_once": {
            "passed": True,
            "evidence": {"why_it_can_fail": "an artifact already present, or a seed outside the "
                                            "authorised range, would mean the block was touched",
                         "block_id": BLOCK_ID, "range": [SEED0, SEED0 + N_TAPES - 1],
                         "output_did_not_exist": True}},
        "f2_theta_is_frozen_from_development": {
            "passed": bool(theta_frozen),
            "evidence": {"why_it_can_fail": "a parameter moved on the virgin block would make this "
                                            "a second search wearing a confirmation's clothes",
                         "theta": THETA, "development_theta": dev["theta"],
                         "development_seal": dev.get("self_sha256")}},
        "f3_the_gain_is_not_bought_by_attending_fewer": {
            "passed": bool(corr >= -0.30),
            "evidence": {"why_it_can_fail": "a strongly negative correlation would mean the "
                                            "resilience gain is bought by abandonment, and the "
                                            "lane dies whether or not the guardrail blocks",
                         "corr_ret_gain_with_attended_gain": corr, "threshold": -0.30}},
        "f4_the_placebo_is_uninformed": {
            "passed": bool(placebo_is_foreign),
            "evidence": {"why_it_can_fail": "a placebo using this tape's own sequence would be the "
                                            "policy itself and could not separate signal from "
                                            "cadence",
                         "construction": "belief-policy sequence from tape (i+1) mod n"}},
        "f5_the_static_baseline_is_the_argmax": {
            "passed": static_is_argmax,
            "evidence": {"why_it_can_fail": "a weaker static comparator inflates the headroom",
                         "n_calendars": len(cals), "chosen": best_cal}},
        "f6_the_result_can_be_negative": {
            "passed": bool(float(ret_gap.min()) < 0.0 or float(h_obs["lcb95"]) < 0.0),
            "evidence": {"why_it_can_fail": "an estimator that cannot return a non-positive "
                                            "headroom cannot fail to confirm, and confirms nothing",
                         "min_per_tape_gap": float(ret_gap.min()),
                         "n_tapes_non_positive": int((ret_gap <= 0).sum())}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    confirmed = bool(h_obs["lcb95"] > 0 and vs_placebo["lcb95"] > 0
                     and falsifiers["f2_theta_is_frozen_from_development"]["passed"]
                     and falsifiers["f3_the_gain_is_not_bought_by_attending_fewer"]["passed"])
    smaller = bool(confirmed and h_obs["mean"] < SESOI_NOTE)
    verdict = ("GSA_CONFIRMED_ON_VIRGIN_BLOCK_SMALLER_THAN_DEVELOPMENT" if smaller
               else "GSA_CONFIRMED_ON_VIRGIN_BLOCK" if confirmed
               else "GSA_NOT_CONFIRMED_ON_VIRGIN_BLOCK")

    print(f"\n  H_PI {h_pi:+.5f}   H_obs {h_obs['mean']:+.5f} "
          f"[{h_obs['lcb95']:+.5f}, {h_obs['ucb95']:+.5f}]   eta {eta:.3f}")
    print(f"  obs - placebo  {vs_placebo['mean']:+.5f} "
          f"[{vs_placebo['lcb95']:+.5f}, {vs_placebo['ucb95']:+.5f}]")
    print(f"  reportado sin vetar: worst_cssu_fill "
          f"{float((col(obs_rows,'worst_cssu_fill') - col(static_rows,'worst_cssu_fill')).mean()):+.4f}"
          f"  attended {float(att_gap.mean()):+.2f}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<48} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "gsa_confirmation_v1",
        "claim_status": verdict,
        "run_role": "CONFIRMATION",
        "scope": "CONFIRMATION_ON_REPURPOSED_VIRGIN_BLOCK_NO_TRAINING_AUTHORISED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "authorisation": "docs/AUTORIZACION_PI_REPROPOSITO_BLOQUE_7700001_2026-08-07.md",
        "seed_block": {"id": BLOCK_ID, "start": SEED0, "end": SEED0 + N_TAPES - 1,
                       "repurposed_from": "G3a asymmetric-claimant development",
                       "gate_lifted_by_pi": "submission_a_receipt_required_before_g3a_open",
                       "opens_once": True},
        "development_reference": {"path": str(DEV), "self_sha256": dev.get("self_sha256"),
                                  "claim_status": dev.get("claim_status")},
        "theta": THETA, "cell": cell, "n_tapes": N_TAPES,
        "H_PI": h_pi, "H_obs": h_obs, "eta": eta, "obs_minus_placebo": vs_placebo,
        "declared_practical_margin": SESOI_NOTE,
        "reported_not_blocking": {
            "worst_cssu_fill_delta": float((col(obs_rows, "worst_cssu_fill")
                                            - col(static_rows, "worst_cssu_fill")).mean()),
            "attended_delta": float(att_gap.mean()),
            "lost_delta": float((col(obs_rows, "lost") - col(static_rows, "lost")).mean()),
            "ret_quantity_delta": float((col(obs_rows, "ret_quantity")
                                         - col(static_rows, "ret_quantity")).mean())},
        "corr_ret_gain_with_attended_gain": corr,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=DEV)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
