#!/usr/bin/env python3
"""A1 procurement-authority screen: does opening Op1/Op2 create decision headroom?

Development screen only. Cannot open confirmation universes and cannot authorize RL.

Context. On 2026-07-28 Garrido authorised widening the decision surface upstream, by
name: *"inclusive aguas arriba de la cadena, en los proveedores... en los de materias
primas, si"*. The procurement family is the cheapest level of the authority ladder
because it needs NO new physics: `op1_rop`, `op2_rop` and `op2_q` already live in
`MFSCSimulation.params` and are re-read by `_op1_contracting` / `_op2_supplier_delivery`
on every cycle. No Gym contract has ever written them.

This screen answers two questions, in order, before any network is trained:

  G0 liveness  Does varying each key actually move a trajectory? A key present in
               `params` is not decision authority until this is demonstrated. Note
               `op1_rop` only gates Op2 under `procurement_contract_mode=
               "causal_coupled"`; under the default `legacy_independent` its loop is
               a bare timeout with no downstream consumer, so the screen runs both
               modes and reports them separately.

  G1 headroom  Over a frozen grid of the three keys, evaluated on common random
               numbers across tapes:

                   B = max_g  mean_t ReT(g, t)      best single setting (static bar)
                   C = mean_t max_g ReT(g, t)       clairvoyant per-tape ceiling
                   H_PI = C - B                     perfect-information headroom

               H_PI is a DIAGNOSTIC CEILING, not an opportunity for a learner: the
               clairvoyant picks per tape using outcomes no deployable policy sees.
               H_PI ~ 0 is nonetheless decisive in the negative direction -- it means
               no policy of any kind, however informed, can profit from this family,
               so A1 closes without training anything.

Metric is the canonical `ret_excel` from `compute_episode_metrics`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK, OPERATIONS  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

# Thesis base values: Op1 ROP 4,032 h (biannual), Op2 ROP 672 h (monthly),
# Op2 Q 190,000 per raw material. The grid spans half to double each base, which
# keeps every point operationally interpretable as a contracting/delivery posture.
BASE = {
    "op1_rop": float(OPERATIONS[1]["rop"]),
    "op2_rop": float(OPERATIONS[2]["rop"]),
    "op2_q": float(OPERATIONS[2]["q"]),
}
GRID = {
    "op1_rop": [0.5, 0.75, 1.0, 1.5, 2.0],
    "op2_rop": [0.5, 0.75, 1.0, 1.5, 2.0],
    "op2_q": [0.5, 0.75, 1.0, 1.5, 2.0],
}


def evaluate(setting: dict[str, float], tape: int, horizon_weeks: int,
             contract_mode: str) -> dict[str, float]:
    """One full-DES episode under a fixed procurement posture, CRN-locked to `tape`."""
    sim = MFSCSimulation(
        shifts=1,
        seed=tape,
        horizon=float(horizon_weeks) * HOURS_PER_WEEK,
        risks_enabled=True,
        risk_level="current",
        strict_exogenous_crn=True,
        procurement_contract_mode=contract_mode,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )
    sim.step(action=dict(setting),
             step_hours=float(horizon_weeks) * HOURS_PER_WEEK)
    m = compute_episode_metrics(sim)
    ledger = sim.flow_ledger()
    return {
        "ret_excel": float(m["ret_excel"]),
        # Uncensored companion: `ret_excel` scores each policy on a different
        # population under risk (policy-dependent order omission), so both are
        # always reported and neither is selected after the fact.
        "ret_excel_full_ledger": float(m["ret_excel_full_ledger"]),
        "flow_fill_rate": float(m["flow_fill_rate"]),
        "lost_orders": float(m["lost_orders"]),
        "service_loss_auc": float(m["service_loss_auc_ration_hours"]),
        "raw_external": float(sim.total_external_raw_material),
        "raw_residual": float(ledger["raw_residual"]),
        "ration_residual": float(ledger["ration_residual"]),
    }


def settings_grid() -> list[dict[str, float]]:
    keys = list(GRID)
    out = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        out.append({k: BASE[k] * mult for k, mult in zip(keys, combo)})
    return out


def liveness(tape: int, horizon_weeks: int, contract_mode: str) -> dict[str, dict]:
    """Per-key: does moving ONLY that key, everything else at base, move the run?"""
    base_row = evaluate(dict(BASE), tape, horizon_weeks, contract_mode)
    report: dict[str, dict] = {}
    for key in BASE:
        rows = []
        for mult in (0.5, 2.0):
            setting = dict(BASE)
            setting[key] = BASE[key] * mult
            rows.append(evaluate(setting, tape, horizon_weeks, contract_mode))
        d_ret = [abs(r["ret_excel"] - base_row["ret_excel"]) for r in rows]
        d_raw = [abs(r["raw_external"] - base_row["raw_external"]) for r in rows]
        report[key] = {
            "base_ret_excel": base_row["ret_excel"],
            "ret_excel_at_half": rows[0]["ret_excel"],
            "ret_excel_at_double": rows[1]["ret_excel"],
            "max_abs_delta_ret": max(d_ret),
            "max_abs_delta_raw_external": max(d_raw),
            # A key is live if it moves ReT, or at minimum moves physical material
            # flow (a key can be physically live while ReT-neutral).
            "live_on_ret": max(d_ret) > 1e-12,
            "live_on_material": max(d_raw) > 1e-9,
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tapes", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=1_210_001)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--contract-modes", nargs="+",
                    default=["legacy_independent", "causal_coupled"])
    ap.add_argument("--liveness-only", action="store_true")
    ap.add_argument("--max-settings", type=int, default=None,
                    help="smoke only: truncate the grid")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/authority_ladder/a1_procurement"))
    args = ap.parse_args()

    tapes = [args.seed_start + i for i in range(args.tapes)]
    grid = settings_grid()
    if args.max_settings is not None:
        grid = grid[: args.max_settings]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    out: dict[str, dict] = {}

    for mode in args.contract_modes:
        print(f"\n=== contract_mode={mode} ===", flush=True)
        live = liveness(tapes[0], args.horizon_weeks, mode)
        for k, v in live.items():
            print(f"  liveness {k:10s} dReT={v['max_abs_delta_ret']:.6g} "
                  f"dRaw={v['max_abs_delta_raw_external']:.6g} "
                  f"live_ret={v['live_on_ret']} live_mat={v['live_on_material']}",
                  flush=True)
        block: dict = {"liveness": live}

        if not args.liveness_only:
            # rows[g][t] = ReT of grid point g on tape t; CRN means the same tape
            # index is the same exogenous stream for every grid point.
            METRICS = ("ret_excel", "ret_excel_full_ledger")
            rows = {k: np.zeros((len(grid), len(tapes)), np.float64) for k in METRICS}
            guard = []
            for gi, setting in enumerate(grid):
                for ti, tape in enumerate(tapes):
                    r = evaluate(setting, tape, args.horizon_weeks, mode)
                    for _k in METRICS:
                        rows[_k][gi, ti] = r[_k]
                    guard.append((r["raw_residual"], r["ration_residual"]))
                print(f"  grid {gi + 1}/{len(grid)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
            rng = np.random.default_rng(20260728)
            by_metric: dict[str, dict] = {}
            for _k in METRICS:
                _a = rows[_k]
                _m = _a.mean(axis=1)
                _b = int(_m.argmax())
                _gap = _a.max(axis=0) - _a[_b, :]
                _boot = np.array([rng.choice(_gap, len(tapes), True).mean()
                                  for _ in range(10_000)])
                by_metric[_k] = {
                    "best_setting": grid[_b],
                    "static_bar_mean": float(_m[_b]),
                    "clairvoyant_ceiling_mean": float(_a.max(axis=0).mean()),
                    "h_pi": float(_a.max(axis=0).mean() - _m[_b]),
                    "h_pi_lcb95": float(np.quantile(_boot, 0.05)),
                    "grid_span_mean": float(_m.max() - _m.min()),
                }
            # Primary reporting stays on the canonical metric; companion disclosed.
            per_setting_mean = rows["ret_excel"].mean(axis=1)
            best_g = int(per_setting_mean.argmax())
            static_bar = float(per_setting_mean[best_g])
            ceiling = float(rows["ret_excel"].max(axis=0).mean())
            # Root-clustered here is simply tape-clustered: one tape, one draw.
            per_tape_gap = rows["ret_excel"].max(axis=0) - rows["ret_excel"][best_g, :]
            boot = np.array([rng.choice(per_tape_gap, len(tapes), True).mean()
                             for _ in range(10_000)])
            block.update({
                "n_settings": len(grid),
                "n_tapes": len(tapes),
                "best_setting": grid[best_g],
                "static_bar_mean_ret": static_bar,
                "clairvoyant_ceiling_mean_ret": ceiling,
                "h_pi": ceiling - static_bar,
                "h_pi_lcb95": float(np.quantile(boot, 0.05)),
                "n_tapes_with_strictly_better_setting": int((per_tape_gap > 1e-12).sum()),
                "by_metric": by_metric,
                "max_abs_mass_residual": float(
                    max(max(abs(a), abs(b)) for a, b in guard)),
            })
            for _k, _v in by_metric.items():
                print(f"    {_k:24s} bar={_v['static_bar_mean']:.8f} "
                      f"H_PI={_v['h_pi']:.6g} LCB95={_v['h_pi_lcb95']:.6g} "
                      f"span={_v['grid_span_mean']:.6g}", flush=True)
            print(f"  BAR={static_bar:.6f}  CEIL={ceiling:.6f}  "
                  f"H_PI={ceiling - static_bar:.6g} "
                  f"LCB95={block['h_pi_lcb95']:.6g}", flush=True)
        out[mode] = block

    payload = {
        "schema_version": "authority_ladder_a1_procurement_v1",
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "ret_excel",
        "base_values": BASE,
        "grid_multipliers": GRID,
        "tapes": tapes,
        "horizon_weeks": args.horizon_weeks,
        "h_pi_is_diagnostic_ceiling_not_learner_opportunity": True,
        "results": out,
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    path = args.output_dir / "screen_result.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
