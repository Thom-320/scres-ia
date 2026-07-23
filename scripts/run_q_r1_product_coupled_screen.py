#!/usr/bin/env python3
"""Learner-blind screen of the product-coupled R24 carrier (Q-R1 risk door).

Contract: contracts/q_r1_product_coupled_execution_amendment_v1.json (frozen
2026-07-23) under parent contracts/q_r1_product_coupled_risk_door_v1.json.
Structure mirrors scripts/run_thesis_native_dispatch_screen.py.

Per (cell, root): a 12-campaign history whose latent sigma chain persists with
kappa_r; each campaign is one DIRECT SimPy ProgramO episode with ONLY R24
enabled at the thesis "increased" frequency, surge quantities product-assigned
by sigma with probability s_r.  Arms:

* const_0..const_3 -- the four constant weekly-count postures (Discrete(4)
  held all campaign), run directly.
* sigma_oracle -- CLAIRVOYANT: knows the true sigma of each campaign and
  plays the best constant posture for that sigma, where "best" is the frozen
  per-sigma argmax of mean early_ret_complete_cohort over the constant arms
  WITHIN THE SAME CELL.  Because the oracle holds one posture all campaign
  and all arms share the exact CRN surge timeline, its rows are derived from
  the already-run constant rows (no re-simulation; bitwise identical).
* posterior_sigma -- OBSERVABLE: chooses per campaign from the Beta-Bernoulli
  posterior over sigma computed from PREVIOUS campaigns' realized surge
  product assignments only (uniform prior), mapped through the same frozen
  per-sigma posture mapping.  Also derived from constant rows.

No learner return is used anywhere (amendment ``selection_forbidden``).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.q_r1_product_coupled import (  # noqa: E402
    SIGMA_VALUES,
    beta_bernoulli_sigma_posterior,
    demand_initial_regime_chain,
    direct_campaign_metrics,
    run_campaign,
    sigma_path,
)

# Frozen cell grid (amendment ``cells_frozen``): 2x3 lattice + control.
CELLS: dict[str, dict[str, float]] = {
    "k075_s070": {"kappa_r": 0.75, "s_r": 0.70},
    "k075_s085": {"kappa_r": 0.75, "s_r": 0.85},
    "k075_s100": {"kappa_r": 0.75, "s_r": 1.00},
    "k090_s070": {"kappa_r": 0.90, "s_r": 0.70},
    "k090_s085": {"kappa_r": 0.90, "s_r": 0.85},
    "k090_s100": {"kappa_r": 0.90, "s_r": 1.00},
    # Control: composition uninformative -> no reversal expected; a reversal
    # here is an artifact flag (amendment ``cells_frozen.control``).
    "k090_s050_control": {"kappa_r": 0.90, "s_r": 0.50},
}
CELL_ADJACENCY = [
    ["k075_s070", "k075_s085"], ["k075_s085", "k075_s100"],
    ["k090_s070", "k090_s085"], ["k090_s085", "k090_s100"],
    ["k075_s070", "k090_s070"], ["k075_s085", "k090_s085"],
    ["k075_s100", "k090_s100"],
]
SCREEN_ROOTS = list(range(7_590_001, 7_590_025))
PROBE_ROOTS = [7_590_901, 7_590_902]
HOLDOUT_ROOTS = set(range(7_590_101, 7_590_149))  # NEVER opened by this script
CAMPAIGNS_PER_HISTORY = 12
POSTURES = (0, 1, 2, 3)
OBJECTIVE = "early_ret_complete_cohort"


def _count_scheduler() -> dict:
    """The frozen weekly-count scheduler (same source as
    scripts/c6_perbatch_ceiling.py::_count_scheduler; re-read from the
    contract to avoid the torch import chain)."""
    parent = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def task(spec: dict) -> dict:
    """One direct campaign run (top-level so spawn-mode pools work)."""
    started = time.perf_counter()
    row = run_campaign(
        root=spec["root"],
        campaign_index=spec["campaign_index"],
        sigma=spec["sigma"],
        s_r=spec["s_r"],
        kappa=spec["kappa_r"],
        calendar=[spec["posture"]] * 8,
        scheduler=spec["scheduler"],
        initial_regime=spec["initial_regime"],
    )
    row.update(
        {
            "cell": spec["cell"],
            "arm": f"const_{spec['posture']}",
            "derived_from": None,
            "run_seconds": time.perf_counter() - started,
        }
    )
    return row


def _surge_signature(row: dict) -> tuple:
    return tuple(
        (item["time"], item["surge"], item["assigned_product"])
        for item in row["surge_log"]
    )


def _oracle_mapping(cell_rows: list[dict]) -> dict:
    """Frozen per-sigma posture mapping from the cell's constant arms.

    CLAIRVOYANT construction: uses the true sigma labels of the same cell's
    campaigns.  Documented as the H_PI oracle; never available to a learner.
    """
    mapping: dict[str, dict] = {}
    overall: dict[int, list[float]] = {posture: [] for posture in POSTURES}
    for row in cell_rows:
        overall[row["posture"]].append(row[OBJECTIVE])
    overall_mean = {
        posture: (sum(values) / len(values) if values else float("-inf"))
        for posture, values in overall.items()
    }
    overall_best = max(POSTURES, key=lambda posture: overall_mean[posture])
    for sigma in SIGMA_VALUES:
        by_posture: dict[int, list[float]] = {posture: [] for posture in POSTURES}
        for row in cell_rows:
            if row["sigma"] == sigma:
                by_posture[row["posture"]].append(row[OBJECTIVE])
        if not any(by_posture.values()):
            mapping[sigma] = {
                "posture": int(overall_best),
                "mean_objective": overall_mean[overall_best],
                "n_campaign_rows": 0,
                "fallback_overall_best": True,
            }
            continue
        means = {
            posture: (sum(values) / len(values) if values else float("-inf"))
            for posture, values in by_posture.items()
        }
        best = max(POSTURES, key=lambda posture: means[posture])
        mapping[sigma] = {
            "posture": int(best),
            "mean_objective": means[best],
            "n_campaign_rows": len(by_posture[best]),
            "fallback_overall_best": False,
        }
    mapping["_overall_best_posture"] = int(overall_best)
    mapping["_overall_best_mean"] = overall_mean[overall_best]
    mapping["_objective"] = OBJECTIVE
    mapping["_clairvoyant_note"] = (
        "per-sigma argmax over constant postures within the same cell; "
        "uses true sigma labels -> H_PI oracle only, never a learner input"
    )
    return mapping


def _derive_row(base: dict, arm: str, extra: dict) -> dict:
    row = dict(base)
    row["arm"] = arm
    row["derived_from"] = f"const_{base['posture']}"
    row["run_seconds"] = 0.0
    row.update(extra)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--roots", nargs="+", type=int, default=None)
    ap.add_argument("--campaigns", type=int, default=CAMPAIGNS_PER_HISTORY)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hard-cap-seconds", type=float, default=21_600.0)
    ap.add_argument(
        "--probe",
        action="store_true",
        help="smoke mode: calibration probes 7590901+, 1 cell, 4 campaigns",
    )
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.probe:
        roots = args.roots if args.roots is not None else PROBE_ROOTS
        cells = args.cells if args.cells != list(CELLS) else ["k090_s085"]
        campaigns = min(int(args.campaigns), 4)
        if any(not 7_590_901 <= root <= 7_590_920 for root in roots):
            raise SystemExit("probe mode only accepts roots 7590901-7590920")
    else:
        roots = args.roots if args.roots is not None else SCREEN_ROOTS
        cells = args.cells
        campaigns = int(args.campaigns)
    if any(root in HOLDOUT_ROOTS for root in roots):
        raise SystemExit("holdout roots 7590101-7590148 must stay untouched")
    unknown = [cell for cell in cells if cell not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells: {unknown}")

    scheduler = _count_scheduler()
    started = time.perf_counter()

    # ------------------------------------------------------------------
    # Pass 1: constant postures (direct DES), per (cell, root, campaign).
    # ------------------------------------------------------------------
    specs: list[dict] = []
    for cell in cells:
        kappa_r = CELLS[cell]["kappa_r"]
        s_r = CELLS[cell]["s_r"]
        for root in roots:
            sigmas = sigma_path(root, campaigns, kappa_r)
            demand_chain = demand_initial_regime_chain(root, campaigns)
            for campaign_index in range(campaigns):
                for posture in POSTURES:
                    specs.append(
                        {
                            "cell": cell,
                            "kappa_r": kappa_r,
                            "s_r": s_r,
                            "root": int(root),
                            "campaign_index": campaign_index,
                            "sigma": sigmas[campaign_index],
                            "initial_regime": demand_chain[campaign_index],
                            "posture": int(posture),
                            "scheduler": scheduler,
                        }
                    )
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        for index, row in enumerate(ex.map(task, specs), 1):
            rows.append(row)
            if index % 50 == 0:
                print(
                    f"  pass1 {index}/{len(specs)} "
                    f"({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
            if time.perf_counter() - started > args.hard_cap_seconds:
                raise TimeoutError("hard cap exceeded in pass 1")

    # ------------------------------------------------------------------
    # Self-check: CRN -- identical surge timeline across the 4 postures.
    # ------------------------------------------------------------------
    signatures: dict[tuple, tuple] = {}
    crn_violations: list[dict] = []
    for row in rows:
        key = (row["cell"], row["root"], row["campaign_index"])
        signature = _surge_signature(row)
        if key not in signatures:
            signatures[key] = signature
        elif signatures[key] != signature:
            crn_violations.append(
                {"key": list(key), "posture": row["posture"]}
            )
    if crn_violations:
        raise SystemExit(f"CRN violation: surge logs differ across arms: {crn_violations[:5]}")

    # ------------------------------------------------------------------
    # Derived arms: sigma_oracle (clairvoyant) + posterior_sigma (observable).
    # Both hold a constant posture per campaign, so their rows ARE the CRN-
    # identical constant rows (documented; no re-simulation).
    # ------------------------------------------------------------------
    oracle_mappings: dict[str, dict] = {}
    derived: list[dict] = []
    for cell in cells:
        cell_rows = [row for row in rows if row["cell"] == cell]
        mapping = _oracle_mapping(cell_rows)
        oracle_mappings[cell] = mapping
        by_key = {
            (row["root"], row["campaign_index"], row["posture"]): row
            for row in cell_rows
        }
        for root in roots:
            # per-campaign surge logs shared by all arms (CRN-verified above);
            # take them from posture 0.
            campaign_logs = [
                by_key[(root, campaign_index, 0)]["surge_log"]
                for campaign_index in range(campaigns)
            ]
            for campaign_index in range(campaigns):
                sigma_true = by_key[(root, campaign_index, 0)]["sigma"]
                oracle_posture = int(mapping[sigma_true]["posture"])
                derived.append(
                    _derive_row(
                        by_key[(root, campaign_index, oracle_posture)],
                        "sigma_oracle",
                        {
                            "chosen_posture": oracle_posture,
                            "clairvoyant": True,
                        },
                    )
                )
                posterior = beta_bernoulli_sigma_posterior(
                    campaign_logs[:campaign_index]
                )
                believed_sigma = "C" if posterior["p_sigma_c"] >= 0.5 else "H"
                posterior_posture = int(mapping[believed_sigma]["posture"])
                derived.append(
                    _derive_row(
                        by_key[(root, campaign_index, posterior_posture)],
                        "posterior_sigma",
                        {
                            "chosen_posture": posterior_posture,
                            "clairvoyant": False,
                            "p_sigma_c": posterior["p_sigma_c"],
                            "posterior_n_c": posterior["n_c"],
                            "posterior_n_h": posterior["n_h"],
                            "believed_sigma": believed_sigma,
                        },
                    )
                )
    rows.extend(derived)

    # ------------------------------------------------------------------
    # Self-check: baseline regression (risk-off + coupling-off == plain
    # ProgramOFullDESSimulation, same seed) -- run in-process, first root.
    # ------------------------------------------------------------------
    from supply_chain.program_o_full_des import ProgramOFullDESSimulation
    from supply_chain.q_r1_product_coupled import ProductCoupledProgramODES

    check_root = int(roots[0])
    check_regime = demand_initial_regime_chain(check_root, 1)[0]
    common = dict(
        seed=check_root * 100,
        calendar=(2,) * 8,
        scheduler=scheduler,
        regime_persistence=0.90,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
        initial_regime=check_regime,
    )
    sim_coupled_off = ProductCoupledProgramODES(
        product_coupling_enabled=False, **common
    ).run_contract()
    sim_plain = ProgramOFullDESSimulation(**common).run_contract()
    metrics_coupled_off = direct_campaign_metrics(sim_coupled_off)
    metrics_plain = direct_campaign_metrics(sim_plain)
    baseline_check = {
        "root": check_root,
        "state_hash_equal": sim_coupled_off.aggregate_state_hash()
        == sim_plain.aggregate_state_hash(),
        "early_ret_equal": metrics_coupled_off["early_ret_complete_cohort"]
        == metrics_plain["early_ret_complete_cohort"],
        "early_ret_value": metrics_plain["early_ret_complete_cohort"],
    }
    if not (baseline_check["state_hash_equal"] and baseline_check["early_ret_equal"]):
        raise SystemExit(f"baseline regression self-check FAILED: {baseline_check}")

    # ------------------------------------------------------------------
    # Self-check: determinism -- rerun the first spec, expect identical row.
    # ------------------------------------------------------------------
    rerun = task(dict(specs[0]))
    original = rows[0]
    compare_keys = [
        OBJECTIVE, "early_ret_visible", "worst_product_fill", "ret_excel",
        "unresolved_orders", "lost_orders", "n_surges", "aggregate_state_hash",
    ]
    determinism_check = {
        "spec": {
            key: specs[0][key]
            for key in ("cell", "root", "campaign_index", "posture", "sigma")
        },
        "identical": all(rerun[key] == original[key] for key in compare_keys)
        and _surge_signature(rerun) == _surge_signature(original),
    }
    if not determinism_check["identical"]:
        raise SystemExit(f"determinism self-check FAILED: {determinism_check}")

    # ------------------------------------------------------------------
    # Self-check: empirical s_r per cell (constant arms only).
    # ------------------------------------------------------------------
    empirical_s_r: dict[str, dict] = {}
    for cell in cells:
        const_rows = [
            row for row in rows
            if row["cell"] == cell and row["arm"].startswith("const_")
            and row["posture"] == 0  # one arm; logs identical across postures
        ]
        n_surges = sum(row["n_surges"] for row in const_rows)
        n_favored = sum(row["n_surges_favored"] for row in const_rows)
        empirical_s_r[cell] = {
            "target_s_r": CELLS[cell]["s_r"],
            "n_surges": int(n_surges),
            "n_favored": int(n_favored),
            "empirical": (n_favored / n_surges) if n_surges else None,
        }

    # ------------------------------------------------------------------
    # Per-arm cost accounting + development summary (no gate claims).
    # ------------------------------------------------------------------
    direct_rows = [row for row in rows if row["derived_from"] is None]
    run_seconds = [row["run_seconds"] for row in direct_rows]
    cost = {
        "direct_runs": len(direct_rows),
        "mean_seconds_per_campaign_run": sum(run_seconds) / len(run_seconds),
        "max_seconds_per_campaign_run": max(run_seconds),
        "wall_seconds": time.perf_counter() - started,
    }
    summary: dict[str, dict] = {}
    for cell in cells:
        cell_summary: dict[str, float] = {}
        for arm in [f"const_{p}" for p in POSTURES] + ["sigma_oracle", "posterior_sigma"]:
            values = [
                row[OBJECTIVE] for row in rows
                if row["cell"] == cell and row["arm"] == arm
            ]
            cell_summary[arm] = sum(values) / len(values) if values else None
        best_const = max(cell_summary[f"const_{p}"] for p in POSTURES)
        cell_summary["best_constant_mean"] = best_const
        cell_summary["h_pi_hat_dev"] = cell_summary["sigma_oracle"] - best_const
        cell_summary["h_obs_hat_dev"] = cell_summary["posterior_sigma"] - best_const
        summary[cell] = cell_summary

    for row in rows:
        row.pop("scheduler", None)

    out = {
        "schema_version": "q_r1_product_coupled_screen_v1",
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "contracts/q_r1_product_coupled_execution_amendment_v1.json",
        "parent_contract": "contracts/q_r1_product_coupled_risk_door_v1.json",
        "probe_mode": bool(args.probe),
        "cells": {cell: CELLS[cell] for cell in cells},
        "cell_adjacency": CELL_ADJACENCY,
        "roots": [int(root) for root in roots],
        "campaigns_per_history": campaigns,
        "objective": OBJECTIVE,
        "oracle_mappings": oracle_mappings,
        "self_checks": {
            "baseline_regression": baseline_check,
            "crn_surge_logs_identical_across_arms": not crn_violations,
            "determinism": determinism_check,
            "empirical_s_r": empirical_s_r,
            "cost": cost,
        },
        "summary_development_only": summary,
        "rows": rows,
        "selection_performed": False,
        "learner_return_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(
        f"rows={len(rows)} direct={len(direct_rows)} "
        f"elapsed={out['elapsed_seconds']:.0f}s -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
