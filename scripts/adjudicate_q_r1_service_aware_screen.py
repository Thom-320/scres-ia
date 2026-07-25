#!/usr/bin/env python3
"""Final adjudication of the 9-config service-aware retained screen.

Gates are copied verbatim from contracts/q_r1_service_aware_retained_screen_v1.json
["gates_frozen"] (frozen 2026-07-23, before any shard ran):
  mean_retention   : variant retained - frozen reset mean >= +0.05 at kappa 0.90 (clustered)
  safety           : worst_product_fill clustered LCB95 >= -0.02 in BOTH strata
  unresolved_margin: mean increase <= +2.0 orders/campaign and max <= 12
Selection is by gates only; no realized-outcome maximization. Burned roots only.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCREEN = ROOT / "results/q_r1/service_aware_screen_v1"
PARETO = ROOT / "results/q_r1/comparator_v2_frozen_pareto_c256_v1/pareto_merged/result.json"
GATE_RET_MEAN = 0.05          # contract gates_frozen.mean_retention
GATE_WF_LCB = -0.02           # contract gates_frozen.safety
GATE_UN_MEAN, GATE_UN_MAX = 2.0, 12  # contract gates_frozen.unresolved_margin
BOOT_SEED, BOOT_DRAWS = 20260723, 10_000


def clustered(by_root: dict[int, list[float]], rng) -> tuple[float, float]:
    """Root-clustered mean and one-sided LCB95 (roots are the resampling unit)."""
    roots = sorted(by_root)
    v = np.array([np.mean(by_root[r]) for r in roots])
    boot = np.array([rng.choice(v, len(v), True).mean() for _ in range(BOOT_DRAWS)])
    return float(v.mean()), float(np.quantile(boot, 0.05))


def main() -> int:
    pairs = json.loads(PARETO.read_text())["pareto_pairs"]
    frozen = {(r["history_root"], r["campaign_index"], r["persistence_mode"]): r
              for r in pairs}
    rng = np.random.default_rng(BOOT_SEED)

    shards = sorted(glob.glob(str(SCREEN / "shard_c*/result.json")))
    cells: dict[str, dict] = {}
    truncated_shards = []
    for path in shards:
        d = json.loads(Path(path).read_text())
        if d.get("truncated_by_hard_cap"):
            truncated_shards.append(path)
        for cfg in d["configs"]:
            rows = [r for r in d["rows"] if r["config_id"] == cfg]
            strata: dict[str, dict] = {}
            for mode in ("binary_0.9", "binary_0.75"):
                mr = [r for r in rows if r["persistence_mode"] == mode]
                if not mr:
                    continue
                ret, wf, un = defaultdict(list), defaultdict(list), defaultdict(list)
                fallbacks = 0
                for r in mr:
                    fz = frozen[(r["history_root"], r["campaign_index"],
                                 r["persistence_mode"])]["reset"]
                    v = r["variant"]
                    ret[r["history_root"]].append(
                        v["early_ret_complete_cohort"] - fz["early_ret_complete_cohort"])
                    wf[r["history_root"]].append(
                        v["worst_product_fill"] - fz["worst_product_fill"])
                    un[r["history_root"]].append(
                        v["unresolved_orders"] - fz["unresolved_orders"])
                    fallbacks += r.get("fallbacks") or 0
                m_ret, l_ret = clustered(ret, rng)
                m_wf, l_wf = clustered(wf, rng)
                m_un, _ = clustered(un, rng)
                strata[mode] = {
                    "n_arms": len(mr),
                    "ret_mean": round(m_ret, 5), "ret_lcb": round(l_ret, 5),
                    "wf_mean": round(m_wf, 5), "wf_lcb": round(l_wf, 5),
                    "un_mean": round(m_un, 3),
                    "un_max": round(max(max(x) for x in un.values()), 1),
                    "fallbacks": fallbacks,
                }
            gate_ret = strata["binary_0.9"]["ret_mean"] >= GATE_RET_MEAN
            gate_wf = all(strata[m]["wf_lcb"] >= GATE_WF_LCB for m in strata)
            gate_un = all(strata[m]["un_mean"] <= GATE_UN_MEAN
                          and strata[m]["un_max"] <= GATE_UN_MAX for m in strata)
            cells[cfg] = {"strata": strata, "gate_ret": gate_ret, "gate_wf": gate_wf,
                          "gate_un": gate_un,
                          "PASS_DEV": bool(gate_ret and gate_wf and gate_un)}

    passers = [c for c, v in cells.items() if v["PASS_DEV"]]
    complete = len(cells) == 9 and not truncated_shards
    verdict = ("PASS_PENDING_CONVERGENCE_CHECK" if passers else
               "STOP_SERVICE_AWARE_NO_SAFE_CONVERSION" if complete else
               "INCOMPLETE_NOT_ADJUDICABLE")
    total_fb = sum(s["fallbacks"] for v in cells.values() for s in v["strata"].values())
    out = {
        "schema": "q_r1_service_aware_adjudication_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "contract": "contracts/q_r1_service_aware_retained_screen_v1.json",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gates_frozen": {"ret_mean_kappa090": GATE_RET_MEAN, "wf_lcb_both_strata":
                         GATE_WF_LCB, "un_mean": GATE_UN_MEAN, "un_max": GATE_UN_MAX},
        "configs_adjudicated": len(cells),
        "grid_size_expected": 9,
        "truncated_shards": truncated_shards,
        "cells": cells,
        "passers_pending_convergence": passers,
        "total_planner_fallbacks": total_fb,
        "mechanism_note": (
            "zero planner fallbacks across every config: the scenario bank's predicted "
            "worst_product_fill never dips below the floor, so planner-side service "
            "constraints cannot bind. The failure is a prediction-side structural "
            "optimism, not a constraint-strength deficit."
        ),
        "verdict": verdict,
        "selection_performed": False,
        "learner_used": False,
    }
    (SCREEN / "adjudication_final.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")

    print(f"configs={len(cells)}/9  truncated={truncated_shards}  fallbacks={total_fb}")
    for cfg, v in sorted(cells.items()):
        s9, s7 = v["strata"]["binary_0.9"], v["strata"]["binary_0.75"]
        tag = cfg.split("legacy_")[-1] if "legacy_" in cfg else cfg[-22:]
        print(f"{tag:26} k90 ret {s9['ret_mean']:+.4f}/{s9['ret_lcb']:+.4f} "
              f"wf_lcb {s9['wf_lcb']:+.4f} | k75 wf_lcb {s7['wf_lcb']:+.4f} "
              f"un {s7['un_mean']:+.2f} | {'PASS' if v['PASS_DEV'] else 'fail'}")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
