#!/usr/bin/env python3
"""Burned H3 retained-MPC family calibration after a material D3 bound."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_t_full_des_mpc import FullDEST0Config  # noqa: E402
from supply_chain.q_r1_retained_learning import (  # noqa: E402
    PERSISTENCE_MODES, build_parameter_history, controller_prefix,
    evaluate_calendar, retained_belief_path,
)

CONFIGS = ("scenario", "robust", "constraint_aware")


def lcb(by_root: dict[int, list[float]], seed: int = 20260721) -> float:
    roots = sorted(by_root); means = np.asarray([np.mean(by_root[root]) for root in roots])
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(means), size=(5000, len(means)))
    return float(np.quantile(means[draws].mean(axis=1), .025))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results/q_r1/d1_demand_memory_v1/result.json")
    parser.add_argument("--histories", type=int, default=24)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=7_570_201)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--hard-cap-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/q_r1/d3_structured_calibration_v1/result.json")
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    source_rows = source["rows"]
    source_map = {(r["persistence_mode"],r["history_root"],r["campaign_index"],r["arm"]):r for r in source_rows}
    sched=scheduler(); rows=[]; started=time.perf_counter()
    for mode in PERSISTENCE_MODES:
        for offset in range(args.histories):
            history=build_parameter_history(history_root=args.seed_start+offset,campaigns=args.campaigns,persistence_mode=mode,scheduler=sched)
            beliefs=retained_belief_path(history)
            for campaign_index,campaign in enumerate(history):
                reset=source_map[(mode,campaign.history_root,campaign_index,"reset_exact_bayes_mpc")]
                continuation=tuple(reset["calendar"])
                scenario=source_map[(mode,campaign.history_root,campaign_index,"retained_exact_bayes_mpc")]
                rows.append({**{k:scenario[k] for k in scenario},"config":"scenario_h3_p4"})
                for planner_mode in ("robust","constraint_aware"):
                    if time.perf_counter()-started>args.hard_cap_seconds: raise TimeoutError("D3 calibration hard cap")
                    config=FullDEST0Config(horizon=3,mode=planner_mode,particles=args.particles,worst_product_floor=.70)
                    prefix,detail=controller_prefix(campaign=campaign,belief=beliefs[campaign_index],scheduler=sched,config=config,decisions=2)
                    calendar=tuple(prefix)+continuation[2:]
                    metrics=evaluate_calendar(campaign=campaign,calendar=calendar,scheduler=sched)
                    rows.append({
                        "persistence_mode":mode,"history_root":campaign.history_root,"campaign_index":campaign_index,
                        "theta_id":f"rho{int(campaign.theta[0]*100)}_share{int(campaign.theta[1]*100)}",
                        "config":f"{planner_mode}_h3_p4","calendar":list(calendar),"online_ms":detail.get("online_ms",0.0),**metrics,
                    })
                print(json.dumps({"mode":mode,"root":campaign.history_root,"campaign":campaign_index}),flush=True)
    persistent=[r for r in rows if r["persistence_mode"]!="iid" and r["campaign_index"]>0]
    roots=sorted({r["history_root"] for r in persistent})
    scores={config:np.mean([r["early_ret_2w"] for r in persistent if r["config"]==config]) for config in ("scenario_h3_p4","robust_h3_p4","constraint_aware_h3_p4")}
    selected=max(scores,key=lambda config:(scores[config],config))
    by_root=defaultdict(list); favorable=[]; fill=[]
    for root in roots:
        for mode in ("persistent_0p75","persistent_0p90"):
            for campaign in range(1,args.campaigns):
                chosen=next(r for r in rows if r["persistence_mode"]==mode and r["history_root"]==root and r["campaign_index"]==campaign and r["config"]==selected)
                baseline=source_map[(mode,root,campaign,"retained_exact_bayes_mpc")]
                delta=chosen["early_ret_2w"]-baseline["early_ret_2w"]
                by_root[root].append(delta); favorable.append(delta>0); fill.append(chosen["worst_product_fill"]-baseline["worst_product_fill"])
    deltas=[v for root in roots for v in by_root[root]]
    payload={
        "schema_version":"q_r1_d3_structured_calibration_v1","claim_status":"EXPLORATORY_NO_CLAIM",
        "created_at":datetime.now(timezone.utc).isoformat(),"source":str(args.source),"scores":scores,
        "selected_universal_h3":selected,"selected_minus_scenario_mean":float(np.mean(deltas)),
        "selected_minus_scenario_lcb95":lcb(by_root),"favorable_fraction":float(np.mean(favorable)),
        "mean_worst_product_delta":float(np.mean(fill)),
        "h8_preflight":{"one_decision_seconds":{"scenario":8.49,"robust":8.24,"constraint_aware":8.30},"projected_campaign_evaluations":1728,"verdict":"STOP_H8_COMPUTE_INFEASIBLE_FOR_BURNED_CALIBRATION"},
        "rows":rows,"elapsed_seconds":time.perf_counter()-started,"learner_authorized":False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in payload.items() if k!="rows"},indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
