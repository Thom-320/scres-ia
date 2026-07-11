#!/usr/bin/env python3
"""Program D D1-v2 exact-prefix counterfactual branching gate."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.config import BACKORDER_QUEUE_CAP, HOURS_PER_DAY  # noqa:E402
from supply_chain.l_program_env import CampaignTape, GarridoLearningEnv  # noqa:E402
from supply_chain.program_d import PROXY_PATH, RULES, exogenous_hash, paired_bootstrap  # noqa:E402
from supply_chain.ret_thesis import compute_order_level_ret_excel_formula  # noqa:E402

FRONTIER_DIR=Path("results/program_d/d1_v2_frontier")
DEFAULT_OUTPUT=Path("results/program_d/d1_branching")
H72=72.0; H28=28.0*HOURS_PER_DAY


def load_inputs() -> tuple[str,list[CampaignTape]]:
    verdict=json.loads((FRONTIER_DIR/"verdict.json").read_text())
    if not verdict.get("promoted_to_branching"):
        raise RuntimeError("Frontier did not promote; branching is forbidden.")
    tapes=[]
    for split in ("selection","validation"):
        tapes += [CampaignTape.from_mapping(x) for x in json.loads((FRONTIER_DIR/f"{split}_tapes.json").read_text())]
    return str(verdict["best_admissible_constant"]),tapes


def make_env(tape:CampaignTape, rule:str) -> GarridoLearningEnv:
    env=GarridoLearningEnv(max_steps=tape.horizon_weeks,buffer_level=0)
    env.reset(seed=tape.base_seed,options={"campaign_tape":tape,"buffer_level":0,"initial_state_seed":tape.base_seed,"initial_shift":1})
    env.sim.set_backorder_priority_rule(rule)
    return env


def oldest(sim:Any)->float:
    return max([float(sim.env.now)-float(o.OPTj) for o in sim.pending_backorders] or [0.0])


def features(sim:Any, treatment:float, prior_rule:str)->dict[str,float|str]:
    queue=list(sim.pending_backorders); sizes=np.asarray([float(o.remaining_qty) for o in queue] or [0.0]); ages=np.asarray([float(sim.env.now)-float(o.OPTj) for o in queue] or [0.0])
    recent_demand=sum(float(q) for t,q in sim.daily_demand if float(t)>=float(sim.env.now)-7*24)
    recent_delivery=sum(float(q) for t,q in sim.delivery_events if float(t)>=float(sim.env.now)-7*24)
    down=[int(sim.op_down_count.get(i,0)>0) for i in range(9,13)]
    return {"time_offset":float(sim.env.now-treatment),"sb_inventory":float(sim.rations_sb.level),"queue_qty":float(sim.pending_backorder_qty),"queue_count":float(len(queue)),"queue_occupancy":float(len(queue)/BACKORDER_QUEUE_CAP),"contingent_share":float(sum(bool(o.contingent) for o in queue)/max(len(queue),1)),"size_p25":float(np.quantile(sizes,.25)),"size_p50":float(np.quantile(sizes,.5)),"size_p75":float(np.quantile(sizes,.75)),"age_p25":float(np.quantile(ages,.25)),"age_p50":float(np.quantile(ages,.5)),"age_p75":float(np.quantile(ages,.75)),"oldest_age":float(max(ages)),"in_transit":float(sim._in_transit),"op9_down":down[0],"op10_down":down[1],"op11_down":down[2],"op12_down":down[3],"recent_demand":recent_demand,"recent_delivered":recent_delivery,"recent_fill":float(recent_delivery/max(recent_demand,1.0)),"prior_rule":prior_rule,"operational_day":float((sim.env.now-treatment)//24%7)}


def state_hash(sim:Any)->str:
    payload={"time":round(float(sim.env.now),9),"inventory":{k:round(float(v),9) for k,v in sim._inventory_detail().items()},"queue":[(int(o.j),round(float(o.OPTj),9),round(float(o.remaining_qty),9),bool(o.contingent),bool(o.lost)) for o in sim.pending_backorders],"orders":[(int(o.j),round(float(o.OPTj),9),None if o.OATj is None else round(float(o.OATj),9),bool(o.lost),round(float(o.remaining_qty),9)) for o in sim.orders]}
    return sha256(json.dumps(payload,sort_keys=True,separators=(",",":" )).encode()).hexdigest()


def collect_daily(tape:CampaignTape, comparator:str)->list[dict]:
    env=make_env(tape,comparator); sim=env.sim; start=float(env._treatment_start); rows=[]
    for day in range(tape.horizon_weeks*7-28):
        f=features(sim,start,comparator); f.update({"day":day,"state_sha256":state_hash(sim),"r24":float(f["contingent_share"]>0),"downstream":float(any(float(f[k])>0 for k in ("op9_down","op10_down","op11_down","op12_down")))})
        rows.append(f); sim.step(None,24.0)
    env.close(); return rows


def choose_two(rows:list[dict], stratum:str, used:set[int])->list[dict]:
    tests={"nominal":lambda r:r["queue_occupancy"]<.8 and r["oldest_age"]<336 and not r["r24"] and not r["downstream"],"high_occupancy":lambda r:r["queue_occupancy"]>=.8,"high_age":lambda r:r["oldest_age"]>=336,"r24":lambda r:r["r24"]>0,"downstream":lambda r:r["downstream"]>0}
    candidates=[r for r in rows if tests[stratum](r) and int(r["day"]) not in used]
    fallback=False
    if len(candidates)<2:
        fallback=True
        score={"nominal":lambda r:abs(r["queue_occupancy"]-.3)+r["r24"]+r["downstream"],"high_occupancy":lambda r:-r["queue_occupancy"],"high_age":lambda r:-r["oldest_age"],"r24":lambda r:-r["contingent_share"],"downstream":lambda r:-(r["downstream"]+sum(r[k] for k in ("op9_down","op10_down","op11_down","op12_down")))}[stratum]
        candidates=sorted([r for r in rows if int(r["day"]) not in used],key=score)
    if not candidates: candidates=rows
    idx=np.linspace(0,len(candidates)-1,min(2,len(candidates))).round().astype(int)
    picked=[]
    for i in idx:
        row=dict(candidates[int(i)]); row["stratum"]=stratum; row["fallback"]=fallback; used.add(int(row["day"])); picked.append(row)
    while len(picked)<2:
        row=dict(candidates[0]); row["stratum"]=stratum; row["fallback"]=True; picked.append(row)
    return picked


def cohort_metrics(sim:Any, cohort_ids:set[int], decision:float, horizon:float)->dict[str,float]:
    orders=[o for o in sim.orders if int(o.j) in cohort_ids and float(o.OPTj)<=horizon]
    ret=compute_order_level_ret_excel_formula(orders,current_time=horizon)["mean_ret_excel"] if orders else 0.0
    sl=0.0
    for o in orders:
        end=min(float(o.OATj),horizon) if o.OATj is not None else horizon
        sl += max(0.0,end-max(decision,float(o.OPTj)+float(o.LTj)))*float(o.quantity)
    served=[o for o in orders if o.OATj is not None and float(o.OATj)<=horizon]
    lost=[o for o in orders if bool(o.lost)]
    return {"ret_excel":float(ret),"service_loss_auc":float(sl),"lost_orders":float(len(lost)),"served_qty":float(sum(o.quantity for o in served)),"ct_mean":float(np.mean([o.CTj for o in served])) if served else 0.0,"terminal_open_orders":float(sum(o.OATj is None and not o.lost for o in orders)),"cohort_orders":float(len(orders))}


def run_branch(tape:CampaignTape,comparator:str,state:dict,rule:str)->list[dict]:
    env=make_env(tape,comparator); sim=env.sim; start=float(env._treatment_start); decision=start+float(state["day"])*24.0
    while sim.env.now<decision-1e-9: sim.step(None,min(24.0,decision-sim.env.now))
    if state_hash(sim)!=state["state_sha256"]: raise RuntimeError(f"FAIL_CLOSED prefix mismatch {tape.campaign_id} day={state['day']}")
    pre_exo=exogenous_hash(sim,start); open_ids={int(o.j) for o in sim.orders if o.OATj is None and not o.lost}
    sim.set_backorder_priority_rule(rule); sim.step(None,24.0); sim.set_backorder_priority_rule(comparator)
    snapshots=[]
    for horizon in (H72,H28):
        target=decision+horizon
        while sim.env.now<target-1e-9: sim.step(None,min(24.0,target-sim.env.now))
        cohort_ids=open_ids|{int(o.j) for o in sim.orders if decision<=float(o.OPTj)<=target}
        snapshots.append({"horizon_hours":horizon,**cohort_metrics(sim,cohort_ids,decision,target),"risk_sha256":exogenous_hash(sim,start)["risk_sha256"],"demand_sha256":exogenous_hash(sim,start)["demand_sha256"],"pre_risk_sha256":pre_exo["risk_sha256"],"pre_demand_sha256":pre_exo["demand_sha256"],"raw_residual":sim.flow_ledger()["raw_residual"],"ration_residual":sim.flow_ledger()["ration_residual"]})
    env.close(); return snapshots


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); ap.add_argument("--max-tapes",type=int,default=60); ap.add_argument("--n-boot",type=int,default=10000); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    comparator,tapes=load_inputs(); tapes=tapes[:args.max_tapes]; states=[]; branches=[]
    states_path=args.output_dir/"states.csv"; branches_path=args.output_dir/"branch_rows.csv"
    if states_path.exists() and branches_path.exists():
        with states_path.open(newline="",encoding="utf-8") as fh: states=list(csv.DictReader(fh))
        with branches_path.open(newline="",encoding="utf-8") as fh: branches=list(csv.DictReader(fh))
        if len(states)!=len(tapes)*10 or len(branches)!=len(states)*len(RULES)*2:
            raise RuntimeError("Existing branch artifact has unexpected dimensions.")
        print("[d1-branch] reusing completed branch rollouts; recomputing grouped inference",flush=True)
    else:
        for ti,tape in enumerate(tapes,1):
            daily=collect_daily(tape,comparator); used=set()
            for s in ("nominal","high_occupancy","high_age","r24","downstream"):
                for rank,row in enumerate(choose_two(daily,s,used)):
                    row.update({"state_id":f"{tape.campaign_id}:{s}:{rank}:{int(row['day'])}","tape_id":tape.campaign_id,"tape_sha256":tape.digest(),"seed":tape.base_seed,"family":tape.family,"risk_level":tape.risk_level}); states.append(row)
            print(f"[d1-branch] sampled {ti}/{len(tapes)}",flush=True)
        tape_map={t.campaign_id:t for t in tapes}
        for si,state in enumerate(states,1):
            reference={}
            for rule in RULES:
                for m in run_branch(tape_map[state["tape_id"]],comparator,state,rule):
                    row={**{k:v for k,v in state.items() if k not in ("state_sha256",)},"state_sha256":state["state_sha256"],"rule":rule,**m}; branches.append(row)
                    key=m["horizon_hours"]; ex=(m["risk_sha256"],m["demand_sha256"])
                    if key in reference and reference[key]!=ex: raise RuntimeError("FAIL_CLOSED exogenous branch mismatch")
                    reference[key]=ex
            if si%10==0: print(f"[d1-branch] {si}/{len(states)} states",flush=True)
    # Long-horizon oracle with frozen tie order.
    long=[r for r in branches if float(r["horizon_hours"])==H28]; optimal=[]
    for state_id in sorted({r["state_id"] for r in long}):
        candidates=[r for r in long if r["state_id"]==state_id]
        best=sorted(candidates,key=lambda r:(-float(r["ret_excel"]),float(r["service_loss_auc"]),float(r["lost_orders"]),RULES.index(r["rule"])))[0]
        base=next(r for r in candidates if r["rule"]==comparator)
        optimal.append({**best,"base_ret":base["ret_excel"],"base_sl":base["service_loss_auc"],"base_lost":base["lost_orders"]})
    counts=Counter(r["rule"] for r in optimal); n=len(optimal)
    # The experimental unit is the tape, not the sampled state. Average within
    # tape first, then bootstrap the 60 paired tape effects.
    tape_effects=[]
    for tape_id in sorted({r["tape_id"] for r in optimal}):
        rr=[r for r in optimal if r["tape_id"]==tape_id]
        tape_effects.append({"tape_id":tape_id,"ret":float(np.mean([float(r["ret_excel"])-float(r["base_ret"]) for r in rr])),"sl":float(np.mean([(float(r["base_sl"])-float(r["service_loss_auc"]))/max(abs(float(r["base_sl"])),1.0) for r in rr])),"lost":float(np.mean([(float(r["lost_orders"])-float(r["base_lost"]))/max(abs(float(r["base_lost"])),1.0) for r in rr]))})
    ret=paired_bootstrap([r["ret"] for r in tape_effects],seed=0xD130,n_boot=args.n_boot)
    sl_rel=paired_bootstrap([r["sl"] for r in tape_effects],seed=0xD131,n_boot=args.n_boot)
    lost_rel=paired_bootstrap([r["lost"] for r in tape_effects],seed=0xD132,n_boot=args.n_boot)
    shares={k:v/n for k,v in counts.items()}; action_pass=sum(v>=.15 for v in shares.values())>=2 and max(shares.values())<=.85
    oracle_pass=action_pass and sl_rel["mean"]>=.05 and sl_rel["ci95"][0]>0 and ret["ci95"][0]>=0 and lost_rel["ci95"][1]<=.02
    verdict={"kind":"program_d_d1_v2_branching","generated_at_utc":datetime.now(timezone.utc).isoformat(),"git_sha_input":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),"proxy_sha256":sha256(PROXY_PATH.read_bytes()).hexdigest(),"tape_sha256s":sorted({str(r["tape_sha256"]) for r in states}),"comparator":comparator,"inference_unit":"tape (state effects averaged within tape)","n_tapes":len(tapes),"n_states":len(states),"n_branch_rows":len(branches),"primary_endpoint":"ret_excel","guardrail_endpoints":["service_loss_auc","lost_orders","mass_conservation","exogenous_identity"],"optimal_rule_counts":dict(counts),"optimal_rule_shares":shares,"oracle_ret_excel_delta":ret,"oracle_service_loss_relative_reduction":sl_rel,"oracle_lost_relative_increase":lost_rel,"criteria":{"two_actions_15pct_and_none_85pct":action_pass,"service_loss_5pct_ci_positive":sl_rel["mean"]>=.05 and sl_rel["ci95"][0]>0,"ret_codirectional_ci_nonnegative":ret["ci95"][0]>=0,"lost_upper_ci_at_most_2pct":lost_rel["ci95"][1]<=.02},"virgin_tapes_opened":0,"ppo_trained":False,"runtime":{"python":platform.python_version(),"numpy":np.__version__},"promoted_to_observable_tree":bool(oracle_pass),"verdict":"PROMOTE_TO_OBSERVABLE_TREE" if oracle_pass else "STOP_NO_STATE_DEPENDENT_RATIONING_HEADROOM"}
    def write(path,rows):
        with path.open("w",newline="",encoding="utf-8") as fh: w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(args.output_dir/"states.csv",states); write(args.output_dir/"branch_rows.csv",branches); write(args.output_dir/"oracle_rows.csv",optimal); write(args.output_dir/"tape_effects.csv",tape_effects); (args.output_dir/"verdict.json").write_text(json.dumps(verdict,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(verdict,indent=2)); return 0 if oracle_pass else 2

if __name__=="__main__": raise SystemExit(main())
