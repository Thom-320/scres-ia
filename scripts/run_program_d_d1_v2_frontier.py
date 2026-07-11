#!/usr/bin/env python3
"""Run the frozen D1-v2 constant frontier and validation gate."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.program_d import (PROXY_PATH, RULES, make_tapes, paired_bootstrap, run_constant)  # noqa:E402
from supply_chain.l_program_env import CampaignTape  # noqa:E402

DEFAULT_OUTPUT = Path("results/program_d/d1_v2_frontier")


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate_split(split: str, seed_start: int, out: Path) -> tuple[list[dict], list]:
    rows_path = out / f"{split}_rows.csv"
    tapes_path = out / f"{split}_tapes.json"
    if rows_path.exists() and tapes_path.exists():
        with rows_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        tapes = [CampaignTape.from_mapping(row) for row in json.loads(tapes_path.read_text(encoding="utf-8"))]
        if len(rows) == 30 * len(RULES) and len(tapes) == 30:
            print(f"[d1-v2] reusing frozen {split} artifacts", flush=True)
            return rows, tapes
    tapes = make_tapes(split, seed_start)
    rows: list[dict] = []
    for i, tape in enumerate(tapes, 1):
        reference_hashes = None
        for rule in RULES:
            m = run_constant(tape, rule)
            hashes = (m["risk_sha256"], m["demand_sha256"])
            if reference_hashes is None: reference_hashes = hashes
            if hashes != reference_hashes:
                raise RuntimeError(f"FAIL_CLOSED exogenous mismatch {tape.campaign_id} {rule}")
            rows.append({"split": split, "tape_id": tape.campaign_id, "tape_sha256": tape.digest(), "seed": tape.base_seed, "family": tape.family, "risk_level": tape.risk_level, "rule": rule, **m})
        print(f"[d1-v2] {split} {i}/30 {tape.campaign_id}", flush=True)
    write_csv(out / f"{split}_rows.csv", rows)
    (out / f"{split}_tapes.json").write_text(json.dumps([t.payload(include_hash=True) for t in tapes], indent=2), encoding="utf-8")
    return rows, tapes


def by_rule(rows: list[dict], field: str, rule: str) -> np.ndarray:
    return np.asarray([float(r[field]) for r in rows if r["rule"] == rule])


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); ap.add_argument("--n-boot", type=int, default=10000); args=ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection,_=evaluate_split("selection",810001,args.output_dir)
    spt_sl=np.mean(by_rule(selection,"service_loss_auc_ration_hours","spt_contingent")); spt_lost=np.mean(by_rule(selection,"lost_orders","spt_contingent"))
    frontier=[]
    for rule in RULES:
        ret=float(np.mean(by_rule(selection,"ret_excel",rule))); sl=float(np.mean(by_rule(selection,"service_loss_auc_ration_hours",rule))); lost=float(np.mean(by_rule(selection,"lost_orders",rule)))
        sl_deg=(sl-spt_sl)/max(abs(spt_sl),1e-12); lost_deg=(lost-spt_lost)/max(abs(spt_lost),1.0)
        frontier.append({"rule":rule,"ret_excel":ret,"service_loss_auc":sl,"lost_orders":lost,"service_loss_degradation_vs_spt":sl_deg,"lost_degradation_vs_spt":lost_deg,"admissible":bool(sl_deg<=.01 and lost_deg<=.02)})
    admissible=[r for r in frontier if r["admissible"]]
    if not admissible: raise RuntimeError("No admissible comparator; thesis SPT should always be admissible.")
    locked=max(admissible,key=lambda r:r["ret_excel"])["rule"]
    (args.output_dir/"selection_lock.json").write_text(json.dumps({"best_admissible_constant":locked,"frontier":frontier,"locked_before_validation":True},indent=2),encoding="utf-8")
    validation,_=evaluate_split("validation",811001,args.output_dir)
    def delta(field, better_high=True):
        a=by_rule(validation,field,locked); b=by_rule(validation,field,"spt_contingent")
        return a-b if better_high else b-a
    ret=paired_bootstrap(delta("ret_excel"),seed=0xD120,n_boot=args.n_boot)
    sl_abs=paired_bootstrap(delta("service_loss_auc_ration_hours",False),seed=0xD121,n_boot=args.n_boot)
    spt=by_rule(validation,"service_loss_auc_ration_hours","spt_contingent"); chosen=by_rule(validation,"service_loss_auc_ration_hours",locked)
    sl_rel=paired_bootstrap((spt-chosen)/np.maximum(np.abs(spt),1e-12),seed=0xD122,n_boot=args.n_boot)
    order_changed=any(abs(float(r["ret_excel"])-float(next(x["ret_excel"] for x in validation if x["tape_id"]==r["tape_id"] and x["rule"]=="spt_contingent")))>1e-12 for r in validation if r["rule"]==locked)
    conservation=max(abs(float(r["mass_balance_residual"])) for r in validation)<1e-6
    ret_pass=ret["mean"]>=.001 and ret["ci95"][0]>0
    sl_pass=sl_rel["mean"]>=.02 and sl_rel["ci95"][0]>0
    promote=bool(conservation and order_changed and (ret_pass or sl_pass))
    verdict={"kind":"program_d_d1_v2_frontier","generated_at_utc":datetime.now(timezone.utc).isoformat(),"git_sha_input":git_sha(),"proxy_sha256":__import__("hashlib").sha256(PROXY_PATH.read_bytes()).hexdigest(),"downstream_q_source":"figure_6_2_text_2400_2600","primary_metric":"ret_excel","selection_seeds":[810001,810030],"validation_seeds":[811001,811030],"best_admissible_constant":locked,"selection_frontier":frontier,"validation":{"ret_excel_delta":ret,"service_loss_absolute_reduction":sl_abs,"service_loss_relative_reduction":sl_rel,"mass_conservation_pass":conservation,"order_service_changed":order_changed,"ret_gate_pass":ret_pass,"service_gate_pass":sl_pass},"promoted_to_branching":promote,"verdict":"PROMOTE_TO_BRANCHING" if promote else "STOP_NO_VALIDATED_D1_AUTHORITY","runtime":{"python":platform.python_version(),"numpy":np.__version__}}
    (args.output_dir/"verdict.json").write_text(json.dumps(verdict,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(verdict,indent=2)); return 0 if promote else 2

if __name__=="__main__": raise SystemExit(main())
