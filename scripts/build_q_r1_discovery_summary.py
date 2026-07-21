#!/usr/bin/env python3
"""Build the compact, hashed Q-R1 D0-D3 terminal discovery summary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "d0": ROOT / "results/q_r1/d0_cold_start_reanalysis_v1/result.json",
    "d1_raw": ROOT / "results/q_r1/d1_demand_memory_v1/result.json",
    "d1_adjudication": ROOT / "results/q_r1/d1_demand_memory_v1/adjudication.json",
    "d2_raw": ROOT / "results/q_r1/d2_risk_memory_bound_v1/result.json",
    "d2_adjudication": ROOT / "results/q_r1/d2_risk_memory_bound_v1/adjudication.json",
    "d3_bound": ROOT / "results/q_r1/d3_residual_bound_v1/result.json",
    "d3_structured": ROOT / "results/q_r1/d3_structured_calibration_v1/result.json",
    "cvar_secondary": ROOT / "results/q_r1/cvar_secondary_instrument_audit_v1/result.json",
}


def load(name: str) -> dict:
    return json.loads(FILES[name].read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    d0=load("d0"); d1=load("d1_adjudication")["summary"]; d2=load("d2_adjudication")["summary"]
    d3=load("d3_bound"); structured=load("d3_structured")
    d0_90=d0["contrasts"]["0.9"]["retained_posterior"]
    d2_90=d2["by_family_persistence"]["R24:persistent_0p90"]
    payload={
        "schema_version":"q_r1_discovery_terminal_summary_v1","claim_status":"EXPLORATORY_NO_CLAIM",
        "created_at":datetime.now(timezone.utc).isoformat(),
        "north_star":"retained decision knowledge causally improves canonical cold-start ReT after complete physical reset",
        "historical_verdicts_preserved":True,
        "canonical_resilience_is_primary":True,
        "d0_binary_context_early_signal":{
            "mean_early_ret_delta":d0_90["mean_early_ret_delta"],"lcb95":d0_90["early_ret_lcb95_history_clustered"],
            "favorable_fraction":d0_90["favorable_fraction"],"worst_product_delta":d0_90["mean_worst_product_delta"],
            "boundary":"new early endpoint only; original R0 STOP remains unchanged",
        },
        "d1_demand_parameter_verdict":d1["verdict"],
        "d1_p90_retained_mean":d1["by_persistence"]["persistent_0p90"]["arms"]["retained_exact_bayes_mpc"]["mean_early_ret_delta"],
        "d2_r24_verdict":d2["verdict"],
        "d2_r24_p90":{"mean":d2_90["known_level_minus_reset_mean_early_ret_2w"],"lcb95":d2_90["history_clustered_lcb95"],"action_divergence":d2_90["action_divergence"],"worst_product_delta":d2_90["mean_worst_product_delta"],"anti_shedding_pass":d2_90["anti_shedding_pass"]},
        "d3_verdict":d3["verdict"],"d3_prospective_all_modes":d3["pooled_all_modes_descriptive"],
        "d3_posthoc_persistent_signal":d3["primary_persistent"],
        "structured_h3_selected":structured["selected_universal_h3"],"structured_h3_scores":structured["scores"],
        "h8_preflight_verdict":structured["h8_preflight"]["verdict"],
        "learner_training_authorized":False,
        "next_route":"FRESH_BURNED_D3_PERSISTENT_REPLICATION_NO_LEARNER",
        "artifact_sha256":{name:sha(path) for name,path in FILES.items()},
    }
    output=ROOT/"results/q_r1/discovery_terminal_summary_v1.json"
    output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
