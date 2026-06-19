"""Gate M0 — deterministic warm-up trigger time.

Thesis (Garrido-Rios 2017, Sec 6.8) says warm-up completes when the first
Q=5,000-ration batch arrives at Op9. The deterministic estimate is 838.8 h.

This probe verifies:
  1. With warmup_trigger='op9_arrival' (thesis-faithful), warmup_time ≈ 838.8 h.
  2. With warmup_trigger='production' (the engine DEFAULT), warmup_time ≈ 814.8 h
     (24h earlier — fires when the batch is produced, not when it reaches Op9).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from supply_chain.supply_chain import MFSCSimulation

EXPECTED_OP9_ARRIVAL = 838.8  # thesis deterministic estimate (Sec 6.8)
TOLERANCE_H = 5.0  # ±5 h tolerance for numerical jitter


def run(trigger: str) -> float:
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=2_000,  # only need to hit warm-up; 2000 h is plenty
        year_basis="thesis",
        deterministic_baseline=True,
        warmup_trigger=trigger,
    ).run()
    return float(sim.warmup_time)


def main() -> None:
    out = {"expected_op9_arrival_h": EXPECTED_OP9_ARRIVAL, "tolerance_h": TOLERANCE_H}
    print(f"Expected warm-up (thesis, op9_arrival): {EXPECTED_OP9_ARRIVAL} h ± {TOLERANCE_H} h")
    print()

    for trigger in ("op9_arrival", "production"):
        wt = run(trigger)
        delta_op9 = wt - EXPECTED_OP9_ARRIVAL if trigger == "op9_arrival" else None
        passed = (
            abs(wt - EXPECTED_OP9_ARRIVAL) <= TOLERANCE_H
            if trigger == "op9_arrival"
            else None
        )
        out[trigger] = {"warmup_time_h": wt}
        if delta_op9 is not None:
            out[trigger]["delta_vs_thesis_h"] = delta_op9
            out[trigger]["passed_gate_M0"] = bool(passed)
        print(
            f"warmup_trigger={trigger:<14} -> warmup_time = {wt:8.2f} h",
            end="",
        )
        if delta_op9 is not None:
            print(
                f"  (Δ vs thesis = {delta_op9:+.2f} h)  "
                f"{'PASS' if passed else 'FAIL'}"
            )
        else:
            # Show how much earlier 'production' fires vs op9_arrival
            out[trigger]["note"] = "Engine default; fires at AL output, not Op9 arrival."
            print("  (engine default; fires at AL output, before Op9 transport)")
    print()

    # Verdict
    op9_wt = out["op9_arrival"]["warmup_time_h"]
    if abs(op9_wt - EXPECTED_OP9_ARRIVAL) <= TOLERANCE_H:
        verdict = "PASS — thesis-faithful trigger reproduces 838.8 h warm-up."
    else:
        verdict = (
            f"FAIL — op9_arrival gave {op9_wt:.2f} h vs thesis 838.8 h. "
            "Possible misunderstanding of AL→Op8→Op9 transit, or PT mismatch."
        )
    out["verdict"] = verdict
    print(verdict)

    out_path = Path(__file__).resolve().parents[1] / "results" / "p0_warmup_det.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
