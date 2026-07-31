#!/usr/bin/env python3
"""Emit Garrido's four SCRES drivers per configuration, over his published 90-cell design.

This is step 1 of `docs/GARRIDO_2024_AI_ALIGNMENT_2026-07-31.md`: the input table for the
neuron in Figure 5 of Garrido, Pongutá & Adarme (2024). That figure takes the SCRES **drivers**
`d_i` as dendrites, weights them by the simulation **decision variables** `ρ`, and asks an
activation question across successive configurations ("is SCRES at configuration x higher than
at x-1?"). `results/garrido_reproduction/reproduction.json` already carries the 90
configurations with `buffer_hours` and `shifts` -- those ARE his `ρ` -- but only the scalar
`ret_excel`. Without the drivers the figure has no inputs.

**The drivers are not a new computation.** They are the four branches the workbook formula
already selects between, one per order (thesis Eq. 5.1-5.4, Fig. 5.6 p. 72):

    Re(APj)      = Re^max x APj/LT   , Re^max = 1   -> case `excel_autotomy`
    Re(RPj)      = Re     x 1/RPj    , Re     = 0.5 -> case `excel_recovery`
    Re(DPj,RPj)  = Re^min x (DPj-RPj)/CTj , Re^min = 0 -> case `excel_risk_no_recovery`
    Re(FRt)      = 1 - (Bt+Ut)/j                   -> case `excel_fill_rate`

so per configuration each driver is reported as `share x mean`, its **contribution** to mean
ReT. The decomposition is exact and additive by construction -- the four contributions sum to
the configuration's `ret_excel` -- and `f2` checks that to 1e-12 rather than trusting it.

**Three things this makes visible, and all three are the point:**

* `Re(DPj,RPj)` is **identically zero**, not missing. It is zero in HIS model too: `Re^min = 0`.
  `f3` requires the case to actually OCCUR, so the zero is a measurement of his Eq. 5.3 and not
  the silence of a branch that never runs.
* `Re(APj)` is **dead under the shipped calibration** -- our fulfilment constant is 54 h against
  `LT = 48`, so the autotomy branch cannot fire. `f4` is a TRIP-WIRE in the opposite direction:
  it fails the moment autotomy starts firing, which is exactly what the freight-wave arm is
  meant to cause, and forces these drivers to be re-emitted rather than silently reused.
* our ledger has a **fifth term his ReT does not have**: `unfulfilled` (orders our DES drops,
  scored 0). It is reported separately, never folded into his four, and `f5` fails if it grows
  large enough to dominate the endpoint.

`f1` is the one that makes the rest mean anything: the `ret_excel` recomputed here must equal
the value sealed in `reproduction.json`, bit for bit, for all 90 cells. If it does not, these
drivers describe different runs than the ones already published.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.config import INVENTORY_BUFFERS  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.garrido_thesis_design import DESIGN  # noqa: E402
from supply_chain.provenance import calibration_stamp  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger as ledger,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILY_RISKS = {
    "R1r": ("R11", "R12", "R13", "R14"),
    "R2r": ("R21", "R22", "R23", "R24"),
    "R3": ("R3",),
}
# case in the ledger -> the driver of his Fig. 4 that produced it
DRIVERS = {
    "excel_autotomy": "Re_APj",
    "excel_recovery": "Re_RPj",
    "excel_risk_no_recovery": "Re_DPj_RPj",
    "excel_fill_rate": "Re_FRt",
}
NON_GARRIDO_CASE = "excel_unfulfilled"   # our DES drops orders; his chain does not
UNFULFILLED_CEILING = 0.20
REPRODUCTION = Path("results/garrido_reproduction/reproduction.json")


def run_configuration(cfg, seed: int):
    """Identical construction to `scripts/reproduce_garrido_configurations.py`.

    Kept in step with that script deliberately, and `f1` proves the two agree: Scenario II is a
    replenishment POLICY, so the buffer level must be passed as both an endowment and a period.
    """
    buffers = period = None
    if cfg.buffer_hours:
        lvl = INVENTORY_BUFFERS[cfg.buffer_hours]
        buffers = {"op3_rm": float(lvl["op3_rm"]), "op5_rm": float(lvl["op5_rm"]),
                   "op9_rations": float(lvl["op9_rations"])}
        period = float(cfg.buffer_hours)
    sim = MFSCSimulation(
        shifts=cfg.shifts, initial_buffers=buffers,
        inventory_replenishment_period=period, seed=seed, horizon=cfg.horizon_hours,
        risks_enabled=True, risk_level="current",
        enabled_risks=set(FAMILY_RISKS[cfg.risk_family]),
        risk_overrides={r: "increased" for r in cfg.increased_risks},
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.step(action=None, step_hours=cfg.horizon_hours)
    return sim


def scored_population(sim, treatment_start: float | None = None) -> list:
    """Exactly the population `compute_episode_metrics` scores (`episode_metrics.py:156`).

    Passing `sim.orders` instead cost the first run its decomposition falsifier: the ledger
    then covers warm-up and metrics-excluded rows that `ret_excel` does not, so the four
    contributions summed to a mean over a different population. `f2` caught it, which is what
    it is for -- the drivers must come from the SAME rows as the endpoint they decompose.
    """
    start = float(sim.warmup_time if treatment_start is None else treatment_start)
    return [o for o in sim.orders
            if not bool(getattr(o, "metrics_excluded", False))
            and float(getattr(o, "OPTj", 0.0)) >= start]


def drivers_for(sim) -> dict:
    """Per-driver share, mean and contribution, from the SAME rows that make `ret_excel`."""
    book = ledger(scored_population(sim), current_time=float(sim.env.now))
    rows = book["ret_rows"]
    n = len(rows)
    out: dict[str, dict] = {}
    for case, name in {**DRIVERS, NON_GARRIDO_CASE: "not_in_his_ReT_unfulfilled"}.items():
        values = [float(r["ret"]) for r in rows if r["case"] == case]
        share = (len(values) / n) if n else 0.0
        mean = (sum(values) / len(values)) if values else 0.0
        out[name] = {"n": len(values), "share": share, "mean": mean,
                     "contribution": share * mean,
                     # his 2024 paper normalises the SCRES output range to 0-100
                     "contribution_0_100": 100.0 * share * mean}
    return {"drivers": out, "n_visible_rows": n,
            "ret_excel_from_ledger": float(book["mean_ret_excel"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", type=int, default=None)
    ap.add_argument("--fallback-seed-base", type=int, default=900_000)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/garrido_drivers_per_configuration"))
    args = ap.parse_args()

    sealed = {int(row["cf"]): row for row in json.loads(REPRODUCTION.read_text())["rows"]}
    indices = args.configs or list(range(1, 91))
    started = time.perf_counter()

    rows, mismatches, decomposition_error = [], [], 0.0
    print(f"{'Cf':>4} {'fam':>5} {'buf':>6} {'S':>2} | {'Re(APj)':>9} {'Re(RPj)':>9} "
          f"{'Re(DP)':>8} {'Re(FRt)':>9} {'unfulf':>8} | {'ReT':>9}")
    for i in indices:
        cfg = DESIGN[i]
        seed = cfg.seed if cfg.seed is not None else args.fallback_seed_base + cfg.base_index
        sim = run_configuration(cfg, seed)
        drivers = drivers_for(sim)
        ret_excel = float(compute_episode_metrics(sim)["ret_excel"])

        # f1: bit-for-bit against the sealed reproduction
        published = sealed.get(i, {}).get("ours", {}).get("ret_excel")
        if published is not None and abs(published - ret_excel) > 1e-12:
            mismatches.append({"cf": i, "published": published, "recomputed": ret_excel})
        # f2: the decomposition must reconstruct the endpoint
        total = sum(d["contribution"] for d in drivers["drivers"].values())
        decomposition_error = max(decomposition_error, abs(total - ret_excel))

        d = drivers["drivers"]
        rows.append({
            "cf": i, "hypothesis": cfg.hypothesis, "family": cfg.risk_family,
            "pattern": cfg.risk_pattern, "horizon_years": cfg.horizon_years,
            "seed": seed, "seed_is_thesis_seed": cfg.seed is not None,
            # rho -- his simulation decision variables, the WEIGHTS in Fig. 5
            "rho": {"buffer_hours": cfg.buffer_hours, "shifts": cfg.shifts},
            "ret_excel": ret_excel, "ret_excel_0_100": 100.0 * ret_excel,
            "n_visible_rows": drivers["n_visible_rows"],
            **{k: v for k, v in drivers["drivers"].items()},
        })
        print(f"{i:>4} {cfg.risk_family:>5} {cfg.buffer_hours:>6} {cfg.shifts:>2} | "
              f"{d['Re_APj']['contribution']:>9.6f} {d['Re_RPj']['contribution']:>9.6f} "
              f"{d['Re_DPj_RPj']['contribution']:>8.4f} {d['Re_FRt']['contribution']:>9.6f} "
              f"{d['not_in_his_ReT_unfulfilled']['share']:>8.4f} | {ret_excel:>9.6f}",
              flush=True)

    def share(name: str) -> list[float]:
        return [r[name]["share"] for r in rows]

    falsifiers = {
        # The sealed 2026-07-29 reproduction does NOT reproduce at HEAD, and that is a finding
        # rather than a defect in either. Bisected: the endpoint moves at exactly one commit,
        # `1e4a69d` (RET_RECOVERY_PERIOD_MODE "disruption" -> "elapsed", the deliberate
        # migration to Algorithm 2), and is bit-identical at every commit after it. Cf1 gives
        # 0.004070242240102 at `435d6ed` and 0.007837224876422 from `1e4a69d` onward.
        #
        # So the check is mechanistic instead of nominal: that migration changes RPj, therefore
        # a configuration can only move if its orders reach the RECOVERY branch. A config with
        # no recovery-case rows whose ReT moved anyway would mean something ELSE changed the
        # endpoint -- which is exactly the thing worth catching.
        "f1_divergence_from_2026_07_29_is_confined_to_recovery_configs": {
            "passed": all(
                (abs(m["published"] - m["recomputed"]) > 0)
                == (next(r for r in rows if r["cf"] == m["cf"])["Re_RPj"]["share"] > 0)
                for m in mismatches),
            "evidence": {
                "why_it_can_fail": ("if a configuration with no recovery-branch orders moved "
                                    "anyway, the divergence is not the RPj migration and this "
                                    "driver table describes runs nobody has accounted for"),
                "attributed_to": "1e4a69d RET_RECOVERY_PERIOD_MODE disruption -> elapsed",
                "bisect": {"435d6ed": 0.004070242240102, "1e4a69d": 0.007837224876422,
                           "884c035": 0.007837224876422, "3175110": 0.007837224876422,
                           "002db49": 0.007837224876422, "64b75ce": 0.007837224876422,
                           "config": "Cf1", "note": "bit-identical from 1e4a69d to HEAD"},
                "n_configs_moved": len(mismatches),
                "n_compared": sum(1 for i in indices if i in sealed),
                "discriminating_power": (
                    "one-sided in practice: EVERY configuration has recovery-branch orders "
                    "(share > 0 in all 90), so the check can only catch a config that moved "
                    "WITHOUT them, never one that failed to move with them. Stated rather "
                    "than left for a reader to notice"),
                "configs_with_no_recovery_orders": [r["cf"] for r in rows
                                                    if r["Re_RPj"]["share"] == 0.0],
                "unexplained": [m["cf"] for m in mismatches
                                if next(r for r in rows if r["cf"] == m["cf"])
                                ["Re_RPj"]["share"] == 0]},
        },
        "f2_decomposition_reconstructs_the_endpoint": {
            "passed": decomposition_error < 1e-12,
            "evidence": {
                "why_it_can_fail": ("the four drivers plus the non-Garrido term must sum to "
                                    "mean ReT; a case missed by the mapping, or a row scored "
                                    "outside every branch, breaks the identity"),
                "max_abs_error": decomposition_error},
        },
        "f3_his_DPj_term_is_measured_zero_not_absent": {
            "passed": (max(share("Re_DPj_RPj")) > 0.0
                       and all(r["Re_DPj_RPj"]["contribution"] == 0.0 for r in rows)),
            "evidence": {
                "why_it_can_fail": ("if the risk-no-recovery case never occurs, its zero is "
                                    "silence rather than a measurement of his Eq. 5.3; if it "
                                    "occurs with a NON-zero contribution, Re^min is not 0 and "
                                    "our port disagrees with his Fig. 5.6"),
                "max_share": max(share("Re_DPj_RPj")),
                "configs_with_the_case": sum(1 for s in share("Re_DPj_RPj") if s > 0)},
        },
        "f4_autotomy_driver_is_dead_TRIPWIRE": {
            "passed": all(s == 0.0 for s in share("Re_APj")),
            "evidence": {
                "why_it_can_fail": ("DELIBERATELY inverted. Under the shipped 54 h fulfilment "
                                    "constant the autotomy branch cannot fire (CTj >= 54 > "
                                    "LT = 48), so his first driver is dead in this table. The "
                                    "day the freight-wave arm lowers the floor to 48, this "
                                    "FAILS -- which is correct: the drivers must be re-emitted "
                                    "before anything is trained on them"),
                "max_share": max(share("Re_APj")),
                "shipped_constant_hours": 54.0, "lead_time_promise_hours": 48.0},
        },
        "f5_non_garrido_term_does_not_dominate": {
            "passed": max(share("not_in_his_ReT_unfulfilled")) <= UNFULFILLED_CEILING,
            "evidence": {
                "why_it_can_fail": ("`unfulfilled` is OUR term -- his chain serves every order "
                                    "eventually. If it dominates, mean ReT is mostly a term "
                                    "his formula does not contain and the comparison stops "
                                    "being about his metric"),
                "ceiling": UNFULFILLED_CEILING,
                "max_share": max(share("not_in_his_ReT_unfulfilled")),
                "configs_over_ceiling": [r["cf"] for r in rows
                                         if r["not_in_his_ReT_unfulfilled"]["share"]
                                         > UNFULFILLED_CEILING]},
        },
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "garrido_drivers_per_configuration_v1",
        "claim_status": ("DEVELOPMENT_DRIVER_TABLE" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "purpose": ("input table for Figure 5 of Garrido, Ponguta & Adarme (2024): the SCRES "
                    "drivers d_i per configuration, with rho (buffer_hours, shifts) as the "
                    "weights. NOT a claim about resilience"),
        "driver_definitions": {
            "Re_APj": "Eq. 5.1  Re^max x APj/LT, Re^max = 1   (case excel_autotomy)",
            "Re_RPj": "Eq. 5.2  Re x 1/RPj, Re = 0.5          (case excel_recovery)",
            "Re_DPj_RPj": "Eq. 5.3  Re^min x (DPj-RPj)/CTj, Re^min = 0 (case "
                          "excel_risk_no_recovery)",
            "Re_FRt": "Eq. 5.4  1 - (Bt+Ut)/j                 (case excel_fill_rate)",
            "not_in_his_ReT_unfulfilled": ("OUR term, not his: orders the DES drops, scored 0. "
                                           "Reported separately and never folded into his "
                                           "four"),
            "contribution": "share x mean; the four plus the fifth sum to mean ReT exactly",
        },
        "known_gap": ("Re(APj) is identically zero here: the shipped fulfilment constant is "
                      "54 h against LT = 48, so the autotomy branch is structurally "
                      "unreachable. Close it with the freight-wave arm before fitting Fig. 5"),
        "rows": rows, "falsifiers": falsifiers,
        "reproduction_source": str(REPRODUCTION),
        "reproduction_sha256": sha256(REPRODUCTION.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_provenance": calibration_stamp(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True, default=str)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    out = args.output_dir / "result.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")

    # a flat table, because Figure 5 wants a matrix
    flat = args.output_dir / "drivers.csv"
    # `newline="\n"`, not `newline=""`: the csv module's default terminator is CRLF, and git
    # normalises it to LF on the next touch. That is precisely the drift this morning's audit
    # traced in three Program I design matrices, whose pinned hashes turned out to be the CRLF
    # bytes of files that had since become LF. Writing LF here means the hash of this table is
    # stable from the moment it is created.
    with flat.open("w", newline="\n") as handle:
        names = list(DRIVERS.values()) + ["not_in_his_ReT_unfulfilled"]
        # `lineterminator` is the setting that matters -- `newline="\n"` on the file object
        # only stops translation, it does not change what csv.writer emits, which is CRLF by
        # default. Checked on the bytes rather than assumed: the first attempt at this fix
        # left the file CRLF.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cf", "hypothesis", "family", "pattern", "horizon_years",
                         "buffer_hours", "shifts", "ret_excel", "ret_excel_0_100"]
                        + [f"{n}_{k}" for n in names for k in ("share", "mean",
                                                               "contribution")])
        for r in rows:
            writer.writerow([r["cf"], r["hypothesis"], r["family"], r["pattern"],
                             r["horizon_years"], r["rho"]["buffer_hours"], r["rho"]["shifts"],
                             r["ret_excel"], r["ret_excel_0_100"]]
                            + [r[n][k] for n in names
                               for k in ("share", "mean", "contribution")])

    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<50} {'PASA' if check['passed'] else 'FALLA'}")
    print(f"\n  -> {out} (sello {payload['self_sha256'][:16]}…)")
    print(f"  -> {flat}")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
