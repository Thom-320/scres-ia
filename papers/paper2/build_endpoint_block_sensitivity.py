#!/usr/bin/env python3
"""Endpoint x block sign-inversion sensitivity table (Paper 2, Conclusion S1).

Re-aggregates two ALREADY-FROZEN artifact sets against the SAME frozen incumbent, with no new
simulation and no new seeds:

  Block A -- corrective confirmation block, 16 tapes
             results/metric_audit/ret_metric_repair_confirmation_v1/result.json
  Block B -- Step-3 pooled block, 12 tapes
             results/step3_*/full/rows.json

The question the table answers: holding policy, comparator and physics fixed, does the SIGN of
the measured advantage depend on which member of the ReT endpoint family is scored, and on which
tape block it is scored over? Reported per risk family (R1r, R2r).

This is a re-aggregation of custodied artifacts, not a measurement: no episode is simulated here.
Every number carries the SHA-256 prefix of the file it came from. Missing fields fail loudly.

Usage:
  build_endpoint_block_sensitivity.py [--out DIR] [--resamples N]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BLOCK_A = ROOT / "results/metric_audit/ret_metric_repair_confirmation_v1/result.json"
BLOCK_A_CONTRACT = ROOT / "contracts/ret_metric_repair_confirmation_v1.json"
BLOCK_B_GLOB = "results/step3_*/full/rows.json"

# The frozen incumbents are read from Block A; Block B is scored against the same posture vector.
POSTURE_TO_ARM = "static_op3rmI{0}_op5rmI{1}_op9rationsI{2}".format

ENDPOINTS = [
    "ret_excel",
    "ret_excel_clipped_0_1",
    "ret_excel_full_ledger",
    "ret_excel_quantity_time_clipped_0_1",
    "ret_thesis",
    "flow_fill_rate",
    "delivered_rations",
]
MPC_ARM = "replay_mpc_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def need(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise SystemExit(f"MISSING FIELD {'/'.join(map(str, keys))}; available: "
                             f"{sorted(cur)[:20] if isinstance(cur, dict) else type(cur)}")
        cur = cur[k]
    return cur


def paired_bootstrap_ci95(deltas, resamples, seed):
    """Percentile CI95 over the paired per-tape differences. Deterministic: seed-addressed."""
    if not deltas:
        raise SystemExit("empty delta vector")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(0.975 * (resamples - 1))]
    return lo, hi


def load_block_b():
    rows = []
    files = sorted(ROOT.glob(BLOCK_B_GLOB))
    if not files:
        raise SystemExit(f"no Block-B rows matched {BLOCK_B_GLOB}")
    for p in files:
        rows.extend(json.loads(p.read_text()))
    src = ", ".join(f"{p.relative_to(ROOT)}@{sha(p)}" for p in files)
    return rows, src, files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper_prep/endpoint_block_inversion_v1")
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260824)
    a = ap.parse_args()

    block_a = json.loads(BLOCK_A.read_text())
    contract_a = json.loads(BLOCK_A_CONTRACT.read_text())
    src_a = f"{BLOCK_A.relative_to(ROOT)}@{sha(BLOCK_A)}"
    rows_b, src_b, files_b = load_block_b()

    out = {
        "schema_version": "endpoint_block_inversion_v1",
        "kind": "REAGGREGATION_OF_FROZEN_ARTIFACTS__NO_NEW_SIMULATION__NO_NEW_SEEDS",
        "contract_of_block_a": need(block_a, "contract_path"),
        "contract_sha256_of_block_a": need(block_a, "contract_sha256"),
        "sources": {"block_a": src_a, "block_b": src_b,
                    "block_a_contract": f"{BLOCK_A_CONTRACT.relative_to(ROOT)}@{sha(BLOCK_A_CONTRACT)}"},
        "bootstrap": {"resamples": a.resamples, "seed": a.bootstrap_seed,
                      "method": "paired per-tape percentile"},
        "families": {},
    }

    for fam, fam_obj in need(block_a, "families").items():
        posture = need(fam_obj, "frozen_incumbent")
        inc_arm = POSTURE_TO_ARM(*posture)
        fam_out = {
            "frozen_incumbent_posture": posture,
            "frozen_incumbent_arm": inc_arm,
            "block_a_verdict": need(fam_obj, "verdict"),
            "block_a": {},
            "block_b": {},
        }

        # ---- Block A: read the frozen per-endpoint contrasts straight out of the artifact -----
        comps = need(fam_obj, "comparisons")
        for ep in ENDPOINTS:
            if ep not in comps:
                continue
            c = comps[ep]
            fam_out["block_a"][ep] = {
                "delta_mean": need(c, "delta_mean"),
                "ci95": need(c, "ci95"),
                "n_positive": need(c, "n_positive"),
                "n_tapes": need(c, "n_tapes"),
            }

        # ---- Block B: recompute the same contrast from the Step-3 rows ------------------------
        sub = [r for r in rows_b if r["family"] == fam]
        if not sub:
            raise SystemExit(f"family {fam} absent from Block B")
        inc = {r["tape_seed"]: r for r in sub if r["arm"] == inc_arm}
        mpc = {r["tape_seed"]: r for r in sub if r["arm"] == MPC_ARM}
        if not inc:
            raise SystemExit(f"incumbent arm {inc_arm} absent from Block B for {fam}")
        if not mpc:
            raise SystemExit(f"arm {MPC_ARM} absent from Block B for {fam}")
        tapes = sorted(set(inc) & set(mpc))
        fam_out["block_b_tapes"] = tapes
        for ep in ENDPOINTS:
            d = [mpc[t][ep] - inc[t][ep] for t in tapes]
            lo, hi = paired_bootstrap_ci95(d, a.resamples, a.bootstrap_seed)
            fam_out["block_b"][ep] = {
                "delta_mean": sum(d) / len(d),
                "ci95": [lo, hi],
                "n_positive": sum(1 for x in d if x > 0),
                "n_tapes": len(d),
            }

        # ---- Block B control: the same contrast against the best static WITHIN the block ------
        statics = sorted({r["arm"] for r in sub if r["arm"].startswith("static_")})
        fam_out["block_b_best_static_control"] = {}
        for ep in ["ret_excel_clipped_0_1", "ret_excel_full_ledger"]:
            best_arm, best_val = None, None
            for arm in statics:
                m = {r["tape_seed"]: r for r in sub if r["arm"] == arm}
                if not set(tapes) <= set(m):
                    continue
                v = sum(m[t][ep] for t in tapes) / len(tapes)
                if best_val is None or v > best_val:
                    best_arm, best_val = arm, v
            m = {r["tape_seed"]: r for r in sub if r["arm"] == best_arm}
            d = [mpc[t][ep] - m[t][ep] for t in tapes]
            fam_out["block_b_best_static_control"][ep] = {
                "best_static_arm": best_arm,
                "delta_mean": sum(d) / len(d),
                "n_positive": sum(1 for x in d if x > 0),
                "n_tapes": len(d),
            }

        # ---- blocks must be disjoint for the cross-block reading to mean anything -------------
        roots_a = sorted(need(contract_a, "roots")[fam])
        fam_out["block_a_tapes"] = roots_a
        fam_out["blocks_disjoint"] = not (set(roots_a) & set(tapes))

        # ---- the sharpest form of the finding: WITHIN one block, two members of the ReT family
        # disagree in sign on the same tapes, same policy pair. This reading needs no assumption
        # that the two blocks are comparable.
        wb = {}
        for label, table in (("block_a", fam_out["block_a"]), ("block_b", fam_out["block_b"])):
            pairs = []
            eps = [e for e in ENDPOINTS if e.startswith("ret_")]
            for i, e1 in enumerate(eps):
                for e2 in eps[i + 1:]:
                    if e1 in table and e2 in table and table[e1]["delta_mean"] * table[e2]["delta_mean"] < 0:
                        pairs.append([e1, e2])
            wb[label] = pairs
        fam_out["within_block_endpoint_sign_disagreements"] = wb

        # ---- the finding itself: which endpoints invert sign across blocks --------------------
        inversions = []
        for ep in ENDPOINTS:
            if ep not in fam_out["block_a"]:
                continue
            sa = fam_out["block_a"][ep]["delta_mean"]
            sb = fam_out["block_b"][ep]["delta_mean"]
            if sa * sb < 0:
                inversions.append(ep)
        fam_out["sign_inversions_across_blocks"] = inversions
        out["families"][fam] = fam_out

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "endpoint_block_inversion_v1.json").write_text(json.dumps(out, indent=1) + "\n")

    # ---- markdown ------------------------------------------------------------------------------
    md = ["# Endpoint x block sign-inversion sensitivity (machine-generated; do not edit)",
          "",
          f"Re-aggregation of frozen artifacts; no new simulation, no new seeds. "
          f"Paired percentile bootstrap, {a.resamples} resamples, seed {a.bootstrap_seed}.",
          "",
          f"- Block A source: `{src_a}`",
          f"- Block B sources: {len(files_b)} files, `results/step3_*/full/rows.json`",
          ""]
    for fam, f in out["families"].items():
        md += [f"## Family {fam} — frozen incumbent `{f['frozen_incumbent_arm']}` "
               f"(posture {f['frozen_incumbent_posture']})", "",
               "| Endpoint | Block A (corrective, n=16) | Block B (Step-3, n=12) | Sign |",
               "|---|---|---|---|"]
        for ep in ENDPOINTS:
            if ep not in f["block_a"]:
                continue
            A, B = f["block_a"][ep], f["block_b"][ep]
            flip = "**INVERTS**" if A["delta_mean"] * B["delta_mean"] < 0 else "stable"
            md.append(
                f"| `{ep}` | {A['delta_mean']:+.7f} "
                f"[{A['ci95'][0]:+.7f}, {A['ci95'][1]:+.7f}] {A['n_positive']}/{A['n_tapes']} "
                f"| {B['delta_mean']:+.7f} "
                f"[{B['ci95'][0]:+.7f}, {B['ci95'][1]:+.7f}] {B['n_positive']}/{B['n_tapes']} "
                f"| {flip} |")
        md.append("")
        md.append("Control — Block B against the best static *within the block* "
                  "(not the frozen incumbent):")
        md.append("")
        md.append("| Endpoint | best static in block | delta mean | favorable |")
        md.append("|---|---|---|---|")
        for ep, c in f["block_b_best_static_control"].items():
            md.append(f"| `{ep}` | `{c['best_static_arm']}` | {c['delta_mean']:+.7f} "
                      f"| {c['n_positive']}/{c['n_tapes']} |")
        md.append("")
        md.append(f"Sign inversions across blocks: "
                  f"{', '.join('`'+e+'`' for e in f['sign_inversions_across_blocks']) or 'none'}")
        md.append("")
        md.append(f"Tape blocks disjoint: **{f['blocks_disjoint']}** "
                  f"(A: {f['block_a_tapes'][0]}-{f['block_a_tapes'][-1]}, "
                  f"B: {f['block_b_tapes'][0]}-{f['block_b_tapes'][-1]}).")
        md.append("")
        for label in ("block_a", "block_b"):
            pairs = f["within_block_endpoint_sign_disagreements"][label]
            txt = "; ".join(f"`{x}` vs `{y}`" for x, y in pairs) or "none"
            md.append(f"Within-{label.replace('_', ' ')} ReT-family sign disagreements "
                      f"(same tapes, same policy pair): {txt}")
        md.append("")
    (a.out / "endpoint_block_inversion_v1.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
