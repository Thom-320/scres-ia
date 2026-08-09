#!/usr/bin/env python3
"""Is a changed source file behaviourally inert on the paths its pins attest?

WHY THIS EXISTS. `scripts/reattest_source_pins.py` refuses to move a pin without a `--cause`
naming the proof that the change preserves behaviour, and it is right to: a pin is provenance, and
moving one without a proven cause is falsifying provenance rather than maintenance. On 2026-07-31
that proof was produced by hand in a throwaway worktree. The drift recurred, so the proof is now a
script -- and a script that fails when the change is NOT inert.

WHAT IT DOES. Runs the SAME episodes under two trees -- the tree the pin attests, checked out in a
git worktree, and the working tree -- and compares every moment exactly. Same seeds, same
configurations, same entry point (`arm_runner.episode_moments`), one subprocess per tree so each
imports its own `supply_chain` package and neither can shadow the other.

WHAT WOULD MAKE IT FAIL, which is the only interesting question about a check like this:

  * any moment differing by more than `--tolerance` (default 0.0 -- EXACT) on any configuration;
  * a configuration that raises under one tree and not the other;
  * a control that must differ and does not. The controls are the point: a harness that runs
    nothing, or runs the same tree twice, would report perfect agreement. `f3` therefore requires
    the two trees to be DIFFERENT files, and `f4` requires a deliberately perturbed configuration
    to produce a DIFFERENT answer under the same tree -- so the comparator is known to be able to
    see a difference at all.

WHAT IT DOES NOT DO. It does not certify that a gated feature is correct when switched ON, only
that leaving it at its default reproduces the attested tree. That is exactly what a pin claims, and
claiming more from it would be the over-reading the pin exists to prevent.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
OUT = Path("results/source_pin_inertness/result.json")

#: Configurations that already existed before the change under test. Deliberately spans the
#: thesis-native default, both risk families, a second shift and a buffer posture, because a change
#: gated OFF by default is inert on the default path by construction and proving only that would be
#: proving nothing.
CONFIGURATIONS: list[dict] = [
    {"label": "thesis_default", "kwargs": {}},
    {"label": "S2", "kwargs": {"shifts": 2}},
    {"label": "S3", "kwargs": {"shifts": 3}},
    {"label": "risks_current", "kwargs": {"risks_enabled": True, "risk_level": "current"}},
    {"label": "risks_increased", "kwargs": {"risks_enabled": True, "risk_level": "increased"}},
    {"label": "risks_current_S2", "kwargs": {"risks_enabled": True, "risk_level": "current",
                                             "shifts": 2}},
    {"label": "buffered", "kwargs": {"initial_buffers": {"rations_al": 20000.0},
                                     "inventory_replenishment_period": 168.0}},
]
SEEDS = [8600001, 8600002, 8600003]

#: Runs inside each tree, with that tree first on `sys.path`. Kept as source text rather than an
#: importable module so the pinned tree -- which has never heard of this script -- can run it.
PROBE = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from supply_chain.arm_runner import episode_moments
from supply_chain.supply_chain import MFSCSimulation
spec = json.loads(sys.argv[2])
out = []
for cfg in spec["configurations"]:
    for seed in spec["seeds"]:
        try:
            sim = MFSCSimulation(seed=seed, horizon=spec["horizon"], **cfg["kwargs"])
            sim.run()
            row = {k: float(v) for k, v in episode_moments(sim).items()}
            out.append({"label": cfg["label"], "seed": seed, "moments": row, "error": None})
        except Exception as exc:                                   # noqa: BLE001
            out.append({"label": cfg["label"], "seed": seed, "moments": None,
                        "error": f"{type(exc).__name__}: {exc}"})
print(json.dumps(out))
'''


def run_probe(tree: Path, spec: dict) -> list[dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(PROBE)
        probe = fh.name
    done = subprocess.run([sys.executable, probe, str(tree), json.dumps(spec)],
                          capture_output=True, text=True, cwd=str(tree))
    if done.returncode != 0:
        raise SystemExit(f"probe failed in {tree}:\n{done.stderr[-3000:]}")
    return json.loads(done.stdout)


def digest(path: Path) -> str | None:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def compare(a: list[dict], b: list[dict], tolerance: float) -> list[dict]:
    diffs = []
    for old, new in zip(a, b):
        key = f"{old['label']}::{old['seed']}"
        if (old["error"] is None) != (new["error"] is None):
            diffs.append({"cell": key, "moment": None, "pinned": old["error"],
                          "current": new["error"], "kind": "raised_under_one_tree_only"})
            continue
        if old["error"] is not None:
            continue
        for moment, value in old["moments"].items():
            other = new["moments"].get(moment)
            if other is None or abs(float(other) - float(value)) > tolerance:
                diffs.append({"cell": key, "moment": moment, "pinned": value,
                              "current": other, "kind": "moment_differs"})
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pinned-commit", required=True,
                    help="the commit whose tree the stale pin attests")
    ap.add_argument("--source", default="supply_chain/supply_chain.py")
    ap.add_argument("--horizon", type=float, default=26 * 168.0,
                    help="hours per episode; short on purpose -- this is a provenance check")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="0.0 means EXACT; raise it only with a stated reason")
    ap.add_argument("--output", type=Path, default=ROOT / OUT)
    args = ap.parse_args()

    worktree = Path(tempfile.mkdtemp(prefix="pin-inertness-"))
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(worktree),
                    args.pinned_commit], cwd=ROOT, check=True)
    try:
        spec = {"configurations": CONFIGURATIONS, "seeds": SEEDS, "horizon": args.horizon}
        pinned = run_probe(worktree, spec)
        current = run_probe(ROOT, spec)
        diffs = compare(pinned, current, args.tolerance)

        # f4's control: the same tree, one configuration deliberately perturbed. If this does NOT
        # differ, the comparator cannot see a difference and every green above is meaningless.
        perturbed = run_probe(ROOT, {"configurations": [
            {"label": "control_perturbed", "kwargs": {"shifts": 3}}],
            "seeds": SEEDS[:1], "horizon": args.horizon})
        baseline = run_probe(ROOT, {"configurations": [
            {"label": "control_perturbed", "kwargs": {"shifts": 1}}],
            "seeds": SEEDS[:1], "horizon": args.horizon})
        control_differs = bool(compare(baseline, perturbed, 0.0))

        old_digest = digest(worktree / args.source)
        new_digest = digest(ROOT / args.source)
        cells = len(CONFIGURATIONS) * len(SEEDS)
        moments = sum(len(r["moments"]) for r in pinned if r["moments"])

        falsifiers = {
            "f1_every_cell_ran_under_both_trees": {
                "passed": all(r["error"] is None for r in pinned + current),
                "computed_from": {"cells": cells,
                                  "errors": [r["error"] for r in pinned + current
                                             if r["error"] is not None]}},
            "f2_no_moment_differs": {
                "passed": not diffs,
                "computed_from": {"n_differences": len(diffs), "tolerance": args.tolerance,
                                  "differences": diffs[:20]}},
            "f3_the_two_trees_are_actually_different_files": {
                "passed": old_digest is not None and new_digest is not None
                and old_digest != new_digest,
                "computed_from": {"pinned": old_digest, "current": new_digest},
                "why_this_can_fail": "comparing a tree with itself reports perfect agreement"},
            "f4_the_comparator_can_see_a_difference": {
                "passed": control_differs,
                "computed_from": {"control": "shifts 1 vs 3 on the current tree",
                                  "differs": control_differs},
                "why_this_can_fail": "a harness that runs nothing agrees with everything"},
        }
        ok = all(f["passed"] for f in falsifiers.values())
        payload = {
            "schema_version": "source_pin_inertness_v1",
            "claim_status": ("SOURCE_CHANGE_IS_BEHAVIOURALLY_INERT_ON_THE_TESTED_PATHS" if ok
                             else "SOURCE_CHANGE_IS_NOT_INERT__PIN_MUST_NOT_MOVE"),
            "source": args.source,
            "pinned_commit": args.pinned_commit,
            "pinned_sha256": old_digest,
            "current_sha256": new_digest,
            "configurations": [c["label"] for c in CONFIGURATIONS],
            "seeds": SEEDS,
            "horizon": args.horizon,
            "n_cells": cells,
            "n_moments_compared": moments,
            "tolerance": args.tolerance,
            "differences": diffs,
            "falsifiers": falsifiers,
            "scope": ("provenance check over DEFAULT-path behaviour; it does not certify a gated "
                      "feature switched ON, only that leaving it at its default reproduces the "
                      "attested tree"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"{payload['claim_status']}")
        print(f"  {cells} cells x {moments // max(cells, 1)} moments, {len(diffs)} differences")
        for name, f in falsifiers.items():
            print(f"  {'PASA' if f['passed'] else 'FALLA'}  {name}")
        print(f"  -> {args.output}")
        return 0 if ok else 1
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                       cwd=ROOT, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
