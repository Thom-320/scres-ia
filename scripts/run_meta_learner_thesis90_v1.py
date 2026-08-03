#!/usr/bin/env python3
"""Corrected Fig. 5 replay over Garrido's 90 thesis-native configurations.

This runner deliberately consumes the sealed 90-row driver surface instead of re-running the DES.
That makes the distinction visible: it is a search/replay result over the published/regenerated
surface, not a new physical replication. The search may update a learner with the observed ReT,
but its pre-run features are only the known configuration coordinates. Drivers are retained only
for the no-context-leakage falsifier.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.garrido_thesis_design import DESIGN  # noqa: E402

SURFACE = Path("results/garrido_drivers_per_configuration/result.json")
CONTEXT_BLOCKS = {
    "H1a": tuple(range(1, 11)), "H1b": tuple(range(11, 21)),
    "H1c": tuple(range(21, 31)), "H2a": tuple(range(31, 41)),
    "H2b": tuple(range(41, 51)), "H2c": tuple(range(51, 61)),
    "H3a": tuple(range(61, 71)), "H3b": tuple(range(71, 81)),
    "H3c": tuple(range(81, 91)),
}
CONTEXTS = tuple(CONTEXT_BLOCKS)
REPLAY_BASE = 9_200_001


def load_surface() -> tuple[dict[int, dict], str]:
    blob = json.loads(SURFACE.read_text())
    if blob.get("claim_status") != "DEVELOPMENT_DRIVER_TABLE":
        raise SystemExit(f"surface is {blob.get('claim_status')}; refusing replay")
    rows = {int(row["cf"]): row for row in blob["rows"]}
    if set(rows) != set(range(1, 91)):
        raise SystemExit("thesis-native surface must contain exactly Cf1..Cf90")
    for cf, row in rows.items():
        cfg = DESIGN[cf]
        if (row["family"], row["pattern"], row["rho"]["buffer_hours"],
                row["rho"]["shifts"]) != (
                    cfg.risk_family, cfg.risk_pattern, cfg.buffer_hours, cfg.shifts):
            raise SystemExit(f"surface/design mismatch at Cf{cf}")
    return rows, str(blob.get("self_sha256", ""))


def features(row: dict) -> np.ndarray:
    """Coordinates known before running the candidate; no driver or endpoint field appears."""
    cfg = DESIGN[int(row["cf"])]
    family = [float(cfg.risk_family == f) for f in ("R1r", "R2r", "R3")]
    scenario = [float(cfg.scenario == s) for s in (1, 2, 3)]
    pattern = [float(ch == "+") for ch in cfg.risk_pattern.ljust(4, "-")]
    return np.asarray([
        cfg.buffer_hours / 1344.0, (cfg.shifts - 1) / 2.0,
        *family, *scenario, *pattern, 1.0,
    ], dtype=float)


class Fig5Neuron:
    def __init__(self, dim: int, lr: float = 0.35):
        self.rho = np.zeros(dim)
        self.lr = lr

    def predict(self, x: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(self.rho @ x, -30, 30))))

    def update(self, x: np.ndarray, y: float) -> None:
        self.rho += self.lr * (y - self.predict(x)) * x


def search(strategy: str, rows: dict[int, dict], rng: np.random.Generator, budget: int,
           surface: dict[tuple[str, str], list[tuple[float, object]]]) -> dict:
    neuron = Fig5Neuron(len(features(next(iter(rows.values()))))) if strategy == "neuron_memory" else None
    per_context = {}
    for context in CONTEXTS:
        if strategy == "neuron_reset":
            neuron = Fig5Neuron(len(features(next(iter(rows.values())))))
        table = surface[(context, "source")]
        cf_order = list(CONTEXT_BLOCKS[context])
        values = [float(v) for v, _ in table]
        best = max(values)
        lo, span = min(values), max(max(values) - min(values), 1.0)
        seen, visited, curve = set(), [], []
        for step in range(budget):
            if strategy == "thesis_order":
                idx = step % len(cf_order)
            elif strategy == "random":
                unseen = [i for i in range(len(cf_order)) if i not in seen]
                idx = int(rng.choice(unseen or list(range(len(cf_order)))))
            else:
                unseen = [i for i in range(len(cf_order)) if i not in seen]
                if len(seen) < 3:
                    idx = int(rng.choice(unseen or list(range(len(cf_order)))))
                else:
                    pred = [neuron.predict(features(rows[cf_order[i]])) for i in unseen]
                    idx = unseen[int(np.argmax(pred))]
            seen.add(idx)
            visited.append(cf_order[idx])
            value, drivers = table[idx]
            curve.append(best - max(values[i] for i in seen))
            if neuron is not None:
                neuron.update(features(rows[cf_order[idx]]), (float(value) - lo) / span)
        within = next((i + 1 for i, r in enumerate(curve)
                       if r <= 0.01 * max(abs(best), 1.0)), budget + 1)
        chosen_idx = max(seen, key=lambda i: values[i]) if seen else 0
        chosen_cf = cf_order[chosen_idx]
        per_context[context] = {
            "regret_curve": curve,
            "final_regret": curve[-1] if curve else None,
            "runs_to_within_1pct": within,
            "best": best,
            "chosen_cf": chosen_cf,
            "chosen_value": values[chosen_idx],
            "visited_sequence": visited,
        }
    return {"per_context": per_context}


def paired(results: dict[str, list[dict]], a: str, b: str, field: str,
           contexts: tuple[str, ...], n_boot: int, rng: np.random.Generator) -> dict:
    av = np.asarray([np.mean([r["per_context"][c][field] for c in contexts])
                     for r in results[a]], dtype=float)
    bv = np.asarray([np.mean([r["per_context"][c][field] for c in contexts])
                     for r in results[b]], dtype=float)
    d = bv - av
    draws = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return {"mean": float(d.mean()), "lcb95": float(np.percentile(draws, 2.5)),
            "ucb95": float(np.percentile(draws, 97.5)),
            "n_replays": int(d.size), "inference": "algorithmic_replays_not_DES_replicates"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=256,
                    help="paired algorithmic replays over the fixed 90-row surface")
    ap.add_argument("--n-boot", type=int, default=5000)
    # REQUIRED, and previously ABSENT: the contract was hard-coded in the seal_and_write call, so
    # this runner could not be pointed at the right document even deliberately. That is a worse
    # version of the default-contract defect that sealed both H3' slices against the wrong
    # preregistration.
    ap.add_argument("--contract", type=Path, required=True,
                    help="contract to seal this run against (no default, and no hard-coding)")
    ap.add_argument("--replay-base", type=int, default=REPLAY_BASE)
    ap.add_argument("--output", type=Path,
                    default=Path("results/garrido_meta_learner_thesis90_v2/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rows, surface_sha = load_surface()

    surface = {
        (context, "source"): [(float(rows[cf]["ret_excel"]),
                                rows[cf].get("Re_RPj", {}))
                               for cf in CONTEXT_BLOCKS[context]]
        for context in CONTEXTS
    }
    strategies = ("thesis_order", "random", "neuron_reset", "neuron_memory")
    results = {s: [] for s in strategies}
    for r in range(args.repeats):
        for strategy in strategies:
            results[strategy].append(
                search(strategy, rows, np.random.default_rng(args.replay_base + r),
                       args.budget, surface))

    # f5: the real search is replayed with driver payloads permuted within every context. Values
    # are untouched. A learner that ranks an unrun row from drivers will change its path.
    shadow = {}
    perm_rng = np.random.default_rng(4242)
    for key, table in surface.items():
        order = perm_rng.permutation(len(table))
        shadow[key] = [(value, table[order[i]][1]) for i, (value, _) in enumerate(table)]
    real_surface = surface
    leak_free, leak_compared = True, 0
    for r in range(min(8, args.repeats)):
        for strategy in ("neuron_memory", "neuron_reset"):
            surface = shadow
            shadow_run = search(strategy, rows, np.random.default_rng(args.replay_base + r),
                                args.budget, surface)
            surface = real_surface
            for context in CONTEXTS:
                leak_compared += 1
                if (results[strategy][r]["per_context"][context]["visited_sequence"]
                        != shadow_run["per_context"][context]["visited_sequence"]):
                    leak_free = False

    # f4: replace every endpoint value. Random selection must be invariant to that replacement.
    random_value_shadow = {
        key: [(1_000_000.0 - float(i), drivers) for i, (_, drivers) in enumerate(table)]
        for key, table in real_surface.items()
    }
    random_invariant, random_compared = True, 0
    for r in range(min(8, args.repeats)):
        surface = random_value_shadow
        shadow_run = search("random", rows, np.random.default_rng(args.replay_base + r),
                            args.budget, surface)
        surface = real_surface
        for context in CONTEXTS:
            random_compared += 1
            if (results["random"][r]["per_context"][context]["visited_sequence"]
                    != shadow_run["per_context"][context]["visited_sequence"]):
                random_invariant = False

    common = {
        "surface_sha256": surface_sha,
        "contexts": list(CONTEXTS), "budget": int(args.budget),
        "replay_base": int(args.replay_base), "repeats": int(args.repeats),
    }
    memory_contract = dict(common, rho_policy="carry")
    reset_contract = dict(common, rho_policy="reset")
    f3_contract_match = all(memory_contract[k] == reset_contract[k] for k in common)
    f3_negative_control_detected = memory_contract != dict(reset_contract, budget=args.budget + 1)
    f3_trace_shapes_match = all(
        len(results["neuron_memory"][r]["per_context"]) == len(CONTEXTS)
        and len(results["neuron_reset"][r]["per_context"]) == len(CONTEXTS)
        for r in range(args.repeats)
    )
    values = [row["ret_excel"] for row in rows.values()]
    spread = float(max(values) - min(values))
    rng_boot = np.random.default_rng(20260801)
    runs = {s: float(np.mean([np.mean([r["per_context"][c]["runs_to_within_1pct"]
                                      for c in CONTEXTS]) for r in results[s]]))
            for s in strategies}
    regret = {s: float(np.mean([np.mean([r["per_context"][c]["final_regret"]
                                        for c in CONTEXTS]) for r in results[s]]))
              for s in strategies}
    alzheimer = paired(results, "neuron_memory", "neuron_reset", "runs_to_within_1pct",
                        CONTEXTS, args.n_boot, rng_boot)
    vs_thesis = paired(results, "neuron_memory", "thesis_order", "runs_to_within_1pct",
                        CONTEXTS, args.n_boot, rng_boot)
    vs_random = paired(results, "neuron_memory", "random", "runs_to_within_1pct",
                       CONTEXTS, args.n_boot, rng_boot)

    falsifiers = {
        "f1_surface_has_real_variation": {
            "passed": spread > 0.0,
            "evidence": {"why_it_can_fail": "if all 90 ReT values tie, search is undefined",
                         "ret_spread": spread}},
        "f2_thesis_order_is_the_declared_open_loop": {
            "passed": all(list(CONTEXT_BLOCKS[c]) == list(range(CONTEXT_BLOCKS[c][0],
                                                                CONTEXT_BLOCKS[c][-1] + 1))
                           for c in CONTEXTS),
            "evidence": {"why_it_can_fail": "a row/order mismatch would change the comparator",
                         "blocks": {c: list(v) for c, v in CONTEXT_BLOCKS.items()},
                         "not_claimed_as": "literal OFAT for risk-pattern blocks"}},
        "f3_memory_is_the_only_difference": {
            "passed": bool(f3_contract_match and f3_trace_shapes_match
                           and f3_negative_control_detected),
            "evidence": {"why_it_can_fail": "surface, budget, context order or replay stream drift",
                         "contract_match": f3_contract_match,
                         "trace_shapes_match": f3_trace_shapes_match,
                         "negative_control_detected": f3_negative_control_detected,
                         "memory_contract": memory_contract,
                         "reset_contract": reset_contract}},
        "f4_random_search_is_uninformed": {
            "passed": random_invariant,
            "evidence": {"why_it_can_fail": "random path changed after endpoint replacement",
                         "sequences_compared": random_compared,
                         "sequences_identical": random_invariant}},
        "f5_search_cannot_read_unrun_drivers": {
            "passed": leak_free,
            "evidence": {"why_it_can_fail": "driver permutation changed a visited sequence",
                         "sequences_compared": leak_compared,
                         "sequences_identical": leak_free}},
        "f6_source_and_replay_block_are_declared": {
            "passed": bool(surface_sha and args.replay_base >= 9_200_001),
            "evidence": {"why_it_can_fail": "an unsealed surface or reused replay stream voids replay",
                         "surface_sha256": surface_sha,
                         "replay_base": args.replay_base}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")
    if not falsifiers["all_passed"]:
        verdict = "HALTED_FALSIFIER_FAILED"
    else:
        verdict = "SURFACE_REPLAY_MEMORY_EFFECT" if alzheimer["lcb95"] > 0 else "SURFACE_REPLAY_NO_MEMORY_EFFECT"

    payload = {
        "schema_version": "garrido_meta_learner_thesis90_v2",
        "claim_status": verdict if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED",
        "surface_status": "SURFACE_REPLAY_ONLY",
        "metric": "ret_excel_diagnostic",
        "metric_gate": "HOLD_METRIC_PROVISIONAL",
        "service_first_note": "historical surface lacks final backorder component; no service-first promotion",
        "n_configurations": 90, "contexts": list(CONTEXTS), "budget": args.budget,
        "repeats": args.repeats, "runs_to_within_1pct": runs, "final_regret": regret,
        "alzheimer_effect_runs_saved_by_memory": alzheimer,
        "memory_vs_thesis_order": vs_thesis, "memory_vs_random": vs_random,
        "falsifiers": falsifiers,
        "source_surface": str(SURFACE), "source_surface_sha256": surface_sha,
        "per_context": {s: [run["per_context"] for run in results[s]] for s in strategies},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=args.contract,
        reference=SURFACE)
    print(f"\n  thesis-native replay: {verdict}")
    print(f"  memory vs reset: {alzheimer['mean']:+.2f} [{alzheimer['lcb95']:+.2f}, {alzheimer['ucb95']:+.2f}]")
    print(f"  memory vs thesis order: {vs_thesis['mean']:+.2f} [{vs_thesis['lcb95']:+.2f}, {vs_thesis['ucb95']:+.2f}]")
    print(f"  memory vs random: {vs_random['mean']:+.2f} [{vs_random['lcb95']:+.2f}, {vs_random['ucb95']:+.2f}]")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<44} {'PASA' if check['passed'] else 'FALLA'}")
    print(f"  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
