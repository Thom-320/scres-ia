#!/usr/bin/env python3
"""Fase 1b: separate RETENTION from NEURAL CARRIER in the outer loop.

Amendment: `docs/ENMIENDA_NOMBRES_Y_LENGUAJE_PERMITIDO_2026-08-12.md`

The outer loop has been narrated as one result and is two:

  RETENTION       AUC(reset) - AUC(retained), per family.  Alive: 6/6 under simultaneous inference.
  NEURAL CARRIER  AUC(best classical retained carrier) - AUC(neuron_memory).  NEVER MEASURED.

The second is the one a neural-premium claim needs, and nobody computed it. What is visible in the
sealed ladder points the other way: `ucb1_transfer` has a BETTER point estimate than
`neuron_memory` on the ladder's own primary metric, and `search_ladder_v5` still seals
THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH.

Lower AUC is better, so every estimand here is signed so that POSITIVE means the neuron wins.

No seed is opened and no search is run: the per-seed AUC arrays are read from the sealed replay.
Selection of the best classical carrier happens INSIDE each bootstrap resample, so the advantage
is not inflated by picking the winner on the same data that scores it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402

CONTRACT = Path("docs/ENMIENDA_NOMBRES_Y_LENGUAJE_PERMITIDO_2026-08-12.md")
OUT = Path("results/program_n/outer_loop_carrier/result.json")
LADDER = Path("results/search_ladder_v5/result.json")

NEURAL_CARRIER = "neuron_memory"
#: Every carrier that retains state across contexts and is NOT a network. Read from the sealed
#: artifact's own `memory_arms` list, minus the neural one, so the pool cannot be curated here.
NEURAL_RESET = "neuron_reset"
N_BOOT = 20_000
BOOT_SEED = 20260812


def paired_boot(diffs_by_arm: dict[str, np.ndarray], rng) -> dict:
    """Best-of-class selected INSIDE each resample. Positive means the neuron wins."""
    arms = sorted(diffs_by_arm)
    matrix = np.vstack([diffs_by_arm[a] for a in arms])              # (arms, seeds)
    n = matrix.shape[1]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        # within the resample: pick the classical carrier that looks best, then score the neuron
        # against THAT one. Selecting outside the bootstrap is what inflates a winner's margin.
        resampled = matrix[:, idx[b]].mean(axis=1)
        draws[b] = resampled.min()
    point = float(matrix.mean(axis=1).min())
    return {"mean": point,
            "lcb95": float(np.quantile(draws, 0.025)),
            "ucb95": float(np.quantile(draws, 0.975)),
            "n_seeds": int(n), "n_boot": N_BOOT,
            "arms_in_pool": arms,
            "hardest_arm": arms[int(np.argmin(matrix.mean(axis=1)))]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", type=Path, default=LADDER)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    d = json.loads(args.ladder.read_text())
    per_arm = d["per_arm"]
    memory_arms = list(d["memory_arms"])
    classical_memory = [a for a in memory_arms if a != NEURAL_CARRIER]

    auc = {a: np.asarray(per_arm[a]["auc"], dtype=float) for a in per_arm if "auc" in per_arm[a]}
    neuron = auc[NEURAL_CARRIER]

    # POSITIVE = the neuron wins, because lower AUC is better.
    diffs = {a: auc[a] - neuron for a in classical_memory}
    carrier = paired_boot(diffs, np.random.default_rng(BOOT_SEED))

    # The retention contrast, recomputed here only so both estimands sit in one artifact.
    retention = {"mean": float((auc[NEURAL_RESET] - neuron).mean()),
                 "n_seeds": int(neuron.size),
                 "note": "neuron_reset - neuron_memory on the same tapes; positive means "
                         "retention helps. This is the estimand that survives 6/6 in "
                         "results/retention_simultaneous, and it is NOT a neural claim."}

    ranking = {a: float(auc[a].mean()) for a in sorted(auc, key=lambda k: auc[k].mean())}
    neuron_rank = list(ranking).index(NEURAL_CARRIER) + 1

    checks = {
        "h1_the_pool_comes_from_the_artifact": F.check(
            set(classical_memory) == set(memory_arms) - {NEURAL_CARRIER},
            "a hand-curated pool is how a comparator class gets quietly narrowed; this reads "
            "`memory_arms` from the sealed ladder and removes only the neural entry",
            computed_from={"n_memory_arms": len(memory_arms),
                           "n_classical": len(classical_memory)},
            pool=classical_memory),
        "h2_selection_happens_inside_the_bootstrap": F.check(
            carrier["n_boot"] == N_BOOT and len(carrier["arms_in_pool"]) > 1,
            "choosing the best classical carrier on the full sample and then testing against it "
            "on that same sample inflates the neuron's margin by the winner's curse",
            computed_from={"n_boot": N_BOOT, "n_arms": len(carrier["arms_in_pool"])}),
        "h3_the_neural_carrier_beats_its_class": F.check(
            carrier["lcb95"] > 0.0,
            "this is the estimand a neural premium in the outer loop requires, and the sealed "
            "ladder already shows ucb1_transfer ahead of neuron_memory on point estimate, so it "
            "can fail and is expected to",
            computed_from={"mean": carrier["mean"], "lcb95": carrier["lcb95"]}),
        "h4_retention_and_carrier_are_reported_apart": F.check(
            retention["mean"] != carrier["mean"],
            "if the two estimands were the same number, narrating them as one would have been "
            "harmless; they are not, and conflating them is what this artifact exists to stop",
            computed_from={"retention_mean": retention["mean"],
                           "carrier_mean": carrier["mean"]}),
    }
    checks["custody"] = {
        "passed": None, "not_applicable": True,
        "evidence": {"why_it_can_fail": "it cannot: the per-seed AUC arrays are read from a sealed "
                                        "declared replay. No seed is opened and no search runs.",
                     "seeds_opened": 0, "searches_run": 0, "source": str(args.ladder)}}
    summary = F.summarise(checks)

    status = ("NEURAL_CARRIER_PREMIUM_IN_THE_OUTER_LOOP" if carrier["lcb95"] > 0.0
              else "RETENTION_YES_NEURAL_CARRIER_NO")

    payload = {
        "schema_version": "program_n_outer_loop_carrier_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "READJUDICATION_OF_A_SEALED_REPLAY",
        "scope": "SPLITS_RETENTION_FROM_NEURAL_CARRIER_NO_SEEDS_NO_SEARCHES",
        "endpoint": "auc_regret_norm__lower_is_better__signed_so_positive_favours_the_neuron",
        "neural_carrier": NEURAL_CARRIER, "classical_memory_pool": classical_memory,
        "carrier_premium": carrier, "retention_contrast": retention,
        "mean_auc_by_arm_best_first": ranking, "neuron_rank_of": [neuron_rank, len(ranking)],
        "falsifiers": checks, "falsifier_summary": summary,
        "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT, reference=args.ladder)

    print(f"\nveredicto: {status}\n")
    print(f"  ranking AUC (menor es mejor), la neurona es la {neuron_rank}a de {len(ranking)}:")
    for a, v in list(ranking.items())[:6]:
        mark = "  <- la red" if a == NEURAL_CARRIER else ""
        print(f"    {a:26}{v:.6f}{mark}")
    print(f"\n  PORTADOR NEURAL  neurona - mejor clasico con memoria")
    print(f"    {carrier['mean']:+.6f} [{carrier['lcb95']:+.6f}, {carrier['ucb95']:+.6f}]"
          f"   rival mas duro: {carrier['hardest_arm']}")
    print(f"  RETENCION        reset - retenida")
    print(f"    {retention['mean']:+.6f}")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:48} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
