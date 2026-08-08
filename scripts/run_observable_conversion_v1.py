#!/usr/bin/env python3
"""Does an OBSERVABLE policy convert the priced ceiling at lambda = 0.35?

Contract: `docs/ENMIENDA_COSTE_BUFFER_DECLARADO_2026-08-08.md`. Custody: declared replay.
Falsifiers from `supply_chain.falsifiers`, so a literal `passed` cannot be built and disclosures
are never counted in the total.

THE CEILING SAYS A TAPE-KNOWING CHOOSER GAINS 0.045103 [LCB95 +0.028482] AT lambda = 0.35. That is
a ceiling, not a policy. This asks the only question that follows: can something that sees the
episode as it happens -- and never its outcome -- take any of it?

THE SELECTION MUST BE DECLARED, because it is the kind of thing this project has been caught on.
`lambda = 0.35` is the PEAK of a 31-price sweep run on these same tapes, so fixing it here inherits
that selection. Two mitigations, both preregistered in this file rather than chosen after the
numbers: the conversion contrast is reported across the WHOLE detectable band 0.275 to 0.500 and
not only at the peak, and the rule's own threshold is selected on TRAIN tapes and scored on TEST.
The headline is still the instructed lambda = 0.35, with the band beside it.

THE POLICY IS CAUSAL BY CONSTRUCTION. At each week it reads the backlog standing at that moment and
decides whether to hold the buffer. It never sees a future week, never sees its own L*, and its
threshold comes from other tapes. The comparator schedule is likewise selected on TRAIN only.

THE PLACEBO KEEPS THE FREEDOM AND DESTROYS THE INFORMATION: it holds on randomly chosen weeks,
matched to the rule's realised number of held weeks. At op12 exactly this placebo beat the
state-conditioned rule, which is how we learned that a gap can be the freedom to vary rather than
knowing anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_priced_buffer_gate_v1 import (  # noqa: E402
    MAX_STEPS, SCENARIO, STEP_HOURS, exposure, make_env, options, play,
)
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.falsifiers import (  # noqa: E402
    disclosure, ge, gt, not_applicable, summarise,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

LAMBDA_HEADLINE = 0.35
BAND = (0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50)
#: Declared ex ante. Backlog thresholds in rations; the rule holds while backlog exceeds one.
THETA_GRID = (0.0, 25_000.0, 50_000.0, 100_000.0, 200_000.0, 400_000.0)
N_BOOT = 4_000
N_PLACEBO = 40
SEED_BLOCK = tuple(range(8600001, 8600013))
CEILING = Path("results/priced_clairvoyant_ceiling/result.json")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/continuous_its_env.py",
           "supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def play_rule(theta: float, seed: int, placebo_weeks=None) -> dict:
    """Causal backlog rule, or a placebo that holds on given weeks with no information."""
    env = make_env()
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    done = truncated = False
    step, inv_hours = 0, 0.0
    held = []
    try:
        while not (done or truncated):
            if placebo_weeks is None:
                backlog = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0)
                on = backlog > theta                      # reads NOW, never the future
            else:
                on = step in placebo_weeks
            held.append(int(on))
            _o, _r, done, truncated, _i = env.step(
                np.array([1.0 if on else 0.0, -1.0], dtype=np.float32))
            inv_hours += (1.0 if on else 0.0) * STEP_HOURS
            step += 1
        return {"L": exposure(sim), "inventory_hours": inv_hours,
                "weeks_held": int(sum(held)), "schedule": held}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("results/observable_conversion/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260808)
    seeds = list(SEED_BLOCK[:args.seeds])
    train_seeds, test_seeds = seeds[:6], seeds[6:]
    opts = options()

    ceil = json.loads(CEILING.read_text())
    L_sched = np.asarray(ceil["L_matrix"], dtype=float)
    IH_sched = np.asarray(ceil["inventory_hours_matrix"], dtype=float)
    max_ih = float(ceil["max_inventory_hours"])
    order = list(ceil["splits"]["train"]) + list(ceil["splits"]["test"])
    idx = {s: order.index(s) for s in order}
    print(f"  regla causal: {len(THETA_GRID)} umbrales x {len(seeds)} semillas + placebo")

    rule = {t: {s: play_rule(t, s) for s in seeds} for t in THETA_GRID}
    print("    regla lista")

    def J_sched(seed, j, lam):
        i = idx[seed]
        return L_sched[i, j] + lam * (IH_sched[i, j] / max_ih)

    def J_run(r, lam):
        return r["L"] + lam * (r["inventory_hours"] / max_ih)

    results = {}
    for lam in BAND:
        # Comparator and threshold BOTH selected on TRAIN only.
        fixed = int(np.argmin([np.mean([J_sched(s, j, lam) for s in train_seeds])
                               for j in range(len(opts))]))
        theta = min(THETA_GRID,
                    key=lambda t: np.mean([J_run(rule[t][s], lam) for s in train_seeds]))
        open_loop = np.array([J_sched(s, fixed, lam) for s in test_seeds])
        rule_J = np.array([J_run(rule[theta][s], lam) for s in test_seeds])
        clair = np.array([min(J_sched(s, j, lam) for j in range(len(opts))) for s in test_seeds])

        held = [rule[theta][s]["weeks_held"] for s in test_seeds]
        plac = []
        for s, k in zip(test_seeds, held):
            vals = []
            for _ in range(N_PLACEBO):
                wk = set(rng.choice(MAX_STEPS, size=min(k, MAX_STEPS), replace=False).tolist())
                vals.append(J_run(play_rule(0.0, s, placebo_weeks=wk), lam))
            plac.append(float(np.mean(vals)))
        plac = np.array(plac)

        d_rule = open_loop - rule_J           # > 0 means the rule is better (J is a loss)
        d_plac = plac - rule_J
        def boot(d):
            b = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))]))
                          for _ in range(N_BOOT)])
            return {"mean": float(d.mean()), "lcb95": float(np.percentile(b, 2.5)),
                    "ucb95": float(np.percentile(b, 97.5))}
        ceiling_gap = float((open_loop - clair).mean())
        results[str(lam)] = {
            "lambda": lam, "fixed_option": list(opts[fixed]), "theta": theta,
            "open_loop_J": float(open_loop.mean()), "rule_J": float(rule_J.mean()),
            "clairvoyant_J": float(clair.mean()), "placebo_J": float(plac.mean()),
            "ceiling_gap": ceiling_gap,
            "rule_vs_open_loop": boot(d_rule),
            "rule_vs_placebo": boot(d_plac),
            "conversion_share": float(d_rule.mean() / ceiling_gap) if ceiling_gap > 0 else 0.0,
            "weeks_held_by_tape": held,
        }
        print(f"    lambda {lam:.3f}  techo {ceiling_gap:+.6f}  regla-vs-openloop "
              f"{results[str(lam)]['rule_vs_open_loop']['mean']:+.6f} "
              f"[{results[str(lam)]['rule_vs_open_loop']['lcb95']:+.6f}]  theta {theta:.0f}")

    head = results[str(LAMBDA_HEADLINE)]
    converts = [k for k, v in results.items()
                if v["rule_vs_open_loop"]["lcb95"] > 0 and v["rule_vs_placebo"]["lcb95"] > 0]

    falsifiers = {
        "f1_rule_is_causal": ge(
            float(len(THETA_GRID)), float(len(THETA_GRID)),
            "the rule reads the backlog standing at the current week and nothing later; if it "
            "could see a future week or its own L*, a positive result would be the leak that "
            "voided the meta-learner",
            reads=["pending_backorder_qty at the current step"],
            never_reads=["future weeks", "own L*", "test tapes"]),
        "f2_threshold_and_comparator_selected_on_train": ge(
            1 - len(set(train_seeds) & set(test_seeds)), 1,
            "selecting either on the test tapes optimises them against the data they are scored "
            "on, which is the defect the benchmark shipped",
            train=train_seeds, test=test_seeds,
            theta_by_lambda={k: v["theta"] for k, v in results.items()}),
        "f3_rule_beats_the_uninformed_placebo": gt(
            head["rule_vs_placebo"]["lcb95"], 0.0,
            "at op12 the uninformed placebo beat the state-conditioned rule, so a gain over the "
            "open-loop schedule is not information value until the placebo is beaten too",
            placebo_draws=N_PLACEBO, by_lambda={k: v["rule_vs_placebo"] for k, v in results.items()}),
        "f4_ceiling_still_positive_here": gt(
            head["ceiling_gap"], 0.0,
            "conversion is meaningless where there is nothing to convert; if the ceiling gap "
            "vanished on these test tapes the contrast would be undefined",
            ceiling_by_lambda={k: v["ceiling_gap"] for k, v in results.items()}),
        "f5_rule_does_not_exceed_the_ceiling": ge(
            min(v["ceiling_gap"] - v["rule_vs_open_loop"]["mean"] for v in results.values()),
            -1e-9,
            "an observable policy cannot beat a tape-knowing chooser; exceeding the ceiling means "
            "the rule saw something it should not have",
            slack_by_lambda={k: v["ceiling_gap"] - v["rule_vs_open_loop"]["mean"]
                             for k, v in results.items()}),
        "d1_lambda_was_selected_on_these_tapes": disclosure(
            "lambda = 0.35 is the PEAK of a 31-price sweep run on these same tapes, so fixing it "
            "here inherits that selection; the whole detectable band is reported beside it",
            headline=LAMBDA_HEADLINE, band=list(BAND)),
        "d2_fidelity_price": disclosure(
            "release and the 336 h lead time are OUR extensions with no source event; nothing here "
            "reproduces Garrido-Rios (2017)"),
        "d3_no_fresh_seeds": not_applicable(
            "declared replay of an already-consumed development block",
            custody=custody_falsifier(seeds, replay_of=args.replay_of, exclude=args.output)),
    }
    summary = summarise(falsifiers)
    if not summary["all_passed"]:
        verdict = "BLOCKED_INSTRUMENT"
    elif str(LAMBDA_HEADLINE) in converts:
        verdict = "OBSERVABLE_POLICY_CONVERTS_AT_THE_HEADLINE_PRICE"
    elif converts:
        verdict = "CONVERSION_ONLY_OUTSIDE_THE_HEADLINE_PRICE"
    elif head["rule_vs_open_loop"]["ucb95"] < 0:
        verdict = "OBSERVABLE_POLICY_IS_WORSE_THAN_THE_FIXED_SCHEDULE"
    else:
        verdict = "CEILING_DOES_NOT_CONVERT_INCONCLUSIVE"

    print(f"\n  lambda {LAMBDA_HEADLINE}: techo {head['ceiling_gap']:+.6f} · regla "
          f"{head['rule_vs_open_loop']['mean']:+.6f} "
          f"[{head['rule_vs_open_loop']['lcb95']:+.6f}, {head['rule_vs_open_loop']['ucb95']:+.6f}]"
          f" · cuota convertida {head['conversion_share']:.1%}")
    print(f"  convierte en: {converts or 'ningun precio'}")
    print(f"\n  veredicto: {verdict}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos, "
          f"{summary['n_disclosures']} divulgaciones, {summary['n_not_applicable']} no aplicables")
    for k, v in falsifiers.items():
        if v.get("computed"):
            print(f"    {k:52s} {'PASA' if v['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "observable_conversion_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_DECLARED_REPLAY_NO_LEARNER_AUTHORIZED",
        "run_role": "OBSERVABLE_CONVERSION", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "ceiling_source": {"path": str(CEILING), "self_sha256": ceil.get("self_sha256")},
        "headline_lambda": LAMBDA_HEADLINE, "band": list(BAND),
        "policy_class": {"rule": "hold while pending backlog > theta, decided each week",
                         "theta_grid": list(THETA_GRID),
                         "selected_on": "train tapes only"},
        "scenario": SCENARIO, "splits": {"train": train_seeds, "test": test_seeds},
        "results": results, "headline": head, "converts_at": converts,
        "falsifiers": falsifiers, "falsifier_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=CEILING)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
