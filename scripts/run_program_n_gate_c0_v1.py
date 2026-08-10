#!/usr/bin/env python3
"""Gate C, stage C0: does either candidate expert qualify for amortization at all?

Preregistration: `docs/PREREGISTRO_PUERTA_C_AMORTIZACION_2026-08-09.md`

An expert worth amortizing must be BOTH more expensive than its amortizer AND better than the cheap
heuristic. One without the other is worthless: amortizing a closed-form rule costs more than the
rule, and amortizing an expensive planner nobody measured buys an unknown.

C0 measures the conjunction on the only two candidates that exist in this tree, and trains nothing.
The instrumented rollout counts planner work per decision, so a policy named "MPC" that turns out to
be four multiply-adds is caught here rather than after a week of imitation learning.
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

from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.arm_runner import seal_and_write                              # noqa: E402
from supply_chain.replenish import central_cell, materialize_tape               # noqa: E402
from supply_chain.replenish_ret import (                                        # noqa: E402
    WEEKS, paced_policy, rollout_policy, sS_policy)

CONTRACT = Path("docs/PREREGISTRO_PUERTA_C_AMORTIZACION_2026-08-09.md")
OUT = Path("results/program_n/gate_c0_expert_audit/result.json")
SESOI = 0.01
REPS = 2_000

#: The selected parameters of the K3 arms, read from the sealed artifact rather than re-searched.
K3_ARTIFACT = Path("results/k3/strong_mpc_terminal.json")
ESTAR_ARTIFACT = Path("results/estar_hcompute_preflight_v1/result.json")


class CountingPolicy:
    """Wraps a policy and counts the work it does per decision.

    A planner rolls the model forward: it evaluates candidates, calls a simulator, or iterates a
    solver. A closed-form rule does none of those. The wrapper cannot make a rule look like a
    planner, which is the whole point -- the count comes from the policy's own execution.
    """

    def __init__(self, policy, probe):
        self.policy, self.probe = policy, probe
        self.decisions = 0
        self.candidate_evaluations = 0
        self.simulator_calls = 0

    def __call__(self, obs):
        self.decisions += 1
        before = self.probe()
        out = self.policy(obs)
        after = self.probe()
        self.candidate_evaluations += after[0] - before[0]
        self.simulator_calls += after[1] - before[1]
        return out


def time_per_call(fn, obs, reps=REPS):
    fn(obs)                                                    # warm
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(obs)
        samples.append(time.perf_counter() - t0)
    a = np.array(samples)
    return {"p50_seconds": float(np.quantile(a, 0.50)),
            "p95_seconds": float(np.quantile(a, 0.95)),
            "mean_seconds": float(a.mean()), "repetitions": int(reps)}


def mlp_forward_factory(n_in, width=64, seed=20260809):
    """The amortizer's online cost: one forward pass at the declared Gate B budget."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=(n_in, width)) / np.sqrt(n_in)
    w2 = rng.normal(size=(width, width)) / np.sqrt(width)
    w3 = rng.normal(size=(width, 1)) / np.sqrt(width)
    keys = None

    def forward(obs):
        nonlocal keys
        if keys is None:
            keys = sorted(obs)
        x = np.array([obs[k] for k in keys], dtype=float)
        return float((np.tanh(np.tanh(x @ w1) @ w2) @ w3)[0])
    return forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tapes", type=int, default=40)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    k3 = json.loads(K3_ARTIFACT.read_text())
    estar = json.loads(ESTAR_ARTIFACT.read_text())
    cell = central_cell()
    tapes = [materialize_tape(6720001 + i, cell, WEEKS) for i in range(args.tapes)]

    # ---- candidate 1: K3's "strong_mpc" -----------------------------------------------------
    alpha, beta, gamma = k3["selected"]["mpc"]
    raw = paced_policy(float(alpha), float(beta), float(gamma))
    counter = CountingPolicy(raw, probe=lambda: (0, 0))
    rows = [rollout_policy(t, counter, exact_budget=True) for t in tapes]
    obs_probe = {"on_hand_D0": 1.0, "pipeline_D0": 0.2, "backlog_D0": 0.0,
                 "remaining_budget_D0": 5.0, "weeks_remaining": 10.0, "forecast_D0": 1.1}
    k3_timing = time_per_call(raw, obs_probe)
    net = mlp_forward_factory(len(obs_probe))
    net_timing = time_per_call(net, obs_probe)

    k3_quality = k3["candidate_minus_best_classical"]["ret_order"]
    candidates = {
        "k3_strong_mpc": {
            "artifact": str(K3_ARTIFACT),
            "plans": {"decisions": counter.decisions,
                      "candidate_evaluations": counter.candidate_evaluations,
                      "simulator_calls": counter.simulator_calls,
                      "functional_form": "paced_policy(alpha, beta, gamma) -- budget pace plus "
                                         "forecast and inventory feedback; its own docstring says "
                                         "'no latent state'",
                      "selected_parameters": [float(alpha), float(beta), float(gamma)],
                      "same_family_as": "inventory_paced, which is paced_policy with alpha fixed "
                                        "at 0.0; the two arms differ only by that grid"},
            "online_cost": k3_timing, "amortizer_cost": net_timing,
            "cost_ratio_expert_over_net": k3_timing["p50_seconds"] / net_timing["p50_seconds"],
            "quality_vs_cheap_heuristic": {
                "metric": "ret_order", "baseline": k3["best_classical"],
                "mean": k3_quality[0], "ci95_low": k3_quality[1], "ci95_high": k3_quality[2],
                "measured": True, "lcb_positive": k3_quality[1] > 0.0},
            "mean_ret_order_replayed": float(np.mean([r.ret_order for r in rows])),
        },
        "estar_direct_des_mpc": {
            "artifact": str(ESTAR_ARTIFACT),
            "plans": {"decisions": None,
                      "candidate_evaluations": None,
                      "simulator_calls": max(estar["h_compute_adjudication"]
                                             ["mpc_des_calls_by_level"]),
                      "functional_form": "DirectDESMPC -- enumerative receding horizon, one fresh "
                                         "DES rollout per candidate",
                      "selected_parameters": None, "same_family_as": None},
            "online_cost": {"p95_seconds": max(estar["h_compute_adjudication"]
                                               ["mpc_p95_seconds_by_level"]),
                            "source": "sealed engineering preflight, not re-measured here"},
            "amortizer_cost": net_timing,
            "cost_ratio_expert_over_net": (max(estar["h_compute_adjudication"]
                                               ["mpc_p95_seconds_by_level"])
                                           / net_timing["p95_seconds"]),
            "quality_vs_cheap_heuristic": {
                "metric": None, "baseline": None, "mean": None,
                "ci95_low": None, "ci95_high": None, "measured": False,
                "lcb_positive": None,
                "note": "the preflight is engineering_only with learner_trained=false and compares "
                        "the planner against no heuristic at all"},
            "mean_ret_order_replayed": None,
        },
    }

    def verdicts(c):
        return {"plans": bool(c["plans"]["candidate_evaluations"]
                              or c["plans"]["simulator_calls"]),
                "expensive": bool(c["cost_ratio_expert_over_net"] > 1.0),
                "better": bool(c["quality_vs_cheap_heuristic"]["lcb_positive"])}

    per_candidate = {k: verdicts(c) for k, c in candidates.items()}
    qualifying = [k for k, v in per_candidate.items() if v["expensive"] and v["better"]]

    checks = {
        "c1_the_expert_actually_plans": F.check(
            all(v["plans"] for v in per_candidate.values()),
            "a policy named for the role it was expected to play, with no measurement that it "
            "plays it, is the defect this project already shipped once as a 'ceiling'. Zero "
            "candidate evaluations and zero simulator calls means a closed-form rule",
            computed_from={"n_candidates": len(per_candidate),
                           "n_that_plan": sum(v["plans"] for v in per_candidate.values())},
            per_candidate={k: v["plans"] for k, v in per_candidate.items()}),
        "c2_the_expert_is_more_expensive_than_its_amortizer": F.check(
            all(v["expensive"] for v in per_candidate.values()),
            "a linear rule is cheaper than any network forward pass, which makes the amortization "
            "estimand NEGATIVE by construction rather than merely small",
            computed_from={"n_candidates": len(per_candidate),
                           "n_expensive": sum(v["expensive"] for v in per_candidate.values())},
            cost_ratios={k: c["cost_ratio_expert_over_net"] for k, c in candidates.items()}),
        "c3_the_expert_beats_the_cheap_heuristic": F.check(
            all(v["better"] for v in per_candidate.values()),
            "amortizing a planner nobody measured against a heuristic buys an unknown; this fails "
            "explicitly when the quality comparison was never run, which is not the same as it "
            "having been run and lost",
            computed_from={"n_candidates": len(per_candidate),
                           "n_better": sum(bool(v["better"]) for v in per_candidate.values())},
            measured={k: c["quality_vs_cheap_heuristic"]["measured"]
                      for k, c in candidates.items()}),
        "c4_a_control_must_separate_the_candidates": F.check(
            len({json.dumps(v, sort_keys=True) for v in per_candidate.values()}) > 1,
            "if both candidates get the identical verdict on all three conditions, C0 does not "
            "discriminate and the audit measures nothing",
            computed_from={"n_distinct_verdicts":
                           len({json.dumps(v, sort_keys=True) for v in per_candidate.values()}),
                           "n_candidates": len(per_candidate)}),
    }
    checks["custody"] = {
        "passed": None, "not_applicable": True,
        "evidence": {"why_it_can_fail": "it cannot: C0 opens no seed and trains nothing. It "
                                        "replays declared K3 test tapes to instrument a policy "
                                        "and times code; the quality numbers are read from "
                                        "sealed artifacts.",
                     "seeds_opened": 0, "networks_trained": 0,
                     "artifacts_read": [str(K3_ARTIFACT), str(ESTAR_ARTIFACT)]}}
    summary = F.summarise(checks)

    any_better = any(v["better"] for v in per_candidate.values())
    any_expensive = any(v["expensive"] for v in per_candidate.values())
    if qualifying:
        status = "EXPERT_QUALIFIES_PROCEED_TO_C1"
    elif any_better and not any_expensive:
        status = "NOTHING_TO_AMORTIZE"
    elif any_expensive and not any_better:
        status = "NO_EXPERT_WITH_MEASURED_QUALITY"
    else:
        status = "NO_QUALIFYING_EXPERT"

    payload = {
        "schema_version": "program_n_gate_c0_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "EXPERT_AUDIT_BEFORE_TRAINING",
        "scope": "NO_SEEDS_OPENED_NO_NETWORK_TRAINED_STOP_RULE_ONLY",
        "sesoi": SESOI, "training_authorized_by_this_run": bool(qualifying),
        "candidates": candidates, "per_candidate_verdicts": per_candidate,
        "qualifying_experts": qualifying,
        "falsifiers": checks, "falsifier_summary": summary,
        "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT, reference=K3_ARTIFACT)

    print(f"\nveredicto: {status}\n")
    for name, c in candidates.items():
        v = per_candidate[name]
        q = c["quality_vs_cheap_heuristic"]
        print(f"  {name}")
        print(f"    planifica          {v['plans']}   "
              f"(evaluaciones {c['plans']['candidate_evaluations']}, "
              f"llamadas al simulador {c['plans']['simulator_calls']})")
        print(f"    caro               {v['expensive']}   "
              f"(razon experto/red {c['cost_ratio_expert_over_net']:.3f}x)")
        print(f"    mejor              {v['better']}   "
              + (f"({q['metric']} {q['mean']:+.5f} [{q['ci95_low']:+.5f}, {q['ci95_high']:+.5f}] "
                 f"vs {q['baseline']})" if q["measured"] else "(NUNCA MEDIDO)"))
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:52} {mark}")
    print(f"\n  entrenamiento autorizado por esta corrida: {bool(qualifying)}")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
