#!/usr/bin/env python3
"""Factorial retention experiment: one frozen checkpoint evaluated four ways.

Contract: contracts/q_r1_matched_retention_factorial_v3.json (frozen, receipt alongside).
Design adopted from the Codex lineage's DRAFT; see the contract's `adoption` block.

The predecessor confounded three factors. This separates two of them on a single frozen
checkpoint, so no difference between arms can be attributed to different weights:

    P0_H0  prior 0.5           hidden reset at every campaign boundary
    P1_H0  carried prior       hidden reset
    P0_H1  prior 0.5           hidden retained across the whole history
    P1_H1  carried prior       hidden retained          <- canonical training arm

    explicit prior value          = P1_H0 - P0_H0
    raw recurrent memory value    = P0_H1 - P0_H0
    recurrent residual give prior = P1_H1 - P1_H0
    interaction                   = (P1_H1 - P1_H0) - (P0_H1 - P0_H0)

Blocking gates, all enforced before any arm value is printed:
  * rho identity: a campaign shared with the enumerated frontier set must rebuild to the
    rho=0.90 skeleton hash, and an identical calendar must match the exact lookup to 1e-9;
  * arm parity: the four arms differ ONLY in the two declared factor levels;
  * the static bar is selected on the selection split and written before grading.

Usage:
    .venv/bin/python scripts/run_q_r1_factorial_v3.py --seeds 5 --output-dir <dir>
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.c6_perbatch_ceiling import OBJECTIVE  # noqa: E402
from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.oracle_capture import (  # noqa: E402
    BOOT_SEED, calendar_index, load_campaigns, pooled_capture,
)
from supply_chain.oracle_curve_v2 import (  # noqa: E402
    HistorySpec, MetaCampaignEnv, load_history,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    simulate_full_des_frontier,
)
from supply_chain.retained_context_discovery import build_campaign_history  # noqa: E402

CONTRACT = ROOT / "contracts/q_r1_matched_retention_factorial_v3.json"
RECEIPT = ROOT / "contracts/q_r1_matched_retention_factorial_v3_freeze_receipt.json"
FRONTIERS = ROOT / "results/fig5_surrogate_v1/frontiers"
RHO, SHARE, CAMPAIGNS = 0.90, 0.90, 12
KAPPAS = (0.75, 0.90)
ARMS = (("P0_H0", False, False), ("P1_H0", True, False),
        ("P0_H1", False, True), ("P1_H1", True, True))
IDENTITY_PROBE = (7_570_801, 6, 0.90, "1ab4ec34")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authority() -> dict:
    contract = json.loads(CONTRACT.read_text())
    receipt = json.loads(RECEIPT.read_text())
    if contract["status"] != "FROZEN_PROSPECTIVE_UNOPENED":
        raise SystemExit("contract is not frozen and unopened")
    if receipt["contract_sha256"] != sha256(CONTRACT):
        raise SystemExit("contract bytes no longer match the freeze receipt")
    if receipt["confirmation_roots_opened"]:
        raise SystemExit("receipt marks the confirmation block opened")
    declared = contract["physical_contract"]["within_campaign_regime_persistence_rho"]
    if abs(float(declared) - RHO) > 1e-12:
        raise SystemExit(f"contract rho {declared} != runner rho {RHO}")
    return contract


# The 4-action COUNT scheduler, which is what the env, the enumerated 4^8 frontiers and the
# frozen comparator all use. c6_perbatch_ceiling.SCHED_PATTERN is the 8-action PER-BATCH
# mapping of the C6 gate; using it here silently reinterpreted action 1 as [P_C,P_H,P_H]
# instead of [P_H,P_C,P_H], so the reward was computed on different physics than the grading.
COUNT_SCHEDULER = scheduler()


def objective_of(skeleton, calendar) -> float:
    metrics = simulate_full_des_frontier(
        skeleton=skeleton, scheduler=COUNT_SCHEDULER,
        calendars=np.asarray([calendar], dtype=np.uint8), include_q_r1_metrics=True)
    return float(np.asarray(metrics[OBJECTIVE])[0])


def gate_rho_identity(campaigns_by_key) -> dict:
    """Blocking: the construction must be the rho=0.90 one, proven by hash and by value."""
    root, index, kappa, expected = IDENTITY_PROBE
    history = build_campaign_history(
        history_root=root, campaigns=CAMPAIGNS, kappa=kappa, scheduler=scheduler(),
        regime_persistence=RHO, dominant_share=SHARE)
    digest = history[index].skeleton.skeleton_sha256
    if not digest.startswith(expected):
        raise SystemExit(f"rho identity gate FAILED: {digest[:8]} != {expected}")
    key = (root, index, "binary_0.9")
    probe_calendar = [1, 2, 0, 3, 1, 2, 0, 1]
    simulated = objective_of(history[index].skeleton, probe_calendar)
    lookup = float(campaigns_by_key[key].labels[calendar_index(probe_calendar)])
    if abs(simulated - lookup) > 1e-9:
        raise SystemExit(f"rho identity gate FAILED on value: {simulated} vs {lookup}")
    return {"skeleton_sha256_prefix": digest[:16], "expected_prefix": expected,
            "value_agreement_abs_delta": abs(simulated - lookup), "passed": True}


def make_env(histories, *, prior: bool, hidden: bool, seed: int) -> MetaCampaignEnv:
    env = MetaCampaignEnv(scheduler=scheduler(), histories=histories, retained=prior,
                          objective_fn=objective_of, rng_seed=seed)
    env.hidden_retained = bool(hidden)  # evaluation-time factor, not an env difference
    return env


def gate_arm_parity(histories, seed: int) -> dict:
    """Blocking: the four arms may differ only in the two declared factor levels."""
    configs = {}
    for name, prior, hidden in ARMS:
        env = make_env(histories, prior=prior, hidden=hidden, seed=seed)
        cfg = env.configuration()
        cfg.pop("histories", None)
        configs[name] = (json.dumps(cfg, sort_keys=True), prior, hidden)
    base = configs["P0_H0"][0]
    for name, (cfg, _p, _h) in configs.items():
        if cfg != base:
            raise SystemExit(f"arm parity gate FAILED: {name} differs beyond its factors")
    levels = {n: (p, h) for n, (_c, p, h) in configs.items()}
    if len(set(levels.values())) != 4:
        raise SystemExit("arm parity gate FAILED: the four factor combinations are not distinct")
    return {"identical_configuration_outside_factors": True, "factor_levels": {
        n: {"explicit_prior": p, "hidden_retained": h} for n, (p, h) in levels.items()}}


def rollout(model, env: MetaCampaignEnv, spec: HistorySpec) -> list[dict]:
    """One meta-episode. The hidden state persists iff this arm retains it."""
    obs, _ = env.reset(options={"history": spec})
    state, done, first = None, False, True
    rows: list[dict] = []
    while not done:
        if env.hidden_retained:
            episode_start = np.array([first])
        else:
            episode_start = np.array([not env.last_calendar])
        action, state = model.predict(obs, state=state, episode_start=episode_start,
                                      deterministic=True)
        first = False
        obs, _r, done, _t, info = env.step(int(action))
        if info.get("campaign_boundary"):
            rows.append(info)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-indices", nargs="+", type=int, default=None,
                        help="run only these positions of the contract's optimizer_seeds")
    parser.add_argument("--total-timesteps", type=int, default=96_000)
    parser.add_argument("--bar-roots", type=int, default=8,
                        help="selection-split roots enumerated for the static bar")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")

    from sb3_contrib import RecurrentPPO  # noqa: PLC0415

    contract = authority()
    splits = contract["data_splits"]
    train_lo, train_hi = splits["training_history_roots"]
    sel_lo, sel_hi = splits["checkpoint_selection_history_roots"]
    declared = list(splits["optimizer_seeds"])
    seeds = ([declared[i] for i in args.seed_indices] if args.seed_indices
             else declared[: args.seeds])
    train_specs = [HistorySpec(r, k, CAMPAIGNS)
                   for r in range(train_lo, train_hi + 1) for k in KAPPAS]
    sel_specs = [HistorySpec(r, k, CAMPAIGNS)
                 for r in range(sel_lo, sel_lo + args.bar_roots) for k in KAPPAS]
    started = time.perf_counter()

    # ---- gates ---------------------------------------------------------------
    test_campaigns = load_campaigns(FRONTIERS)
    by_key = {c.key: c for c in test_campaigns}
    gates = {"rho_identity": gate_rho_identity(by_key),
             "arm_parity": gate_arm_parity(train_specs, seeds[0])}
    print(f"[gates] rho identity OK ({gates['rho_identity']['skeleton_sha256_prefix']}), "
          f"arm parity OK ({time.perf_counter()-started:.0f}s)", flush=True)

    # ---- static bar on the selection split, written before any grading -------
    all_calendars = np.load(FRONTIERS / "calendars.npz")["calendars"]
    sched = sched_cache = scheduler()
    labels = []
    for spec in sel_specs:
        campaigns, _ = load_history(spec, sched_cache)
        for campaign in campaigns:
            metrics = simulate_full_des_frontier(
                skeleton=campaign.skeleton, scheduler=COUNT_SCHEDULER,
                calendars=all_calendars, include_q_r1_metrics=True)
            labels.append(np.asarray(metrics[OBJECTIVE], dtype=float))
    stack = np.vstack(labels)
    bar_row = int(stack.mean(axis=0).argmax())
    bar_calendar = [(bar_row // 4 ** (7 - w)) % 4 for w in range(8)]
    bar = {c.key: float(c.labels[bar_row]) for c in test_campaigns}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "static_bar.json").write_text(json.dumps({
        "calendar": bar_calendar, "frontier_row": bar_row,
        "selected_on_roots": [sel_lo, sel_lo + args.bar_roots - 1],
        "selection_campaigns": int(stack.shape[0]),
        "selected_before_any_arm_was_graded": True,
    }, indent=1, sort_keys=True) + "\n")
    print(f"[bar] {bar_calendar} on {stack.shape[0]} selection campaigns "
          f"({time.perf_counter()-started:.0f}s)", flush=True)

    test_specs = sorted({HistorySpec(c.history_root, c.kappa, CAMPAIGNS)
                         for c in test_campaigns}, key=lambda s: (s.history_root, s.kappa))

    def grade_arm(model, prior: bool, hidden: bool) -> dict:
        env = make_env(test_specs, prior=prior, hidden=hidden, seed=0)
        calendars = {}
        for spec in test_specs:
            mode = "binary_0.75" if spec.kappa < 0.85 else "binary_0.9"
            for row in rollout(model, env, spec):
                key = (row["history_root"], row["campaign_index"], mode)
                if key in by_key:
                    calendars[key] = row["calendar"]
        present = [c for c in test_campaigns if c.key in calendars]
        pooled = pooled_capture(present, calendars, bar, np.random.default_rng(BOOT_SEED))
        return {**pooled,
                "mean_label": float(np.mean([c.value_of(calendars[c.key]) for c in present])),
                "exact_optimum_hits": int(sum(
                    1 for c in present if c.ceiling - c.value_of(calendars[c.key]) <= 1e-9)),
                "distinct_calendars": len({tuple(v) for v in calendars.values()}),
                "n_graded": len(present)}

    # ---- train the canonical arm, then evaluate the frozen checkpoint 4 ways -
    runs = []
    for seed in seeds:
        train_env = make_env(train_specs, prior=True, hidden=True, seed=seed)
        model = RecurrentPPO("MlpLstmPolicy", train_env, seed=seed, n_steps=96,
                             batch_size=96, verbose=0)
        model.learn(total_timesteps=args.total_timesteps)
        selection = float(np.mean([row["objective"] for spec in sel_specs
                                   for row in rollout(model, make_env(
                                       sel_specs, prior=True, hidden=True, seed=seed), spec)]))
        arms = {name: grade_arm(model, prior, hidden) for name, prior, hidden in ARMS}
        estimands = {
            "explicit_prior_value": arms["P1_H0"]["pooled_ratio"] - arms["P0_H0"]["pooled_ratio"],
            "raw_recurrent_memory_value": arms["P0_H1"]["pooled_ratio"] - arms["P0_H0"]["pooled_ratio"],
            "recurrent_residual_given_prior": arms["P1_H1"]["pooled_ratio"] - arms["P1_H0"]["pooled_ratio"],
            "total_retained_treatment": arms["P1_H1"]["pooled_ratio"] - arms["P0_H0"]["pooled_ratio"],
        }
        estimands["interaction"] = (estimands["recurrent_residual_given_prior"]
                                   - estimands["raw_recurrent_memory_value"])
        runs.append({"seed": seed, "selection_mean_ret": selection, "arms": arms,
                     "estimands": estimands})
        print(f"[seed {seed}] sel {selection:.4f} | " + " ".join(
            f"{n} {arms[n]['pooled_ratio']:+.4f}" for n, _p, _h in ARMS)
            + f" | prior {estimands['explicit_prior_value']:+.4f}"
              f" memory {estimands['raw_recurrent_memory_value']:+.4f}"
              f" interaction {estimands['interaction']:+.4f}"
              f" ({time.perf_counter()-started:.0f}s)", flush=True)

    payload = {
        "schema": "q_r1_matched_retention_factorial_v3_run",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT),
        "claim_status": "DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "static_bar": {"calendar": bar_calendar, "frontier_row": bar_row},
        "rho": RHO, "dominant_share": SHARE, "campaigns_per_history": CAMPAIGNS,
        "training_roots": [train_lo, train_hi],
        "selection_roots": [sel_lo, sel_hi],
        "confirmation_roots_opened": False,
        "development_evaluation_set": "burned enumerated campaigns 7570801-24",
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"-> {args.output_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
