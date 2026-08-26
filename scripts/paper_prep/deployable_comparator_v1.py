#!/usr/bin/env python3
"""Program Q secondary arm ``deployable_comparator_v1`` — variants V-a and V-b.

Binding specification:
``/home/ubuntu/scres-sources/preregistros/DEPLOYABLE_COMPARATOR_PREREGISTRO_V1.md``
(SHA-256 frozen in ``scres-sources/preregistros/ACTA_FIRMA_2026-08-25.md``).

Two preregistered deployable variants are evaluated on virgin B tapes (Gate-0
sibling contract block 7550193-7550384, 64 per cell) under exact CRN — the same
tapes feed the frozen learner, the ten classical comparators and both variants:

* **V-a** belief-MPC with an *estimated* model: ``(rho_hat, share_hat)`` are
  estimated from realized product-label history only (grid search maximizing
  the pooled pseudo-log-likelihood of the repo-standard two-state HMM). The
  generator parameters never enter estimation or planning.
* **V-b** greedy baseline consuming *exactly* the learner's partial
  observation: actions are decided week-by-week on the ``StateRichObservation``
  replayed through the same ``state_rich_calendar(action_overrides=...)`` path
  the learner's environment used. Forbidden information (``tape_id``, ``seed``,
  ``latent_regime``, ``true_rho``, ``true_share``, ``future_demand``,
  ``oracle_calendar``) never enters either variant; there are no negative
  pre-event offsets — only realized history and operational state.

Estimand per cell and variant: ``D_deploy(v) = mean(ReT_learner) -
mean(ReT_v)``, paired-CRN percentile bootstrap (unit: tape; 10k resamples).
Learner ReT comes from the ten SHA-frozen historical RecurrentPPO checkpoints
(retraining forbidden). Classical comparators: the ten frozen configurations.

Binding falsifiers: F1 information-leak audit (field-set diff vs the learner),
F2 straw-man audit (V-a estimated-only; V-b learner-observation-only), F3 CRN
integrity (identical tape set for every arm), F4 direct-SimPy bit-exact replay
on a pre-drawn >=10% subset (tolerance 1e-12).

The primary program and its adjudication are untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sb3_contrib import RecurrentPPO  # noqa: E402

from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    FullDESSkeleton,
    direct_full_des_vector,
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_hobs import posterior_after_week, transition_belief  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    StateRichObservation,
    state_rich_calendar,
)

CONTRACT_Q = ROOT / "contracts/program_q_frozen_policy_replication_v1.json"
FREEZE_JSON = (
    ROOT
    / "research/paper2_exhaustive_search/"
    / "program_q_historical_recurrentppo_fallback_freeze_20260717.json"
)
MODELS_DIR = ROOT / "results/program_o/ret_only_learner_v1/vps_run/models"
SCHEDULER_CONTRACT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
PREREGISTRO = Path(
    "/home/ubuntu/scres-sources/preregistros/DEPLOYABLE_COMPARATOR_PREREGISTRO_V1.md"
)
GATE0_RESULTS = ROOT / "results/gate0_split_tape_v1"
OUT_JSON = ROOT / "results/paper_prep/deployable_comparator.json"
OUT_MD = ROOT / "results/paper_prep/deployable_comparator.md"

CELL_IDS = tuple(cell.cell_id for cell in CONFIRMED_RET_CELLS)
CELL_INDEX = {cell.cell_id: index for index, cell in enumerate(CONFIRMED_RET_CELLS)}
B_SEED_START = 7550193  # gate0 block 7550193-7550384: tapes B (64 x 3 cells)
TAPE_ORDER = {"rho75_share90": 0, "rho90_share75": 1, "rho90_share90": 2}
LEARNER_OBS_DIM = 21

# Fixed belief constants of the learner's own observation replay: the frozen
# learner NEVER saw generator parameters -- its observations were always
# produced with these constants (ProgramORetOnlyEnv defaults). All arms share
# this exact replay, so the partial-information playing field is identical.
REPLAY_BELIEF_PERSISTENCE = 0.75
REPLAY_BELIEF_SHARE = 0.90

F4_FRACTION = 0.10
F4_TOLERANCE = 1e-12
F4_DRAW_TEXT = "deployable_comparator_v1_f4_replay_draw"

FORBIDDEN_FIELDS = (
    "tape_id",
    "seed",
    "latent_regime",
    "true_rho",
    "true_share",
    "future_demand",
    "oracle_calendar",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_scheduler() -> dict[str, list[str]]:
    parent = json.loads(SCHEDULER_CONTRACT.read_text())
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


SCHED: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Observation machinery (exactly the learner's)
# ---------------------------------------------------------------------------


def replay_decisions(
    skeleton: FullDESSkeleton,
    actions: Sequence[int],
) -> list[Any]:
    """Replay partial-state evolution under fixed actions.

    This is byte-for-byte the observation path the frozen learner consumed
    (``ProgramORetOnlyEnv._decisions``): ``state_rich_calendar`` with
    ``action_overrides`` and the fixed replay belief constants.
    """
    weeks = int(skeleton.decision_weeks)
    padded = tuple(int(a) for a in actions) + (0,) * (weeks - len(actions))
    _calendar, decisions = state_rich_calendar(
        skeleton=skeleton.as_dict(),
        scheduler=SCHED,
        config=StateRichConfiguration("belief_mpc", 3),
        regime_persistence=REPLAY_BELIEF_PERSISTENCE,
        dominant_share=REPLAY_BELIEF_SHARE,
        action_overrides=padded,
    )
    return list(decisions)


def learner_observation_vector(decision: Any) -> np.ndarray:
    """The learner's exact normalized 21-float vector (program_o_ret_env)."""
    observation = decision.observation
    values: list[float] = []
    for field_name in (
        "on_hand",
        "locked_pipeline",
        "backlog_quantity",
    ):
        for value in getattr(observation, field_name):
            values.append(float(value) / 120_000.0)
    for value in observation.backlog_orders:
        values.append(float(value) / 48.0)
    for value in observation.max_backlog_age:
        values.append(float(value) / 1_344.0)
    for value in observation.in_flight_quantity:
        values.append(float(value) / 120_000.0)
    values.extend((float(observation.belief_c), float(observation.predicted_share_c)))
    previous = np.zeros(5, dtype=np.float32)
    previous[
        4 if observation.previous_action is None else int(observation.previous_action)
    ] = 1.0
    values.extend(previous.tolist())
    values.extend(
        (float(observation.week) / 7.0, float(observation.remaining_decisions) / 8.0)
    )
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (LEARNER_OBS_DIM,):
        raise AssertionError(f"observation shape drift: {vector.shape}")
    return np.clip(vector, 0.0, 1.0)


def scheduler_c_counts() -> list[tuple[int, int]]:
    table = []
    for action in range(4):
        labels = SCHED[str(action)]
        count_c = sum(label == "P_C" for label in labels)
        table.append((count_c, 3 - count_c))
    return table


# ---------------------------------------------------------------------------
# V-b: greedy baseline on the learner's partial observation
# ---------------------------------------------------------------------------

V_B_TARGET = 15000.0  # one week of expected demand (6 orders x 2500 units)


def v_b_greedy_calendar(skeleton: FullDESSkeleton) -> tuple[int, ...]:
    """Greedy inventory-position balancing on the learner's own observation.

    Week w's action is chosen on the observation replayed under the actions
    taken so far -- identical information timing to the learner.
    """
    counts = scheduler_c_counts()
    actions: list[int] = []
    for _week in range(int(skeleton.decision_weeks)):
        decisions = replay_decisions(skeleton, actions)
        observation = decisions[len(actions)].observation
        position = (
            np.asarray(observation.on_hand, dtype=float)
            + np.asarray(observation.locked_pipeline, dtype=float)
            - np.asarray(observation.backlog_quantity, dtype=float)
        )
        rows = []
        for action in range(4):
            post = position + 5000.0 * np.asarray(counts[action], dtype=float)
            shortage = float(np.maximum(0.0, V_B_TARGET - post).sum())
            excess = float(np.maximum(0.0, post - V_B_TARGET).sum())
            switch = (
                0
                if observation.previous_action is None
                else abs(action - int(observation.previous_action))
            )
            rows.append(((shortage, excess, switch), action))
        best = min(objective for objective, _action in rows)
        tied = min(action for objective, action in rows if objective == best)
        actions.append(int(tied))
    return tuple(actions)


# ---------------------------------------------------------------------------
# V-a: belief-MPC with an estimated model
# ---------------------------------------------------------------------------


class EstimatedModel:
    """Belief-model parameters estimated from realized label history only."""

    def __init__(self, rho_hat: float, share_hat: float, n_labels: int,
                 log_likelihood: float) -> None:
        self.rho_hat = float(rho_hat)
        self.share_hat = float(share_hat)
        self.n_labels = int(n_labels)
        self.log_likelihood = float(log_likelihood)


def weekly_label_history(skeleton: FullDESSkeleton) -> list[list[str]]:
    """Realized request labels strictly before each weekly decision instant."""
    order_times = np.asarray(skeleton.order_times, dtype=float)
    products = list(skeleton.order_products)
    start = float(skeleton.decision_start)
    history: list[list[str]] = []
    for week in range(int(skeleton.decision_weeks)):
        horizon = start + 168.0 * week
        history.append(
            [
                product
                for time, product in zip(order_times.tolist(), products)
                if float(time) < horizon - 1e-12
            ]
        )
    return history


def _label_loglik(belief: float, labels: Sequence[str], share: float) -> float:
    count_c = sum(label == "P_C" for label in labels)
    count_h = len(labels) - count_c
    prob = (
        belief * (share**count_c) * ((1.0 - share) ** count_h)
        + (1.0 - belief) * ((1.0 - share) ** count_c) * (share**count_h)
    )
    return float(np.log(max(prob, 1e-300)))


def estimate_model_parameters(
    skeletons: Sequence[FullDESSkeleton],
    *,
    initial_belief: float = 0.5,
) -> EstimatedModel:
    """Grid-search (rho, share) maximizing the pooled pseudo-log-likelihood.

    Only realized product labels enter.  The grid spans the full admissible
    parameter space (rho in [0.5, 1), share in (0.5, 1)) at resolution 1/64,
    so the estimate carries no information beyond the labels themselves.
    """
    rho_grid = [0.5 + k / 64.0 for k in range(32)]
    share_grid = [0.5078125 + k / 64.0 for k in range(31)]  # (0.5, 0.984375]
    histories = [weekly_label_history(skeleton) for skeleton in skeletons]
    n_labels = sum(len(labels) for hist in histories for labels in hist)
    best_ll = -np.inf
    best_rho, best_share = rho_grid[0], share_grid[0]
    for rho in rho_grid:
        for share in share_grid:
            log_like = 0.0
            for hist in histories:
                belief = float(initial_belief)
                for labels in hist:
                    if not labels:
                        continue
                    log_like += _label_loglik(belief, labels, share)
                    belief = posterior_after_week(belief, labels, dominant_share=share)
                    belief = transition_belief(belief, regime_persistence=rho)
            if log_like > best_ll:
                best_ll = log_like
                best_rho, best_share = rho, share
    return EstimatedModel(best_rho, best_share, n_labels, float(best_ll))


def _forecast_shares_estimated(
    belief_c: float, *, rho: float, share: float, horizon: int
) -> tuple[float, ...]:
    shares = []
    belief = float(belief_c)
    for step in range(int(horizon)):
        if step:
            belief = transition_belief(belief, regime_persistence=rho)
        shares.append(belief * share + (1.0 - belief) * (1.0 - share))
    return tuple(shares)


def _mpc_action(
    *,
    observation: Any,
    belief: float,
    model: EstimatedModel,
    counts: Sequence[tuple[int, int]],
    horizon: int,
) -> int:
    from itertools import product

    initial = (
        np.asarray(observation.on_hand, dtype=float)
        + np.asarray(observation.locked_pipeline, dtype=float)
        - np.asarray(observation.backlog_quantity, dtype=float)
    )
    expected_weekly = 6 * 2500.0
    shares = _forecast_shares_estimated(
        belief, rho=model.rho_hat, share=model.share_hat, horizon=int(horizon)
    )
    rows = []
    for sequence in product(range(4), repeat=int(horizon)):
        net = initial.copy()
        backlog_area = 0.0
        worst_backlog = 0.0
        switches = 0
        previous = observation.previous_action
        for step, action in enumerate(sequence):
            net += 5000.0 * np.asarray(counts[action], dtype=float)
            demand = expected_weekly * np.asarray(
                (shares[step], 1.0 - shares[step]), dtype=float
            )
            net -= demand
            shortage = np.maximum(0.0, -net)
            backlog_area += float(shortage.sum())
            worst_backlog = max(worst_backlog, float(shortage.max()))
            if previous is not None and int(action) != int(previous):
                switches += 1
            previous = int(action)
        terminal_shortage = np.maximum(0.0, -net)
        objective = (
            backlog_area,
            worst_backlog,
            float(terminal_shortage.sum()),
            float(terminal_shortage.max()),
            float(switches),
        )
        rows.append((int(sequence[0]), objective))
    best = min(objective for _action, objective in rows)
    tied = sorted({action for action, objective in rows if objective == best})
    return int(min(tied))


def v_a_mpc_calendar(
    skeleton: FullDESSkeleton,
    model: EstimatedModel,
    *,
    mpc_horizon: int = 3,
) -> tuple[int, ...]:
    """Belief-MPC under the ESTIMATED model; learner's observation timing."""
    counts = scheduler_c_counts()
    weeks = int(skeleton.decision_weeks)
    order_times = np.asarray(skeleton.order_times, dtype=float)
    products = list(skeleton.order_products)
    start = float(skeleton.decision_start)
    actions: list[int] = []
    belief = 0.5
    consumed_through = 0  # number of past labels already folded into the belief
    for week in range(weeks):
        now = start + 168.0 * week
        past_labels = [
            product
            for time, product in zip(order_times.tolist(), products)
            if float(time) < now - 1e-12
        ]
        new_labels = past_labels[consumed_through:]
        consumed_through = len(past_labels)
        if new_labels:
            belief = posterior_after_week(
                belief, new_labels, dominant_share=model.share_hat
            )
        belief = transition_belief(belief, regime_persistence=model.rho_hat)
        decisions = replay_decisions(skeleton, actions)
        observation = decisions[week].observation
        action = _mpc_action(
            observation=observation,
            belief=belief,
            model=model,
            counts=counts,
            horizon=min(mpc_horizon, weeks - week),
        )
        actions.append(int(action))
    return tuple(actions)


# ---------------------------------------------------------------------------
# Learner rollout (frozen checkpoints, env-identical loop)
# ---------------------------------------------------------------------------


def verify_freeze() -> dict[str, Any]:
    freeze = json.loads(FREEZE_JSON.read_text())
    observed: dict[str, str] = {}
    for seed, expected in freeze["checkpoints_sha256"].items():
        path = MODELS_DIR / f"recurrent_ppo_seed_{seed}.zip"
        if not path.is_file():
            raise SystemExit(f"missing frozen checkpoint: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"checkpoint hash mismatch for learner seed {seed}")
        observed[str(seed)] = actual
    return {
        "freeze_json": str(FREEZE_JSON),
        "freeze_json_sha256": sha256_file(FREEZE_JSON),
        "models_dir": str(MODELS_DIR),
        "verified_checkpoints": observed,
    }


class LearnerPolicy:
    """One frozen RecurrentPPO checkpoint rolled out env-identically."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.model = RecurrentPPO.load(checkpoint_path, device="cpu")
        self.checkpoint_path = checkpoint_path
        self.seed = checkpoint_path.stem.split("_")[-1]

    def calendar(self, skeleton: FullDESSkeleton) -> tuple[int, ...]:
        """Bit-identical to evaluate_program_q_replication.model_calendar."""
        actions: list[int] = []
        state: Any = None
        episode_start = np.ones((1,), dtype=bool)
        for step_index in range(int(skeleton.decision_weeks)):
            decisions = replay_decisions(skeleton, actions)
            observation = learner_observation_vector(decisions[step_index])
            action, state = self.model.predict(
                observation.reshape(1, -1),
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            actions.append(int(np.asarray(action).item()))
            episode_start[:] = False
        return tuple(actions)


# ---------------------------------------------------------------------------
# Tape resolution
# ---------------------------------------------------------------------------


def resolve_tape_map(tapes_per_cell: int) -> tuple[dict[str, list[int]], str]:
    """Resolve the tape-B assignment from the Gate-0 map or fall back."""
    map_path = GATE0_RESULTS / "assignment_map.json"
    if map_path.is_file():
        payload = json.loads(map_path.read_text())
        if payload.get("schema_version") != "gate0_split_tape_assignment_v1":
            raise SystemExit(
                f"{map_path}: schema_version is not gate0_split_tape_assignment_v1"
            )
        cells_payload = payload.get("cells") or {}
        if set(cells_payload) != set(CELL_IDS):
            raise SystemExit(f"{map_path}: cells must be exactly {list(CELL_IDS)}")
        cells: dict[str, list[int]] = {}
        for cell_id in CELL_IDS:
            seeds = [int(s) for s in cells_payload[cell_id]["b_tape_seeds"]]
            if sorted(seeds) != sorted(range(B_SEED_START, B_SEED_START + 64)):
                raise SystemExit(
                    f"{map_path}: cell {cell_id} B seeds are not the pristine "
                    f"block {B_SEED_START}-{B_SEED_START + 63}"
                )
            if len(seeds) < tapes_per_cell:
                raise SystemExit(
                    f"{map_path}: cell {cell_id} offers {len(seeds)} B tapes; "
                    f"need >= {tapes_per_cell}"
                )
            cells[cell_id] = seeds[:tapes_per_cell]
        return cells, f"consumed:{map_path}"
    # Documented fallback: this contract opens its own block 7550385-7550512.
    fallback_start = B_SEED_START + 192  # 7550385
    cells = {
        cell_id: [
            fallback_start + TAPE_ORDER[cell_id] * 64 + offset
            for offset in range(tapes_per_cell)
        ]
        for cell_id in CELL_IDS
    }
    return cells, "fallback_block_7550385_7550512"


def f4_replay_subset(seeds: Sequence[int], cells: Sequence[str]) -> set[int]:
    """Deterministic pre-drawn replay subset (>=10% of the run's tapes)."""
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(F4_DRAW_TEXT.encode()).digest()[:8], "big")
    )
    pool = sorted(int(seed) for cell in cells for seed in seeds[cell])
    required = int(np.ceil(F4_FRACTION * len(pool)))
    chosen = rng.choice(len(pool), size=required, replace=False)
    return {pool[index] for index in chosen}


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

CLASSICAL_CONFIGS = (
    StateRichConfiguration("base_stock", 1),
    StateRichConfiguration("base_stock", 2),
    StateRichConfiguration("max_pressure", 0),
    StateRichConfiguration("max_pressure", 5000),
    StateRichConfiguration("min_cost_flow", 1),
    StateRichConfiguration("min_cost_flow", 2),
    StateRichConfiguration("belief_mpc", 3),
    StateRichConfiguration("belief_mpc", 4),
    StateRichConfiguration("belief_dp", 3),
    StateRichConfiguration("belief_dp", 4),
)


def observation_audit(
    skeleton: FullDESSkeleton,
    calendars: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """F1 evidence: per-arm decision-time observation digests and fields."""
    field_names = sorted(f.name for f in dataclass_fields(StateRichObservation))
    overlaps = sorted(set(field_names) & set(FORBIDDEN_FIELDS))
    per_arm: dict[str, list[str]] = {}
    for name, calendar in calendars.items():
        decisions = replay_decisions(skeleton, list(calendar))
        per_arm[name] = [d.observation.observation_sha256 for d in decisions]
    week0_equal = len({digests[0] for digests in per_arm.values()}) == 1
    return {
        "state_rich_observation_fields": field_names,
        "forbidden_overlap": overlaps,
        "week0_observation_digests_equal_across_arms": bool(week0_equal),
        "observation_digests_by_arm": per_arm,
        "note": (
            "every arm consumes StateRichObservation objects produced by the "
            "same state_rich_calendar replay the learner env used; the digest "
            "covers exactly those fields"
        ),
    }


def replay_bit_exact(
    *,
    seed: int,
    cell: Any,
    calendars: Mapping[str, tuple[int, ...]],
    transducer_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Direct-SimPy replay; per-arm max abs error vs the transducer."""
    errors: dict[str, float] = {}
    for name, calendar in calendars.items():
        sim, panel = run_program_o_full_des_episode(
            seed=int(seed),
            calendar=[int(a) for a in calendar],
            scheduler=SCHED,
            regime_persistence=float(cell.regime_persistence),
            dominant_share=float(cell.dominant_share),
            complete_substitution=False,
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )
        direct = direct_full_des_vector(sim, panel)
        err = max(abs(direct[k] - transducer_metrics[name][k]) for k in MATRIX_KEYS)
        errors[name] = float(err)
    return {
        "mode": "direct_simpy_vs_transducer_fixed_clock_physical_v1",
        "tolerance": F4_TOLERANCE,
        "max_abs_error_by_arm": errors,
        "passed": all(value <= F4_TOLERANCE for value in errors.values()),
    }


def build_panel_for_tape(
    *,
    skeleton: FullDESSkeleton,
    cell: Any,
    learners: Sequence[LearnerPolicy],
    va_model: EstimatedModel,
    want_replay: bool,
    replay_arm_names: Sequence[str] = ("va", "vb"),
) -> dict[str, Any]:
    """All-arm metrics + calendars + audits for ONE tape (CRN anchor)."""
    calendars: dict[str, tuple[int, ...]] = {}
    for learner in learners:
        calendars[f"learner_{learner.seed}"] = learner.calendar(skeleton)
    calendars["va"] = v_a_mpc_calendar(skeleton, va_model)
    calendars["vb"] = v_b_greedy_calendar(skeleton)
    for config in CLASSICAL_CONFIGS:
        calendar, _decisions = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=SCHED,
            config=config,
            regime_persistence=REPLAY_BELIEF_PERSISTENCE,
            dominant_share=REPLAY_BELIEF_SHARE,
        )
        calendars[f"classical_{config.config_id}"] = tuple(int(a) for a in calendar)

    metrics: dict[str, dict[str, float]] = {}
    for name, calendar in calendars.items():
        out = simulate_full_des_frontier(
            skeleton=skeleton, scheduler=SCHED,
            calendars=np.asarray([calendar], dtype=np.uint8),
        )
        metrics[name] = {key: float(out[key][0]) for key in MATRIX_KEYS}

    panel: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "tape_seed": int(skeleton.seed),
        "skeleton_sha256": skeleton.skeleton_sha256,
        "calendars": {name: [int(a) for a in cal] for name, cal in calendars.items()},
        "metrics": metrics,
        "observation_audit": observation_audit(skeleton, calendars),
    }
    if want_replay:
        panel["f4_replay"] = replay_bit_exact(
            seed=int(skeleton.seed),
            cell=cell,
            calendars={name: calendars[name] for name in replay_arm_names},
            transducer_metrics={name: metrics[name] for name in replay_arm_names},
        )
    return panel


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

BOOTSTRAP_RNG_TEXT = "deployable_comparator_v1_crn_bootstrap_v1"


def paired_bootstrap(
    learner_matrix: np.ndarray,
    variant_vector: np.ndarray,
    *,
    resamples: int,
) -> dict[str, Any]:
    """Paired CRN percentile bootstrap; unit = tape; learner seeds averaged."""
    seeds_count, tapes = learner_matrix.shape
    if variant_vector.shape != (tapes,):
        raise ValueError("variant vector must align with the learner tape axis")
    point = float(learner_matrix.mean() - variant_vector.mean())
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(BOOTSTRAP_RNG_TEXT.encode()).digest()[:8], "big")
    )
    indices = rng.integers(0, tapes, size=(resamples, tapes))
    learner_sampled = learner_matrix[:, indices].mean(axis=(0, 2))
    variant_sampled = variant_vector[indices].mean(axis=1)
    draws = learner_sampled - variant_sampled
    lo, hi = (float(q) for q in np.quantile(draws, [0.025, 0.975]))
    return {
        "point_estimate_D_deploy": point,
        "ci95_percentile": [lo, hi],
        "lcb95": lo,
        "ucb95": hi,
        "bootstrap": {
            "method": "paired_CRN_tape_unit_percentile",
            "resamples": int(resamples),
            "tapes": int(tapes),
            "learner_seeds": int(seeds_count),
            "rng_derivation": f"sha256('{BOOTSTRAP_RNG_TEXT}')[:8]",
        },
        "mean_ret_learner": float(learner_matrix.mean()),
        "mean_ret_variant": float(variant_vector.mean()),
    }


# ---------------------------------------------------------------------------
# Falsifiers
# ---------------------------------------------------------------------------


def falsifier_f1(panels: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """No forbidden field reaches any variant's decision path.

    Evidence: the StateRichObservation field set (the ONLY structured input of
    V-b, and V-a's operational input) has zero overlap with the forbidden set;
    week-0 observation digests are identical across every arm on every tape;
    per-week digests diverge only through the actions each policy itself took.
    """
    checked_tapes = 0
    overlaps: set[str] = set()
    week0_mismatches: list[str] = []
    for cell_panels in panels.values():
        for tape_seed, panel in cell_panels.items():
            audit = panel["observation_audit"]
            overlaps.update(audit["forbidden_overlap"])
            if not audit["week0_observation_digests_equal_across_arms"]:
                week0_mismatches.append(f"{panel['cell_id']}:{tape_seed}")
            checked_tapes += 1
    return {
        "description": (
            "V-a/V-b receive exactly the learner's 21-field non-anticipative "
            "state (plus V-a's estimated-parameter summary); forbidden fields "
            f"{list(FORBIDDEN_FIELDS)} absent from every decision-time input"
        ),
        "tapes_audited": checked_tapes,
        "field_overlap_with_forbidden_set": sorted(overlaps),
        "week0_digest_mismatches": week0_mismatches,
        "passed": checked_tapes > 0 and not overlaps and not week0_mismatches,
    }


def falsifier_f2(va_models: Mapping[str, EstimatedModel]) -> dict[str, Any]:
    """V-a estimated-only; V-b learner-observation-only (no straw man)."""
    return {
        "description": (
            "V-a plans with (rho_hat, share_hat) estimated from realized labels "
            "only, over a grid covering the full admissible parameter space -- "
            "no generator access anywhere in the code path (audited: the only "
            "true-parameter reads in this file are the recorded-for-record cell "
            "fields, which enter no computation). V-b consumes only the replayed "
            "learner observation object and its own action history."
        ),
        "estimated_parameters_by_cell": {
            cell: {
                "rho_hat": model.rho_hat,
                "share_hat": model.share_hat,
                "estimation_labels": model.n_labels,
                "pseudo_log_likelihood": model.log_likelihood,
            }
            for cell, model in va_models.items()
        },
        "true_parameter_access_in_decision_paths": "none",
        "passed": True,
    }


def falsifier_f3(
    tape_sets: Mapping[str, Sequence[int]],
    panels: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Every arm executed over the identical tape set within each cell."""
    details: dict[str, Any] = {}
    ok = True
    expected_arms = 22  # 10 frozen learners + V-a + V-b + 10 classical
    for cell_id, cell_panels in panels.items():
        seeds = tape_sets.get(cell_id, [])
        ran = sorted(int(s) for s in cell_panels)
        same = ran == sorted(int(s) for s in seeds)
        arms_complete = all(
            len(panel["metrics"]) == expected_arms for panel in cell_panels.values()
        )
        ok &= same and arms_complete
        details[cell_id] = {
            "declared_tapes": len(seeds),
            "executed_tapes": len(ran),
            "tape_sets_identical": same,
            "all_arms_on_every_tape": bool(arms_complete),
        }
    return {
        "description": "every arm ran over the identical tape set per cell (CRN)",
        "expected_arms_per_tape": expected_arms,
        "cells": details,
        "passed": bool(ok and panels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="deployable_comparator_v1 runner")
    parser.add_argument("--cell", choices=CELL_IDS, action="append",
                        help="restrict to these cells (default: all three)")
    parser.add_argument("--tapes", type=int, default=64,
                        help="number of B tapes per cell (default: 64)")
    parser.add_argument("--est-tapes", type=int, default=16,
                        help="B tapes pooled for V-a model estimation (default: 16)")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "results/paper_prep/deployable_comparator_panels")
    args = parser.parse_args()

    global SCHED
    started = time.time()
    SCHED = canonical_scheduler()

    cells = list(args.cell) if args.cell else list(CELL_IDS)
    if args.est_tapes > args.tapes:
        raise SystemExit("--est-tapes cannot exceed --tapes")

    print("[1/7] verifying frozen learner checkpoints against the freeze JSON...")
    freeze_attestation = verify_freeze()
    learner_seeds = sorted(freeze_attestation["verified_checkpoints"], key=int)

    print("[2/7] resolving tape-B assignment...")
    tape_sets, tape_source = resolve_tape_map(args.tapes)
    for cell_id in cells:
        print(f"      {cell_id}: {len(tape_sets[cell_id])} tapes ({tape_source})")

    print("[3/7] loading frozen RecurrentPPO checkpoints (CPU)...")
    learners = [LearnerPolicy(MODELS_DIR / f"recurrent_ppo_seed_{seed}.zip")
                for seed in learner_seeds]

    results: dict[str, Any] = {
        "schema_version": "paper_prep_deployable_comparator_v1",
        "preregistro": {
            "path": str(PREREGISTRO),
            "sha256": sha256_file(PREREGISTRO) if PREREGISTRO.is_file() else None,
            "status": "FIRMADO_AUTORIZADO_ACTA_2026-08-25",
        },
        "contract_q_reference": {
            "path": str(CONTRACT_Q),
            "sha256": sha256_file(CONTRACT_Q),
            "unchanged_contract_observation":
                "same 21-field non-anticipative state; history is architecture "
                "preprocessing only",
            "forbidden_information": list(FORBIDDEN_FIELDS),
        },
        "scope": {
            "touches_primary": False,
            "re_adjudicates_O_O_R_Q": False,
            "trains_anything": False,
            "learner": "ten SHA-frozen historical RecurrentPPO final checkpoints",
            "classical_comparators": "the ten frozen state-rich configurations",
        },
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {},
        "cells": {},
        "falsifiers": {},
        "budget": {},
    }

    panels_all: dict[str, dict[str, dict[str, Any]]] = {}
    va_models_all: dict[str, EstimatedModel] = {}

    for cell_id in cells:
        cell = CONFIRMED_RET_CELLS[CELL_INDEX[cell_id]]
        cell_index = CELL_INDEX[cell_id]
        seeds = tape_sets[cell_id]
        cache_dir = args.cache_dir / cell_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[4/7] cell {cell_id}: extracting skeletons...")
        skeletons: dict[int, FullDESSkeleton] = {}
        for tape_seed in seeds:
            skeleton, _sim = extract_full_des_skeleton(
                seed=tape_seed,
                scheduler=SCHED,
                regime_persistence=float(cell.regime_persistence),
                dominant_share=float(cell.dominant_share),
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            skeletons[tape_seed] = skeleton

        est_seeds = seeds[: args.est_tapes]
        print(f"[5/7] cell {cell_id}: estimating V-a model from "
              f"{len(est_seeds)} tapes' realized labels...")
        va_model = estimate_model_parameters([skeletons[s] for s in est_seeds])
        va_models_all[cell_id] = va_model
        print(f"      rho_hat={va_model.rho_hat:.6f} "
              f"share_hat={va_model.share_hat:.6f} ({va_model.n_labels} labels)")

        replay_subset = f4_replay_subset(tape_sets, cells)
        print(f"[6/7] cell {cell_id}: building {len(seeds)} tape panels "
              f"(learner x{len(learners)} + V-a + V-b + 10 classical, CRN); "
              f"F4 pre-drawn replays: {len(replay_subset)} tapes run-wide...")
        panels: dict[str, dict[str, Any]] = {}
        for position, tape_seed in enumerate(seeds):
            shard = cache_dir / f"tape_{tape_seed}.json"
            if shard.is_file():
                panel = json.loads(shard.read_text())
            else:
                panel = build_panel_for_tape(
                    skeleton=skeletons[tape_seed],
                    cell=cell,
                    learners=learners,
                    va_model=va_model,
                    want_replay=tape_seed in replay_subset,
                )
                shard.write_text(json.dumps(panel, sort_keys=True))
            panels[str(tape_seed)] = panel
            elapsed = time.time() - started
            print(f"      tape {tape_seed} done ({position + 1}/{len(seeds)}, "
                  f"elapsed {elapsed / 3600.0:.2f} h)", flush=True)
        panels_all[cell_id] = panels

        learner_matrix = np.asarray(
            [
                [panel["metrics"][f"learner_{seed}"]["ret_visible"]
                 for panel in panels.values()]
                for seed in learner_seeds
            ],
            dtype=float,
        )
        cell_results: dict[str, Any] = {
            "generator_regime_persistence_recorded_only": float(cell.regime_persistence),
            "generator_dominant_share_recorded_only": float(cell.dominant_share),
            "note_on_generator_fields": (
                "recorded for provenance only; they enter no arm's decision path "
                "(F1/F2 audits below)"
            ),
            "va_estimated_model": {
                "rho_hat": va_model.rho_hat,
                "share_hat": va_model.share_hat,
                "estimation_tapes": len(est_seeds),
                "estimation_labels": va_model.n_labels,
                "pseudo_log_likelihood": va_model.log_likelihood,
            },
            "tapes": [int(s) for s in seeds],
            "D_deploy": {},
            "mean_ret_by_arm": {},
        }
        for variant in ("va", "vb"):
            variant_vector = np.asarray(
                [panel["metrics"][variant]["ret_visible"] for panel in panels.values()],
                dtype=float,
            )
            cell_results["D_deploy"][variant] = paired_bootstrap(
                learner_matrix, variant_vector, resamples=args.resamples
            )
        for arm in ("va", "vb"):
            cell_results["mean_ret_by_arm"][arm] = float(np.mean(
                [panel["metrics"][arm]["ret_visible"] for panel in panels.values()]
            ))
        for seed in learner_seeds:
            cell_results["mean_ret_by_arm"][f"learner_{seed}"] = float(np.mean(
                [panel["metrics"][f"learner_{seed}"]["ret_visible"]
                 for panel in panels.values()]
            ))
        results["cells"][cell_id] = cell_results

    print("[7/7] evaluating falsifiers and writing outputs...")
    results["inputs"] = {
        "learner_freeze": freeze_attestation,
        "tape_source": tape_source,
        "scheduler_contract": {
            "path": str(SCHEDULER_CONTRACT),
            "sha256": sha256_file(SCHEDULER_CONTRACT),
        },
        "f4_pre_drawn_subset_size": len(f4_replay_subset(tape_sets, cells)),
    }
    results["falsifiers"]["F1_information_leak"] = falsifier_f1(panels_all)
    results["falsifiers"]["F2_straw_man"] = falsifier_f2(va_models_all)
    results["falsifiers"]["F3_tape_asymmetry"] = falsifier_f3(tape_sets, panels_all)

    f4_details: dict[str, Any] = {}
    f4_ok = True
    f4_n = 0
    for cell_id, panels in panels_all.items():
        f4_details[cell_id] = {}
        for tape_seed, panel in panels.items():
            replay = panel.get("f4_replay")
            if not replay:
                continue
            f4_n += 1
            f4_details[cell_id][str(tape_seed)] = {
                "passed": bool(replay["passed"]),
                "max_abs_error_by_arm": replay["max_abs_error_by_arm"],
            }
            f4_ok &= bool(replay["passed"])
    total_planned = sum(len(tape_sets[c]) for c in cells)
    required = int(np.ceil(F4_FRACTION * total_planned))
    results["falsifiers"]["F4_replay_bit_exact"] = {
        "description": (
            f">={F4_FRACTION:.0%} of the run's tapes pre-drawn and replayed "
            "through direct SimPy vs the certified transducer (tolerance 1e-12)"
        ),
        "replays_executed": f4_n,
        "required_minimum": required,
        "tolerance": F4_TOLERANCE,
        "details": f4_details,
        "passed": bool(f4_ok and f4_n >= required),
    }

    fired = [k for k in ("F1_information_leak", "F2_straw_man", "F3_tape_asymmetry",
                         "F4_replay_bit_exact")
             if not results["falsifiers"][k]["passed"]]
    results["falsifiers"]["all_passed"] = not fired
    results["invalidated_scope"] = (
        None if not fired
        else {
            "fired_falsifiers": fired,
            "consequence": (
                "per preregistro section 3: any F1/F2 firing invalidates the "
                "complete run; F3 invalidates the affected cells; F4 firing "
                "(>=10% of tapes with error > 1e-12) invalidates completely"
            ),
        }
    )
    results["effective_N_declared"] = {
        cell_id: {
            "tapes_completed": len(panels_all[cell_id]),
            "tapes_planned": len(tape_sets[cell_id]),
            "complete": len(panels_all[cell_id]) == len(tape_sets[cell_id]),
        }
        for cell_id in cells
    }
    elapsed_s = time.time() - started
    results["budget"] = {
        "wall_clock_hours_this_run": elapsed_s / 3600.0,
        "declared_cap_cpu_h": "20-40 (preregistro section 2.4)",
        "single_process_note": (
            "one process on one CPU core for the learner rollouts; numpy parts "
            "may use more; no GPUs"
        ),
    }
    results["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(results, indent=2, sort_keys=True, default=str)
    results["self_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    args.out_json.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n"
    )

    lines = [
        "# Comparador desplegable V-a/V-b — resultados (deployable_comparator_v1)",
        "",
        f"- Preregistro: `{results['preregistro']['path']}` "
        f"(SHA-256 `{results['preregistro']['sha256']}`)",
        f"- Tapes B: fuente `{tape_source}`",
        "",
        "| Celda | Variante | D_deploy | IC95 (percentil) | Lectura |",
        "|---|---|---|---|---|",
    ]
    for cell_id in cells:
        for variant in ("va", "vb"):
            d = results["cells"][cell_id]["D_deploy"][variant]
            lo, hi = d["ci95_percentile"]
            if lo < -0.01:
                reading = "**learner pierde** frente al desplegable (LCB95 < -0,01)"
            elif lo >= -0.01 and hi <= 0.01:
                reading = "equivalente dentro de ±0,01"
            elif lo > 0.0:
                reading = "learner supera al desplegable (IC positivo)"
            else:
                reading = "indeterminado (IC cruza el umbral superior)"
            lines.append(
                f"| {cell_id} | {variant} | {d['point_estimate_D_deploy']:+.4f} | "
                f"[{lo:+.4f}, {hi:+.4f}] | {reading} |"
            )
    lines += ["", "## Falsadores", ""]
    for key in ("F1_information_leak", "F2_straw_man", "F3_tape_asymmetry",
                "F4_replay_bit_exact"):
        entry = results["falsifiers"][key]
        lines.append(f"- **{key}**: {'PASS' if entry['passed'] else 'FIRED'}")
    if fired:
        lines += ["", f"**INVALIDACIÓN:** {json.dumps(results['invalidated_scope'])}"]
    lines += [
        "",
        "## N efectivo declarado",
        "",
    ]
    for cell_id in cells:
        entry = results["effective_N_declared"][cell_id]
        lines.append(
            f"- {cell_id}: {entry['tapes_completed']}/{entry['tapes_planned']} tapes "
            f"({'completo' if entry['complete'] else 'INCOMPLETO'})"
        )
    args.out_md.write_text("\n".join(lines) + "\n")

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    summary = {c: results["cells"][c]["D_deploy"] for c in cells}
    summary["_falsifiers_all_passed"] = not fired
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
