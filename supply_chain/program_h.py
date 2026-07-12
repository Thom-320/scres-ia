"""Program H O0 belief-state and regret-aware policies on frozen Program G physics."""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .program_g import (
    ACTIONS, CSSU_CAP, DEMAND_DAYS, MULT, SB_INITIAL, S1_DAILY, STATIONARY, TEMPO,
    _week_step, metrics_all, observe,
)

ARM = "TRS"
PRIOR = np.array([STATIONARY[s] for s in TEMPO], dtype=float)


def tempo_belief(tape, week: int) -> np.ndarray:
    """Exact O0 tempo marginal for the frozen four-week episode.

    Minimum dwell is four weeks, so each CSSU's tempo is constant over this episode. Signals
    whose target lies outside the episode were generated as non-surge and carry no tempo
    information; they are intentionally ignored.
    """
    q = float(tape.cell["signal_q"])
    lead = int(tape.cell["lead_weeks"])
    out = np.tile(PRIOR, (2, 1))
    for i in range(2):
        likelihood = np.ones(3)
        for u in range(week + 1):
            if u + lead >= tape.weeks:
                continue
            y = int(tape.signal[u, i])
            p_one = np.array([1.0 - q, 1.0 - q, q])
            likelihood *= p_one if y else (1.0 - p_one)
        post = PRIOR * likelihood
        out[i] = post / post.sum()
    return out


def o0_features(inv, sb, tape, week: int) -> np.ndarray:
    base = observe(inv, sb, tape, week).astype(float)
    belief = tempo_belief(tape, week).reshape(-1)
    history = np.zeros((tape.weeks, 2), dtype=float)
    history[:week + 1] = tape.signal[:week + 1]
    return np.concatenate([base, belief, history.reshape(-1)])


def _state_after_prefix(tape, prefix: Sequence[str]):
    inv = np.zeros(2); sb = float(SB_INITIAL)
    for w, action in enumerate(prefix):
        inv, sb, _ = _week_step(inv, sb, action, tape.demand[w], tape.r22[w], True)
    return inv, sb


def exact_sequence_scores(tape) -> dict[tuple[str, ...], float]:
    return {seq: metrics_all(tape, seq, ARM)["ret_order"]
            for seq in itertools.product(ACTIONS, repeat=tape.weeks)}


def fit_regret_q_policy(tapes, *, random_state: int = 20260713):
    """Fit action-value regressors to exact counterfactual terminal ReT, not action labels."""
    rows, targets = [], []
    for tape in tapes:
        scores = exact_sequence_scores(tape)
        for w in range(tape.weeks):
            for prefix in itertools.product(ACTIONS, repeat=w):
                inv, sb = _state_after_prefix(tape, prefix)
                rows.append(o0_features(inv, sb, tape, w))
                q = []
                for action in ACTIONS:
                    compatible = [v for seq, v in scores.items()
                                  if seq[:w] == prefix and seq[w] == action]
                    q.append(max(compatible))
                targets.append(q)
    X, Y = np.asarray(rows), np.asarray(targets)
    models = []
    for a in range(len(ACTIONS)):
        # Single-process execution avoids macOS joblib workers lingering after the fully
        # materialized verdict; it does not change the frozen estimator or random state.
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                                    random_state=random_state + a, n_jobs=1)
        model.fit(X, Y[:, a]); models.append(model)
    return models


def regret_q_actions(tape, models) -> tuple[str, ...]:
    inv = np.zeros(2); sb = float(SB_INITIAL); actions = []
    for w in range(tape.weeks):
        x = o0_features(inv, sb, tape, w).reshape(1, -1)
        q = [float(m.predict(x)[0]) for m in models]
        action = ACTIONS[int(np.argmax(q))]
        actions.append(action)
        inv, sb, _ = _week_step(inv, sb, action, tape.demand[w], tape.r22[w], True)
    return tuple(actions)


def expected_weekly_demand(tape, week: int) -> np.ndarray:
    belief = tempo_belief(tape, week)
    multipliers = np.array([MULT["low"], MULT["routine"], float(tape.cell["surge_mult"])])
    # Expected daily base is 2500, divided equally, six days.
    return belief @ multipliers * (2500.0 / 2.0 * DEMAND_DAYS)


def belief_rollout_actions(tape, *, lookahead: int | None = None) -> tuple[str, ...]:
    """Point-based posterior-mean rollout; enumerates all actions in its frozen horizon."""
    inv = np.zeros(2); sb = float(SB_INITIAL); chosen = []
    for w in range(tape.weeks):
        horizon = tape.weeks - w if lookahead is None else min(lookahead, tape.weeks - w)
        demand_forecast = [expected_weekly_demand(tape, min(w + h, tape.weeks - 1))
                           for h in range(horizon)]
        best_seq, best_loss = None, np.inf
        for seq in itertools.product(ACTIONS, repeat=horizon):
            ii, ss, loss = inv.copy(), sb, 0.0
            for h, action in enumerate(seq):
                # O0 does not expose route state; use its frozen marginal expectation by
                # evaluating the no-closure modal trajectory, not the actual future tape.
                ii, ss, unmet = _week_step(ii, ss, action, demand_forecast[h], np.zeros(2), True)
                loss += unmet
            if loss < best_loss:
                best_seq, best_loss = seq, loss
        action = best_seq[0]
        chosen.append(action)
        inv, sb, _ = _week_step(inv, sb, action, tape.demand[w], tape.r22[w], True)
    return tuple(chosen)


def full_information_oracle_actions(tape) -> tuple[str, ...]:
    scores = exact_sequence_scores(tape)
    return max(scores, key=scores.get)


def filter_log_loss(tapes) -> tuple[float, float]:
    losses, prior_losses = [], []
    for tape in tapes:
        for w in range(tape.weeks):
            belief = tempo_belief(tape, w)
            for i in range(2):
                truth = int(tape.z[w, i])
                losses.append(-np.log(max(belief[i, truth], 1e-12)))
                prior_losses.append(-np.log(max(PRIOR[truth], 1e-12)))
    return float(np.mean(losses)), float(np.mean(prior_losses))
