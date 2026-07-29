"""Track B port of ``FutureCreditRewardWrapper`` (from
``scripts/run_cf20_learning_repair.py``, Track A/continuous_its lane).

Potential-based reward shaping (Ng, Harada & Russell 1999): gives IMMEDIATE
reward signal for reducing backlog/lost-order exposure and increasing
downstream (theatre) coverage, instead of waiting weeks for the eventual
Excel-ReT payoff to propagate back through PPO's discounted credit. This is
isolated arm 1 of the two the user picked (the other is retargeting the
belief-encoder's prediction risk, in ``scripts/build_track_b_v10_belief_pretrain_dataset.py``
--target-risk). It does NOT touch observation or architecture -- only the
TRAINING reward. Evaluation always uses the unmodified Garrido Excel ReT
(``order_ret_excel_mean``), never the shaped reward.

Two modes, same semantics as the Track A version:

- ``ReT_excel_delta_bootstrap``: per-step penalty proportional to pending
  backorder qty and lost-order count, small coverage bonus. Not policy
  invariant by construction (unlike PBRS), but gives an immediate signal
  correlated with the eventual ReT hit instead of waiting for it to resolve.
- ``ReT_excel_terminal_shaped``: potential-based shaping,
  ``r' = r + gamma * Phi(s') - Phi(s)``, with
  ``Phi(s) = -alpha*pending_norm - beta*lost_norm + eta*theatre_coverage``.
  Policy-invariant in the tabular-optimal-policy sense (Ng-Harada-Russell): in
  the limit of exact optimization, the optimal policy is unchanged; in
  practice with function approximation and finite training it primarily
  speeds up and stabilizes credit assignment, which is exactly the problem
  diagnosed in ``docs/EXCEL_REWARD_PREPOSITIONING_AUDIT_2026-06-27.md`` and
  ``docs/TRACK_B_PREVENTIVE_LEARNING_ROOT_CAUSE_AND_ALTERNATIVES_2026-07-04.md``.

``rations_theatre.level / 1e5`` reuses the exact normalization already used
for this field elsewhere in the observation pipeline (supply_chain.py:1311),
for consistency rather than inventing a new constant.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

SUPPORTED_MODES = ("ReT_excel_delta_bootstrap", "ReT_excel_terminal_shaped")


class TrackBFutureCreditRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        mode: str,
        pending_scale: float = 1_000_000.0,
        lost_scale: float = 100.0,
        pbrs_alpha: float = 1.0,
        pbrs_beta: float = 0.5,
        pbrs_eta: float = 0.05,
        pbrs_gamma: float = 0.99,
    ) -> None:
        super().__init__(env)
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown mode={mode!r}. Expected one of: {SUPPORTED_MODES}.")
        self.mode = mode
        self.pending_scale = float(pending_scale)
        self.lost_scale = float(lost_scale)
        self.pbrs_alpha = float(pbrs_alpha)
        self.pbrs_beta = float(pbrs_beta)
        self.pbrs_eta = float(pbrs_eta)
        self.pbrs_gamma = float(pbrs_gamma)
        self._prev_phi = 0.0

    def _sim(self):
        return getattr(self.unwrapped, "sim", None)

    def _state_terms(self) -> dict[str, float]:
        sim = self._sim()
        if sim is None:
            return {"pending_norm": 0.0, "lost_norm": 0.0, "coverage": 0.0}
        pending_norm = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0) / self.pending_scale
        lost_norm = float(getattr(sim, "total_unattended_orders", 0.0) or 0.0) / self.lost_scale
        try:
            coverage = float(sim.rations_theatre.level) / 1e5
        except Exception:
            coverage = 0.0
        return {
            "pending_norm": float(np.clip(pending_norm, 0.0, 10.0)),
            "lost_norm": float(np.clip(lost_norm, 0.0, 10.0)),
            "coverage": float(np.clip(coverage, 0.0, 1.0)),
        }

    def _phi(self) -> float:
        terms = self._state_terms()
        return (
            -self.pbrs_alpha * terms["pending_norm"]
            - self.pbrs_beta * terms["lost_norm"]
            + self.pbrs_eta * terms["coverage"]
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base_reward = float(reward)
        terms = self._state_terms()
        if self.mode == "ReT_excel_delta_bootstrap":
            shaped = (
                base_reward
                - 0.05 * terms["pending_norm"]
                - 0.02 * terms["lost_norm"]
                + 0.005 * terms["coverage"]
            )
        else:
            phi = self._phi()
            shaped = base_reward + self.pbrs_gamma * phi - self._prev_phi
            self._prev_phi = phi
        info = dict(info)
        info["future_credit_reward_mode"] = self.mode
        info["future_credit_base_reward"] = base_reward
        info["future_credit_shaped_reward"] = float(shaped)
        info["future_credit_terms"] = terms
        return obs, float(shaped), terminated, truncated, info


class TrackBBeliefConditionedFutureCreditRewardWrapper(gym.Wrapper):
    """Combined Arm 1 + Arm 2, v2 (per user's 5-point design review 2026-07-04).

    ``p_adv = max(0, (p_belief - p_base) / (1 - p_base))``: excess belief over
    the risk's own base rate, not the raw probability -- a risk with a high
    permanent base rate must not look "always imminent" and blanket-reward
    readiness forever; only genuinely elevated risk should.

    ``readiness = 0.5*theatre_coverage + 0.25*op10_slack + 0.25*op12_slack``:
    R22 destroys ops [4, 8, 10, 12] -- LOC/dispatch nodes, not just the final
    theatre buffer -- so readiness must reflect buffer at op10/op12 too.
    Reuses ``op10_queue_pressure_norm``/``op12_queue_pressure_norm``, already
    present in the v10 observation and already normalized as
    ``buffer_level / (dispatch_q_max * lookahead_cycles)`` -- high value means
    high buffer relative to need (slack), not stress, despite the name.

    ``Phi(s) = -alpha*pending_norm - beta*lost_norm
               + kappa*p_adv*readiness - rho*(1-p_adv)*resource_posture``

    The last term is the fix for the known failure mode (Ruta A, Arm 1): an
    agent that is simply "always ready" is not preventive, it is permanently
    expensive. Penalizing an expensive resource posture specifically when
    there is NO elevated risk belief (``1-p_adv`` weight) removes the reward
    for blanket readiness -- readiness only pays when ``p_adv`` is genuinely
    elevated. ``resource_posture`` reuses the same shift/op10/op12 "intensity"
    definition as ``row_from_step`` elsewhere in the audit scripts, for
    consistency, computed from the PREVIOUS step's realized action (a valid
    state descriptor, not the action about to be chosen).

    All components are logged into ``info`` (``belief_prob``, ``p_adv``,
    ``readiness``, ``resource_posture``, ``phi_terms``) so a post-hoc trace
    can distinguish "learned to anticipate" from "found another baseline
    posture" without re-deriving anything.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        encoder_path: str | Path,
        head_path: str | Path,
        belief_target_index: int = 0,
        belief_base_rate: float = 0.0607,
        pending_scale: float = 1_000_000.0,
        lost_scale: float = 100.0,
        pbrs_alpha: float = 1.0,
        pbrs_beta: float = 0.5,
        pbrs_kappa: float = 0.2,
        pbrs_rho: float = 0.05,
        pbrs_exposure: float = 0.0,
        pbrs_backlog_age: float = 0.0,
        pbrs_tail: float = 0.0,
        pbrs_gamma: float = 0.99,
        mode_label: str = "belief_conditioned_pbrs_v2",
        belief_mask_forecast: bool = False,
        features_dim: int = 64,
        hidden_width: int = 64,
    ) -> None:
        super().__init__(env)
        from scripts.belief_extractor import MLPBeliefExtractor
        from scripts.run_track_b_smoke import extract_downstream_multipliers
        from supply_chain.external_env_interface import get_observation_fields

        self.pending_scale = float(pending_scale)
        self.lost_scale = float(lost_scale)
        self.pbrs_alpha = float(pbrs_alpha)
        self.pbrs_beta = float(pbrs_beta)
        self.pbrs_kappa = float(pbrs_kappa)
        self.pbrs_rho = float(pbrs_rho)
        self.pbrs_exposure = float(pbrs_exposure)
        self.pbrs_backlog_age = float(pbrs_backlog_age)
        self.pbrs_tail = float(pbrs_tail)
        self.pbrs_gamma = float(pbrs_gamma)
        self.mode_label = str(mode_label)
        self.belief_mask_forecast = bool(belief_mask_forecast)
        self.belief_target_index = int(belief_target_index)
        self.belief_base_rate = float(belief_base_rate)
        self._extract_downstream_multipliers = extract_downstream_multipliers
        self._prev_phi = 0.0
        self._last_resource_posture = 0.0

        v10_fields = list(get_observation_fields("v10"))
        self._op10_pressure_idx = v10_fields.index("op10_queue_pressure_norm")
        self._op12_pressure_idx = v10_fields.index("op12_queue_pressure_norm")
        self._oldest_backorder_age_idx = v10_fields.index("oldest_backorder_age_norm")
        self._rolling_backorder_idx = v10_fields.index("rolling_backorder_rate_4w")
        self._backorder_rate_idx = v10_fields.index("backorder_rate")
        self._forecast_indices = tuple(
            v10_fields.index(name)
            for name in ("risk_forecast_48h_norm", "risk_forecast_168h_norm")
            if name in v10_fields
        )

        self._trunk = MLPBeliefExtractor(
            env.observation_space, features_dim=features_dim, hidden_width=hidden_width
        )
        self._trunk.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        self._trunk.eval()
        head_state = torch.load(head_path, map_location="cpu")
        n_targets = head_state["weight"].shape[0]
        self._head = torch.nn.Linear(features_dim, n_targets)
        self._head.load_state_dict(head_state)
        self._head.eval()

    def _sim(self):
        return getattr(self.unwrapped, "sim", None)

    def _belief_prob(self, obs: np.ndarray) -> float:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        if self.belief_mask_forecast:
            for idx in self._forecast_indices:
                if obs.shape[0] > idx:
                    obs[idx] = 0.0
        with torch.no_grad():
            x = torch.as_tensor(obs).reshape(1, -1)
            logits = self._head(self._trunk(x))
            prob = torch.sigmoid(logits)[0, self.belief_target_index]
        return float(prob.item())

    def _state_terms(self, obs: np.ndarray) -> dict[str, float]:
        sim = self._sim()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        pending_norm = 0.0
        lost_norm = 0.0
        if sim is not None:
            pending_norm = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0) / self.pending_scale
            lost_norm = float(getattr(sim, "total_unattended_orders", 0.0) or 0.0) / self.lost_scale
            try:
                theatre_coverage = float(sim.rations_theatre.level) / 1e5
            except Exception:
                theatre_coverage = 0.0
        else:
            theatre_coverage = 0.0

        op10_slack = float(obs[self._op10_pressure_idx]) if obs.shape[0] > self._op10_pressure_idx else 0.0
        op12_slack = float(obs[self._op12_pressure_idx]) if obs.shape[0] > self._op12_pressure_idx else 0.0
        oldest_backorder_age = (
            float(obs[self._oldest_backorder_age_idx])
            if obs.shape[0] > self._oldest_backorder_age_idx
            else 0.0
        )
        rolling_backorder = (
            float(obs[self._rolling_backorder_idx])
            if obs.shape[0] > self._rolling_backorder_idx
            else 0.0
        )
        backorder_rate = (
            float(obs[self._backorder_rate_idx])
            if obs.shape[0] > self._backorder_rate_idx
            else 0.0
        )
        readiness = 0.5 * theatre_coverage + 0.25 * op10_slack + 0.25 * op12_slack

        belief_prob = self._belief_prob(obs)
        p_adv = max(0.0, (belief_prob - self.belief_base_rate) / (1.0 - self.belief_base_rate))

        return {
            "pending_norm": float(np.clip(pending_norm, 0.0, 10.0)),
            "lost_norm": float(np.clip(lost_norm, 0.0, 10.0)),
            "theatre_coverage": float(np.clip(theatre_coverage, 0.0, 1.0)),
            "op10_slack": float(np.clip(op10_slack, 0.0, 1.0)),
            "op12_slack": float(np.clip(op12_slack, 0.0, 1.0)),
            "readiness": float(np.clip(readiness, 0.0, 1.0)),
            "exposure_gap": float(np.clip(1.0 - readiness, 0.0, 1.0)),
            "oldest_backorder_age": float(np.clip(oldest_backorder_age, 0.0, 1.0)),
            "rolling_backorder": float(np.clip(rolling_backorder, 0.0, 1.0)),
            "backorder_rate": float(np.clip(backorder_rate, 0.0, 1.0)),
            "belief_prob": belief_prob,
            "p_adv": float(np.clip(p_adv, 0.0, 1.0)),
            "resource_posture": self._last_resource_posture,
        }

    def _phi(self, obs: np.ndarray) -> tuple[float, dict[str, float]]:
        terms = self._state_terms(obs)
        phi = (
            -self.pbrs_alpha * terms["pending_norm"]
            - self.pbrs_beta * terms["lost_norm"]
            + self.pbrs_kappa * terms["p_adv"] * terms["readiness"]
            - self.pbrs_rho * (1.0 - terms["p_adv"]) * terms["resource_posture"]
            - self.pbrs_exposure * terms["p_adv"] * terms["exposure_gap"]
            - self.pbrs_backlog_age * terms["p_adv"] * terms["oldest_backorder_age"]
            - self.pbrs_tail * terms["p_adv"] * (
                0.5 * terms["rolling_backorder"] + 0.5 * terms["backorder_rate"]
            )
        )
        return phi, terms

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_resource_posture = 0.0
        self._prev_phi, _ = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base_reward = float(reward)

        shift = float(info.get("shifts_active", 1))
        shift_norm = (shift - 1.0) / 2.0
        op10_mult, op12_mult = self._extract_downstream_multipliers(info)
        op10_norm = float(np.clip((op10_mult - 1.25) / 0.75, 0.0, 1.0))
        op12_norm = float(np.clip((op12_mult - 1.25) / 0.75, 0.0, 1.0))
        self._last_resource_posture = float(np.clip((shift_norm + op10_norm + op12_norm) / 3.0, 0.0, 1.0))

        phi, terms = self._phi(obs)
        shaped = base_reward + self.pbrs_gamma * phi - self._prev_phi
        self._prev_phi = phi
        info = dict(info)
        info["future_credit_reward_mode"] = self.mode_label
        info["future_credit_base_reward"] = base_reward
        info["future_credit_shaped_reward"] = float(shaped)
        info["future_credit_phi"] = float(phi)
        info["future_credit_belief_prob"] = terms["belief_prob"]
        info["future_credit_p_adv"] = terms["p_adv"]
        info["future_credit_readiness"] = terms["readiness"]
        info["future_credit_resource_posture"] = terms["resource_posture"]
        info["future_credit_terms"] = terms
        return obs, float(shaped), terminated, truncated, info
