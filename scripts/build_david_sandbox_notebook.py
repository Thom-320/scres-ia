#!/usr/bin/env python3
"""Build David's auditable Program O-R architecture laboratory notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "david_sandbox_program_o_ret.ipynb"


def _source(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


cells = [
    markdown(
        r"""
        # Laboratorio Program O-R de David — PPO+DMLPA, DMLPA posicional y RecurrentPPO

        Este notebook permite **ver, editar y auditar** la arquitectura de David usando el entorno
        incremental real de Program O-R. Todo lo ejecutado aquí es desarrollo: usa exclusivamente
        seeds `949*`, nunca abre calibración/confirmación científica y no puede producir un claim.

        ## Qué queda visible

        - `FriendDMLPAFaithful`: arquitectura DMLPA original, sin positional encoding.
        - `FriendDMLPAPositional`: la misma arquitectura con PE sinusoidal y `LayerNorm`.
        - Historia apilada configurable. Con el default, cuatro estados de 21 variables forman
          cuatro tokens temporales; David puede cambiar la longitud o el agrupamiento.
        - `RecurrentPPO` como baseline de memoria.
        - Hook explícito para `SAC-Discrete + DMLPA`. **SB3 SAC no sirve aquí** porque la acción
          `k∈{0,1,2,3}` es discreta; David puede conectar su implementación sin cambiar el evaluador.

        ## Preset por defecto

        `preliminary`: 3 seeds del optimizador × 50,000 pasos, 12 tapes por cada una de las tres
        celdas. Es suficientemente serio para descartar arquitecturas malas, pero sigue siendo mucho
        menor que el contrato científico (10 seeds × 200,000 pasos y 48 tapes selladas por celda).

        El resultado principal del notebook es `H_learned` contra los 65,536 calendarios open-loop y
        `H_neural` contra la mejor configuración clásica elegida **por media del panel**, nunca por tape.
        """
    ),
    code(
        r"""
        # Celda 1 — configuración editable + guardia de seeds
        import json, math, sys, time
        from collections import deque
        from pathlib import Path
        from typing import Any

        import gymnasium as gym
        import numpy as np
        import pandas as pd
        import torch
        from einops import rearrange

        ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
        sys.path.insert(0, str(ROOT))

        # ── EDITA ESTO ────────────────────────────────────────────────────────────────
        PRESET = "preliminary"  # "quick" | "preliminary" | "extended"
        MODEL_KINDS_TO_RUN = ["ppo_dmlpa_positional"]
        # Para comparar las tres familias:
        # MODEL_KINDS_TO_RUN = ["ppo_dmlpa_faithful", "ppo_dmlpa_positional", "recurrent_ppo"]
        # SAC de David, después de implementar build_sac_discrete_dmlpa():
        # MODEL_KINDS_TO_RUN = ["sac_discrete_dmlpa"]

        HISTORY_LENGTH = 4
        DMLPA_FEATURES_DIM = 120
        DMLPA_HIDDEN = 100
        DMLPA_NHEAD = 12
        DMLPA_NUM_LAYERS = 4
        SAVE_DEV_ARTIFACTS = True

        PRESETS = {
            "quick": dict(total_timesteps=10_000, optimizer_seeds=[9201], eval_tapes_per_cell=4),
            "preliminary": dict(total_timesteps=50_000, optimizer_seeds=[9201, 9202, 9203], eval_tapes_per_cell=12),
            "extended": dict(total_timesteps=100_000, optimizer_seeds=[9201, 9202, 9203, 9204, 9205], eval_tapes_per_cell=24),
        }
        CFG = PRESETS[PRESET]
        TOTAL_TIMESTEPS = CFG["total_timesteps"]
        OPTIMIZER_SEEDS = CFG["optimizer_seeds"]
        EVAL_TAPES_PER_CELL = CFG["eval_tapes_per_cell"]

        DEV_TRAIN_SEEDS = (949100001, 949199999)
        DEV_EVAL_SEEDS = list(range(949200001, 949200001 + EVAL_TAPES_PER_CELL))
        FORBIDDEN = [(7420001, 7430999), (747000000, 748999999), (7480001, 7480999)]

        def assert_dev_seed(seed: int) -> None:
            for lo, hi in FORBIDDEN:
                if lo <= int(seed) <= hi:
                    raise RuntimeError(f"SEED PROHIBIDA {seed}: rango científico [{lo}, {hi}]")

        for seed in [DEV_TRAIN_SEEDS[0], DEV_TRAIN_SEEDS[1], *DEV_EVAL_SEEDS]:
            assert_dev_seed(seed)

        allowed = {
            "ppo_dmlpa_faithful", "ppo_dmlpa_positional", "recurrent_ppo",
            "sac_discrete_dmlpa", "ppo_mlp",
        }
        unknown = set(MODEL_KINDS_TO_RUN) - allowed
        if unknown:
            raise ValueError(f"MODEL_KINDS desconocidos: {sorted(unknown)}")

        print("Guardia OK: solo seeds de desarrollo.")
        print({"preset": PRESET, "models": MODEL_KINDS_TO_RUN, "timesteps_per_seed": TOTAL_TIMESTEPS,
               "optimizer_seeds": OPTIMIZER_SEEDS, "eval_tapes_per_cell": EVAL_TAPES_PER_CELL})
        """
    ),
    code(
        r"""
        # Celda 2 — entorno real + wrapper de historia (editable)
        from supply_chain.program_o_ret_env import ProgramORetOnlyEnv, CONFIRMED_RET_CELLS
        from scripts.evaluate_program_o_ret_learner import (
            GUARDRAIL_KEYS, RESOURCE_EQUALITY_KEYS, derive_placebo_calendars,
            encode_calendar, scheduler, trajectory_audit,
        )
        from supply_chain.program_o_full_des_transducer import (
            extract_full_des_skeleton, full_action_calendars, simulate_full_des_frontier,
        )

        SCHED = scheduler()

        class HistoryStackWrapper(gym.Wrapper):
            '''Concatena los últimos H estados; ceros representan historia no observada al reset.'''

            def __init__(self, env: gym.Env, history_length: int = 1):
                super().__init__(env)
                if history_length < 1:
                    raise ValueError("history_length debe ser >= 1")
                self.history_length = int(history_length)
                self.base_dim = int(np.prod(env.observation_space.shape))
                self.history = deque(maxlen=self.history_length)
                low = np.tile(env.observation_space.low, self.history_length).astype(np.float32)
                high = np.tile(env.observation_space.high, self.history_length).astype(np.float32)
                self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

            def _stack(self) -> np.ndarray:
                return np.concatenate(tuple(self.history)).astype(np.float32, copy=False)

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                self.history.clear()
                for _ in range(self.history_length - 1):
                    self.history.append(np.zeros_like(obs))
                self.history.append(np.asarray(obs, dtype=np.float32))
                return self._stack(), info

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.history.append(np.asarray(obs, dtype=np.float32))
                return self._stack(), reward, terminated, truncated, info

        def history_for(model_kind: str) -> int:
            # RecurrentPPO ya mantiene memoria interna. Los DMLPA reciben historia explícita.
            return 1 if model_kind in {"recurrent_ppo", "ppo_mlp"} else HISTORY_LENGTH

        def make_env(model_kind: str = "ppo_dmlpa_positional"):
            base = ProgramORetOnlyEnv(
                scheduler=SCHED,
                tape_seed_start=DEV_TRAIN_SEEDS[0],
                tape_seed_end=DEV_TRAIN_SEEDS[1],
            )
            return HistoryStackWrapper(base, history_length=history_for(model_kind))

        probe = make_env(MODEL_KINDS_TO_RUN[0])
        obs, info = probe.reset()
        print("observación base=21 | historia=", history_for(MODEL_KINDS_TO_RUN[0]),
              "| shape entregada=", obs.shape, "| acciones=", probe.action_space)
        print("celdas=", [cell.cell_id for cell in CONFIRMED_RET_CELLS])
        """
    ),
    code(
        r"""
        # Celda 3 — ARQUITECTURA DE DAVID: visible, editable y auditable
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        class FriendDMLPAFaithful(BaseFeaturesExtractor):
            '''DMLPA original de David: embedding + Transformer, sin posición explícita.'''

            def __init__(self, observation_space, factor: int = 1, features_dim: int = 120,
                         hidden_dim: int = 100, nhead: int = 12, num_layers: int = 4):
                super().__init__(observation_space, features_dim)
                flat_dim = int(np.prod(observation_space.shape))
                if flat_dim % factor != 0:
                    raise ValueError(f"Observation dimension {flat_dim} no es divisible por factor={factor}")
                if features_dim % nhead != 0:
                    raise ValueError("features_dim debe ser divisible por nhead")
                self.obs_dimension = flat_dim // factor
                self.factor = int(factor)
                self.latent_rw = torch.nn.Sequential(
                    torch.nn.Linear(self.obs_dimension, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, features_dim),
                )
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=features_dim, nhead=nhead, batch_first=True
                )
                self.accumulated = torch.nn.TransformerEncoder(layer, num_layers=num_layers)

            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                observations = rearrange(observations, "b (d k) -> b d k", d=self.factor)
                observations = self.latent_rw(observations)
                observations = self.accumulated(observations)
                return observations[:, -1, :]


        class FriendDMLPAPositional(BaseFeaturesExtractor):
            '''DMLPA de David con posición sinusoidal y LayerNorm.'''

            def __init__(self, observation_space, factor: int = 1, features_dim: int = 120,
                         hidden_dim: int = 100, nhead: int = 12, num_layers: int = 4):
                super().__init__(observation_space, features_dim)
                flat_dim = int(np.prod(observation_space.shape))
                if flat_dim % factor != 0:
                    raise ValueError(f"Observation dimension {flat_dim} no es divisible por factor={factor}")
                if features_dim % nhead != 0:
                    raise ValueError("features_dim debe ser divisible por nhead")
                self.obs_dimension = flat_dim // factor
                self.factor = int(factor)
                self.latent_rw = torch.nn.Sequential(
                    torch.nn.Linear(self.obs_dimension, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, features_dim),
                )
                self.pre_norm = torch.nn.LayerNorm(features_dim)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=features_dim, nhead=nhead, batch_first=True
                )
                self.accumulated = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
                self.register_buffer("pos_encoding", self.build_sinusoidal_pe(factor, features_dim))

            @staticmethod
            def build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
                pe = torch.zeros(seq_len, d_model)
                position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                return pe.unsqueeze(0)

            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                observations = rearrange(observations, "b (d k) -> b d k", d=self.factor)
                observations = self.latent_rw(observations)
                observations = observations + self.pos_encoding
                observations = self.pre_norm(observations)
                observations = self.accumulated(observations)
                return observations[:, -1, :]


        # Alias de notebooks anteriores.
        DMLPA = FriendDMLPAPositional

        def build_policy_kwargs(model_kind: str) -> dict[str, Any] | None:
            if model_kind in {"ppo_mlp", "recurrent_ppo"}:
                return None
            if model_kind in {"ppo_dmlpa_faithful", "sac_discrete_dmlpa"}:
                extractor = FriendDMLPAFaithful
            elif model_kind == "ppo_dmlpa_positional":
                extractor = FriendDMLPAPositional
            else:
                raise ValueError(f"MODEL_KIND desconocido: {model_kind}")
            factor = history_for(model_kind)
            return {
                "features_extractor_class": extractor,
                "features_extractor_kwargs": {
                    "factor": factor,
                    "features_dim": DMLPA_FEATURES_DIM,
                    "hidden_dim": DMLPA_HIDDEN,
                    "nhead": DMLPA_NHEAD,
                    "num_layers": DMLPA_NUM_LAYERS,
                },
                "net_arch": dict(pi=[128, 64], vf=[128, 64]),
            }

        print("Arquitecturas DMLPA cargadas.")
        print("Nota: factor=1 crea un Transformer de un token; con historia=4 usamos 4 tokens × 21 variables.")
        """
    ),
    code(
        r"""
        # Celda 4 — constructores de agentes + AUDIT de la red antes de entrenar
        from stable_baselines3 import PPO
        from sb3_contrib import RecurrentPPO

        def build_sac_discrete_dmlpa(env, seed: int):
            '''HOOK DE DAVID: conectar aquí SAC-Discrete. SB3.SAC NO admite Discrete(4).'''
            raise NotImplementedError(
                "Pega aquí tu SAC-Discrete y conserva la interfaz learn(total_timesteps=...) "
                "+ predict(obs, deterministic=True). No uses stable_baselines3.SAC vanilla."
            )

        def build_agent(model_kind: str, seed: int):
            env = make_env(model_kind)
            common = dict(env=env, verbose=0, seed=int(seed), learning_rate=3e-4, gamma=0.99)
            if model_kind in {"ppo_dmlpa_faithful", "ppo_dmlpa_positional", "ppo_mlp"}:
                kwargs = build_policy_kwargs(model_kind)
                return PPO(
                    "MlpPolicy", n_steps=512, batch_size=64, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01, policy_kwargs=kwargs, **common,
                )
            if model_kind == "recurrent_ppo":
                return RecurrentPPO(
                    "MlpLstmPolicy", n_steps=512, batch_size=64, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(lstm_hidden_size=64, net_arch=dict(pi=[64, 64], vf=[64, 64])),
                    **common,
                )
            if model_kind == "sac_discrete_dmlpa":
                return build_sac_discrete_dmlpa(env, seed)
            raise ValueError(model_kind)

        def architecture_audit(agent, model_kind: str) -> dict[str, Any]:
            policy = getattr(agent, "policy", agent)
            total = sum(p.numel() for p in policy.parameters())
            trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            extractor = getattr(policy, "features_extractor", None)
            report = {
                "model_kind": model_kind,
                "policy_class": type(policy).__name__,
                "total_parameters": int(total),
                "trainable_parameters": int(trainable),
                "history_length": history_for(model_kind),
                "observation_shape": tuple(agent.get_env().observation_space.shape)
                    if hasattr(agent, "get_env") else None,
            }
            print("\n" + "═" * 78)
            print(json.dumps(report, indent=2))
            if extractor is not None:
                print("\nFEATURE EXTRACTOR COMPLETO:\n", extractor)
                dummy = torch.zeros(2, *agent.get_env().observation_space.shape, device=agent.device)
                with torch.no_grad():
                    output = extractor(dummy)
                print("entrada dummy:", tuple(dummy.shape), "→ salida extractor:", tuple(output.shape))
                if hasattr(extractor, "pos_encoding"):
                    print("positional encoding:", tuple(extractor.pos_encoding.shape))
            print("═" * 78)
            return report

        # Construye un ejemplar antes de gastar compute: David ve exactamente qué se entrenará.
        AUDIT_AGENT = build_agent(MODEL_KINDS_TO_RUN[0], OPTIMIZER_SEEDS[0])
        ARCHITECTURE_REPORT = architecture_audit(AUDIT_AGENT, MODEL_KINDS_TO_RUN[0])
        del AUDIT_AGENT
        """
    ),
    code(
        r"""
        # Celda 5 — entrenamiento multiseed
        RUN_ROOT = ROOT / "outputs" / "david_sandbox" / f"{PRESET}_{int(time.time())}"
        if SAVE_DEV_ARTIFACTS:
            RUN_ROOT.mkdir(parents=True, exist_ok=True)

        agents: dict[tuple[str, int], Any] = {}
        training_rows = []
        for model_kind in MODEL_KINDS_TO_RUN:
            for seed in OPTIMIZER_SEEDS:
                print(f"\nEntrenando {model_kind} seed={seed} pasos={TOTAL_TIMESTEPS:,}")
                started = time.time()
                agent = build_agent(model_kind, seed)
                agent.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
                elapsed = time.time() - started
                agents[(model_kind, seed)] = agent
                training_rows.append({"model": model_kind, "seed": seed, "timesteps": TOTAL_TIMESTEPS,
                                      "elapsed_seconds": elapsed})
                if SAVE_DEV_ARTIFACTS and hasattr(agent, "save"):
                    agent.save(RUN_ROOT / f"{model_kind}_seed{seed}.zip")
                print(f"listo en {elapsed/60:.1f} min")

        display(pd.DataFrame(training_rows))
        """
    ),
    code(
        r"""
        # Celda 6 — panel de evaluación: misma familia física, tapes NUEVAS de desarrollo
        from supply_chain.program_o_state_rich import finite_state_rich_configurations, state_rich_calendar

        ALL_CALS = full_action_calendars()
        CLASSICAL_CONFIGS = finite_state_rich_configurations()
        METRIC_KEYS = tuple(dict.fromkeys((
            "ret_visible", *GUARDRAIL_KEYS, *RESOURCE_EQUALITY_KEYS,
            "ret_visible_cvar10", "service_loss_auc", "unresolved_orders", "max_backlog_age",
        )))

        def rollout_calendar(agent, model_kind: str, skeleton, cell_index: int, tape_seed: int) -> tuple[int, ...]:
            env = make_env(model_kind)
            obs, _ = env.reset(options={"tape_seed": tape_seed, "cell_index": cell_index, "skeleton": skeleton})
            state, episode_start, actions = None, True, []
            terminated = truncated = False
            while not (terminated or truncated):
                prediction = agent.predict(
                    obs, state=state, episode_start=np.array([episode_start]), deterministic=True
                )
                action, state = prediction if isinstance(prediction, tuple) else (prediction, None)
                action = int(np.asarray(action).reshape(-1)[0])
                obs, _, terminated, truncated, _ = env.step(action)
                actions.append(action)
                episode_start = bool(terminated or truncated)
            return tuple(actions)

        # Cache compartido: todos los modelos se comparan en exactamente las mismas trayectorias.
        eval_cache: dict[str, list[dict[str, Any]]] = {}
        for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
            rows = []
            print(f"Construyendo panel {cell.cell_id} ({len(DEV_EVAL_SEEDS)} tapes)...")
            for tape_seed in DEV_EVAL_SEEDS:
                assert_dev_seed(tape_seed)
                skeleton, _ = extract_full_des_skeleton(
                    seed=tape_seed, scheduler=SCHED,
                    regime_persistence=cell.regime_persistence,
                    dominant_share=cell.dominant_share,
                    downstream_freight_physics_mode="fixed_clock_physical_v1",
                )
                raw_panel = simulate_full_des_frontier(
                    skeleton=skeleton, scheduler=SCHED, calendars=ALL_CALS
                )
                panel = {key: np.asarray(raw_panel[key]) for key in METRIC_KEYS}
                classical_indices = []
                for config in CLASSICAL_CONFIGS:
                    calendar, _ = state_rich_calendar(
                        skeleton=skeleton.as_dict(), scheduler=SCHED, config=config,
                        # Modelo permitido fijo: no recibe rho/share verdaderos de la celda.
                        regime_persistence=0.75, dominant_share=0.90,
                    )
                    classical_indices.append(encode_calendar(tuple(calendar)))
                rows.append({"seed": tape_seed, "skeleton": skeleton, "panel": panel,
                             "classical_indices": np.asarray(classical_indices, dtype=np.int64)})
            eval_cache[cell.cell_id] = rows

        print("Panel listo:", {k: len(v) for k, v in eval_cache.items()})
        """
    ),
    code(
        r"""
        # Celda 7 — seleccionar comparadores por MEDIA y evaluar todos los agentes
        benchmark_cache: dict[str, dict[str, Any]] = {}
        learner_cache: dict[tuple[str, str], dict[str, Any]] = {}

        for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
            rows = eval_cache[cell.cell_id]
            open_ret = np.stack([row["panel"]["ret_visible"] for row in rows])
            best_open_index = int(open_ret.mean(axis=0).argmax())

            classical_ret = np.stack([
                row["panel"]["ret_visible"][row["classical_indices"]] for row in rows
            ])
            best_classical_index = int(classical_ret.mean(axis=0).argmax())
            best_classical_name = CLASSICAL_CONFIGS[best_classical_index].config_id
            benchmark_cache[cell.cell_id] = {
                "best_open_index": best_open_index,
                "best_classical_index": best_classical_index,
                "best_classical_name": best_classical_name,
            }

            for model_kind in MODEL_KINDS_TO_RUN:
                seed_metrics, seed_calendars = [], []
                for optimizer_seed in OPTIMIZER_SEEDS:
                    agent = agents[(model_kind, optimizer_seed)]
                    metrics, calendars = [], []
                    for row in rows:
                        calendar = rollout_calendar(
                            agent, model_kind, row["skeleton"], cell_index, row["seed"]
                        )
                        index = encode_calendar(calendar)
                        calendars.append(calendar)
                        metrics.append({key: float(row["panel"][key][index]) for key in METRIC_KEYS})
                    seed_metrics.append(metrics)
                    seed_calendars.append(calendars)
                learner_cache[(model_kind, cell.cell_id)] = {
                    "metrics": seed_metrics, "calendars": seed_calendars,
                }
            print(cell.cell_id, "best open=", best_open_index,
                  "best classical=", best_classical_name)
        """
    ),
    code(
        r"""
        # Celda 8 — métricas importantes (bootstrap exploratorio, NO inferencia contractual)
        BOOTSTRAP_RESAMPLES = 2_000

        def two_way_lcb95(delta: np.ndarray, rng_seed: int = 920100) -> float:
            '''Bootstrap seeds×tapes simple; orientativo, no reemplaza max-t contractual.'''
            delta = np.asarray(delta, dtype=float)
            rng = np.random.default_rng(rng_seed)
            sims = np.empty(BOOTSTRAP_RESAMPLES)
            for b in range(BOOTSTRAP_RESAMPLES):
                si = rng.integers(0, delta.shape[0], delta.shape[0])
                ti = rng.integers(0, delta.shape[1], delta.shape[1])
                sims[b] = delta[np.ix_(si, ti)].mean()
            return float(np.quantile(sims, 0.05))

        summaries, detailed = [], {}
        for model_kind in MODEL_KINDS_TO_RUN:
            for cell in CONFIRMED_RET_CELLS:
                rows = eval_cache[cell.cell_id]
                bench = benchmark_cache[cell.cell_id]
                payload = learner_cache[(model_kind, cell.cell_id)]
                learner = np.asarray([
                    [m["ret_visible"] for m in seed_rows] for seed_rows in payload["metrics"]
                ])
                open_values = np.asarray([
                    row["panel"]["ret_visible"][bench["best_open_index"]] for row in rows
                ])
                classical_values = np.asarray([
                    row["panel"]["ret_visible"][
                        row["classical_indices"][bench["best_classical_index"]]
                    ] for row in rows
                ])
                d_open = learner - open_values[None, :]
                d_classical = learner - classical_values[None, :]
                mean_by_tape = learner.mean(axis=0)

                audits = [trajectory_audit(calendars) for calendars in payload["calendars"]]
                placebo_seed_passes = {name: 0 for name in ("modal", "phase_only", "frequency_matched")}
                for seed_idx, calendars in enumerate(payload["calendars"]):
                    placebos = derive_placebo_calendars(calendars, rng_seed=OPTIMIZER_SEEDS[seed_idx])
                    learner_mean = learner[seed_idx].mean()
                    for name, calendar in placebos.items():
                        idx = encode_calendar(calendar)
                        placebo_mean = np.mean([row["panel"]["ret_visible"][idx] for row in rows])
                        placebo_seed_passes[name] += int(learner_mean > placebo_mean)

                resource_spread = max(
                    float(row["panel"][key].max() - row["panel"][key].min())
                    for row in rows for key in RESOURCE_EQUALITY_KEYS
                )
                guardrail_deltas = {}
                for key in GUARDRAIL_KEYS:
                    learner_key = np.asarray([[m[key] for m in sr] for sr in payload["metrics"]]).mean()
                    classical_key = np.mean([
                        row["panel"][key][row["classical_indices"][bench["best_classical_index"]]]
                        for row in rows
                    ])
                    guardrail_deltas[key] = float(learner_key - classical_key)

                row_summary = {
                    "model": model_kind,
                    "cell": cell.cell_id,
                    "H_learned": float(d_open.mean()),
                    "H_learned_LCB05_dev": two_way_lcb95(d_open),
                    "H_neural": float(d_classical.mean()),
                    "H_neural_LCB05_dev": two_way_lcb95(d_classical, rng_seed=920101),
                    "fav_tapes_open": int(np.sum(mean_by_tape > open_values)),
                    "fav_tapes_classical": int(np.sum(mean_by_tape > classical_values)),
                    "positive_seeds_both": int(np.sum((d_open.mean(axis=1) > 0) & (d_classical.mean(axis=1) > 0))),
                    "feedback_seeds": int(sum(bool(a["passed"]) for a in audits)),
                    "resource_spread": resource_spread,
                    "best_classical": bench["best_classical_name"],
                }
                summaries.append(row_summary)
                detailed[f"{model_kind}::{cell.cell_id}"] = {
                    **row_summary,
                    "trajectory_audits": audits,
                    "placebo_seed_passes": placebo_seed_passes,
                    "guardrail_deltas_vs_classical": guardrail_deltas,
                }

        summary_df = pd.DataFrame(summaries)
        display(summary_df)
        print("\nLectura rápida: H_learned>0 = aprende frente a horarios fijos; H_neural>=0 = iguala/supera clásico.")
        print("Los LCB son exploratorios y NO simultáneos. resource_spread debe ser exactamente 0.")
        for key, value in detailed.items():
            print("\n", key)
            print(" placebos (seeds que ganan):", value["placebo_seed_passes"], "/", len(OPTIMIZER_SEEDS))
            print(" guardrails vs clásico:", value["guardrail_deltas_vs_classical"])
        """
    ),
    code(
        r"""
        # Celda 9 — guardar reporte auditable de desarrollo
        report = {
            "status": "SANDBOX_DEVELOPMENT_ONLY_NOT_PROMOTABLE",
            "preset": PRESET,
            "models": MODEL_KINDS_TO_RUN,
            "total_timesteps_per_seed": TOTAL_TIMESTEPS,
            "optimizer_seeds": OPTIMIZER_SEEDS,
            "history_length": HISTORY_LENGTH,
            "eval_tapes": DEV_EVAL_SEEDS,
            "architecture": ARCHITECTURE_REPORT,
            "training": training_rows,
            "summary": summaries,
            "details": detailed,
            "claim_boundary": (
                "A positive sandbox result is only a hypothesis. Freeze architecture and hyperparameters, "
                "preregister fresh seeds, then evaluate once with the scientific evaluator."
            ),
        }
        if SAVE_DEV_ARTIFACTS:
            path = RUN_ROOT / "development_report.json"
            path.write_text(json.dumps(report, indent=2, default=str) + "\n")
            print("Reporte guardado en:", path)
        else:
            print(json.dumps(report["summary"], indent=2))
        """
    ),
    markdown(
        r"""
        ## Cómo debe usarlo David

        1. Ejecutar primero `quick` para verificar que su código corre.
        2. Dejar visible el audit de arquitectura: extractor completo, parámetros, entrada y salida.
        3. Usar `preliminary` sin cambiar tapes ni métricas para comparar propuestas.
        4. No seleccionar un modelo porque ganó una sola seed o una sola celda.
        5. Un candidato merece preregistro nuevo únicamente si muestra, de forma estable:
           - `H_learned > 0`;
           - `H_neural >= 0`;
           - feedback real y derrota de placebos;
           - recursos exactamente iguales;
           - guardrails sin deterioro material.

        `RecurrentPPO` sigue disponible como baseline. La corrida científica RecurrentPPO histórica ya
        terminó: superó open-loop pero no al mejor controlador clásico. Este notebook no la reabre; permite
        comprobar si la representación DMLPA de David aporta algo nuevo bajo un contrato prospectivo.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(OUTPUT)
