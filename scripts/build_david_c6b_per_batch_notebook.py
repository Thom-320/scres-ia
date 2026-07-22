#!/usr/bin/env python3
"""Build David's prospective C6-B notebook with 24 causal batch epochs."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "scresia_david_C6B_physical_perbatch_FINAL.ipynb"
ENV_SOURCE = (ROOT / "supply_chain" / "program_o_per_batch_env.py").read_text()


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
        # Notebook 6 · C6-B de David — 24 decisiones físicas reales por lote

        Este notebook prueba una pregunta nueva y prospectiva: **¿la memoria de DMLPA aporta valor
        cuando el controlador observa y actúa en 24 epochs físicos reales?**

        En cada epoch el transductor exacto se detiene inmediatamente antes de la llegada de un lote
        de 5,000 unidades. El agente observa el estado generado por todas las acciones, demandas y
        releases anteriores; elige `0=P_H` o `1=P_C`; el lote entra al inventario y el DES avanza
        hasta la siguiente llegada. No son tres bits ficticios con estado semanal repetido.

        ## Default `SERIOUS`

        Este notebook está preparado para que **10bits solamente seleccione `Run all`**. Durante la
        corrida muestra qué modelo está entrenando, la seed, tiempo transcurrido y un heartbeat cada
        60 segundos. Al terminar entrega una interpretación en español, una tarjeta para pantallazo y
        un ZIP pequeño listo para enviar a Thomas.

        Sin editar nada ejecuta cinco brazos, tres seeds y el mismo presupuesto
        de 200,192 pasos por seed:

        - RecurrentPPO con la arquitectura histórica MLP-LSTM: baseline en este mismo C6-B.
        - PPO + DMLPA con historia física completa de 24 epochs.
        - PPO + la misma DMLPA pero stack 1: ablación directa de memoria.
        - RecurrentPPO + DMLPA stack 24.
        - SAC categórico discreto + DMLPA stack 24.

        Los controladores estructurados reciben la misma observación y la misma acción binaria. Se
        elige una configuración usando tapes de selección separadas y se evalúa una sola vez en tapes
        de desarrollo disjuntas. El notebook no abre seeds científicas ni modifica Program Q/Q-R1.

        > **Autoridad física nueva:** elegir el destino del lote justo antes de su llegada a Op8 es
        > una hipótesis C6-B de desarrollo. Requiere validación de Garrido antes de cualquier claim de
        > dominio. Un PASS aquí autoriza preregistrar; no autoriza publicación ni despliegue.

        **Tiempo orientativo:** extrapolando el smoke PPO medido en un Mac CPU, los cuatro brazos de
        la familia PPO suman aproximadamente **12 horas** para las tres seeds `serious`. No se incluye
        SAC en esa cifra: su costo depende mucho de GPU y de las actualizaciones off-policy, y puede
        dominar la corrida. Para `serious`, usa GPU y evita asumir que terminará en una sesión corta.
        """
    ),
    code(
        r"""
        # 1 — CONFIGURACIÓN. El default ya está listo para Run all.
        import ast, hashlib, html, importlib.metadata, inspect, json, math, os, platform, shutil, subprocess, sys, threading, time
        from collections import deque
        from pathlib import Path
        from typing import Any

        RUN_PROFILE = os.environ.get("DAVID_C6B_PROFILE", "serious")
        MODEL_KINDS = [
            "recurrent_ppo_mlp",
            "ppo_dmlpa_stack24",
            "ppo_dmlpa_stack1",
            "recurrent_ppo_dmlpa_stack24",
            "sac_discrete_dmlpa_stack24",
        ]
        FRAME_STACK = 24
        FEATURES_DIM = 120
        DMLPA_HIDDEN = 100
        DMLPA_HEADS = 12
        DMLPA_LAYERS = 4
        AUTO_DOWNLOAD_AUDIT = True  # intenta descargar el ZIP pequeño al finalizar
        HEARTBEAT_SECONDS = 60       # muestra que la corrida sigue viva

        MODEL_LABELS = {
            "recurrent_ppo_mlp": "Baseline RecurrentPPO (MLP-LSTM)",
            "ppo_dmlpa_stack24": "PPO + DMLPA con memoria física stack 24",
            "ppo_dmlpa_stack1": "PPO + DMLPA sin memoria (stack 1)",
            "recurrent_ppo_dmlpa_stack24": "RecurrentPPO + DMLPA stack 24",
            "sac_discrete_dmlpa_stack24": "SAC categórico discreto + DMLPA stack 24",
        }

        PROFILES = {
            "debug": dict(timesteps=768, optimizer_seeds=[9201], selection_tapes=2, eval_tapes=2),
            "screen": dict(timesteps=50_000, optimizer_seeds=[9201, 9202, 9203], selection_tapes=6, eval_tapes=8),
            "serious": dict(timesteps=200_192, optimizer_seeds=[9201, 9202, 9203], selection_tapes=12, eval_tapes=24),
        }
        CFG = PROFILES[RUN_PROFILE]
        TOTAL_TIMESTEPS = CFG["timesteps"]
        OPTIMIZER_SEEDS = CFG["optimizer_seeds"]
        SELECTION_TAPES_PER_CELL = CFG["selection_tapes"]
        EVAL_TAPES_PER_CELL = CFG["eval_tapes"]

        TRAIN_TAPE_START, TRAIN_TAPE_END = 972_000_001, 972_099_999
        SELECTION_SEEDS = list(range(982_000_001, 982_000_001 + SELECTION_TAPES_PER_CELL))
        EVAL_SEEDS = list(range(983_000_001, 983_000_001 + EVAL_TAPES_PER_CELL))
        ALLOWED_TAPE_RANGES = ((972_000_001, 972_099_999), (982_000_001, 982_099_999), (983_000_001, 983_099_999))

        def assert_dev_tape(seed: int) -> None:
            if not any(lo <= int(seed) <= hi for lo, hi in ALLOWED_TAPE_RANGES):
                raise RuntimeError(f"TAPE PROHIBIDA: {seed}")

        for tape in (TRAIN_TAPE_START, TRAIN_TAPE_END, *SELECTION_SEEDS, *EVAL_SEEDS):
            assert_dev_tape(tape)
        total_jobs = len(MODEL_KINDS) * len(OPTIMIZER_SEEDS)
        print("=" * 76)
        print("NOTEBOOK 6 · PLAN DE EJECUCIÓN")
        print("Perfil:", RUN_PROFILE, "| trabajos de entrenamiento:", total_jobs)
        print("Pasos por modelo/seed:", f"{TOTAL_TIMESTEPS:,}", "| frame stack:", FRAME_STACK)
        print("Seeds de optimización:", OPTIMIZER_SEEDS)
        for number, kind in enumerate(MODEL_KINDS, 1):
            print(f"  {number}. {MODEL_LABELS[kind]} [{kind}]")
        print("Al finalizar se mostrará el veredicto y se generará un ZIP de auditoría.")
        print("=" * 76)
        """
    ),
    code(
        r"""
        # 2 — Repositorio y dependencias
        IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
        IN_KAGGLE = Path("/kaggle/working").exists()
        GIT_URL = "https://github.com/Thom-320/scres-ia.git"
        GIT_BRANCH = "qr1-c1-natural-continuation"
        CORE_COMMIT = "49f7802baedb47e0b1d23e23fa317504be059b71"
        PACKAGES = [
            "simpy>=4.1", "numpy>=1.26", "gymnasium>=1.3", "stable-baselines3>=2.9",
            "sb3-contrib>=2.9", "torch>=2.1", "einops>=0.8", "pandas>=2.2",
        ]

        def run(command):
            print("$", " ".join(map(str, command)))
            result = subprocess.run(command, text=True, capture_output=True)
            print((result.stdout or "")[-1600:]); print((result.stderr or "")[-1600:])
            if result.returncode:
                raise RuntimeError(command)

        if IN_COLAB:
            REPO = Path("/content/scresia_c6b")
        elif IN_KAGGLE:
            REPO = Path("/kaggle/working/scresia_c6b")
        else:
            REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
        if IN_COLAB or IN_KAGGLE:
            run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
            if REPO.exists():
                shutil.rmtree(REPO)
            run(["git", "clone", "--depth", "20", "--branch", GIT_BRANCH, GIT_URL, str(REPO)])
            run(["git", "-C", str(REPO), "checkout", "--detach", CORE_COMMIT])
        sys.path.insert(0, str(REPO))
        print("repo:", REPO)
        """
    ),
    markdown(
        r"""
        ## 3 — Contrato físico C6-B embebido · ⛔ NO CAMBIAR

        Esta celda contiene el entorno auditado. Para cada acción, el tiempo físico avanza hasta una
        llegada de lote distinta. Al terminal, un replay independiente verifica OAT, backlog, inventario
        y métricas contra el transductor vectorizado.
        """
    ),
    code(ENV_SOURCE),
    code(
        r"""
        # 4 — Entorno + historia con máscara explícita de padding
        import gymnasium as gym
        import numpy as np
        import pandas as pd
        import torch
        from einops import rearrange
        from scripts.evaluate_program_o_ret_learner import scheduler

        SCHED = scheduler()

        class MaskedHistoryStack(gym.Wrapper):
            '''Apila observaciones y añade un bit de validez por token.'''
            def __init__(self, env: gym.Env, history_length: int):
                super().__init__(env)
                self.history_length = int(history_length)
                self.base_dim = int(env.observation_space.shape[0])
                self.history = deque(maxlen=self.history_length)
                token_low = np.concatenate([env.observation_space.low, [0.0]]).astype(np.float32)
                token_high = np.concatenate([env.observation_space.high, [1.0]]).astype(np.float32)
                self.observation_space = gym.spaces.Box(
                    np.tile(token_low, self.history_length),
                    np.tile(token_high, self.history_length),
                    dtype=np.float32,
                )

            def _token(self, observation, valid=1.0):
                return np.concatenate([np.asarray(observation, np.float32), [valid]]).astype(np.float32)

            def _stack(self):
                return np.concatenate(tuple(self.history)).astype(np.float32, copy=False)

            def reset(self, **kwargs):
                observation, info = self.env.reset(**kwargs)
                self.history.clear()
                for _ in range(self.history_length - 1):
                    self.history.append(self._token(np.zeros_like(observation), valid=0.0))
                self.history.append(self._token(observation, valid=1.0))
                return self._stack(), info

            def step(self, action):
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.history.append(self._token(observation, valid=1.0))
                return self._stack(), reward, terminated, truncated, info

        def history_for(model_kind: str) -> int:
            if model_kind in {"recurrent_ppo_mlp", "ppo_dmlpa_stack1"}:
                return 1
            return FRAME_STACK

        def make_env(model_kind: str, start=TRAIN_TAPE_START, end=TRAIN_TAPE_END):
            raw = ProgramOPerBatchEnv(
                scheduler=SCHED, tape_seed_start=start, tape_seed_end=end
            )
            return MaskedHistoryStack(raw, history_for(model_kind))

        probe = make_env("ppo_dmlpa_stack24")
        obs, info = probe.reset(options={"tape_seed": EVAL_SEEDS[0], "cell_index": 2})
        times = [info["raw_observation"]["decision_time"]]
        for _ in range(23):
            obs, _, done, _, info = probe.step(0)
            if not done:
                times.append(info["raw_observation"]["decision_time"])
        assert len(times) == 24 and all(b > a for a, b in zip(times, times[1:]))
        print("PRECHECK OK: 24 epochs físicos distintos; obs raw=21; token DMLPA=22; stack=", FRAME_STACK)
        """
    ),
    markdown(
        r"""
        ## 5 — ARQUITECTURA DE DAVID · ✅ PUEDES EDITAR AQUÍ

        Puedes revisar y modificar embedding, atención, dimensión y capas. La máscara impide que el
        Transformer atienda a frames de padding. No cambies el bit final de validez ni el contrato de
        observación del entorno.
        """
    ),
    code(
        r"""
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        class DavidDMLPAPositional(BaseFeaturesExtractor):
            def __init__(self, observation_space, factor=24, features_dim=120,
                         hidden_dim=100, nhead=12, num_layers=4):
                super().__init__(observation_space, features_dim)
                self.factor = int(factor)
                flat_dim = int(observation_space.shape[0])
                if flat_dim % self.factor:
                    raise ValueError("observation dimension must be divisible by factor")
                self.token_dim = flat_dim // self.factor
                if self.token_dim != OBSERVATION_DIM + 1:
                    raise ValueError("each token must contain raw observation + validity bit")
                if features_dim % nhead:
                    raise ValueError("features_dim must be divisible by nhead")
                self.embedding = torch.nn.Sequential(
                    torch.nn.Linear(OBSERVATION_DIM, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, features_dim),
                )
                self.pre_norm = torch.nn.LayerNorm(features_dim)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=features_dim, nhead=nhead, batch_first=True
                )
                self.transformer = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
                self.register_buffer("positional", self._sinusoidal(self.factor, features_dim))

            @staticmethod
            def _sinusoidal(length, width):
                pe = torch.zeros(length, width)
                position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
                scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
                pe[:, 0::2] = torch.sin(position * scale)
                pe[:, 1::2] = torch.cos(position * scale)
                return pe.unsqueeze(0)

            def forward(self, observations):
                tokens = rearrange(observations, "b (t d) -> b t d", t=self.factor)
                valid = tokens[:, :, -1] > 0.5
                embedded = self.pre_norm(self.embedding(tokens[:, :, :-1]) + self.positional)
                encoded = self.transformer(embedded, src_key_padding_mask=~valid)
                return encoded[:, -1, :]

        print("DavidDMLPAPositional cargada. Esta es la única clase de arquitectura que David debe editar.")
        """
    ),
    code(
        r"""
        # 6 — Agentes: mismas acciones, información, tapes y presupuesto
        from stable_baselines3 import PPO
        from sb3_contrib import RecurrentPPO
        from scripts.discrete_sac_dmlpa import DiscreteSACAgent, DiscreteSACConfig

        def dmlpa_policy_kwargs(model_kind):
            return {
                "features_extractor_class": DavidDMLPAPositional,
                "features_extractor_kwargs": {
                    "factor": history_for(model_kind), "features_dim": FEATURES_DIM,
                    "hidden_dim": DMLPA_HIDDEN, "nhead": DMLPA_HEADS,
                    "num_layers": DMLPA_LAYERS,
                },
                "net_arch": dict(pi=[128, 64], vf=[128, 64]),
            }

        def build_agent(model_kind, optimizer_seed):
            env = make_env(model_kind)
            common = dict(env=env, seed=int(optimizer_seed), verbose=1, learning_rate=3e-4, gamma=0.99)
            if model_kind in {"ppo_dmlpa_stack24", "ppo_dmlpa_stack1"}:
                return PPO("MlpPolicy", n_steps=768, batch_size=96, gae_lambda=0.95,
                           ent_coef=0.01, policy_kwargs=dmlpa_policy_kwargs(model_kind), **common)
            if model_kind == "recurrent_ppo_mlp":
                return RecurrentPPO(
                    "MlpLstmPolicy", n_steps=768, batch_size=96, gae_lambda=0.95,
                    ent_coef=0.01,
                    policy_kwargs=dict(lstm_hidden_size=64, net_arch=dict(pi=[64, 64], vf=[64, 64])),
                    **common,
                )
            if model_kind == "recurrent_ppo_dmlpa_stack24":
                kwargs = dmlpa_policy_kwargs(model_kind); kwargs["lstm_hidden_size"] = 64
                return RecurrentPPO("MlpLstmPolicy", n_steps=768, batch_size=96,
                                    gae_lambda=0.95, ent_coef=0.01, policy_kwargs=kwargs, **common)
            if model_kind == "sac_discrete_dmlpa_stack24":
                kwargs = dmlpa_policy_kwargs(model_kind)["features_extractor_kwargs"]
                learning_starts = 744 if RUN_PROFILE == "debug" else 2_000
                return DiscreteSACAgent(
                    env=env, seed=int(optimizer_seed), features_dim=FEATURES_DIM,
                    extractor_factory=lambda: DavidDMLPAPositional(env.observation_space, **kwargs),
                    config=DiscreteSACConfig(buffer_size=100_000, batch_size=256,
                                             learning_starts=learning_starts, learning_rate=3e-4),
                )
            raise ValueError(model_kind)

        architecture_reports = {}
        for kind in MODEL_KINDS:
            agent = build_agent(kind, OPTIMIZER_SEEDS[0])
            policy = getattr(agent, "policy", agent)
            architecture_reports[kind] = {
                "modelo_legible": MODEL_LABELS[kind],
                "policy": type(policy).__name__,
                "history": history_for(kind),
                "parameters": int(sum(p.numel() for p in policy.parameters())),
                "action_space": str(agent.get_env().action_space),
            }
        architecture_df = pd.DataFrame(architecture_reports).T
        print("\nMODELOS QUE SE VAN A CORRER")
        display(architecture_df)
        print(architecture_df.to_string())
        """
    ),
    code(
        r"""
        # 7 — Entrenamiento multiseed con progreso y tiempo visible
        RUN_ROOT = REPO / "outputs" / "david_c6b" / f"{RUN_PROFILE}_{int(time.time())}"
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_STARTED = time.time()

        class TrainingHeartbeat:
            def __init__(self, label, every_seconds=HEARTBEAT_SECONDS):
                self.label = label; self.every_seconds = every_seconds
                self.stop_event = threading.Event(); self.started = None; self.thread = None
            def __enter__(self):
                self.started = time.time()
                def beat():
                    while not self.stop_event.wait(self.every_seconds):
                        elapsed = time.time() - self.started
                        total = time.time() - RUN_STARTED
                        print(f"  ⏱ SIGUE CORRIENDO · {self.label} · modelo {elapsed/60:.1f} min · total {total/60:.1f} min", flush=True)
                self.thread = threading.Thread(target=beat, daemon=True); self.thread.start()
                return self
            def __exit__(self, exc_type, exc, tb):
                self.stop_event.set(); self.thread.join(timeout=2)

        agents = {}; training_rows = []
        total_jobs = len(MODEL_KINDS) * len(OPTIMIZER_SEEDS); job_number = 0
        for kind in MODEL_KINDS:
            for optimizer_seed in OPTIMIZER_SEEDS:
                job_number += 1
                label = MODEL_LABELS[kind]
                print("\n" + "=" * 76)
                print(f"TRABAJO {job_number}/{total_jobs}: {label}")
                print(f"ID={kind} | seed={optimizer_seed} | pasos={TOTAL_TIMESTEPS:,} | historia={history_for(kind)}")
                print("Estado: ENTRENANDO", flush=True)
                started = time.time(); agent = build_agent(kind, optimizer_seed)
                with TrainingHeartbeat(f"{kind} seed={optimizer_seed}"):
                    agent.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
                elapsed = time.time() - started
                agents[(kind, optimizer_seed)] = agent
                training_rows.append({"model": kind, "model_label": label, "seed": optimizer_seed,
                                      "timesteps": TOTAL_TIMESTEPS, "history": history_for(kind),
                                      "elapsed_seconds": elapsed, "steps_per_second": TOTAL_TIMESTEPS / elapsed})
                suffix = ".pt" if kind.startswith("sac_discrete") else ".zip"
                agent.save(RUN_ROOT / f"{kind}_seed{optimizer_seed}{suffix}")
                completed_average = (time.time() - RUN_STARTED) / job_number
                remaining_minutes = completed_average * (total_jobs - job_number) / 60.0
                print(f"Estado: TERMINADO · {elapsed/60:.2f} min · {TOTAL_TIMESTEPS/elapsed:.1f} pasos/s")
                print(f"Progreso global: {job_number}/{total_jobs} · ETA aproximado restante: {remaining_minutes:.1f} min")
        training_df = pd.DataFrame(training_rows)
        print("\nENTRENAMIENTO COMPLETO · tiempo total:", f"{(time.time()-RUN_STARTED)/60:.2f} min")
        display(training_df)
        """
    ),
    code(
        r"""
        # 8 — Selección separada del mejor controlador estructurado
        def rollout_structured(config, skeleton, cell_index, tape_seed):
            env = ProgramOPerBatchEnv(scheduler=SCHED, tape_seed_start=tape_seed, tape_seed_end=tape_seed)
            _, _ = env.reset(options={"skeleton": skeleton, "tape_seed": tape_seed, "cell_index": cell_index})
            done = False
            while not done:
                action = structured_action(env.raw_observation(), config)
                _, _, done, _, info = env.step(action)
            return info["metrics"]

        structured_configs = structured_configurations()
        selection_rows = []
        selection_skeletons = {}
        for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
            for tape_seed in SELECTION_SEEDS:
                skeleton, _ = extract_full_des_skeleton(
                    seed=tape_seed, scheduler=SCHED,
                    regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share,
                    downstream_freight_physics_mode="fixed_clock_physical_v1")
                selection_skeletons[(cell_index, tape_seed)] = skeleton
                for config in structured_configs:
                    metrics = rollout_structured(config, skeleton, cell_index, tape_seed)
                    selection_rows.append({"cell": cell.cell_id, "seed": tape_seed,
                                           "config": config.config_id, "ret_visible": metrics["ret_visible"]})
        selection_df = pd.DataFrame(selection_rows)
        means = selection_df.groupby("config").ret_visible.mean().sort_values(ascending=False)
        BEST_STRUCTURED_ID = str(means.index[0])
        BEST_STRUCTURED = next(c for c in structured_configs if c.config_id == BEST_STRUCTURED_ID)
        print("MEJOR ESTRUCTURADO SELECCIONADO SOLO EN TAPES SEPARADAS:", BEST_STRUCTURED_ID)
        display(means.to_frame("mean_selection_ret"))
        """
    ),
    code(
        r"""
        # 9 — Evaluación emparejada: misma física, acción, observación y tapes
        def rollout_agent(agent, model_kind, skeleton, cell_index, tape_seed):
            env = make_env(model_kind, tape_seed, tape_seed)
            observation, _ = env.reset(options={"skeleton": skeleton, "tape_seed": tape_seed,
                                                "cell_index": cell_index})
            state = None; episode_start = np.ones((1,), dtype=bool); done = False
            while not done:
                prediction = agent.predict(observation, state=state,
                                           episode_start=episode_start, deterministic=True)
                action, state = prediction if isinstance(prediction, tuple) else (prediction, None)
                observation, _, done, _, info = env.step(int(np.asarray(action).reshape(-1)[0]))
                episode_start[:] = done
            return info["metrics"]

        def rollout_random_binary(skeleton, cell_index, tape_seed, policy_seed):
            env = ProgramOPerBatchEnv(scheduler=SCHED, tape_seed_start=tape_seed, tape_seed_end=tape_seed)
            env.reset(options={"skeleton": skeleton, "tape_seed": tape_seed, "cell_index": cell_index})
            rng = np.random.default_rng(int(policy_seed) + 31 * int(tape_seed)); done = False
            while not done:
                _, _, done, _, info = env.step(int(rng.integers(0, 2)))
            return info["metrics"]

        evaluation_rows = []; evaluation_started = time.time(); evaluation_tape_number = 0
        total_evaluation_tapes = len(CONFIRMED_RET_CELLS) * len(EVAL_SEEDS)
        RESOURCE_KEYS = ("gross_policy_batch_slots", "gross_production_quantity",
                         "charged_daily_dispatch_slots", "charged_downstream_vehicle_hours")
        for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
            for tape_seed in EVAL_SEEDS:
                evaluation_tape_number += 1
                print(f"EVALUACIÓN {evaluation_tape_number}/{total_evaluation_tapes}: {cell.cell_id} tape={tape_seed}")
                skeleton, _ = extract_full_des_skeleton(
                    seed=tape_seed, scheduler=SCHED,
                    regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share,
                    downstream_freight_physics_mode="fixed_clock_physical_v1")
                structured = rollout_structured(BEST_STRUCTURED, skeleton, cell_index, tape_seed)
                for kind in MODEL_KINDS:
                    for optimizer_seed in OPTIMIZER_SEEDS:
                        metrics = rollout_agent(agents[(kind, optimizer_seed)], kind, skeleton,
                                                cell_index, tape_seed)
                        row = {"model": kind, "optimizer_seed": optimizer_seed,
                               "cell": cell.cell_id, "tape_seed": tape_seed,
                               "ret_visible": metrics["ret_visible"],
                               "delta_structured": metrics["ret_visible"] - structured["ret_visible"],
                               "worst_delta_structured": metrics["worst_product_fill"] - structured["worst_product_fill"]}
                        for key in RESOURCE_KEYS:
                            row[f"resource_delta::{key}"] = metrics[key] - structured[key]
                        evaluation_rows.append(row)
                for optimizer_seed in OPTIMIZER_SEEDS:
                    metrics = rollout_random_binary(skeleton, cell_index, tape_seed, optimizer_seed)
                    row = {"model": "random_binary", "optimizer_seed": optimizer_seed,
                           "cell": cell.cell_id, "tape_seed": tape_seed,
                           "ret_visible": metrics["ret_visible"],
                           "delta_structured": metrics["ret_visible"] - structured["ret_visible"],
                           "worst_delta_structured": metrics["worst_product_fill"] - structured["worst_product_fill"]}
                    for key in RESOURCE_KEYS:
                        row[f"resource_delta::{key}"] = metrics[key] - structured[key]
                    evaluation_rows.append(row)
        evaluation_df = pd.DataFrame(evaluation_rows)
        print("Evaluación terminada en", f"{(time.time()-evaluation_started)/60:.2f} min")
        result_summary = evaluation_df.groupby(["model", "cell"]).ret_visible.agg(["mean", "std", "count"])
        print("\nRESULTADOS MEDIOS POR MODELO Y CELDA")
        display(result_summary)
        """
    ),
    code(
        r"""
        # 10 — Veredicto claro: RecurrentPPO, estructurado y memoria
        BOOTSTRAP_RESAMPLES = 5_000
        def two_way_lcb(delta, seed=20260722):
            delta = np.asarray(delta, dtype=float)
            rng = np.random.default_rng(seed); sims = np.empty(BOOTSTRAP_RESAMPLES)
            for i in range(BOOTSTRAP_RESAMPLES):
                si = rng.integers(0, delta.shape[0], delta.shape[0])
                ti = rng.integers(0, delta.shape[1], delta.shape[1])
                sims[i] = delta[np.ix_(si, ti)].mean()
            return float(np.quantile(sims, 0.05))

        def matrix(model, cell, column):
            subset = evaluation_df[(evaluation_df.model == model) & (evaluation_df.cell == cell)]
            return subset.pivot(index="optimizer_seed", columns="tape_seed", values=column).to_numpy()

        verdict_rows = []
        candidates = [kind for kind in MODEL_KINDS if kind != "recurrent_ppo_mlp"]
        for kind in candidates:
            for cell in [c.cell_id for c in CONFIRMED_RET_CELLS]:
                candidate_ret = matrix(kind, cell, "ret_visible")
                recurrent_ret = matrix("recurrent_ppo_mlp", cell, "ret_visible")
                random_ret = matrix("random_binary", cell, "ret_visible")
                delta_rppo = candidate_ret - recurrent_ret
                delta_random = candidate_ret - random_ret
                delta_structured = matrix(kind, cell, "delta_structured")
                worst = matrix(kind, cell, "worst_delta_structured")
                resource_cols = [c for c in evaluation_df.columns if c.startswith("resource_delta::")]
                subset = evaluation_df[(evaluation_df.model == kind) & (evaluation_df.cell == cell)]
                resource_max = max(
                    float(np.max(np.abs(subset[column].to_numpy(dtype=float))))
                    for column in resource_cols
                )
                row = {"model": kind, "cell": cell,
                       "delta_vs_RecurrentPPO": float(delta_rppo.mean()),
                       "delta_vs_RecurrentPPO_LCB05": two_way_lcb(delta_rppo),
                       "delta_vs_random": float(delta_random.mean()),
                       "delta_vs_random_LCB05": two_way_lcb(delta_random, 20260726),
                       "delta_vs_structured": float(delta_structured.mean()),
                       "delta_vs_structured_LCB05": two_way_lcb(delta_structured, 20260723),
                       "worst_product_delta_LCB05": two_way_lcb(worst, 20260724),
                       "resource_max_abs": resource_max}
                row["strong_cell_pass"] = bool(
                    row["delta_vs_RecurrentPPO_LCB05"] > 0.0
                    and row["delta_vs_structured_LCB05"] >= 0.01
                    and row["worst_product_delta_LCB05"] >= -0.02
                    and resource_max == 0.0)
                verdict_rows.append(row)

        verdict_df = pd.DataFrame(verdict_rows)
        display(verdict_df)
        final_verdict = {}
        for kind in candidates:
            rows = verdict_df[verdict_df.model == kind]
            final_verdict[kind] = {
                "beat_recurrent_ppo_all_cells": bool((rows.delta_vs_RecurrentPPO_LCB05 > 0).all()),
                "learned_signal_vs_random_all_cells": bool((rows.delta_vs_random_LCB05 > 0).all()),
                "beat_structured_plus_0p01_all_cells": bool((rows.delta_vs_structured_LCB05 >= 0.01).all()),
                "strong_goal_all_cells": bool(rows.strong_cell_pass.all()),
            }

        # La prueba directa de memoria mantiene PPO y DMLPA idénticos; solo cambia stack 24 vs 1.
        memory_rows = []
        for cell in [c.cell_id for c in CONFIRMED_RET_CELLS]:
            stack24 = matrix("ppo_dmlpa_stack24", cell, "ret_visible")
            stack1 = matrix("ppo_dmlpa_stack1", cell, "ret_visible")
            delta = stack24 - stack1
            memory_rows.append({"cell": cell, "memory_delta": float(delta.mean()),
                                "memory_delta_LCB05": two_way_lcb(delta, 20260725),
                                "memory_helped": bool(two_way_lcb(delta, 20260725) > 0.0)})
        memory_df = pd.DataFrame(memory_rows); display(memory_df)
        final_verdict["DMLPA_MEMORY"] = {
            "helped_all_cells": bool(memory_df.memory_helped.all()),
            "interpretation": "stack24 beat the otherwise identical stack1 PPO only if all paired LCB05 values exceed zero",
        }
        any_strong = any(
            value.get("strong_goal_all_cells", False)
            for value in final_verdict.values() if isinstance(value, dict)
        )
        if RUN_PROFILE != "serious":
            RUN_OUTCOME = "SMOKE_ONLY_NO_SCIENTIFIC_CONCLUSION"
        elif any_strong and final_verdict["DMLPA_MEMORY"]["helped_all_cells"]:
            RUN_OUTCOME = "C6B_DEVELOPMENT_PASS_TO_PREREGISTRATION"
        else:
            RUN_OUTCOME = "C6B_DEVELOPMENT_NO_GO_UNDER_TESTED_ENVELOPE"
        print("\nVEREDICTO FINAL")
        print(json.dumps(final_verdict, indent=2))
        print("RESULTADO GLOBAL:", RUN_OUTCOME)
        print("Incluso un PASS solo autoriza preregistrar; NO es claim científico ni validación Garrido.")
        """
    ),
    code(
        r"""
        # 11 — Interpretación directa + archivo pequeño para enviar y auditar
        from IPython.display import FileLink, HTML, Javascript, display
        from urllib.parse import quote

        trained_means = (evaluation_df[evaluation_df.model != "random_binary"]
                         .groupby("model").ret_visible.mean().sort_values(ascending=False))
        best_observed_value = float(trained_means.iloc[0])
        best_observed_models = [
            str(kind) for kind, value in trained_means.items()
            if np.isclose(float(value), best_observed_value, atol=1e-12, rtol=0.0)
        ]
        best_observed_label = "EMPATE: " + " | ".join(MODEL_LABELS.get(kind, kind) for kind in best_observed_models)
        if len(best_observed_models) == 1:
            best_observed_label = MODEL_LABELS.get(best_observed_models[0], best_observed_models[0])
        total_training_minutes = sum(row["elapsed_seconds"] for row in training_rows) / 60.0
        operator_rows = []
        for kind in candidates:
            flags = final_verdict[kind]
            operator_rows.append({
                "modelo": MODEL_LABELS[kind],
                "señal_aprendizaje_vs_azar": "SÍ" if flags["learned_signal_vs_random_all_cells"] else "NO",
                "ganó_a_RecurrentPPO": "SÍ" if flags["beat_recurrent_ppo_all_cells"] else "NO",
                "ganó_al_estructurado": "SÍ" if flags["beat_structured_plus_0p01_all_cells"] else "NO",
                "objetivo_fuerte_completo": "SÍ" if flags["strong_goal_all_cells"] else "NO",
            })
        operator_df = pd.DataFrame(operator_rows)

        interpretation_lines = [
            "NOTEBOOK 6 · RESUMEN PARA THOMAS",
            "=" * 72,
            f"Resultado global: {RUN_OUTCOME}",
            f"Perfil ejecutado: {RUN_PROFILE}",
            f"Modelos entrenados: {len(MODEL_KINDS)} tipos x {len(OPTIMIZER_SEEDS)} seeds = {len(training_rows)} trabajos",
            f"Pasos por trabajo: {TOTAL_TIMESTEPS:,}",
            f"Tiempo total de entrenamiento: {total_training_minutes:.2f} minutos",
            f"Mejor media observada: {best_observed_label}",
            f"Controlador estructurado seleccionado: {BEST_STRUCTURED_ID}",
            "",
            "LECTURA AUTOMÁTICA POR CANDIDATO",
        ]
        for kind in candidates:
            flags = final_verdict[kind]
            interpretation_lines.extend([
                f"- {MODEL_LABELS[kind]}",
                f"    Señal frente a política aleatoria: {'SÍ' if flags['learned_signal_vs_random_all_cells'] else 'NO'}",
                f"    Superó RecurrentPPO en todas las celdas: {'SÍ' if flags['beat_recurrent_ppo_all_cells'] else 'NO'}",
                f"    Superó estructurado +0.01 en todas: {'SÍ' if flags['beat_structured_plus_0p01_all_cells'] else 'NO'}",
                f"    Cumplió objetivo fuerte completo: {'SÍ' if flags['strong_goal_all_cells'] else 'NO'}",
            ])
        interpretation_lines.extend([
            "",
            "MEMORIA DMLPA",
            "- Stack 24 superó stack 1 en todas las celdas: " +
            ("SÍ" if final_verdict["DMLPA_MEMORY"]["helped_all_cells"] else "NO"),
            "",
            "INTERPRETACIÓN",
        ])
        if RUN_PROFILE != "serious":
            interpretation_lines.append(
                "Esta fue una corrida de prueba. Verifica que el código funciona, pero NO permite concluir que aprendió o ganó científicamente."
            )
        elif RUN_OUTCOME == "C6B_DEVELOPMENT_PASS_TO_PREREGISTRATION":
            interpretation_lines.append(
                "PASS de desarrollo: existe evidencia para preregistrar una confirmación. Aún requiere validación física de Garrido."
            )
        else:
            interpretation_lines.append(
                "NO-GO bajo el sobre probado: ningún candidato satisfizo simultáneamente todas las reglas congeladas."
            )
        interpretation_lines.extend([
            "",
            "LÍMITE DEL CLAIM",
            "C6-B usa autoridad física nueva por lote. Incluso un PASS no autoriza publicación ni despliegue sin Garrido y preregistro.",
        ])
        EXECUTIVE_TEXT = "\n".join(interpretation_lines) + "\n"
        print("\n" + EXECUTIVE_TEXT)

        report = {"status": "C6B_DEVELOPMENT_ONLY_NOT_PROMOTABLE", "profile": RUN_PROFILE,
                  "models": MODEL_KINDS, "timesteps_per_seed": TOTAL_TIMESTEPS,
                  "optimizer_seeds": OPTIMIZER_SEEDS, "frame_stack": FRAME_STACK,
                  "best_structured": BEST_STRUCTURED_ID, "architecture": architecture_reports,
                  "training": training_rows, "verdict": verdict_rows,
                  "memory_ablation": memory_rows, "final_verdict": final_verdict,
                  "run_outcome": RUN_OUTCOME,
                  "best_observed_models": best_observed_models,
                  "best_observed_ret": best_observed_value,
                  "total_training_minutes": total_training_minutes,
                  "claim_boundary": "New per-batch authority; requires Garrido face validation and a fresh preregistered run."}
        AUDIT_DIR = RUN_ROOT / "AUDITORIA_PARA_ENVIAR"
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        (AUDIT_DIR / "RESUMEN_PARA_THOMAS.txt").write_text(EXECUTIVE_TEXT)
        (AUDIT_DIR / "development_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        architecture_df.to_csv(AUDIT_DIR / "modelos_y_arquitecturas.csv")
        training_df.to_csv(AUDIT_DIR / "tiempos_de_entrenamiento.csv", index=False)
        selection_df.to_csv(AUDIT_DIR / "seleccion_controlador_estructurado.csv", index=False)
        evaluation_df.to_csv(AUDIT_DIR / "resultados_fila_a_fila.csv", index=False)
        verdict_df.to_csv(AUDIT_DIR / "veredicto_por_modelo_y_celda.csv", index=False)
        operator_df.to_csv(AUDIT_DIR / "interpretacion_directa_por_modelo.csv", index=False)
        memory_df.to_csv(AUDIT_DIR / "ablacion_memoria_stack24_vs_stack1.csv", index=False)
        source_notebook = REPO / "notebooks" / "scresia_david_C6B_physical_perbatch_FINAL.ipynb"
        if source_notebook.exists():
            shutil.copy2(source_notebook, AUDIT_DIR / source_notebook.name)

        environment = {
            "python": sys.version, "platform": platform.platform(),
            "core_commit": CORE_COMMIT,
            "public_notebook": "https://github.com/Thom-320/scres-ia/blob/qr1-c1-natural-continuation/notebooks/scresia_david_C6B_physical_perbatch_FINAL.ipynb",
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__, "pandas": pd.__version__,
            "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            "sb3_contrib": importlib.metadata.version("sb3-contrib"),
        }
        (AUDIT_DIR / "entorno_de_ejecucion.json").write_text(json.dumps(environment, indent=2) + "\n")

        outcome_color = "#166534" if "PASS" in RUN_OUTCOME else "#991b1b" if "NO_GO" in RUN_OUTCOME else "#92400e"
        screenshot_html = (
            '<div style="font-family:Arial,sans-serif;border:3px solid #111827;border-radius:16px;padding:22px;max-width:1050px;background:#fff">'
            '<h1 style="margin:0 0 8px">Notebook 6 · Resultado C6-B</h1>'
            f'<div style="font-size:22px;font-weight:700;color:white;background:{outcome_color};padding:12px;border-radius:8px">{html.escape(RUN_OUTCOME)}</div>'
            f'<p><b>Perfil:</b> {html.escape(RUN_PROFILE)} · <b>Trabajos:</b> {len(training_rows)} · <b>Tiempo de entrenamiento:</b> {total_training_minutes:.2f} min</p>'
            f'<p><b>Mejor media observada:</b> {html.escape(best_observed_label)}</p>'
            '<h2>¿Quién aprendió y quién ganó?</h2>' + operator_df.to_html(index=False) +
            '<h2>¿Sirvió la memoria?</h2>' + memory_df.to_html(index=False, float_format=lambda x: f"{x:.4f}") +
            '<p style="font-weight:700">Un smoke no permite conclusión científica. Un PASS serious solo autoriza preregistrar y requiere validación Garrido.</p></div>'
        )
        (AUDIT_DIR / "REPORTE_VISUAL_PARA_PANTALLAZO.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Resultado C6-B</title>" + screenshot_html
        )
        display(HTML(screenshot_html))

        artifacts = sorted(path for path in AUDIT_DIR.iterdir() if path.is_file())
        checksums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in artifacts]
        (AUDIT_DIR / "files.sha256").write_text("\n".join(checksums) + "\n")
        AUDIT_ZIP = Path(shutil.make_archive(str(RUN_ROOT / "C6B_AUDITORIA_PARA_ENVIAR"), "zip", root_dir=AUDIT_DIR))

        try:
            relative_zip = AUDIT_ZIP.relative_to(Path.cwd())
        except ValueError:
            relative_zip = AUDIT_ZIP
        download_href = "/files/" + quote(str(relative_zip))
        display(HTML(
            f'<div style="padding:18px;background:#e0f2fe;border:2px solid #0284c7;border-radius:12px;font-size:18px">'
            f'<b>ARCHIVO LISTO PARA ENVIAR</b><br><a id="c6b-download" href="{download_href}" download>'
            f'DESCARGAR {html.escape(AUDIT_ZIP.name)}</a><br><small>{html.escape(str(AUDIT_ZIP))}</small></div>'
        ))
        print("ARCHIVO FINAL PARA ENVIAR:", AUDIT_ZIP)
        print("SHA256:", hashlib.sha256(AUDIT_ZIP.read_bytes()).hexdigest())

        if AUTO_DOWNLOAD_AUDIT and IN_COLAB:
            from google.colab import files
            files.download(str(AUDIT_ZIP))
        elif AUTO_DOWNLOAD_AUDIT and IN_KAGGLE:
            display(Javascript("setTimeout(function(){document.getElementById('c6b-download').click();}, 1500);"))
            print("Kaggle: se intentó la descarga automática. Si el navegador la bloquea, usa el botón azul de arriba.")
        else:
            print("VPS/local: descarga el ZIP desde la ruta mostrada arriba.")
        """
    ),
    markdown(
        r"""
        ## Instrucciones finales para 10bits / David

        1. Activa GPU e Internet en Kaggle o Colab.
        2. Selecciona **Run all** y no cierres la sesión.
        3. Si aparece `SIGUE CORRIENDO`, el proceso está vivo: no lo reinicies.
        4. Al terminar, toma un pantallazo de la tarjeta **Notebook 6 · Resultado C6-B**.
        5. Envía el archivo `C6B_AUDITORIA_PARA_ENVIAR.zip` a Thomas. Contiene resumen legible,
           tablas completas, tiempos, versiones, checksums y el notebook fuente; no contiene los
           pesos grandes de los modelos.
        6. Si Kaggle bloquea la descarga automática, pulsa el botón azul final. El ZIP también queda
           guardado en `/kaggle/working/scresia_c6b/outputs/david_c6b/...` y aparecerá en Output tras
           guardar una versión.

        ### ✅ PUEDES CAMBIAR

        - La clase `DavidDMLPAPositional`.
        - `FEATURES_DIM`, hidden size, heads y capas, conservando dimensiones compatibles.
        - `MODEL_KINDS` para depuración y `RUN_PROFILE="debug"` para un smoke.
        - `AUTO_DOWNLOAD_AUDIT=False` si el navegador no permite descargas automáticas.
        - Después del smoke, vuelve a `serious` y usa **Run all**.

        ### ⛔ NO CAMBIES

        - La física C6-B, el momento de decisión, reward, observación, máscaras o tapes.
        - Los namespaces `972*`, `982*`, `983*` ni ninguna seed científica/reservada.
        - El controlador estructurado después de ver evaluación: se selecciona únicamente en tapes separadas.
        - Las acciones, información, presupuesto o métricas de un brazo sin cambiarlos para todos.
        - La lógica del veredicto ni el margen `+0.01` después de observar resultados.

        ### Qué debe ganar

        1. RecurrentPPO entrenado desde cero en **este mismo entorno C6-B**.
        2. El mejor controlador estructurado con la misma acción binaria e información.
        3. La DMLPA stack 24 debe superar a la misma DMLPA stack 1 para atribuir valor a memoria.
        4. Debe conservar peor-producto y recursos exactos.

        Si stack 24 no supera stack 1, DMLPA tiene capacidad de memoria pero la historia no produjo
        valor incremental bajo este contrato. Si gana, sigue siendo evidencia de desarrollo: la nueva
        autoridad de asignar cada lote debe validarse con Garrido antes de una corrida científica.
        """
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"david-c6b-{index:02d}"

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
