#!/usr/bin/env python3
"""Build the self-contained notebook David runs (and we run) end to end.

Design goals, in priority order:
  1. Run-all works. No hidden state, no manual steps between cells.
  2. David can read it without this repo's context: every cell says what it does, which
     knobs are his to turn, and which must stay frozen for the result to be comparable.
  3. The final cells answer his three questions in plain language -- did it beat the static
     policy, how close is it to the MPC, is the run even valid -- and write one downloadable
     JSON so we can audit how it ran.

Usage:  .venv/bin/python scripts/build_david_notebook_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks/SCRESIA_DAVID_RL_vs_MPC.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------- 0. portada
md(r"""
# SCRES-IA — ¿Puede el RL ganarle al MPC? Notebook de experimentos

**Para David.** Este notebook corre de principio a fin (`Run all`) y al final te dice, en
lenguaje claro:

1. ¿Tu política **le ganó a la mejor política estática**? (la barra que pide Garrido)
2. ¿**Qué tan cerca quedó del MPC**, el controlador estructurado que hoy es el mejor?
3. ¿El run **es válido**? (chequeos automáticos; si algo falla, te lo dice y no reporta números)

---

## Lo que ya sabemos, para que no repitas trabajo

| Hecho medido | Número |
|---|---|
| Mejor calendario estático `[0,0,3,3,3,3,3,3]` (elegido fuera de muestra) | ReT medio **0.6894** en nuestras campañas |
| MPC de creencia retenida — **el rival a vencer** | **+0.0280** sobre la estática |
| Aprendices neuronales probados hasta ahora (RecurrentPPO, PPO-MLP) | **por debajo** de la estática |
| Techo clarividente exacto (ninguna política puede superarlo) | +0.1309 sobre la estática |

O sea: el MPC captura ~21% del margen disponible y **nadie ha superado la barra estática con
una red todavía**. Ese es el hueco que estos experimentos atacan.

> **Sobre los números de arriba:** el *calendario* de la barra estática es fijo, pero su ReT
> medio depende de qué campañas evalúes. Por eso el notebook **recalcula la barra sobre tus
> mismas campañas de evaluación** (Celda 5) en vez de usar el 0.6894 de las nuestras. La
> comparación siempre es contra la barra medida en tu propio conjunto, que es la única justa.
> Las raíces que usa son de *desarrollo*: sirven para decidir, no para reportar como
> confirmatorio.

## Dónde creemos que está la oportunidad (y por qué)

El episodio de una campaña son solo **8 decisiones**, así que apilar más frames *dentro* de una
campaña no agrega información que exista. La señal está en **cruzar el límite de campaña**: el
MPC gana precisamente porque arrastra su creencia de una campaña a la siguiente. Por eso este
notebook apila frames **a través de campañas** (`FRAME_STACK`), que es tu propuesta llevada al
lugar donde hay señal.

Tu arquitectura DMLPA encaja naturalmente: el transformer atiende sobre los `FRAME_STACK`
frames apilados, así que un stack más grande le da más contexto histórico que resumir.
""")

# ---------------------------------------------------------------- 1. setup
md(r"""
## Celda 1 — Instalación y descarga del código

**Qué hace:** instala dependencias y trae el repositorio con el simulador y los artefactos
congelados (barra estática, contratos, campañas).

**Qué puedes cambiar:** nada aquí. Si estás en Colab/Kaggle déjalo tal cual.

**Si falla:** casi siempre es la versión de `torch`. Reinicia el runtime y vuelve a correr.
""")

code(r"""
# --- Celda 1: setup -------------------------------------------------------------------
import os, subprocess, sys, importlib

REPO_URL    = "https://github.com/Thom-320/scres-ia.git"
REPO_BRANCH = "codex/q-r1-comparator-reconciliation"   # rama con el entorno más reciente
REPO_DIR    = "/content/scres-ia" if os.path.isdir("/content") else "./scres-ia"

def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

if not os.path.isdir(REPO_DIR):
    sh(f"git clone --depth 1 --branch {REPO_BRANCH} {REPO_URL} {REPO_DIR}")
else:
    print(f"[ok] repositorio ya presente en {REPO_DIR}")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

for pkg, pip_name in [("simpy", "simpy"), ("stable_baselines3", "stable-baselines3"),
                      ("sb3_contrib", "sb3-contrib"), ("einops", "einops")]:
    try:
        importlib.import_module(pkg)
        print(f"[ok] {pkg}")
    except ImportError:
        sh(f"{sys.executable} -m pip -q install {pip_name}")

import torch
print(f"[ok] torch {torch.__version__} | GPU: {torch.cuda.is_available()}")
""")

# ---------------------------------------------------------------- 2. config
md(r"""
## Celda 2 — **LA ÚNICA CELDA QUE DEBES EDITAR**

Todo lo que puedes tocar está aquí. Lo de abajo está marcado como *congelado*: si lo cambias,
tus resultados dejan de ser comparables con los nuestros y con los de Garrido.

### Los tres modos

| `PRESET` | Para qué sirve | Cuánto tarda (GPU) |
|---|---|---|
| `"smoke"` | ¿El notebook corre sin errores? No mires los números. | ~3 min |
| `"signal"` | ¿Hay señal? ¿Vale la pena el run largo? | ~30-45 min |
| `"final"`  | Resultado definitivo, varias semillas, reportable. | ~4-6 h |

**Recomendación:** corre `smoke` una vez, luego `signal`. Solo si `signal` muestra que te
acercas a la estática, corre `final`.
""")

code(r"""
# --- Celda 2: CONFIGURACIÓN (edita SOLO esta celda) -----------------------------------

PRESET = "smoke"          # "smoke" | "signal" | "final"

# ---- ARQUITECTURA --------------------------------------------------------------------
# "dmlpa_positional" : TU arquitectura completa (transformer + posición sinusoidal +
#                      LayerNorm). Es el DEFAULT porque es la más fuerte que has construido.
# "dmlpa_faithful"   : tu DMLPA original, sin posición explícita. Úsala para aislar
#                      cuánto aporta el positional encoding.
# "mlp"              : red feed-forward simple. Es el control honesto: si DMLPA no le gana
#                      a esto, la complejidad no se está pagando.
# "lstm"             : RecurrentPPO. Memoria recurrente en vez de atención.
ARCHITECTURE = "dmlpa_positional"

# ---- FRAME STACK (tu propuesta) ------------------------------------------------------
# Cuántas observaciones pasadas ve la política en cada decisión.
# El transformer de DMLPA atiende sobre estos frames: más stack = más contexto histórico.
#
#   8  = una campaña completa (8 decisiones)
#   16 = dos campañas  <- recomendado para empezar; cruza el límite de campaña
#   24 = tres campañas
#   32 = cuatro campañas (más lento, más memoria)
#
# IMPORTANTE: el valor DEBE dividir exactamente la observación apilada, y el notebook lo
# verifica por ti. Cruzar el límite de campaña es donde creemos que está la señal.
FRAME_STACK = 16

# ---- ALGORITMO -----------------------------------------------------------------------
# "ppo"  : recomendado. Funciona con las 4 arquitecturas.
# "qrdqn": distribucional, off-policy. Alternativa si PPO se estanca.
#
# NOTA SOBRE SAC: SAC es para acciones CONTINUAS. Aquí el espacio de acción es discreto
# (4 acciones: cuántos lotes de 3 son P_C). SAC no aplica sin cambiar el contrato de
# decisión, y cambiarlo rompe la comparabilidad con el MPC y con la tesis. QRDQN es el
# equivalente distribucional para acciones discretas: úsalo si quieres esa familia.
ALGORITHM = "ppo"

# ---- HIPERPARÁMETROS DE TU ARQUITECTURA (puedes tocarlos) ----------------------------
DMLPA_FEATURES_DIM = 120   # dimensión latente. Debe ser divisible por DMLPA_NHEAD.
DMLPA_HIDDEN_DIM   = 100   # capa oculta del embedding previo al transformer
DMLPA_NHEAD        = 12    # cabezas de atención
DMLPA_NUM_LAYERS   = 4     # capas del transformer. Más = más capacidad y más lento.

LEARNING_RATE = 3e-4       # 1e-4 si el entrenamiento se ve inestable
ENT_COEF      = 0.01       # entropía: súbelo (0.02-0.05) si colapsa a acción constante

# ---- SEMILLAS ------------------------------------------------------------------------
# En "final" se usan varias para que el resultado no dependa de una semilla afortunada.
SEEDS_OVERRIDE = None      # None = usa las del preset. O pon [1, 2, 3] para fijarlas.

# =======================================================================================
# ==== A PARTIR DE AQUÍ: CONGELADO. No lo cambies o pierdes comparabilidad. =============
# =======================================================================================
FROZEN = {
    "campaigns_per_history": 12,     # 12 campañas por meta-episodio
    "decisions_per_campaign": 8,     # 8 decisiones semanales
    "rho": 0.90,                     # persistencia de régimen DENTRO de la campaña
    "dominant_share": 0.90,          # cuota del producto dominante
    "kappas": (0.75, 0.90),          # persistencia de conocimiento ENTRE campañas
    "objective": "early_ret_complete_cohort",   # la métrica ReT canónica
    "static_bar_calendar": [0, 0, 3, 3, 3, 3, 3, 3],  # barra elegida fuera de muestra
    "mpc_advantage_over_static": 0.0280313677,        # el rival, medido
}

PRESETS = {
    "smoke":  dict(timesteps=2_000,   eval_roots=2,  seeds=[1]),
    "signal": dict(timesteps=60_000,  eval_roots=6,  seeds=[1, 2]),
    "final":  dict(timesteps=240_000, eval_roots=12, seeds=[1, 2, 3, 4, 5]),
}
CFG = dict(PRESETS[PRESET])
if SEEDS_OVERRIDE is not None:
    CFG["seeds"] = list(SEEDS_OVERRIDE)

print(f"PRESET={PRESET}  arquitectura={ARCHITECTURE}  frame_stack={FRAME_STACK}")
print(f"algoritmo={ALGORITHM}  timesteps={CFG['timesteps']:,}  semillas={CFG['seeds']}")
print(f"campañas de evaluación: {CFG['eval_roots']} historias x 12 campañas x 2 kappas")
if DMLPA_FEATURES_DIM % DMLPA_NHEAD:
    raise ValueError(f"DMLPA_FEATURES_DIM ({DMLPA_FEATURES_DIM}) debe ser divisible por "
                     f"DMLPA_NHEAD ({DMLPA_NHEAD})")
print("[ok] configuración válida")
""")

# ---------------------------------------------------------------- 3. entorno
md(r"""
## Celda 3 — El entorno (simulador) con frame stack entre campañas

**Qué hace:** construye el entorno de decisión sobre el simulador de eventos discretos de la
cadena de suministro militar, y le apila `FRAME_STACK` observaciones pasadas **cruzando el
límite de campaña**.

**Lo importante, en una frase:** cada meta-episodio son 12 campañas seguidas; el estado físico
(inventario, backlog, pipeline) se reinicia entre campañas, pero **la historia observada no** —
y ahí es donde tu política puede aprender lo que el MPC obtiene con su creencia bayesiana.

**Qué puedes cambiar:** nada de esta celda. El frame stack se controla desde la Celda 2.
""")

code(r"""
# --- Celda 3: entorno con frame stack entre campañas ----------------------------------
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces

from scripts.evaluate_program_q_replication import scheduler
from supply_chain.oracle_curve_v2 import HistorySpec, MetaCampaignEnv
from supply_chain.program_o_full_des_transducer import simulate_full_des_frontier

SCHED = scheduler()

def objective_of(skeleton, calendar) -> float:
    '''ReT canónica de un calendario en una campaña. Es la métrica que se reporta.'''
    m = simulate_full_des_frontier(skeleton=skeleton, scheduler=SCHED,
                                   calendars=np.asarray([calendar], dtype=np.uint8),
                                   include_q_r1_metrics=True)
    return float(np.asarray(m[FROZEN["objective"]])[0])


class FrameStackAcrossCampaigns(gym.Wrapper):
    '''Apila las últimas N observaciones SIN vaciar la pila al cambiar de campaña.

    Ese detalle es el punto: un frame stack normal se reinicia en cada episodio y la
    política pierde justo la información que distingue al MPC retenido.
    '''

    def __init__(self, env, n_frames: int):
        super().__init__(env)
        self.n_frames = int(n_frames)
        base = int(env.observation_space.shape[0])
        self.base_dim = base
        self.frames = deque(maxlen=self.n_frames)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(base * self.n_frames,), dtype=np.float32)

    def _stacked(self):
        return np.concatenate(list(self.frames), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()                      # solo al iniciar el meta-episodio
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)                  # NO se vacía en el límite de campaña
        return self._stacked(), reward, terminated, truncated, info


def make_env(roots, *, frame_stack: int, seed: int = 0):
    specs = [HistorySpec(r, k, FROZEN["campaigns_per_history"])
             for r in roots for k in FROZEN["kappas"]]
    env = MetaCampaignEnv(scheduler=SCHED, histories=specs, retained=True,
                          objective_fn=objective_of, rng_seed=seed)
    env.hidden_retained = True
    return FrameStackAcrossCampaigns(env, frame_stack), specs


TRAIN_ROOTS = list(range(7_680_001, 7_680_001 + 8))
EVAL_ROOTS  = list(range(7_690_001, 7_690_001 + CFG["eval_roots"]))

_probe, _ = make_env(TRAIN_ROOTS[:1], frame_stack=FRAME_STACK)
print(f"[ok] observación base = {_probe.base_dim} dims")
print(f"[ok] observación apilada = {_probe.observation_space.shape[0]} dims "
      f"({FRAME_STACK} frames x {_probe.base_dim})")
print(f"[ok] acciones = {_probe.action_space.n} (cuántos de los 3 lotes semanales son P_C)")
print(f"[ok] entrenamiento: {len(TRAIN_ROOTS)} historias | evaluación: {len(EVAL_ROOTS)} "
      f"historias (disjuntas)")
""")

# ---------------------------------------------------------------- 4. arquitecturas
md(r"""
## Celda 4 — Las arquitecturas, incluida la tuya

**Qué hace:** define las cuatro opciones. `dmlpa_positional` es la default.

**Cómo lee DMLPA el frame stack:** la observación apilada de `FRAME_STACK × 22` se reordena a
una secuencia de `FRAME_STACK` tokens de 22 dims; el transformer atiende sobre ellos y se toma
el último. Es decir, **`FRAME_STACK` es literalmente la longitud de secuencia que ve tu
transformer** — por eso subirlo es tu palanca directa.

**Qué puedes cambiar:** los hiperparámetros están en la Celda 2, no aquí.
""")

code(r"""
# --- Celda 4: arquitecturas -----------------------------------------------------------
import math
import torch
from einops import rearrange
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DMLPAFaithful(BaseFeaturesExtractor):
    '''DMLPA original de David: embedding + Transformer, sin posición explícita.'''

    def __init__(self, observation_space, factor=1, features_dim=120,
                 hidden_dim=100, nhead=12, num_layers=4):
        super().__init__(observation_space, features_dim)
        flat = int(observation_space.shape[0])
        if flat % factor:
            raise ValueError(f"obs {flat} no divisible por factor={factor}")
        self.factor, self.obs_dimension = int(factor), flat // int(factor)
        self.latent_rw = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dimension, hidden_dim), torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, features_dim))
        layer = torch.nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead,
                                                 batch_first=True)
        self.accumulated = torch.nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, obs):
        x = rearrange(obs, "b (d k) -> b d k", d=self.factor)
        return self.accumulated(self.latent_rw(x))[:, -1, :]


class DMLPAPositional(DMLPAFaithful):
    '''DMLPA con posición sinusoidal + LayerNorm. LA ARQUITECTURA COMPLETA DE DAVID.'''

    def __init__(self, observation_space, factor=1, features_dim=120,
                 hidden_dim=100, nhead=12, num_layers=4):
        super().__init__(observation_space, factor, features_dim, hidden_dim,
                         nhead, num_layers)
        self.pre_norm = torch.nn.LayerNorm(features_dim)
        pe = torch.zeros(self.factor, features_dim)
        pos = torch.arange(0, self.factor, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, features_dim, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / features_dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))

    def forward(self, obs):
        x = rearrange(obs, "b (d k) -> b d k", d=self.factor)
        x = self.pre_norm(self.latent_rw(x) + self.pos_encoding)
        return self.accumulated(x)[:, -1, :]


def build_model(env, seed: int):
    '''Devuelve el modelo listo para entrenar según ARCHITECTURE y ALGORITHM.'''
    from stable_baselines3 import PPO
    from sb3_contrib import QRDQN, RecurrentPPO

    common = dict(seed=seed, verbose=0, learning_rate=LEARNING_RATE)

    if ARCHITECTURE == "lstm":
        return RecurrentPPO("MlpLstmPolicy", env, n_steps=96, batch_size=96,
                            ent_coef=ENT_COEF, **common), "RecurrentPPO + LSTM"

    if ARCHITECTURE in ("dmlpa_positional", "dmlpa_faithful"):
        cls = DMLPAPositional if ARCHITECTURE == "dmlpa_positional" else DMLPAFaithful
        pk = dict(features_extractor_class=cls,
                  features_extractor_kwargs=dict(
                      factor=FRAME_STACK, features_dim=DMLPA_FEATURES_DIM,
                      hidden_dim=DMLPA_HIDDEN_DIM, nhead=DMLPA_NHEAD,
                      num_layers=DMLPA_NUM_LAYERS))
        nice = ("DMLPA positional (transformer + PE + LayerNorm)"
                if ARCHITECTURE == "dmlpa_positional" else "DMLPA faithful (transformer)")
    else:
        pk, nice = None, "PPO + MLP (control)"

    if ALGORITHM == "qrdqn":
        return QRDQN("MlpPolicy", env, policy_kwargs=pk, **common), nice + " + QRDQN"
    return PPO("MlpPolicy", env, n_steps=96, batch_size=96, ent_coef=ENT_COEF,
               policy_kwargs=pk, **common), nice + " + PPO"


_m, _name = build_model(_probe, seed=0)
_n = sum(p.numel() for p in _m.policy.parameters())
print(f"[ok] arquitectura: {_name}")
print(f"[ok] parámetros entrenables: {_n:,}")
print(f"[ok] el transformer ve una secuencia de {FRAME_STACK} tokens")
del _m
""")

# ---------------------------------------------------------------- 5. referencias
md(r"""
## Celda 5 — Las dos referencias contra las que te vamos a medir

**Qué hace:** calcula el ReT de la **mejor política estática** sobre exactamente las mismas
campañas donde se evaluará tu política, y fija el nivel del **MPC**.

- **Barra estática**: el calendario fijo `[0,0,3,3,3,3,3,3]`, elegido *fuera* de la muestra de
  evaluación. Es la barra de Garrido: si no le ganas a esto, no hay aprendizaje demostrado.
- **MPC**: el controlador de creencia retenida. Su ventaja medida es **+0.0280** sobre la
  estática. Es el rival real. (Calcularlo aquí costaría ~5 min por campaña, así que se usa el
  valor congelado que ya medimos.)

**Qué puedes cambiar:** nada.
""")

code(r"""
# --- Celda 5: referencias -------------------------------------------------------------
from supply_chain.oracle_curve_v2 import load_history

def static_reference(roots):
    '''ReT medio del mejor calendario estático sobre las campañas de evaluación.'''
    cal, vals = FROZEN["static_bar_calendar"], []
    for root in roots:
        for kappa in FROZEN["kappas"]:
            campaigns, _ = load_history(
                HistorySpec(root, kappa, FROZEN["campaigns_per_history"]), SCHED)
            for c in campaigns:
                vals.append(objective_of(c.skeleton, cal))
    return float(np.mean(vals)), len(vals)

STATIC_MEAN, N_EVAL = static_reference(EVAL_ROOTS)
MPC_MEAN = STATIC_MEAN + FROZEN["mpc_advantage_over_static"]

print(f"campañas de evaluación : {N_EVAL}")
print(f"BARRA ESTÁTICA         : {STATIC_MEAN:.4f}   <- hay que superarla")
print(f"MPC (rival real)       : {MPC_MEAN:.4f}   (+{FROZEN['mpc_advantage_over_static']:.4f})")
""")

# ---------------------------------------------------------------- 6. entrenamiento
md(r"""
## Celda 6 — Entrenamiento

**Qué hace:** entrena una política por semilla sobre historias **disjuntas** de las de
evaluación (para que el resultado no sea memorización).

**Qué mirar mientras corre:** el `ReT medio` de cada semilla. Si se queda pegado y el número
de *calendarios distintos* es 1, la política colapsó a una acción constante — sube `ENT_COEF`.
""")

code(r"""
# --- Celda 6: entrenamiento -----------------------------------------------------------
import time

def evaluate(model, roots, frame_stack):
    '''Rueda la política determinista sobre las campañas de evaluación.'''
    env, specs = make_env(roots, frame_stack=frame_stack)
    values, calendars = [], []
    for spec in specs:
        obs, _ = env.reset(options={"history": spec})
        state, done, first = None, False, True
        while not done:
            action, state = model.predict(obs, state=state,
                                          episode_start=np.array([first]),
                                          deterministic=True)
            first = False
            obs, _r, done, _t, info = env.step(int(action))
            if info.get("campaign_boundary"):
                values.append(float(info["objective"]))
                calendars.append(tuple(info["calendar"]))
    return float(np.mean(values)), len(values), len(set(calendars))

runs = []
for seed in CFG["seeds"]:
    t0 = time.perf_counter()
    env, _ = make_env(TRAIN_ROOTS, frame_stack=FRAME_STACK, seed=seed)
    model, arch_name = build_model(env, seed=seed)
    model.learn(total_timesteps=CFG["timesteps"])
    mean_ret, n_camp, n_cal = evaluate(model, EVAL_ROOTS, FRAME_STACK)
    runs.append(dict(seed=seed, mean_ret=mean_ret, n_campaigns=n_camp,
                     distinct_calendars=n_cal, seconds=time.perf_counter() - t0))
    print(f"semilla {seed}: ReT medio {mean_ret:.4f} | vs estática "
          f"{mean_ret - STATIC_MEAN:+.4f} | calendarios distintos {n_cal} "
          f"| {runs[-1]['seconds']:.0f}s")
""")

# ---------------------------------------------------------------- 7. veredicto
md(r"""
## Celda 7 — **EL VEREDICTO**

Esta es la celda que responde tus tres preguntas. Léela y decide con ella.
""")

code(r"""
# --- Celda 7: veredicto ---------------------------------------------------------------
best = max(runs, key=lambda r: r["mean_ret"])
mean_all = float(np.mean([r["mean_ret"] for r in runs]))
vs_static_best, vs_static_mean = best["mean_ret"] - STATIC_MEAN, mean_all - STATIC_MEAN
vs_mpc = best["mean_ret"] - MPC_MEAN
pct_of_mpc = (vs_static_best / FROZEN["mpc_advantage_over_static"] * 100
              if FROZEN["mpc_advantage_over_static"] else float("nan"))
seeds_above = sum(1 for r in runs if r["mean_ret"] > STATIC_MEAN)
collapsed = sum(1 for r in runs if r["distinct_calendars"] <= 1)

print("=" * 78)
print(f"  {arch_name}   |   frame stack = {FRAME_STACK}   |   preset = {PRESET}")
print("=" * 78)
print(f"  Barra estática ............ {STATIC_MEAN:.4f}")
print(f"  MPC (rival) ............... {MPC_MEAN:.4f}")
print(f"  Tu política (mejor semilla) {best['mean_ret']:.4f}")
print(f"  Tu política (media)  ...... {mean_all:.4f}")
print("-" * 78)

if PRESET == "smoke":
    print("  MODO SMOKE: el notebook corre. NO interpretes estos números.")
elif vs_static_best <= 0:
    print(f"  1) ¿Le ganaste a la estática?  NO  ({vs_static_best:+.4f})")
    print(f"     {seeds_above} de {len(runs)} semillas la superan.")
    print("     -> Todavía no hay aprendizaje demostrado bajo el criterio de Garrido.")
elif seeds_above < len(runs):
    print(f"  1) ¿Le ganaste a la estática?  PARCIAL  ({vs_static_best:+.4f} la mejor)")
    print(f"     Solo {seeds_above} de {len(runs)} semillas. Inestable: corre más semillas.")
else:
    print(f"  1) ¿Le ganaste a la estática?  SÍ  ({vs_static_best:+.4f}, {seeds_above}/"
          f"{len(runs)} semillas)")
    print("     -> Es el primer aprendiz que lo logra. Avísanos.")

print(f"  2) ¿Qué tan cerca del MPC?     {vs_mpc:+.4f}  "
      f"({max(pct_of_mpc, 0):.0f}% de su ventaja)")
if vs_mpc > 0:
    print("     -> LE GANASTE AL MPC. Este es el resultado que buscamos hace meses.")

print(f"  3) ¿El run es válido?")
problems = []
if collapsed:
    problems.append(f"{collapsed}/{len(runs)} semillas colapsaron a acción constante "
                    f"(sube ENT_COEF)")
if len(runs) < 2 and PRESET != "smoke":
    problems.append("una sola semilla: no se puede distinguir señal de suerte")
if best["n_campaigns"] != N_EVAL:
    problems.append(f"campañas evaluadas {best['n_campaigns']} != esperadas {N_EVAL}")
for p in problems:
    print(f"     [!] {p}")
if not problems:
    print("     [ok] sin problemas detectados")
print("=" * 78)
""")

# ---------------------------------------------------------------- 8. export
md(r"""
## Celda 8 — Archivo de resultados para enviar

**Qué hace:** escribe un JSON con todo lo necesario para que auditemos cómo corriste el
experimento y si el resultado es válido, y lo descarga.

**Mándanos ese archivo.** Con él sabemos exactamente qué arquitectura, qué frame stack, qué
semillas y qué números salieron, sin tener que preguntarte nada.
""")

code(r"""
# --- Celda 8: exportar resultados -----------------------------------------------------
import json, datetime, platform

payload = {
    "schema": "scresia_david_rl_vs_mpc_v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "preset": PRESET,
    "claim_status": ("SMOKE_NO_INTERPRETABLE" if PRESET == "smoke"
                     else "DEVELOPMENT_NO_CONFIRMATORY_CLAIM"),
    "configuration": {
        "architecture": ARCHITECTURE, "architecture_name": arch_name,
        "algorithm": ALGORITHM, "frame_stack": FRAME_STACK,
        "dmlpa": dict(features_dim=DMLPA_FEATURES_DIM, hidden_dim=DMLPA_HIDDEN_DIM,
                      nhead=DMLPA_NHEAD, num_layers=DMLPA_NUM_LAYERS),
        "learning_rate": LEARNING_RATE, "ent_coef": ENT_COEF,
        "timesteps": CFG["timesteps"], "seeds": CFG["seeds"],
    },
    "frozen_contract": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in FROZEN.items()},
    "splits": {"train_roots": TRAIN_ROOTS, "eval_roots": EVAL_ROOTS,
               "disjoint": not set(TRAIN_ROOTS) & set(EVAL_ROOTS)},
    "references": {"static_bar_mean_ret": STATIC_MEAN, "mpc_mean_ret": MPC_MEAN,
                   "eval_campaigns": N_EVAL},
    "runs": runs,
    "verdict": {
        "best_mean_ret": best["mean_ret"], "mean_over_seeds": mean_all,
        "vs_static_best": vs_static_best, "vs_static_mean": vs_static_mean,
        "vs_mpc_best": vs_mpc, "percent_of_mpc_advantage": pct_of_mpc,
        "seeds_above_static": seeds_above, "seeds_total": len(runs),
        "collapsed_seeds": collapsed, "problems": problems,
    },
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "gpu": torch.cuda.is_available()},
}

name = (f"scresia_david_{ARCHITECTURE}_fs{FRAME_STACK}_{PRESET}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json")
with open(name, "w") as fh:
    json.dump(payload, fh, indent=1, sort_keys=True)
print(f"[ok] resultados escritos en {name}")

try:
    from google.colab import files
    files.download(name)
except Exception:
    print("(si no estás en Colab, descarga el archivo desde el panel de archivos)")
""")

md(r"""
## Qué hacer según lo que te salga

| Resultado | Siguiente paso |
|---|---|
| Colapso a acción constante | Sube `ENT_COEF` a 0.03-0.05 y baja `LEARNING_RATE` a 1e-4 |
| Por debajo de la estática, estable | Sube `FRAME_STACK` a 24 o 32: más contexto entre campañas |
| Cerca de la estática | Corre `PRESET="final"` con las 5 semillas |
| Le ganas a la estática | Mándanos el JSON enseguida: es el primer aprendiz que lo logra |
| DMLPA ≈ MLP | El positional encoding no está pagando: compara `dmlpa_faithful` vs `mlp` |

**Comparaciones que nos sirven más:** corre el mismo `PRESET` cambiando **una sola cosa** cada
vez (arquitectura, o frame stack, o algoritmo) y mándanos los JSON. Así aislamos qué factor
aporta, que es exactamente lo que no pudimos hacer cuando cambiamos varias cosas a la vez.
""")

nb = {
    "cells": [
        {"cell_type": kind, "metadata": {},
         **({"source": src.splitlines(keepends=True)} if kind == "markdown"
            else {"source": src.splitlines(keepends=True), "outputs": [],
                  "execution_count": None})}
        for kind, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"notebook escrito: {OUT}")
print(f"celdas: {len(CELLS)} ({sum(1 for k, _ in CELLS if k == 'code')} de código)")
