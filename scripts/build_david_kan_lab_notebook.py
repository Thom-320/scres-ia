#!/usr/bin/env python3
"""Generate David's KAN laboratory notebook for Colab and Kaggle.

WHY A NEW LAB. The Program O-R workbench gives KAN a four-action discrete problem with 21
observation features and 8 decisions per episode. A lookup table wins there, and we measured it:
Delta_N is negative in all three O-R cells. Handing a high-capacity approximator a problem that
small is not a fair test of the approximator -- it is a test of whether extra capacity can hurt.

This lab moves the same question to the richest decision problem in the repository: Track B, with
an 8-dimensional continuous action and 101 observation features over 104 weekly decisions. That is
Garrido's own 28 July instruction -- add decision variables, do not lengthen the episode -- applied
to the architecture comparison.

THE FAIRNESS FIX DAVID ASKED FOR. His current run trains 2,496,295 parameters against an MLP
baseline two orders of magnitude smaller, and he said so himself: the comparison is not fair unless
the parameter counts match. Every architecture here is auto-sized to a shared budget and the
notebook refuses to train if they land more than 10% apart.

THE SPEED FIX HE HAS NOT SEEN. His cell builds `KAN(width=[...])` with pykan's defaults, so
`auto_save`, `save_act` and `symbolic_enabled` are all ON -- that is why his run printed
"checkpoint directory created" and why it crawls. This repository already measured ~160x forward
slowdown from those flags and ships `scripts/real_kan_extractor.py` with them off.

Output: notebooks/scresia_david_kan_lab.ipynb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BRANCH = "david/kan-lab"
REPO = "Thom-320/scres-ia"
NOTEBOOK = Path("notebooks/scresia_david_kan_lab.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


CELLS = [
    md(f"""# SCRES-IA — laboratorio KAN de David

**Pregunta:** ¿puede una KAN aprender una política mejor que un MLP **con el mismo presupuesto de
parámetros**?

Tres cosas que este cuaderno cambia respecto al de Program O-R:

**1 · El entorno tiene sitio para que una arquitectura importe.** O-R da `Discrete(4)`, 21
observaciones y 8 decisiones por episodio; ahí una tabla de consulta gana, y lo medimos —`Δ_N`
negativo en las tres celdas—. Aquí se usa **Track B: acción continua de 8 dimensiones, 101
observaciones, 104 decisiones semanales**. Es la instrucción de Garrido del 28 de julio —añadir
variables de decisión, no alargar el episodio— aplicada a la comparación de arquitecturas.

**2 · Los parámetros se igualan.** Tu objeción era correcta: tu corrida entrenaba **2.496.295**
parámetros contra un MLP dos órdenes de magnitud menor. Aquí **todas** las arquitecturas se
dimensionan a un presupuesto común y el cuaderno **se niega a entrenar** si quedan a más del 10 %.

**3 · La KAN va rápida.** Tu celda construye `KAN(width=[...])` con los valores por defecto de
pykan, así que `auto_save`, `save_act` y `symbolic_enabled` quedan **encendidos** — por eso te
imprimió *"checkpoint directory created"* y por eso va lenta. Medimos **~160× de sobrecoste** en el
forward por esos flags. Aquí se usan apagados.

> Todo es **desarrollo**. Un buen resultado selecciona un candidato para un experimento
> preregistrado; no es evidencia de paper por sí solo.
"""),

    code(f"""# 0) LA ÚNICA CELDA QUE TIENES QUE TOCAR
RUN_PROFILE   = 'preliminary'   # 'smoke' (cableado, ~2 min) | 'preliminary' (1 semilla) | 'final' (multi-semilla)
ARCH          = 'KAN'           # 'KAN' | 'MLP' | 'DMLPA' | 'CUSTOM'
MEMORY_ARM    = 'independent'   # 'independent' (compara arquitecturas) | 'persistent' (aprendizaje entre corridas)

TARGET_PARAMS = 200_000         # presupuesto COMPARTIDO de parámetros del extractor
PARAM_TOLERANCE = 0.10          # el cuaderno aborta si dos arquitecturas quedan a más de esto

DEVICE        = 'auto'          # 'auto' mide CPU vs GPU y elige | 'cpu' | 'cuda'
N_ENVS        = 'auto'          # 'auto' = núcleos-1 (máx 8). El cuello es el DES, no la red.

HISTORY_LEN   = 8               # frame stack: 101 x HISTORY_LEN entra a la red
OBS_VERSION   = 'v10'           # v7=52 | v8=79 | v9=89 | v10=101 features
MAX_STEPS     = 104             # decisiones por episodio (semanales)

GIT_URL, GIT_BRANCH = 'https://github.com/{REPO}.git', '{BRANCH}'

# Perfiles. El presupuesto real lo fija la SONDA DE TIEMPO de la celda 3, que mide esta máquina
# y te dice cuántas semillas caben; estos son sólo los puntos de partida.
if RUN_PROFILE == 'smoke':
    TOTAL_STEPS, N_STEPS, SEEDS = 2_048, 256, [9491]
    EVAL_EPISODES = 3
elif RUN_PROFILE == 'preliminary':
    TOTAL_STEPS, N_STEPS, SEEDS = 30_000, 512, [9491]
    EVAL_EPISODES = 12
elif RUN_PROFILE == 'final':
    # Medido: ~97 pasos/s con 4 envs paralelos y KAN de 200k parámetros.
    # 150k x 5 semillas ~ 2,2 h de entrenamiento + evaluación. Cabe en las 9 h con holgura.
    TOTAL_STEPS, N_STEPS, SEEDS = 150_000, 512, [9491, 9492, 9493, 9494, 9495]
    EVAL_EPISODES = 24
else:
    raise ValueError(RUN_PROFILE)

print(f'{{ARCH}} · {{RUN_PROFILE}} · {{TOTAL_STEPS:,}} pasos x {{len(SEEDS)}} semilla(s) '
      f'· frame stack {{HISTORY_LEN}} · obs {{OBS_VERSION}}')"""),

    code("""# 1) Setup portátil: local, Colab o Kaggle
from pathlib import Path
import os, subprocess, sys, time

def run(cmd, cwd=None):
    print('+', ' '.join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)

cwd = Path.cwd().resolve()
if (cwd / 'supply_chain').exists():
    ROOT = cwd
elif Path('/kaggle/working').exists():
    ROOT = Path('/kaggle/working/scres-ia')
elif Path('/content').exists():
    ROOT = Path('/content/scres-ia')
else:
    raise RuntimeError('Corre desde el repo, Colab o Kaggle')

if not (ROOT / '.git').exists():
    # SPARSE CHECKOUT. El repo entero pesa ~860 MB (resultados sellados de otros carriles) y en
    # Colab/Kaggle eso son minutos de descarga por nada: este cuaderno sólo necesita el simulador
    # y dos scripts. --filter=blob:none trae los blobs bajo demanda.
    try:
        run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
             '--branch', GIT_BRANCH, GIT_URL, ROOT])
        run(['git', 'sparse-checkout', 'set', 'supply_chain', 'scripts', 'contracts',
             'notebooks'], cwd=ROOT)
    except subprocess.CalledProcessError:
        print('  sparse checkout no disponible; clone completo (tarda más)')
        run(['git', 'clone', '--depth', '1', '--branch', GIT_BRANCH, GIT_URL, ROOT])
os.chdir(ROOT)
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    import gymnasium, stable_baselines3, simpy, kan
except ImportError:
    run([sys.executable, '-m', 'pip', 'install', '-q', 'gymnasium>=1.2',
         'stable-baselines3>=2.7', 'sb3-contrib>=2.7', 'simpy>=4.1', 'pykan==0.2.8',
         'einops', 'pandas>=2.0', 'matplotlib>=3.7'])
print('ROOT =', ROOT)
print('commit =', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())"""),

    code("""# 2) El entorno, y por qué éste y no el de Program O-R
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, gymnasium as gym
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from supply_chain.external_env_interface import make_track_b_env

class HistoryStack(gym.Wrapper):
    \"\"\"Frame stack causal: apila las últimas HISTORY_LEN observaciones. Nunca mira el futuro.\"\"\"
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n
        d = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (d * n,), dtype=np.float32)
    def reset(self, **kw):
        o, i = self.env.reset(**kw)
        self.buf = [o] * self.n
        return np.concatenate(self.buf).astype(np.float32), i
    def step(self, a):
        o, r, term, trunc, i = self.env.step(a)
        self.buf = self.buf[1:] + [o]
        return np.concatenate(self.buf).astype(np.float32), r, term, trunc, i

def make_env(seed=None):
    e = make_track_b_env(observation_version=OBS_VERSION, max_steps=MAX_STEPS)
    e = HistoryStack(e, HISTORY_LEN)
    if seed is not None:
        e.reset(seed=seed)
    return e

import multiprocessing as mp
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

n_envs = max(1, min(8, (os.cpu_count() or 2) - 1)) if N_ENVS == 'auto' else int(N_ENVS)

def make_vec(seed=None):
    \"\"\"Envs en paralelo. El cuello de botella es el DES (simpy, puro Python en CPU), no la red:
    con 4 procesos medimos 97 pasos/s contra 59 con uno solo. La GPU no toca esa parte.\"\"\"
    if n_envs == 1:
        return DummyVecEnv([lambda: make_env(seed)])
    # start_method='fork' EXPLÍCITO. Con 'spawn' (macOS/Windows) los procesos hijos vuelven a
    # importar el módulo y una celda de notebook se re-ejecutaría entera. Colab y Kaggle son
    # Linux y tienen fork; fuera de ahí degradamos a un solo entorno en vez de romper.
    if 'fork' not in mp.get_all_start_methods():
        print('  sin fork disponible; uso DummyVecEnv (1 entorno)')
        return DummyVecEnv([lambda: make_env(seed)])
    try:
        return SubprocVecEnv([lambda: make_env(None) for _ in range(n_envs)],
                             start_method='fork')
    except Exception as exc:
        print(f'  SubprocVecEnv no disponible ({type(exc).__name__}); uso DummyVecEnv')
        return DummyVecEnv([lambda: make_env(seed)])

_e = make_env(0)
FLAT_DIM = _e.observation_space.shape[0]
print(f'Track B      : acción {_e.action_space}  ·  obs {_e.observation_space.shape}  ·  '
      f'{MAX_STEPS} decisiones/episodio')
print(f'Program O-R  : acción Discrete(4)          ·  obs (21,)        ·  8 decisiones/episodio')
print()
print('8 variables de decisión continuas contra 4 acciones discretas. Ahí es donde una')
print('arquitectura puede marcar diferencia; en O-R una tabla de consulta ya gana.')
print()
print(f'núcleos detectados: {os.cpu_count()}  ->  {n_envs} entornos en paralelo')"""),

    code("""# 3) SONDA DE DISPOSITIVO Y TIEMPO — mide ESTA máquina; no asume nada
KAGGLE_LIMIT_S = 9 * 3600
SAFETY = 0.80          # 20 % de margen: guardar, evaluar, y que la sesión no te corte

print('GPU visible:', torch.cuda.is_available(),
      f'({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else '')
print()
print('Por qué esto se mide y no se decide de antemano:')
print('  · el cuello de botella es el DES (simpy, Python puro, CPU) -- la GPU no lo toca;')
print('  · SB3 avisa que PPO con MlpPolicy suele ir MEJOR en CPU: los minibatches son de 64')
print('    y el trasiego CPU<->GPU pesa más que el cálculo;')
print('  · pero una KAN de 200k parámetros tiene un forward bastante más caro que un MLP,')
print('    así que en TU arquitectura puede salir a cuenta. Se mide y se elige.')

def time_device(build_fn, device, steps=512):
    venv = make_vec(0)
    try:
        m = build_fn(venv, 0, device=device)
        t0 = time.time()
        m.learn(total_timesteps=steps, progress_bar=False)
        return steps / (time.time() - t0)
    except Exception as exc:
        print(f'  {device}: falló ({type(exc).__name__}: {exc})')
        return 0.0
    finally:
        venv.close()

def pick_device(build_fn):
    if DEVICE != 'auto':
        return DEVICE, None
    speeds = {'cpu': time_device(build_fn, 'cpu')}
    if torch.cuda.is_available():
        speeds['cuda'] = time_device(build_fn, 'cuda')
    for d, v in speeds.items():
        print(f'  {d:<5} {v:7.1f} pasos/s')
    best = max(speeds, key=speeds.get)
    if len(speeds) > 1:
        other = min(speeds, key=speeds.get)
        ratio = speeds[best] / max(speeds[other], 1e-9)
        print(f'  -> elijo {best} ({ratio:.2f}x sobre {other})')
        if best == 'cpu':
            print('     Nota de Kaggle: si la GPU no ayuda, apágala en Settings y no gastes')
            print('     cuota (30 h/semana). En Colab, un runtime CPU se preempta menos.')
    return best, speeds

def budget_report(steps_per_s):
    per_seed = TOTAL_STEPS / max(steps_per_s, 1e-9)
    usable = KAGGLE_LIMIT_S * SAFETY
    fits = int(usable // per_seed) if per_seed > 0 else 0
    print(f'  coste por semilla  : {per_seed/60:,.1f} min ({TOTAL_STEPS:,} pasos)')
    print(f'  semillas que caben : {fits}  (en {usable/3600:.1f} h útiles de 9 h)')
    if fits < len(SEEDS):
        print(f'  AVISO: pediste {len(SEEDS)} y sólo caben {fits}. Baja TOTAL_STEPS o SEEDS.')
        print('         Aun así, la celda 6 guarda CADA semilla al terminarla: un corte de')
        print('         sesión no borra lo ya hecho.')
    else:
        print(f'  holgura            : {(usable - per_seed*len(SEEDS))/60:.0f} min de sobra')
    return fits

print()
print('Ejecuta la celda 3-bis DESPUÉS de la 5 (necesita el constructor de modelos).')"""),



    md("""## 4) TU CELDA — la arquitectura vive aquí

Edita libremente. Las tres funciones tienen que devolver un `BaseFeaturesExtractor` y aceptar un
vector plano de `101 × HISTORY_LEN`.

**Lo único que no debes tocar** es que se dimensionen al mismo `TARGET_PARAMS`: es lo que hace que
«la KAN gana» signifique algo. Si quieres cambiar el presupuesto, cambia `TARGET_PARAMS` en la
celda 0 y **todas** las arquitecturas se re-dimensionan juntas."""),

    code("""# ===== LA ARQUITECTURA DE DAVID — EDITABLE =====
import math
from einops import rearrange
from real_kan_extractor import RealKANFeaturesExtractor   # pykan con los flags APAGADOS

def count_params(module):
    return sum(p.numel() for p in module.parameters())

def size_to_budget(factory, lo, hi, budget, tol=PARAM_TOLERANCE):
    \"\"\"Busca el ancho que acerca la arquitectura al presupuesto compartido.

    Sin esto la comparación mide capacidad, no arquitectura -- que es justo la objeción que
    levantaste sobre el preprint anti-KAN.
    \"\"\"
    best, best_err = None, float('inf')
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(factory(mid))
        err = abs(n - budget) / budget
        if err < best_err:
            best, best_err = mid, err
        if n < budget:
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_err

# ---- KAN -------------------------------------------------------------------------------------
def kan_factory(hidden):
    space = gym.spaces.Box(-np.inf, np.inf, (FLAT_DIM,), dtype=np.float32)
    return RealKANFeaturesExtractor(space, features_dim=64, hidden_width=int(hidden),
                                    grid=3, k=3)

# ---- MLP -------------------------------------------------------------------------------------
class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, hidden=256):
        super().__init__(observation_space, features_dim)
        d = int(observation_space.shape[0])
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, features_dim), nn.LayerNorm(features_dim))
    def forward(self, x):
        return self.net(x.float())

def mlp_factory(hidden):
    space = gym.spaces.Box(-np.inf, np.inf, (FLAT_DIM,), dtype=np.float32)
    return MLPExtractor(space, features_dim=64, hidden=int(hidden))

# ---- DMLPA (tu transformer) ------------------------------------------------------------------
class DMLPA(BaseFeaturesExtractor):
    \"\"\"Tu DMLPA, con el frame stack como secuencia de tokens.\"\"\"
    def __init__(self, observation_space, factor=HISTORY_LEN, features_dim=120,
                 hidden_dim=100, nhead=12, num_layers=2, ff_mult=4, use_kan=False):
        super().__init__(observation_space, features_dim)
        flat = int(observation_space.shape[0])
        if flat % factor:
            raise ValueError(f'{flat} no es divisible por factor={factor}')
        if features_dim % nhead:
            raise ValueError('features_dim debe ser divisible por nhead')
        self.obs_dimension, self.factor = flat // factor, factor
        if use_kan:
            from kan import KAN
            self.latent_rw = KAN(width=[self.obs_dimension, hidden_dim, features_dim],
                                 grid=3, k=3, auto_save=False, save_act=False,
                                 symbolic_enabled=False)
        else:
            self.latent_rw = nn.Sequential(nn.Linear(self.obs_dimension, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, features_dim))
        self.pre_norm = nn.LayerNorm(features_dim)
        # dim_feedforward EXPLÍCITO. El valor por defecto de PyTorch es 2048, y con d_model=120
        # eso son ~492k parámetros POR CAPA: tu DMLPA arrastraba ~1,1 M de feedforward que
        # probablemente no pretendías, y por eso no podía compararse a 200k con nadie.
        layer = nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead, batch_first=True,
                                           dim_feedforward=ff_mult * features_dim)
        self.accumulated = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.register_buffer('pos', self._pe(factor, features_dim))
    @staticmethod
    def _pe(seq, d):
        pe = torch.zeros(seq, d)
        pos = torch.arange(seq).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe.unsqueeze(0)
    def forward(self, x):
        x = rearrange(x.float(), 'b (d k) -> b d k', d=self.factor)
        x = self.latent_rw(x)
        x = self.pre_norm(x + self.pos)
        return self.accumulated(x)[:, -1, :]

def dmlpa_factory(width):
    # El ancho que se escala es d_model (múltiplo de nhead), no hidden_dim: el grueso de los
    # parámetros vive en el transformer, así que mover hidden_dim no mueve el total.
    d = max(12, int(width) // 12 * 12)
    return DMLPA(gym.spaces.Box(-np.inf, np.inf, (FLAT_DIM,), dtype=np.float32),
                 hidden_dim=max(32, d), features_dim=d, nhead=12, num_layers=2)

FACTORIES = {'KAN': (kan_factory, 4, 64), 'MLP': (mlp_factory, 8, 512),
             'DMLPA': (dmlpa_factory, 12, 480)}
print('arquitecturas disponibles:', list(FACTORIES))"""),

    code("""# 5) IGUALDAD DE PARÁMETROS — el cuaderno aborta si no se cumple
sizes = {}
for name, (factory, lo, hi) in FACTORIES.items():
    width, err = size_to_budget(factory, lo, hi, TARGET_PARAMS)
    n = count_params(factory(width))
    sizes[name] = {'width': width, 'params': n, 'error': err}
    print(f'  {name:<6} ancho={width:<4} parámetros={n:>9,}  desviación={err:6.1%}')

worst = max(v['error'] for v in sizes.values())
if worst > PARAM_TOLERANCE:
    raise SystemExit(
        f'Las arquitecturas quedan a {worst:.1%} del presupuesto (tolerancia {PARAM_TOLERANCE:.0%}).\\n'
        'Ajusta TARGET_PARAMS o los rangos de FACTORIES. Comparar capacidades distintas\\n'
        'mide capacidad, no arquitectura -- que es exactamente tu objeción al preprint.')
print(f'\\nOK: las tres dentro del {PARAM_TOLERANCE:.0%}. La comparación es de arquitectura.')

def build_model(env, seed, device='cpu'):
    if ARCH == 'CUSTOM':
        return build_custom_model(env, seed, device)   # defínela tú en la celda 4
    factory, _, _ = FACTORIES[ARCH]
    width = sizes[ARCH]['width']
    kwargs = {'KAN': {'features_extractor_class': RealKANFeaturesExtractor,
                      'features_extractor_kwargs': {'features_dim': 64, 'hidden_width': width,
                                                    'grid': 3, 'k': 3}},
              'MLP': {'features_extractor_class': MLPExtractor,
                      'features_extractor_kwargs': {'features_dim': 64, 'hidden': width}},
              'DMLPA': {'features_extractor_class': DMLPA,
                        'features_extractor_kwargs': {
                            'hidden_dim': max(32, int(width) // 12 * 12),
                            'features_dim': max(12, int(width) // 12 * 12),
                            'nhead': 12, 'num_layers': 2}}}[ARCH]
    kwargs['net_arch'] = dict(pi=[64, 64], vf=[64, 64])
    return PPO('MlpPolicy', env, seed=seed, device=device, learning_rate=3e-4,
               n_steps=N_STEPS, batch_size=64, gamma=0.99, gae_lambda=0.95,
               clip_range=0.2, ent_coef=0.01, policy_kwargs=kwargs, verbose=0)"""),

    code("""# 3-bis) Elegir dispositivo y comprobar que el plan cabe en las 9 h
BEST_DEVICE, device_speeds = pick_device(build_model)
print()
fits = budget_report((device_speeds or {}).get(BEST_DEVICE) or time_device(build_model, BEST_DEVICE))
print(f'\\ndispositivo elegido: {BEST_DEVICE}  ·  {n_envs} entornos en paralelo')"""),

    md("""## 6) Los dos brazos, y qué mide cada uno

**`independent`** — cada semilla arranca de cero. Es lo **correcto para comparar arquitecturas**:
las semillas son réplicas independientes y sus intervalos de confianza son válidos.

**`persistent`** — los pesos cruzan de una semilla a la siguiente. Esto es lo que pediste, y **no
es un arreglo del anterior: es otro experimento**. Mide aprendizaje *entre corridas* (el efecto
Alzheimer de Garrido), y sin su control `reset` no se puede distinguir de «entrenó 5 veces más».
Por eso el brazo persistente corre **siempre** con su gemelo reseteado."""),

    code("""# 6) Entrenar y evaluar, con checkpoint por semilla (un corte de Kaggle no lo pierde todo)
from datetime import datetime, timezone
OUT = ROOT / 'outputs' / 'david_kan_lab' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True, exist_ok=True)

def evaluate(model, episodes, seed0=777_000):
    rets = []
    for k in range(episodes):
        env = make_env()
        obs, _ = env.reset(seed=seed0 + k)
        done, total = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r); done = term or trunc
        rets.append(total)
    return float(np.mean(rets)), float(np.std(rets)), rets

def train_arm(arm):
    rows, carried = [], None
    for i, seed in enumerate(SEEDS):
        env = make_vec(seed)
        model = build_model(env, seed, device=BEST_DEVICE)
        if arm == 'persistent' and carried is not None:
            model.policy.load_state_dict(carried)       # los pesos cruzan la frontera
        t0 = time.time()
        model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)
        mean, sd, _ = evaluate(model, EVAL_EPISODES)
        carried = {k: v.clone() for k, v in model.policy.state_dict().items()}
        rows.append({'arm': arm, 'arch': ARCH, 'seed': seed, 'order': i,
                     'ret_mean': mean, 'ret_sd': sd,
                     'params': sizes[ARCH]['params'] if ARCH in sizes else -1,
                     'train_s': time.time() - t0})
        pd.DataFrame(rows).to_csv(OUT / f'{ARCH}_{arm}.csv', index=False)   # checkpoint
        model.save(OUT / f'{ARCH}_{arm}_seed{seed}')
        print(f'  {arm:<11} semilla {seed}  ReT {mean:+.5f} ± {sd:.5f}  '
              f'({time.time()-t0:.0f}s)  [{i+1}/{len(SEEDS)}]')
    return pd.DataFrame(rows)

arms = ['independent'] if MEMORY_ARM == 'independent' else ['persistent', 'independent']
results = pd.concat([train_arm(a) for a in arms], ignore_index=True)
results.to_csv(OUT / 'results.csv', index=False)
display(results)"""),

    code("""# 7) Δ_efficiency: calidad CONTRA coste. Es el estimando que el contrato E* declara.
def ms_per_decision(model, n=200):
    env = make_env(); obs, _ = env.reset(seed=0)
    for _ in range(20):
        model.predict(obs, deterministic=True)          # calentamiento
    t0 = time.time()
    for _ in range(n):
        model.predict(obs, deterministic=True)
    return 1000.0 * (time.time() - t0) / n

probe = build_model(make_vec(0), 0, device=BEST_DEVICE)
row = {'arch': ARCH, 'params': sizes.get(ARCH, {}).get('params', -1),
       'ms_per_decision': ms_per_decision(probe),
       'ret_mean': float(results.query("arm == 'independent'")['ret_mean'].mean()),
       'ret_sd': (float(results.query("arm == 'independent'")['ret_mean'].std())
                  if len(SEEDS) > 1 else None),
       'n_seeds': len(SEEDS)}
pd.DataFrame([row]).to_csv(OUT / 'efficiency.csv', index=False)
import json as _json
print(_json.dumps(row, indent=2))
if len(SEEDS) < 2:
    print('\\nUNA SOLA SEMILLA: esto es un humo. La desviación entre semillas es lo que decide')
    print('si una arquitectura gana, y con n=1 no existe. Usa RUN_PROFILE=final.')
print()
print('Regla de lectura fijada de antemano: si dos arquitecturas EMPATAN en ReT')
print('(intervalos que se solapan), gana la más barata en parámetros y ms/decisión.')
print('Medimos eso mismo para surrogates y ganó una neurona de 5 parámetros.')"""),

    md("""## Reglas de interpretación

1. **Compara sobre las mismas cintas de evaluación**, no sobre la recompensa de entrenamiento.
2. **Una semilla es un humo.** Para afirmar que una arquitectura gana hacen falta varias, y con el
   brazo `independent` — el `persistent` no tiene réplicas independientes por construcción.
3. **Empate en calidad lo gana el más barato.** Ese es `Δ_efficiency` y es un resultado publicable
   por sí solo: en nuestro banco de surrogates, KAN y MLP empataron y ganó una neurona de **5**
   parámetros con 30× menos coste por decisión.
4. **Un modelo que repite un único plan descubrió un horario, no demostró feedback.**
5. Si encuentras algo prometedor: **congela el código y los hiperparámetros primero**, y luego se
   escribe un contrato preregistrado. Un resultado elegido después de mirar es selección post-hoc.
6. Esto es desarrollo. No altera ningún veredicto sellado del proyecto.
"""),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=NOTEBOOK)
    args = ap.parse_args()
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
            "accelerator": "None",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"  {args.output}  ({len(CELLS)} celdas)")
    print(f"  Colab : https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{args.output}")
    print(f"  Kaggle: https://www.kaggle.com/kernels/welcome?src=https://github.com/{REPO}/blob/"
          f"{BRANCH}/{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
