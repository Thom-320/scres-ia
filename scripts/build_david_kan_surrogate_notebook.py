#!/usr/bin/env python3
"""David's most promising notebook: KAN where it has a MEASURED advantage.

WHY THIS ONE AND NOT THE OTHERS. Every environment we searched today is closed. The expanded buffer
contract is saturated upward -- x10 on any node moves resilience by exactly 0.000000. The 288/4608
configuration surface has one invariant argmax. The v0 recovery surface picks [0, 672, 168] in 42
of 42 selections across twelve seeds. Where the optimal action does not vary, no learner can beat a
constant, because the optimal policy IS a constant.

There is exactly one place where a KAN has a measured, parameter-matched advantage, and it is not
as a policy. As a SUPERVISED SURROGATE of the design surface -- four design coordinates in,
resilience out -- KAN reaches held-out R2 of 0.9673-0.9978 against a 529-parameter MLP's
0.7424-0.9599, ahead in all six contexts (results/kan_interpretability). And in the search ladder,
surrogate_kan already TIES the 5-parameter neuron at -0.0010 [-0.0061, +0.0043]
(results/search_surrogates). Parity is demonstrated; the open question is whether the surrogate's
accuracy edge converts into a better SEARCH.

That is also Garrido's Fig. 5 position: the neuron sits between node 3 and node 8, learning the
design surface across runs. And it is the one task the KANbeFair preprint itself concedes to KAN --
function representation.

IT RUNS ON THE CACHED SURFACE. No DES, no GPU, minutes not hours. The whole comparison is
searching an already-computed 288-point surface under a budget of 24 evaluations.

Output: notebooks/scresia_david_kan_surrogate_search.ipynb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BRANCH, REPO = "david/kan-lab", "Thom-320/scres-ia"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


CELLS = [
    md("""# SCRES-IA — la KAN donde sí tiene ventaja medida

**Este cuaderno sustituye al laboratorio anterior.** Aquel te ponía a comparar arquitecturas como
políticas de control, y eso ya está medido: **no separan**. Éste te pone en el único sitio donde la
KAN **sí** gana con parámetros igualados.

## Por qué aquí y no en los otros sitios

Hoy cerramos tres entornos, todos por el mismo motivo:

| entorno | qué se midió |
|---|---|
| contrato expandido de buffers | **saturado**: ×10 en cualquier nodo mueve la resiliencia `+0,000000` |
| superficie de configuración 288/4.608 | un solo argmax invariante al régimen |
| superficie de recuperación v0 (12 semillas) | `[0, 672, 168]` elegida **42 de 42 veces** |

> Donde la acción óptima no varía, **ninguna red puede batir a una constante — porque la política
> óptima *es* una constante.** No es un fallo de las redes.

## Dónde sí hay ventaja, y está medida

**La KAN como surrogate supervisado de la superficie de diseño.** A **532 contra 529 parámetros**,
evaluado sobre un cuarto retenido:

| contexto | KAN R²_out | MLP R²_out |
|---|---:|---:|
| R1r | **0,9978** | 0,9418 |
| **R2r** | **0,9777** | **0,7424** |
| R1r+R2r | 0,9945 | 0,9425 |
| R1r\\|esc | 0,9915 | 0,9312 |
| R2r\\|esc | 0,9673 | 0,8912 |
| R1r+R2r\\|esc | 0,9908 | 0,9599 |

Y en la escalera de búsqueda, `surrogate_kan` **ya empata** con la neurona de 5 parámetros:
Δ −0,0010 [−0,0061 · +0,0043].

**La paridad está demostrada. La pregunta abierta es tuya:** ¿esa ventaja en precisión se convierte
en **mejor búsqueda**? Es la posición de la Fig. 5 de Garrido —la neurona entre el nodo ③ y el ⑧,
aprendiendo la superficie entre corridas— y es la única tarea que el propio preprint KANbeFair le
concede a la KAN: **representación de funciones**.

## Lo que corre y lo que cuesta

**Sobre la superficie ya cacheada. Sin DES, sin GPU, minutos.** Buscas 288 configuraciones con
presupuesto 24, con la KAN como surrogate dentro del bucle, contra los mismos comparadores.
"""),

    md("""> ## ⚠️ En Kaggle: enciende **Internet** antes de ejecutar
>
> El error `CalledProcessError ... exit status 128` en el `git clone` **no es del repositorio** —
> es público y la rama existe; lo comprobamos con `git ls-remote` anónimo.
>
> Es que **el kernel tenía Internet apagado**. Al abrir el cuaderno por link, Kaggle no aplica el
> `kernel-metadata.json` que trae `enable_internet: true`, así que arranca en OFF.
>
> **Panel derecho → `Internet` → ON**, y vuelve a ejecutar. La celda 1 ahora lo detecta y te lo
> dice en una línea en vez de escupir un traceback.
"""),

    code("""# 1) Setup — comprueba Internet ANTES de intentar el clone
import os, socket, subprocess, sys, json
from pathlib import Path

def internet_ok(host='github.com', port=443, timeout=5):
    try:
        socket.create_connection((host, port), timeout=timeout); return True
    except OSError:
        return False

if not internet_ok():
    raise SystemExit(
        'SIN INTERNET. En Kaggle: panel derecho -> Internet -> ON, y vuelve a ejecutar.\\n'
        'El clone falla con exit 128 por esto, no por el repositorio: es publico y la rama existe.')

GIT_URL, GIT_BRANCH = 'https://github.com/{repo}.git', '{branch}'
ROOT = Path('/kaggle/working/scres-ia') if Path('/kaggle').exists() else Path('/content/scres-ia')
if not ROOT.exists():
    subprocess.check_call(['git','clone','--depth','1','--branch',GIT_BRANCH,GIT_URL,str(ROOT)])
os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy','scikit-learn','scipy',
                       'torch','pykan==0.2.8'])
print('commit:', subprocess.check_output(['git','rev-parse','HEAD']).decode().strip())"""
         .replace('{repo}', REPO).replace('{branch}', BRANCH)),

    code("""# 2) La superficie cacheada y los comparadores que ya existen. No se re-simula nada.
from run_search_comparator_ladder_v2 import (
    COORDS, N_CFG, BUDGET, GP_N_INIT, Surface, load_cache,
    arm_random, arm_ofat, arm_gp_ei, arm_ucb1, make_neuron_arm,
)
import numpy as np

surface, contexts, seeds = load_cache(Path('results/surface_cache/wrap288_v1'))
print(f'{len(contexts)} contextos x {len(seeds)} semillas x {N_CFG} configuraciones')
print(f'presupuesto por contexto: {BUDGET} evaluaciones')"""),

    code("""# 3) TU BRAZO: la KAN como surrogate DENTRO de la busqueda.
# Ajusta sobre lo ya visitado, puntua lo no visitado, evalua, repite. Esa es la Fig. 5 de Garrido.
# El esqueleto esta completo salvo el modelo: mete el tuyo donde dice.
import torch

TARGET_PARAMS = 532          # el conteo con el que la KAN gana como surrogate. Igualalo si cambias.

def fit_surrogate(kind, x, y, steps=300):
    torch.manual_seed(9201)
    if kind == 'kan':
        from kan import KAN
        model = KAN(width=[x.shape[1], 5, 1], grid=5, k=3,
                    auto_save=False, save_act=False, symbolic_enabled=False)
    elif kind == 'mlp':
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(x.shape[1], 88), nn.Tanh(), nn.Linear(88, 1))
    else:
        raise ValueError(kind)
    xt = torch.tensor(x, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32).view(-1,1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        opt.zero_grad(); ((model(xt)-yt)**2).mean().backward(); opt.step()
    return model, sum(p.numel() for p in model.parameters())

def make_surrogate_arm(kind):
    def arm(s, rng, budget):
        for idx in rng.permutation(N_CFG)[:GP_N_INIT]:
            s.select(int(idx))
        while len(s.visited) < budget:
            seen = sorted(s._seen)
            y = np.array([s.value_of_visited(i) for i in seen])
            lo, hi = y.min(), y.max()
            yn = (y - lo)/(hi - lo) if hi > lo else np.zeros_like(y)
            model, _ = fit_surrogate(kind, COORDS[seen], yn)
            cand = s.unvisited
            with torch.no_grad():
                score = model(torch.tensor(COORDS[cand], dtype=torch.float32)).numpy().ravel()
            s.select(cand[int(score.argmax())])
    return arm

print('brazos surrogate listos: kan, mlp')"""),

    code("""# 4) La comparacion. Mismo presupuesto, misma cinta, mismos contextos.
# Se reporta AUC de regret (coste de busqueda) y % del techo (lo que pidio Garrido).
def evaluate(build_arm, label):
    aucs, finals = [], []
    for r, seed in enumerate(seeds):
        rng = np.random.default_rng(90_000 + r)
        fn = build_arm()
        a, f = [], []
        for ctx in contexts:
            s = Surface(surface[(ctx, seed)]); fn(s, rng, BUDGET)
            curve = s.regret_curve(); denom = BUDGET*abs(s.best) or 1.0
            a.append(float(np.sum(curve))/denom); f.append(curve[-1]/(abs(s.best) or 1.0))
        aucs.append(np.mean(a)); finals.append(np.mean(f))
    print(f'  {label:<22} AUC {np.mean(aucs):.5f}   % del techo {100*(1-np.mean(finals)):6.2f}%')
    return np.array(aucs)

retained = {'rho': np.zeros(COORDS.shape[1]+1)}
res = {}
res['neuron_memory (5 par.)'] = evaluate(lambda: make_neuron_arm(retained), 'neuron_memory')
res['surrogate_KAN']          = evaluate(lambda: make_surrogate_arm('kan'), 'surrogate_KAN')
res['surrogate_MLP']          = evaluate(lambda: make_surrogate_arm('mlp'), 'surrogate_MLP')
res['gp_ei']                  = evaluate(lambda: arm_gp_ei, 'gp_ei')
res['ucb1']                   = evaluate(lambda: arm_ucb1, 'ucb1')
res['ofat (tesis)']           = evaluate(lambda: arm_ofat, 'ofat')
res['random']                 = evaluate(lambda: arm_random, 'random')"""),

    code("""# 5) Contraste pareado por semilla contra la neurona de 5 parametros
base = res['neuron_memory (5 par.)']
rng = np.random.default_rng(20260807)
print('  Delta vs neuron_memory (negativo = MEJOR que la neurona)')
for k, v in res.items():
    if k.startswith('neuron'): continue
    d = v - base
    draws = [float(np.mean(d[rng.integers(0,len(d),len(d))])) for _ in range(5000)]
    lo, hi = np.percentile(draws,2.5), np.percentile(draws,97.5)
    tag = 'GANA' if hi < 0 else ('pierde' if lo > 0 else 'empata')
    print(f'    {k:<24} {d.mean():+.5f}  [{lo:+.5f} . {hi:+.5f}]  {tag}')"""),

    md("""## Cómo se lee, fijado antes de correr

* **`surrogate_KAN` con IC95 enteramente por debajo de cero** → **la ventaja de precisión SÍ se
  convierte en mejor búsqueda.** Sería el primer positivo neural del proyecto, y se preregistra
  confirmación en bloque virgen antes de reclamar nada.
* **IC cruza cero** → **empate**, que es lo que ya sabemos y sigue siendo publicable: una KAN de 532
  parámetros iguala a una neurona de 5 en la tarea donde la KAN es más precisa. Eso acota la
  contribución de la arquitectura.
* **IC enteramente por encima de cero** → la KAN busca **peor** pese a ajustar mejor, y ése también
  es un resultado: precisión del surrogate ≠ calidad de búsqueda.

## Lo que ya sabemos de un smoke, para que no te sorprenda

Corrido aquí con 1 semilla y 2 contextos, presupuesto 24:

| brazo | AUC | % del techo |
|---|---:|---:|
| `surrogate_KAN` | **0,03835** | 100,00 % |
| `surrogate_MLP` | **0,03835** | 100,00 % |
| `neuron_memory` | 0,04272 | 100,00 % |
| `ofat` | 0,11561 | 100,00 % |

Dos cosas: **los surrogates salen por delante de la neurona** en esa rebanada —así que el cuaderno
no está amañado para perder—, y **KAN y MLP dan el MISMO AUC hasta el quinto decimal**. Con 8 puntos
iniciales y la misma semilla eligen las mismas configuraciones. **Necesitas las 12 semillas × 6
contextos** para que aparezca cualquier separación; con menos, un empate no significa nada.

**Compara siempre `surrogate_KAN` contra `surrogate_MLP` a parámetros igualados**, no contra la
neurona sola. Si cambias la anchura de la KAN, ajusta la del MLP para que los conteos queden a menos
del 10 %; si no, la comparación no vale y es la objeción que tú mismo pusiste.

> Todo esto es **desarrollo** sobre cintas ya quemadas. Un buen resultado selecciona un candidato
> para un experimento preregistrado; no es evidencia de paper por sí solo.
"""),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=Path("notebooks/scresia_david_kan_surrogate_search.ipynb"))
    args = ap.parse_args()
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"  {args.output}  ({len(CELLS)} celdas)")
    print(f"  Colab : https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{args.output}")
    print(f"  Kaggle: https://www.kaggle.com/kernels/welcome?src=https://github.com/{REPO}/blob/"
          f"{BRANCH}/{args.output}   (enciende Internet en el panel derecho)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
