# Investigación del fix de RNG para el confound de régimen/riesgo — 2026-07-03

Sigue a `docs/TRACK_B_COUNTERFACTUAL_RNG_ENTANGLEMENT_FINDING_2026-07-03.md`. Ahí until confirmé
que el contrafactual por empalme es inválido. Aquí investigo qué haría falta para arreglarlo de
raíz en el simulador, y qué tan grande es el radio de impacto.

## Hallazgo 0 (nuevo, más grave de lo que pensaba): Track B ni siquiera usa el modo de streams ya validado

`MFSCSimulation.__init__` (supply_chain.py:250-264) ya soporta dos modos:

- `seed_stream_mode="single"` (default si no se pasa nada): `self.rng = np.random.default_rng(seed)`
  y luego `self.demand_rng = self.rng` / `self.risk_rng = self.rng` — **son literalmente el mismo
  objeto Generator.**
- `seed_stream_mode="split"` (equivalente a `strict_exogenous_crn=True`): en el working tree actual usa
  `np.random.SeedSequence(seed).spawn(4)` y produce streams independientes para `rng`/`demand_rng`/`risk_rng`/
  `regime_rng`.

`docs/GARRIDO_DES_FREEZE_STATUS_2026-06-26.md` confirma que la configuración congelada y validada
contra el gate de fidelidad (H2/H3) usa explícitamente `seed_stream_mode = split`.

Pero **`supply_chain/env_experimental_shifts.py:2475`** (el env que usa Track B) construye
`MFSCSimulation(...)` sin pasar `seed_stream_mode` ni `strict_exogenous_crn` en ningún punto del
archivo (verificado, cero ocurrencias). Es decir: **Track B corre en modo `"single"` — demanda,
riesgo y ruido general comparten un solo stream —, no en el modo `split` que el propio proyecto ya
validó y adoptó como estándar para la réplica de Garrido.**

## Hallazgo 1 (matiz importante): el regimen adaptativo ya tiene stream propio en el working tree, pero Track B no lo activa

En el working tree actual, `_adaptive_regime_controller` y `_update_adaptive_forecasts` ya leen
`self.regime_rng` (supply_chain.py:1740/1745/1760), no `self.rng`. Esto implementa la capa de fix
que se habia propuesto para el regimen.

Pero mientras Track B siga construyendo `MFSCSimulation(...)` sin pasar `seed_stream_mode="split"`
ni `strict_exogenous_crn=True`, `self.regime_rng = self.rng` por default (modo `"single"`). Es decir:
el stream dedicado existe en codigo, pero queda colapsado al mismo generador compartido para Track B.

## Por qué esto importa: `_adaptive_regime_controller` solo corre bajo `adaptive_benchmark_v2`

Buena noticia parcial: `_adaptive_regime_controller` solo se lanza
`if self.adaptive_benchmark_enabled` (supply_chain.py:850-851), y eso solo es `True` para los
niveles de riesgo `adaptive_benchmark*` (línea 464) — **no se ejecuta durante el gate de fidelidad
H2/H3**, que usa niveles de riesgo estáticos (`current`/`increased`/`severe`). Es decir: el
Hallazgo 1 es ortogonal al gate de fidelidad ya validado — arreglarlo no exige re-validar H2/H3.
El Hallazgo 0, en cambio, si se corrige, sí cambia el modo de streams que usa TODO Track B
(entrenamiento y evaluación), aunque ese modo específico (`split`) ya está validado como fiel para
la réplica base de Garrido.

## El fix completo (dos capas)

**Fix A — activar el modo ya validado para Track B.** Pasar `strict_exogenous_crn=True` (o
`seed_stream_mode="split"`) en la construcción de `MFSCSimulation` dentro de
`env_experimental_shifts.py`/`env.py`. Separa `risk_rng`/`demand_rng` del stream general
(`_pt`, R14). No requiere nueva validación de fidelidad porque el modo ya está validado — solo
faltaba conectarlo a Track B.

**Fix B — stream dedicado para régimen.** Ya aparece implementado en el working tree:
`MFSCSimulation.__init__` crea `regime_rng` cuando `seed_stream_mode="split"`, y el controlador/
forecast adaptativo ya lo usan. No obstante, esta capa es **inerte para Track B** mientras no se
active el modo split desde `env_experimental_shifts.py`.

Ambas capas juntas son necesarias para que el empalme dentro de episodio (`R_full - R_reset(w)`)
sea válido de raíz. En el working tree actual, la capa B ya esta codificada; falta la activacion de
la capa A para Track B. Sin Fix A, `rng`/`demand_rng`/`risk_rng`/`regime_rng` siguen siendo el mismo
objeto bajo el modo default `"single"`.

## Radio de impacto si se implementa

- Cambia la trayectoria de riesgo/demanda simulada para **cualquier seed**, aunque las
  distribuciones subyacentes no cambian (mismas tablas de riesgo, misma lógica) — solo cambia qué
  número aleatorio específico cae en qué sorteo. Las conclusiones cualitativas (PPO/Real-KAN
  ganan a estática y heurísticas) casi con certeza sobreviven, pero **no puede garantizarse sin
  volver a correr todo**: reentrenar y reevaluar Track A/B, y regenerar E1-E6 y el propio
  no-forecast confirm run para que el paquete de evidencia sea internamente consistente bajo la
  nueva configuración.
- El gate de fidelidad H2/H3 (réplica base de Garrido) no necesita re-validarse por la capa B (el
  controlador de régimen no corre ahí). Para la capa A tampoco, porque el modo `split` ya es el
  modo validado — de hecho, aplicar la capa A haría que Track B esté MÁS alineado con lo ya validado,
  no menos.
- Este es un cambio de alto impacto en código central ya validado. **No lo implementaría sin
  aprobación explícita del usuario**, dado que invalida la comparabilidad directa con todos los
  resultados de Track A/B generados hasta ahora (aunque no necesariamente los invalida
  *cualitativamente* — hay que volver a correr para saberlo con certeza).

## Validación empírica del fix (2026-07-03, tras implementarlo)

Implementé la capa A (`strict_exogenous_crn=True` en `env_experimental_shifts.py:2481`) y confirmé
que la capa B (`regime_rng`) ya estaba codificada. Repetí exactamente la prueba de divergencia que
había usado para encontrar el problema original (splice de 4 pasos pre-riesgo sobre `ppo_mlp`,
seed=1, episodio=1, usando el propio `audit_track_b_risk_event_counterfactual.py` de Codex):

```text
Antes del fix:  868/876 eventos compartidos entre full y reset ... en realidad 683/876 (78%)
Despues del fix: conteos por risk_id, full vs reset:
  R11: 130 vs 130   R13: 79 vs 79   R24: 38 vs 38
  R22: 7 vs 7        R12: 3 vs 3    R23: 2 vs 2   R21: 1 vs 1
  R14: 621 vs 622   (unica diferencia, 1 evento de 621)
```

**Los 8 riesgos discretos "verdaderos" (R11, R12, R13, R21, R22, R23, R24) quedan perfectamente
identicos entre `full` y `reset` tras el fix** — cero divergencia, con solo 4 de 104 pasos
sustituidos. Solo R14 muestra una diferencia mínima (1 evento de ~621, 0.16%), y eso es **esperado
y correcto**, no un defecto: R14 usa `self.rng.binomial(produced, p)` donde `produced` es la
producción real del día — su tasa depende legítimamente de cuánto se produce, que depende de las
acciones. Ya estaba documentado como "frecuente_tasa" y requiere tratamiento aparte; no es parte
del confound de régimen que este fix corrige.

**Conclusión de la validación: el fix funciona.** El contrafactual `R_full - R_reset(w)` ahora es
válido para R11/R12/R13/R21/R22/R23/R24 evaluados bajo el entorno corregido. Sigue sin ser
aplicable de forma limpia a R14 sin lógica adicional (sería necesario fijar `produced` o tratarlo
por separado).

## Coordinación con Codex (2026-07-03, mismo día)

Codex llegó al mismo diseño de forma independiente y ya lo implementó en el mismo working tree
(verificado con `diff` byte a byte entre mi copia local de `supply_chain.py`/
`env_experimental_shifts.py` y la copia en el VPS — idénticas). Codex ya lanzó el reentrenamiento
de PPO+MLP bajo la config corregida en el VPS (`tmux` session `track_b_fixed_rng_5seed`,
`outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03/`, 5 seeds x 60k). Para no
duplicar, lancé el complemento: Real-KAN bajo la misma config corregida
(`track_b_real_kan_fixed_rng_confirm_5seed_60k_2026-07-03`, misma escala/hiperparámetros que
`confirm_5seed_60k_h104`). Tuve que instalar `pykan`/`scikit-learn`/`sympy`/`tqdm`/`pyyaml` en el
`.venv` del VPS (no estaban, solo se habían usado para KAN localmente hasta ahora). Ambos jobs
corren en paralelo en el mismo VPS (6 vCPU, ~3 ocupados entre los dos). Watcher local:
`scripts/watch_track_b_fixed_rng_vps.sh` (rsync automático cuando terminen).

## Recomendación (actualizada — el usuario aprobó implementar)

1. **Hecho**: fix implementado y validado (arriba). Reentrenamiento de PPO+MLP (Codex) y Real-KAN
   (yo) en curso en el VPS bajo la config corregida, 5 seeds x 60k cada uno.
2. **Alcance deliberadamente acotado**: NO estoy re-entrenando todo Track A/B ni regenerando
   E1-E6/no-forecast — eso seguiría siendo una decisión aparte, mucho más cara, para una v2 del
   paper. El alcance actual es el mínimo necesario para tener un par PPO+MLP/Real-KAN
   autoconsistente bajo el entorno corregido, suficiente para re-habilitar el contrafactual
   `R_full - R_reset(w)` de forma válida sobre riesgos frecuentes reales.
3. Los resultados actuales del paper (Track A/B headline, E1-E6, no-forecast, el documento para
   Garrido) siguen siendo válidos como están — no se tocan ni se re-etiquetan por este cambio.
   Están generados bajo la config de RNG anterior, documentada como tal.
4. Cuando terminen ambos reentrenamientos: re-correr `audit_track_b_risk_event_counterfactual.py`
   (o una versión corregida) sobre los nuevos checkpoints para obtener, por fin, un delta ReT Excel
   causal válido para R11/R12/R13/R21/R22/R23/R24 (R14 sigue necesitando tratamiento aparte).
