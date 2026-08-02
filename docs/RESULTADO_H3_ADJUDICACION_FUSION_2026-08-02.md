# Resultado — **H3′ SOSTENIDA** con n = 120, y por qué esto sí es la confirmación preregistrada

**Artefacto:** `results/garrido_h3_merge_adjudication/result.json` (sello `1ac02efa1618e5a9…`) ·
contrato `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md` · runner
`scripts/adjudicate_h3_merge_v1.py` · **ninguna semilla nueva**.

## 1. El resultado

**Estimando de H3′, tal como fue reescrito:** *varianza del **coste de búsqueda entre contextos**,
por réplica* — no la media. Un aprendiz con memoria debería hacer el coste de búsqueda **más
uniforme** entre contextos de riesgo.

| contraste | media | IC95 | n |
|---|---:|---|---:|
| **memoria vs reinicio** | **+9,3144** | **[+2,3491 · +16,3474]** | 120 |
| memoria vs OFAT (la tesis) | +16,2203 | [+9,6107 · +22,7405] | 120 |

| estrategia | varianza media del coste de búsqueda |
|---|---:|
| **`neuron_memory`** | **44,26** |
| `neuron_reset` | 53,58 |
| `ofat` (el diseño de la tesis) | 60,48 |
| `random` | 67,50 |

**`LCB95 = +2,3491 > 0` → `H3_PRIME_SUSTAINED_AT_N120`.** La regla de lectura estaba fijada de
antemano y no se ha movido.

**Los cuatro falsadores de fusión PASAN**: semillas disjuntas (90 + 30), diseño idéntico
(`budget`, `factors`, `contexts`, `metric`, `n_configurations`), fuente idéntica (los siete hashes
de módulo, el del entry script y el del contrato), y `n = 120` exactamente el contratado.

**Diagnóstico por rebanada, que no es la adjudicación:** local +10,27 [+2,36 · +18,19] y VPS +6,46
[−7,31 · +20,33]. **La rebanada de 30 sola no alcanza** — que es precisamente por qué el contrato
fijó 120 y por qué la fusión era necesaria en vez de opcional.

## 2. Por qué esto SÍ es la confirmación, y me corrijo

Vengo diciendo que esto era «evidencia de desarrollo sobre semillas quemadas». **Era
sobre-conservador, y el propio contrato lo desmiente.** Su texto dice:

> *«Se abre un bloque **nuevo y virgen** de 120 réplicas con semillas `6 000 001…6 000 120`,
> disjuntas de todas las anteriores.»*

**El bloque se abrió UNA vez, y se abrió PARA este contrato.** Lo que falló no fue la ciencia sino
la etiqueta: el runner selló contra su ruta de contrato **por defecto**
(`PREREGISTRO_META_APRENDIZ`) en vez de la de H3′. Las re-ejecuciones reproducen los originales
**al último decimal en los dos lados (14/14)**, con el contrato correcto y manifiesto de módulos.

> Eso es un **re-sellado**, no una re-tirada. No hubo peeking —las 12 réplicas exploratorias
> quedan aparte y declaradas—, la regla de lectura nunca se tocó, y `f5`, el falsador que valida
> el arreglo de la fuga de julio, pasa en ambas.

El contrato también especifica la fusión: *«se fusionan por concatenación de réplicas, que es
válido porque cada réplica es independiente y lleva su propia semilla CRN»*. El adjudicador hace
exactamente eso.

## 3. Un defecto de puntería que el contrato me obligó a cazar

Estuve a punto de adjudicar sobre `reset − memoria` en **corridas medias** —el efecto Alzheimer,
+7,27 y +7,61—. **Eso habría resuelto una hipótesis distinta de la preregistrada.** H3′ es la
**varianza entre contextos**, y la función que la calcula ya existía en
`run_h1_h3_v2.py:191`; se levanta verbatim en vez de reinventarla.

Las dos cantidades son reales y ambas favorecen a la memoria, pero **no son intercambiables**, y
el manuscrito debe usar cada una para su hipótesis.

## 4. Qué hipótesis del borrador v.0 cubre esto

| hipótesis del borrador | qué la mide | estado |
|---|---|---|
| **H3 — Volatility Reduction** *«learning-enabled models reduce performance variance across heterogeneous disruption intensities»* | **exactamente este estimando**: varianza del coste de búsqueda entre seis contextos de riesgo | **SOSTENIDA, n = 120, LCB95 +2,35** |
| **H4 — Path Dependency** *«resilience at t is positively influenced by accumulated learning»* | efecto Alzheimer, `reset − memoria` | **+7,27** [6,75 · 7,78] (local) y **+7,61** [6,61 · 8,66] (VPS), sellados |
| **H2 — Adaptation / learning curve** | curvas de regret por contexto (`per_context.regret_curve`) | **datos presentes, sin adjudicar** |
| **H1 — Learning Effect (recovery time)** | — | **no evaluable**: TTR censurado; se sustituyó por `service_loss_auc` |

**`L_{t−1}` como variable de estado endógena** —la contribución formal que el borrador declara—
es precisamente `ρ` sobreviviendo la frontera de contexto, y **su precio está medido**.

## 5. Lo que sigue sin estar establecido

* La **identidad de fuente del snapshot original del VPS** no se demostró y ya no es
  reconstruible; lo que se demostró es identidad entre las dos re-ejecuciones.
* `f6 = DECLARED_REPLAY` en ambas rebanadas: es correcto, porque **re-ejecutan su propio bloque**.
  No es una segunda confirmación independiente y no debe presentarse como tal.
* **La fusión no autoriza nada sobre prima neural.** H3′ compara **memoria contra reinicio del
  mismo aprendiz**; no dice que una red gane a un control estructurado.
