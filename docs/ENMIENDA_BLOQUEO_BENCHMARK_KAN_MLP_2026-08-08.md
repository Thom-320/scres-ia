# Enmienda — el benchmark KAN–MLP queda bloqueado, y el scheduler ya soporta el eje que faltaba

## Parte A — `results/kan_mlp_r2_benchmark/result.json` se reetiqueta

Sello `02a674fe…`, commit `8f2bcc23`. **Se conserva y se reetiqueta:**

> **`BLOCKED_INSTRUMENT — LEAKED_DECISION_TIMING_AND_TEST_SELECTED_COMPARATOR`**

Un auditor externo listó ocho defectos. Los verifiqué contra el runner y **los ocho son ciertos**.
Los tres que anulan la comparación:

**1. Viaje en el tiempo.** El contexto del surrogate son las cuatro primeras semanas de backlog
tomadas del **calendario 0**, que ya tiene el buffer encendido en esas semanas. Luego se elige un
calendario completo —incluidas ellas— y se evalúa desde `t = 0`. La política usa información
posterior a decisiones que supuestamente debía tomar. **Mi `f4_surrogate_reads_no_outcome = True`
era falso**, y estaba hardcodeado, que es exactamente el defecto sobre el que este proyecto ya tiene
una memoria escrita.

**2. El comparador open-loop se eligió sobre el test.** `best_fixed = argmin(L_te.mean(axis=0))`
optimiza el calendario fijo sobre las mismas seis tapes con las que se evalúa. Debe elegirse sobre
entrenamiento.

**3. Los parámetros no estaban emparejados.** KAN 372/744/1488 contra MLP 257/513/1025 — **un 45 %
más**. Y mi `f1` toleraba `|K−M| ≤ 2·min(K,M)`, que admite hasta 3:1: un falsador que
prácticamente no puede detectar un mal emparejamiento.

Y cinco más, todos ciertos: el SESOI se implementó sobre la interacción **absoluta** cuando el
preregistro exige **relativa ≥ 5 %**; una sola semilla de optimizador, así que no separa
arquitectura de suerte de inicialización; la demanda fue `thesis_uniform` y **no**
`garrido_seasonal_v1`; no hubo TOST, ni memoria, ni HPO equivalente, ni sensibilidad riesgo por
riesgo; y **«ocho falsadores pasan» era inflado** — `f7` es `NOT_APPLICABLE`, y `f2`, `f3`, `f4` y
`f6` estaban esencialmente codificados a `True`, así que sólo `f1`, `f5` y `f8` computaban algo.

## Lo que sí queda, y con su alcance

> **`DEVELOPMENT_PLATEAU_OBSERVED_ON_SIX_REPLAY_TAPES`**

En esas seis tapes, MLP eligió el calendario 0, KAN eligió 17, 19 y 24, y **los cinco igualaron
exactamente el mínimo por tape**, igual que la regla causal. Los calendarios `{0, 17, 19, 24}`
pertenecen a una meseta óptima **en esas tapes**.

**Se retira la frase más amplia** —«todos los calendarios envolventes que cubren las semanas
tempranas son óptimos»—: el artefacto guarda el spread medio, no la matriz completa
`tape × 26 calendarios`, así que esa afirmación excede lo serializado. También se retiran
**«Q1 contestada»**, **«paridad de parámetros»**, **«ocho falsadores pasan»** y **«MLP domina por
parsimonia»**.

**Y se retira «la propia construcción de Garrido».** Su Fig. 5 recibe los drivers SCRES `d_i`
ponderados por `ρ_i` y produce la métrica SCRES. El runner recibe `codificación del calendario +
backlog temprano → L*`. Es una operacionalización razonable del puente ③→⑧ de su Fig. 2, pero es
**una construcción nueva**, no una reproducción literal. Y KAN y MLP pertenecen a la **misma
categoría** de reconocimiento de patrones, así que una comparación entre ellos no puede responder
su Q1 —qué *categoría* de IA imita mejor el aprendizaje— por muy limpia que fuera.

## Parte B — el scheduler ya soporta el cambio de familia

`R2_DISTRIBUTION_FAMILY_CHANGE` pasa de `NOT_IMPLEMENTED` a **`IMPLEMENTED`**.

`supply_chain.py` acepta `risk_occurrence_family_by_id` con `uniform` (fuente) · `exponential` ·
`lognormal`, aplicado en `_sample_uniform_risk_window`, que es el único punto de muestreo de
ocurrencia. Las alternativas son **procesos de renovación** y devuelven ventana igual a su propio
retardo, de modo que el tail queda en cero y el bucle extrae el siguiente inter-arribo.

**Moment-matched sobre el inter-arribo, y ahí cometí y corregí un error que importa.** Bajo
`thesis_window` el bucle espera `delay` y luego el resto de la ventana, así que ocurre exactamente
un evento por ventana de longitud `b`: **el inter-arribo medio es `b`, no `(a+b)/2`**, que es el
desplazamiento medio *dentro* de la ventana. Igualar el desplazamiento **duplicaba la tasa de
eventos**, de 10,4 a 19,3 por episodio, y habría confundido forma distributiva con frecuencia
media — el confundido exacto que este brazo existe para evitar.

Medido sobre 12 semillas tras la corrección:

| familia | eventos R2 / episodio | sd | rango |
|---|---:|---:|---|
| `uniform` (fuente) | 10,25 | **0,83** | 9–12 |
| `exponential` | 9,58 | **3,25** | 5–17 |
| `lognormal` | 9,33 | **2,53** | 6–15 |

**Misma frecuencia media, entre 3 y 4 veces la dispersión.** Eso es exactamente «más aleatorios y
complejos» sin ser «más frecuentes», que es lo que Garrido pidió.

R1 y R3 conservan su familia por defecto y sólo admiten encendido/apagado y escalado de parámetros,
como el PI aclaró.

## Estado

```text
R2_DISTRIBUTION_FAMILY_CHANGE             IMPLEMENTED  (uniform · exponential · lognormal)
GARRIDO_SEASONAL_DEMAND_IN_BENCHMARK      NOT_USED
R2_PER_EPISODE_RANDOMIZATION              NOT_IMPLEMENTED
MODERN_PER_RISK_ARCHITECTURE_SENSITIVITY  NOT_RUN
KAN_MLP_BENCHMARK                         BLOCKED_INSTRUMENT — sucesor v2 pendiente
```

El sucesor `run_kan_mlp_r2_benchmark_v2.py` debe corregir, antes de correr: elección ex ante o
prefijo común de cuatro semanas; open-loop elegido **sólo** sobre entrenamiento; parámetros
emparejados a **±5 %** con 10–20 semillas de optimizador; mismo presupuesto de HPO pero
hiperparámetros propios por arquitectura; interacción **relativa** con TOST a ±5 %; demanda
estacional en ambos brazos; y la matriz `L[entorno, tape, calendario]` serializada entera.
