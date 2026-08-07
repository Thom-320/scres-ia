# Decisión del PI — endpoint de resiliencia y apertura de Program L

**Fecha:** 2026-08-07 · **Autoridad:** PI, bajo la constante permanente *«no dependemos de Garrido;
donde la tesis no fija un hecho, lo decidimos, lo declaramos como nuestro supuesto y lo pagamos»*.

Dos decisiones. La primera fija un endpoint; la segunda abre un carril. Este documento declara
ambas, **mide qué compran y qué no**, y deja preregistrado lo que sigue.

---

## Decisión 1 — El endpoint primario es la resiliencia

> **Declaración:** la doctrina de esta línea de investigación evalúa por **resiliencia**. No se
> impone un piso vinculante por peor producto. Es **nuestro supuesto declarado**, no un hecho de la
> tesis ni de Garrido.

### Alcance: prospectivo, y sólo prospectivo

Rige la regla R4 de `GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_1.md`: **un guardarraíl no se
retira después de ver quién gana.** Esta decisión define el endpoint de campañas futuras. No
promociona ningún resultado pasado ni convierte un STOP en un PASS.

### Qué compra para Program O: menos de lo que parece, y la medición lo dice

Se verificó `results/program_o/fixed_clock_hobs_corrective_validation_v1/` (resultado
`3d3ff5b37510a993…`, contrato `8cb665df8fad6bdf…`, ejecutado en `7a05d448`).

**Cuatro de cinco compuertas pasan:**

```
primary_pass    true      placebo_pass  true      physical_pass  true
action_pass     true      guardrail_pass  FALSE
```

**El primario está confirmado en las tres celdas:**

| celda | Δ medio | LCB95 simultáneo | tapes favorables |
|---|---:|---:|---:|
| `rho75_share90` | +0,09852 | **+0,06595** | 44/48 |
| `rho90_share75` | +0,07347 | **+0,04303** | 42/48 |
| `rho90_share90` | +0,09974 | **+0,05860** | 46/48 |

27/27 placebos superados (mínimo LCB95 +0,00716). 1.451 replays físicos, **0 fallos**. Un solo
vector de recursos programados — los recursos son literalmente iguales. `unique_action_sequences`
36–48 por celda: la política varía de verdad.

**Lo único que falló son dos estimandos, ambos el mismo:**

```
rho75_share90::ret_visible_cvar10   estimate +0,03502   sim_lcb95 −0,00858
rho90_share75::ret_visible_cvar10   estimate +0,01954   sim_lcb95 −0,01551
all_other_guardrails_pass: true
```

Nótese: **ambos puntos estimados son positivos.** Y de forma decisiva para esta decisión:

```
lost_orders     −0,0        lost_quantity   −0,0
ret_full         0,0        quantity_ret_full  0,0
```

**La política no abandona nada.** Cero pedidos perdidos, cero cantidad perdida. La preocupación por
producto abandonado —que es la que motivó el piso— **no es lo que cerró Program O**.

### Y aun así, la decisión no rescata Program O. Medido:

La inferencia usa *studentized one-sided max-t* sobre **69 estimandos**, crítico **2,8358**.
Reconstruyendo el SE desde `(estimate − sim_lcb95)/crítico`:

| celda | estimate | SE | **t** | LCB simultáneo (k=69) | LCB marginal (z=1,645) | LCB con k=12 (≈2,50) |
|---|---:|---:|---:|---:|---:|---:|
| `rho75_share90` | 0,03502 | 0,01537 | **2,28** | −0,00858 | **+0,00973** | −0,00342 |
| `rho90_share75` | 0,01954 | 0,01236 | **1,58** | −0,01551 | **−0,00079** | −0,01136 |

**Aunque se eliminara la multiplicidad por completo**, `rho90_share75` sigue en −0,00079. El fallo no
es sólo penalización por 69 estimandos: la estimación de cola es **ruidosa** (t = 2,28 y 1,58).

> **Conclusión honesta: la decisión del PI no convierte el STOP de Program O en un PASS, y ninguna
> reducción de la familia de estimandos lo haría.** El contrato prohíbe además un segundo rescate
> (`second_rescue_forbidden: true`).

### Lo que sí cambia, y es real

El artefacto **ya declara** `classical_primary_ret_advantage_confirmed: true`. Eso no se modifica
aquí: se cita. Lo que cambia es **cómo se reporta el conjunto**.

| antes | después de esta decisión |
|---|---|
| «Program O CERRÓ en STOP» | «Program O **confirmó prospectivamente** su primario en las tres celdas (LCB95 +0,043…+0,066, 27/27 placebos, 1.451 replays sin fallo), y **falló un estimando de cola que esta línea declara no primario**» |

No es una promoción: es la misma evidencia, dejando de encabezar con un guardarraíl que ya no es el
objetivo. **El proyecto pasa de “ninguna instancia positiva” a “una confirmación prospectiva
positiva sobre el endpoint declarado”.** El fallo de CVaR se reporta como **limitación declarada**,
con su número, no como resultado terminal.

### Precio del supuesto, declarado

1. **Cualquier revisor que evalúe por seguridad de cola tiene un contraejemplo cuantificado**
   (−0,0086 y −0,0155 simultáneos). Se publica, no se esconde.
2. **`ret_excel` queda prohibido como endpoint bajo esta decisión.** Sin piso por producto y con una
   métrica que censura pedidos omitidos, abandonar sería política ganadora — está medido: la
   partición que maximiza `ret_excel` entrega **50 % de fill**; la que lo minimiza, **80 %**. Con el
   piso retirado, **la métrica debe ser la que no se puede engañar abandonando**: `full_ledger` /
   Cobb-Douglas. El piso y la métrica censurada no pueden caer los dos.
3. La decisión **no** autoriza semillas nuevas. El registro sigue en
   `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`.

---

## Decisión 2 — Program L se abre bajo supuestos declarados

> **Declaración:** asumimos **dos rutas reales, flota finita con retorno obligatorio, y aviso
> predespacho imperfecto**. Son **nuestros supuestos**, no hechos verificados por Garrido.
> `blocked_domain_fact` deja de ser estado terminal para este carril.

El carril estaba en `ROUTE_FAMILY_DOMAIN_BLOCKED` esperando la validación facial de Garrido
(`results/paper2_search/program_l_corrective_audit.json`). Bajo la constante permanente del PI, ese
bloqueo se levanta declarando los supuestos y pagándolos.

### Las tres relajaciones que se declaran, con su precio

Tomadas literalmente de `program_l_route_recourse_screen.json`:

| relajación | qué asume la tesis | precio |
|---|---|---|
| almacenamiento CSSU finito (2–8 días de cobertura) | ilimitado | el buffer deja de ser gratis; el óptimo puede ser interior |
| disrupciones de ruta persistentes ~3–5 días | R22 ≈ 24 h | alarga la ventana donde la ruta importa |
| vehículo único finito con retorno obligatorio | disponibilidad dada | crea la contención; **es el mecanismo** |

La tercera es la que hace el carril interesante: **es contención sobre un recurso escaso y no
fungible**, que es el único sitio donde este proyecto ha medido headroom real (`H_PI` 0,1515 con el
nulo fungible en exactamente 0).

### El estado real de la evidencia: no es un nulo, es un gradiente truncado

`results/paper2_search/program_l_full_des_gate.json`, FULL-DES, identidad flags-off probada contra
`ProgramEConvoyEnv`, 56 días, sin aprendiz:

| `n_r22` | `dur_h` | `H_PI` | `H_obs` | placebo | bate placebo |
|---:|---:|---:|---:|---:|---|
| 1 | 24 | −0,000493 | −0,00836 | −0,04961 | sí |
| 2 | 24 | −0,000337 | −0,00754 | −0,05311 | sí |
| 4 | 24 | +0,000415 | −0,00728 | −0,04571 | sí |
| 2 | 72 | +0,001703 | −0,00375 | −0,04830 | sí |
| 4 | 72 | +0,002215 | −0,00353 | −0,04575 | sí |
| 4 | 120 | +0,005390 | −0,00239 | −0,04805 | sí |
| **6** | **120** | +0,003885 | **+0,00316** | −0,04005 | sí |
| **8** | **72** | +0,003913 | **+0,00402** | −0,01567 | sí |

**Ambas cantidades crecen monótonamente con la intensidad de disrupción, y `H_obs` cruza a positivo
en las dos celdas más estresadas — que son el borde de la rejilla.** El gate no midió un techo: se
quedó sin estrés antes de que el efecto se estableciera.

Dos razones independientes por las que este negativo no es limpio:

1. **La rejilla está truncada** exactamente donde el signo cambia.
2. **La métrica es `ret_excel` canónico**, que bajo régimen de riesgo pierde monotonía y ordena la
   peor postura primero, por censura dependiente de la política (18,6 % vs 3,9 % de pedidos
   omitidos). Un negativo medido con un instrumento no monótono no es un negativo.

Aplica la instrucción permanente: **un negativo bajo la física vieja no es un negativo bajo la
física nueva** — y aquí cambiaron la física *y* la métrica.

### Lo que se corre, en orden

**L-0 · Extender la rejilla más allá del borde.** `n_r22 ∈ {6, 8, 10, 12}` × `dur_h ∈ {72, 120, 168}`,
mismo instrumento, sin tocar nada más. Responde: ¿el gradiente sigue subiendo o se aplana?
Es el experimento más barato del repositorio y decide si el carril existe.

**L-1 · Re-medir con métrica no censurada.** Las mismas ocho celdas del gate original bajo
`full_ledger` y bajo Cobb-Douglas. Falsador que puede fallar: si el gradiente desaparece al cambiar
de métrica, entonces era un artefacto de censura y el carril se cierra por esa vía.

**L-2 · Gate de headroom oracle-first, antes de cualquier aprendiz.** `H_regime` con LCB95, placebo
desinformado obligatorio, y el control decisivo: **con la flota hecha fungible, el headroom debe caer
a 0**. Si no cae, no estamos midiendo contención.

**L-3 · Frontera clásica completa antes de red alguna.** constante → umbral → política de creencia
interpretable. La escalera de esta semana existe precisamente porque sin suelo no se lee nada.

**Sólo si L-0…L-3 pasan** se preregistra un aprendiz. No antes.

### Por qué este carril y no otro

`program_l_route_recourse_env` es la **única decisión del repositorio que sólo existe si algo falla**:
`Discrete(3)` HOLD/ROUTE_1/ROUTE_2 con aviso predespacho imperfecto. Una constante no puede ser
competitiva por construcción, porque la ruta óptima depende de una contingencia observable. Es lo
contrario de todo lo que medimos esta semana, donde la constante llegaba a 96,57 de 100.

---

## Resumen de decisiones

```
ENDPOINT_PRIMARIO           resiliencia; sin piso vinculante por peor producto
ALCANCE                     PROSPECTIVO — no promociona nada pasado (R4)
PROGRAM_O                   NO rescatado; medido: falla incluso sin multiplicidad (−0,00079)
PROGRAM_O_REPORTE           primario CONFIRMADO se cita como titular; CVaR como limitación declarada
METRICA_OBLIGATORIA         full_ledger / Cobb-Douglas — ret_excel PROHIBIDO sin el piso
PROGRAM_L                   ABIERTO bajo tres supuestos declarados y pagados
PROGRAM_L_ORDEN             L-0 rejilla → L-1 métrica → L-2 headroom+nulo fungible → L-3 frontera clásica
SEMILLAS_NUEVAS             NO autorizadas; el inventario sigue incompleto
```

## Custodia

Documento datado; no se edita en sitio. Los supuestos declarados aquí son **nuestros** y así deben
aparecer en cualquier manuscrito: nunca presentados como si vinieran de la tesis.
