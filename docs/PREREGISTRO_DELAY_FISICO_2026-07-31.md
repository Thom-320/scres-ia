# Preregistro — el delay de cumplimiento: cadencia física, tope de `APj`, y el iterativo como respaldo

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Sucede a `docs/RESULTADO_DELAY_DISTRIBUCION_2026-07-30.md`, que se detuvo en su falsador 4.

**Este contrato se ejecuta con `supply_chain/arm_runner.py`**, no con un runner copiado. Los
cuatro runners del 2026-07-30 se derivaron entre sí con `sed` y compartían las mismas cinco
violaciones del contrato maestro; el módulo compartido las corrige de una vez. Enmienda
vigente: `contracts/paper_b_v2_amendment_2026-07-31.json`.

---

## 1. Por qué el ajuste probablemente sobra: la tesis especifica la cadencia

§6.3 (pp. 85-86) define el último tramo:

| operación | PT | cadencia |
|---|---|---|
| Op9 SB | 24 h | «at a **daily freight rate** (ROP = 24 hours)» |
| Op10 SB→CSSU | 24 h | «daily freight rate (ROP = 24 hours)» |
| Op11 CSSU | **0** («nuisance factor») | «daily freight rate (ROP = 24 hours)» |
| Op12 CSSU→tropas | 24 h | «daily freight rate (ROP = 24 hours)» |

`24 + 0 + 24 = 48 = LT`, y §6.8.2 lo confirma: disponible en Op9, la tropa se abastece
«within a pre-set lead-time of 48 hours». **La dispersión de `CTj` es la espera a la siguiente
ola de flete en cada tramo. Cero parámetros libres.**

Predicción analítica (`U(0,24)` por tramo) contra lo observado:

| | p1 | p5 | p25 | p50 |
|---|---:|---:|---:|---:|
| 3 tramos, cadencia de la tesis | 57,41 | 64,09 | **75,54** | 84,00 |
| **Garrido observado** | 48,41 | 50,42 | **75,00** | 101,45 |

El p25 acierta sin ajustar nada. El extremo bajo no, y **eso es esperable**: las olas son
**compartidas** entre órdenes, no sorteos independientes, así que solo el DES lo reproduce.

Y el sim **ya corre esas olas** (`supply_chain.py:4229`, `:4270`) — pero mueven **raciones**.
El `OATj` de la **orden** se estampa con la constante plana en `:2569`, por **ambas** rutas de
cumplimiento: medido, `op9_linked` da el mismo piso 54,00 con 52,5% modal que
`legacy_theatre_stock`.

## 2. El segundo defecto, cruzado en el mismo diseño

Algoritmo 1 (p.68) línea 3: `APj = ΣRcr − Σ(R1r ∩ … ∩ Rc4)`. **Sin tope.**
`supply_chain.py:6009` aplica `min(total_disruption_hours, LTj)`.

Medido en sus 96 filas de autotomía: **12 (12,5%) exceden `LT = 48`**, hasta **48,0418**.
Nuestro tope las truncaría a 48,0 exacto. Además `total_disruption_hours` se acumula con `+=`
en seis sitios sin restar solapamientos, que el algoritmo exige explícitamente.

## 3. Brazos

| | tope `APj` ON (statu quo) | tope `APj` OFF (Algoritmo 1) |
|---|---|---|
| **delay constante 54** | **A** | **A′** |
| **delay físico (olas)** | **F** | **F′** |

**Brazo F — cero parámetros.** El pedido espera la siguiente ola real de flete en
Op10/Op11/Op12 más los PT de la tesis: registrar la última ola por tramo en los bucles
existentes y calcular `transit = Σ_tramos (espera_a_la_siguiente_ola + PT_tramo)` en
`_finalize_order_after_fulfillment_delay`. **Sin RNG**: determinista dado el estado del sim,
lo que hace de los cuantiles una predicción y no un ajuste.

**Brazo I — punto fijo acotado. SOLO se ejecuta si F y F′ fallan §6.**

- objetivo: `p25` y `p50` del `CTj` realizado **de órdenes no bloqueadas** contra 75,00 y
  101,45. *(El contrato anterior declaró este filtro y el código lo omitió; aquí es falsador.)*
- forma: **una sola**, declarada ahora — `48,0074 + Weibull(k, λ)` — no tres;
- iteración: `θ_{k+1} = θ_k · (objetivo / realizado_k)` sobre la escala `λ`, con `k` fijo;
- **tope duro de 6 iteraciones**, parada en error relativo < 5%;
- si no converge en 6, **se reporta el no convergido**; ampliar el tope está prohibido;
- `p1`, `p5`, `p95` **reservados**: no entran en la iteración y se comparan al final.

## 4. Predicción, en `d_k`, la misma escala que la regla

1. **F mejora `d_k(autotomy_share)` en ambas familias.** Es la razón de ser del cambio.
2. **`ret_mean`: sin dirección declarada, deliberadamente.** Bajar el piso sube `0,5/RPj`;
   alargar el cuerpo hacia el p50 de 101 lo baja. No sé cuál domina y no voy a fingir que sí.
   Conserva veto igual.
3. **El tope OFF sube `APj` solo en órdenes de autotomía**, así que su efecto sobre `ret_mean`
   debería ser pequeño y positivo en la rama `APj/LT`.
4. **Nada sobre `rpj_p95` ni sobre la saturación.** Cuatro mecanismos refutados el 2026-07-30;
   no propongo un quinto.

## 5. Falsadores — cada uno con un modo de fallo real

El defecto dominante del 2026-07-30 fue publicar falsadores que no podían fallar. Cada uno de
estos declara **por qué puede fallar**, y `arm_runner.run_falsifiers()` guarda su evidencia
junto al booleano para que la vacuidad sea visible en el artefacto.

1. **A reproduce el bloque congelado en los SEIS momentos**, contra los valores sin redondear
   (no `0,007`). *Puede fallar:* cualquier perturbación del default lo rompe.
2. **`min(CTj)` en F cae en `[48,00, 48,20]`.** *Puede fallar:* la cadencia lo **predice**;
   nada lo ajusta. Si F produce 54, la implementación no conectó las olas.
3. **`CTj` con más de 500 valores distintos POR CORRIDA**, no agregando raíces. *Puede
   fallar:* hoy son 40 en un episodio, con 65,9% en un solo valor.
4. **Con tope OFF, ningún `APj` excede `CTj`.** *Puede fallar:* es una cota física, no una
   identidad aritmética.
5. **A y A′ salen BIT-IDÉNTICOS.** Bajo delay constante `CTj = 54 > 48` siempre, la autotomía
   nunca dispara y `APj ≡ 0`, así que el tope no puede tener efecto. **Declarado ahora en vez
   de descubrirse después**, y el factor del tope solo es observable en `F` contra `F′`.
6. **La prueba 96/98 — la que nunca se corrió.** Con el piso y la banda de F, nuestras órdenes
   en `[48,0074, 48,06]` deben clasificarse como él clasifica las suyas: ≥ 90 de 98
   equivalentes. *Puede fallar:* el 2026-07-30 la tolerancia resultó irrelevante porque `CTj`
   era masa puntual; con una distribución real deja de serlo.
7. **`epsilon` barrido.** Si el conjunto no dominado se mueve con `epsilon`, se reporta
   **inestable** en vez de mostrarse. *El contrato maestro lo exige y ninguno de los cuatro
   runners lo implementó.*

## 6. Criterio de aceptación

**Dominancia sobre los seis momentos**, `EPSILON = 0,5` barrido, ambas familias, referencia
`fidelity_reference_v3`. **La salida es el conjunto no dominado, nunca un ganador**, y
`sum_dk` **no puede usarse para rankear** (contrato maestro: «do NOT collapse it with
weights»).

**Un brazo entra en el conjunto adoptable si y solo si:**

* `d_k(autotomy_share)` **mejora en ambas familias**; **y**
* `d_k(ret_mean)` **no empeora más de `EPSILON` en ninguna**; **y**
* ningún otro momento empeora más allá de `EPSILON` en ninguna familia; **y**
* **los siete falsadores de §5 pasan** — conjunto que las reglas del 2026-07-30 omitieron en
  código; **y**
* el conjunto es `epsilon`-estable.

**Si el conjunto no dominado es todo el factorial, la respuesta honesta es que no
discrimina**, y se reporta así.

**Excluido de la aceptación:** `scored_orders_per_year`, hasta que exista una referencia v4
con el denominador de ventana puntuada (enmienda §2). Se reporta, no se puntúa.

**Prohibido** elegir brazo por el `H_PI` que produzca, por el signo de un contraste
MPC-contra-estático, por que una familia cruce un umbral de servicio, o por que el resultado
sea publicable.

## 7. Declarado por adelantado

| ítem | valor |
|---|---|
| constante **no** tocada | `LEAD_TIME_PROMISE = 48` (§6.8.2 p.111) |
| instrumento | `supply_chain/arm_runner.py` (obligatorio) |
| brazos | A, A′, F, F′; **I solo si F y F′ fallan** |
| parámetros libres en F/F′ | **ninguno** |
| forma de I | `48,0074 + Weibull(k, λ)`, declarada antes de correr |
| raíces | **2.700.001–2.700.012**, disjuntas de todo bloque previo |
| raíces de regresión | 2.600.001–12, solo para el falsador 1 |
| familias | R1r y R2r, ambas objetivo |
| momento excluido | `scored_orders_per_year` (§6) |
| criterio | conjunto no dominado, `epsilon` barrido, `ret_mean` con veto |
| predicción | §4, en `d_k`, con §4.2 **sin dirección** |

## 8. Alcance

**Nada se reetiqueta.** Si un brazo se adopta, **abre un cuerpo de resultados nuevo**.

**Fuera de alcance:** la saturación de `RPj` (cuatro mecanismos refutados, ninguno propuesto),
el clamp (medido, no adoptado, `a0912bd`), el multiplicador serial (medido, no adoptado,
`8a6aa16`) y la referencia v4 (enmienda §2, su propio trabajo).

## 9. Firma

Requiere aprobación del PI.

La decisión que no me corresponde: si conectar el `OATj` de la orden a la cadencia de flete
que la tesis especifica es un cambio de modelo aceptable para el paper, o si la línea de
reproducción conserva la constante y la cadencia se abre en paralelo. Mi lectura: la
constante ya está documentada como fallida en tres frentes en `config.py`, y el tercero —ser
constante donde la fuente es distribución— no lo arregla ningún número. Pero es un cambio de
modelo, no de código.
