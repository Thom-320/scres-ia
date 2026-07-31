# Preregistro — el turno de 8 h y la espera por capacidad

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Se ejecuta con `supply_chain/arm_runner.py`. Enmienda vigente:
`contracts/paper_b_v2_amendment_2026-07-31.json`.

Sucede a `docs/DISPERSION_CTJ_RESUELTA_2026-07-31.md` (`f1dfd2f`), que descompuso su `CTj`
sobre 21.667 filas.

---

## 1. Qué se añade, y por qué no hay nada que ajustar

    CTj = 48  +  k · 24  +  δ

| término | qué es | fuente | ¿lo tenemos? |
|---|---|---|---|
| **48** | `LT` = Op10 + Op11 + Op12 = 24 + 0 + 24 | tesis §6.3, §6.8.2 | **sí** |
| **k · 24** | días esperados, un flete diario | «daily freight rate (ROP = 24 hours)», §6.3 | **no** — `k ≡ 0` |
| **δ** | posición dentro del turno de **8 h** | `HOURS_PER_SHIFT = 8`, `S = 1` | **no** — `δ ≡ 0` |

**Ningún término tiene parámetro libre.** 48 sale de los PT de la tesis, 24 de su ROP
declarado, 8 de `HOURS_PER_SHIFT` con `S = 1`. Las tres constantes ya están en `config.py`;
lo que falta es aplicarlas a la pierna de cumplimiento, que hoy corre 24/7 y sirve toda orden
en la primera ola.

Esto **no es una calibración**. Es física declarada que el modelo no ejecuta.

## 2. Brazos — factorial 2×2, cada celda con firma distinta

| | capacidad OFF | capacidad ON |
|---|---|---|
| **turno OFF** | **A** (statu quo) | **C** |
| **turno ON** | **S** | **SC** |

La ventaja del diseño: **cada brazo tiene una firma predicha distinta y verificable por
separado**, así que un fallo señala qué término falló, no «algo falló».

| brazo | firma predicha |
|---|---|
| A | `CTj ≡ 48` — masa puntual (lo que ya medimos) |
| **S** | `CTj ∈ [48, 56]`, `δ ~ U(0,8)`, **sin bandas** (`k ≡ 0`) |
| **C** | picos discretos en `48 + k·24`, **sin dispersión intra-banda** (`δ ≡ 0`) |
| **SC** | la estructura completa: bandas con huecos vacíos **y** `δ ~ U(0,8)` |

## 3. Predicciones, declaradas con número antes de correr

**En forma** (más fuertes que cualquier momento, y ninguna usa un parámetro libre):

1. **En S y SC, `δ` es `U(0,8)`.** Criterio: los cuantiles p25/p50/p75 dentro de **±0,25 h**
   de 2,0 / 4,0 / 6,0. *Puede fallar:* nada garantiza que la ventana de turno produzca una
   uniforme; si el servicio se concentra al abrir el turno, `δ` saldrá sesgado a 0.
2. **En C y SC aparecen bandas en `48 + k·24` con huecos vacíos.** Criterio: `[60,72)` y
   `[84,96)` contienen **< 1%** de las órdenes cada uno, y `[72,84)` contiene **> 10%**.
   *Puede fallar:* si la cola no se satura, todo queda en `k = 0` y no hay bandas.
3. **En SC, `p25` y `p50` del `CTj` reconstruido caen a **±10%** de 75,00 y 101,45.**
   Simulado con la `k` observada, el modelo da 74,96 y 101,80. *Puede fallar:* la `k` que
   produzca nuestra cola no tiene por qué ser la suya.

**En `d_k`**, la misma escala que la regla de aceptación:

4. **`ret_mean`: sin dirección declarada.** Alargar `CTj` sube `RPj` y por tanto baja
   `0,5/RPj`; pero también cambia la mezcla de ramas. No lo sé y no voy a fingirlo.
   Conserva veto.
5. **`rpj_mean` y `rpj_p95` mejoran en SC**, porque `RPj ≈ CTj` y nuestro `CTj` pasa de 48 a
   una mediana de ~101, que es la suya.

**Predicción diagnóstica separada, y no es criterio de aceptación:** bajo el predicado de
banda con `tol = 0,05`, si `δ ~ U(0,8)` entonces `autotomy_share = 0,05/8 = **0,625%**`
contra su **0,443%** observado. Se reporta; **no** se puntúa, porque cambiar el predicado es
una decisión de otro contrato.

## 4. Falsadores — cada uno con su modo de fallo

1. **A reproduce el bloque de referencia bajo `arm_runner.py`** en los seis momentos, valores
   sin redondear. *Requiere una corrida base nueva*: los artefactos del 2026-07-30 usan la
   base de año vieja y la población mixta y **no son comparables** — el falsador 1 del
   contrato anterior solo registraba por esto, y aquí se cierra de verdad.
2. **`min(CTj)` en todos los brazos ≥ 48,00 y ninguna orden con `CTj < LT`.**
3. **En S, `k ≡ 0`** (`CTj ≤ 56` para toda orden no bloqueada) — aísla el turno.
   *Puede fallar:* si la implementación del turno también retrasa olas.
4. **En C, `δ ≡ 0`** (todo `CTj` a menos de 0,01 h de `48 + k·24`) — aísla la capacidad.
   *Puede fallar:* si la espera por capacidad introduce fracciones.
5. **El turno no toca aguas arriba:** `risk_events` bit-idéntico entre A y S. *Puede fallar:*
   si el calendario de turno se aplica a operaciones que no son la pierna de cumplimiento.
6. **`CTj` con más de 500 valores distintos por corrida en SC.** *Puede fallar:* es el
   falsador que tumbó el factorial anterior, con 36.
7. **La prueba 96/98** sobre la banda del piso.
8. **`epsilon` barrido**; conjunto que se mueve con `epsilon` se reporta inestable.

## 5. Criterio de aceptación

**Dominancia sobre los seis momentos**, `EPSILON = 0,5` barrido, ambas familias, referencia
`fidelity_reference_v3`. **La salida es el conjunto no dominado, nunca un ganador**, y
`sum_dk` **no puede rankear**.

Un brazo entra en el conjunto adoptable si y solo si:

* **las tres predicciones de forma (§3.1-3.3) se cumplen** para ese brazo; **y**
* `d_k(ret_mean)` **no empeora más de `EPSILON` en ninguna familia**; **y**
* ningún otro momento puntuado empeora más allá de `EPSILON`; **y**
* **los ocho falsadores pasan**; **y**
* el conjunto es `epsilon`-estable.

**Momento excluido de la puntuación:** `scored_orders_per_year`, hasta que exista una
referencia v4 con denominador de ventana puntuada (enmienda §2). Se reporta, no se puntúa.

**Si las firmas de forma se cumplen pero los momentos no mejoran**, eso se reporta como tal:
sería evidencia de que reproducimos su mecánica de tiempos sin reproducir su `ReT`, que es un
resultado, no un fracaso.

**Prohibido** elegir brazo por el `H_PI`, por el signo de un contraste MPC-contra-estático,
por que una familia cruce un umbral de servicio, o por que el resultado sea publicable.

## 6. Declarado por adelantado

| ítem | valor |
|---|---|
| constantes **no** tocadas | `LEAD_TIME_PROMISE = 48`, `HOURS_PER_SHIFT = 8`, ROP = 24 |
| parámetros libres | **ninguno, en ningún brazo** |
| brazos | A, S, C, SC |
| predicado de autotomía | `le` (el default) en los cuatro; la banda solo como diagnóstico §3 |
| raíces | **2.800.001–2.800.012**, disjuntas de todo bloque previo |
| familias | R1r y R2r, ambas objetivo |
| instrumento | `supply_chain/arm_runner.py` (obligatorio) |
| criterio | conjunto no dominado + las tres firmas de forma |
| predicción | §3, con §3.4 **sin dirección** |

## 7. Alcance

**Nada se reetiqueta.** Si un brazo se adopta, abre un cuerpo de resultados nuevo.

**Fuera de alcance:** el predicado de banda (medido, no adoptado), el tope de `APj` (la unión
lo vuelve redundante, `64b75ce`), el clamp de `RPj`, el multiplicador serial, y la referencia
v4.

## 8. Firma

Requiere aprobación del PI.

La decisión que no me corresponde: si aplicar el calendario de turno a la pierna de
cumplimiento es un cambio de modelo aceptable para el paper. Mi lectura: `HOURS_PER_SHIFT` y
el ROP diario **ya son parámetros declarados de la tesis que el modelo no ejecuta en esa
pierna**, así que esto es completar una implementación, no calibrar. Pero mueve el `CTj` de
toda orden, y eso es tuyo.
