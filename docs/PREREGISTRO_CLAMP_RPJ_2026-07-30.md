# Preregistro — el clamp de RPj: `max(earliest, OPTj)` contra la ventana del Algoritmo 2

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Ninguna constante cambiada. Toda cifra congelada permanece como fue reportada.

---

## 1. Qué se cambia

`supply_chain.py:5933`, rama `elapsed`:

```python
eff_risk_start = max(earliest_risk_start, order.OPTj)
order.RPj = max(0.0, order.OATj - eff_risk_start)
```

`earliest_risk_start` incluye riesgos **ya corriendo** cuando la orden se colocó, y el
clamp reescribe su inicio a `OPTj`.

El Algoritmo 2 (p.69) condiciona la atribución:

> 2: **IF the impact of at least one Rcr ∈ Ω manifests within the interval [OPTj, OATj]**
> AND CTj > LTj,
> 3: THEN, RPj = (OATj – first-R⁰cr)

| brazo | R⁰ admisible | si no hay ninguno |
|---|---|---|
| **C** (statu quo) | el más temprano, recortado a `OPTj` | no ocurre: siempre hay |
| **W** (ventana) | el más temprano **estrictamente dentro** de `(OPTj, OATj]` | falla la línea 2: `RPj = 0` |

**Sub-decisión declarada ahora, no después:** en el brazo W, una orden tocada por riesgo
pero sin ningún inicio dentro de su ventana **no cumple la línea 2**, luego `RPj = 0` y cae
en la rama que el ledger ya asigna a ese caso. No se le inventa un `RPj` sustituto.

Ninguna constante se toca en ningún brazo. Esto es una condición lógica del algoritmo, no
una calibración.

## 2. La firma objetivo, medida antes de declarar nada

Sus nueve hojas R1r, 21.561 filas con `RPj > 0`:

* **`RPj > CTj` ocurre 0 veces.** Razón `RPj/CTj` máxima = 1,0000 exacta.
* razón p50 = 0,958; p75 = 0,990; p90 y más = 1,0000. **Para la mayoría de sus órdenes
  `RPj ≈ CTj`**, y en 16,9% es exactamente igual.
* pero **`RPj` satura**:

| `CTj` | n | `RPj` p50 | `RPj` p95 | `RPj` max |
|---|---:|---:|---:|---:|
| [48, 168) | 13.851 | 75,2 | 126,3 | 152,1 |
| [168, 500) | 2.145 | 243,1 | 420,7 | 487,3 |
| [500, 1.000) | 1.972 | 413,3 | 835,6 | 991,9 |
| [1.000, 5.000) | 3.289 | **404,3** | 504,7 | 1.156,1 |
| [5.000, ∞) | 304 | **400,2** | 476,1 | 556,5 |

`RPj` sigue a `CTj` hasta ~500 h y luego **deja de crecer**, estabilizándose cerca de 400.

## 3. Predicción, en la MISMA escala que la regla de aceptación

Este es el defecto que cometí en el preregistro anterior —declaré la predicción sobre el
momento crudo y la regla sobre `d_k`—, así que aquí todo va en `d_k`.

1. **W mejora `d_k(rpj_p95)` en R1r.** Dirección declarada, magnitud no.
2. **W probablemente NO lo cierra**, y lo digo antes de medir. La §2 muestra que su `RPj`
   satura para `CTj` grande; la lectura de ventana predice `RPj ≈ CTj` también para órdenes
   largas, porque en un régimen denso en riesgos el primer inicio dentro de la ventana llega
   pronto. **Espero mejora parcial y un residuo de saturación sin explicar.** Si W cerrara
   el momento por completo, eso sería sorprendente y exigiría auditoría, no celebración.
3. **Riesgo declarado sobre el momento protegido, y es adverso.** `ReT = 0,5/RPj` en la rama
   de recuperación, así que **reducir `RPj` sube `ret_mean`**. Hoy `ret_mean` en R1r es
   0,007 contra una referencia de 0,006 — ya está por encima. **W lo empuja en la dirección
   equivocada.** Esta es la tensión real del cambio y la razón por la que `ret_mean`
   conserva veto.

## 4. Falsadores del instrumento

1. **Regresión del brazo C.** Corrido sobre las raíces 2.300.001–12, el brazo C debe
   reproducir `rpj_p95 = 2405,5` y `ret_mean = 0,007` del artefacto ya congelado
   `results/metric_audit/procurement_delay_reading_v1/result.json`. Si no, el instrumento
   está mal y no se reporta nada.
2. **`RPj ≤ CTj` en ambos brazos, sin excepción.** Sus datos lo cumplen 21.561/21.561.
3. **En el brazo W, toda orden con `RPj > 0` debe tener al menos un inicio de riesgo
   estrictamente dentro de `(OPTj, OATj]`.** Verificable orden por orden.

## 5. Criterio de aceptación

Dominancia sobre los seis momentos, referencia `fidelity_reference_v3`, `EPSILON = 0,5`, y
**las dos familias son co-objetivo** — a diferencia del test anterior, el clamp afecta a R1r
y a R2r por igual, así que aquí no hay familia de control.

**W se adopta si y solo si:**

* `d_k(rpj_p95)` **mejora en R1r**; **y**
* `d_k(ret_mean)` **no empeora más de `EPSILON` en NINGUNA de las dos familias**; **y**
* ningún otro momento empeora más allá de `EPSILON` en ninguna familia; **y**
* los tres falsadores de §4 pasan.

**Si W mejora `rpj_p95` y degrada `ret_mean`, no se adopta**, y el intercambio se reporta
como medido. Dado §3.3 ese es el desenlace más probable, y es un resultado publicable: diría
que la atribución literal del Algoritmo 2 y la magnitud de ReT de Garrido son mutuamente
inconsistentes en nuestro modelo, que es exactamente el tipo de hallazgo que este proyecto
debe reportar.

**Prohibido** elegir el brazo por el `H_PI` que produzca, por el signo de cualquier contraste
MPC-contra-estático, por que una familia cruce un umbral de servicio, o por que el resultado
sea publicable.

## 6. Corrección: la autotomía NO es un test de esto

En `docs/RECURRENCIA_REFUTADA_Y_EL_CLAMP_2026-07-30.md` escribí que `autotomy_share = 0`
quedaría como «test independiente y gratis» del clamp, porque `RPj = CTj > LTj` hace
inalcanzable la rama `CTj <= LTj`. **Está mal, y lo mido:**

    LT = 48    delay de cumplimiento = 54,0
    nuestro CTj: min = 54,00   p1 = 54,00   p50 = 54,00
    órdenes con CTj <= 48:  0 / 416

Con `GARRIDO_FULFILLMENT_DELAY_HOURS = 54 > LT = 48`, **ninguna orden puede calificar jamás**
para autotomía. Lo fija esa constante, no el clamp, y W no la moverá. `autotomy_share`
**queda fuera** del criterio de aceptación; si se mueve, se reporta como anomalía a
investigar, no como confirmación.

Eso convierte `autotomy_share = 0,000` (`d_k` 11,2 en R1r, 4,6 en R2r) en un **defecto
separado y ya localizado**: la constante de 54 h. Es su propio preregistro y no este.

## 7. Declarado por adelantado

| ítem | valor |
|---|---|
| constantes barridas | **ninguna** |
| brazos | C (clamp) contra W (ventana del Algoritmo 2) |
| sub-decisión | sin inicio en ventana → `RPj = 0` (§1) |
| raíces | **2.400.001–2.400.012**, disjuntas de todo bloque previo |
| raíces de regresión | 2.300.001–12, solo para el falsador 1 |
| familias | R1r y R2r, **ambas objetivo**, sin control |
| configuración | `S = 1`, buffers 0, nivel «+» de la Tabla 6.12 |
| modo RPj | `elapsed`; `procurement_delay_accumulation = "serial"` (el default) |
| referencia | `fidelity_reference_v3` (sha `31ecf9f9dae8058a`) |
| criterio | dominancia seis-momentos, `EPSILON = 0,5`, `ret_mean` con veto |
| predicción | §3, en `d_k` |

## 8. Alcance

**Nada se reetiqueta.** Program Q, la confirmación H2/H3, el buffer gate, las 90
configuraciones y la frontera conjunta conservan sus cifras. Si W se adopta, **abre un cuerpo
de resultados nuevo** y ambos se reportan con su atribución declarada.

**Fuera de alcance:** la saturación de §2 (mecanismo no identificado; W no se propone como su
explicación), la constante de 54 h y la autotomía (§6), y `ret_above_one_share` (3,9 / 4,0).

## 9. Firma

Requiere aprobación del PI antes de ejecutar.

La decisión que no me corresponde: si la lectura literal del Algoritmo 2 debe adoptarse aun
cuando degrade `ret_mean`. Mi lectura, y es solo eso: el clamp **contradice** la línea 2 del
algoritmo, así que es un defecto con independencia de qué momento mejore — pero adoptar un
cambio que empeora el endpoint del manuscrito es una decisión de proyecto, no de código, y
por eso la regla de §5 le da veto a `ret_mean` en lugar de resolverlo yo.
