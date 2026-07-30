# Las dos preguntas a Garrido, respondidas desde la tesis

> **CORREGIDO 2026-07-30.** La decisión de §Pregunta 1 —«`LT` debe ser 54»— **es
> incorrecta y queda retirada**. La tesis fija el lead time en 48 h explícitamente en
> **§6.8.2, p.111**: *«the availability of finished products at this point allows troops to
> be supplied within a pre-set lead-time of 48 hours»*. Lo desalineado es nuestro pipeline,
> no la métrica. Lo que sí se sostiene de esa sección es la semántica de igualdad
> (`CTj = LTj`, sin caso `CTj < LTj`) y los dos defectos de implementación. El sucesor
> correcto es `PREREGISTRO_LEAD_TIME_2026-07-30.md`, que barre el **delay de cumplimiento**
> con `LT` fijo en 48 y un criterio de fidelidad multi-momento.

**Estado:** `DECISION_DOCUMENT_NOTHING_APPLIED`. Ninguna constante fue cambiada. Este
documento fija la decisión y su evidencia; la aplicación requiere el mismo preregistro que
la reparación de ReT.

Las dos preguntas que le íbamos a hacer a Garrido **están respondidas en la tesis de 2017**.
No hace falta esperarlo. La segunda incluso viene con una regla de decisión explícita que
no habíamos visto.

---

## Pregunta 1 — el piso de 54 h y la rama de autotomía

### La tesis no tiene el caso `CTj < LTj`

Tres lugares independientes, todos con **igualdad**, no con `≤`:

- **§5.5.1 (p.67):** *«a partial disruption occurs when fluctuations … do **not** interrupt
  the flow of supplies to end-users, i.e., the SC cycle time of the order j **is equal to**
  its lead time (CTj = LTj)»*
- **Algoritmo 1 (p.68):** `IF el impacto de al menos un Rσ ∈ Ω se manifiesta en [OPTj, OATj]`
  **`AND CTj = LTj`**`, THEN APj = ΣRσ − Σ(solapamientos)`
- **p.72:** *«The range of values of APj is 0 < APj ≤ LT, **whenever CTj = LTj**»*

Y el Algoritmo 2 (p.69) dispara `RPj` con **`CTj > LTj`**.

**CORREGIDO:** un borrador anterior decía que «la partición es binaria y exhaustiva» y que
«no existe más rápido que la promesa». Eso es sobrelectura. Los algoritmos **especifican**
`CT = LT` y `CT > LT`; simplemente **no definen** `CT < LT`. Silencio no es prueba de
imposibilidad física. Lo que sí queda establecido es la **discrepancia semántica**: la tesis
activa `APj` con igualdad y nuestro código con `<=`.

### Consecuencia: `LT` es el horario, no una aspiración

En el modelo de Garrido `LTj = ‾LT` es el **tiempo de entrega programado** de la cadena.
`CTj = LTj` significa *«llegó cuando estaba previsto, a pesar de que ocurrió un riesgo
durante su ciclo»* — que es exactamente lo que el efecto de autotomía describe.

Nuestro pipeline tiene un ciclo programado de **54 h**, y lo comparamos contra un
`LEAD_TIME_PROMISE = 48` que no es nuestro horario. Por eso la rama nunca dispara.

### Dos defectos concretos de implementación

1. **`LEAD_TIME_PROMISE = 48` está mal citado, pero el valor es correcto.** `config.py:118`
   lo atribuye a §6.3.4, que es «Demand for combat rations» y no define lead time alguno.
   **La fuente correcta es §6.8.2, p.111**, que declara un *pre-set lead-time* de 48 horas.
   **CORREGIDO:** un borrador anterior afirmaba que «el 48 solo es la cadencia ROP de
   Op7/Op8». Eso es **falso** — §6.8.2 lo enuncia directamente como lead time. Solo hay que
   arreglar la cita, no el número.
2. **`supply_chain.py:5810` usa `CTj <= LTj`** donde la tesis usa `CTj = LTj`.

### Lo que se recupera al alinear `LT` con nuestro propio horario

Medido sobre la postura incumbente, raíz 1.900.001, 52 semanas:

| familia | órdenes puntuadas | `CTj = 54` | `CTj ≤ 48` | autotomía hoy |
|---|---:|---:|---:|---:|
| R1r | 287 | **278 (96,9%)** | **0** | 0,00% |
| R2r | 258 | **180 (69,8%)** | **0** | 0,00% |

**CORREGIDO:** estos conteos **no** demuestran que 54 sea el horario de Garrido. Demuestran
que **nuestra implementación concentra las entregas ahí**, que es un hecho sobre nuestro
pipeline y no sobre su modelo. Con `LT = 48` fijo (§6.8.2), lo que muestran es cuán lejos
está nuestro delay ajustado del lead time que la tesis declara.

**Y la diferencia entre familias es la señal que la métrica debería estar capturando.**
Bajo riesgos frecuentes y leves solo el 3% de las órdenes llega tarde; bajo riesgos raros y
severos, el 30%. Hoy esa distinción es invisible porque todo se fuerza a la rama de
recuperación.

### Decisión

**`LT` debe ser el tiempo de ciclo programado del modelo, y en nuestro pipeline eso es
54 h.** No es una elección de calibración: es leer la semántica de la tesis, donde `LT` es
el horario y no una meta.

La decisión **no se aplica aquí**. Cambia toda cifra de ReT del proyecto, así que va por
preregistro, con el mismo tratamiento que la reparación de la cola: contrato congelado,
raíces nuevas, y la métrica histórica reportada al lado, nunca sustituida.

Queda por decidir en ese preregistro si el predicado pasa a igualdad estricta (`CTj = LTj`,
la letra de la tesis) o se mantiene `≤` con `LT = 54` (equivalente para el caso modal, más
tolerante a granularidad). Recomiendo igualdad estricta con una tolerancia declarada, y
medir ambas.

---

## Pregunta 2 — los costos de κ

### La tesis los excluye a propósito, y dice por qué

**§8.5.2, «From the non-inclusion of the cost factor» (p.147):**

> *«Cost factor was not considered in the analysis as a critical variable mainly for the
> reasons explained in Section 3.2 … "For military-SCs only, in conditions of war, **the
> cost of shortages of troops is always higher than the result of adding, holding,
> replenishing, or obsolescence inventory costs and/or labour cost**." … However … if the
> military-SCs operated in peace mode, their operating conditions would be quite similar to
> commercial-SCs, therefore further research should include the cost factor.»*

No es un vacío: es una exclusión razonada **con un orden explícito**.

### Eso es una regla de decisión, no una ausencia

La proposición establece una **restricción ordinal**: en guerra el costo de faltante supera
la *suma* de costos de inventario y trabajo.

**CORREGIDO:** un borrador escribía esto como `c_b >> c_i, c_p, c_h, c_l, c_u, c_o`,
presentándolo como derivado de la tesis. **No lo es.** La tesis no identifica nuestros siete
coeficientes ni justifica ese símbolo. La formulación defendible es: *usamos `c_b` como
**proxy ordinal declarado** del costo militar de faltante y exploramos multiplicadores
monótonos; no son precios ni coeficientes calibrados por la tesis.*

Y esto **refuerza el hallazgo que ya teníamos**: nuestro port de Cobb-Douglas coloca en
R2r una postura con 76% de fill y 16 órdenes perdidas cerca del frente, precisamente
porque `c = 1` trata el faltante como cualquier otro costo. La tesis dice que eso está
mal para modo guerra.

### Decisión

**Dos vectores declarados, ninguno inventado:**

1. **`c = 1` en los siete** — réplica de la asunción (6) del §3.1 del artículo de 2024 de
   Garrido, etiquetada como *replicación*, no como calibración del MFSC. Es lo que ya
   tenemos.
2. **Vector modo-guerra** — `c_b` dominante sobre los demás, derivado de la proposición de
   §8.5.2 de la tesis. La magnitud del dominio es nuestra y debe barrerse, no fijarse: un
   barrido sobre `c_b ∈ {2, 5, 10, 50}` con el resto en 1, reportando en qué punto (si
   alguno) el ranking se estabiliza.

Lo que **no** haremos es inventar precios con unidades y año. La tesis no los tiene, y
`§8.6.2` dice que integrar los cuatro sub-indicadores en un óptimo *«might be a good
starting point for a new promising research avenue»* — o sea, el panel es trabajo nuevo,
no reproducción.

---

## Un tercer hallazgo, no buscado

**§8.6.1 (p.148):** *«the SC lead-time could be an additional factor to include within the
analysis of buffer-related strategies»*.

Garrido nombra el lead time como factor pendiente en su propia agenda, y es exactamente la
constante de la pregunta 1. La decisión sobre `LT` no es solo una corrección de fidelidad:
es una de las líneas que él mismo dejó abiertas.

---

## Qué sigue

1. Preregistrar el cambio de `LT` — contrato congelado, raíces nuevas, métrica histórica
   como segunda columna. Es el único de los dos que mueve todas las cifras.
2. Añadir el vector modo-guerra al barrido económico de κ que ya existe
   (`results/cobb_douglas/economic_sensitivity_v1/`).
3. La carta a Garrido se acorta: las dos preguntas dejan de ser bloqueantes y pasan a ser
   validación. Conviene reescribirla para presentarle **nuestras decisiones y su evidencia
   en la tesis**, pidiéndole confirmación en vez de instrucción.
