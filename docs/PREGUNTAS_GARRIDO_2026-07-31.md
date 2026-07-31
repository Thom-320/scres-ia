# Preguntas para Garrido — quedan **dos**, y no son las que íbamos a mandar

**Evidencia:** `results/metric_audit/garrido_ledger_conventions/result.json` (sello
`f89041450a3f0065…`), sobre sus **47.780 filas** entregadas · **Runner:**
`scripts/audit_garrido_ledger_conventions.py`.

Antes de escribirle, probé si sus propios datos ya contestaban lo que le iba a preguntar. **Una
de las dos preguntas se contestó sola, la otra cambió de objeto**, y por el camino salió una
corrección a un párrafo nuestro.

---

## ❌ RETIRADA — «¿tu `RPj` satura por diseño?»

**Contestada con su ledger. No se le pregunta.** Cuatro hipótesis estructurales, todas
decidibles sobre sus columnas, y las cuatro resueltas:

| hipótesis | resultado |
|---|---|
| `DPj = CTj` en toda fila demorada | **42.814 de 42.814 — 100,00%** |
| `RPj ≤ DPj` siempre | **100,00%** |
| `APj > 0 ⇒ RPj = 0` (ramas exclusivas) | **128 de 128 — 100,00%** |
| ¿techo duro en `RPj`? | **No.** Máximo 7.116 h, y solo el **0,005%** de las filas a menos del 1% de él |

Como `DPj ≡ CTj` y `RPj ≤ DPj`, la diferencia `DPj − RPj` **es** el retardo entre la colocación
del pedido y el primer onset de riesgo: exactamente `RPj = OATj − primer R⁰`, el **Algoritmo 2
tal como está publicado (p. 69)**. No hay saturación no documentada que preguntarle.

> **Y corrige un párrafo nuestro.** Nuestra sección de validación decía que su `RPj` satura,
> «correlaciona 0,88 con el conteo de riesgos y solo 0,37 con el tiempo de ciclo, aplanándose
> cerca de 400 h». Medido sobre el ledger completo: **`corr(RPj, CTj) = 0,582` y
> `corr(RPj, n riesgos) = 0,347`** — **el orden es el contrario** — y **no hay techo**. Las
> cifras anteriores salieron de un subconjunto que nunca se nombró. Ya está corregido en
> `docs/MANUSCRIPT_MODEL_VALIDATION_SECTION_2026-07-31.md`.

Nuestros periodos de recuperación son largos porque **nuestros riesgos empiezan antes respecto
al ciclo**, no porque su modelo haga algo que no vemos.

---

## ⚠️ TRANSFORMADA — la convención de simultaneidad ya no es la pregunta

Íbamos a preguntarle: *«cuando un pedido se solicita en el instante `t` y hay eventos en `t`,
¿la foto de `Bt`/`Ut` los ve o no?»*. Eso mantiene toda la métrica v2 en «provisional».

**Medido: la convención es inmaterial en sus datos.** Reconstruí el ledger bajo las **cuatro**
convenciones de empate posibles y solo **2 filas de 47.780 (0,004%)** distinguen entre ellas.
Un empate depende únicamente de las marcas de tiempo, así que ese conteo vale **sea lo que sea**
`∑Bt`. Elegir una convención u otra no puede cambiar sus números.

**Pero la reconstrucción fracasó, y eso sí es una pregunta.** Ninguna de las cuatro reproduce su
columna `∑Bt`: la mejor acierta el **1,09%** de las filas. Así que `∑Bt` **no es** el número de
pedidos pendientes en el momento de la solicitud, que es lo que nuestro código asume. Lo que sí
sabemos de ella, medido en las 20 hojas:

* **su máximo es exactamente 60,0** — el cap de la cola de backorders de su §6.5.4;
* el **13,0%** de las filas está en el cap, y 41.553 por debajo;
* no crece monótonamente con `j`, así que tampoco es un acumulado simple.

### **Pregunta 1 (la que sí se le manda)**

> En las hojas `CFi`, la columna **`∑Bt`** está acotada exactamente en 60 y no coincide con el
> número de pedidos pendientes de entrega en el instante `OPTj` (reconstruido, coincide en el
> 1% de las filas). **¿Qué cuenta exactamente `∑Bt` en cada fila?** ¿Es el tamaño de la lista
> de backorders del Op9 en ese instante, con el tope de 60 de §6.5.4, o un acumulado por
> periodo `t` como sugiere la Tabla 6.25 — y en ese caso, cuál es el periodo?

Es una pregunta **mejor** que la original: no depende de una convención de empate que ya
sabemos irrelevante, y su respuesta cierra la definición de la única columna del ledger que no
hemos podido reconstruir.

---

## **Pregunta 2 (nueva, la levantó la medición)**

`DPj = CTj` en el 100% de sus filas demoradas, y su Fig. 6.6 lista **`DPj − RPj`** como uno de
los cuatro sub-indicadores. Pero su Eq. 5.3 lo pondera con `Re^min = 0`, así que ese
sub-indicador **es idénticamente cero** en toda la tesis.

> **¿`DPj − RPj` está previsto como término inerte —presente en la estructura pero con peso
> cero por construcción— o `Re^min = 0` es un valor provisional de la Fig. 5.6 que en el
> modelo operativo debería ser distinto?**

Precedente: usted ya nos aclaró una vez que el `Re = 0,5` de la Fig. 5.6 es **ilustrativo** y
que la ponderación operativa debe usar `Re = 1` (registrado en `supply_chain/config.py:856`).
La misma duda se aplica a `Re^min`.

Importa de forma concreta: con `Re^min = 0`, **la neurona de su Fig. 5 tiene tres de sus cuatro
dendritas muertas** en nuestra reproducción de sus 90 configuraciones — una por esto, otra
porque nuestra autotomía no dispara, y la tercera queda viva solo en R2r/R3.

---

## Lo que NO le preguntamos, y por qué

| iba a preguntarse | por qué se retira |
|---|---|
| ¿satura `RPj`? | contestado: no hay techo, y `DPj − RPj` es el Algoritmo 2 publicado |
| convención de simultaneidad `Bt`/`Ut` | inmaterial: **2 filas de 47.780** distinguen entre las cuatro convenciones |

**Nada de esto se contestó con una opinión: se contestó con sus 47.780 filas.** El principio que
deja la sesión es que una pregunta al autor solo se justifica cuando su propio material no puede
responderla — y la mitad de las nuestras sí podía.
