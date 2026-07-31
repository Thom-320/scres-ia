# Result — the A/A'/F/F' factorial: HALTED, and both failures are informative

**Status:** `HALTED_FALSIFIER_FAILED`. Executes
`docs/PREREGISTRO_DELAY_FISICO_2026-07-31.md` on `supply_chain/arm_runner.py`. Artifact
`results/metric_audit/delay_physical_arms_v1/result.json`, sealed. Roots 2,700,001–12.
**No moment is reported**, per §6.

## 1. Falsifiers

| falsador | resultado |
|---|---|
| f1 bloque de regresión del brazo A | PASA (registra, no puertea — ver §4) |
| f2 `min(CTj)` de F en `[48,00; 48,20]` | **PASA** — 48,000 |
| f3 `CTj` deja de ser masa puntual (>500 distintos/corrida) | **FALLA** — 36 |
| f4 `APj` nunca excede `CTj` | **FALLA** — 1.716 violaciones en F′ |
| f5 A ≡ A′ bit-idénticos | **PASA** |
| f6 etiquetado de la banda del piso | PASA |
| f7 conjunto `epsilon`-estable | PASA |

## 2. f3 — la cadencia de flete NO produce la distribución

| brazo | `min` | distintos/corrida | modal | p25 | p50 |
|---|---:|---:|---:|---:|---:|
| Garrido | 48,007 | — | — | **75,00** | **101,45** |
| A constante | 54,000 | 36 | 60,7% | 54,00 | 54,00 |
| **F olas** | **48,000** | **36** | **60,7%** | 48,00 | 48,00 |

F **baja el piso exactamente donde la tesis lo predice** —48,000 contra su 48,0074, y ese
falsador se declaró antes de correr— pero **la distribución no aparece**: misma cuenta de
valores distintos, misma cuota modal. Solo desplazó la masa puntual de 54 a 48.

**La causa, medida:** la finalización de la orden **ya está sincronizada con las olas**. Se
dispara desde `_op12_deliver`, que corre *en* una ola, así que la espera a la siguiente
siempre vale cero y `transit` colapsa a `24 + 0 + 24 = 48` exacto.

Esto **refuta la hipótesis tal como la implementé**, y afina el diagnóstico: nuestras órdenes
ya viajan en las olas; la constante se estampaba *encima*. Quitarla deja el `LT` determinista
al desnudo, no una distribución. La dispersión de Garrido tiene que venir de otro sitio —
disponibilidad de stock, lotes de 2.400–2.600 raciones contra el tamaño del pedido, o cola —
y **no de la cadencia**, que era mi hipótesis y ahora está medida y descartada.

## 3. f4 — quitar el tope destapa un doble conteo, y eso confirma la lectura del Algoritmo 1

**1.716 órdenes en F′ tienen `APj > CTj`**, que es físicamente imposible: el periodo de
autonomía no puede exceder el ciclo que lo contiene. En A′ hay **cero**, porque bajo delay
constante la autotomía nunca dispara (f5).

La causa es exactamente el segundo defecto que el preregistro §2 declaró: `APj` se construye
desde `total_disruption_hours`, que se acumula con `+=` en **seis sitios** sin restar
solapamientos, mientras el Algoritmo 1 exige `APj = ΣRcr − Σ(R1r ∩ … ∩ Rc4)`.

**El tope `min(total, LTj)` estaba enmascarando ese doble conteo.** Por eso la lectura
correcta no es «quitar el tope»: es **restar los solapamientos primero**, y entonces el tope
sobra por construcción. Quitarlo sin lo otro produce un valor sin sentido físico.

Es un resultado, no un fallo: el brazo F′ existía para medir esto y lo midió.

## 4. Corrección sobre f1

El falsador 1 **registra en vez de puertear**, y hay que decirlo. Los artefactos del
2026-07-30 se produjeron con la base de año equivocada y la población mixta; bajo el
instrumento reparado los `d_k` **no son comparables** con ellos. Un bloque de regresión
cruzado necesita una corrida de referencia nueva bajo `arm_runner.py`, y hasta entonces esa
compuerta no puede cerrar. Declararla como «PASA» sin esta nota sería exactamente el tipo de
falsador vacío que este contrato existe para eliminar.

## 5. Qué sobrevive

* **f2 pasó, y era una predicción.** La cadencia de la tesis fija el piso en 48 h y el modelo
  lo produce sin ajustar nada. La aritmética `24 + 0 + 24 = 48 = LT` es correcta y ahora está
  medida en el DES, no solo en el papel.
* **f5 pasó, y se declaró por adelantado.** A y A′ salieron bit-idénticos porque bajo delay
  constante `CTj = 54 > LT = 48` siempre. Declararlo antes evitó leer un no-efecto como
  confirmación.
* **El artefacto está sellado pese al halt**, con `contract_sha256`, provenance y
  `self_sha256` — lo que la ruta de halt anterior no hacía.

## 6. Qué NO hago

**No lanzo el brazo I.** El contrato §3 lo condiciona a que F y F′ *fallen la aceptación*, y
aquí no llegaron a evaluarse: los falsadores pararon la corrida antes. Pasar al ajuste
iterativo ahora sería saltarme mi propia condición.

**No quito el tope de `APj`.** f4 muestra que sin restar solapamientos el resultado es
imposible. La resta de solapamientos es su propio trabajo y su propio contrato.

## 7. Lo que queda planteado

1. **Restar solapamientos en `total_disruption_hours`** (Algoritmo 1, p.68). Con eso el tope
   sobra y `APj` recupera sentido físico. Es la pieza mejor definida que queda.
2. **De dónde sale la dispersión de su `CTj`**, ya que no es la cadencia. Candidatos
   medibles: disponibilidad de stock, el lote de 2.400–2.600 raciones contra el tamaño del
   pedido, la cola de pendientes.
3. **Una referencia v4** con el denominador de ventana puntuada, sin la cual
   `scored_orders_per_year` no se puede puntuar (enmienda §2).
