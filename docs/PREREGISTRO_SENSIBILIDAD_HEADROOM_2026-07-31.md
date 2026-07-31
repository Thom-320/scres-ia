# Preregistro — mapa de headroom: ¿nodos o continuidad, y dónde?

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Se corre **una vez**.

## 1. La pregunta

Liberar el espacio a continuo no volvió el problema no lineal (el lineal **subió** a 0,9823).
Garrido pidió además **añadir nodos**. Antes de gastar en topología nueva:

> **¿El cuello es la resolución de las variables o la cantidad de nodos — y dónde pondríamos la
> próxima variable de decisión?**

El criterio **no** es «qué factor mueve la métrica». Un factor con efecto enorme pero óptimo
**invariante** se resuelve con una constante y no da headroom a ninguna política. El headroom
vive en dos sitios, y ambos se miden:

* **`S_T − S1`** — la fracción de varianza que un factor solo aporta **interactuando**. Si
  `S_T ≈ S1` en todos, la superficie es **aditiva**: ni más resolución ni una política aprendida
  pueden ganar, y un nodo nuevo solo ayudaría si interactúa.
* **desplazamiento del `argmax` entre regímenes de riesgo** — si el mejor ajuste de un factor
  **se mueve** con el régimen, una política dependiente del estado paga; si no, no.

## 2. Método — práctica estándar en cadenas de suministro, tres etapas

1. **Morris (efectos elementales)** — cribado barato, `μ*` (importancia) y `σ` (no
   linealidad/interacción). Descarta inertes; no cuantifica.
2. **Sobol (descomposición de varianza)** — muestreo Saltelli, estimadores **Jansen**, IC
   bootstrap. Da `S1` y `S_T`. **Es la etapa que contesta la pregunta.**
3. **Barrido 1-D por régimen** — `argmax` de cada factor dentro de cada régimen de riesgo, y su
   dispersión.

**Las tres, sobre las dos métricas de resiliencia**: `ret_excel` (canónica, 2017) y el índice
**Cobb-Douglas** (IJPR 2024). Que un factor tenga headroom en una y no en la otra es un
resultado en sí.

## 3. Factores — 20, todos contra el contrato de acción VIGENTE

> **Defecto encontrado al preparar esto:** `scripts/run_program_i_sensitivity.py` **no corre en
> HEAD** — usa `op8_rop`, que no está en `sim.params`. La pantalla Morris sellada del
> 2026-07-12 (320 filas) por tanto **no es reproducible hoy**. Se reporta; no se reutiliza su
> mapeo. Los factores de abajo se validan contra `sim.params` antes de correr (`f6`).

| # | factor | nodo | rango |
|---|---|---|---|
| 1 | `op1_rop` | proveedor externo (aguas arriba) | 1.008 – 8.064 |
| 2 | `op2_q` | lote del proveedor | 95.000 – 380.000 |
| 3 | `op2_rop` | revisión del proveedor | 168 – 1.344 |
| 4 | `op3_q` | pedido WDC | 7.750 – 47.000 |
| 5 | `op3_rop` | revisión WDC | 84 – 336 |
| 6 | `batch_size` | lote de ensamblaje | 2.500 – 10.000 |
| 7 | `assembly_shifts` | capacidad (turnos) | 1 – 3 |
| 8 | `op9_rop` | despacho batallón | 12 – 48 |
| 9 | `op9_q_max` | cantidad batallón | 1.200 – 5.200 |
| 10 | `op10_rop` | transporte aguas abajo | 12 – 48 |
| 11 | `op10_q_max` | cantidad aguas abajo | 1.200 – 5.200 |
| 12 | `op12_rop` | último tramo | 12 – 48 |
| 13 | `op12_q_max` | cantidad último tramo | 1.200 – 5.200 |
| 14 | `op3_rm` | **buffer** materia prima WDC | 0 – 122.880 |
| 15 | `op5_rm` | **buffer** materia prima ensamblaje | 0 – 122.880 |
| 16 | `op9_rations` | **buffer** raciones batallón | 0 – 126.000 |
| 17 | `risk_frequency_scale` | riesgo (permiso suyo) | 0,5 – 2,0 |
| 18 | `risk_impact_scale` | riesgo (permiso suyo) | 0,5 – 2,0 |
| 19 | `risk_family_selector` | **qué riesgos activos** — R1r / R2r / R3 | 3 bins |
| 20 | `demand_level` | demanda | 0,75 – 1,5 |

Los buffers respetan los topes de su Tabla 6.16. Los rangos de decisión cubren y exceden los
valores por defecto sin salirse de lo físicamente admisible.

## 4. Diseño y coste

* **Sobol**: `N = 256`, `k = 20` → `N(k+2) = 5.632` corridas. Medido: 0,076 s/corrida → **~7 min**.
* **Morris**: 20 trayectorias × 21 = 420 corridas → ~35 s.
* **Barrido por régimen**: 20 factores × 9 puntos × 3 regímenes × 3 semillas = 1.620 → ~2 min.
* Horizonte **52 semanas**, declarado: más corto que sus 10–20 años, así que **no** se compara
  con sus valores absolutos, solo consigo mismo.
* Semillas **4.200.001+**, vírgenes.

Cobb-Douglas exige muestrear por periodos, lo que multiplica el coste por ~300; por eso se
calcula sobre el **subdiseño Morris** (420 corridas) y sobre el barrido por régimen, **no** sobre
las 5.632 de Sobol. Declarado aquí, no descubierto después.

## 5. Predicción, antes de mirar

1. **La suma de `S1` estará por encima de 0,85** — superficie mayoritariamente aditiva,
   coherente con el `R² = 0,982` lineal.
2. **Los factores de riesgo (17–19) tendrán la mayor `S_T − S1`**: la interacción decisión ×
   riesgo es donde puede haber política.
3. **Los buffers (14–16) tendrán `S_T ≈ S1`**: efecto grande, aditivo, óptimo en el tope.
   Si es así, **añadir más nodos de buffer no genera headroom** y la instrucción del 28 de julio
   necesita otro tipo de nodo.
4. **El `argmax` de la mayoría de factores NO se moverá entre regímenes**, replicando la
   invariancia ya medida sobre 45 perfiles de riesgo.

Predigo, en suma, **que el mapa dirá que el problema no es la resolución NI el número de nodos
de buffer, sino la ausencia de interacción decisión × riesgo**. Si la predicción 2 falla y algún
factor de decisión tiene interacción alta, **ése es el sitio donde poner la próxima variable** y
sería la mejor noticia posible.

## 6. Falsadores

| # | qué | puede fallar porque |
|---|---|---|
| f1 | **el estimador de Sobol reproduce Ishigami** (`S1`, `S_T` en forma cerrada) con error < 0,02 | sin esto, cualquier índice es indistinguible de un bug. **Ya verificado: error 8e-4**, incluido el caso `S1 = 0`, `S_T = 0,244` de pura interacción |
| f2 | Morris y Sobol coinciden en el orden de los 3 primeros | si el cribado barato y la descomposición cara se contradicen, uno está mal |
| f3 | los factores en su valor por defecto reproducen el `ret_excel` del bloque congelado | un mapeo de acción roto invalidaría todo |
| f4 | ningún punto fuera de rango; cada factor continuo con ≥ 100 valores distintos | un mapeo colapsado haría el diseño vacuo |
| f5 | determinismo: misma semilla → mismos índices a 1e-9 | — |
| f6 | **todas las claves de acción existen en `sim.params`** | es exactamente el defecto que dejó a Program I sin correr |

## 7. Regla de lectura, declarada

* `Σ S1 > 0,85` ⇒ superficie aditiva ⇒ **el cuello no es la resolución**.
* factor con `S_T − S1 > 0,05` **y** `argmax_span > 20%` del rango ⇒ **candidato a variable de
  decisión**.
* si **ningún** factor cumple ambas ⇒ el headroom no está en este espacio, y la siguiente
  hipótesis es la topología (nodos nuevos), **no** más resolución.

**Prohibido**: re-muestrear, ampliar `N`, o cambiar rangos después de ver resultados.
