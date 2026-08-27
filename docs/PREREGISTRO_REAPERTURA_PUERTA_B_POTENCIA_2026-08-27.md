# PREREGISTRO — Reapertura de la Puerta B por potencia (`gate_b_reopening_power_v1`)

| Campo | Valor |
|---|---|
| Fecha | 2026-08-27 |
| Estado | **FIRMADO — congelado antes de abrir el bloque** |
| Reabre | `results/program_n/gate_b_readjudication/result.json` (governing run `gate_b_widened_class`) |
| Bloque de semillas | **9700001–9700112**, virgen |
| Runner | `scripts/run_program_n_gate_b_v1.py` (pipeline existente, no un script nuevo) |
| Salida | `results/program_n/gate_b_reopening_power_v1/result.json` |

## 1. Por qué se reabre

La readjudicación del 2026-08-12 declaró `arms_that_pass = []` contra la clase
no-neuronal ancha y con eso se cerró la prima neural de predicción. La auditoría
de potencia del 2026-08-27 midió que **esa puerta no podía ver el efecto que
declaraba buscar**:

- decidió sobre **5 folds de validación cruzada solapados**, desbalanceados
  2/2/2/1/1, con conjuntos de entrenamiento compartiendo 4–6 de 7 semillas;
- la corrección de Nadeau–Bengio infla el SE ≈1,5 ⇒ **todo intervalo era ~33 %
  demasiado estrecho**;
- **MDE80 entre 0,091 y 0,292** contra un SESOI declarado de **0,05**, es decir
  1,83× a 5,84× el SESOI;
- potencia 0,072–0,345 al SESOI declarado;
- **10 de 12 contrastes de arquitectura tenían UCB95 por encima del SESOI**,
  hasta +0,2436.

Un nulo producido por un diseño que no podía ver el efecto no es un hallazgo. El
veredicto correcto de aquella puerta no era `CLOSED` sino
**`UNDETERMINED_UNDERPOWERED`**. Esta reapertura existe para convertir esa
indeterminación en un veredicto, en cualquiera de las dos direcciones.

**Esto no rescata nada por decreto.** Dos derrotas del programa NO se reabren
porque no son problemas de potencia y así queda declarado aquí: el control
(`gate_a2_track_b`, −0,5594, efecto/MDE80 = −2,13, pierde por 2 MDE) y la
decisión (`phase3_decision_surrogate`, con `h_decision` 6,499e-05 medido al 13 %
de precisión relativa, 154× bajo la barra). Ningún *n* hace existir un premio
ausente.

## 2. Potencia, calculada desde el artefacto sellado

sd por contraste recomputada de `results/program_n/gate_b_widened_class/result.json`
campo `per_fold`; *n* requerido para MDE80 = SESOI = 0,05 con
(z_{0,975}+z_{0,80}) = 2,8016.

| contraste que gobierna el veredicto | media | sd | MDE80 con n=5 | **n necesario** |
|---|---|---|---|---|
| `recurrent − gbdt_lagged` | −0,0300 | 0,0655 | 0,0820 | **14** |
| `mlp_tuned − gbdt_lagged` | −0,1331 | 0,1210 | 0,1516 | **46** |
| `mlp_tuned − gaussian_process` | +0,0342 | 0,1106 | 0,1386 | **39** |
| `kan_tuned − gbdt_lagged` | −0,1501 | 0,1519 | 0,1903 | **73** |
| `kan_tuned − gaussian_process` | +0,0172 | 0,1575 | 0,1973 | **78** |
| peor de los 33 (`kan − train_cell_mean`) | +0,0258 | 0,1766 | 0,2213 | **98** |

**n = 112** cubre los 33 contrastes con 14 % de margen. Se eligió 112 y no 48
precisamente porque con 48 KAN habría quedado indeterminado y habríamos repetido
el defecto que esta reapertura corrige.

## 3. Diseño

- **Unidad de análisis: la semilla.** `grouped_folds` asigna cada semilla a
  exactamente un fold de test por zancada, de modo que cada semilla produce
  exactamente una puntuación retenida. Esas puntuaciones son independientes entre
  semillas; los folds no lo son. El runner emite ahora `per_seed` además de
  `per_fold`; **`per_fold` se conserva sin tocar** para que la reapertura sea
  numéricamente comparable con la corrida sellada.
- **K = 8 folds** de 14 semillas. K no es la unidad de análisis: sólo gobierna el
  ajuste. 8 ajustes sobre 112 semillas de datos.
- **Endpoint primario:** `held_out_r2` sobre `R_cobb_douglas`. Idéntico al de la
  puerta que se reabre — cambiarlo sería sustituir el estimando.
- **Comparador:** el máximo sobre la clase no-neuronal ANCHA (15 miembros),
  **reseleccionado dentro de cada remuestreo bootstrap**. Nunca una línea base
  única preespecificada: ése fue el defecto que infló la prima original de 2/3
  brazos aprobados a 0/3 al corregirlo.
- **Bootstrap:** 10.000 remuestreos **sobre semillas**, pareado por semilla.
- **SESOI = 0,05**, heredado verbatim de la puerta que se reabre. No se mueve.

## 4. Regla de decisión, fijada antes de correr

Para cada brazo neural *k* ∈ {`mlp_tuned`, `kan_tuned`, `recurrent`}, con
Δ_k = R²(k) − R²(mejor no-neuronal reseleccionado):

1. **`PREMIUM_k`** si LCB95(Δ_k) > SESOI.
2. **`EQUIVALENT_k`** si TOST rechaza a ±SESOI, es decir UCB95(Δ_k) < +0,05 **y**
   LCB95(Δ_k) > −0,05.
3. **`UNDETERMINED_k`** en cualquier otro caso.

El veredicto global es `PREMIUM` si algún brazo da `PREMIUM_k`; `EQUIVALENT` si
los tres dan `EQUIVALENT_k`; `UNDETERMINED` en cualquier otro caso.

**La tercera salida es la que hace honesto este diseño.** La puerta que se reabre
sólo podía producir «no rechazo» y lo reportó como cierre. Aquí, por primera vez
en el programa, un resultado nulo puede **certificar equivalencia** en vez de
limitarse a fallar en rechazar.

## 5. Falsadores — cada uno puede PASAR y puede FALLAR

**F1 — `mde_medido_bajo_el_sesoi`.** El MDE80 recomputado desde la sd **medida en
las 112 semillas** debe ser ≤ 0,05 en los 33 contrastes.
*Puede fallar:* si la sd por semilla resulta mayor que la sd por fold —
plausible, porque el fold promedia 14 semillas y amortigua varianza — 112 podría
no bastar. *Puede pasar:* si sd_semilla ≈ sd_fold, n=112 deja MDE80 ≈ 0,047 en el
peor contraste. **Si falla, el veredicto es `UNDETERMINED_UNDERPOWERED` otra vez
y se publica como tal**, con el *n* que sí haría falta.

**F2 — `unidad_de_remuestreo_es_la_semilla`.** Cada semilla del bloque debe
aparecer en `per_seed` exactamente una vez por brazo, y la unión de
`seed_of_fold` debe ser el bloque completo sin repetición.
*Puede fallar* si `grouped_folds` cambia o si un fold queda vacío.
*Ya verificado en smoke*: 4 semillas, 2 folds, cada una puntuada una vez.

**F3 — `sin_regresion_contra_el_codigo_sellado`.** Con las mismas entradas, el
`per_fold` del runner parcheado debe ser **idéntico** al del código anterior.
*Puede fallar* si el parche altera cualquier ajuste. **Ya ejecutado:
`max |dif| = 0,0` exacto sobre 17 brazos.**

**F4 — `comparador_reseleccionado_dentro_del_bootstrap`.** El índice del mejor
no-neuronal debe variar entre remuestreos.
*Puede fallar* si un miembro domina en las 10.000 réplicas — y entonces hay que
declararlo, porque un máximo que nunca cambia no añade la varianza de selección
que se pretende capturar.

**F5 — `custodia_del_bloque`.** Las 112 semillas deben estar fuera de todo bloque
declarado, verificado por `scripts/seed_collision_scan.py` **antes** de correr.
*Puede fallar:* el escaneo cubre 5.456 archivos rastreados y 47.395 enteros tipo
semilla. **Ya ejecutado: 0 colisiones**, informe en
`results/seed_scans/gate_b_reopening_2026-08-27.json`.

**F6 — `el_orden_de_los_brazos_no_predice_el_resultado`.** El ρ de Spearman entre
la puntuación de cada brazo y el orden de la semilla debe tener CI95 que contenga
0. *Puede fallar:* es exactamente la firma que delató el placebo con deriva de
`grid_transfer` (ρ −0,19 a −0,36 en los cuatro brazos `*_marginal`).

**F7 — `KAN_reportado_pase_o_falle`.** `kan_tuned` se corre y se reporta con su
intervalo aunque quede indeterminado. *Puede fallar* si el brazo se cae por NaN,
y en ese caso se declara cuántos folds se perdieron.

## 6. Compromiso

Se publica **gane o pierda**. Si el resultado es `EQUIVALENT`, es la primera
certificación de equivalencia del programa y responde la Pregunta 1 de Garrido
con un intervalo en vez de un adjetivo. Si es `PREMIUM`, la lectura vigente del
repositorio —que la escalera neural es cero o negativa en los cinco peldaños—
queda parcialmente **retractada por nosotros**, y así se dirá. Si es
`UNDETERMINED`, se publica el *n* que haría falta.

Este preregistro **no autoriza** entrenar controladores, no toca guardarraíles de
cola, no reabre Program O ni Program Q, y no cambia ningún estimando sellado.
