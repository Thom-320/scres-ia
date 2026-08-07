# Enmienda 1 al registro de decisión del PI — 7 de agosto de 2026

**Predecesor:** `docs/DECISION_PI_ENDPOINT_Y_APERTURA_PROGRAM_L_2026-08-07.md`
(sha `52ed092402bfd14d…`, commit `1157eec`)
**Motivo:** L-0 corrió, y su resultado (a) cierra el carril que el predecesor abrió, y (b) obliga a
corregir una cantidad mal nombrada en ese mismo documento.

**Filas superseded:** Decisión 2 completa. **Filas intactas:** Decisión 1 (endpoint), sin cambios.

---

## E1 · Corrección: lo que el predecesor llamó `H_PI` no es `H_PI`

El predecesor citó una columna `H_PI` del JSON almacenado como si fuera headroom de información
perfecta. **No lo es.** El instrumento actual
(`research/paper2_exhaustive_search/program_l_full_des_gate.py`, sha `45168695dfd317ee…`) emite:

```
"H_PI_certified": false
"heuristic_true_state_delta": <la columna en cuestión>
nota: "diagnostic true-state rule is NOT H_PI; comparator frontier incomplete"
```

Es una **regla miope de estado verdadero**, no un oráculo de horizonte completo. El JSON almacenado
lleva la etiqueta antigua porque se generó antes de esa corrección del script.

**Corrección:** en cualquier cita futura, esa columna es `heuristic_true_state_delta`, diagnóstica y
no certificada. `H_PI` de Program L **nunca se midió**.

Esto también invalida el `eta` / `eta_diagnostic` como magnitud legible: es un cociente de dos
cantidades minúsculas y toma valores de −8,60 a +73,16 sin significado. No se reporta.

---

## E2 · Decisión 2 revocada: Program L cierra

> **`CLOSE_PROGRAM_L_ROUTE_FAMILY`** — lectura B, adjudicada por el PI después de que **ambas
> lecturas se pusieran por escrito sin resolver** (`RESULTADO_PROGRAM_L_L0_2026-08-07.md`,
> commit `4d82765`).

Certificado legible por máquina: `results/paper2_search/program_l_l0_adjudication.json`.

### Qué encontró L-0

18 celdas, 40 tapes idénticas (8500001–8500040) en todas, **cero valores de semilla nuevos**.

1. **El gradiente es real y está reproducido** — 8/8 celdas originales dentro de su propio IC95.
2. **Pero no es una rampa: es una U invertida.** `H_obs` sube de −0,00704 (cobertura 0,02) a
   **+0,00541 (cobertura 0,54)** y cae a −0,03484 (cobertura 1,25). La rejilla original **no estaba
   truncada por debajo de un techo: estaba truncada en el pico**.
3. **0 de 18 celdas alcanzan `LCB95 > 0`.** El máximo es +0,00541 con `LCB95` −0,00735.
4. **El pico es ~28× menor** que el headroom de contención de Program O (0,1515).
5. **El mecanismo existe pero es de tipo «alternar», no «saber»:** `best_static` pasa a `alternate`
   exactamente en las dos celdas del pico, y ahí la política informada por la señal no bate al
   estático de forma significativa. Misma forma que op12.

### Lo que la premisa de apertura resultó ser

El predecesor abrió el carril porque el nulo «no era un nulo, era un gradiente truncado». **Era un
gradiente**, y estaba reproducido — pero truncado **en el pico**, no antes de él. La premisa era
correcta en la forma y equivocada en la dirección: extender la rejilla no encontró más señal, encontró
el otro lado de la colina.

Los tres supuestos de dominio declarados en el predecesor (almacenamiento CSSU finito, disrupciones
persistentes, vehículo único con retorno) **se mantienen declarados como nuestros**. No se retiran:
son las condiciones bajo las que este negativo es válido.

### El carril se cierra con sus reaperturas nombradas

Rige la instrucción permanente: *un negativo bajo la física vieja no es un negativo bajo la física
nueva*. Este cierre **no borra el carril**.

**Reabre si:**

- **Cambia la métrica.** Este screen usó `ret_excel` canónico, que pierde monotonía bajo riesgo.
  **L-1 nunca se corrió.** Es la reapertura más barata y la más justificada.
- **Se aísla la contención.** El nulo de flota fungible (**L-2**) nunca se midió, así que aquí no se
  demostró que la contención fuera el mecanismo — solo se asumió.
- **Garrido aporta física R03** distinta de nuestras tres relajaciones declaradas.
- **El disparo de despacho pasa a ser variable de decisión.** Este screen **congeló un trigger de
  umbral de staging compartido por todas las políticas** para aislar la elección de ruta: el
  *timing* de despacho estuvo constante por construcción, y nunca se probó.

**No reabre por:** extender más la rejilla en frecuencia o duración (la saturación se alcanzó y se
pasó), ni por entrenar un aprendiz sobre el mismo contrato (no hay headroom que capturar).

---

## E3 · Dos defectos de preregistro, congelados como regla

L-0 falló contra su propio preregistro de dos maneras, y ambas son de diseño, no de medición.

### R6 · Un falsador debe poder pasar, no solo poder fallar

`F1` aplicó un **test de signo a una cantidad que cruza cero**. En la celda (4,120) el valor es
−0,00239 y +0,00201 en dos corridas con semiancho 0,01126: indistinguibles de cero y entre sí. El
test no podía pasar de forma fiable, así que no discriminaba lo que decía discriminar.

> Complemento a `falsifier-must-be-seen-to-fail`: además de **poder fallar**, un falsador debe
> **poder pasar cuando la hipótesis es cierta**. Antes de congelarlo, comprobar que su criterio es
> estable en el régimen donde se va a evaluar — y nunca aplicar un test de signo a una cantidad cuyo
> intervalo contiene cero.

### R7 · Las ramas de una regla de decisión deben particionar, no enumerar

Escribí cuatro ramas y la frase «no hay cuarta salida». El resultado cayó **entre dos**: `OPEN_L1`
pedía `LCB95 > 0` (no se cumplió) y `CLOSE` pedía «`H_obs` ≤ 0 en toda la extensión» cuando su
intención declarada tres líneas después era «ninguna celda supera la barra». Dos celdas positivas y
no significativas cayeron en la grieta.

> Una regla de decisión debe **particionar el espacio de resultados**: cada rama definida sobre el
> mismo estadístico que la barra usa, y una rama final `else` explícita. Enumerar casos plausibles no
> es particionar.

### Y la regla que sí funcionó

R4 —*un guardarraíl no se retira después de ver quién gana*— se aplicó **en la dirección incómoda**:
el defecto de redacción era mío y la resolución me habría favorecido, así que ambas lecturas se
escribieron sin resolver y la decisión fue del PI. Es el precedente que la hace creíble: una regla
que solo se invoca cuando conviene no es una regla.

---

## Estado tras esta enmienda

```
ENDPOINT_PRIMARIO      resiliencia, sin piso por peor producto        (Decisión 1, intacta)
METRICA_OBLIGATORIA    full_ledger / Cobb-Douglas; ret_excel prohibido como endpoint
PROGRAM_L              CERRADO — CLOSE_PROGRAM_L_ROUTE_FAMILY, con 4 reaperturas nombradas
PROGRAM_L_H_PI         NUNCA MEDIDO — la columna citada era heuristic_true_state_delta
L-1 / L-2 / L-3        NO corridos; L-1 y L-2 son las reaperturas vivas
APRENDIZ               NO autorizado
SEMILLAS NUEVAS        NO autorizadas; 0 valores abiertos por L-0
REGLAS NUEVAS          R6 (un falsador debe poder pasar), R7 (las ramas particionan)
```

## Custodia

Documento datado; no se edita en sitio. Una corrección se emite como `…_ENMIENDA_2.md`.
