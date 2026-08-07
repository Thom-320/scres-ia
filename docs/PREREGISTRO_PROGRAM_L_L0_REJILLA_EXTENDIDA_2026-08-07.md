# Preregistro L-0 — extender la rejilla de Program L más allá del cambio de signo

**Fecha:** 2026-08-07, **escrito antes de correr**
**Autoriza:** `docs/DECISION_PI_ENDPOINT_Y_APERTURA_PROGRAM_L_2026-08-07.md` (commit `1157eec`)
**Instrumento:** `research/paper2_exhaustive_search/program_l_full_des_gate.py`
**Clase:** desarrollo. Sin aprendiz, sin adjudicación, **sin semillas nuevas**.

---

## Por qué se corre

`results/paper2_search/program_l_full_des_gate.json` no muestra un nulo: muestra un **gradiente
monótono** que cruza a positivo en las dos celdas más estresadas, **que son el borde de la rejilla**.
La rejilla se quedó sin estrés antes de que el efecto se estableciera.

L-0 pregunta una sola cosa: **¿el gradiente sigue subiendo más allá del borde, o se aplana?**

## Una complicación que obliga a ampliar el diseño

El script del repo **ya no es el que produjo el JSON almacenado**. El actual emite
`schema: program_l_full_des_development_screen_v2`, renombra `H_PI` → `heuristic_true_state_delta`,
añade `H_PI_certified: False` y declara en su propia nota: *«diagnostic true-state rule is NOT H_PI;
comparator frontier incomplete»*.

Es decir: **la cantidad que la decisión del PI citó como `H_PI` no es H_PI certificado.** Es una regla
miope de estado verdadero, diagnóstica. Se corrige aquí antes de que el error se propague.

Consecuencia de diseño: L-0 **no puede** limitarse a añadir celdas nuevas y compararlas contra
números producidos por un instrumento superseded. Debe **reproducir las 8 celdas originales** con el
instrumento actual, en la misma corrida.

## Diseño

**Instrumento sin derivar.** Se añade al script existente un `--cells`, `--n-tapes`, `--out` y
`--label`; el comportamiento por defecto queda idéntico. No se sed-deriva un runner (precedente:
`--arms` en `2407ab5`; regla de `contract-discipline-failures`).

**Tapes.** `n_tapes=40`, `seed0=8_500_001` → semillas **8500001–8500040**, exactamente las mismas del
gate original y **las mismas en todas las celdas**. No se abre ningún valor de semilla nuevo, lo que
respeta `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`. El diseño queda pareado entre celdas.

**Rejilla — 18 celdas.**

- *Reproducción (8):* `1x24, 2x24, 4x24, 2x72, 4x72, 4x120, 6x120, 8x72`
- *Extensión (10):* `6x72, 6x168, 8x120, 8x168, 10x72, 10x120, 10x168, 12x72, 12x120, 12x168`

Episodio = 56 días = 1.344 h. En `12x168` los eventos suman 2.016 h sobre 1.344: **saturación
deliberada**. Se busca el punto donde Op8 está caído casi siempre y ya no queda nada que esquivar.

**Métrica.** `ret_excel` canónico, **igual que el gate original y a propósito**. L-0 responde
«¿el gradiente continúa?» sobre el mismo instrumento. Cambiar la métrica es **L-1**, y mezclarlas
aquí haría que un cambio de signo no se pudiera atribuir. `ret_excel` no vuelve a usarse como
endpoint más allá de este screen de continuidad.

## Falsadores — y por qué cada uno puede fallar

| # | falsador | **por qué puede fallar** |
|---|---|---|
| **F1** | Las 8 celdas de reproducción recuperan el patrón de signos almacenado, y `H_obs` no difiere del valor almacenado por más de su propio IC95. | El script cambió después de generar el JSON. **Si difieren, el gradiente que motivó esta decisión era un artefacto de un instrumento superseded**, y L-0 no tiene premisa. Es el falsador que más probablemente mata la línea. |
| **F2** | `real_beats_placebo` se cumple en las 18 celdas. | El placebo enruta al azar el 50 % de las veces. Si lo iguala, el valor está en *variar de ruta*, no en *saber cuál* — que es exactamente lo que ya nos pasó en op12, donde el placebo desinformado ganó. |
| **F3** | La saturación se alcanza: en el extremo superior `H_obs` deja de crecer. | Si `H_obs` sigue subiendo monótonamente hasta `12x168`, la rejilla **sigue truncada** y L-0 no concluye: habría que extenderla otra vez. Un falsador que puede dejar el resultado indefinido, y se declara así de antemano. |
| **F4** | `best_static` deja de ser `const_R1` en al menos una celda extrema. | En las 8 celdas originales fue `const_R1` **siempre**. Si con Op8 caído casi todo el episodio la ruta 2 sigue sin ser el mejor estático, la ruta alterna no tiene valor ni en retrospectiva, y el mecanismo no existe. |

## Regla de decisión, congelada antes de correr

```
si F1 falla                              -> L0_INSTRUMENT_MISMATCH
                                            se retira el gradiente del registro de decisión
                                            y se re-adjudica Program L desde cero

si F1 pasa y alguna celda de extensión
   tiene H_obs LCB95 > 0 y bate placebo  -> OPEN_L1  (re-medir con métrica no censurada)

si F1 pasa, H_obs <= 0 en toda la
   extension y F3 confirma saturacion    -> CLOSE_PROGRAM_L_ROUTE_FAMILY
                                            negativo limpio: el carril no tiene headroom
                                            ni con la contención llevada al extremo

si F3 falla (sigue subiendo en el borde) -> L0_STILL_TRUNCATED (sin conclusión; extender)
```

**No hay cuarta salida.** En particular, un `H_obs` positivo cuyo LCB95 cruce cero **no abre L-1**.

## Lo que L-0 explícitamente NO hace

- No entrena nada.
- No certifica `H_PI`. La cantidad reportada es `heuristic_true_state_delta`, diagnóstica.
- No adjudica el carril: `OPEN_L1` es permiso para medir mejor, no un claim.
- No sustituye a L-2 (nulo de flota fungible) ni a L-3 (frontera clásica).

## Salida

`results/paper2_search/program_l_l0_extended_grid.json`, etiqueta
`program_l_l0_extended_grid_v1`. **No sobrescribe** `program_l_full_des_gate.json`.
