# Resultado L-0 — Program L, rejilla extendida

**Fecha:** 2026-08-07 · **Preregistro:** `docs/PREREGISTRO_PROGRAM_L_L0_REJILLA_EXTENDIDA_2026-08-07.md` (`1f95366`)
**Artefacto:** `results/paper2_search/program_l_l0_extended_grid.json`, etiqueta `program_l_l0_extended_grid_v1`
**Instrumento:** `research/paper2_exhaustive_search/program_l_full_des_gate.py`, 18 celdas × 40 tapes
(8500001–8500040, idénticas en todas las celdas) × 6 políticas. Sin aprendiz, sin semillas nuevas.

---

## El resultado, en una frase

> **El gradiente existe, es real y está reproducido — pero no es una rampa: es una U invertida cuyo
> pico no alcanza significación. Ninguna de las 10 celdas de extensión tiene `LCB95 > 0`.**

Y hay que decir esto antes que nada: **`F1` falló por la letra de su propia redacción**, y la
resolución de esa discrepancia no me corresponde tomarla después de ver los datos.

---

## F1 · Reproducción — falla por la letra, pasa por la sustancia

Preregistrado con **dos** condiciones: (a) recuperar el patrón de signos, (b) no diferir del valor
almacenado por más de su propio IC95.

| celda | `H_obs` almacenado | `H_obs` L-0 | \|dif\| | semiancho IC95 | (b) dentro | (a) mismo signo |
|---|---:|---:|---:|---:|---|---|
| (1, 24) | −0,00836 | −0,00704 | 0,00132 | 0,00458 | sí | sí |
| (2, 24) | −0,00754 | −0,00581 | 0,00173 | 0,00660 | sí | sí |
| (4, 24) | −0,00728 | −0,00666 | 0,00063 | 0,00617 | sí | sí |
| (2, 72) | −0,00375 | −0,00011 | 0,00363 | 0,00613 | sí | sí |
| (4, 72) | −0,00353 | −0,00080 | 0,00273 | 0,00580 | sí | sí |
| **(4, 120)** | **−0,00239** | **+0,00201** | 0,00439 | 0,01126 | sí | **NO** |
| (6, 120) | +0,00316 | +0,00541 | 0,00226 | 0,01276 | sí | sí |
| (8, 72) | +0,00402 | +0,00506 | 0,00104 | 0,01176 | sí | sí |

- **Condición (b): 8/8.** Ninguna celda se mueve más que su propio intervalo. La reproducción
  cuantitativa es excelente.
- **Condición (a): 7/8.** La celda (4, 120) cambia de signo, de −0,00239 a +0,00201.

**`F1` = FALLA**, tal como está escrito.

### Por qué falló, y por qué eso es un defecto mío

Apliqué un **criterio de signo a una cantidad que atraviesa cero**. En (4, 120) el valor es
−0,0024 en una corrida y +0,0020 en otra, con un semiancho de 0,0113: **las dos mediciones son
indistinguibles de cero y entre sí**. Un test de signo ahí no puede pasar de forma fiable, y por
tanto no discrimina lo que decía discriminar.

El falsador estaba mal diseñado. Pero la regla de decisión que congelé dice:

```
si F1 falla -> L0_INSTRUMENT_MISMATCH
               se retira el gradiente del registro de decisión
               y se re-adjudica Program L desde cero
```

**No resuelvo esto yo.** Reinterpretar un falsador después de ver que falla es exactamente la regla
R4 de la enmienda 1 —*un guardarraíl no se retira después de ver quién gana*— y da igual que ahora
me convenga la dirección contraria a la de ayer. Las dos lecturas posibles se ponen ante el PI en la
sección final.

---

## F2 · Placebo — pasa, con una advertencia

`real_beats_placebo` es **True en las 18 celdas**. Pero el placebo es muy malo (−0,007 a −0,042) y
la política de señal ronda cero: **le gana al placebo sin ganarle al cero**. Batir a este placebo es
una barra baja y no debe reportarse como evidencia de valor.

## F3 · Saturación — pasa, y con forma

Ordenando por horas de evento sobre las 1.344 h del episodio:

| celda | h evento | cobertura | `H_obs` | LCB95 | Δdespachos | mejor estático |
|---|---:|---:|---:|---:|---:|---|
| (1, 24) | 24 | 0,02 | −0,00704 | −0,01161 | −1,4 | `const_R1` |
| (2, 24) | 48 | 0,04 | −0,00581 | −0,01241 | −1,4 | `const_R1` |
| (4, 24) | 96 | 0,07 | −0,00666 | −0,01283 | −1,1 | `const_R1` |
| (2, 72) | 144 | 0,11 | −0,00011 | −0,00625 | −0,8 | `const_R1` |
| (4, 72) | 288 | 0,21 | −0,00080 | −0,00660 | +0,0 | `const_R1` |
| (6, 72) | 432 | 0,32 | +0,00298 | −0,00445 | +0,8 | `const_R1` |
| (4, 120) | 480 | 0,36 | +0,00201 | −0,00925 | +0,8 | `const_R1` |
| **(8, 72)** | 576 | 0,43 | **+0,00506** | −0,00670 | +1,8 | **`alternate`** |
| **(6, 120)** | 720 | 0,54 | **+0,00541** | −0,00735 | +2,8 | **`alternate`** |
| (10, 72) | 720 | 0,54 | +0,00324 | −0,00757 | +2,1 | `const_R1` |
| (12, 72) | 864 | 0,64 | −0,00111 | −0,01274 | +2,6 | `const_R1` |
| (8, 120) | 960 | 0,71 | −0,00241 | −0,01602 | +3,1 | `const_R1` |
| (6, 168) | 1008 | 0,75 | −0,00102 | −0,01661 | +3,6 | `const_R1` |
| (10, 120) | 1200 | 0,89 | −0,00248 | −0,02125 | +3,7 | `const_R1` |
| (8, 168) | 1344 | 1,00 | −0,01596 | −0,03140 | +4,1 | `const_R1` |
| (12, 120) | 1440 | 1,07 | −0,00658 | −0,02686 | +4,0 | `const_R1` |
| (10, 168) | 1680 | 1,25 | −0,03484 | −0,06432 | +4,4 | `const_R1` |
| (12, 168) | 2016 | 1,50 | −0,02450 | −0,05443 | +4,7 | `const_R1` |

**Es una U invertida limpia.** `H_obs` sube desde −0,0070 con cobertura 0,02, **culmina en +0,0054
alrededor de cobertura 0,43–0,54**, y cae hasta −0,0348 en cobertura 1,25. La saturación se alcanza y
se pasa: **la rejilla original no estaba truncada por debajo de un techo, estaba truncada justo en el
pico.**

Mecanismo coherente: con Op8 caído casi todo el episodio no queda nada que esquivar — la ruta
alterna deja de ser un refugio y el sistema está roto por igual en ambas rutas.

## F4 · El comparador estático — pasa, y en el sitio exacto

`best_static` deja de ser `const_R1` en **(8, 72) y (6, 120)** — que son **precisamente las dos celdas
del pico**. Ahí el mejor estático pasa a ser `alternate`.

Es la mejor evidencia de la corrida de que el mecanismo **existe**: hay una región donde alternar de
ruta bate a insistir siempre en la primaria. Y es también su límite: en esa misma región la política
**informada por la señal** no bate al estático de forma significativa. Es decir, **el valor está en
alternar, no en saber cuándo** — la misma forma que ya medimos en op12, donde el placebo desinformado
ganaba a la regla condicionada al estado.

---

## Contra la regla de decisión congelada

```
OPEN_L1   requiere alguna celda de extensión con H_obs LCB95 > 0
          -> NINGUNA de las 10. La mejor es (6,72) con LCB95 = −0,00445.       NO SE CUMPLE

CLOSE     requiere H_obs <= 0 en TODA la extensión + F3 saturación
          -> F3 sí; pero (6,72)=+0,0030 y (10,72)=+0,0032 son positivos.       NO SE CUMPLE (letra)

STILL_TRUNCATED  requiere que F3 falle
          -> F3 pasa.                                                          NO SE CUMPLE

INSTRUMENT_MISMATCH  requiere que F1 falle
          -> F1 falla por la letra.                                            SE CUMPLE (letra)
```

Dos defectos de mi preregistro salen a la vez:

1. **La rama `CLOSE` está mal redactada.** Pedí «`H_obs` ≤ 0 en toda la extensión» cuando su intención
   —explícita tres líneas más abajo— era «ninguna celda supera la barra»: *«un `H_obs` positivo cuyo
   `LCB95` cruce cero **no** abre L-1»*. Dos celdas positivas y no significativas caen en el hueco
   entre la letra y la intención.
2. **`F1` usa un test de signo sobre una cantidad que cruza cero**, y falla por la única celda donde
   eso era inevitable.

Escribí «no hay cuarta salida». El resultado encontró la grieta entre dos.

---

## Lo que sí se puede afirmar, sea cual sea la resolución

Ambas lecturas coinciden en la sustancia, y **ninguna autoriza un aprendiz**:

1. **Ninguna celda de las 18 alcanza `LCB95 > 0`.** El máximo `H_obs` es +0,0054 con `LCB95` −0,0074.
2. **La saturación se alcanzó y se pasó.** No hay más rejilla que extender por este eje.
3. **El pico es ~28× menor que el headroom medido en contención de Program O** (0,1515), y no separa
   de cero.
4. **El mecanismo existe pero es de tipo «alternar», no de tipo «saber»** (F4 + F2).

Clasificación del artefacto: `DEVELOPMENT_NO_ADJUDICATION_PREREGISTRATION_BRANCH_GAP`.

---

## Las dos lecturas, para el PI

**Lectura A — por la letra: `L0_INSTRUMENT_MISMATCH`.**
`F1` falló; el gradiente sale del registro de decisión y Program L se re-adjudica desde cero. Coste:
tira una reproducción que es 8/8 en la condición cuantitativa por un test mal construido.

**Lectura B — por la intención: `CLOSE_PROGRAM_L_ROUTE_FAMILY`.**
La reproducción es sustantivamente sólida, la saturación se alcanzó, ninguna celda pasa la barra, y
la propia cláusula «un `H_obs` positivo cuyo `LCB95` cruce cero no abre L-1» dice qué hacer con las
dos celdas positivas. El carril cierra con un negativo limpio y bien caracterizado —una U invertida
con su pico localizado— que **es reportable en el manuscrito**.

Mi lectura técnica es que B describe lo que pasó y A castiga un defecto de redacción mío. **Pero la
decisión es del PI**, precisamente porque yo escribí el falsador y ahora me beneficiaría
reinterpretarlo.

Lo que **no** cambia en ninguno de los dos casos: no se abre L-1, no se entrena nada, y no se piden
semillas.

---

## Custodia

Documento datado, no se edita en sitio. El artefacto L-0 **no sobrescribió**
`program_l_full_des_gate.json`. La corrección de que `H_PI` en el JSON almacenado es en realidad
`heuristic_true_state_delta` —no `H_PI` certificado— sigue vigente y afecta a
`DECISION_PI_ENDPOINT_Y_APERTURA_PROGRAM_L_2026-08-07.md`, que debe enmendarse en cualquiera de las
dos lecturas.
