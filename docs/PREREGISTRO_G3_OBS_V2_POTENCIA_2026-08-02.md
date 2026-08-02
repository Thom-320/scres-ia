# Preregistro — G3-obs v2: la misma pregunta, con la potencia que le faltaba

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_g3_obs_conversion.py`.
**Sucede a** `docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md`, cuyo resultado fue
`STOP_G3_OBS_UNDERPOWERED`. **El diseño, los brazos, el endpoint y el SESOI no cambian.** Cambian
dos cosas y ambas están declaradas aquí antes de ver un dato nuevo.

## 1. Autorización de semillas — de dónde viene

**El PI autorizó la apertura en sesión el 2026-08-02**, y la autorización está registrada en el
propio bloque: `research/seed_custody_registry.json`, `g3_obs_v2_powered`, campo `authorization`.

Dejo constancia del razonamiento, porque me equivoqué a medias: argumenté que
`authority_ladder_v1` no podía bloquear al declararse `DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY`.
Una revisión externa corrigió la otra mitad: **que un documento no sea autoridad no convierte la
ausencia de autoridad en permiso.** Las dos cosas son ciertas. La resolución: **el borrador ni
prohíbe ni autoriza; la autoridad es el PI, y el PI autorizó.** El bloque se abre sobre eso.

## 2. El bloque, y el cálculo que fija su tamaño

**`7.800.001 – 7.800.140`**, `RESERVED_NOT_OPENED` en el registro, disjunto de los siete bloques
declarados. No toca `7.700.001–120`, que pertenece a G3a.

De la corrida de 16 semillas, con `n_test = 8`:

    MDE(90 %) = 0,0256  (celda base)     y  0,0286  (freq3_imp2)

Para `MDE ≤ SESOI = 0,010` hace falta `n_test = 8 · (0,0286/0,010)² ≈ 66`.

> **Se fija `n = 140`, partido 70 desarrollo / 70 test.** Da margen sobre los 66 requeridos y
> mantiene la partición 50/50 del contrato original.

**Lo que NO se hace:** no se amplía el bloque de 16 ya visto. Extender una muestra después de
mirar su intervalo es peeking. Las 16 quedan como la corrida **exploratoria** que motivó este
cálculo, y se reportan como tal.

## 3. El SESOI no se mueve, y ésa es la razón de existir de este contrato

**`SESOI = +0,010` de `worst_claimant_fill`.** Aflojarlo para que el resultado anterior «pasara»
habría sido elegir el resultado. Lo que se corrige es la **potencia**, que es un defecto de
diseño, no de umbral.

**Y la regla que sigue en pie:** un resultado subpotenciado **nunca** se reporta como nulo.

## 4. Lo único que cambia además de `n`

**`δ` de `lost_orders` pasa de 0,25 a 0,50 pedidos/episodio**, re-derivado en
`docs/ENMIENDA_G3C_MARGENES_OPERACIONALES_2026-08-02.md`: *un pedido perdido cada dos años*. El
0,25 estaba justificado como «4× la granularidad Monte Carlo», que es un **error de tipo** — la
resolución muestral informa la potencia, no qué daño es aceptable. **Se declara antes de correr**,
sobre un artefacto que quedó `HALTED`.

Los demás márgenes no cambian: `flow_fill_rate` 0,005 · `backorder_qty_final` 1,0 % relativo ·
identidades algebraicas 0,0 exacto.

## 5. Falsadores — los ocho, con `f8` reapuntado

Idénticos al contrato anterior salvo `f8`, que ya no compara contra un bloque cableado sino contra
el **registro central de custodia**: un bloque debe encontrarse `RESERVED_NOT_OPENED` o es
colisión. Ese defecto es real y está documentado: el `f6` del meta-aprendiz devolvía `passed=True`
para semillas que el registro marcaba usadas.

## 6. Reglas terminales, sin cambios

* **`H_obs ≥ SESOI`, `LCB95 > 0`, márgenes respetados, residual NO material** →
  `STRUCTURED_CONTROL_SUFFICES_G3_OBS`. **Sigue siendo el desenlace esperado y un éxito**: el valor
  existe, es desplegable, y un `if` de dos ramas lo agota.
* **`residual_over_simple ≥ SESOI` con `LCB95 > 0`** → `G3_OBS_RESIDUAL_OVER_SIMPLE_RULE`. Abre
  G3c; **nunca** una afirmación de prima.
* **`H_obs < SESOI`, o el realismo lo destruye** → `OBSERVABLE_CONVERSION_FAILS`.
* **`MDE > SESOI` otra vez** → `STOP_G3_OBS_UNDERPOWERED`, y se publica el número.
* **cualquier margen violado** → `STOP_G3_OBS_GUARDRAIL`, sin segundo rescate.

## 7. Alcance

No autoriza entrenar. No reabre Program O ni Program Q. Y el resultado anterior **no era un
negativo**: era un no-resultado por potencia, y así se cita.
