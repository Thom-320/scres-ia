# Preregistro — confirmación prospectiva de la lane GSA sobre el último bloque virgen

**Escrito y commiteado ANTES de abrir el bloque y ANTES de correr.**
Runner: `scripts/run_gsa_confirmation_v1.py`.
Autorización: `docs/AUTORIZACION_PI_REPROPOSITO_BLOQUE_7700001_2026-08-07.md`.
Bloque: **7.700.001–7.700.120**, `g3a_v2_development`, repropósito autorizado por el PI.
**Se abre una vez. No hay segunda.**

## 1. La hipótesis, congelada desde el desarrollo

`results/gsa_resilience_only/result.json` (`GSA_QUALIFIES_UNDER_RESILIENCE_ONLY`, 5/5 falsadores)
midió sobre tres bloques **ya abiertos**:

| bloque | H_obs | IC95 | η | obs − placebo | LCB95 |
|---|---:|---|---:|---:|---:|
| `GP_search_3000001` | +0,01307 | [+0,01018, +0,01609] | 0,905 | +0,06906 | +0,05227 |
| `FRESH_4200001` | +0,01136 | [+0,00861, +0,01425] | 0,815 | +0,07294 | +0,05548 |
| `FRESH_4500001` | +0,01001 | [+0,00719, +0,01292] | 0,784 | +0,07329 | +0,05589 |

θ **congelado**, sin reajuste posible: `signal_q 0,532 · lead 2 · surge_mult 1,946 · persistence
short · commonality 0,887 · r22_prob 0,107`. **Ningún parámetro se selecciona con el bloque nuevo.**

## 2. Lo que la confirmación tiene que reproducir

Las 120 semillas vírgenes se dividen en **una sola celda** de 120 cintas — no se trocean para
buscar la partición favorable.

**Primaria:** `H_obs = media(obs − mejor estático)` sobre `ret_order`, bootstrap sobre cintas.
**Co-primaria obligatoria:** `obs − placebo desinformado`, donde el placebo es la secuencia que la
política de creencia produjo en **otra** cinta.

> **CONFIRMA ⟺ `LCB95(H_obs) > 0` **y** `LCB95(obs − placebo) > 0`.**

Cualquier otra cosa → **`GSA_NOT_CONFIRMED_ON_VIRGIN_BLOCK`**, y la lane se cierra por número. El
bloque queda quemado igual: eso es el precio de una confirmación y está aceptado de antemano.

**Margen práctico declarado ahora, antes de ver nada:** el desarrollo dio H_obs entre +0,0100 y
+0,0131. Se declara que una confirmación con `H_obs` por debajo de **+0,005** —aunque su LCB
excluya el cero— se reporta como **confirmada pero con efecto menor que el de desarrollo**, y el
manuscrito cita el número virgen, no el de desarrollo.

## 3. Reportado y **no bloqueante** — decisión del PI, no del runner

`worst_cssu_fill`, `attended`, `lost`, `ret_quantity`. El PI declaró el 2026-08-07 que **la medida
es la resiliencia**; el coste distributivo se reporta entero y sin suavizar, y **no veta**.

Esto es legítimo aquí y no lo sería con `ret_excel` visible: `ret_order_metrics`
(`program_g.py:320`) marca los no atendidos como perdidos y los puntúa **cero**, así que el
abandono ya está pagado. `f3` lo comprueba en vez de creerle al docstring.

## 4. Falsadores, con por qué cada uno **puede** fallar

| falsador | por qué puede fallar |
|---|---|
| `f1_the_block_is_virgin_and_opened_once` | exige que las 120 semillas estén en `7700001–7700120`, que el registro las marque abiertas por **este** contrato, y que no exista artefacto previo. Falla si el bloque ya se tocó |
| `f2_theta_is_frozen_from_development` | compara θ con el sellado del desarrollo campo a campo. **Falla si algún parámetro se movió**, y entonces esto no sería confirmación sino una segunda búsqueda |
| `f3_the_gain_is_not_bought_by_attending_fewer` | correlación por cinta entre la ganancia de ReT y el cambio de pedidos atendidos; falla si es < −0,30. **La ganancia moriría aunque el guardarraíl ya no bloquee** |
| `f4_the_placebo_is_uninformed` | el placebo debe usar la secuencia de **otra** cinta. Falla si coincide con la propia |
| `f5_the_static_baseline_is_the_argmax` | el estático es el mejor calendario **sobre estas mismas cintas** — el comparador más exigente. Falla si no es el argmax, porque entonces el headroom estaría inflado |
| `f6_the_result_can_be_negative` | comprueba que el estimador admite y reporta `H_obs ≤ 0`. Un estimador que sólo puede confirmar no confirma nada |

## 5. Alcance

**Confirmación prospectiva sobre bloque virgen.** Si pasa, es la **tercera** confirmación del
proyecto y la primera sobre el mecanismo bajo el objetivo declarado. **No autoriza entrenamiento**:
autoriza preregistrar una lane con oracle-first. Si no pasa, la lane se cierra y se dice.
