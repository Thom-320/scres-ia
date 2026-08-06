# Enmienda — bloque limpio, estimando re-congelado, y la autorización que faltaba

**Contrato padre:** `docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md`.
Esta enmienda **sustituye su §4** (bloque y potencia) y **añade la autorización**. Todo lo demás
—estimandos, regla primaria, falsadores— se conserva sin tocar.

## 1. La autorización, que ahora sí existe

> **El PI autorizó explícitamente abrir la confirmación en sesión del 2026-08-05**, tras revisar
> el veredicto `KEEP_THE_RESERVED_BLOCK` y decidir mantener `n = 60`.

La entrada anterior del registro decía `"granted_by": "PI"` **sin que el PI lo hubiera concedido**:
se la escribió a sí mismo un agente. Esa entrada queda como está —en cuarentena— y **no se reusa
su texto de autorización**. El precedente que se sigue es el de `g3_obs_v2_powered`, que registra
la sesión y la fecha en que la decisión se tomó de verdad.

## 2. Bloque nuevo, porque el anterior no se rescata

`8.100.001–8.100.060` está en `ATTEMPTED_NO_SEALED_ARTIFACT`. Hay rebanadas completas de tres
semillas, **pero el builder sólo escribe rebanadas completas**: una semilla simulada a medias no
deja rastro. Contar archivos mide lo que se escribió, no lo que se consumió, así que **no se puede
demostrar qué semillas siguen vírgenes** — que es exactamente por lo que se cuarentenó el bloque
entero.

Reusar sus semillas «no tocadas» sería contradecir nuestra propia cuarentena y convertir ausencia
de artefactos en prueba de virginidad. **Se abre un bloque limpio:**

```text
8.200.001 – 8.200.060      (60 semillas, NO_KNOWN_COLLISION contra registro y artefactos)
```

El bloque quemado **no se recicla ni se libera**: se queda en cuarentena como evidencia del intento.

## 3. La potencia, re-congelada bajo el orden contractual

El §4 original dimensionó con la SD del artefacto alfabético. Bajo el orden contractual
(`results/custody/grid_transfer_confirmation_repower/result.json`):

```text
delta*        = 0,015        FIJADO ANTES, no se mueve
media delta_M = 0,030497
SD pareada    = 0,027527     (era 0,042154 con la carrera alfabética)
```

**`n = 60` se mantiene**, y la razón está en
`results/custody/confirmation_block_size_audit/result.json`: la SD viene de 12 semillas y su banda
del 95 % es `[0,0195, 0,0469]`. A `n = 26` la potencia cae a **0,495** si la SD está en su extremo
alto; a `n = 60` es **0,798**. Para 0,86 robusta harían falta 73, así que 60 ya es el mínimo
sensato y no un exceso.

**`delta*` no se re-deriva de la media observada.** Hacerlo dimensionaría el experimento para
detectar exactamente el efecto ya visto, y sería adecuado por construcción.

## 4. Lo que no cambia

Estimando primario `E[δ_M]` con `δ_M(s) = AUC_marginal(s) − AUC_UCB1_transfer(s)`; secundario
`E[δ_C]`; unidad de inferencia la semilla; bootstrap pareado, 5.000 remuestras, LCB95 al 2,5 %.
**Éxito confirmatorio:** `LCB95(δ_M) > 0` **y** `LCB95(δ_C) > 0` con todos los falsadores pasando.

**Sin decisiones intermedias, sin extender el bloque, sin reasignar semillas y sin cambiar el
estimando después de mirar resultados.** Una corrida parcial no se rescata como confirmación:
consume la virginidad de lo que haya tocado y se cuarentena, como ya pasó una vez.
