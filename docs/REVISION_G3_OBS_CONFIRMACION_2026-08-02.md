# Revisión — la confirmación de G3-obs: el defecto es real, el remedio no

## 1. Lo que NO ocurrió, y hay que decirlo primero

El reporte declara *«PID 84707, activo al 96,7 % de CPU»* y un watcher en `84708`.

**Ninguno de los dos existe, y la corrida no produjo nada:**

```
results/headroom/g3_obs_v2_confirmation_20260802/run.log   0 bytes
result.json                                                 no existe
watcher_status.json                                         no existe
procesos g3_obs vivos                                       0
```

Es la regla que este proyecto ya tiene escrita tras un incidente de julio: **un PID vivo no es
evidencia; hay que confirmar una fila registrada antes de reportar «corriendo».** El bloque
`7.900.001–140` sigue `RESERVED_NOT_OPENED`, que al menos es consistente: **no se quemó nada.**

## 2. El defecto SÍ es real, y está bien visto

`results/headroom/g3_obs_conversion_v2/result.json` sella:

```
contract_path  docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md   70f2e8ad…
```

es decir **el contrato v1**, no `PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md` (`ad0395b5…`),
que es el que fija `n = 140`, el bloque `7.800.001+` y el margen `lost_orders` re-derivado.

**Es el mismo defecto de etiqueta que H3′**: el runner selló contra su ruta por defecto. Detectarlo
antes de citar el resultado es correcto y necesario.

## 3. Por qué abrir un bloque nuevo es el remedio equivocado

**Ya tenemos precedente para exactamente este defecto**, y no fue abrir semillas: para H3′ se
**re-ejecutaron las mismas rebanadas como réplica declarada** bajo el contrato correcto,
reproduciendo al último decimal. Eso arregla la custodia **sin consumir un bloque virgen**.

Y hay una razón de fondo, no sólo de coste. **La corrida de 7.800.001–140 ya pasó CON potencia**:
MDE 0,0092 y 0,0085 contra un SESOI de 0,010, los ocho falsadores en orden, y ya llevaba **su
propia partición 70/70** dev/test. Sobre eso, abrir un segundo bloque no supera el test de simetría:

> **¿Habríamos abierto un segundo bloque si el primero hubiera FALLADO?**

Si la respuesta es no, correr un segundo **sólo cuando el primero salió bien** es selección, por
muy prospectivo que sea el contrato del segundo. Ésa es la misma objeción de multiplicidad que
cuatro revisiones externas nos hicieron con la regla de «parar en la primera prima neural», y que
aceptamos.

## 4. Lo que propongo

**(a) Reparar la custodia donde está el defecto.** Re-ejecutar `7.800.001–140` como
`--replay-of g3_obs_v2_powered` bajo `PREREGISTRO_G3_OBS_V2_POTENCIA` (`ad0395b5…`), con
manifiesto de módulos. Si reproduce, el resultado queda contratado; si no, el artefacto anterior
se anula. **Cero semillas nuevas.**

**(b) Conservar `7.900.001–140` sellado**, para una confirmación **genuinamente independiente**
si el manuscrito la necesita — y decidida **antes** de mirar, con el compromiso explícito de
reportar también un fallo. Un bloque virgen gastado en re-etiquetar ya no está disponible para lo
que sí lo merece.

**(c) Reparar el runner**, que es donde vive la causa: que **exija** `--contract` explícito en vez
de caer a un valor por defecto. Un contrato por defecto es cómo se sellan tres artefactos contra
el documento equivocado sin que nadie lo note.

## 5. Lo que sigue siendo cierto del reporte

* No re-etiquetar la corrida anterior es **correcto** — un artefacto fechado no se edita.
* Que la afirmación confirmatoria **no está concedida** hasta que aterrice, pase los falsadores y
  selle el contrato exacto: **correcto**.
* Que el inventario histórico de semillas sigue declarándose incompleto y eso debe constar en el
  artefacto: **correcto**, y ya está en el propio `seed_custody`.
