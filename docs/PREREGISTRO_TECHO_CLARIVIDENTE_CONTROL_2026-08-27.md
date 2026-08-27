# PREREGISTRO — Techo clarividente de la lane de control (`control_ceiling_v1`)

| Campo | Valor |
|---|---|
| Fecha | 2026-08-27 |
| Pregunta | ¿Puede *cualquier* política de calendario batir al mejor clásico por el SESOI en esta física? |
| Datos | Paneles sellados de Program Q, 256 tapas de confirmación por celda, ya consumidas |
| Cómputo | numpy sobre `.npz` en disco. **Cero simulación, cero semillas nuevas** |
| Autoriza | nada: es descriptivo y sólo puede CERRAR una lane, nunca abrirla |

## Por qué

RecurrentPPO empata al mejor clásico en las tres celdas: Δ_N = −0,00159 / −0,00072
/ −0,00041, con los tres CI95 simultáneos cruzando cero y TOST pasando a ±0,01.
La pregunta del PI es si tunear hiperparámetros —LSTM 64→256, 200k→1M pasos,
on-policy→off-policy según Ni et al.— podría convertir ese empate en prima.

Tunear acerca una política a su óptimo; **no mueve dónde está el óptimo**. Así que
antes de gastar una campaña de entrenamiento hay que medir si existe algo que
ganar. Esta es esa medición, y es barata porque los paneles ya están calculados.

## Estimandos

Con `X_ol[t,k]` el ReT visible de la tapa *t* bajo el calendario *k* (256 × 65.536),
`X_cl[t,c]` el de los diez comparadores clásicos, y `L[s,t]` el del aprendiz:

```
techo_clarividente = mean_t[ max_k X_ol[t,k] ]  −  max_c mean_t[ X_cl[t,c] ]
margen_sobre_aprendiz = mean_t[ max_k X_ol[t,k] ]  −  mean[ L ]
brecha_fijo          = max_k mean_t[ X_ol[t,k] ]  −  max_c mean_t[ X_cl[t,c] ]
```

Bootstrap **sobre tapas**, 10.000 remuestreos, CI percentil. Por celda, y las tres
reportadas juntas.

## La lógica, y por qué un estimador sesgado sirve igual

`mean_t[max_k]` es un máximo **dentro de muestra** sobre 65.536 calendarios, así que
está **inflado** — Gate-0 midió ese sesgo en el mismo entorno: `Δ_bias` de +0,119 a
+0,176, de 2 a 5 veces el propio estimador ingenuo.

Eso no invalida la medición: la convierte en **una cota superior válida**. El
razonamiento es unilateral y sólo se usa en una dirección:

- **Si el techo INFLADO ya queda por debajo del SESOI**, entonces ninguna política
  —ni clarividente, ni tuneada, ni perfecta— puede batir al mejor clásico por el
  SESOI. La lane cierra por aritmética y tunear es imposible, no improbable.
- **Si el techo inflado supera el SESOI, no prueba nada**, porque está inflado. En
  ese caso el resultado es «no cerrado», y haría falta la versión con separación
  selección/evaluación antes de afirmar que hay margen.

Esta asimetría se declara aquí para que no se lea el segundo caso como un positivo.

## Regla de decisión, fijada antes de mirar

SESOI = 0,01, el mismo de Program Q, sobre el mismo endpoint `ret_visible`.

1. `CLOSED_BY_ARITHMETIC` si `UCB95(techo_clarividente) < 0,01` en **alguna** celda.
   Ningún hiperparámetro puede producir prima ahí y se publica así.
2. `LEARNER_AT_CEILING` si `UCB95(margen_sobre_aprendiz) < 0,01` en alguna celda:
   el aprendiz ya está en el techo clarividente y tunear no tiene a dónde ir.
3. `NOT_CLOSED` en cualquier otro caso. **No es un positivo**: significa que la
   cota superior inflada no basta para cerrar, y que decidir exige el estimador
   con separación selección/evaluación estilo Gate-0.

## Falsadores

**F1 — reproducción del ancla.** `max_c mean_t[X_cl]` y `mean[L]` recomputados
deben reproducir `Delta_N` sellado a 1e-9 en las tres celdas.
*Puede fallar* si el panel que leo no es el que produjo el veredicto; entonces no
estoy midiendo el techo de esta lane. *Puede pasar*: `ret_decomposition` ya
reprodujo esos escalares a 1e-9 hoy.

**F2 — el máximo por tapa no es degenerado.** El `argmax_k` debe variar entre
tapas; si todas eligen el mismo calendario, el techo clarividente coincide con el
mejor fijo y hay que decirlo, porque entonces la clarividencia no aporta nada.
*Puede pasar y puede fallar*: en Gate-0 las 24 tapas eligieron 24 calendarios
distintos.

**F3 — orden de magnitud del sesgo.** Se reporta `techo_clarividente −
brecha_fijo`, que es la parte atribuible a personalizar por tapa, junto al
`Δ_bias` de Gate-0 (+0,119 a +0,176) como referencia externa del inflado.
*No puede fallar*: es una divulgación obligatoria, no una prueba. Se declara como
tal.

## Lo que este preregistro NO autoriza

No abre semillas. No entrena nada. No reabre Program Q, cuyo contrato prohíbe
reentrenar. No cambia ningún estimando sellado. Un resultado `NOT_CLOSED` **no**
autoriza una campaña de tuneo: autoriza, como mucho, diseñar el estimador con
separación selección/evaluación que sí podría decidirlo.
