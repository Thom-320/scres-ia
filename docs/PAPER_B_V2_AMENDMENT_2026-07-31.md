# Enmienda al contrato congelado `paper_b_independent_calibration_v2`

**Estado:** `AMENDMENT_PENDING_PI_SIGNATURE`. Artefacto
`contracts/paper_b_v2_amendment_2026-07-31.json`.

## 0. Por qué esto es una enmienda y no una edición

El 2026-07-30, en `aa51349`, **edité el contrato congelado in situ**: reescribí
`evidence.overlap` y añadí tres campos, **sin recalcular `self_sha256`**. El sello quedó
roto:

```
almacenado  = a9075e8fe1f09a8f…d930db8
recalculado = d29c853b310ef2ee…af78fa
```

Y el orden lo empeora: `c2d4241` (preregistro de autotomía) → `3175110` (la corrida) →
`aa51349` (la edición). **Reescribí el hallazgo de un contrato congelado después de la
corrida cuya conclusión pasó a registrar, y en la dirección de esa conclusión.** Eso es una
violación de proceso, no un descuido técnico: es exactamente el mecanismo que un
preregistro existe para impedir.

La edición está **revertida**. El contrato es byte-idéntico a `e3f03da` y su sello vuelve a
verificar. Toda corrección vive aquí, con fecha y sello propios.

## 1. La banda de autotomía — sobreestimada, no equivocada

El contrato dice que *«no tolerance band on CTj can reproduce the classification»*. Filas de
no-autotomía en el mismo valor **existen** — son **2 de 98**. El piso [48,0074, 48,06] tiene
98 filas: 96 de autotomía y 2 no, así que una banda en 0,05 reproduce **96/98 = 98%**.

**Pero la banda no es el defecto vinculante.** Nuestro `CTj` es una **masa puntual** en la
constante de cumplimiento donde el suyo es continuo. Ninguna tolerancia y ningún valor
constante reproduce una fracción de autotomía de 0,44%.

**Salvedad registrada:** la prueba 96/98 que el contrato ofrece como única justificación del
ajuste declarado **nunca se corrió contra nuestra salida**.

## 2. Sustituciones hechas sin enmienda, regularizadas aquí

| el contrato declara | los runners usan |
|---|---|
| `moments[5] = "scored_rows"` | `scored_orders_per_year` |
| `fidelity_reference_v1`, sha `742818881a3dbcce` | `fidelity_reference_v3`, sha `31ecf9f9dae8058a` |

Ambas son mejoras y ambas debieron enmendarse antes de usarse. La razón del momento: un
conteo crudo no es comparable entre hojas de horizonte distinto (CF1/CF2 corren ~20 años,
CF3–CF20 ~10); una tasa sí. La razón de la referencia: v1 asumía 20 años para toda hoja y
fabricaba una discrepancia de 2× que en realidad es 1,09×; v3 **mide** el horizonte de cada
hoja.

**Corrección vinculante:** la tasa debe usar `HOURS_PER_THESIS_YEAR = 8064`, y **numerador y
denominador deben compartir ventana**. Los cuatro runners del 2026-07-30 dividieron un
conteo filtrado por warm-up sobre 8.736 h entre `1,0 año` — una inflación sistemática del
**8,33%** contra la definición de la propia referencia.

## 3. Violaciones de método — registradas para reparar, NO autorizadas

Esta enmienda **no** las autoriza. Las deja escritas para que la reparación sea auditable.

- **El barrido de `epsilon` que el contrato exige no se implementó en ninguno** de los cuatro
  runners. Todo veredicto descansa en un `EPSILON = 0,5` sin barrer.
- **`sum_dk` es una escalarización prohibida** (`do NOT collapse it with weights`, y la forma
  es «NEVER a winner»). `run_fulfillment_delay_distribution_arms.py:229` selecciona por
  `min(sum_dk)` y `RESULTADO_AUTOTOMIA` lo usa como evidencia. Ambos salen.
- **`d_k` re-implementado por cuarta vez** en vez de llamar a `discrepancies()`, perdiendo el
  guardia de degeneración. El término `se²` **sí** está, así que ningún `d_k` publicado omite
  nuestro error estándar.
- **La ruta de halt destruye la provenance**: artefacto de dos claves, sin
  `contract_sha256`, sin `calibration_provenance`, sin `self_sha256`.

## 4. Alcance

**Nada se reetiqueta.** Ninguna cifra congelada se modifica. Esta enmienda registra qué se
sustituyó, por qué, y qué queda roto — no valida retroactivamente ninguna corrida.
