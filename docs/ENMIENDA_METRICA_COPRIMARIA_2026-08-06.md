# Enmienda — co-primarias nativas de Garrido, declaradas antes del resultado extendido

**Sustituye a** `docs/ENMIENDA_METRICA_PRIMARIA_ESTAR_2026-08-06.md` (de hace una hora) en el campo
de primaria, y sólo en ése. Contrato: `contracts/garrido_expanded_des_e_star_v3_metric.json`.

**Escrita mientras `results/cobb_douglas_component_headroom_extended/` todavía NO existe.** Ese
orden es lo que hace válida la declaración y queda verificable: el artefacto extendido llevará una
marca de tiempo posterior a este documento y al commit que lo sella.

## 1. La regla de selección que NO se usa, y por qué

El PI pidió *«veamos cuál nos da mayor headroom o nos es más conveniente»*.

**Esa no puede ser la regla, y el motivo es que no funcionaría ni aunque la aplicáramos.** Ya
conocemos las dos cifras (`results/endpoint_headroom_atlas/result.json`):

| endpoint | H_regime (288) | H_regime (4.608) | umbral |
|---|---:|---:|---:|
| `ret_excel_full_ledger` | +0,00028 | **+0,00978** | 0,05 |
| `cobb_douglas_index` | 0,00000 | 0,00000 | 0,05 |

`full_ledger` tiene más. **Y sigue estando 5× por debajo del umbral.** Elegirlo por eso no compra un
resultado positivo — compra la acusación de haber elegido el endpoint por su respuesta, que es
exactamente el defecto que este contrato prohíbe y que nosotros mismos escribimos hace una hora.

**La conveniencia no está disponible aquí porque no hay nada conveniente que comprar.**

## 2. Lo que sí se cambia, y por qué es mejor

El cambio que el PI pide **sí tiene una justificación legítima, y es fuerte**: las dos son de
**Garrido**, y `service_first_resilience_v2` la inventamos nosotros.

Para un paper con Garrido de coautor y con destino C&IE, una primaria **estipulada por nosotros**
es atacable —*«¿por qué esa elección normativa?»*— mientras que su fórmula del Excel sin el defecto
de censura, y su índice publicado en IJPR 2024, no lo son.

```text
co-primarias  : ret_excel_full_ledger  Y  cobb_douglas_index
guardarraíl   : service_first_resilience_v2  (obligatorio, bloqueante)
panel         : sin cambios
```

**El criterio es el mecanismo, no el resultado:**

* **`ret_excel_full_ledger`** — la fórmula de Garrido puntuando *todos* los pedidos generados, los
  no servidos a 0. **Quita la censura**, que es el mecanismo medido que hace que `ret_excel` premie
  el abandono. Máxima continuidad con la tesis, mínimo defecto.
* **`cobb_douglas_index`** — su índice de 2024, cinco variables físicas, **y sobrevive la prueba de
  abandono medida** (`results/metric_audit/abandonment_v1/`: coincide con el servicio, elige 0,5
  donde el ReT elige 0,1).

**`service_first_v2` baja a guardarraíl, no desaparece.** Es lo único que **no se puede ganar
abandonando por construcción**, así que sigue siendo bloqueante: ninguna configuración se promueve
si empeora el fill del peor reclamante más allá de su margen. Deja de decidir el ranking y pasa a
vetar.

## 3. El precio que se paga por tener dos

**Dos primarias exigen control de multiplicidad, y se paga.** Corrección **Bonferroni sobre 2
endpoints**: el umbral por endpoint pasa de `LCB95 > 0` al equivalente de `alpha/2`, y **un
resultado sólo cuenta si sobrevive la corrección**.

**Prohibido reportar «la mejor de las dos» sin la corrección.** Si una pasa y la otra no, se
reportan **ambas** con su corrección aplicada y se dice cuál pasó — nunca se presenta la ganadora
como si hubiera sido la única declarada.

## 4. Falsadores de esta decisión

| falsador | por qué puede fallar |
|---|---|
| `f1_declared_before_the_extended_result` | la fecha de este documento y de su commit deben preceder al `created_at` de `results/cobb_douglas_component_headroom_extended/result.json`. **Falla si el artefacto ya existía** |
| `f2_multiplicity_is_actually_applied` | todo contraste sobre las co-primarias debe llevar su corrección por 2. Falla si aparece un `LCB95 > 0` sin corregir |
| `f3_the_guardrail_still_blocks` | `service_first_v2` debe poder vetar una promoción. Falla si ninguna configuración es jamás vetada por él, porque entonces es decoración |
| `f4_selection_was_not_on_headroom` | queda registrado que `full_ledger` tiene **más** headroom que Cobb-Douglas y que **ambas** se declaran igual. Falla si alguna vez se descarta una de las dos citando su H |

## 5. Lo que no cambia

`ret_excel` **sigue prohibido como objetivo de entrenamiento y como primaria**: está medido
premiando el abandono. `ret_excel_risk_conditional`, con 65× más headroom normalizado que la
canónica, **sigue descartado** — y ése es el mejor testimonio de que la selección no se hizo por
headroom.

**No abre semillas, no adjudica, no autoriza aprendices.**
