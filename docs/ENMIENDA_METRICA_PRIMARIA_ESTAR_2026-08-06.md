# Enmienda E\* — selección de la métrica primaria de resiliencia

**Contrato padre:** `contracts/garrido_expanded_des_e_star_v1.json`
(`DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`). Su
`metric_hierarchy.primary_selection_status` estaba en
`PENDING_PI_AND_GARRIDO_SIGNATURE_BEFORE_FRESH_DATA`. Esta enmienda lo cierra.

**No se edita el contrato en sitio.** Esta enmienda lo sustituye prospectivamente y sólo en ese
campo.

## 1. El problema de orden, dicho antes que nada

El contrato dice: *«Select exactly one primary resilience endpoint before fresh data; never switch
after inspecting results.»*

**Y nosotros ya inspeccionamos.** `results/endpoint_headroom_atlas/result.json` mide `H_regime`
para siete endpoints sobre dos rejillas, y
`results/cobb_douglas_component_headroom/result.json` lo mide para las cinco variables del índice
de Garrido. Elegir primaria ahora **no puede ser ciego**, y fingir lo contrario sería peor que el
defecto.

La regla que aplicamos, y que es la única que salva la selección post-inspección:

> **Seleccionar después de mirar sólo es un defecto cuando la selección se mueve hacia el endpoint
> FAVORABLE.** Elegir el menos favorable es una auto-penalización y se declara como tal.

`service_first_resilience_v2` tiene `H_regime = 0` con IC95 [0, 0] — **el más desfavorable de todos
los medidos**. `ret_excel_risk_conditional` tiene 65× más headroom normalizado y **se descarta
explícitamente**. Esa asimetría es lo que hace la selección defendible.

## 2. La decisión

```text
primaria           : service_first_resilience_v2
reporte fiel       : ret_excel_full_ledger
panel obligatorio  : el de metric_hierarchy.always_reported_resilience_panel, sin cambios
```

`service_first_resilience_v2` **se añade a `allowed_primary_endpoints`**, donde no estaba.

## 3. Por qué, criterio a criterio — y con qué medición

| criterio | `ret_excel` | `ret_excel_full_ledger` | `cobb_douglas` | **`service_first_v2`** |
|---|---|---|---|---|
| no se gana abandonando | **✗ medido** | ✓ sin censura | ✓ **medido** | ✓✓ por construcción |
| estable a la cadencia | ✓ tras reparación | ✓ | ✓ no usa `RPj` | ✓ |
| fiel a Garrido | ✓✓ es suya | ✓✓ su fórmula sin el defecto | ✓✓ suya, IJPR 2024 | ✗ **estipulada** |
| resolución | alta pero espuria | media | **baja (~1 %)** | media |
| `H_regime` | +0,0005 | +0,0003 | 0,0000 | **0,0000 [0, 0]** |

**El criterio decisivo es el primero, y es el único que está medido en las cuatro.**
`ret_excel` premia el abandono: el reparto que lo maximiza entrega **50 %** de las raciones, el que
lo minimiza entrega **80 %** (`results/sensitivity/contention_headroom_v1_2/`). Ningún endpoint que
pueda ganarse dejando de servir puede ser primario en un paper sobre resiliencia.

`service_first_v2` es el único donde eso es **imposible por construcción**: su término principal es
el fill del **peor** reclamante, y es lexicográfico, nunca escalarizado — no hay ponderación oculta
que se pueda invertir.

## 4. Una corrección que va en acta

Afirmé que Cobb-Douglas «es ciega al servicio». **Es ciega por construcción —no tiene término de
fill entre sus cinco variables— pero NO se comporta así en nuestra cadena**:
`results/metric_audit/abandonment_v1/result.json` → `COBB_DOUGLAS_SURVIVES_THE_ABANDONMENT_TEST`,
`cobb_douglas_agrees_with_service: true`, y **elige 0,5 igual que el servicio mientras el ReT elige
0,1**. Abandonar mueve ε y κ en su contra, así que la ceguera teórica no se materializa.

Eso deja a Cobb-Douglas como **co-primaria legítima**, y la razón por la que no lo es aquí no es la
ceguera sino dos cosas medidas hoy:

* **resolución**: su recorrido entre repartos es 0,5473–0,5519, un **0,85 %**. Ordena bien, no es un
  microscopio;
* **la regla de exponentes se invierte en nuestra cadena**: `0,20/ln(x_max)` da a τ un exponente de
  0,678 contra 0,014 de ζ —**47,5×**— porque τ recorre `[1 · 1,343]`; y τ está muerta en el 18 % de
  las celdas (`results/cobb_douglas_component_headroom/`).

Queda en el panel obligatorio y como diagnóstico, **no como primaria**.

## 5. Lo que esto NO autoriza

No autoriza maximizar `ret_excel` como objetivo de entrenamiento, y la razón es una medición, no una
preferencia: **está medido premiando el abandono**. Si se quiere entrenar contra la fórmula de
Garrido, el objeto correcto es **`ret_excel_full_ledger`**, que es su misma fórmula puntuando
*todos* los pedidos generados y los no servidos a 0 — sin la censura que crea el incentivo.

Y no autoriza cambiar de primaria después de ver un resultado. **Si algún día se cambia, se cambia
por mecanismo declarado y con la primaria anterior reportada al lado.**

## 6. Falsadores de esta decisión

| falsador | por qué puede fallar |
|---|---|
| `f1_the_primary_is_not_the_most_favourable` | `H_regime(primaria) <= H_regime(cualquier otro endpoint del atlas)`. Falla si alguna vez elegimos el endpoint con más headroom |
| `f2_the_primary_cannot_be_won_by_abandoning` | reproducir el barrido de nueve repartos: el argmax de la primaria debe coincidir con el argmax del servicio. Falla si divergen, como diverge `ret_excel` |
| `f3_the_reporting_endpoint_is_garrido_faithful` | `ret_excel_full_ledger` debe usar su fórmula sin modificaciones y diferir de `ret_excel` **sólo** en la población puntuada |

**Alcance:** fija la primaria del contrato E\*. No abre semillas, no adjudica y no autoriza
aprendices.
