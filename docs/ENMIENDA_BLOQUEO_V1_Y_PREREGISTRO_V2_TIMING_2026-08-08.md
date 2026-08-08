# Enmienda — V1 queda bloqueado como instrumento, y V2 se preregistra aquí

## Parte A — `budgeted_timing_headroom` V1 es un diagnóstico, no un veredicto

`results/budgeted_timing_headroom/result.json` (sello `c3f0497c…`, commit `5b7b98ce`) **se conserva
y se reetiqueta**:

> **`BLOCKED_BY_DESIGN_AUDIT_DEVELOPMENT_ONLY`** — su oráculo aproximado, el desajuste de
> discretización del presupuesto, la confusión entre timing e intensidad y una estructura de
> controles negativos inválida impiden **tanto** una interpretación de headroom **como** una de
> ausencia.

Un auditor externo listó nueve bloqueadores. Los verifiqué y **los nueve son ciertos**. Los tres
que anulan cualquier lectura:

**1. El test de ausencia estaba invertido, y es el error de fondo.** Declarar `STOP` cuando
`LCB95 < 0,01` es *«no conseguimos demostrar superioridad»*, que **no es** *«demostramos que no la
hay»*. Para ausencia hace falta el test de equivalencia: `UCB95 < δ`. Es el mismo defecto que el
cierre de Program L ya nos había enseñado y volví a cometer.

**2. Mi frase sobre el clarividente estaba exactamente al revés.** Escribí que un techo
aproximado, al ser cota **inferior** del clarividente verdadero, hacía un STOP *más fuerte*. Es al
contrario:

```
Δ̂ = L(open) − L(oráculo aproximado)      y      L(aprox) ≥ L(verdadero)
⟹  Δ̂ ≤ Δ*     ⟹  un Δ̂ positivo establece headroom; un Δ̂ = 0 NO acota Δ*
```

Un cero de una búsqueda restringida significa **que la búsqueda no encontró el calendario**, no que
no exista.

**3. La barra no tenía las unidades del endpoint.** `L = ration-hours / rations` está **en horas**,
así que `0,01` significaba **36 segundos** de tardanza media por ración. La barra venía de
`H_regime`, que es adimensional, y la trasplanté sin convertirla.

Los otros seis, todos ciertos y todos corregidos en V2: el presupuesto discretizado a 6/13/19
unidades cuando el contrato decía 6,5/13/19,5 (**violé mi propio contrato congelado**); `uniform`
gasta en S2 mientras `contiguous` y `ranked` gastan primero en S3, confundiendo timing con
intensidad; la regla sólo lee `pending_backorder_qty` y por tanto es reactiva, no risk-aware;
R22/R23 no son inmunes causalmente y además R21/R24 siguen activos en `current` en todas las
celdas; `claim_status` se calcula **antes** que `all_passed`, así que el JSON puede decir `STOP` con
falsadores en rojo; y `f2` pasaba con un `or` que sólo exigía no exceder el presupuesto.

**Consecuencia:** V1 no autoriza aprendiz, ni MPC, ni claim de ausencia. Se cita como diagnóstico.

## Parte B — Preregistro de V2, escrito antes del runner

Runner: `scripts/run_exact_timing_headroom_v2.py`. Custodia: réplica declarada, sin semillas nuevas.

### B1. Mucho más estrecho, a propósito

`D0 × R24 × turnos`, y **sólo S1/S2**. Excluir S3 es lo que separa *cuándo* de *cuánto*: cada
política juega **exactamente `K = 13` semanas de S2** de 26. Presupuesto e intensidad **idénticos
por construcción**, así que lo único que varía es el timing. `f1` lo verifica exactamente, no con
tolerancia.

R24 (*contingent demand surge*, op13) es el riesgo alineado con turnos. **R21 se retira de esta
familia**: con el buffer fijado en cero no se está probando honestamente el preposicionamiento, y
le corresponde a la familia de inventario.

### B2. Una clase de oráculo EXACTA, y una búsqueda que no puede concluir ausencia

| clase | contenido | qué puede afirmar |
|---|---|---|
| **exacta** | los **26** inicios del bloque contiguo de `K` semanas, enumerados **todos** | `NO_MATERIAL_HEADROOM_WITHIN_EXACT_CLASS` vía `UCB95 < δ` |
| **enriquecida** | la exacta **más** 150 subconjuntos aleatorios de `K` semanas, el rankeado por presión y el calendario realizado por la regla | sólo `HEADROOM_FOUND` o `HEADROOM_NOT_FOUND_BY_SEARCH` |

Ésta es la familia enriquecida que faltaba, y la separación es lo que hace honesta cada afirmación:
**una búsqueda heurística nunca produce un STOP general.**

### B3. La regla agota el presupuesto sin ver el futuro

```
S2 si la señal lo indica, o si presupuesto_restante == decisiones_restantes
```

Así completa exactamente `K` sin que se le regale información. Si en cambio hubiera que rellenarle
el presupuesto a posteriori, se le estaría dando el futuro.

### B4. Endpoint adimensional

```
L* = Σ qᵢ·[eᵢ − (OPTᵢ + LTᵢ)]₊  /  Σ qᵢ·[T − (OPTᵢ + LTᵢ)]₊
```

Exposición realizada sobre exposición máxima posible, en `[0, 1]`, invariante a la política en el
denominador. **`δ = 0,01` vuelve a significar un punto porcentual de exposición máxima**, que es
una cantidad que se puede defender ante un revisor. Abandonar sigue sin poder mejorarlo: un pedido
no servido toma `eᵢ = T` y satura su propio término.

### B5. El estimando de riesgo, no un control negativo físico

```
Δ_R24 = Δ_política(R24 increased) − Δ_política(R24 current)
```

Dos celdas, misma custodia. R22/R23 **dejan de ser controles negativos decisorios** — la fuente no
establece que sean causalmente inmunes a un cambio de turnos aguas arriba — y pasan a sensibilidad
en una familia posterior.

### B6. La lógica de decisión, y los falsadores bloquean el claim

```
si falla cualquier falsador          -> BLOCKED_INSTRUMENT
si LCB95(Δ) ≥ δ                      -> HEADROOM_ESTABLISHED
si la clase es exacta y UCB95(Δ) < δ -> NO_MATERIAL_HEADROOM_WITHIN_EXACT_CLASS
en otro caso                         -> INCONCLUSIVE
```

**`BLOCKED_INSTRUMENT` se evalúa primero y entra en `claim_status`**, no sólo en el código de
salida. En V1 el JSON decía `STOP` con `f3` en rojo, y eso no puede volver a pasar.

**No hay rama `STOP`.** No superar una barra por abajo no es ausencia, y el vocabulario del
artefacto ya no permite decirlo.

### B7. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_exactly_K_surge_weeks` | si alguna política no juega exactamente `K`, presupuesto e intensidad no son idénticos y el contraste mezcla política con recurso |
| `f2_endpoint_is_dimensionless` | `L*` debe caer en `[0,1]` en toda celda; fuera de rango significa que el denominador no es la exposición máxima |
| `f3_exact_class_is_exhaustive` | deben enumerarse los 26 inicios; enumerar menos convierte la clase exacta en otra heurística |
| `f4_search_contains_the_exact_class` | la enriquecida debe contener estrictamente a la exacta, o «no encontrado por búsqueda» no se puede comparar con «no existe en la clase» |
| `f5_placebo_does_not_match_the_rule` | mismo presupuesto, mismas `K` semanas, sólo cambia *cuándo*; ya falló en op12 |
| `f6_clairvoyant_dominates` | por construcción; una violación es un índice mal puesto |
| `f7_endpoint_discriminates` | el spread entre los 26 calendarios exactos debe superar **2 errores estándar pareados de las diferencias**, no el error del mejor calendario — el defecto de `f4` en V1 |
| `f8_falsifiers_block_the_claim` | control autorreferencial: si algún falsador falla, `claim_status` debe ser `BLOCKED_INSTRUMENT` |
| `f9_no_fresh_seeds` | custodia central |

### B8. Multiplicidad

`K = 2 celdas × 2 contrastes (exacto, enriquecido) = 4`. Holm sobre 4. **Y las proporciones
bootstrap dejan de llamarse p-values**: la decisión se toma con `LCB95`/`UCB95` sobre el intervalo,
que es lo que corresponde a un estimando continuo, y Holm se aplica al nivel del intervalo.
