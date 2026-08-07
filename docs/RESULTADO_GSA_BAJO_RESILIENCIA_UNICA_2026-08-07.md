# Resultado — la lane GSA **califica** bajo el objetivo declarado, y con el placebo que nunca tuvo

**Artefacto:** `results/gsa_resilience_only/result.json` (sello `759c2955cccf4062…`,
`GSA_QUALIFIES_UNDER_RESILIENCE_ONLY`) · **los cinco falsadores PASAN** · preregistro
`docs/PREREGISTRO_GSA_BAJO_RESILIENCIA_UNICA_2026-08-07.md`, commiteado antes de correr ·
600 cintas sobre tres bloques ya abiertos, **cero semillas nuevas** · 20 s de cómputo.

## 1. Los tres bloques

| bloque | H_PI | H_obs | IC95 | η | ΔReT_cantidad |
|---|---:|---:|---|---:|---:|
| `GP_search_3000001` | 0,01445 | **+0,01307** | [+0,01018, +0,01609] | **0,905** | +0,01418 |
| `FRESH_4200001` | 0,01393 | **+0,01136** | [+0,00861, +0,01425] | 0,815 | +0,01284 |
| `FRESH_4500001` | 0,01277 | **+0,01001** | [+0,00719, +0,01292] | 0,784 | +0,01300 |

Positivo, con el cero excluido, **en los tres bloques independientes**. La política observable
captura entre el **78 % y el 91 %** del techo de información perfecta — una η que no se había
visto en ninguna otra lane del proyecto.

## 2. El placebo, que es lo nuevo

`headroom_sensitivity.py` **no tenía placebo**: comparaba la política de creencia contra un
calendario estático y nada más. Añadido aquí con el idioma del repositorio —a cada cinta se le
aplica la secuencia que la política produjo en **otra** cinta, misma distribución de acciones sin
información alineada:

| bloque | obs − placebo | LCB95 |
|---|---:|---:|
| `GP_search_3000001` | +0,06906 | **+0,05227** |
| `FRESH_4200001` | +0,07294 | **+0,05548** |
| `FRESH_4500001` | +0,07329 | **+0,05589** |

**El margen contra el placebo es cinco veces mayor que el H_obs mismo.** Aquí el valor está en la
señal, no en la cadencia — lo contrario de lo que se midió en `op12`, donde el placebo desinformado
ganaba.

## 3. Por qué esto NO es la trampa del abandono

Con `ret_excel` sobre la población visible, «sólo importa la resiliencia» premia abandonar: está
medido que el reparto que la maximiza entrega 50 % de fill contra 80 % del que la minimiza.
**Aquí no aplica.** `ret_order_metrics` (`supply_chain/program_g.py:320`) marca los no atendidos
como perdidos y los puntúa **cero**: el abandono ya está pagado.

Y `f2` lo comprueba en vez de creerle al docstring: correlación por cinta entre la ganancia de ReT
y el cambio en pedidos atendidos = **−0,180 / −0,170 / +0,099**, todas dentro del umbral −0,30.
**La ganancia no se compra atendiendo menos.**

## 4. Lo que cambió, y es una decisión, no un hallazgo

El resultado histórico tenía `qualifies_new_lane: false` por una sola razón, literal:

> `worst_cssu_fill_delta -0.13 << -0.02 fairness guardrail`

y su propio veredicto la llama *"the Program-G concentration/**fairness** artifact"*. Es un
guardarraíl **distributivo**. La decisión del PI del 2026-08-07 —*la medida es la resiliencia*—
lo convierte en resultado reportado. **Queda escrito como decisión del PI, no como hallazgo del
runner**, y el coste distributivo se reporta entero y sin suavizar:

| bloque | Δ fill del peor CSSU | Δ atendidos | Δ perdidos |
|---|---:|---:|---:|
| `GP_search_3000001` | **−0,1400** | −0,09 | +0,09 |
| `FRESH_4200001` | **−0,1250** | −0,14 | +0,14 |
| `FRESH_4500001` | **−0,1225** | −0,28 | +0,28 |

La ganancia de resiliencia es real **y** un CSSU pierde ~13 puntos de fill. Las dos cosas van
juntas en el manuscrito.

## 5. `f1` — segundo acotamiento exacto del hueco A2

`f1` recalculó H_PI en el θ localizado y lo comparó con el valor sellado del 14 de julio:

> **0,014446048488184385 contra 0,014446048488184385** — idéntico hasta el último dígito.

Es la **segunda** familia de artefactos que reproduce exactamente bajo la física de hoy (la
primera fue el meta-aprendiz, `max_abs_delta = 0,0`). El hueco A2 sigue abierto para las demás,
pero se estrecha con evidencia, no con optimismo.

## 6. Alcance

Desarrollo sobre bloques ya abiertos. **No autoriza entrenar nada.** Lo que abre es el derecho a
**preregistrar** una lane con oracle-first — que es exactamente lo que el certificado de
agotamiento fija como regla: *«una respuesta que reabre NO autoriza entrenamiento; autoriza
preregistrar el contrato correspondiente»*.
