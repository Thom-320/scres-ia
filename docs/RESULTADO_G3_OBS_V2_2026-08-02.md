# Resultado — G3-obs v2: **hay conversión observable, y un `if` de dos ramas la agota**

**Artefacto:** `results/headroom/g3_obs_conversion_v2/result.json` (sello `317daf920579ec6e…`) ·
preregistro `docs/PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md` (commiteado antes) · bloque
**`7.800.001–140`**, abierto sobre la autorización del PI registrada en el propio bloque ·
**70 desarrollo / 70 test disjuntas** · **los ocho falsadores PASAN** · **los tres guardarraíles
PASAN**.

**Veredicto: `STRUCTURED_CONTROL_SUFFICES_G3_OBS`** — el desenlace que el contrato declaró
**esperado y exitoso**, no el temido.

## 1. La potencia, que era el problema, está resuelta

| celda | MDE(90 %) | SESOI | |
|---|---:|---:|---|
| `base` | **0,0092** | 0,010 | **CON POTENCIA** |
| `freq3_imp2` | **0,0085** | 0,010 | **CON POTENCIA** |

Con 16 semillas el MDE era 0,0256/0,0286 y el primario **no era interpretable**. Con 140 lo es.
**El SESOI nunca se movió**; lo que se corrigió fue el defecto de diseño.

## 2. Lo positivo: la conversión observable EXISTE

`H_obs` = mejor umbral sobre señal **por ventana** menos la mejor constante:

| celda | `H_obs` | LCB95 |
|---|---:|---:|
| `base` | **+0,0207** | **+0,0147** |
| `freq3_imp2` | **+0,0129** | **+0,0073** |

Ambas medias **≥ SESOI** y ambos LCB95 **> 0**, que es la regla preregistrada. *Matiz honesto:* en
`freq3_imp2` el límite inferior (+0,0073) queda **por debajo** del SESOI, así que la afirmación es
«media en o por encima del SESOI con el intervalo excluyendo el cero», no «todo el intervalo supera
el SESOI».

**Y la señal es real, no cadencia:**

| brazo | base | freq3_imp2 |
|---|---:|---:|
| `uninformed_placebo` | **−0,0049** | **−0,0085** |
| `wrong_claimant` | **−0,2449** | **−0,1978** |

El placebo no informado **pierde** contra la constante en las dos celdas, y apuntar al reclamante
equivocado cuesta entre **−0,20 y −0,24**. La dirección importa; variar por variar no.

> **Es el primer headroom observable establecido de este carril**, sobre un endpoint que castiga
> el abandono y con los tres guardarraíles respetados con márgenes firmados.

## 3. Lo negativo, y ahora **con potencia**: no hay residual sobre el umbral simple

`residual_over_simple` = política tabular de 5 bins **menos** umbral simple:

| celda | media | IC95 |
|---|---:|---|
| `base` | **−0,0022** | [−0,0049 · **+0,0003**] |
| `freq3_imp2` | **+0,0010** | [−0,0011 · **+0,0031**] |

**Ambos intervalos contienen el cero y ambos caben enteros muy por debajo del SESOI de 0,010.**
Con `MDE ≤ 0,0092` esto **ya no es un nulo sin potencia**: es un nulo de grado equivalencia.

> **Una política más rica no aporta nada sobre un `if` de dos ramas.** Es la misma forma del
> desenlace de G2 —donde el lineal con interacciones ganó al umbral y a las redes— y de Program Q
> —donde `Δ_N` fue TOST-equivalente a cero—, ahora en un tercer contrato y **con potencia
> declarada**.

## 4. El coste del realismo, confirmado y ampliado

| degradación vs. acumulado desde el día 1 | base | freq3_imp2 |
|---|---:|---:|
| retardo de 3 días | +0,0080 [+0,0044] | +0,0054 [+0,0011] |
| ruido `σ = 0,30` | +0,0052 [+0,0018] | +0,0033 [−0,0017] |
| ventana de 14 días | +0,0055 [+0,0019] | +0,0037 [−0,0009] |

**Cuatro de los seis excluyen el cero** —frente a los cinco de seis de la corrida de 16, ahora con
potencia—. El acumulado vale +0,0262/+0,0166 y cada limitación se lleva entre un cuarto y un
tercio. **La conversión sobrevive al realismo, pero pagando.**

## 5. Qué cierra esto

Por la regla terminal fijada de antemano, **`STRUCTURED_CONTROL_SUFFICES_G3_OBS` NO abre G3c**:
sólo un residual material sobre el umbral lo habría hecho, y no lo hay. **El carril se cierra sin
construir permanencia mínima ni coste de cambio** — que es exactamente el ahorro que el contrato
existía para producir.

**Y no autoriza entrenar nada.** El gate neuronal exigía residual observable sobre el mejor control
estructurado; el residual es cero-equivalente.

## 6. Para C&IE

Éste es un resultado de **dos caras**, y las dos son publicables:

* **positiva** — existe valor de decisión dependiente del estado, es **observable con información
  desplegable** (backlog por reclamante, previo a la acción), y es **seguro** bajo márgenes de no
  inferioridad firmados;
* **negativa** — **una regla estructurada de dos ramas lo agota**, con potencia suficiente para
  detectar un residual del tamaño que declaramos relevante.

Eso responde la Q1 de Garrido con más precisión que «no hay prima»: **la propiedad que importa es
el feedback dependiente del estado, no la capacidad del aproximador.**
