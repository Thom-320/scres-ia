# Enmienda — dar memoria a los comparadores clásicos

**Escrita ANTES de correr.** Runner: `scripts/run_search_comparator_ladder_v2.py`. Misma caché,
mismo bloque quemado, mismo presupuesto. **Sin semillas nuevas.**

## 1. La asimetría que invalida el titular de la v1

`results/search_ladder/result.json` dio `NEURON_BEATS_THE_FULL_CLASSICAL_LADDER`, con la neurona en
AUC 0,0498 contra 0,0985–0,1648 de los siete clásicos. **Ese titular no es defendible tal cual**, y
el propio artefacto contiene la razón:

```text
neuron_reset  0.10067      ofat  0.10024      gp_ei  0.10862
```

**Sin memoria, la neurona es un buscador del montón.** Toda su ventaja es `ρ` cruzando la frontera
de contexto. Y los siete comparadores **reinician en cada contexto por construcción**. Así que lo
medido no es «una neurona bate a la optimización bayesiana»: es **«un buscador con memoria bate a
buscadores sin memoria»**, que es casi una tautología y que un revisor de C&IE marcaría en la
primera lectura.

La corrección no es rebajar la neurona: es **dar memoria a los clásicos** y volver a medir.

## 2. Los tres brazos nuevos

Todos reciben exactamente la misma información que la neurona: las observaciones de contextos
anteriores, **normalizadas dentro de cada contexto por su propio min/max de prefijo**, porque
`ret_excel_risk_conditional` vale ~0,009 en R1r y ~0,8 en R2r y agrupar sin normalizar sería
aritmética sin sentido. **Ninguno recibe la etiqueta del contexto**, igual que la neurona.

| brazo | qué memoriza |
|---|---|
| `gp_ei_transfer` | el GP se ajusta sobre las observaciones **acumuladas de todos los contextos**, no sólo las del actual. Es BO con prior calentado |
| `ucb1_transfer` | las sumas y conteos por nivel de factor **cruzan** la frontera de contexto |
| `ofat_transfer` | el diseño de la tesis arranca desde **la incumbente del contexto anterior** en vez de desde `DEFAULT` |

`ofat_transfer` es el más importante de los tres y el más barato: es **la propia tesis de Garrido,
con memoria**. Si captura la mayor parte de la ventaja, el hallazgo honesto deja de ser «una red
gana» y pasa a ser **«lo que vale es la memoria entre corridas, y hasta el diseño de la tesis la
aprovecha si se le permite continuar donde se quedó»** — que responde la Q1 de Garrido y le da el
crédito a él.

## 3. Por qué esto puede fallar, y qué significaría

* **la neurona sigue batiendo a los tres con memoria** → la ventaja no es la memoria sola sino
  **cómo** se representa: un vector de pesos sobre coordenadas de diseño generaliza a
  configuraciones no vistas, mientras que un GP acumula puntos y un bandido acumula marginales.
  Sería el resultado fuerte, y **el mecanismo sería declarable**, no misterioso.
* **`ofat_transfer` o `gp_ei_transfer` la alcanzan** → el titular pasa a ser la memoria, no la red.
  **Es un resultado mejor para el paper**, porque separa el ingrediente que importa del que no.
* **la superan** → `CLASSICAL_SEARCH_WITH_MEMORY_WINS`, y la contribución es exactamente la
  frontera que el marco ya eligió.

## 4. Lectura conjunta con `g1`

`results/surface_gates/result.json` mide `H_regime` +0,0038: **el óptimo es común a los seis
contextos**. Por eso la memoria puede funcionar aquí, y hay que escribirlo así en el manuscrito —
la memoria **no está adaptando al régimen, está evitando re-derivar una constante**. Cualquier
lectura que presente esto como adaptación contextual sería falsa, y la mide nuestro propio gate.

**Nada de esto autoriza entrenar una red ni abrir una semilla.**
