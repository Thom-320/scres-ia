# Contrato marco — Programa N: dónde está la prima neural y cómo se cobra

**Fecha:** 2026-08-09. **Congelado antes de la primera corrida del programa.**
**Rol:** `PROGRAM_CONTRACT_NO_EXECUTION_AUTHORISED_BY_ITSELF`.
Cada puerta necesita además su propio preregistro.

## 1. El diagnóstico que motiva el programa

Se reverificaron los **242 artefactos con veredicto** del repositorio. La premisa «RL nunca gana»
es falsa: hay victorias neuronales selladas, y lo que falta es cobrarlas con custodia.

**Por qué pierde RL donde pierde, medido:** en todos los entornos que construimos el estado latente
tiene dos o tres estados **y su modelo generativo es conocido**, así que un filtro bayesiano escrito
a mano es óptimo o casi y una red sólo puede empatar.

* `program_v/prelearner_gate_v1` — privilegiado menos Bayes **+0,00076, UCB95 +0,0023**;
* `headroom/g3_obs_conversion_v2` — `STRUCTURED_CONTROL_SUFFICES_G3_OBS`;
* `headroom/g2_autotomy_threshold` — `THRESHOLD_RULE_SUFFICES`.

**Y el contraejemplo es nuestro:** en `contention_v1` el aprendiz batió al belief-MPC por
**+0,0136 [LCB95 +0,0124]**, y la única diferencia estructural es que el régimen es **semi-Markov
con permanencia mínima**, de modo que el filtro de primer orden está **mal especificado**.

> **La prima neural no vive donde el problema es difícil. Vive donde la creencia exacta no es
> calculable en forma cerrada.**

## 2. Dos primas, separadas y nunca fusionadas

```
Delta_calidad      = V(red) - V(mejor comparador estructurado)
Delta_amortizacion = C_online(planner) - C_online(red),  con no-inferioridad en calidad
```

**Una red que sólo imita al planner más rápido no produce prima de calidad y no se llamará así.**

## 3. La escalera de comparadores, fija y en este orden

```
constante -> regla de umbral -> filtro Bayes -> belief/scenario-MPC
          -> spline-GAM -> MLP -> KAN -> recurrente
```

**La red debe batir al mejor comparador NO neuronal, nunca a la constante.** Ésa fue la omisión que
hizo intestable la prima en `track_b_v1` durante meses, y su propio preregistro lo admite: *«el
orden fue el equivocado»*.

## 4. SESOI y presupuesto

`SESOI = +0,01` en el endpoint congelado de cada puerta. Presupuesto emparejado y reportado por
separado en **parámetros, FLOPs, llamadas al DES y wall-clock**. Para KAN frente a MLP se reportan
las tres comparaciones **por separado**; un KAN estrecho contra un MLP absurdo no demuestra
arquitectura sino que el presupuesto se convirtió en trampa.

## 5. Métrica, con las dos que tenemos

* **Primario: Cobb-Douglas.** Pasa el test de abandono (`metric_audit/abandonment_v1`).
* **Sensibilidad legada: `ret_excel`.** Se reporta y **nunca se entrena sobre ella**: está medido
  que la partición que la maximiza entrega 50,7 % de fill y sacrifica 318.621 raciones, mientras la
  que la minimiza entrega 79,5 % sin sacrificar ninguna.
* Cuando una puerta use un endpoint propio (servicio, AUC de regret), se declara en su preregistro
  y se justifica ahí.

## 6. Falsadores heredados, no reescritos

De `supply_chain/falsifiers.py`: `computed_from` con operandos numéricos, `disclosure()` para lo que
no puede fallar, el **nulo de interacción** para cualquier hueco clarividente, y **un control
obligado a diferir** en toda comparación. Ese último no es retórico: el 2026-08-08 una prueba de
inercia reportó «0 diferencias» porque ambos lados fallaban idénticamente.

## 7. Fidelidad: dos brazos declarados

* **`mfsc_faithful`** — sólo extensiones sobre simplificaciones que la tesis declara (almacén
  finito, coste, multiproducto). Es el **control** y sostiene la reproducción.
* **`mfsc_stressed`** — **autorizado por el PI el 2026-08-09** para cambiar parámetros publicados,
  con tres condiciones que no se negocian:
  1. cada cambio se **declara** y se le **mide el precio de fidelidad** contra el brazo fiel;
  2. el nivel se fija por un **argumento de dominio** y se **barre**, con el extremo fiel como
     control inerte — el protocolo que ya usamos con la vida útil y con los días de suministro;
  3. **ningún resultado del brazo estresado se presenta como reproducción de Garrido-Ríos (2017).**

Advertencia que queda escrita: el `3,17×` de Op2 **es la variable de control del diseño de S** —
aprovisionamiento fijo mientras los turnos escalan, verificado en
`results/procurement_overorder_source/`. Moverlo es legítimo bajo el brazo estresado y **debe
decirse que desactiva el control del experimento original**.

## 8. Regla de parada, escrita antes

Sin `Delta_calidad >= SESOI` contra el mejor estructurado, **no se reclama prima de calidad**. Se
reporta la amortización por separado, o se cierra la puerta. Un resultado nulo en las tres puertas
es un resultado del programa, no un fallo de ejecución.
