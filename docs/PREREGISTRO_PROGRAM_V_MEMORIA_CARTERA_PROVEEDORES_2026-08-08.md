# Program V — memoria en una cartera de proveedores aguas arriba

**Congelado:** 2026-08-08, antes del primer resultado de Program V
**Rol inicial:** `BURNED_PRELEARNER_MECHANISM_GATE`
**Aprendiz neuronal:** no autorizado
**Contrato:** `contracts/program_v_supplier_memory_v1.json`

## 1. Por qué este mecanismo y no otro

La reserva A/B de transporte ya fue Program M y no puede reabrirse cambiando el hazard. El buffer
Op3/Op5/Op9 fue evaluado como postura estática y no mostró heterogeneidad entre tapes. Program V
cambia el derecho de decisión: antes de conocer la realización de la siguiente semana, el operador
compromete una bolsa fija de capacidad entre tres proveedores. La capacidad comprometida a un
proveedor no puede trasladarse después de observar su pérdida de rendimiento.

La tesis incluye procurement y nodos WDC/AL/SB, pero simplifica la disponibilidad de vehículos,
rutas y almacenamiento, y no especifica una cartera adaptativa de proveedores. También identifica
la demanda estacionaria como una limitación y propone integrar forecasting ante cambios de largo
plazo. Por tanto, la cartera, sus yields, el lead de compromiso y el régimen persistente son una
**extensión nuestra**, no parámetros de Garrido-Rios (2017).

Las reuniones de Garrido del 22 y 28 de julio y 7 de agosto de 2026 pidieron: R1 fijo, frecuencia e
impacto R2 aleatorios, demanda estacional, más nodos/derechos de decisión —en particular buffers de
materia prima aguas arriba— y comparación KAN–MLP–MPC. Program V responde a ese pedido sin asumir
que KAN ganará.

## 2. Hipótesis causal

Tres proveedores comparten un presupuesto semanal fijo. En cada semana uno opera en régimen de
degradación severa. El régimen es latente y persistente (`P(stay)=0.88`). Un aviso contemporáneo lo
identifica con exactitud 0.65; los yields realizados de entregas anteriores aportan evidencia
retardada. La acción se compromete una semana antes de la entrega y no se reasigna.

La condición necesaria es una reversión de la acción óptima:

```text
si falla S0: reservar {0, 0.5, 0.5}
si falla S1: reservar {0.5, 0, 0.5}
si falla S2: reservar {0.5, 0.5, 0}
```

La bolsa total pedida es idéntica. El proveedor degradado entrega un factor exógeno cercano a 0.10;
los otros, cercano a 1.00. El stock sólo entra al sistema al llegar una orden; los rechazos del
proveedor nunca se cuentan como inventario destruido.

## 3. `L_(t-1)` identificable

`L_(t-1)` es el posterior causal sobre qué proveedor está degradado, actualizado con avisos y yields
ya realizados. Se contrasta con:

- `reset`: posterior uniforme cada semana, misma señal actual y misma física;
- `delayed`: aviso de la semana anterior;
- `shuffled`: secuencia de avisos de otra posición del mismo tape;
- `last-yield`: regla sin modelo que evita el último proveedor con peor yield;
- `privileged`: ve el régimen verdadero, sólo como diagnóstico.

La red futura no podrá reclamar memoria por recibir una secuencia. Deberá vencer a su reset y a un
filtro Bayesiano con la misma información.

## 4. Gate sin learner

Se ejecutan 60 seeds ya quemadas (`8701001–8701060`), 30 selección y 30 evaluación. Se comparan seis
composiciones constantes, warning, placebos, last-yield, Bayes retenido, Bayes reset y privilegiado.
El endpoint primario es servicio acumulado (`delivered / demanded`) y se reportan backlog AUC,
tiempo de recuperación después de cada shock, coste/volatilidad de compras y masa.

Program V pasa a U1 sólo si simultáneamente:

1. `LCB95(H_priv) >= 0.02` frente al mejor constante;
2. `LCB95(H_obs) >= 0.01` para la mejor política no anticipativa;
3. `LCB95(H_ret) > 0` para Bayes retenido menos reset;
4. retained vence delayed y shuffled;
5. las tres acciones de evitación son óptimas en sus estados vivos;
6. gasto pedido idéntico y residuo de masa `<=1e-9`;
7. no empeora el backlog final ni el tiempo de recuperación frente al constante.

Si Bayes retenido absorbe todo el valor privilegiado, el estado es
`STRUCTURED_BELIEF_SUFFICIENT_FOR_QUALITY`. Eso **no** autoriza una prima de calidad neuronal, pero
sí permite construir el planner DES y probar amortización si su coste online es vinculante.

## 5. Escalera posterior y definición de prima

```text
constante → aviso/last-yield → filtro Bayes → scenario MPC/belief-MPC
→ MPC+spline-GAM → MPC+MLP → MPC+KAN
```

Hay dos estimandos distintos:

- `Delta_quality = V(neural) - V(best_structured)`;
- `Delta_amortization`: no inferioridad en servicio/guardarraíles y reducción predeclarada de
  latencia o llamadas DES frente al planner.

El paper puede demostrar H1–H4 frente al modelo estático y, separadamente, una prima computacional
neuronal frente al planner. No se llamará «prima neural de calidad» a una red que sólo lo imita más
rápido.

## 6. KAN

KAN será un head de valor/residual de baja dimensión sólo si el bake-off se abre. Se emparejará con
MLP y spline-GAM en datos, observaciones, parámetros/FLOPs, tuning, seeds y presupuesto online. La
regla de Garrido se aplica: equivalencia implica elegir el modelo más parsimonioso. Un actor KAN
end-to-end no está autorizado por este contrato.
