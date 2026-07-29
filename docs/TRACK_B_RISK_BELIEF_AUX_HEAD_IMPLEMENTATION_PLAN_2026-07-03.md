# Track B preventivo: memoria de riesgos frecuentes + cabeza auxiliar

Fecha: 2026-07-03

## Idea central

La arquitectura que vale la pena probar no es simplemente "PPO con mas historia". Ya probamos historia de varias formas
(`ppo_mlp_history`, DMLPA/Transformer y RecurrentPPO/LSTM) y no superaron de forma clara al PPO+MLP canonico. La propuesta
mas fuerte es:

```text
PPO o PPO+Real-KAN
  + memoria historica observable de riesgos frecuentes
  + cabeza auxiliar de prediccion de riesgo futuro
  + evaluacion final siempre en ReT Excel de Garrido
```

La intuicion: el agente aprende una creencia interna del tipo "este riesgo comun suele venir pronto" y puede preparar la
cadena antes de que aparezcan backlog, perdida de servicio o congestion downstream. Esa es la version rigurosa del
comportamiento preventivo/biologico que Garrido quiere ver.

## Que ya existe en el repo

### Observacion

Track B ya tiene una base muy aprovechable:

- `v7`: agrega estado del cuello de botella downstream, fill/backorder rolling, forecast 48h/168h y memoria parcial de
  R22/R23/R24.
- `v8`: agrega riesgos realizados por identidad: `active_<risk>`, `recent_<risk>` y duracion reciente para R11, R12, R13,
  R14, R21, R22, R23, R24 y R3.
- `v9`: agrega salud de backlog, tendencias de servicio y throughput del paso anterior.

Esto significa que no partimos de cero. Lo que falta para prevencion real es memoria historica por riesgo comun, no solo
senales activas/recientes.

### Arquitecturas

- `scripts/run_track_b_smoke.py` ya entrena PPO+MLP y RecurrentPPO bajo el contrato Track B.
- `scripts/real_kan_extractor.py` ya conecta `pykan.KAN` oficial como `BaseFeaturesExtractor` de SB3.
- `scripts/dmlpa_extractor.py` ya conecta el extractor DMLPA/Transformer de David.

### Recompensa futura

La idea de recompensa futura existe parcialmente en `FutureCreditRewardWrapper`, pero vive en una linea antigua
(`run_cf20_learning_repair.py`) y no esta portada al contrato Track B 8D. Para esta fase, no empezaria por ahi. Primero
probaria si una representacion predictiva ayuda sin cambiar la recompensa principal.

## Riesgos que conviene aprender primero

No tiene sentido prometer prediccion de black swans. La prevencion debe empezar por riesgos frecuentes, donde el agente
realmente tiene repeticiones suficientes para aprender patrones.

Prioridad recomendada:

1. **R11**: muy frecuente en el regimen actual.
2. **R14**: defectos frecuentes; alta senal operacional.
3. **R24**: demanda/perturbacion downstream relativamente frecuente.
4. **R22/R23**: segunda etapa; menos frecuentes pero relevantes para Op10/Op12.
5. **R3**: solo stress test; no usar como prueba principal de anticipacion.

## Nuevas variables de observacion propuestas

Crear una variante nueva, por ejemplo `v10_risk_memory` o `v8_risk_memory`, que parta de `v8` o `v9` y agregue para cada
riesgo frecuente:

```text
weeks_since_last_Ri
events_count_Ri_8w
events_count_Ri_26w
ewma_Ri_8w
last_duration_Ri_norm
last_impact_Ri_norm
time_since_recovery_Ri
```

Estas variables son defendibles porque solo usan historia observada. No son forecast privilegiado. Si funcionan, el
argumento ante Garrido es mucho mas limpio: "el agente no lee el futuro del simulador; aprende recurrencia desde eventos
pasados".

## Cabeza auxiliar

La cabeza auxiliar debe predecir riesgo futuro comun desde la misma representacion que usa la politica.

Targets por paso `t`:

```text
y_i_1w  = 1 si riesgo Ri inicia en las proximas 1 semanas
y_i_2w  = 1 si riesgo Ri inicia en las proximas 2 semanas
y_i_4w  = 1 si riesgo Ri inicia en las proximas 4 semanas
y_i_8w  = 1 si riesgo Ri inicia en las proximas 8 semanas
h_i_4w  = horas esperadas de disrupcion Ri en las proximas 4 semanas
h_i_8w  = horas esperadas de disrupcion Ri en las proximas 8 semanas
```

Loss:

```text
L_total = L_PPO
        + lambda_cls * BCE(y_pred, y_future)
        + lambda_hours * MSE(h_pred, h_future)
```

La evaluacion final no cambia: sigue siendo ReT Excel. La cabeza auxiliar solo ayuda a formar una representacion interna
predictiva.

## Dos rutas de implementacion

### Ruta A: dos etapas, bajo riesgo

Esta es la que recomiendo primero.

1. Correr politicas/heuristicas y guardar trayectorias con observaciones, acciones y `risk_event_ledger.csv`.
2. Construir labels futuros post-hoc a partir de los eventos realmente ocurridos.
3. Preentrenar el encoder:
   - `MLPBeliefExtractor` para PPO+MLP.
   - `RealKANBeliefExtractor` para PPO+Real-KAN.
4. Inicializar PPO con ese encoder preentrenado.
5. Fine-tune con PPO normal.
6. Evaluar con ReT Excel, misma grilla estatica, mismo CRN, mismas seeds.

Ventaja: evita tocar el loop interno de SB3. Es mucho menos riesgoso y mas facil de auditar.

### Ruta B: PPO con loss auxiliar end-to-end

Mas elegante, pero mas ingenieria:

1. Crear `RiskBeliefRolloutBuffer` que guarde targets auxiliares por paso.
2. Crear `AuxRiskPPO(PPO)` y sobrescribir `train()` para sumar la loss auxiliar.
3. Crear un extractor compartido que exponga:

```python
features = extractor(obs)
risk_logits, risk_hours = extractor.predict_risk(features)
```

4. Entrenar PPO y cabeza auxiliar simultaneamente.

Ventaja: representacion predictiva durante todo el entrenamiento.
Riesgo: es mas facil introducir un bug sutil en PPO/SB3. No la usaria como primer paso.

## Arquitecturas candidatas

### PPO+MLP con cabeza auxiliar

Baseline preventivo mas limpio. Si gana, el mensaje es fuerte: no necesitamos una arquitectura exotica; basta con darle
memoria de riesgo observable y una tarea auxiliar correcta.

### PPO+Real-KAN con cabeza auxiliar

La ruta mas atractiva para Garrido:

- KAN ya gano marginalmente a PPO+MLP en ReT, aunque con mayor costo.
- Sus funciones spline pueden ayudar a explicar umbrales: por ejemplo, cuando `weeks_since_last_R24` pasa cierto punto,
  la politica aumenta `op12_q`.
- Permite una narrativa visual: curvas aprendidas por variable de riesgo, no solo pesos de una red negra.

### DMLPA/DKANA

Se puede mantener como comparador de David, pero no lo haria el primer objetivo del sprint preventivo. La razon es simple:
la atencion no basta como explicacion, y el bakeoff actual mostro que historia+atencion no supera a MLP simple. Si David
quiere defenderlo, el criterio debe ser el mismo: ganar en ReT Excel bajo mismo protocolo.

## Auditoria de prevencion

No basta con que el modelo tenga buen score. Para llamarlo preventivo deben cumplirse tres cosas:

1. **Accion pre-riesgo:** antes de un riesgo frecuente, suben `shift`, `op10_q` u `op12_q` respecto a calma.
2. **Uso de memoria:** al enmascarar o barajar memoria historica de riesgos, cae ReT Excel o desaparece la accion
   pre-riesgo.
3. **Valor en ReT:** la accion previa mejora la ReT Excel final frente a un contrafactual valido.

El auditor actual no puede cerrar esto todavia. Primero hay que instrumentar:

```text
risk_event_ledger.csv
  seed, episode, risk_id, start_step, end_step, operation, duration, severity_proxy
```

Luego se puede correr:

- event-study por riesgo frecuente;
- ablation `memory_zeroed`;
- ablation `memory_scrambled`;
- test sin forecast;
- sensibilidad/occlusion por grupo de features;
- para KAN, curvas parciales/splines si pyKAN permite graficarlas de forma estable.

## Gates de decision

No promover como "preventivo" salvo que:

1. ReT Excel no caiga frente al Track B principal.
2. Mejore contra la grilla estatica bajo el mismo protocolo.
3. El modelo con memoria+aux head supere a la misma arquitectura sin aux head.
4. La degradacion al quitar memoria historica sea medible.
5. La senal se concentre en riesgos frecuentes/aprendibles, no en black swans.
6. La auditoria muestre accion antes del deterioro operativo, no solo despues.

## Sprint recomendado

### Sprint 0: logging

- Agregar `risk_event_ledger.csv` al auditor.
- Confirmar que se puede estratificar por R11/R14/R24.

### Sprint 1: dataset supervisado

- Generar trayectorias con politicas congeladas y/o random/heuristicas.
- Crear labels `P(Ri en 1/2/4/8 semanas)`.
- Entrenar un predictor simple.
- Gate: AUC/Brier/calibracion razonable para riesgos frecuentes.

### Sprint 2: encoder preentrenado + PPO

- `PPO+MLP-belief`.
- `PPO+RealKAN-belief`.
- 3 seeds x 30k smoke.
- Si hay senal, 5 seeds x 60k.

### Sprint 3: confirmatorio

- 10 seeds.
- Sin forecast como principal.
- Forecast solo como sonda.
- Resultado reportado como mejora porcentual vs grilla estatica y ReT Excel interna.

## Mensaje para Garrido

El resultado actual demuestra aprendizaje adaptativo. La siguiente version busca aprendizaje preventivo: el agente no solo
ve que la cadena se dano, sino que aprende la recurrencia de riesgos comunes y prepara la cadena antes. Para hacerlo sin
darle un oraculo, usaremos memoria historica observable y una cabeza auxiliar de prediccion; la metrica final seguira
siendo la resiliencia del Excel de Garrido.

## Referencias metodologicas

- RUDDER / return decomposition para credito temporal en recompensas retrasadas:
  https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html
- Integrated Gradients para atribucion de features:
  https://arxiv.org/abs/1703.01365
- SHAP para importancia de features:
  https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Precaucion: atencion no debe tratarse automaticamente como explicacion:
  https://aclanthology.org/N19-1357/
