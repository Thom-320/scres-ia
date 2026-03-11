**Presentación guiada en formato de preguntas y respuestas**

Esta versión sirve para ensayar como si Garrido te fuera guiando con preguntas. La idea es que respondas corto, claro y siempre regreses al eje del proyecto.

## 1. ¿Qué hiciste después del DES?

Después del DES, el siguiente paso fue convertir la simulación en una plataforma de decisión. Con el DES puro puedo evaluar políticas fijas, pero no aprender decisiones adaptativas. Por eso lo envolví como un entorno RL: el agente observa el estado operativo, toma una acción, el simulador avanza y luego evaluamos el resultado.

## 2. ¿Por qué no quedarte solo con el DES?

Porque con DES solo podía comparar reglas fijas o manuales. Si quiero que el sistema ajuste decisiones según el estado del momento, necesito un problema secuencial de control. El RL no reemplaza el DES; usa el DES como entorno.

## 3. ¿Qué observa el agente? ¿Por qué son 15 dimensiones?

La observación tiene 15 dimensiones porque resume el estado operativo mínimo que quería exponer en cada decisión semanal:

1. `raw_material_wdc_norm`: inventario de materia prima en WDC  
2. `raw_material_al_norm`: inventario de materia prima en la línea de ensamblaje  
3. `rations_al_norm`: inventario de raciones en ensamblaje  
4. `rations_sb_norm`: inventario de raciones en Supply Battalion  
5. `rations_cssu_norm`: inventario de raciones en CSSU  
6. `rations_theatre_norm`: inventario de raciones en teatro  
7. `fill_rate`: tasa de servicio  
8. `backorder_rate`: tasa de backorders  
9. `assembly_line_down`: si la línea de ensamblaje está caída  
10. `any_location_down`: si alguna localización crítica está caída  
11. `op9_down`: si Op9 está caída  
12. `op11_down`: si Op11 está caída  
13. `time_fraction`: progreso del episodio  
14. `pending_batch_fraction`: lote pendiente relativo  
15. `contingent_demand_fraction`: demanda contingente pendiente relativa

La idea no fue reconstruir todo el estado interno del DES, sino exponer un snapshot operativo compacto y útil para control.

## 4. ¿Por qué dices que hay observabilidad parcial?

Porque el DES interno tiene más estado del que el agente ve: por ejemplo, órdenes en tránsito o tiempos residuales internos. Entonces no lo vendo como un MDP exacto probado, sino como una aproximación Markoviana práctica para control secuencial.

## 5. ¿Cuál es el espacio de acción real?

El entorno actual usa **5 dimensiones de acción**, no 4.

Las primeras 4 son continuas y modifican políticas de inventario:

1. `op3_q_multiplier_signal`
2. `op9_q_multiplier_signal`
3. `op3_rop_multiplier_signal`
4. `op9_rop_multiplier_signal`

La quinta controla capacidad:

5. `assembly_shift_signal`

## 6. ¿Entonces por qué alguien diría “cuatro estados”?

Probablemente porque está pensando solo en las cuatro dimensiones de inventario. Pero el entorno completo tiene cinco acciones: cuatro de inventario y una de turnos.

## 7. ¿Qué hacen esas 5 acciones exactamente?

Las dimensiones `0–3` no escogen cantidades arbitrarias; generan multiplicadores sobre parámetros base:

- `op3_q`
- `op9_q`
- `op3_rop`
- `op9_rop`

Esos multiplicadores van aproximadamente de `0.5x` a `2.0x` sobre el valor base.

La dimensión `4` decide el número de turnos:

- señal `< -0.33`  → `S1`
- señal entre `-0.33` y `0.33` → `S2`
- señal `>= 0.33` → `S3`

O sea: 5 dimensiones continuas de acción, pero la última se discretiza en tres regímenes de capacidad.

## 8. ¿Por qué escogiste justamente esas acciones?

Porque quería una interfaz compacta y operativamente plausible:

- Op3 controla buffer upstream
- Op9 controla buffer downstream
- shifts controla capacidad de ensamblaje

No intenté controlar todas las 13 operaciones. La meta fue un benchmark tratable e interpretable.

## 9. ¿Qué es Stable-Baselines3?

Es una librería estándar de Python que ya trae implementaciones confiables de algoritmos de reinforcement learning como PPO. La usé para no implementar PPO desde cero y poder concentrarme en el entorno, la reward y la evaluación.

## 10. ¿Qué es el MLP aquí?

Es la red neuronal simple que PPO usa para mapear la observación a una acción y estimar valor. La elegí porque la entrada ya es un vector compacto de variables operativas. Para esta etapa era la arquitectura estándar y suficiente.

## 11. ¿Por qué PPO?

Porque es un baseline robusto y ampliamente usado para control secuencial. Quería primero validar el carril experimental con un método estándar antes de hablar de arquitecturas más sofisticadas.

## 12. ¿Qué descubriste con eso?

Tres cosas importantes:

1. la acción de turnos sí cambia materialmente el sistema  
2. `ReT_thesis` no era buena reward de entrenamiento  
3. separar `ReT_thesis` como métrica de evaluación y `control_v1` como reward de entrenamiento deja el carril adaptativo mejor planteado

## 13. ¿Entonces ya ganó PPO?

No lo presento así. Lo correcto es decir que ya hay evidencia preliminar de políticas adaptativas competitivas en una región estrecha, y que la robustez la estamos cerrando con corridas más largas y más seeds.
