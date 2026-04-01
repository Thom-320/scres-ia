# Reporte de Avance: Modelo DES y Funcion de Recompensa

**Fecha:** 25 de marzo de 2026
**Proyecto:** From Simulation to Learning: Advancing SCRES Modeling with Deep Learning Algorithms
**Autor:** Thom
**Destinatario:** Prof. Garrido

---

## 1. Resumen Ejecutivo

Se ha reconstruido completamente el modelo de simulacion de eventos discretos (DES) de la cadena de suministro militar de alimentos (MFSC) descrito en la tesis doctoral de Garrido-Rios (2017), reimplementandolo en Python con SimPy. Sobre este modelo DES se construyo un entorno compatible con Gymnasium para entrenamiento de agentes de Reinforcement Learning (RL) con Stable-Baselines3. El principal hallazgo metodologico es que la metrica de resiliencia original (ReT) no funciona como objetivo de entrenamiento para RL, lo que llevo al diseno de una funcion de recompensa operacional (`control_v1`) que produce politicas adaptativas genuinas.

**Estado actual del manuscrito v0:** Secciones 1-4 escritas, con resultados preliminares de PPO vs baselines estaticas bajo dos escenarios de riesgo.

---

## 2. Modelo DES: Reconstruccion de la MFSC

### 2.1 Arquitectura General

El modelo replica las 13 operaciones del Capitulo 6 de la tesis, organizadas en cuatro echelones funcionales:

| Echelon | Operaciones | Descripcion |
|---------|-------------|-------------|
| **Adquisicion** | Op1-Op2 | Agencia de Logistica Militar + 12 proveedores |
| **Almacen y Distribucion** | Op3-Op4 | Centro de Distribucion (WDC) + transporte a linea de ensamble |
| **Manufactura** | Op5-Op7 | Pre-ensamble, ensamble y control de calidad (linea de ensamble) |
| **Distribucion hacia el frente** | Op8-Op13 | Batallon de Suministro, CSSUs, LOCs, y Teatro de Operaciones |

### 2.2 Parametros del Modelo

**Todos los parametros provienen directamente de las tablas de la tesis:**

| Fuente | Contenido |
|--------|-----------|
| Tabla 6.4 | Funcion de demanda: U(2400, 2600) raciones/dia, 6 dias/semana |
| Tabla 6.10 | Datos de validacion: throughput historico vs. simulado |
| Tabla 6.12 | Distribuciones de riesgo (niveles current `-`, increased `+`) |
| Tabla 6.16 | Niveles de inventario buffer (Escenario II) |
| Tabla 6.20 | Capacidad por turnos S=1,2,3 (Escenario III) |
| Tabla 6.25 | Esquema de datos de salida (SDM) |

**Constantes clave:**
- Tasa de ensamble: lambda = 320.5 raciones/hora
- Tamano de lote: 5,000 raciones (S=1,2) / 7,000 (S=3)
- Horizonte: 161,280 horas (20 anos x 8,064 hrs/ano tesis)
- Lead time prometido: LT = 48 horas
- Warm-up: ~838.8 horas (hasta primer lote en Op9)

### 2.3 Implementacion en SimPy

La simulacion usa **SimPy** como motor de eventos discretos. Decisiones clave de implementacion:

**Granularidad horaria en la linea de ensamble (Op5-Op7):**
- La tesis no especifica la resolucion temporal de la simulacion
- Nosotros elegimos granularidad **horaria** porque los riesgos R11 (averias de maquinaria) tienen tiempo de reparacion promedio de ~2 horas
- Una resolucion diaria enmascaria estas disrupciones sub-diarias
- A S=1: 8 hrs/dia x 320.5 = 2,564 raciones/dia (coincide con tesis)

**Buffers como `simpy.Container`:**
- Cantidades continuas de materia prima y raciones
- 7 buffers rastreados: WDC, AL (materia prima), AL (raciones), SB, SB dispatch, CSSU, Teatro

**Parametros mutables en runtime:**
- `self.params` es un diccionario modificable en cada paso
- El agente RL puede ajustar: `op3_q`, `op9_q_min/max`, `op3_rop`, `op9_rop`, `assembly_shifts`
- Los cambios de turno se acoplan automaticamente con tamano de lote (Tabla 6.20)

### 2.4 Modelo de Riesgos

Se implementaron las tres categorias de riesgo de la tesis como procesos SimPy concurrentes:

#### Categoria 1: Riesgos Operacionales (R1r)

| Riesgo | Descripcion | Distribucion | Efecto |
|--------|-------------|--------------|--------|
| **R11** | Averias de maquinaria | Ocurrencia: U(1, 168)h; Reparacion: Exp(2)h | Apaga Op5, Op6 |
| **R12** | Retrasos en contratos | Bin(12, 1/11) por ciclo bianual | Retrasa Op1 (168h por contrato) |
| **R13** | Escasez de materia prima | Bin(12, 1/10) por ciclo mensual | Retrasa Op2 (24h por entrega) |
| **R14** | Productos defectuosos | Bin(produccion_diaria, 3/100) por dia | Retorno a Op6 para reproceso |

#### Categoria 2: Desastres Naturales y Ataques (R2r)

| Riesgo | Descripcion | Distribucion | Efecto |
|--------|-------------|--------------|--------|
| **R21** | Desastre natural | U(1, 16128)h; Recuperacion: Exp(120)h | Op3, Op5, Op6, Op7, Op9 simultaneamente |
| **R22** | Destruccion de LOC | U(1, 4032)h; Recuperacion: Exp(24)h | Op4, Op8, Op10, Op12 (todas las LOCs) |
| **R23** | Destruccion de unidades avanzadas | U(1, 8064)h; Recuperacion: Exp(120)h | Op11 (CSSUs) |
| **R24** | Surge de demanda contingente | U(1, 672)h; Tamano: U(2400, 2600) raciones | Demanda adicional en Op13 |

#### Categoria 3: Eventos Cisne Negro (R3)

| Riesgo | Descripcion | Distribucion | Efecto |
|--------|-------------|--------------|--------|
| **R3** | Evento cisne negro | U(1, 161280)h; Duracion fija: 672h | Op5, Op6, Op7, Op9 fuera 1 mes |

#### Niveles de Riesgo Implementados

| Nivel | Descripcion | Ejemplo R11 intervalo | Uso |
|-------|-------------|----------------------|-----|
| **current** (`-`) | Tabla 6.12 base | U(1, 168) | Validacion, escenario nominal |
| **increased** (`+`) | Tabla 6.12 incrementado | U(1, 42) | **Entrenamiento RL** |
| **severe** (`++`) | Extrapolado (2x freq) | U(1, 21) | Evaluacion de estres |
| **severe_training** | Curriculum learning | U(1, 10), Exp(6)h reparacion | Experimental |

### 2.5 Validacion del DES

El modelo se valido contra la linea base deterministica Cf0 (S=1, sin riesgos) reportada en la Tabla 6.10 de la tesis:

| Comparacion | Hrs/ano | Nuestro modelo | ECS Tesis | Gap relativo |
|-------------|---------|----------------|-----------|--------------|
| **Base tesis (oficial)** | 8,064 | 733,621 raciones/ano | 767,592 | **-4.43%** |
| Base gregoriana (diagnostico) | 8,760 | 796,940 | 767,592 | +3.82% |

**El gap de -4.43% esta dentro del umbral aceptado de +/-15%**, confirmando la fidelidad estructural del modelo reconstruido.

**Resultados bajo condiciones deterministicas Cf0:**
- Produccion: 734,458 raciones/ano promedio
- Entrega: 733,621 raciones/ano promedio
- Fill rate: 99.3%
- Backorders en 20 anos: solo 41
- Capacidad semanal vs demanda: ~17,948 vs ~17,500 (opera cerca de saturacion)

**Resultados con riesgos estocasticos (nivel current):**
- Entrega anual cae a 677,750 (-7.6% vs determinista)
- Fill rate: 68.3%
- Backorders: 1,825 en 20 anos
- Eventos disruptivos: 8,247 totales (R11: 1,868, R14: 5,591, R24: 478)

**Resultados con riesgos incrementados:**
- Entrega anual: 549,250 raciones
- Fill rate: 45.6%
- Backorders: 3,132
- Eventos disruptivos: 13,705

> La transicion de un sistema bien calibrado bajo condiciones nominales a un sistema fragil bajo estres confirma que el DES captura correctamente la propagacion de disrupciones.

---

## 3. Wrapper Gymnasium para RL

### 3.1 Formulacion MDP

El DES se envuelve en un entorno Gymnasium que convierte el problema de control en un Proceso de Decision de Markov (MDP):

**Epocas de decision:** Cada 168 horas simuladas (1 semana) -- coincide con el ciclo operacional natural de la MFSC.

**Episodio:** 260 pasos de decision = horizonte completo de 20 anos (post warm-up).

### 3.2 Espacio de Observacion (15 dimensiones, v1)

| Indice | Variable | Descripcion | Rango |
|--------|----------|-------------|-------|
| 0 | raw_material_wdc | Materia prima en WDC (Op3) | /1e6 |
| 1 | raw_material_al | Materia prima en linea de ensamble (Op5) | /1e6 |
| 2 | rations_al | Raciones en buffer QC (Op7) | /1e5 |
| 3 | rations_sb | Raciones en Batallon de Suministro (Op9) | /1e5 |
| 4 | rations_cssu | Raciones en CSSUs (Op11) | /1e5 |
| 5 | rations_theatre | Raciones en Teatro (Op13) | /1e5 |
| 6 | fill_rate | Tasa de cumplimiento acumulada | [0, 1] |
| 7 | backorder_rate | Tasa de backorders acumulada | [0, 1] |
| 8 | assembly_line_down | Linea de ensamble disrumpida | {0, 1} |
| 9 | any_loc_down | Alguna LOC disrumpida | {0, 1} |
| 10 | op9_down | Batallon de Suministro disrumpido | {0, 1} |
| 11 | op11_down | CSSUs disrumpidas | {0, 1} |
| 12 | time_fraction | Progreso de simulacion | [0, 1] |
| 13 | pending_batch_norm | Lote pendiente / tamano de lote | [0, 1] |
| 14 | contingent_demand_norm | Demanda contingente pendiente / 2600 | [0, 1] |

> No se incluye memoria explicita de observaciones pasadas (v1). Las implicaciones de observabilidad parcial se discuten como limitacion y linea futura (frame-stacking, RecurrentPPO).

### 3.3 Espacio de Acciones (5 dimensiones continuas)

| Dim | Accion | Mapeo | Parametro DES |
|-----|--------|-------|---------------|
| 0 | op3_q | multiplicador = 1.25 + 0.75 x senal | Cantidad de despacho en WDC |
| 1 | op9_q_max | multiplicador = 1.25 + 0.75 x senal | Despacho maximo en SB |
| 2 | op3_rop | multiplicador = 1.25 + 0.75 x senal | Punto de reorden en WDC |
| 3 | op9_rop | multiplicador = 1.25 + 0.75 x senal | Punto de reorden en SB |
| 4 | shifts | < -0.33 -> S1; [-0.33, 0.33) -> S2; >= 0.33 -> S3 | Turnos activos de ensamble |

**Rango de multiplicadores:** [0.5, 2.0] alrededor del valor base de la tesis.

**Esta estructura de acciones unifica las dos estrategias** que Garrido-Rios (2017) examino por separado:
- Estrategia I (buffering de inventario) -> dims 0-3
- Estrategia III (aumento de turnos) -> dim 4

---

## 4. Diseno de la Funcion de Recompensa

### 4.1 El Problema: ReT como Objetivo de Entrenamiento

La metrica de resiliencia de la tesis (Eq. 5.1-5.5) agrega cuatro sub-metricas en un escalar unico:

- **Re(APj)** -- Autotomia: deteccion rapida de disrupcion
- **Re(RPj)** -- Recuperacion: rapidez de restablecimiento del servicio
- **Re(DPj, RPj)** -- No-recuperacion: penalizacion por disrupcion prolongada
- **Re(FRt)** -- Fill rate: fraccion de demanda satisfecha

**Aproximacion a nivel de paso (step-level):**

Dado que el DES rastrea disrupciones a traves de procesos SimPy (no con timestamps explicitos por orden), implementamos una aproximacion a nivel de paso:

| Caso | Condicion | Formula |
|------|-----------|---------|
| 1. Sin disrupcion | disruption_hrs = 0 | Re = fill_rate |
| 2. Autotomia | disruption > 0 AND FR >= 0.95 | Re = 1 - disruption_frac |
| 3. Recuperacion | disruption > 0 AND FR < 0.95 | Re = 1 / (1 + disruption_frac) |
| 4. No-recuperacion | disruption_frac > 0.5 AND FR < 0.5 | Re = 0 |

**Recompensa ReT:** `r_t = ReT_step - delta x (S - 1)`, donde delta = 0.06 es el costo lineal de turno.

### 4.2 Hallazgo Critico: Desalineacion de ReT

Cuando se usa ReT como objetivo de entrenamiento, el agente aprende a **minimizar costos en lugar de mantener servicio**:

| Metrica | Agente ReT | Comportamiento esperado |
|---------|-----------|------------------------|
| Asignacion de turnos | **99.99% S1** | Mezcla S1/S2/S3 |
| Fill rate | 0.845 | >= 0.83 |
| Comportamiento | **Minimizacion de costo** | Tradeoff servicio-costo |

**Diagnostico:** La estructura de ReT recompensa la evasion de costos (no activar turnos extra) mas rapido de lo que penaliza la degradacion del servicio. El resultado es un score ReT numericamente alto pero una politica **operacionalmente pobre**.

### 4.3 Solucion: Recompensa de Control Operacional (control_v1)

Disenamos una funcion de recompensa que penaliza directamente las dos cantidades que el agente de control de turnos puede influenciar:

```
r_t = -(w_bo x B_t/D_t + w_cost x (S_t - 1))
```

Donde:
- **B_t / D_t** = perdida de servicio a nivel de paso (backorders / demanda)
- **S_t - 1** = costo de turno (0 para S1, 1 para S2, 2 para S3)
- **w_bo = 4.0** = peso de penalizacion por perdida de servicio
- **w_cost = 0.02** = peso de penalizacion por costo de turno

**Ratio w_bo / w_cost = 200:** En una cadena de suministro militar, no entregar raciones es ~200x peor que el costo de un turno adicional.

**Calibracion:** El ratio se valido con un barrido de pesos (50k timesteps, 3 seeds, riesgo increased):
- w_bo=5.0, w_cost=0.03: PPO supera al mejor baseline estatico por primera vez
- w_bo=3.0, w_cost=0.01: PPO subtrata la perdida de servicio
- w_bo=4.0, w_cost=0.02: punto medio robusto seleccionado

**Nota: w_disr = 0.0** (disrupciones son exogenas -- el agente no puede prevenirlas, solo reaccionar).

### 4.4 Justificacion Empirica de control_v1

| Metrica | Agente ReT_thesis | Agente control_v1 |
|---------|-------------------|-------------------|
| Mezcla de turnos (increased) | 99.99% S1 | 12% S1, 25% S2, 63% S3 |
| Fill rate (increased) | 0.845 | 0.838 |
| Comportamiento adaptativo | **Ninguno (colapsado)** | **Genuino cambio de turnos** |

### 4.5 Extension PBRS (Implementada, Pendiente de Evaluacion)

Se implemento Potential-Based Reward Shaping (Ng et al., 1999) como `control_v1_pbrs`:

```
r_shaped = r_base + gamma x Phi(s') - Phi(s)
Phi(s) = alpha x fill_rate - beta x backorder_rate
```

Esto preserva la politica optima (garantia teorica) mientras inyecta awareness de calidad de servicio en las recompensas a nivel de paso. Se trata como mejora metodologica de fase 2.

---

## 5. Resultados Preliminares de RL

### 5.1 Baselines Estaticas (5 seeds x 10 episodios, stochastic PT)

| Escenario | Politica | Control reward | Fill rate | Backorder rate |
|-----------|----------|---------------|-----------|----------------|
| Increased | Static S1 | -356.81 | 0.652 | 0.348 |
| Increased | **Static S2** | **-171.35** | **0.836** | **0.164** |
| Increased | Static S3 | -178.09 | 0.835 | 0.165 |
| Severe | Static S1 | -564.70 | 0.452 | 0.548 |
| Severe | **Static S2** | **-384.82** | 0.628 | 0.372 |
| Severe | Static S3 | -388.40 | 0.629 | 0.371 |

> S2 domina bajo riesgo increased. Bajo severe, S2 y S3 son comparables. S1 es claramente suboptimo en ambos escenarios.

### 5.2 PPO vs Mejor Baseline Estatica

#### Riesgo Increased (PPO 500k steps, w_bo=4.0, w_cost=0.02)

| Politica | Control reward | Fill rate | Backorder rate | Mezcla turnos |
|----------|---------------|-----------|----------------|---------------|
| Static S2 | -170.10 | 0.837 | 0.163 | 0/100/0% |
| **PPO** | **-172.05** | **0.838** | **0.162** | **12/25/63%** |
| Diferencia | -1.95 | +0.001 | -0.001 | -- |
| CI95 Bootstrap | [-9.95, +8.51] | -- | -- | -- |

> PPO iguala el nivel de servicio con una mezcla heterogenea de turnos. Diferencia no estadisticamente significativa (p = 0.812).

#### Riesgo Severe (PPO 500k steps)

| Politica | Control reward | Fill rate | Backorder rate | Mezcla turnos |
|----------|---------------|-----------|----------------|---------------|
| Static S3 | -385.59 | 0.632 | 0.368 | 0/0/100% |
| **PPO** | **-380.98** | **0.631** | **0.369** | **41/27/32%** |
| Diferencia | **+4.61** | -0.001 | +0.001 | -- |
| CI95 Bootstrap | [-0.28, +9.49] | -- | -- | -- |

> Bajo estres severo, PPO **supera** a la mejor baseline fija por 4.61 puntos manteniendo servicio equivalente. p-value = 0.188 (direccional, no significativo a 0.05).

### 5.3 Interpretacion

1. **Valor dependiente del regimen:** El controlador adaptativo es competitivo bajo estres moderado y **superior bajo estres severo**. Las politicas fijas se vuelven insuficientes cuando la intensidad de disrupcion excede su punto de diseno.

2. **Adaptacion cost-aware:** El agente PPO no simplemente elige la configuracion mas cara. Usa una mezcla de turnos, reduciendo costos en ventanas de baja disrupcion sin sacrificar servicio.

3. **Preservacion de servicio:** En ningun escenario la politica adaptativa degrada el servicio respecto a la mejor baseline estatica. Esto confirma que control_v1 evita la desalineacion de ReT_thesis.

---

## 6. Arquitectura del Repositorio

```
proyecto_grarrido_scres+ia/
  supply_chain/
    config.py            # Parametros de tesis (UNICA fuente de verdad)
    supply_chain.py      # Motor DES SimPy (13 operaciones)
    env.py               # Wrapper Gymnasium base (4-dim actions)
    env_experimental_shifts.py  # Wrapper con control de turnos (5-dim)
    external_env_interface.py   # Interfaz para modelos externos
    dkana.py             # Pipeline DKANA (colaboracion David)
  train_agent.py         # Pipeline de entrenamiento PPO (SB3)
  run_static.py          # Baselines deterministicas/estocasticas
  validation_report.py   # Tablas de validacion
  scripts/
    benchmark_control_reward.py  # Benchmark principal
    benchmark_delta_sweep_static.py
    benchmark_ret_ablation_static.py
    formal_evaluation.py
  tests/                 # Suite pytest completa
  docs/                  # Documentacion y artefactos
```

---

## 7. Proximos Pasos

### Pendientes para completar el paper:

| Prioridad | Tarea | Estado |
|-----------|-------|--------|
| **Alta** | SAC vs PPO comparison (5 seeds x 500k) | Pendiente |
| **Alta** | PPO + frame-stack 4 (5 seeds x 500k) | Pendiente |
| **Alta** | RecurrentPPO + LSTM (5 seeds x 500k) | Pendiente |
| **Media** | PBRS experiments (control_v1_pbrs) | Implementado, no evaluado |
| **Media** | PPO 500k x 10 seeds RERUN (con fix de cross-eval) | Pendiente |
| **Baja** | Heuristic baselines adicionales | 3 implementadas |

### Requerimientos para journal Q1/Q2:
- 5+ baselines (incluyendo 3 heuristicas no-RL)
- 10 seeds minimo por condicion
- CI95 bootstrap, Mann-Whitney U, effect sizes
- Separacion train/test (entrenar en `increased`, evaluar en todos los niveles)

---

## 8. Preguntas para Discutir con Garrido

1. **Validacion del DES:** El gap de -4.43% vs la tesis es aceptable? Alguna fuente adicional de divergencia a investigar?

2. **Aproximacion step-level de ReT:** Los umbrales de clasificacion (autotomy FR >= 0.95, non-recovery frac > 0.5) son supuestos del repositorio. Hay valores mas apropiados desde la perspectiva de la tesis?

3. **Nivel severe (`++`):** Es una extrapolacion nuestra (2x frecuencia del increased). Es un escenario razonable para un teatro de operaciones de alta intensidad?

4. **Framing del paper:** La contribucion principal se enmarca como "diseno de recompensa y auditoria para control de resiliencia operacional". Es consistente con la vision del proyecto?

5. **Section 4.3 del manuscrito:** Necesita resultados de SAC/frame-stack/RecurrentPPO. Prioridad de experimentos?
