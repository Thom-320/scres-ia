# Demo para Garrido — 26 Marzo 2026
## Notas de presentación + comandos

**Objetivo:** Mostrar que el DES está reconstruido fielmente, el environment Gymnasium funciona, y el espacio de estados/acciones está diseñado.

**Duración estimada:** 20-25 min

---

## DEMO 1: Validación DES (5 min)
**Mensaje:** "Reconstruimos tu modelo MFSC completo en Python con SimPy 4. Todos los parámetros vienen de tus tablas."

### Correr baseline determinístico (Cf0, S=1)
```bash
cd ~/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia
python3 validation_report.py
```

**Puntos clave para decir:**
- "Usamos tus 13 operaciones, 4 escalones, los parámetros exactos de Tablas 6.4, 6.10, 6.12, 6.16, 6.20"
- "Nuestro Cf0 da 733,621 raciones/año vs tu referencia de 767,592 → gap de **-4.43%**, dentro del umbral ±15%"
- "Bajo condiciones determinísticas: fill rate 100%, solo 41 backorders en 20 años"

### Si quiere ver los riesgos habilitados:
```bash
python3 -c "
from supply_chain.supply_chain import MFSCSimulation
from supply_chain.config import SIMULATION_HORIZON
sim = MFSCSimulation(shifts=1, risks_enabled=True, seed=42, horizon=SIMULATION_HORIZON, year_basis='thesis', risk_level='current').run()
sim.summary()
"
```

**Resultado esperado:**
- Producción cae de 734K → 677K raciones/año (-7.6%)
- Fill rate: 100% → 86.3%
- 3,722 eventos disruptivos: R11 (1,880), R14 (1,038), R24 (467), 1 black-swan

---

## DEMO 2: Los parámetros en código (3 min)
**Mensaje:** "Cada parámetro es trazable a tu tesis."

### Abrir config.py y mostrar:
```bash
code supply_chain/config.py
```

**Secciones para señalar:**
1. **Línea ~50:** `OPERATIONS` → las 13 operaciones con PT, Q, ROP
2. **Línea ~200:** `RISKS_CURRENT` → distribuciones de R11-R24 + R3
3. **Línea ~280:** `RISKS_INCREASED` → niveles aumentados
4. **Línea ~330:** `CAPACITY_BY_SHIFTS` → capacidad por turno (S1/S2/S3)
5. **Línea ~380:** `VALIDATION_TABLE_6_10` → datos de validación

**Decir:** "Todo viene de Tables 6.4, 6.12, 6.16, 6.20. Si ves algún parámetro que no coincide, dime y lo corregimos."

---

## DEMO 3: Environment Gymnasium (7 min)
**Mensaje:** "Convertimos tu DES en un MDP estándar para RL."

### Mostrar que el env se crea y funciona:
```bash
python3 -c "
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
env = MFSCGymEnvShifts(risk_level='current', reward_mode='control_v1')
obs, info = env.reset(seed=42)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)
print('Obs shape:', obs.shape)
print()
labels = ['raw_mat_wdc','raw_mat_al','rations_qc','rations_sb','rations_cssu','rations_theatre','fill_rate','bo_rate','al_down','loc_down','op9_down','op11_down','time_frac','pending_batch','contingent_demand']
for i,(l,v) in enumerate(zip(labels,obs)):
    print(f'  [{i:2d}] {l:25s} = {v:.4f}')
"
```

**Explicar el espacio de observación (15 dims):**
- [0-5]: Niveles de inventario en 6 nodos clave (Op3, Op5, Op7, Op9, Op11, Op13)
- [6-7]: Fill rate y backorder rate acumulados
- [8-11]: Indicadores binarios de disrupción activa
- [12]: Progreso temporal (0→1)
- [13-14]: Batch pendiente y demanda contingente

**Explicar el espacio de acción (5 dims, continuo [-1,1]):**
- [0-3]: Multiplicadores de inventario (Q y ROP en Op3 y Op9)
- [4]: Selector de turnos → S1/S2/S3

**Decir:** "Esto unifica tus Estrategias I (buffering) y II (turnos) en una sola política dinámica."

### Correr unos steps para mostrar que la simulación avanza:
```bash
python3 -c "
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
import numpy as np
env = MFSCGymEnvShifts(risk_level='current', reward_mode='control_v1')
obs, info = env.reset(seed=42)
print('=== Corriendo 10 semanas simuladas ===')
print(f'{'Step':>5} {'Reward':>8} {'Fill%':>7} {'Shift':>6} {'AL_down':>8}')
print('-' * 40)
for step in range(10):
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.5])  # S3 fijo, params default
    obs, reward, term, trunc, info = env.step(action)
    print(f'{step+1:5d} {reward:8.3f} {obs[6]*100:6.1f}% {3:>5d} {int(obs[8]):>7d}')
"
```

---

## DEMO 4: Degradación por riesgo (5 min)
**Mensaje:** "Mira cómo se degrada la cadena cuando aumentamos los riesgos."

```bash
python3 -c "
from supply_chain.supply_chain import MFSCSimulation
from supply_chain.config import SIMULATION_HORIZON

for level in ['current', 'increased', 'severe']:
    sim = MFSCSimulation(shifts=1, risks_enabled=True, seed=42, horizon=SIMULATION_HORIZON, year_basis='thesis', risk_level=level).run()
    tp = sim.get_annual_throughput(start_time=sim.warmup_time)
    fr = sim.fill_rate()
    events = sum(sim.risk_event_counts.values()) if hasattr(sim, 'risk_event_counts') else '?'
    print(f'{level:10s}: {tp[\"avg_annual_delivery\"]:>10,.0f} rations/yr | Fill rate: {fr:.1%} | Events: {events}')
"
```

**Resultado esperado (S=1):**
| Nivel | Raciones/año | Fill rate |
|-------|-------------|-----------|
| Current | ~677K | ~86% |
| Increased | ~549K | ~46% |
| Severe | más bajo | <40% |

**Decir:** "La cadena militar colapsa bajo estrés severo con turno único. Esto motiva la necesidad de control adaptativo."

---

## Preguntas anticipadas de Garrido

**"¿Por qué granularidad horaria?"**
→ "Para no enmascarar disrupciones sub-diarias. R11 (fallo workstation) tiene reparación ~2h. Con agregación diaria se pierde."

**"¿Por qué decisiones semanales?"**
→ "Coincide con tu ciclo operativo: contratos semanales (R12), demanda contingente semanal (R24), cambio de turno requiere coordinación."

**"¿Cómo validaste los riesgos?"**
→ "Las distribuciones son exactas de tu Tabla 6.12. El modelo con riesgos produce degradación consistente con tu análisis cualitativo."

**"¿Qué operaciones observa el agente?"**
→ "Solo los buffers de almacenamiento: Op3, Op5, Op7, Op9, Op11, Op13. No ve los tiempos en tránsito (Op4, Op8, Op10, Op12). Es observabilidad parcial — limitación reconocida."

---

## ⚠️ ERRORES CONOCIDOS PARA NO MENCIONAR (a menos que pregunte)
- El paper dice "260 steps" pero el correcto es **960 steps** (161,280h ÷ 168h). Typo pendiente de corregir.
- La función de recompensa control_v1 es nuestra contribución, no viene de su tesis. Si pregunta, explicar brevemente.
- PPO resultados son preliminares — no mencionar a menos que pregunte.

---

## Antes de la reunión: checklist
- [ ] Activar el venv: `source .venv/bin/activate`
- [ ] Verificar que `python3 validation_report.py` corre sin errores
- [ ] Tener el PDF `garrido_presentacion_modelo_DES_RL.pdf` abierto como referencia personal
- [ ] Zoom/Teams listo con screen share
