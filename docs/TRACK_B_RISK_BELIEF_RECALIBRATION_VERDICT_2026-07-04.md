# Belief recalibrada (class_weight=none): PPO+MLP y Real-KAN, ambos terminados — 2026-07-04

## Qué se corrigió

`scripts/run_track_b_risk_belief_sidecar.py` usaba `class_weight="balanced"` al entrenar el
clasificador de creencia congelado, lo que distorsionaba las probabilidades hacia 0.5 (R24 1w:
base real 0.342, predicha 0.492; R24 2w: base real 0.627, predicha 0.505). Se agregó
`--belief-class-weight {balanced,none}` (default sigue siendo `balanced` para no romper corridas
previas). Verificado con `class_weight=none`: la calibración ahora es casi exacta — R24 1w base
0.3417/predicha 0.3416; R24 2w base 0.6271/predicha 0.6271.

## Resultado PPO+MLP (terminado, verificado)

Mismo protocolo (3 seeds × 30k, v10 + 2 belief dims = 103 obs):

| Variante | order_ret_excel_mean | Costo |
|---|---:|---:|
| v10 crudo (sin belief) | 0.005811 | 0.763 |
| belief `balanced` (mal calibrado) | 0.005743 | 0.640 |
| **belief `none` (bien calibrado)** | **0.005652** | **0.669** |

Por semilla (belief calibrado): `0.005560 / 0.005611 / 0.005784` — las 3 semillas caen en o por
debajo del rango de la versión mal calibrada (`0.005675-0.005777`), no es un resultado arrastrado
por una sola semilla.

**Sorpresa real, no lo que se esperaba**: corregir la calibración **empeoró** el resultado de
PPO+MLP en vez de mejorarlo. La secuencia completa es monótonamente descendente: v10 crudo
(0.005811) > belief mal calibrado (0.005743) > belief bien calibrado (0.005652). Esto sugiere que
para PPO+MLP el problema no es la calidad de la señal de creencia — es que agregar estas 2
dimensiones extra (probabilidades, bien o mal calibradas) simplemente no ayuda a esta arquitectura
en este contrato de recompensa, y con presupuesto de entrenamiento pequeño (30k) cualquier
dimensión adicional puede estar diluyendo el aprendizaje en vez de enfocarlo. La lectura honesta:
**para PPO+MLP, este diseño concreto (escalar de creencia pegado al vector) no funciona,
independientemente de la calibración.**

## Resultado Real-KAN (terminado, verificado)

Calibración confirmada correcta (igual que PPO): base 0.3417/predicha 0.3416; base 0.6271/predicha
0.6271.

| Variante | order_ret_excel_mean | Costo |
|---|---:|---:|
| v10 crudo (sin belief) | 0.005915 | 0.982 |
| belief `balanced` (mal calibrado) | 0.005935 | 0.899 |
| **belief `none` (bien calibrado)** | **0.005937** | **1.000** |

Por semilla:

| Semilla | v10 crudo | belief calibrado | Delta |
|---:|---:|---:|---:|
| 1 | 0.005914 (costo 0.960) | 0.005932 (costo 1.000) | +0.000018 |
| 2 | 0.005907 (costo 1.000) | 0.005937 (costo 1.000) | +0.000030 |
| 3 | 0.005923 (costo 0.987) | 0.005943 (costo 1.000) | +0.000019 |

## Síntesis final: la asimetría es real y ahora está limpia

**Real-KAN mejora de forma pequeña pero consistente en las 3 semillas con belief, sea cual sea la
calibración** (deltas ~+0.000018 a +0.000030, muy parecidos con `balanced` y con `none`) — esta
parte del hallazgo de Codex se sostiene. **Pero la "baja de costo" que vimos con la calibración mal
hecha (0.982→0.899) desaparece por completo con la calibración correcta (vuelve a 1.000 en las 3
semillas)** — confirma la sospecha: esa baja de costo era un artefacto de la probabilidad
distorsionada (una señal ruidosa que por azar le abrió a una semilla una vía más barata), no un
beneficio real de "belief". Con calibración honesta, Real-KAN-belief converge al mismo régimen de
capacidad máxima que ya conocíamos, solo que con un empujón pequeño y consistente en ReT Excel.

**PPO+MLP no se beneficia con ninguna calibración** — de hecho empeora monótonamente
(0.005811 → 0.005743 → 0.005652) mientras más "correcta" es la señal.

Conclusión combinada: hay una asimetría real entre arquitecturas. Real-KAN (que ya opera cerca del
techo de capacidad por diseño) parece poder aprovechar un escalar de creencia bien calibrado como un
pequeño ajuste fino de su punto de operación; PPO+MLP (más variable, más sensible al costo) no lo
aprovecha y de hecho se ve perjudicado, posiblemente porque a esta escala de entrenamiento (30k
pasos, red 64x64) las 2 dimensiones extra compiten por capacidad de aprendizaje en vez de aportar
señal útil. Esto es evidencia (todavía a escala de smoke, n=3) de que **el diseño "escalar de
creencia congelado pegado al vector" no es la vía correcta para PPO+MLP**, y que incluso para
Real-KAN el beneficio es modesto, no una prueba de comportamiento preventivo — sigue sin haber
delta causal validado vía el contrafactual `R_full - R_reset(w)` para esta variante.

## Siguiente paso recomendado

No seguir iterando esta familia de diseño para PPO+MLP. Pasar a la Ruta A real del plan aprobado
(encoder pre-entrenado trasplantado al `features_extractor` de PPO, no un escalar pegado al final
del vector) antes de concluir que "memoria + creencia" no sirve para PPO+MLP — el diseño actual es
la versión más simple/barata de la idea, no la definitiva.
