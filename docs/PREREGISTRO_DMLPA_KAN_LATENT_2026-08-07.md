# Preregistro — la arquitectura de David, corrida como toca: ¿la KAN del `latent_rw` ayuda?

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_dmlpa_kan_latent_v1.py`.
Semillas 9491–9495, de desarrollo y ya abiertas. **Cero semillas nuevas.**

## 1. De quién es el defecto, con la evidencia

El cuaderno que le mandamos (`david/kan-lab:scripts/build_david_kan_lab_notebook.py:404`) declara

```python
hidden_dim=100, nhead=12, num_layers=2, ff_mult=4, use_kan=False
```

y la celda que David ejecutó declara `use_kan=True`. **Su DMLPA lleva la KAN por diseño y nosotros
se la habíamos apagado en silencio; él la repuso.** El defecto de origen es nuestro.

Lo que sí rompió eso: `DMLPA` y `DMLPA_KAN` pasaron a ser **el mismo objeto** —su
`parameter_matching` los da idénticos, 225.410 parámetros y 12,71 % de desviación— y la etiqueta
`DMLPA` de su corrida dejó de significar lo mismo que la nuestra. Y **este repositorio nunca tuvo la
rama KAN**: `run_architecture_bakeoff_v1.DMLPA` estaba cableada al MLP. Lo que hemos medido como
«DMLPA» es su arquitectura **sin** su KAN.

## 2. La pregunta, que sigue abierta

> **A presupuesto de parámetros igualado, ¿poner una KAN en el `latent_rw` del transformer de David
> cambia el ReT frente a poner un MLP?**

## 3. El diseño de un solo factor

David comparó, sin querer, dos cosas a la vez: `features_dim` 60 contra 84 **y** el `latent_rw`.
Aquí se fija el transformer entero —`features_dim = 84`, `nhead = 12`, `num_layers = 2`,
`ff_mult = 4`— y se ajusta **sólo `hidden_dim`**, que no tiene la restricción de divisibilidad:

| brazo | `latent_rw` | `hidden_dim` | parámetros | desviación |
|---|---|---:|---:|---:|
| `dmlpa_mlp` | `Linear→GELU→Linear` | 152 | 200.052 | **0,03 %** |
| `dmlpa_kan` | `KAN([101, h, 84], grid=3, k=3)` | 10 | 199.082 | **0,46 %** |

Los dos dentro del **0,5 %**, muy por debajo de nuestra tolerancia del 10 % — y de la del 30 % que
David tuvo que usar porque la rejilla de múltiplos de 12 no dejaba otra.

**Un hecho que ya es resultado antes de entrenar:** a presupuesto igual, la KAN sólo permite
`hidden_dim = 10` contra 152 del MLP. Las aristas de una KAN son caras; el ancho que compran es
quince veces menor.

## 4. Lo que este experimento NO cubre, declarado ahora

La versión de David difiere de la nuestra en **dos** sitios, no uno. El otro es el orden de la
normalización:

* nuestra: `pre_norm(latent_rw(x) + pos)`
* la suya: `pre_norm(latent_rw(x)) + pos`

**Aquí sólo se prueba el `latent_rw`.** El orden de normalización queda sin examinar, y cambiarlo
además haría que el contraste midiera dos cosas a la vez — exactamente el defecto que este
preregistro corrige.

## 5. Diseño y regla de lectura, fijadas ahora

100.000 pasos, 5 semillas (9491–9495), 24 episodios de evaluación con `seed0 = 777_000`,
hiperparámetros de PPO idénticos a los del bake-off (`n_steps 512`, `ent_coef 0,01`,
`learning_rate 3e-4`, `batch_size 64`), 8 entornos, CPU.

**Sobre el arnés reparado**: `results/determinism_repair_control/` cerró en
`DETERMINISM_REPAIRED_SEED_IS_A_REPLICATION_UNIT_AGAIN`, así que por primera vez la semilla es
unidad de réplica y este contraste es pareable.

Primaria: ReT medio, contraste pareado por semilla `kan − mlp`, bootstrap sobre semillas.

| resultado | veredicto |
|---|---|
| `LCB95 > 0` | **`KAN_LATENT_HELPS`** |
| `UCB95 < 0` | **`KAN_LATENT_HURTS`** |
| el IC cruza cero | **`KAN_LATENT_INDISTINGUISHABLE`** — y entonces, por la regla de eficiencia ya congelada, **gana la más barata por decisión**, que se mide |

## 6. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_parameters_are_matched_within_our_tolerance` | falla si algún brazo se sale del 10 %. Comparar capacidades distintas mide capacidad, no arquitectura |
| `f2_only_the_latent_rw_differs` | comprueba que `features_dim`, `nhead`, `num_layers` y `ff_mult` coinciden. Falla si el contraste arrastra un segundo factor |
| `f3_the_two_arms_are_behaviourally_distinct` | huellas de salida sobre una entrada fija; si coinciden, se entrenó dos veces el mismo modelo |
| `f4_the_harness_reproduces` | re-entrena una celda dos veces y exige el mismo ReT. **Falla si la reparación de determinismo no se sostiene a este presupuesto**, y entonces el contraste pareado no vale |
| `f5_no_new_seeds` | 9491–9495, de desarrollo |

**Alcance:** desarrollo. No adjudica el manuscrito, no autoriza confirmación —no quedan bloques
vírgenes— y no reabre C1.
