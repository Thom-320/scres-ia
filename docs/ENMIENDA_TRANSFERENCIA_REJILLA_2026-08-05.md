# Enmienda — la transferencia de rejilla 288 → 4.608

**Escrita ANTES de correr.** Runner: `scripts/run_grid_transfer_v1.py`. Opera sobre las dos cachés
selladas, bloque quemado `5.300.001–012`, réplica declarada. **Sin semillas nuevas.**

## 1. El único eje de transferencia que sigue vivo

`results/surface_gates/result.json` cerró la transferencia **entre contextos**: `H_regime`
+0,0038 contra un umbral de 0,05, con el óptimo común a los seis. Queda la transferencia **entre
rejillas**, y es la que separa a un método que aprende **la forma de la superficie** de uno que
memoriza **puntos**.

## 2. El diseño, y por qué es justo

**Entrenamiento**: cada método con memoria recorre su carrera de seis contextos sobre la rejilla de
**288**. **Transferencia**: el **mismo estado retenido** busca sobre la de **4.608**, con el mismo
presupuesto `B = 24` por contexto. **Control**: el mismo método arrancando **de cero** sobre 4.608.

El espacio de coordenadas cambia de 4 factores a 6, y ésa es exactamente la dificultad. El mapeo es
inmediato y **no inventa información**: la rejilla de 288 **es** el subgrid `op3_rm = op5_rm = 0`,
así que toda observación de entrenamiento vive en el espacio de 6 dimensiones con sus dos últimas
coordenadas en cero. Cada familia hereda lo que le corresponde y nada más:

| método | qué transfiere | qué NO sabe de los factores nuevos |
|---|---|---|
| neurona | `ρ` de 7 componentes | sus dos pesos nunca recibieron gradiente: quedan en 0 |
| UCB1 | sumas y conteos por nivel | los niveles nuevos tienen conteo 0 y se exploran primero |
| OFAT | la incumbente | los dos factores nuevos son los que le quedan por barrer |
| GP-EI | sus observaciones | el GP extrapola en las dos dimensiones nuevas |

**No se afirmará que BO no puede transferir.** El GP calentado es un brazo de pleno derecho y se
mide igual que los demás. Que un prior GP no cruce un cambio de espacio de diseño es una hipótesis
nuestra, y las hipótesis se miden.

## 3. Placebos, sin los cuales esto no vale nada

* **arranque en frío** — el mismo método sin estado retenido. Es el control primario.
* **réplica marginal** — reproduce la distribución marginal de visitas del brazo transferido
  ignorando el estado. **La decisiva**: si el brazo transferido no la bate, lo que transfirió es
  una tabla de consulta, no la forma de la superficie.
* **estado permutado** — se transfiere un `ρ` con sus componentes barajadas. Si funciona igual, lo
  que transfiere es la magnitud y no la dirección.

## 4. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_null_subgrid_reproduces_the_288_cache` | las 288 celdas con `op3_rm = op5_rm = 0` deben coincidir **bit a bit** con la caché sellada anterior. Ancla externa: la escribió una corrida previa |
| `f2_the_new_factors_move_the_endpoint` | si mover 140.000 unidades de materia prima no cambia nada, los factores son decoración — y este proyecto **ya midió** 4,56 M de unidades moviendo exactamente cero ReT |
| `f3_transfer_beats_its_marginal_replay` | ver §3 |
| `f4_budgets_are_matched` | contados del log de accesos, no afirmados |
| `f5_no_fresh_seeds` | custodia central, réplica declarada |

## 5. Reglas de lectura

* **algún método transfiere con LCB95 > 0 sobre su arranque en frío Y sobre su réplica marginal** →
  `GRID_TRANSFER_ESTABLISHED`, y el paper puede nombrar **cuál** y con qué representación.
* **transfieren varios** → se reporta el orden y, en empate, decide `Delta_efficiency`.
* **ninguno** → `NO_GRID_TRANSFER`: lo que la memoria compra es **dentro** de un espacio de diseño
  fijo, no a través de un cambio de espacio. Es un límite nítido y publicable, y cierra la última
  puerta abierta a un aprendiz.

**Nada de esto autoriza entrenar ni abrir semillas.**
