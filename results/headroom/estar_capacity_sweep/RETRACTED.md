# RETIRADO — instrumento inválido (`f2` tautológico, `f6` cableado)

`result.json` de este directorio **no es adjudicable**. Dos de sus seis falsadores no podían fallar:

* **`f2`** evaluaba `len({demanda redondeada}) >= 1`, que es verdadero por construcción. No comparaba
  contra ninguna corrida sin capacidad, así que no establecía invarianza de la demanda exógena.
* **`f6`** estaba escrito literalmente como `"passed": True`. No calculaba ningún UCB ni daño contra
  comparador alguno.

Sucesor: `results/headroom/estar_capacity_sweep_v1_1/result.json`, con ambos implementados contra el
**brazo nulo sin capacidad sobre la misma cinta**. Con `f6` de verdad, **falla** — así que el
veredicto descriptivo de este directorio (`ARGMAX_MOVES_WITHOUT_VALUE`) tampoco se sostiene tal cual.

`estar_capacity_sweep_INFLATED_H_REGIME/` sigue conservado por separado: retiene el defecto de
fórmula (clarividencia por semilla) que ya se había corregido aquí.
