# Enmienda — la rejilla extendida `wrap288_compat_extended_v1` (4.608)

**Escrita ANTES de correr.** Runner: `scripts/build_extended_surface_v1.py`.
**Semillas: ninguna nueva.** La extensión añade **configuraciones**, no cintas: se corre sobre el
mismo bloque quemado `5.300.001–012`, réplica declarada. Ésa es la razón por la que este eje **no
requiere firma de apertura**: no hay nada que abrir.

## 1. Qué abre, y por qué es el último eje vivo

`results/surface_gates/result.json` cerró la transferencia **entre contextos**: `H_regime` +0,0038,
el óptimo es común. Queda un solo eje de transferencia con contenido: **entre rejillas**. Entrenar
un buscador sobre 288 configuraciones y evaluarlo sobre 4.608 es la prueba que separa a un método
que aprende **la forma de la superficie** de uno que memoriza **puntos**.

Y es la instrucción de Garrido del 28 de julio —añadir variables de decisión, no alargar el
episodio— ejecutada donde puede pagar.

## 2. Los dos factores nuevos, y su mapeo congelado

Los buffers aguas arriba **ya están cableados** (`initial_buffers` → `op3_rm` en el WDC, `op5_rm`
en el AL). Lo único que faltaba era exponerlos como factores. Niveles, con **la misma convención
horas→unidades que ya usa `op9_rations`** (`h × 2.500 / 24`), para que la rejilla sea internamente
consistente y no una escala inventada:

```text
op3_rm ∈ {0, 17_500, 70_000, 140_000}      # 0, 168 h, 672 h, 1.344 h de materia prima en WDC
op5_rm ∈ {0, 17_500, 70_000, 140_000}      # lo mismo en AL
```

Rejilla: `6 × 3 × 4 × 4 × 4 × 4 = 4.608`. A 72 ms por episodio, `4.608 × 6 × 12 = 331.776`
episodios ≈ **6,6 h**.

**Se llama `wrap288_compat_extended_v1`, no «réplica de la Tabla 6.16».** Conserva exactamente la
semántica del runner Q2, incluido `op9_rations = buffer_hours × 2.500 / 24`. Es una extensión
compatible, y decirlo de otro modo sería sobreafirmar.

## 3. El nulo, que aquí es gratis y más fuerte que el de CSSU

El subgrid `op3_rm = op5_rm = 0` **es por definición** la superficie de 288: ambos runners ya pasan
esas claves con valor `0.0`.

> **`f1_null_subgrid_reproduces_the_frozen_288`** — para las 288 configuraciones con
> `op3_rm = op5_rm = 0`, el valor debe coincidir **bit a bit** con la caché sellada
> `results/surface_cache/wrap288_v1`, en los seis contextos y las doce semillas.
>
> *Por qué puede fallar:* si exponer los dos factores cambió cualquier cosa de la física —orden de
> consumo del RNG incluido— las 20.736 celdas dejan de coincidir. Es un ancla **externa**: la caché
> de 288 la escribió una corrida anterior, así que el chequeo no puede satisfacerse por acuerdo del
> código consigo mismo.

## 4. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_null_subgrid_reproduces_the_frozen_288` | ver §3 |
| `f2_the_new_factors_actually_move_the_endpoint` | el rango del endpoint sobre configuraciones que sólo difieren en `op3_rm`/`op5_rm` debe ser > 0 en algún contexto. *Puede fallar*: si mover 140.000 unidades de materia prima no cambia nada, los factores son decoración y hay que escribirlo — este proyecto **ya midió** que 4,56 M de unidades de materia prima movieron exactamente cero ReT |
| `f3_no_fresh_seeds` | custodia central, réplica declarada del mismo bloque |

## 5. Qué se hará con la superficie, y en qué orden

1. **Los dos gates** (`g2` separabilidad con CV dejando semilla fuera, `g1` `H_regime`) sobre la
   rejilla extendida. Son go/no-go y se corren **antes** que nada.
2. **La escalera de comparadores v2** completa sobre 4.608.
3. **La transferencia de rejilla**: cada método con memoria entrena sobre 288 y se evalúa sobre
   4.608, con su gemelo sin memoria como control. **No se afirmará que BO no puede transferir**: el
   GP con prior calentado se mide igual que los demás.

**Nada de esto adjudica, abre semillas ni autoriza entrenar.**
