# Enmienda — signo, SESOI, horizonte, y un eje que el código no soporta

**Enmienda a** `docs/PREREGISTRO_GARRIDO_R2_RANDOMIZED_BENCHMARK_V1.md` y a
`docs/ENMIENDA_ALCANCE_R1_R3_BENCHMARK_2026-08-08.md`, escrita **antes** del runner. Corrige tres
defectos que un auditor externo señaló y añade un cuarto que apareció al ir a implementar.

## 1. El signo del contraste estaba invertido

`L*` es una **pérdida**: menor es mejor. Escribí la ventaja como `(KAN − MLP)`, que es **positiva
cuando KAN pierde**. Correcto:

```
A_e = E[ L*_MLP − L*_KAN ]        ventaja de KAN en el entorno e, positiva = KAN mejor
Δ   = A_R2estresado − A_baseline  la interacción, positiva = el estrés R2 favorece a KAN
```

## 2. El SESOI del 5 % no estaba definido

Garrido dijo «5–10 %» sin unidad, y yo lo copié sin fijarla, así que no existía. Queda:

> **SESOI = reducción relativa del 5 % de `L*` respecto del brazo comparado**, es decir
> `A_e / E[L*_MLP] ≥ 0,05`.

Relativa y no en puntos absolutos porque `L*` recorre 0,23–0,73 entre celdas, y cinco puntos
significan cosas muy distintas en cada extremo. La interacción `Δ` se reporta en la misma escala
relativa.

## 3. El horizonte no permite hablar de R21

Medido en `results/lever_redundancy_diagnostic/result.json` (sello `81085aa7…`), eventos máximos
por episodio: **R21 = 1,00** · R22 = 5,08 · R23 = 2,17 · **R24 = 32,08**.

La ventana de fuente de R21 llega a 16.128 h —unas 96 semanas— contra episodios de 26. **Ningún
resultado de este benchmark podrá hablar de R21**, y se declara aquí en vez de descubrirse después.
El benchmark se lee sobre **R24**, que es el riesgo con exposición suficiente y además el alineado
con capacidad.

## 4. El eje que Garrido pidió no es implementable hoy sin tocar el simulador

Fui a implementar el cambio de **familia** de distribución de R21–R24 y no existe. El soporte de
`exponential / lognormal / weibull` en `supply_chain.py:852-866` es para el **retardo de
cumplimiento**, no para la ocurrencia de riesgos, que está fijada como `uniform` o `binomial` en
`RISK_DEFINITIONS`. Cambiar la familia exige tocar el scheduler de riesgos.

**No se sustituye en silencio por un escalado paramétrico.** El brazo queda declarado:

```text
R2_PARAMETRIC_STRESS_WITHIN_SOURCE_FAMILY   IMPLEMENTED  (multiplicadores por ID, R21-R24)
R2_DISTRIBUTION_FAMILY_CHANGE               NOT_IMPLEMENTED — requiere cambio de código
```

El benchmark corre sobre el primero, y **su título y su claim dirán eso**. El segundo queda como
trabajo declarado, no como algo que se dio por hecho.

## 5. Lo que el benchmark sí puede responder

Con R1 y R3 idénticos entre brazos y sus familias congeladas, y R2 escalado por ID dentro de su
familia:

> **¿El estrés paramétrico de R2 beneficia diferencialmente a KAN frente a MLP, a paridad de
> parámetros, de interacciones con el DES y de búsqueda de hiperparámetros?**

No responde si un **cambio de familia** lo haría. Ésa es la petición literal de Garrido y queda
pendiente con su motivo técnico.

## 6. Sigue vigente

Los cuatro comparadores (MLP · KAN · regla · mejor calendario open-loop), los tres presupuestos
paramétricos emparejados al 25/50/100 %, el endpoint adimensional, la ausencia de rama `STOP`, los
falsadores fijando el `claim_status`, y **la divulgación colocada antes del resultado**: el
diagnóstico de palancas mide `M_S = 0` en las nueve celdas —los turnos no aportan nada dado el
buffer— y una clase exacta cuyo techo clarividente es `UCB95 ≤ 0,0028`. **El margen que cualquier
arquitectura puede disputar está acotado por eso, y se dice antes de medir.**
