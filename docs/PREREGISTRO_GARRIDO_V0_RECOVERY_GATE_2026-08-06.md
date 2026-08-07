# Preregistro de desarrollo — gate temporal para cerrar H1–H4 de v.0

**Estado:** escrito antes de ejecutar el gate. **Desarrollo sobre semillas ya usadas; no
confirmación y ningún resultado direccional requerido.**

## Pregunta que resuelve

Los resultados existentes cierran el lazo entre corridas de Garrido, pero no convierten por ello
los cuatro enunciados de v.0 en resultados:

- H1 exige tiempo de recuperación; `system_ttr_mean` quedó censurado al 100 % y devolvía cero por
  vacuidad.
- H2 exige una curva de mejora a través de exposiciones sucesivas, con estimando y regla de lectura.
- H3 exige varianza del **desempeño** entre intensidades o riesgos, no varianza del coste de búsqueda.
- H4 exige que el estado acumulado previo cambie el desempeño posterior, no sólo el orden de visitas.

Este gate pregunta si la física existente contiene un estimando temporal vivo antes de construir
un nuevo aprendiz o abrir semillas.

## Entorno congelado para el gate

- Ocho campañas aisladas `R11`–`R24`, una por riesgo de la tesis.
- Demanda idéntica por semilla y postura; cada riesgo se empareja con una corrida placebo que sólo
  marca la misma ventana temporal y no altera la física.
- Evento a la semana 8; horizonte 20 semanas; ventana de recuperación `tau = 8 semanas`.
- Recuperación: siete días consecutivos con servicio al menos al 95 % del basal, usando el panel
  temporal existente.
- Duraciones y magnitudes: medianas redondeadas de las cintas de desarrollo del Paso 3 ya abiertas
  el 2026-08-06. R11 y R22, riesgos frecuentes y cortos, se representan como un único clúster de
  8 y 4 eventos respectivamente. Estos valores son una extensión empírica declarada, no texto de
  la tesis.
- Seis posturas centinela fijadas antes del resultado: cero, cada nodo alto por separado, una
  postura interior y los tres nodos altos.
- Semillas `5.300.001–002`, **replay declarado** del bloque quemado `garrido_q2_des288`.

## Estimando que repara H1

Se usa tiempo restringido a restauración:

```text
RTTR_tau = min(TTR, tau)
```

Si el riesgo no causa degradación adicional frente a su placebo, `RTTR_tau = 0`: el choque fue
absorbido. Si hay degradación y no se observa restauración antes de `tau`, `RTTR_tau = tau`.
Nunca se imputa cero a una observación censurada.

La búsqueda posterior utilizará una clave escalar recuperación-primero:

```text
U = (1 - RTTR_tau/tau)
    + 1e-3 (1 - excess_service_loss_auc / (demand * tau))
    + 1e-6 flow_fill_rate
```

Los dos últimos términos están acotados y sólo desempatan: un día de TTR domina toda su suma.

## Gates, sin dirección favorable

1. `g1_shocks_have_incremental_service_effect`: al menos 6 de 8 contextos muestran mediana de
   pérdida de servicio incremental positiva o caída de servicio > 0,01 frente al placebo.
2. `g2_postures_change_recovery`: al menos 4 de 8 contextos cambian `RTTR_tau` en 24 h o cambian
   el AUC incremental normalizado en 0,01 entre posturas centinela.
3. `g3_recovery_is_observed_somewhere`: al menos 25 % de las celdas impactadas recuperan antes de
   `tau`; si no, H1 seguiría siendo casi enteramente una comparación de límites administrativos.
4. Todas las corridas tienen exactamente un clúster, comparten la cinta de demanda por semilla,
   producen endpoints acotados y declaran el replay de custodia.

Los tres gates científicos positivos autorizan construir la superficie completa de 216 posturas.
Un fallo detiene esa construcción; no autoriza cambiar duraciones, riesgos o `tau` mirando el
resultado. Una corrección de software sólo puede reparar un falsador con un mutante que reproduzca
el defecto.

## Diseño posterior, sólo si pasa

Tres ciclos balanceados de las ocho campañas. En cada campaña cada buscador dispone del mismo
presupuesto DES y despliega su incumbente. Se compararán UCB1, GP-EI, OFAT, azar, neurona lineal,
MLP y KAN; cada familia tendrá retención y, donde sea definible, ablación reset. H1 se leerá sobre
`RTTR_tau`, H2 sobre el cambio entre ciclos, H3 sobre la varianza de desempeño entre riesgos y H4
sobre retenido menos reset. El descriptor de contexto contiene sólo el riesgo declarado y sus
propiedades físicas, nunca resultados no ejecutados ni estadísticas de superficie completa.
