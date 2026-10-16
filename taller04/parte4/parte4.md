# Parte 4 — Ajuste fino y monitoreo

Cifras: `taller04/tablas/tablas.md` (generadas de los CSV y los spans). Corrida base: 6 tareas, 53/53
comprobaciones automáticas, 35/35 del juez, 19/19 prohibidos ausentes.

## 4.a — ¿Hace falta ajuste fino?

No. Ninguno de nuestros fallos está en la rama de «ajustar pesos» del árbol de la sesión 19
(mejor prompt → recuperar lo que falta → ajustar pesos):

1. **S1 (T4, T5) y C (T3): llamadas del programador vacías por longitud.** El modelo razonó hasta el
   tope de 32 768 tokens sin escribir el script (573 s y 65 536 tokens perdidos solo en S1). Es control
   del razonamiento: **prompt** (la regla «razona lo justo» de la v2) y configuración del tope, no
   conocimiento que le falte al modelo.
2. **S1/T5 rechazada por el crítico.** Reconstruyó el DataFrame porque «no encontró un parquet previo»:
   le **faltaba información** en su entrada (rama «recuperar lo que falta»: qué artefactos de la T3 recibe
   la T5), y la regla «carga lo que ya produjo una subtarea previa».
3. **H (T1, T3): dos scripts rotos** (un `NameError`, un `assert` mal planteado). El bucle crítico →
   programador ya los corrigió en el intento siguiente; es un error de programación sin patrón que
   unos pesos nuevos arreglen mejor que la regla «relee el script» del **prompt**.

La calidad ya es la máxima que el golden mide. Un ajuste fino añadiría datos que no tenemos, un
entrenamiento y un modelo propio que mantener, sin un fallo que corregir. Además, con 20+ casos
no se podría distinguir una mejora pequeña del azar.

**Si decidiéramos ajustar.** Modelo de pesos abiertos de la tabla: `open_weight_razonamiento`
(gpt-oss:20b, Ollama). Datos: **preferencias** (pares rechazado/aprobado del crítico para la misma
subtarea, sacados de `traza.jsonl`). La base solo da 3 pares (S1/T5, H/T1, H/T3), y harían falta
cientos: habría que correr el solver sobre muchas tareas, y los scripts aprobados (23) servirían
para un ajuste supervisado previo. Matrices: Hu et al. (2021, Tabla 5 y §7.1) obtienen el mejor
resultado, con un presupuesto fijo, adaptando **Wq y Wv con r = 4**. Dettmers et al. (2023, §4)
muestran que esa práctica estándar no replica el ajuste completo y que hace falta LoRA en **todas
las capas lineales**: con QLoRA (NF4 + doble cuantización) un modelo de 20B cabe en una sola GPU.
Sabríamos que mejoró con el mismo golden ampliado, los mismos verificadores y varias repeticiones:
menos vacías por longitud, menos rechazos y menos tokens del programador, sin perder ninguna de las
53 comprobaciones.

## 4.b — Tres alertas

Umbrales sacados de la corrida base (tabla «Mediciones para los umbrales»). Con 6 corridas, cada una
vale 16,7 % de cualquier tasa: por eso la tercera alerta se calcula sobre una ventana más larga.

| # | Alerta | Umbral (y de dónde sale) | Ventana | Qué miro primero en los spans |
|---|---|---|---|---|
| 1 | Latencia de una corrida | > **1 800 s**. Base: p50 469 s, p95 1 534 s, máx. 1 759 s (S1); el umbral queda sobre el máximo medido, así que no salta con una corrida normal | Cada corrida, al cerrar su span `corrida`; resumen diario | `agent:programador` y sus `generation` con `fin=length` (vacías por longitud): en la base explican 573 s de los 1 759 de S1 |
| 2 | Tokens por corrida | > **350 000** (70 % del presupuesto de 500 000 del freno). Base: p50 104 264, máx. 334 343 (S1) | Cada corrida | Las `generation` del programador con salida de 32 768 tokens (el tope) y sus reintentos; luego el redactor (14 % de los tokens) |
| 3 | Disparos de guardrails de entrada (`entrada_inyeccion` redactada o `guarda_ast` bloqueada) | > **10 %** de las corridas (más de 2 de 20). Base: 0 disparos en las 4 tareas sin ataque y 0/5 falsos positivos fuera del banco; con 20 corridas la resolución es 5 %, por debajo del umbral | Ventana móvil de 20 corridas | El `metadata.retiradas` del span `entrada_inyeccion` (el fragmento retirado): ¿un ataque real o un falso positivo sobre una orden legítima? |
