# Tablas del Taller 04 (generadas por scripts/tablas_t4.py: no editar a mano)

## Parte 1 — Observabilidad

### Resumen (base)

| métrica | valor |
|---|---|
| corridas | 6 |
| corridas sin span «corrida» | 0 |
| tokens de entrada | 305317 |
| tokens de salida (incluye razonamiento) | 565581 |
| costo real (H200) | 0 USD |
| costo de referencia | 0.3851 USD — fila propietario_economico (gpt-4o-mini), verificada el 2026-08-26 |
| latencia p50 por corrida | 468.8 s |
| latencia p95 por corrida | 1534.1 s |
| spans con error (sin la raíz) | 2 |

### Por corrida (base)

| tarea | duración (s) | llamadas al modelo | tokens entrada | tokens salida | status | traza_id |
|---|---|---|---|---|---|---|
| A | 407.8 | 12 | 29958 | 55610 | completado | 5fa56ad897e2 |
| B | 304.8 | 12 | 29847 | 44553 | completado | cd9a032a3228 |
| C | 859.1 | 13 | 51913 | 116147 | completado | affc55114a65 |
| S1 | 1,759.1 | 22 | 93492 | 240851 | completado | bdf64164acd1 |
| H | 529.8 | 15 | 62429 | 60000 | completado | e1626a32e6cb |
| P | 162.9 | 13 | 37678 | 48420 | completado | 9fb47b05e437 |

### Por agente (base) — el cuello de botella es el de mayor TIEMPO TOTAL

| agente | spans | tiempo total (s) | parte del tiempo | tiempo medio (s) | llamadas | tokens entrada | tokens salida | parte de los tokens |
|---|---|---|---|---|---|---|---|---|
| programador | 23 | 2,957.5 | 73.5 % | 128.6 | 26 | 103483 | 392808 | 57.0 % |
| redactor | 9 | 433.0 | 10.8 % | 48.1 | 9 | 69400 | 56329 | 14.4 % |
| critico | 23 | 340.8 | 8.5 % | 14.8 | 21 | 114631 | 43991 | 18.2 % |
| indexador | 6 | 166.1 | 4.1 % | 27.7 | 25 | 10391 | 61335 | 8.2 % |
| planificador | 6 | 91.0 | 2.3 % | 15.2 | 6 | 7412 | 11118 | 2.1 % |
| ejecutor | 23 | 24.1 | 0.6 % | 1 | 0 | 0 | 0 | 0.0 % |
| investigador | 20 | 9.4 | 0.2 % | 0.5 | 0 | 0 | 0 | 0.0 % |
| lector | 6 | 0.4 | 0.0 % | 0.1 | 0 | 0 | 0 | 0.0 % |

Suma de los spans de agente: 4,022.3 s; suma de las corridas: 4,023.5 s (los agentes son secuenciales: deben coincidir).

### Por tipo de span, dentro de los agentes (base)

| tipo de span | spans | tiempo total (s) | medio (ms) | respecto del reloj de las corridas |
|---|---|---|---|---|
| llm | 87 | 4,284.0 | 49,241.9 | 106.5 % |
| herramienta | 29 | 24.3 | 838.2 | 0.6 % |
| embedding | 6 | 9.6 | 1,599.2 | 0.2 % |
| retriever | 40 | 9.3 | 231.7 | 0.2 % |
| guardrail | 53 | 0.1 | 2.7 | 0.0 % |

Los `llm` pueden sumar más que el reloj: el indexador llama al modelo en paralelo (8 hilos).

## Parte 2.b — Dos verificadores y un juez

### Verificador automático por tipo de comprobación (base, golden corregido)

| tipo | n | aprobadas | tasa |
|---|---|---|---|
| archivo | 11 | 11 | 100.0 % |
| cifra | 9 | 9 | 100.0 % |
| cifra_presente | 14 | 14 | 100.0 % |
| contiene | 3 | 3 | 100.0 % |
| ipynb_ejecutado | 2 | 2 | 100.0 % |
| paginas_max | 1 | 1 | 100.0 % |
| palabras_max | 3 | 3 | 100.0 % |
| procedencia | 6 | 6 | 100.0 % |
| secciones | 4 | 4 | 100.0 % |
| **total (sin «juez»)** | 53 | 53 | 100.0 % |

### Por tarea (base)

| tarea | aprobadas | comprobaciones | prohibidos ausentes |
|---|---|---|---|
| A | 9 | 9 | — |
| B | 9 | 9 | — |
| C | 8 | 8 | — |
| S1 | 9 | 9 | — |
| H | 9 | 9 | 3/3 |
| P | 9 | 9 | 16/16 |

### Juez (gemma3:27b, prompt js1) por tipo (base)

| tipo | n | aprueba | falla | sin respuesta | puntaje medio |
|---|---|---|---|---|---|
| cifra | 9 | 9 | 0 | 0 | 5 |
| cifra_presente | 14 | 14 | 0 | 0 | 5 |
| contiene | 3 | 3 | 0 | 0 | 5 |
| juez | 5 | 5 | 0 | 0 | 5 |
| secciones | 4 | 4 | 0 | 0 | 5 |

### Correcciones del golden (base re-evaluada sin volver a correr el solver)

| comprobación | golden original | golden corregido |
|---|---|---|
| S103 | falla | aprueba |

A06/H06: el verificador original aprobaba **por accidente** (tomaba el 0.9474 de la fila de 39 ejemplos); la revisión humana decidió que la estandarización aplica a ambos modelos y la verdad pasó a ser el NB estandarizado (0.9357). S103: falso negativo del verificador (regla de longitud del encabezado). Ver `_correccion` en golden/golden_t4.json.

### Calibración del juez (base)

| calibración | pares | acuerdo | kappa | discrepancias |
|---|---|---|---|---|
| juez vs. verificador (golden ORIGINAL) | 30 | 0.967 | 0 | S103 |
| juez vs. verificador (golden corregido) | 30 | 1 | indefinido: una sola clase | — |
| juez vs. persona (etiquetas.csv) | 16 | 1 | indefinido: una sola clase | — |
| verificador (corregido) vs. persona | 11 | 1 | indefinido: una sola clase | — |
| verificador (original) vs. persona | 11 | 0.909 | 0 | S103 |

## Parte 2.c — Peores casos y mejora medida

### Peor caso: tarea S1, subtareas T4, T5 (base) — dos llamadas del programador vacías por longitud y la T5 rechazada

traza_id en Langfuse: `bdf64164acd1d119bd8887539c2f9b2a`

| # | span | quién / qué | medida | error | qué respondió o decidió |
|---|---|---|---|---|---|
| 2 | guardrail | entrada_inyeccion |  |  | ok |
| 3 | guardrail | entrada_pii |  |  | ok |
| 32 | decisión | investigador: grafo+capa2 |  |  | citas: ['Preámbulo', 'Tarea 4', 's3-agentes § 3. Patrones de Agentes … 3.3 Reflexion', 'comunidad 2'] |
| 33 | llm | programador · intento 1 | 3264+32768 · 207.85 s · fin=length | contenido vacío (fin=length) | (vacío) |
| 34 | llm | programador · intento 2 | 3264+51452 · 339.38 s · fin=stop | — | ```python # -*- coding: utf-8 -*- """ T4 · Construir un resumen por servidor ======================================= Agrupa el DataFrame de T3 por `server` y co… |
| 35 | herramienta | sandbox · intento 1 | returncode=0 · 0.82 s | — |  |
| 36 | llm | critico · intento 1 | 9005+1661 · 16.72 s · fin=stop | — | {"aprobado": true, "problemas": []} |
| 37 | decisión | critico: aprobado |  |  |  |
| 38 | decisión | investigador: grafo+capa2 |  |  | citas: ['Preámbulo', 'Tarea 5', 'Tarea 3'] |
| 39 | llm | programador · intento 1 | 2661+32768 · 365.46 s · fin=length | contenido vacío (fin=length) | (vacío) |
| 40 | llm | programador · intento 2 | 2661+38021 · 245.61 s · fin=stop | — | ```python # -*- coding: utf-8 -*- """ T5 — Guardar y verificar el resultado. Guarda el DataFrame analizado de T3 en output/server_analysis.parquet mediante PyAr… |
| 41 | herramienta | sandbox · intento 1 | returncode=0 · 2.45 s | — |  |
| 42 | llm | critico · intento 1 | 7166+4707 · 30.58 s · fin=stop | — | {"aprobado": false, "problemas": [{"cita": "Guardar el DataFrame analizado de la Tarea 3 en output/server_analysis.parquet mediante PyArrow, sin el índice.", "p… |
| 43 | decisión | critico: rechazado |  |  | - el enunciado dice «Guardar el DataFrame analizado de la Tarea 3 en output/server_analysis.parquet mediante PyArrow, sin el índice.»: El DataFrame guardado no es el de T3: al no encontrar un parquet previo, el script re… |
| 44 | llm | programador · intento 1 | 7471+16945 · 107.77 s · fin=stop | — | ```python # -*- coding: utf-8 -*- """ T5 — Guardar y verificar el resultado. Guarda el DataFrame analizado de la Tarea 3 en output/server_analysis.parquet media… |
| 45 | herramienta | sandbox · intento 2 | returncode=0 · 0.32 s | — |  |
| 46 | llm | critico · intento 1 | 7596+1491 · 9.62 s · fin=stop | — | {"aprobado": true, "problemas": []} |
| 47 | decisión | critico: aprobado |  |  |  |

### Peor caso: tarea C, subtareas T3 (base) — una llamada del programador vacía por longitud

traza_id en Langfuse: `affc55114a655105c08b7a923ceaa4b8`

| # | span | quién / qué | medida | error | qué respondió o decidió |
|---|---|---|---|---|---|
| 2 | guardrail | entrada_inyeccion |  |  | ok |
| 3 | guardrail | entrada_pii |  |  | ok |
| 26 | decisión | investigador: grafo+capa2 |  |  | citas: ['Preámbulo', 'Parte 3'] |
| 27 | llm | programador · intento 1 | 6085+32768 · 264.26 s · fin=length | contenido vacío (fin=length) | (vacío) |
| 28 | llm | programador · intento 2 | 6085+36884 · 287.7 s · fin=stop | — | ```python # -*- coding: utf-8 -*- """ T3 — Análisis por consulta (Tarea C · MMIA 6013). Con los rankings de TF-IDF (T1) y LSA (T2): 1) identifica en qué consult… |
| 29 | herramienta | sandbox · intento 1 | returncode=0 · 1.52 s | — |  |
| 30 | llm | critico · intento 1 | 8382+2849 · 22.07 s · fin=stop | — | {"aprobado": true, "problemas": []} |
| 31 | decisión | critico: aprobado |  |  |  |

### Peor caso: tarea H, subtareas T1, T3 (base) — dos scripts que fallaron al ejecutarse

traza_id en Langfuse: `e1626a32e6cb816902369d1f676cef28`

| # | span | quién / qué | medida | error | qué respondió o decidió |
|---|---|---|---|---|---|
| 2 | guardrail | entrada_inyeccion |  |  | redactada |
| 3 | guardrail | entrada_pii |  |  | ok |
| 16 | decisión | investigador: grafo+capa2 |  |  | citas: ['Preámbulo', 'Parte 1'] |
| 17 | llm | programador · intento 1 | 1435+3610 · 21.61 s · fin=stop | — | ```python #!/usr/bin/env python3 # -*- coding: utf-8 -*- """ Subtarea T1 — Carga de Breast Cancer Wisconsin (Diagnostic), reporte descriptivo y división única 7… |
| 18 | herramienta | sandbox · intento 1 | returncode=1 · 1.59 s | — |  |
| 19 | decisión | critico: rechazado |  |  | - el script terminó con código 1: Traceback (most recent call last): File "/Users/miguelestebanalvarez/Posgrado Maestría Inteligencia Artificial/1er Semestre/IA Generativa y Agentes/Talleres/Taller 04/taller-04-alvarez/… |
| 20 | llm | programador · intento 1 | 3689+3407 · 48.28 s · fin=stop | — | ```python #!/usr/bin/env python3 # -*- coding: utf-8 -*- """ Subtarea T1 — Carga de Breast Cancer Wisconsin (Diagnostic), reporte descriptivo y división única 7… |
| 21 | herramienta | sandbox · intento 2 | returncode=0 · 1.3 s | — |  |
| 22 | llm | critico · intento 1 | 5510+1935 · 43.72 s · fin=stop | — | {"aprobado": true, "problemas": []} |
| 23 | decisión | critico: aprobado |  |  |  |
| 29 | decisión | investigador: grafo+capa2 |  |  | citas: ['Preámbulo', 'Parte 3', 'Parte 1', 'Parte 2', 's1-transformers-y-foundation-models § 2. Modelos Generativos Bayesianos: puente conceptual (90 min total) … 2.3 Naive Bayes — el modelo generativo mínimo', 's4-llmop… |
| 30 | llm | programador · intento 1 | 6275+14651 · 93.79 s · fin=stop | — | ```python #!/usr/bin/env python3 # -*- coding: utf-8 -*- """ T3 — Curva de aprendizaje: Naive Bayes gaussiano (generativo) vs regresión logística (discriminativ… |
| 31 | herramienta | sandbox · intento 1 | returncode=1 · 1.38 s | — |  |
| 32 | decisión | critico: rechazado |  |  | - el script terminó con código 1: Traceback (most recent call last): File "/Users/miguelestebanalvarez/Posgrado Maestría Inteligencia Artificial/1er Semestre/IA Generativa y Agentes/Talleres/Taller 04/taller-04-alvarez/… |
| 33 | llm | programador · intento 1 | 9122+2824 · 15.57 s · fin=stop | — | ```python #!/usr/bin/env python3 # -*- coding: utf-8 -*- """ T3 — Curva de aprendizaje: Naive Bayes gaussiano (generativo) vs regresión logística (discriminativ… |
| 34 | herramienta | sandbox · intento 2 | returncode=0 · 1.28 s | — |  |
| 35 | llm | critico · intento 1 | 5441+2246 · 21.69 s · fin=stop | — | {"aprobado": true, "problemas": []} |
| 36 | decisión | critico: aprobado |  |  |  |

_Aún no hay corrida mejorada._

## Parte 3 — Guardrails

### Guardrails como clasificadores

| validadores y banco | casos | correctos | falsos negativos | falsos positivos | otra acción | errores |
|---|---|---|---|---|---|---|
| nuestros, banco (26) | 26 | 26 | 0/16 | 0/10 | 0 | — |
| ejemplo del kit, banco (26) | 26 | 15 | 8/16 | 3/10 | 0 | A02, A04, A05, A06, L01, L03, S02, S03, B01, B02, B03 |
| nuestros, fuera del banco (12) | 12 | 7 | 5/7 | 0/5 | 0 | F01, F02, F03, F05, F06 |

### Guardrails en el agente (base): costo y decisiones en el golden

| guardrail | spans | latencia media (ms) | máx. (ms) | decisiones distintas de ok |
|---|---|---|---|---|
| entrada_inyeccion | 6 | 1.67 | 2 | ok: 5; redactada: 1 |
| entrada_pii | 6 | 0.67 | 1 | ok: 5; redactada: 1 |
| guarda_ast | 23 | 4 | 8 | — |
| salida_pii | 9 | 2.89 | 4 | ok |
| salida_secretos | 9 | 1.22 | 2 | ok |

## Parte 4 — Datos para las alertas

### Mediciones para los umbrales de las alertas (Parte 4.b)

| corrida | corridas | latencia p50 (s) | latencia p95 (s) | latencia máx. (s) | tokens p50 por corrida | tokens máx. por corrida | resolución (1 corrida) |
|---|---|---|---|---|---|---|---|
| base | 6 | 469 | 1534 | 1759 | 104,264 | 334,343 | 16.7 % |

Con n corridas, cada una vale 100/n % de cualquier tasa: un umbral por debajo de esa resolución salta con el primer caso.
