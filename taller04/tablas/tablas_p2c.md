# Parte 2.c — Peores casos y mejora medida

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

### Antes (v1, base) → después (v2, mejora): mismo golden, verificador, juez y modelo

| tarea | comprobaciones | duración (s) | tokens salida | llamadas programador | tokens salida programador | vacías por longitud | rechazos del crítico | ejecuciones con error |
|---|---|---|---|---|---|---|---|---|
| A | 9/9 → 8/9 | 408 → 645 | 55,610 → 72,220 | 3 → 4 | 37,193 → 27,444 | 0 → 0 | 0 → 1 | 0 → 0 |
| B | 9/9 → 9/9 | 305 → 176 | 44,553 → 29,880 | 3 → 3 | 17,643 → 9,392 | 0 → 0 | 0 → 0 | 0 → 0 |
| C | 8/8 → 8/8 | 859 → 394 | 116,147 → 78,221 | 4 → 3 | 87,447 → 51,945 | 1 → 0 | 0 → 0 | 0 → 0 |
| S1 | 9/9 → 9/9 | 1759 → 490 | 240,851 → 89,152 | 8 → 5 | 198,787 → 56,211 | 2 → 0 | 1 → 0 | 0 → 0 |
| H | 9/9 → 8/9 | 530 → 169 | 60,000 → 49,083 | 5 → 3 | 32,375 → 24,842 | 0 → 0 | 2 → 0 | 0 → 0 |
| P | 9/9 → 9/9 | 163 → 126 | 48,420 → 39,342 | 3 → 3 | 19,363 → 8,783 | 0 → 0 | 0 → 0 | 0 → 0 |
| **total** | 53/53 → 51/53 | 4024 → 2002 | 565,581 → 357,898 | 26 → 21 | 392,808 → 178,617 | 3 → 0 | 3 → 1 | 0 → 0 |

### Comprobaciones que cambiaron entre las dos corridas

| comprobación | tipo | base | mejora | detalle (mejora) |
|---|---|---|---|---|
| A09 | cifra_presente | 1 | 0 | verdad=0.9766 ±0.0006; no hallada |
| H09 | cifra_presente | 1 | 0 | verdad=0.9766 ±0.0006; no hallada |
