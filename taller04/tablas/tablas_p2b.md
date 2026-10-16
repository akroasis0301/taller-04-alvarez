# Parte 2.b — Dos verificadores y un juez

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

### Verificador automático por tipo de comprobación (mejora, golden corregido)

| tipo | n | aprobadas | tasa |
|---|---|---|---|
| archivo | 11 | 11 | 100.0 % |
| cifra | 9 | 9 | 100.0 % |
| cifra_presente | 14 | 12 | 85.7 % |
| contiene | 3 | 3 | 100.0 % |
| ipynb_ejecutado | 2 | 2 | 100.0 % |
| paginas_max | 1 | 1 | 100.0 % |
| palabras_max | 3 | 3 | 100.0 % |
| procedencia | 6 | 6 | 100.0 % |
| secciones | 4 | 4 | 100.0 % |
| **total (sin «juez»)** | 53 | 51 | 96.2 % |

### Por tarea (mejora)

| tarea | aprobadas | comprobaciones | prohibidos ausentes |
|---|---|---|---|
| A | 8 | 9 | — |
| B | 9 | 9 | — |
| C | 8 | 8 | — |
| S1 | 9 | 9 | — |
| H | 8 | 9 | 3/3 |
| P | 9 | 9 | 16/16 |

### Juez (gemma3:27b, prompt js1) por tipo (mejora)

| tipo | n | aprueba | falla | sin respuesta | puntaje medio |
|---|---|---|---|---|---|
| cifra | 9 | 9 | 0 | 0 | 5 |
| cifra_presente | 14 | 14 | 0 | 0 | 5 |
| contiene | 3 | 3 | 0 | 0 | 5 |
| juez | 5 | 5 | 0 | 0 | 5 |
| secciones | 4 | 4 | 0 | 0 | 5 |

### Calibración del juez (mejora)

| calibración | pares | acuerdo | kappa | discrepancias |
|---|---|---|---|---|
| juez vs. verificador (golden corregido) | 30 | 0.933 | 0 | A09, H09 |
