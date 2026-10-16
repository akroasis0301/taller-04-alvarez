# Reporte — Procesamiento reproducible de telemetría de servidores de IA

## Tarea 1 - Representar las mediciones

La clase `ServerMeasurements` almacena la matriz NumPy `values` y la tupla `feature_names`, y valida en el constructor los tres casos exigidos: matriz no bidimensional, número de columnas distinto de la cantidad de nombres y datos no numéricos. Las tres validaciones se ejecutaron y las tres lanzan `ValueError` (no bidimensional: True; columnas distintas de nombres: True; no numéricos: True). `build_measurements(df)` selecciona `gpu_utilization`, `cpu_utilization` y `memory_gb` en ese orden y las convierte a una matriz `float64` de forma (300, 3). El `repr` obtenido es:

```
ServerMeasurements(shape=(300, 3), features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))
```

## Tarea 2 - Normalizar y calcular un indicador de carga

Sobre la matriz de la Tarea 1 se calcularon, con operaciones vectorizadas, la media por columna [57.9931, 50.5416, 29.4553] y la desviación estándar [14.4301, 12.6985, 6.5368]. Ninguna desviación resultó cero (0 de 3 columnas), por lo que la división del Z-score usó las desviaciones medidas; el resguardo contra división por cero permanece implementado. `zscores` tiene forma (300, 3) y `load_score`, calculado como `zscores @ [0.50, 0.30, 0.20]`, tiene forma (300,).

## Tarea 3 - Identificar observaciones para revisión

Se creó una copia del DataFrame original (que permanece sin modificar) y se agregaron las columnas `load_score` y `requires_review` (tipo bool) con la regla vectorizada `load_score > 1.5` o `temperature_c > 80`. El DataFrame analizado tiene 300 filas y 9 columnas: server, timestamp, gpu_utilization, cpu_utilization, memory_gb, power_w, temperature_c, load_score, requires_review. En total, 49 de las 300 observaciones quedaron marcadas con `requires_review=True`. La fila 23 es:

|    | server     | timestamp           | load_score | temperature_c | requires_review |
|----|------------|---------------------|------------|---------------|-----------------|
| 23 | AI-SRV-01  | 2026-08-24 09:55:00 | 3.1362     | 66.97         | True            |

## Tarea 4 - Construir un resumen por servidor

La agrupación por `server` (orden alfabético) produce una fila por servidor con exactamente las columnas exigidas, en este orden:

| server     | observations | mean_power_w | max_temperature_c | mean_load | review_count |
|------------|--------------|--------------|-------------------|-----------|--------------|
| AI-SRV-01  | 50           | 363.4598     | 86.50             | -0.6145   | 2            |
| AI-SRV-02  | 50           | 390.4990     | 76.47             | -0.4177   | 1            |
| AI-SRV-03  | 50           | 416.6088     | 86.50             | -0.0859   | 2            |
| AI-SRV-04  | 50           | 434.8684     | 80.86             | -0.0030   | 5            |
| AI-SRV-05  | 50           | 471.1808     | 86.50             | 0.4536    | 12           |
| AI-SRV-06  | 50           | 495.7276     | 88.01             | 0.6676    | 27           |

Cada servidor aporta 50 observaciones. La potencia media crece de 363.4598 W en AI-SRV-01 a 495.7276 W en AI-SRV-06, y `mean_load` pasa de -0.6145 a 0.6676; AI-SRV-06 concentra la mayor cantidad de revisiones (27), frente a 1 de AI-SRV-02.

## Tarea 5 - Guardar y verificar el resultado

El DataFrame analizado de la Tarea 3 se guardó en `output/server_analysis.parquet` mediante PyArrow, sin el índice. Al volver a leer el archivo, las cuatro verificaciones del round trip resultaron verdaderas:

```
Round trip verificado: output/server_analysis.parquet
Filas: True
Columnas: True
Esquema: True
Valores: True
```

En detalle: filas, 300 antes y 300 después; columnas, las mismas 9 antes y después; esquema idéntico antes y después (server: large_string; timestamp: timestamp[us]; gpu_utilization, cpu_utilization, memory_gb, power_w, temperature_c y load_score: double; requires_review: bool); y valores iguales columna a columna. El contrato de los datos se cumple y el archivo Parquet queda como entregable verificado.
