import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# T3: copia del DataFrame original + columnas load_score y
# requires_review (bool), regla vectorizada:
#   requires_review = (load_score > 1.5) OR (temperature_c > 80)
# ---------------------------------------------------------------

# 1) Cargar datos originales (timestamp con parse dates)
df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

columnas_originales = list(df.columns)
original_snapshot = df.copy(deep=True)  # referencia para verificar que el original no cambia

# 2) Cargar zscores y pesos de la T2 (no se recalculan)
with open("entrada/T2/resultados.json", "r", encoding="utf-8") as f:
    t2 = json.load(f)

zscores = np.asarray(t2["zscores"], dtype=float)   # forma (300, 3)
pesos = np.asarray(t2["pesos"], dtype=float)       # [0.50, 0.30, 0.20]
assert zscores.shape[0] == len(df), "los zscores de T2 deben alinear con las filas del DataFrame"
load_score = zscores @ pesos  # vectorizado, heredado de T2
assert load_score.shape == (len(df),)

# 3) Copia del original (el original no se modifica) y agregado de columnas
df_t3 = df.copy(deep=True)
df_t3["load_score"] = load_score
df_t3["requires_review"] = (df_t3["load_score"] > 1.5) | (df_t3["temperature_c"] > 80)

# 4) Verificaciones del criterio de éxito
assert df_t3["requires_review"].dtype == bool, "requires_review debe ser bool"
assert df_t3["load_score"].dtype == float
assert list(df_t3.columns) == columnas_originales + ["load_score", "requires_review"]
# el original queda intacto (mismas columnas y mismos valores)
assert list(df.columns) == columnas_originales
assert "load_score" not in df.columns and "requires_review" not in df.columns
assert df.equals(original_snapshot)

fila = df_t3.iloc[23]
assert fila["server"] == "AI-SRV-01"
assert fila["timestamp"] == pd.Timestamp("2026-08-24 09:55:00")
assert abs(float(fila["load_score"]) - 3.136218) < 5e-7
assert abs(float(fila["temperature_c"]) - 66.97) < 1e-6
assert bool(fila["requires_review"]) is True

# 5) Guardar el parquet exigido por el enunciado (sin índice)
Path("output").mkdir(parents=True, exist_ok=True)
df_t3.to_parquet("output/server_analysis.parquet", index=False)

# 6) resultados.json (contrato de la subtarea)
resultados = {
    "subtarea": "T3",
    "supuestos": [
        "load_score se toma de los zscores producidos por la T2 (entrada/T2/resultados.json) con los pesos [0.50, 0.30, 0.20].",
        "Las columnas se agregan solo a la copia df_t3; el DataFrame original permanece sin cambios.",
        "requires_review se calcula vectorizado como (load_score > 1.5) OR (temperature_c > 80) y el DataFrame analizado se guarda sin índice en output/server_analysis.parquet.",
    ],
    "n_filas": int(len(df_t3)),
    "columnas": list(df_t3.columns),
    "load_score": [float(x) for x in df_t3["load_score"]],
    "requires_review": [bool(x) for x in df_t3["requires_review"]],
    "n_requires_review": int(df_t3["requires_review"].sum()),
    "n_por_condicion": {
        "load_score_gt_1.5": int((df_t3["load_score"] > 1.5).sum()),
        "temperature_c_gt_80": int((df_t3["temperature_c"] > 80).sum()),
    },
    "fila_23": {
        "server": str(fila["server"]),
        "timestamp": fila["timestamp"].isoformat(),
        "load_score": float(fila["load_score"]),
        "temperature_c": float(fila["temperature_c"]),
        "requires_review": bool(fila["requires_review"]),
    },
    "original_intacto": bool(df.equals(original_snapshot)),
    "requires_review_dtype": str(df_t3["requires_review"].dtype),
    "archivo_parquet": "output/server_analysis.parquet",
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# 7) Resumen
print("T3 completada.")
print(f"Filas: {len(df_t3)} | Columnas: {list(df_t3.columns)}")
print(f"Filas que requieren revisión: {int(df_t3['requires_review'].sum())} "
      f"(load_score>1.5: {int((df_t3['load_score'] > 1.5).sum())}, "
      f"temperature_c>80: {int((df_t3['temperature_c'] > 80).sum())})")
print(f"Fila 23: server={fila['server']}, timestamp={fila['timestamp']}, "
      f"load_score={fila['load_score']:.6f}, temperature_c={fila['temperature_c']}, "
      f"requires_review={bool(fila['requires_review'])}")
print(f"Original intacto: {df.equals(original_snapshot)}")
print("Parquet guardado en output/server_analysis.parquet")
