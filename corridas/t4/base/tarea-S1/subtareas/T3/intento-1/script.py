# T3: Copia del DataFrame original + columnas load_score (de T2) y requires_review (bool),
# calculadas de forma vectorizada. Verificación de la fila 23 y de que el original queda intacto.

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULT = Path("resultados.json")

UMBRAL_LOAD = 1.5
UMBRAL_TEMP = 80.0

# ---------- 1) DataFrame original (no se modifica nunca) ----------
df_original = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])

# ---------- 2) load_score calculado en T2 ----------
with open(RUTA_T2, "r", encoding="utf-8") as f:
    res_t2 = json.load(f)

load_score_t2 = np.asarray(res_t2["load_score"], dtype=float)
if load_score_t2.ndim != 1 or load_score_t2.shape[0] != len(df_original):
    raise ValueError(
        f"load_score de T2 con forma {load_score_t2.shape} no coincide con "
        f"las {len(df_original)} filas del CSV."
    )

# Comprobación de alineación: recalcular load_score con medias/stds/pesos de T2
features = list(res_t2["feature_names"])
X = df_original[features].to_numpy(dtype=float)
means = np.asarray(res_t2["means"], dtype=float)
stds = np.asarray(res_t2["stds"], dtype=float)
pesos = np.asarray(res_t2["pesos"], dtype=float)
zscores_check = (X - means) / np.where(stds == 0.0, 1.0, stds)
load_score_check = zscores_check @ pesos
max_diff_load = float(np.max(np.abs(load_score_check - load_score_t2)))

# ---------- 3) Copia + columnas nuevas (todo vectorizado, sin bucles) ----------
df = df_original.copy()  # copia explícita: el original no se toca
df["load_score"] = load_score_t2
df["requires_review"] = (df["load_score"] > UMBRAL_LOAD) | (df["temperature_c"] > UMBRAL_TEMP)
df["requires_review"] = df["requires_review"].astype(bool)  # tipo bool declarado

# ---------- 4) Verificaciones ----------
def dataframes_iguales(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    return all(a[c].equals(b[c]) for c in a.columns)

df_releido = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
original_intacto = bool(
    ("load_score" not in df_original.columns)
    and ("requires_review" not in df_original.columns)
    and dataframes_iguales(df_original, df_releido)
    and dataframes_iguales(df[df_original.columns], df_original)
)

fila = df.loc[23]
checks_fila23 = {
    "server_ok": str(fila["server"]) == "AI-SRV-01",
    "timestamp_ok": bool(pd.Timestamp(fila["timestamp"]) == pd.Timestamp("2026-08-24 09:55:00")),
    "load_score_ok": bool(abs(float(fila["load_score"]) - 3.136218) <= 1e-6),
    "temperature_c_ok": bool(abs(float(fila["temperature_c"]) - 66.97) <= 1e-9),
    "requires_review_ok": bool(fila["requires_review"]) is True,
}
fila_23_coincide = bool(all(checks_fila23.values()))

regla_ok = bool(
    df["requires_review"].equals(
        ((df["load_score"] > UMBRAL_LOAD) | (df["temperature_c"] > UMBRAL_TEMP)).astype(bool)
    )
)
dtype_bool_ok = bool(df["requires_review"].dtype == bool)

# ---------- 5) Guardar el DataFrame analizado (archivo exigido por el enunciado) ----------
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(RUTA_PARQUET, engine="pyarrow", index=True)

# ---------- 6) resultados.json (contrato de la subtarea) ----------
resultados = {
    "subtarea": "T3_dataframe_con_load_score_y_requires_review",
    "n_filas": int(df.shape[0]),
    "n_columnas": int(df.shape[1]),
    "columnas": list(df.columns),
    "columnas_nuevas": ["load_score", "requires_review"],
    "requires_review_dtype": str(df["requires_review"].dtype),
    "regla": "requires_review = (load_score > 1.5) | (temperature_c > 80)",
    "umbral_load_score": UMBRAL_LOAD,
    "umbral_temperature_c": UMBRAL_TEMP,
    "vectorizado": True,
    "original_intacto": original_intacto,
    "load_score_reciclado_max_diff_vs_T2": max_diff_load,
    "fila_23": {
        "server": str(fila["server"]),
        "timestamp": str(pd.Timestamp(fila["timestamp"])),
        "load_score": float(fila["load_score"]),
        "temperature_c": float(fila["temperature_c"]),
        "requires_review": bool(fila["requires_review"]),
    },
    "fila_23_verificaciones": checks_fila23,
    "fila_23_coincide": fila_23_coincide,
    "regla_coherente_en_todas_las_filas": regla_ok,
    "requires_review_es_bool": dtype_bool_ok,
    "criterio_exito_ok": bool(fila_23_coincide and original_intacto and dtype_bool_ok),
    "n_requires_review_true": int(df["requires_review"].sum()),
    "n_requires_review_false": int((~df["requires_review"]).sum()),
    "n_load_score_mayor_1_5": int((df["load_score"] > UMBRAL_LOAD).sum()),
    "n_temperature_mayor_80": int((df["temperature_c"] > UMBRAL_TEMP).sum()),
    "n_ambas_condiciones": int(
        ((df["load_score"] > UMBRAL_LOAD) & (df["temperature_c"] > UMBRAL_TEMP)).sum()
    ),
    "requires_review": [bool(v) for v in df["requires_review"].tolist()],
    "parquet_guardado": str(RUTA_PARQUET),
    "parquet_existe": bool(RUTA_PARQUET.exists()),
}

with open(RUTA_RESULT, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------- 7) Resumen ----------
print("=== T3: DataFrame con load_score y requires_review ===")
print(f"Filas x columnas: {df.shape[0]} x {df.shape[1]} | Columnas: {list(df.columns)}")
print(f"requires_review dtype: {df['requires_review'].dtype} (bool esperado: {dtype_bool_ok})")
print(f"Fila 23: server={fila['server']}, timestamp={fila['timestamp']}, "
      f"load_score={fila['load_score']:.6f}, temperature_c={fila['temperature_c']}, "
      f"requires_review={fila['requires_review']}")
print(f"Fila 23 coincide con lo esperado: {fila_23_coincide} {checks_fila23}")
print(f"Original intacto: {original_intacto}")
print(f"requires_review=True: {resultados['n_requires_review_true']} filas "
      f"(load>1.5: {resultados['n_load_score_mayor_1_5']}, temp>80: {resultados['n_temperature_mayor_80']}, "
      f"ambas: {resultados['n_ambas_condiciones']})")
print(f"Parquet guardado en: {RUTA_PARQUET} (existe: {RUTA_PARQUET.exists()})")
print(f"Criterio de éxito global: {resultados['criterio_exito_ok']}")
