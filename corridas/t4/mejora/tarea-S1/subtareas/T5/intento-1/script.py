# T5 - Guardar el DataFrame analizado de la Tarea 3 en output/server_analysis.parquet
# con PyArrow (sin índice), releerlo y verificar el round trip:
# filas antes == filas después, columnas, esquema y valores iguales.
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_SALIDA = Path("output/server_analysis.parquet")
RUTA_REL = "output/server_analysis.parquet"
T3_DIR = Path("entrada/T3")
COLUMNAS_T3 = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
               "memory_gb", "power_w", "temperature_c", "load_score", "requires_review"]

# ------------------------------------------------------------------ 1) DataFrame analizado (T3)
try:
    with open(T3_DIR / "resultados.json", encoding="utf-8") as fh:
        t3 = json.load(fh)
except (FileNotFoundError, json.JSONDecodeError):
    t3 = {}
n_esperado = int(t3.get("n_filas", -1))

df_analizado, fuente = None, None
candidatos = []
if T3_DIR.is_dir():
    candidatos += sorted(T3_DIR.glob("*.parquet")) + sorted(T3_DIR.glob("*.csv"))
if RUTA_SALIDA.is_file():
    candidatos.append(RUTA_SALIDA)

for f in candidatos:
    try:
        cand = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
    except Exception:
        continue
    cand = cand.loc[:, [c for c in cand.columns
                        if not str(c).startswith("Unnamed:")
                        and not str(c).startswith("__index_level")]]
    if not all(c in cand.columns for c in COLUMNAS_T3):
        continue
    if n_esperado > 0 and len(cand) != n_esperado:
        continue
    if not np.issubdtype(cand["timestamp"].dtype, np.datetime64):
        cand["timestamp"] = pd.to_datetime(cand["timestamp"])
    cand["requires_review"] = cand["requires_review"].astype(bool)
    cand = cand[COLUMNAS_T3 + [c for c in cand.columns if c not in COLUMNAS_T3]]
    df_analizado, fuente = cand.reset_index(drop=True), str(f)
    break

if df_analizado is None:
    # Respaldo: reconstruir el DataFrame analizado con el CSV original y el
    # load_score ya calculado por la T3 (no se recalcula ninguna métrica).
    assert "load_score" in t3, "Sin DataFrame de T3 y sin load_score en entrada/T3/resultados.json"
    df_base = pd.read_csv("data/server_measurements.csv")
    df_base.columns = [str(c).strip().lower().replace(" ", "_") for c in df_base.columns]
    df_base = df_base.loc[:, [c for c in df_base.columns if not c.startswith("Unnamed:")]]
    df_base["timestamp"] = pd.to_datetime(df_base["timestamp"])
    load_score_t3 = np.asarray(t3["load_score"], dtype=float)
    assert len(load_score_t3) == len(df_base), "load_score de T3 y filas del CSV no coinciden"
    df_analizado = df_base.copy()
    df_analizado["load_score"] = load_score_t3
    df_analizado["requires_review"] = ((df_analizado["load_score"] > 1.5) |
                                       (df_analizado["temperature_c"] > 80)).astype(bool)
    df_analizado = df_analizado[COLUMNAS_T3].reset_index(drop=True)
    fuente = "data/server_measurements.csv + entrada/T3/resultados.json (load_score)"

if "load_score" in t3:
    ls_json = np.asarray(t3["load_score"], dtype=float)
    if len(ls_json) == len(df_analizado):
        igual = bool(np.allclose(ls_json, df_analizado["load_score"].to_numpy(dtype=float),
                                 rtol=0, atol=0, equal_nan=True))
        print(f"load_score consistente con entrada/T3/resultados.json: {igual}")

# ------------------------------------------------------------------ 2) Guardar con PyArrow, sin índice
RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
tabla_escrita = pa.Table.from_pandas(df_analizado, preserve_index=False)
pq.write_table(tabla_escrita, str(RUTA_SALIDA))

# ------------------------------------------------------------------ 3) Releer y verificar el round trip
tabla_leida = pq.read_table(str(RUTA_SALIDA))
df_leido = tabla_leida.to_pandas().reset_index(drop=True)

filas_ok = len(df_analizado) == len(df_leido)
columnas_ok = list(df_analizado.columns) == list(df_leido.columns)

esquema_antes = pa.Table.from_pandas(df_analizado, preserve_index=False).schema
esquema_despues = pa.Table.from_pandas(df_leido, preserve_index=False).schema
esquema_ok = bool(esquema_antes.equals(esquema_despues))

def valores_iguales_col(x, y):
    a, b = x.to_numpy(), y.to_numpy()
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.allclose(a, b, rtol=0, atol=0, equal_nan=True))
    return bool(np.array_equal(a, b))

a_df = df_analizado.reset_index(drop=True)
b_df = df_leido.reset_index(drop=True)
valores_ok = bool(all(valores_iguales_col(a_df[c], b_df[c]) for c in a_df.columns))

# ------------------------------------------------------------------ 4) Resultados
esq_antes = [f"{fl.name}: {fl.type}" for fl in esquema_antes]
esq_despues = [f"{fl.name}: {fl.type}" for fl in esquema_despues]

rev = df_analizado[df_analizado["requires_review"]]
fila_ejemplo = {}
if len(rev):
    r0 = rev.iloc[0]
    fila_ejemplo = {"indice": int(rev.index[0]), "server": str(r0["server"]),
                    "timestamp": pd.Timestamp(r0["timestamp"]).isoformat(),
                    "load_score": float(r0["load_score"]),
                    "temperature_c": float(r0["temperature_c"]),
                    "requires_review": bool(r0["requires_review"])}

resultados = {
    "subtarea": "T5",
    "supuestos": [
        "El DataFrame analizado de la T3 se carga de entrada/T3/ (o de output/server_analysis.parquet si la T3 ya lo dejó ahí); si no existe ningún archivo con el esquema completo, se reconstruye con data/server_measurements.csv y el load_score guardado en entrada/T3/resultados.json.",
        "El parquet se escribe con PyArrow (pa.Table.from_pandas + pq.write_table) sin incluir el índice (preserve_index=False).",
        "El esquema se compara con Schema.equals de PyArrow sin metadatos de pandas; los valores se comparan columna a columna con igualdad exacta (NaN cuenta como igual a NaN).",
    ],
    "ruta_parquet": RUTA_REL,
    "fuente_dataframe": fuente,
    "n_filas_antes": int(len(df_analizado)),
    "n_filas_despues": int(len(df_leido)),
    "columnas_antes": list(df_analizado.columns),
    "columnas_despues": list(df_leido.columns),
    "esquema_antes": esq_antes,
    "esquema_despues": esq_despues,
    "verificacion": {"filas": bool(filas_ok), "columnas": bool(columnas_ok),
                     "esquema": bool(esquema_ok), "valores": bool(valores_ok)},
    "round_trip_verificado": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
    "n_requires_review": int(df_analizado["requires_review"].sum()),
    "fila_ejemplo_requires_review": fila_ejemplo,
}
with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2)

print(f"Round trip verificado: {RUTA_REL}")
print(f"Filas: {filas_ok}")
print(f"Columnas: {columnas_ok}")
print(f"Esquema: {esquema_ok}")
print(f"Valores: {valores_ok}")
print("-" * 60)
print(f"Fuente del DataFrame analizado: {fuente}")
print(f"Filas: {len(df_analizado)} | Columnas: {len(df_analizado.columns)}")
print(f"Observaciones que requieren revisión: {resultados['n_requires_review']}")
print(f"Parquet escrito: {RUTA_REL} ({RUTA_SALIDA.stat().st_size} bytes)")
