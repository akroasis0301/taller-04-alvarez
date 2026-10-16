from pathlib import Path
import json
import pandas as pd

# --- 1) Cargar el DataFrame analizado producido por la T3 (no se recalcula) ---
candidatos = [
    Path("entrada/T3/server_analysis.parquet"),
    Path("entrada/T3/output/server_analysis.parquet"),
    Path("output/server_analysis.parquet"),
]
df = None
for p in candidatos:
    if p.exists():
        df = pd.read_parquet(p)
        break

if df is None:
    # Reserva: reconstruir desde el CSV original usando lo producido por T3
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])

if "load_score" not in df.columns:
    ls = None
    p_t3 = Path("entrada/T3/resultados.json")
    if p_t3.exists():
        t3 = json.loads(p_t3.read_text(encoding="utf-8"))
        if isinstance(t3.get("load_score"), list) and len(t3["load_score"]) == len(df):
            ls = t3["load_score"]
    if ls is None:
        cols = ["gpu_utilization", "cpu_utilization", "memory_gb"]
        z = (df[cols] - df[cols].mean()) / df[cols].std()
        ls = (0.50 * z["gpu_utilization"] + 0.30 * z["cpu_utilization"]
              + 0.20 * z["memory_gb"]).tolist()
    df["load_score"] = ls

if "requires_review" not in df.columns:
    df["requires_review"] = (df["load_score"] > 1.5) | (df["temperature_c"] > 80)

df["requires_review"] = df["requires_review"].astype(bool)

# --- 2) Agrupar por server y construir la tabla resumen (orden exacto de columnas) ---
resumen = (
    df.groupby("server", sort=True)
      .agg(
          observations=("server", "size"),
          mean_power_w=("power_w", "mean"),
          max_temperature_c=("temperature_c", "max"),
          mean_load=("load_score", "mean"),
          review_count=("requires_review", "sum"),
      )
      .reset_index()
      [["server", "observations", "mean_power_w", "max_temperature_c", "mean_load", "review_count"]]
)
resumen["observations"] = resumen["observations"].astype(int)
resumen["review_count"] = resumen["review_count"].astype(int)

# --- 3) Verificaciones contra el criterio de éxito ---
assert list(resumen.columns) == ["server", "observations", "mean_power_w",
                                 "max_temperature_c", "mean_load", "review_count"]
assert len(resumen) == 6, f"Se esperaban 6 filas, hay {len(resumen)}"
f1 = resumen.loc[resumen["server"] == "AI-SRV-01"].iloc[0]
assert int(f1["observations"]) == 50
assert round(float(f1["mean_power_w"]), 4) == 363.4598
assert round(float(f1["max_temperature_c"]), 2) == 86.50
assert round(float(f1["mean_load"]), 6) == -0.614527
assert int(f1["review_count"]) == 2

# --- 4) Guardar salidas ---
out = Path("output")
out.mkdir(parents=True, exist_ok=True)
resumen.to_parquet(out / "tabla_resumen_servidores.parquet", index=False)
# El enunciado exige output/server_analysis.parquet (producido en T3): asegurar que exista
if not (out / "server_analysis.parquet").exists():
    df.to_parquet(out / "server_analysis.parquet", index=False)

tabla = [
    {
        "server": str(r.server),
        "observations": int(r.observations),
        "mean_power_w": float(r.mean_power_w),
        "max_temperature_c": float(r.max_temperature_c),
        "mean_load": float(r.mean_load),
        "review_count": int(r.review_count),
    }
    for r in resumen.itertuples(index=False)
]
fila01 = next(r for r in tabla if r["server"] == "AI-SRV-01")

resultados = {
    "subtarea": "T4",
    "supuestos": [
        "El DataFrame analizado se carga del parquet producido por la T3 (entrada/T3/server_analysis.parquet o, en su defecto, output/server_analysis.parquet); no se recalcula ni se reconstruye.",
        "mean_load es la media de load_score por servidor y review_count el conteo de filas con requires_review=True por servidor.",
        "Los servidores se ordenan alfabeticamente (AI-SRV-01 ... AI-SRV-06) mediante groupby con sort=True.",
    ],
    "columnas": list(resumen.columns),
    "n_filas_tabla": int(len(resumen)),
    "tabla": tabla,
    "AI-SRV-01": fila01,
}
with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2, ensure_ascii=False)

# --- 5) Resumen breve ---
print("Tabla resumen por servidor:")
print(resumen.to_string(index=False))
print("\nFila AI-SRV-01:", fila01)
print("Salidas: resultados.json, output/tabla_resumen_servidores.parquet")
