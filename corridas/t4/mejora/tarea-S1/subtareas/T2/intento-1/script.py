import json
from pathlib import Path

import numpy as np
import pandas as pd

# Parámetros fijos de la subtarea
TOL = 1e-10
FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS = np.array([0.50, 0.30, 0.20])

# ------------------------------------------------------------------
# 1) Matriz de características de la Tarea 1:
#    cargarla de entrada/T1/ si está guardada; si no, reconstruir
#    desde data/server_measurements.csv con la misma selección de T1.
# ------------------------------------------------------------------
t1_dir = Path("entrada/T1")
X = None
fuente = None

if t1_dir.exists():
    for npy in sorted(t1_dir.glob("*.npy")):
        try:
            arr = np.load(npy)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[1] == len(FEATURES):
            X = arr.astype(np.float64)
            fuente = str(npy)
            break
    if X is None:
        for pq in sorted(t1_dir.glob("*.parquet")):
            t1df = pd.read_parquet(pq)
            if all(c in t1df.columns for c in FEATURES):
                X = t1df[FEATURES].to_numpy(dtype=np.float64)
                fuente = str(pq)
                break

if X is None:
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
    X = df[FEATURES].to_numpy(dtype=np.float64)
    fuente = "data/server_measurements.csv"

# ------------------------------------------------------------------
# 2) Media y desviación estándar por columna + Z-score vectorizado
#    (ddof=0, default de numpy, para que cada columna quede con std 1)
# ------------------------------------------------------------------
means = X.mean(axis=0)
stds = X.std(axis=0)
safe_stds = np.where(stds == 0.0, 1.0, stds)  # evita división por cero
zscores = (X - means) / safe_stds             # vectorizado, sin for

# ------------------------------------------------------------------
# 3) load_score = zscores @ [0.50, 0.30, 0.20] (vectorizado)
# ------------------------------------------------------------------
load_score = zscores @ PESOS

# ------------------------------------------------------------------
# 4) Verificaciones del criterio de éxito
# ------------------------------------------------------------------
col_means = zscores.mean(axis=0)
col_stds = zscores.std(axis=0)
media0 = bool(np.all(np.abs(col_means) <= TOL))
std1 = bool(np.all(np.abs(col_stds - 1.0) <= TOL))
producto_ok = bool(np.allclose(load_score, zscores @ PESOS, rtol=0.0, atol=1e-12))

assert zscores.shape == (300, 3), f"forma inesperada de zscores: {zscores.shape}"
assert load_score.shape == (300,), f"forma inesperada de load_score: {load_score.shape}"
assert media0, f"medias de zscores fuera de tolerancia: {col_means}"
assert std1, f"stds de zscores fuera de tolerancia: {col_stds}"
assert producto_ok, "load_score no coincide con zscores @ pesos"

# Consistencia con las medias reportadas por la Tarea 1
t1_json = t1_dir / "resultados.json"
consistente_t1 = None
if t1_json.exists():
    t1 = json.loads(t1_json.read_text(encoding="utf-8"))
    t1_means = np.array(
        [t1["estadisticas_descriptivas"][f]["media"] for f in FEATURES]
    )
    consistente_t1 = bool(np.allclose(means, t1_means, rtol=0.0, atol=1e-9))
    assert consistente_t1, f"medias no coinciden con T1: {means} vs {t1_means}"

# ------------------------------------------------------------------
# 5) Salidas impresas exigidas
# ------------------------------------------------------------------
print("means =", means)
print("zscores.shape =", zscores.shape)
print("load_score.shape =", load_score.shape)
print("stds =", stds)
print("media de zscores por columna =", col_means)
print("std de zscores por columna   =", col_stds)
print(
    "load_score resumen: media=%.10f std=%.10f min=%.6f max=%.6f"
    % (load_score.mean(), load_score.std(), load_score.min(), load_score.max())
)

# ------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "fuente_matriz": fuente,
    "feature_names": FEATURES,
    "means": means.tolist(),
    "stds": stds.tolist(),
    "stds_usadas_para_division": safe_stds.tolist(),
    "n_stds_cero": int(np.sum(stds == 0.0)),
    "pesos": PESOS.tolist(),
    "zscores_shape": [int(zscores.shape[0]), int(zscores.shape[1])],
    "load_score_shape": [int(load_score.shape[0])],
    "zscores": zscores.tolist(),
    "load_score": load_score.tolist(),
    "load_score_resumen": {
        "media": float(load_score.mean()),
        "desv_std": float(load_score.std()),
        "min": float(load_score.min()),
        "max": float(load_score.max()),
    },
    "verificacion": {
        "tolerancia": TOL,
        "medias_zscores": col_means.tolist(),
        "stds_zscores": col_stds.tolist(),
        "max_abs_media_zscores": float(np.max(np.abs(col_means))),
        "max_abs_1_menos_std_zscores": float(np.max(np.abs(col_stds - 1.0))),
        "media_cero_ok": media0,
        "std_uno_ok": std1,
        "load_score_igual_a_zscores_por_pesos": producto_ok,
        "medias_coinciden_con_T1": consistente_t1,
    },
    "supuestos": [
        "La desviación estándar se calcula con ddof=0 (default de numpy) para que cada "
        "columna de zscores tenga media 0 y desviación exactamente 1 con tolerancia 1e-10.",
        "Si alguna desviación estándar fuera 0 se reemplaza por 1.0 antes de dividir "
        "(los zscores de esa columna quedarían en 0); con estos datos ninguna std es 0.",
        "La matriz de características es la de la Tarea 1 (gpu_utilization, cpu_utilization, "
        "memory_gb en ese orden); el script la carga de entrada/T1/ si está guardada y, en su "
        "defecto, la reconstruye desde data/server_measurements.csv con la misma selección.",
        "output/server_analysis.parquet es entregable de una subtarea posterior (resumen por "
        "servidor) y no se genera en T2.",
    ],
}

Path("resultados.json").write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("resultados.json escrito. Verificaciones: media0=%s, std1=%s, producto_ok=%s"
      % (media0, std1, producto_ok))
