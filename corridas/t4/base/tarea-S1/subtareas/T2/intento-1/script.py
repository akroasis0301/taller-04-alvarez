# -*- coding: utf-8 -*-
"""
T2 - Normalizar (Z-score) la matriz de T1 y calcular load_score de forma vectorizada.

Entradas:
  - data/server_measurements.csv            (datos de la tarea)
  - entrada/T1/results.json                 (subtarea previa: de donde se toman feature_names)

Salidas:
  - resultados.json                         (contrato de la subtarea: todas las cifras pedidas)
  - output/server_analysis.parquet          (archivo exigido por el enunciado: df + load_score)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T1_JSON = Path("entrada/T1/results.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")

PESOS = np.array([0.50, 0.30, 0.20], dtype=np.float64)
FEATURES_DEFECTO = ["gpu_utilization", "cpu_utilization", "memory_gb"]
MEANS_ESPERADAS = np.array([57.99313333, 50.54163333, 29.45533333], dtype=np.float64)
TOL = 1e-10

np.set_printoptions(precision=8)


def obtener_feature_names():
    """Recupera los nombres de características declarados por T1 (con reserva por defecto)."""
    if RUTA_T1_JSON.exists():
        try:
            t1 = json.loads(RUTA_T1_JSON.read_text(encoding="utf-8"))
            nombres = t1.get("feature_names")
            if isinstance(nombres, list) and len(nombres) == 3:
                return [str(n) for n in nombres]
        except (OSError, json.JSONDecodeError):
            pass
    return list(FEATURES_DEFECTO)


def main():
    feature_names = obtener_feature_names()

    # --- Matriz de T1: mismas características y mismo orden que build_measurements ---
    df = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
    faltantes = [c for c in feature_names if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el CSV: {faltantes}")
    values = df[feature_names].to_numpy(dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("La matriz de características debe ser bidimensional")

    # --- Media y desviación estándar por columna (vectorizado, ddof=0) ---
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)

    # --- Z-score vectorizado evitando división por cero ---
    # Si una std es 0, se usa 1.0 como denominador: la columna centrada queda en 0.
    safe_stds = np.where(stds == 0.0, 1.0, stds)
    zscores = (values - means) / safe_stds  # broadcasting: sin for sobre observaciones

    # --- load_score vectorizado ---
    load_score = zscores @ PESOS

    # --- Verificaciones del criterio de éxito ---
    col_means = zscores.mean(axis=0)
    col_stds = zscores.std(axis=0, ddof=0)
    max_abs_media = float(np.max(np.abs(col_means)))
    max_abs_std_menos_uno = float(np.max(np.abs(col_stds - 1.0)))
    media_cero_ok = bool(max_abs_media <= TOL)
    desviacion_uno_ok = bool(max_abs_std_menos_uno <= TOL)

    # load_score debe coincidir con zscores @ pesos (verificación independiente)
    load_recalc = (zscores * PESOS).sum(axis=1)
    load_max_diff = float(np.max(np.abs(load_score - load_recalc)))
    load_ok = bool(load_max_diff <= 1e-10)

    # means impresas deben ser [57.99313333 50.54163333 29.45533333]
    means_coinciden = bool(
        np.allclose(np.round(means, 8), MEANS_ESPERADAS, rtol=0.0, atol=1e-9)
    )

    n_stds_cero = int(np.sum(stds == 0.0))

    # --- Archivo exigido por el enunciado: parquet con load_score agregado a df ---
    df_salida = df.copy()
    df_salida["load_score"] = load_score
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_salida.to_parquet(RUTA_PARQUET, index=False)

    # --- Contrato de la subtarea ---
    resultados = {
        "subtarea": "T2_normalizar_y_calcular_load_score",
        "feature_names": feature_names,
        "n_observaciones": int(values.shape[0]),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "std_ddof": 0,
        "means_esperadas": MEANS_ESPERADAS.tolist(),
        "means_coinciden_esperadas": means_coinciden,
        "zscores_shape": list(zscores.shape),
        "load_score_shape": list(load_score.shape),
        "zscores_medias_por_columna": col_means.tolist(),
        "zscores_desviaciones_por_columna": col_stds.tolist(),
        "tolerancia": TOL,
        "max_abs_media_columna": max_abs_media,
        "max_abs_desviacion_menos_uno": max_abs_std_menos_uno,
        "media_cero_ok": media_cero_ok,
        "desviacion_uno_ok": desviacion_uno_ok,
        "pesos": PESOS.tolist(),
        "load_score": load_score.tolist(),
        "load_score_resumen": {
            "media": float(load_score.mean()),
            "desviacion_estandar": float(load_score.std(ddof=0)),
            "minimo": float(load_score.min()),
            "maximo": float(load_score.max()),
        },
        "load_score_max_diff_vs_producto": load_max_diff,
        "load_score_coincide": load_ok,
        "columnas_con_std_cero": n_stds_cero,
        "division_por_cero_evitada": True,
        "vectorizado_sin_for_sobre_observaciones": True,
        "parquet_escrito": str(RUTA_PARQUET),
        "parquet_filas": int(len(df_salida)),
        "parquet_columnas": list(df_salida.columns),
    }
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Resumen breve ---
    print("means:", means)
    print("stds :", stds)
    print("zscores.shape    ->", zscores.shape)
    print("load_score.shape ->", load_score.shape)
    print("media por columna de zscores   :", col_means)
    print("desv. est. por columna zscores :", col_stds)
    print(f"max|media col| = {max_abs_media:.3e} (tol 1e-10) -> {media_cero_ok}")
    print(f"max|std - 1|   = {max_abs_std_menos_uno:.3e} (tol 1e-10) -> {desviacion_uno_ok}")
    print(f"load_score == zscores @ [0.50, 0.30, 0.20]: max diff = {load_max_diff:.3e} -> {load_ok}")
    print(
        f"load_score: media={load_score.mean():.6f}, "
        f"min={load_score.min():.6f}, max={load_score.max():.6f}"
    )
    print(f"Parquet escrito en {RUTA_PARQUET} ({len(df_salida)} filas)")


if __name__ == "__main__":
    main()
