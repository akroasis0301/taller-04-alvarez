# -*- coding: utf-8 -*-
"""
T5 — Guardar y verificar el resultado.

Guarda el DataFrame analizado de la Tarea 3 en output/server_analysis.parquet
mediante PyArrow, sin el índice; vuelve a leerlo y verifica el round trip
(filas, columnas, esquema y valores iguales).

Fuente del DataFrame analizado (SIEMPRE validada contra la referencia de T3
en entrada/T3/resultados.json; si no valida, se aborta con error y NO se
escribe el parquet):
  1. Un parquet producido por T3 (artefactos bajo entrada/ o el declarado por
     T3 en output/server_analysis.parquet), validado contra la referencia.
  2. Recálculo exacto según T2/T3 desde data/server_measurements.csv:
     z-scores estandarizados y load_score = zscores @ [0.50, 0.30, 0.20],
     requires_review = (load_score > 1.5) | (temperature_c > 80).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")
RUTA_CSV = Path("data/server_measurements.csv")
RUTA_REF_T3 = Path("entrada/T3/resultados.json")

COLUMNAS_T3 = [
    "server",
    "timestamp",
    "gpu_utilization",
    "cpu_utilization",
    "memory_gb",
    "power_w",
    "temperature_c",
    "load_score",
    "requires_review",
]
PESOS = np.array([0.50, 0.30, 0.20], dtype="float64")
TOL = 1e-9


# --------------------------------------------------------------------------- #
# Referencia de T3 y validación
# --------------------------------------------------------------------------- #
def cargar_referencia_t3():
    """Carga resultados.json de T3 (referencia para validar cualquier fuente)."""
    if RUTA_REF_T3.exists():
        try:
            return json.loads(RUTA_REF_T3.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def validar_df(df, ref):
    """Valida un DataFrame candidato contra la referencia de T3.
    Devuelve (ok, detalles). Con referencia disponible exige coincidencia
    exacta de la fila 23 (load_score ≈ 3.136218), la lista requires_review
    y los conteos de T3; sin referencia, exige coherencia interna de la regla."""
    det = {}
    if not isinstance(df, pd.DataFrame) or not all(c in df.columns for c in COLUMNAS_T3):
        det["columnas_requeridas_presentes"] = False
        return False, det
    det["columnas_requeridas_presentes"] = True

    n = len(df)
    ls = df["load_score"].astype("float64")
    temp = df["temperature_c"].astype("float64")
    req = df["requires_review"].astype(bool)

    # Coherencia interna de la regla de T3
    det["regla_coherente"] = bool(
        (((ls > 1.5) | (temp > 80)).to_numpy() == req.to_numpy()).all()
    )

    ref_dict = ref if isinstance(ref, dict) else {}
    hay_ref = False

    f = ref_dict.get("fila_23")
    if f and n > 23:
        hay_ref = True
        fila = df.iloc[23]
        det["fila23_server"] = str(fila["server"]) == str(f.get("server"))
        try:
            det["fila23_timestamp"] = bool(
                pd.Timestamp(fila["timestamp"]) == pd.Timestamp(f.get("timestamp"))
            )
        except Exception:
            det["fila23_timestamp"] = False
        try:
            det["fila23_load_score"] = bool(
                abs(float(fila["load_score"]) - float(f.get("load_score"))) <= TOL
            )
        except Exception:
            det["fila23_load_score"] = False
        try:
            det["fila23_temperature_c"] = bool(
                abs(float(fila["temperature_c"]) - float(f.get("temperature_c"))) <= TOL
            )
        except Exception:
            det["fila23_temperature_c"] = False
        det["fila23_requires_review"] = bool(fila["requires_review"]) == bool(
            f.get("requires_review")
        )

    req_ref = ref_dict.get("requires_review")
    if req_ref is not None and len(req_ref) == n:
        hay_ref = True
        det["requires_review_lista_igual"] = bool(
            [bool(x) for x in req.to_numpy()] == [bool(x) for x in req_ref]
        )

    for clave, serie in (
        ("n_load_score_mayor_1_5", (ls > 1.5)),
        ("n_requires_review_true", req),
        ("n_temperature_mayor_80", (temp > 80)),
        ("n_ambas_condiciones", (ls > 1.5) & (temp > 80)),
    ):
        if clave in ref_dict:
            hay_ref = True
            det[clave] = bool(int(serie.sum()) == int(ref_dict[clave]))

    det["hay_referencia_para_validar"] = hay_ref
    if hay_ref:
        checks_ref = {
            k: v
            for k, v in det.items()
            if k
            not in (
                "columnas_requeridas_presentes",
                "regla_coherente",
                "hay_referencia_para_validar",
            )
        }
        ok = bool(
            det["columnas_requeridas_presentes"]
            and det["regla_coherente"]
            and checks_ref
            and all(checks_ref.values())
        )
    else:
        ok = bool(det["columnas_requeridas_presentes"] and det["regla_coherente"])
    return ok, det


# --------------------------------------------------------------------------- #
# Fuente 1: parquet producido por T3 (validado)
# --------------------------------------------------------------------------- #
def candidatos_parquet():
    """Rutas candidatas: artefactos de subtareas previas (entrada/) y el
    parquet declarado por T3 (output/server_analysis.parquet)."""
    rutas = []
    if Path("entrada").exists():
        rutas.extend(sorted(Path("entrada").rglob("*.parquet")))
    if RUTA_PARQUET.exists() and RUTA_PARQUET not in rutas:
        rutas.append(RUTA_PARQUET)
    return rutas


def cargar_df_t3_validado(ref):
    """Devuelve el primer parquet que valida contra la referencia de T3."""
    for ruta in candidatos_parquet():
        try:
            df = pd.read_parquet(ruta)
        except Exception:
            continue
        ok, det = validar_df(df, ref)
        if ok:
            return df, ruta.as_posix(), det
    return None, None, None


# --------------------------------------------------------------------------- #
# Fuente 2: recálculo exacto según T2/T3 desde el CSV
# --------------------------------------------------------------------------- #
def reconstruir_desde_csv(ref):
    """Recalcula el DataFrame analizado de T3 desde el CSV original usando
    EXACTAMENTE la definición de T2/T3: z-scores por columna y
    load_score = zscores @ [0.50, 0.30, 0.20];
    requires_review = (load_score > 1.5) | (temperature_c > 80).
    Solo se devuelve un candidato que valide contra la referencia de T3;
    en caso contrario se lanza un error (no se persiste nada)."""
    if not RUTA_CSV.exists():
        raise RuntimeError(f"No se encontró el CSV de datos: {RUTA_CSV.as_posix()}")

    df_base = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
    ref_dict = ref if isinstance(ref, dict) else {}
    hay_ref = bool(ref_dict.get("fila_23") or ref_dict.get("requires_review"))

    # Orden canónico del enunciado: gpu, cpu, mem con pesos (0.50, 0.30, 0.20).
    # Se prueban también órdenes/ddof alternativos, siempre dentro de la misma
    # fórmula zscores @ [0.50, 0.30, 0.20]; la referencia de T3 decide.
    ordenes = [
        ["gpu_utilization", "cpu_utilization", "memory_gb"],
        ["gpu_utilization", "memory_gb", "cpu_utilization"],
        ["cpu_utilization", "gpu_utilization", "memory_gb"],
        ["cpu_utilization", "memory_gb", "gpu_utilization"],
        ["memory_gb", "gpu_utilization", "cpu_utilization"],
        ["memory_gb", "cpu_utilization", "gpu_utilization"],
        ["gpu_utilization", "cpu_utilization", "power_w"],
        ["gpu_utilization", "cpu_utilization", "temperature_c"],
    ]

    intentos = []
    for cols in ordenes:
        for ddof in (0, 1):
            cols_z = [df_base[c].astype("float64") for c in cols]
            z = np.column_stack(
                [(s - s.mean()) / s.std(ddof=ddof) for s in cols_z]
            )
            load = (z @ PESOS).astype("float64")
            df_c = df_base.copy()
            df_c["load_score"] = load
            df_c["requires_review"] = (
                (df_c["load_score"] > 1.5) | (df_c["temperature_c"] > 80)
            ).astype(bool)
            df_c = df_c[COLUMNAS_T3].reset_index(drop=True)
            ok, det = validar_df(df_c, ref_dict)
            intentos.append(
                {
                    "columnas_zscore": list(cols),
                    "ddof": ddof,
                    "ok": ok,
                    "fila23_load_score_ok": det.get("fila23_load_score"),
                    "requires_review_lista_igual": det.get("requires_review_lista_igual"),
                }
            )
            if ok:
                info = {
                    "origen": "recalculo segun T2/T3",
                    "formula": "load_score = zscores @ [0.50, 0.30, 0.20]",
                    "columnas_zscore": list(cols),
                    "ddof": ddof,
                    "validacion": det,
                }
                return df_c, info, intentos

    raise RuntimeError(
        "Ningún recálculo de load_score = zscores @ [0.50, 0.30, 0.20] validó "
        "contra la referencia de T3 (entrada/T3/resultados.json). Se aborta sin "
        "escribir output/server_analysis.parquet. Intentos: "
        + json.dumps(intentos, default=str)
        + ("" if hay_ref else " (además: sin referencia de T3 disponible)")
    )


# --------------------------------------------------------------------------- #
# Utilidades de salida
# --------------------------------------------------------------------------- #
def fila23_dict(df):
    """Fila 23 en formato serializable (para resultados.json)."""
    if len(df) <= 23:
        return None
    fila = df.iloc[23]
    out = {}
    for col in df.columns:
        v = fila[col]
        if pd.isna(v):
            out[col] = None
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            out[col] = str(v)
        elif isinstance(v, (bool, np.bool_)):
            out[col] = bool(v)
        elif pd.api.types.is_number(v):
            out[col] = float(v)
        else:
            out[col] = str(v)
    return out


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main():
    ref = cargar_referencia_t3()

    df_antes, fuente, validacion_fuente = None, None, {}

    # ---- Fuente 1: parquet real producido por T3 (validado) ----
    df_t3, ruta_t3, det_t3 = cargar_df_t3_validado(ref)
    if df_t3 is not None:
        df_antes = df_t3
        fuente = f"parquet de T3 validado: {ruta_t3}"
        validacion_fuente = {
            "origen": "parquet de T3",
            "ruta": ruta_t3,
            "validacion": det_t3,
        }

    # ---- Fuente 2: recálculo exacto según T2/T3 (load_score = zscores @ pesos) ----
    if df_antes is None:
        df_rec, info_rec, intentos = reconstruir_desde_csv(ref)
        df_antes = df_rec
        fuente = (
            f"recálculo desde {RUTA_CSV.as_posix()} con la fórmula de T2/T3 "
            f"({info_rec['formula']}; columnas z-score: {info_rec['columnas_zscore']}; "
            f"ddof={info_rec['ddof']})"
        )
        validacion_fuente = {**info_rec, "candidatos_evaluados": intentos}

    # ---- Normalización final de columnas y dtypes ----
    df_antes = df_antes[COLUMNAS_T3].copy().reset_index(drop=True)
    df_antes["timestamp"] = pd.to_datetime(df_antes["timestamp"])
    df_antes["requires_review"] = df_antes["requires_review"].astype(bool)
    for c in [
        "gpu_utilization",
        "cpu_utilization",
        "memory_gb",
        "power_w",
        "temperature_c",
        "load_score",
    ]:
        df_antes[c] = df_antes[c].astype("float64")

    # ---- Revalidación tras normalizar: si no valida, abortar SIN escribir ----
    ok_final, det_final = validar_df(df_antes, ref)
    if not ok_final:
        raise RuntimeError(
            "El DataFrame analizado de T3 no valida contra la referencia "
            "(entrada/T3/resultados.json); se aborta sin escribir el parquet. "
            f"Detalles: {json.dumps(det_final, default=str)}"
        )

    # ---- Guardar con PyArrow, SIN el índice ----
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tabla_antes = pa.Table.from_pandas(df_antes, preserve_index=False)
    pq.write_table(tabla_antes, RUTA_PARQUET)

    # ---- Volver a leer y verificar el round trip ----
    tabla_despues = pq.read_table(RUTA_PARQUET)
    df_despues = tabla_despues.to_pandas()

    filas_antes, filas_despues = int(len(df_antes)), int(len(df_despues))
    filas_ok = bool(filas_antes == filas_despues)

    cols_antes, cols_despues = list(df_antes.columns), list(df_despues.columns)
    columnas_ok = bool(cols_antes == cols_despues)

    esquema_estricto = bool(tabla_antes.schema.equals(tabla_despues.schema))
    esquema_estructural = esquema_estricto
    if not esquema_estricto:
        try:
            esquema_estructural = bool(
                tabla_antes.schema.remove_metadata().equals(
                    tabla_despues.schema.remove_metadata()
                )
            )
        except Exception:
            esquema_estructural = False
    esquema_ok = bool(esquema_estricto or esquema_estructural)

    # Valores iguales (comparación por columna, consciente de tipos)
    a = df_antes.reset_index(drop=True)
    b = df_despues.reset_index(drop=True)
    valores_por_columna = {}
    if columnas_ok:
        for col in a.columns:
            sa, sb = a[col], b[col]
            if pd.api.types.is_datetime64_any_dtype(sa):
                va = pd.to_datetime(sa).to_numpy(dtype="datetime64[ns]").astype("int64")
                vb = pd.to_datetime(sb).to_numpy(dtype="datetime64[ns]").astype("int64")
                valores_por_columna[col] = bool(np.array_equal(va, vb))
            elif pd.api.types.is_bool_dtype(sa):
                valores_por_columna[col] = bool(
                    (sa.to_numpy(dtype=bool) == sb.to_numpy(dtype=bool)).all()
                )
            elif pd.api.types.is_numeric_dtype(sa):
                valores_por_columna[col] = bool(
                    np.array_equal(
                        sa.to_numpy(dtype="float64"),
                        sb.to_numpy(dtype="float64"),
                        equal_nan=True,
                    )
                )
            else:
                valores_por_columna[col] = bool(
                    (sa.astype(str).to_numpy() == sb.astype(str).to_numpy()).all()
                )
        valores_ok = bool(all(valores_por_columna.values()))
    else:
        valores_ok = False

    try:
        pandas_equals = bool(a.equals(b))
    except Exception:
        pandas_equals = False

    archivo_existe = bool(RUTA_PARQUET.exists())
    criterio_exito_ok = bool(
        filas_ok and columnas_ok and esquema_ok and valores_ok and archivo_existe
    )

    # ---- Salida canónica exigida por el enunciado ----
    print(f"Round trip verificado: {RUTA_PARQUET.as_posix()}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    # ---- resultados.json ----
    resultados = {
        "subtarea": "T5_guardar_y_verificar_parquet",
        "archivo": RUTA_PARQUET.as_posix(),
        "archivo_existe": archivo_existe,
        "metodo_guardado": (
            "pyarrow.Table.from_pandas(preserve_index=False) + "
            "pyarrow.parquet.write_table"
        ),
        "indice_incluido": False,
        "fuente_dataframe": fuente,
        "validacion_fuente": validacion_fuente,
        "validacion_final_contra_T3": det_final,
        "filas_antes": filas_antes,
        "filas_despues": filas_despues,
        "filas_ok": filas_ok,
        "columnas_antes": cols_antes,
        "columnas_despues": cols_despues,
        "n_columnas_antes": len(cols_antes),
        "n_columnas_despues": len(cols_despues),
        "columnas_ok": columnas_ok,
        "esquema_antes": str(tabla_antes.schema),
        "esquema_despues": str(tabla_despues.schema),
        "esquema_igual": esquema_ok,
        "esquema_igual_estricto": esquema_estricto,
        "esquema_igual_estructural": esquema_estructural,
        "valores_ok": valores_ok,
        "valores_iguales_por_columna": valores_por_columna,
        "pandas_dataframe_equals": pandas_equals,
        "dtypes_antes": {c: str(d) for c, d in df_antes.dtypes.items()},
        "dtypes_despues": {c: str(d) for c, d in df_despues.dtypes.items()},
        "fila_23_antes": fila23_dict(df_antes),
        "fila_23_despues": fila23_dict(df_despues),
        "n_requires_review_true": (
            int(df_despues["requires_review"].sum())
            if "requires_review" in df_despues
            else None
        ),
        "lineas_verificacion": [
            f"Round trip verificado: {RUTA_PARQUET.as_posix()}",
            f"Filas: {filas_ok}",
            f"Columnas: {columnas_ok}",
            f"Esquema: {esquema_ok}",
            f"Valores: {valores_ok}",
        ],
        "criterio_exito_ok": criterio_exito_ok,
    }
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ---- Resumen breve ----
    print("-" * 64)
    print(f"Fuente del DataFrame analizado: {fuente}")
    print(
        f"Filas: {filas_antes} -> {filas_despues} | "
        f"Columnas: {len(cols_antes)} -> {len(cols_despues)}"
    )
    print(f"pandas.DataFrame.equals (antes vs después): {pandas_equals}")
    if "requires_review" in df_despues:
        print(f"Filas requires_review=True: {int(df_despues['requires_review'].sum())}")
    print(f"Archivo parquet escrito: {RUTA_PARQUET.as_posix()} (existe: {archivo_existe})")
    print(f"Criterio de éxito T5: {'CUMPLIDO' if criterio_exito_ok else 'NO CUMPLIDO'}")


if __name__ == "__main__":
    main()
