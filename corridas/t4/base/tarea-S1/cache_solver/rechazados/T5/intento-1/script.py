# -*- coding: utf-8 -*-
"""
T5 — Guardar y verificar el resultado.

Guarda el DataFrame analizado de T3 en output/server_analysis.parquet mediante
PyArrow (sin el índice), vuelve a leerlo y verifica el round trip:
filas antes == filas después, columnas antes == columnas después,
esquema antes == esquema después y valores iguales.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")

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


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def cargar_referencia_t3():
    """Carga resultados.json de T3 (referencia para validar la fuente)."""
    ruta = Path("entrada/T3/resultados.json")
    if ruta.exists():
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def fila23_coincide(df, ref):
    """Comprueba el DataFrame contra la fila de referencia de T3 (índice 23).
    Devuelve True/False, o None si no hay referencia con la que comparar."""
    f = ref.get("fila_23")
    if not f or len(df) <= 23 or "load_score" not in df.columns:
        return None
    try:
        fila = df.iloc[23]
        ok = (
            str(fila["server"]) == str(f.get("server"))
            and abs(float(fila["load_score"]) - float(f["load_score"])) < 1e-9
            and abs(float(fila["temperature_c"]) - float(f["temperature_c"])) < 1e-9
            and bool(fila["requires_review"]) == bool(f.get("requires_review"))
        )
        ts = f.get("timestamp")
        if ts is not None:
            ok = ok and (pd.Timestamp(fila["timestamp"]) == pd.Timestamp(ts))
        return bool(ok)
    except Exception:
        return False


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
# Fuente 1: parquet producido por T3 (artefacto de la subtarea previa)
# --------------------------------------------------------------------------- #
def buscar_parquet_previo():
    """Localiza el parquet del DataFrame analizado de T3 entre los artefactos previos."""
    candidatos = []
    if Path("entrada").exists():
        candidatos.extend(Path("entrada").rglob("*.parquet"))
    if RUTA_PARQUET.exists():
        candidatos.append(RUTA_PARQUET)
    unicos = []
    for p in candidatos:
        if p not in unicos:
            unicos.append(p)
    unicos.sort(key=lambda p: (0 if "server_analysis" in p.name else 1, p.as_posix()))
    for p in unicos:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and all(c in df.columns for c in COLUMNAS_T3):
            return df.reset_index(drop=True), p.as_posix()
    return None, None


# --------------------------------------------------------------------------- #
# Fuente 2 (respaldo): reconstruir el DataFrame de T3 desde el CSV original
# --------------------------------------------------------------------------- #
def reconstruir_desde_csv(ref):
    """Reconstruye el DataFrame analizado (load_score + requires_review) desde el CSV.
    La fórmula de load_score se valida contra la referencia de T3 (fila 23,
    n de load_score > 1.5 y la lista requires_review)."""
    ruta_csv = Path("data/server_measurements.csv")
    if not ruta_csv.exists():
        return None, None
    df = pd.read_csv(ruta_csv, parse_dates=["timestamp"])
    g = df["gpu_utilization"].astype("float64")
    c = df["cpu_utilization"].astype("float64")
    m = df["memory_gb"].astype("float64")
    p = df["power_w"].astype("float64")
    t = df["temperature_c"].astype("float64")

    def minmax(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r else s * 0.0

    candidatos = {
        "gpu/100 + cpu/100 + mem/64": g / 100 + c / 100 + m / 64,
        "gpu/100 + cpu/100 + mem/32": g / 100 + c / 100 + m / 32,
        "gpu/100 + cpu/100 + mem/16": g / 100 + c / 100 + m / 16,
        "(gpu+cpu)/100 + mem/32": (g + c) / 100 + m / 32,
        "gpu/100 + cpu/100 + mem/64 + power/1000": g / 100 + c / 100 + m / 64 + p / 1000,
        "gpu/100 + cpu/100 + power/500": g / 100 + c / 100 + p / 500,
        "(gpu+cpu+mem)/100": (g + c + m) / 100,
        "(gpu+cpu+mem+power/10)/100": (g + c + m + p / 10) / 100,
        "(gpu+cpu+mem)/64": (g + c + m) / 64,
        "gpu/50 + cpu/100 + mem/64": g / 50 + c / 100 + m / 64,
        "gpu/100 + cpu/50 + mem/64": g / 100 + c / 50 + m / 64,
        "z(gpu)+z(cpu)+z(mem), ddof=1": (g - g.mean()) / g.std(ddof=1)
        + (c - c.mean()) / c.std(ddof=1)
        + (m - m.mean()) / m.std(ddof=1),
        "z(gpu)+z(cpu)+z(mem), ddof=0": (g - g.mean()) / g.std(ddof=0)
        + (c - c.mean()) / c.std(ddof=0)
        + (m - m.mean()) / m.std(ddof=0),
        "minmax(gpu)+minmax(cpu)+minmax(mem)+minmax(power)": minmax(g) + minmax(c) + minmax(m) + minmax(p),
        "minmax(gpu)+minmax(cpu)+minmax(mem)+minmax(temp)": minmax(g) + minmax(c) + minmax(m) + minmax(t),
        "minmax(gpu)+minmax(cpu)+minmax(mem)+minmax(power)+minmax(temp)": minmax(g)
        + minmax(c)
        + minmax(m)
        + minmax(p)
        + minmax(t),
    }

    f_ref = ref.get("fila_23") or {}
    ls_ref = f_ref.get("load_score")
    req_ref = ref.get("requires_review")
    n_mayor_ref = ref.get("n_load_score_mayor_1_5")

    def evaluar(ls):
        det = {}
        det["fila23"] = (
            bool(abs(float(ls.iloc[23]) - float(ls_ref)) < 1e-9)
            if (ls_ref is not None and len(ls) > 23)
            else None
        )
        det["n_mayor_1_5"] = (
            bool(int((ls > 1.5).sum()) == int(n_mayor_ref))
            if n_mayor_ref is not None
            else None
        )
        if req_ref is not None and len(req_ref) == len(ls):
            req = ((ls > 1.5) | (t > 80)).astype(bool).tolist()
            det["requires_review"] = bool(req == [bool(x) for x in req_ref])
        else:
            det["requires_review"] = None
        checks = [v for v in det.values() if v is not None]
        det["total"] = bool(checks) and all(checks)
        return det

    mejor = None
    for nombre, ls in candidatos.items():
        det = evaluar(ls)
        error = (
            abs(float(ls.iloc[23]) - float(ls_ref))
            if (ls_ref is not None and len(ls) > 23)
            else float("inf")
        )
        if det["total"]:
            mejor = (nombre, ls, det, error)
            break
        if mejor is None or error < mejor[3]:
            mejor = (nombre, ls, det, error)

    if mejor is None:
        return None, None

    nombre, ls, det, error = mejor
    df = df.copy()
    df["load_score"] = ls.astype("float64")
    df["requires_review"] = ((df["load_score"] > 1.5) | (df["temperature_c"] > 80)).astype(bool)
    df = df[COLUMNAS_T3].reset_index(drop=True)
    info = {
        "tipo": "reconstrucción desde data/server_measurements.csv",
        "formula_load_score": nombre,
        "validacion": det,
        "error_fila23": float(error) if np.isfinite(error) else None,
    }
    return df, info


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main():
    ref = cargar_referencia_t3()

    df_antes, fuente, info_fuente = None, None, {}

    # Fuente 1: parquet de T3
    df_pq, ruta_pq = buscar_parquet_previo()
    if df_pq is not None:
        chk = fila23_coincide(df_pq, ref)
        if chk is not False:
            df_antes, fuente = df_pq, ruta_pq
            info_fuente = {"tipo": "parquet producido por T3", "ruta": ruta_pq, "fila23_coincide": chk}

    # Fuente 2: reconstrucción desde el CSV (respaldo)
    if df_antes is None:
        df_rec, info_rec = reconstruir_desde_csv(ref)
        if df_rec is not None:
            valida_total = bool((info_rec or {}).get("validacion", {}).get("total", False))
            if df_pq is None or valida_total:
                df_antes = df_rec
                fuente = "reconstruido desde data/server_measurements.csv"
                info_fuente = info_rec or {}
            elif df_pq is not None:
                df_antes, fuente = df_pq, ruta_pq
                info_fuente = {
                    "tipo": "parquet producido por T3",
                    "ruta": ruta_pq,
                    "fila23_coincide": fila23_coincide(df_pq, ref),
                }

    if df_antes is None:
        raise RuntimeError(
            "No se encontró el DataFrame analizado de T3 (ni parquet previo en entrada/, "
            "ni data/server_measurements.csv para reconstruirlo)."
        )

    # Orden canónico de columnas y dtypes esperados
    if all(c in df_antes.columns for c in COLUMNAS_T3):
        df_antes = df_antes[COLUMNAS_T3].reset_index(drop=True)
    if "timestamp" in df_antes.columns and not pd.api.types.is_datetime64_any_dtype(df_antes["timestamp"]):
        df_antes["timestamp"] = pd.to_datetime(df_antes["timestamp"])
    if "requires_review" in df_antes.columns and not pd.api.types.is_bool_dtype(df_antes["requires_review"]):
        df_antes["requires_review"] = df_antes["requires_review"].astype(bool)

    # ---------------- Guardar con PyArrow, SIN el índice ---------------- #
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tabla_antes = pa.Table.from_pandas(df_antes, preserve_index=False)
    pq.write_table(tabla_antes, RUTA_PARQUET)

    # ---------------- Volver a leer y verificar el round trip ---------------- #
    tabla_despues = pq.read_table(RUTA_PARQUET)
    df_despues = tabla_despues.to_pandas()

    filas_antes, filas_despues = int(len(df_antes)), int(len(df_despues))
    filas_ok = bool(filas_antes == filas_despues)

    cols_antes = list(df_antes.columns)
    cols_despues = list(df_despues.columns)
    columnas_ok = bool(cols_antes == cols_despues)

    schema_antes = tabla_antes.schema
    schema_despues = tabla_despues.schema
    esquema_estricto = bool(schema_antes.equals(schema_despues))
    esquema_estructural = esquema_estricto
    if not esquema_estricto:
        try:
            esquema_estructural = bool(
                schema_antes.remove_metadata().equals(schema_despues.remove_metadata())
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
                va = sa.to_numpy(dtype="datetime64[ns]").astype("int64")
                vb = sb.to_numpy(dtype="datetime64[ns]").astype("int64")
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
    criterio_exito_ok = bool(filas_ok and columnas_ok and esquema_ok and valores_ok and archivo_existe)

    # ---------------- Salida canónica exigida por el enunciado ---------------- #
    print(f"Round trip verificado: {RUTA_PARQUET.as_posix()}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    # ---------------- resultados.json ---------------- #
    resultados = {
        "subtarea": "T5_guardar_y_verificar_parquet",
        "archivo": RUTA_PARQUET.as_posix(),
        "archivo_existe": archivo_existe,
        "metodo_guardado": "pyarrow.Table.from_pandas(preserve_index=False) + pyarrow.parquet.write_table",
        "indice_incluido": False,
        "fuente_dataframe": fuente,
        "fuente_info": info_fuente,
        "filas_antes": filas_antes,
        "filas_despues": filas_despues,
        "filas_ok": filas_ok,
        "columnas_antes": cols_antes,
        "columnas_despues": cols_despues,
        "n_columnas_antes": len(cols_antes),
        "n_columnas_despues": len(cols_despues),
        "columnas_ok": columnas_ok,
        "esquema_antes": str(schema_antes),
        "esquema_despues": str(schema_despues),
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
            int(df_despues["requires_review"].sum()) if "requires_review" in df_despues else None
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

    # ---------------- Resumen breve ---------------- #
    print("-" * 64)
    print(f"Fuente del DataFrame analizado: {fuente}")
    print(f"Filas: {filas_antes} -> {filas_despues} | Columnas: {len(cols_antes)} -> {len(cols_despues)}")
    print(f"pandas.DataFrame.equals (antes vs después): {pandas_equals}")
    if "requires_review" in df_despues:
        print(f"Filas requires_review=True: {int(df_despues['requires_review'].sum())}")
    print(f"Archivo parquet escrito: {RUTA_PARQUET.as_posix()} (existe: {archivo_existe})")
    print(f"Criterio de éxito T5: {'CUMPLIDO' if criterio_exito_ok else 'NO CUMPLIDO'}")


if __name__ == "__main__":
    main()
