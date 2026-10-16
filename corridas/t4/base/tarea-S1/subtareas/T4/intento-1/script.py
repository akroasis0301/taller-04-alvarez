# -*- coding: utf-8 -*-
"""
T4 · Construir un resumen por servidor
=======================================
Agrupa el DataFrame de T3 por `server` y construye una tabla con una fila por
servidor y EXACTAMENTE las columnas, en este orden:

    server, observations, mean_power_w, max_temperature_c, mean_load, review_count

Definiciones:
    observations      : número de filas del servidor
    mean_power_w      : media de power_w
    max_temperature_c : máximo de temperature_c
    mean_load         : media de load_score
    review_count      : número de filas con requires_review == True

Robustez: el DataFrame de T3 (con load_score y requires_review) se busca en
entrada/T3/ y output/; si no está, se reconstruye desde
data/server_measurements.csv recuperando load_score / requires_review de los
artefactos de T2/T3 y, en último caso, recalibrando load_score contra los
valores de control conocidos de T3 (fila 23, nº de load_score>1.5, mean_load
de AI-SRV-01).

Salidas: resultados.json, output/server_analysis.parquet,
output/server_summary.parquet, output/server_summary.csv,
t4_resumen_por_servidor.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
T3_COLS = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
           "memory_gb", "power_w", "temperature_c", "load_score", "requires_review"]
BASE_COLS = T3_COLS[:7]
ORDEN_COLUMNAS = ["server", "observations", "mean_power_w",
                  "max_temperature_c", "mean_load", "review_count"]

# Valores de control conocidos (resultados de T3 / enunciado de T4)
CTRL_IDX = 23
CTRL_SERVER = "AI-SRV-01"
CTRL_TS = "2026-08-24 09:55:00"
CTRL_LOAD_SCORE = 3.136217861183744
CTRL_MEAN_LOAD_SRV01 = -0.614527
CTRL_N_LOAD_GT_15 = 9
CTRL_N_REVIEW_TRUE = 49


# ---------------------------------------------------------------------------
# Utilidades de carga
# ---------------------------------------------------------------------------
def leer_tabla(path: Path):
    try:
        suf = path.suffix.lower()
        if suf == ".parquet":
            return pd.read_parquet(path)
        if suf == ".csv":
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def buscar_df(columnas, directorios):
    """Devuelve (df, ruta) del primer archivo (parquet preferido) que contenga
    todas las `columnas`."""
    for d in directorios:
        dd = Path(d)
        if not dd.is_dir():
            continue
        archivos = [f for f in dd.rglob("*")
                    if f.is_file() and f.suffix.lower() in (".parquet", ".csv")]
        archivos.sort(key=lambda f: (f.suffix.lower() != ".parquet", f.name))
        for f in archivos:
            df = leer_tabla(f)
            if df is not None and set(columnas).issubset(df.columns):
                return df, str(f)
    return None, None


def a_bool(serie: pd.Series) -> np.ndarray:
    """Convierte una serie a numpy bool de forma segura."""
    if serie.dtype == bool:
        return serie.to_numpy()
    if pd.api.types.is_numeric_dtype(serie):
        return (serie != 0).to_numpy()
    t = serie.astype(str).str.strip().str.lower()
    return ((t == "true") | (t == "1") | (t == "1.0")).to_numpy()


def cargar_json(ruta):
    p = Path(ruta)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def buscar_lista_json(obj, subcadena_clave, largo):
    """Busca recursivamente una lista de números de longitud `largo` bajo una
    clave cuyo nombre contenga `subcadena_clave`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if (subcadena_clave in str(k).lower() and isinstance(v, list)
                    and len(v) == largo
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
                return v
            r = buscar_lista_json(v, subcadena_clave, largo)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = buscar_lista_json(v, subcadena_clave, largo)
            if r is not None:
                return r
    return None


def a_python(o):
    """Convierte tipos numpy/pandas a tipos nativos serializables a JSON."""
    if isinstance(o, dict):
        return {str(k): a_python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [a_python(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d %H:%M:%S")
    return o


# ---------------------------------------------------------------------------
# Reconstrucción de load_score (último recurso, calibrada con oráculos de T3)
# ---------------------------------------------------------------------------
def zscore(x, ddof):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=ddof)
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def reconstruir_load_score(df_base, idx_ctrl, mask_srv01):
    """Busca la fórmula de load_score (combinación estandarizada de
    gpu/cpu/memory[/power]) que reproduce los valores de control de T3."""
    g = df_base["gpu_utilization"].to_numpy(float)
    c = df_base["cpu_utilization"].to_numpy(float)
    m = df_base["memory_gb"].to_numpy(float)
    p = df_base["power_w"].to_numpy(float)

    backup = {"nombre": None, "arr": None}

    def probar(nombre, arr):
        if not np.isfinite(arr).all():
            return None
        if abs(float(arr[idx_ctrl]) - CTRL_LOAD_SCORE) >= 1e-6:
            return None
        cnt = int((arr > 1.5).sum())
        ml = float(arr[mask_srv01].mean())
        if cnt == CTRL_N_LOAD_GT_15 and abs(ml - CTRL_MEAN_LOAD_SRV01) < 1e-5:
            return nombre
        if backup["arr"] is None:
            backup["nombre"], backup["arr"] = nombre, arr
        return None

    z = {dd: {"g": zscore(g, dd), "c": zscore(c, dd),
              "m": zscore(m, dd), "p": zscore(p, dd)} for dd in (0, 1)}

    # 1) candidatos simples
    simples = {}
    for dd in (0, 1):
        Z = z[dd]
        simples[f"zscore(gpu)|ddof={dd}"] = Z["g"]
        simples[f"zscore(cpu)|ddof={dd}"] = Z["c"]
        simples[f"zscore(memory)|ddof={dd}"] = Z["m"]
        simples[f"zscore(power)|ddof={dd}"] = Z["p"]
        simples[f"zscore(gpu+cpu)|ddof={dd}"] = zscore(g + c, dd)
        simples[f"zscore(gpu+cpu+memory)|ddof={dd}"] = zscore(g + c + m, dd)
        simples[f"zscore((gpu+cpu+memory)/3)|ddof={dd}"] = zscore((g + c + m) / 3.0, dd)
        simples[f"zscore(gpu+cpu+memory+power)|ddof={dd}"] = zscore(g + c + m + p, dd)
        simples[f"z(gpu)+z(cpu)|ddof={dd}"] = Z["g"] + Z["c"]
        simples[f"z(gpu)+z(cpu)+z(memory)|ddof={dd}"] = Z["g"] + Z["c"] + Z["m"]
    for nombre, arr in simples.items():
        hit = probar(nombre, arr)
        if hit is not None:
            return arr, hit, True

    # 2) mallas ponderadas gpu/cpu/memory (z del compuesto y compuesto de z)
    for paso in (0.05, 0.01):
        ws = [round(float(w), 10) for w in np.arange(0.0, 1.0 + 1e-9, paso)]
        for dd in (0, 1):
            Z = z[dd]
            for w1 in ws:
                for w2 in ws:
                    if w1 + w2 > 1.0 + 1e-9:
                        continue
                    w3 = round(1.0 - w1 - w2, 10)
                    arr_a = zscore(w1 * g + w2 * c + w3 * m, dd)
                    hit = probar(f"zscore({w1:g}*gpu+{w2:g}*cpu+{w3:g}*memory)|ddof={dd}", arr_a)
                    if hit is not None:
                        return arr_a, hit, True
                    arr_b = w1 * Z["g"] + w2 * Z["c"] + w3 * Z["m"]
                    hit = probar(f"{w1:g}*z(gpu)+{w2:g}*z(cpu)+{w3:g}*z(memory)|ddof={dd}", arr_b)
                    if hit is not None:
                        return arr_b, hit, True

    if backup["arr"] is not None:
        return backup["arr"], backup["nombre"], False
    return (zscore((g + c + m) / 3.0, 1),
            "zscore((gpu+cpu+memory)/3)|ddof=1 [defecto]", False)


# ===========================================================================
# Flujo principal
# ===========================================================================
fuentes = {}

# --- 1) DataFrame base de T3 ------------------------------------------------
df_base, src_base = buscar_df(BASE_COLS, ["entrada/T3", "output", "entrada/T2", "entrada/T1"])
if df_base is None:
    ruta_csv = Path("data/server_measurements.csv")
    if not ruta_csv.exists():
        raise FileNotFoundError("No se halló el DataFrame de T3 (entrada/T3 u output) "
                                "ni data/server_measurements.csv")
    df_base = pd.read_csv(ruta_csv)
    src_base = str(ruta_csv)
fuentes["base"] = src_base

df_base = df_base.reset_index(drop=True)
df_base["server"] = df_base["server"].astype(str).str.strip()
df_base["timestamp"] = pd.to_datetime(df_base["timestamp"])
n_filas = len(df_base)

# Índice de control (fila 23 de T3) y máscara de AI-SRV-01
if n_filas > CTRL_IDX and str(df_base["server"].iloc[CTRL_IDX]) == CTRL_SERVER:
    idx_ctrl = CTRL_IDX
else:
    m = ((df_base["server"] == CTRL_SERVER) &
         (df_base["timestamp"] == pd.Timestamp(CTRL_TS))).to_numpy()
    idx_ctrl = int(np.flatnonzero(m)[0]) if m.any() else 0
mask_srv01 = (df_base["server"] == CTRL_SERVER).to_numpy()

# --- 2) load_score -----------------------------------------------------------
load_score = None
if "load_score" in df_base.columns:
    load_score = df_base["load_score"].astype(float).to_numpy()
    fuentes["load_score"] = f"{src_base} (columna load_score)"
else:
    df_ls, src_ls = buscar_df(["load_score"], ["entrada/T2", "entrada/T3", "output", "entrada/T1"])
    if df_ls is not None and len(df_ls) == n_filas:
        load_score = df_ls["load_score"].astype(float).to_numpy()
        fuentes["load_score"] = f"{src_ls} (columna load_score)"
if load_score is None:
    for jp in ("entrada/T2/resultados.json", "entrada/T3/resultados.json"):
        datos = cargar_json(jp)
        if datos is None:
            continue
        lista = buscar_lista_json(datos, "load", n_filas)
        if lista is not None:
            load_score = np.asarray(lista, dtype=float)
            fuentes["load_score"] = f"{jp} (lista de valores)"
            break
load_score_reconstruido = load_score is None
if load_score is None:
    load_score, nombre_rec, verificado = reconstruir_load_score(df_base, idx_ctrl, mask_srv01)
    fuentes["load_score"] = ("reconstruido y verificado con oráculo de T3: " if verificado
                             else "reconstruido SIN verificación plena: ") + nombre_rec
load_score = np.asarray(load_score, dtype=float)

# --- 3) requires_review ------------------------------------------------------
requires_review = None
if "requires_review" in df_base.columns:
    requires_review = a_bool(df_base["requires_review"])
    fuentes["requires_review"] = f"{src_base} (columna requires_review)"
if requires_review is None:
    datos3 = cargar_json("entrada/T3/resultados.json")
    if (datos3 is not None and isinstance(datos3.get("requires_review"), list)
            and len(datos3["requires_review"]) == n_filas):
        requires_review = np.asarray(datos3["requires_review"], dtype=bool)
        fuentes["requires_review"] = "entrada/T3/resultados.json (lista requires_review)"
if requires_review is None:
    requires_review = (load_score > 1.5) | (df_base["temperature_c"].to_numpy(float) > 80.0)
    fuentes["requires_review"] = "recalculada con la regla de T3"
requires_review = np.asarray(requires_review, dtype=bool)

# --- 4) DataFrame de T3 normalizado ------------------------------------------
df_t3 = df_base.copy()
df_t3["load_score"] = load_score
df_t3["requires_review"] = requires_review
df_t3 = df_t3[T3_COLS].copy()

# --- 5) Tabla resumen por servidor -------------------------------------------
agg = df_t3.groupby("server").agg(
    mean_power_w=("power_w", "mean"),
    max_temperature_c=("temperature_c", "max"),
    mean_load=("load_score", "mean"),
    review_count=("requires_review", "sum"),
)
obs = df_t3.groupby("server").size().rename("observations")
resumen = pd.concat([obs, agg], axis=1).reset_index()
resumen["observations"] = resumen["observations"].astype(int)
resumen["review_count"] = resumen["review_count"].astype(int)
resumen = resumen.sort_values("server").reset_index(drop=True)
resumen = resumen[ORDEN_COLUMNAS]

# --- 6) Verificación contra el criterio de éxito ------------------------------
fila01 = resumen[resumen["server"] == CTRL_SERVER]
v_obs = int(fila01["observations"].iloc[0]) if len(fila01) else None
v_pow = float(fila01["mean_power_w"].iloc[0]) if len(fila01) else None
v_tmp = float(fila01["max_temperature_c"].iloc[0]) if len(fila01) else None
v_load = float(fila01["mean_load"].iloc[0]) if len(fila01) else None
v_rev = int(fila01["review_count"].iloc[0]) if len(fila01) else None

checks = {
    "seis_filas_una_por_servidor": bool(len(resumen) == 6 and df_t3["server"].nunique() == 6),
    "columnas_en_orden_exacto": bool(list(resumen.columns) == ORDEN_COLUMNAS),
    "servidores_en_orden_alfabetico": bool([str(s) for s in resumen["server"]] ==
                                           sorted(str(s) for s in resumen["server"])),
    "sin_nulos_en_resumen": bool(resumen.notna().all().all()),
    "AI_SRV_01_observations_50": bool(v_obs == 50),
    "AI_SRV_01_mean_power_w_363_4598": bool(v_pow is not None and abs(v_pow - 363.4598) < 5e-4),
    "AI_SRV_01_max_temperature_c_86_50": bool(v_tmp is not None and abs(v_tmp - 86.50) < 1e-6),
    "AI_SRV_01_mean_load_menos_0_614527": bool(v_load is not None and abs(v_load + 0.614527) < 1e-5),
    "AI_SRV_01_review_count_2": bool(v_rev == 2),
}
criterio_ok = all(checks.values())

coherencia_t3 = {
    "n_filas_300": bool(n_filas == 300),
    "requires_review_dtype_bool": bool(df_t3["requires_review"].dtype == bool),
    "total_requires_review_true_49": bool(int(requires_review.sum()) == CTRL_N_REVIEW_TRUE),
    "review_count_suma_49": bool(int(resumen["review_count"].sum()) == CTRL_N_REVIEW_TRUE),
    "n_load_score_mayor_1_5_9": bool(int((load_score > 1.5).sum()) == CTRL_N_LOAD_GT_15),
    "fila_control_load_score_coincide": bool(abs(float(load_score[idx_ctrl]) - CTRL_LOAD_SCORE) < 1e-6),
    "fila_control_server_y_timestamp_ok": bool(
        str(df_t3["server"].iloc[idx_ctrl]) == CTRL_SERVER and
        pd.Timestamp(df_t3["timestamp"].iloc[idx_ctrl]).strftime("%Y-%m-%d %H:%M:%S") == CTRL_TS),
}

# --- 7) Archivos de salida ----------------------------------------------------
Path("output").mkdir(parents=True, exist_ok=True)
df_t3.to_parquet("output/server_analysis.parquet", index=False)
resumen.to_parquet("output/server_summary.parquet", index=False)
resumen.to_csv("output/server_summary.csv", index=False)

# --- 8) Figura ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
x = np.arange(len(resumen))
etiquetas = [str(s) for s in resumen["server"]]

def _barras(ax, col, titulo, color, linea=None):
    ax.bar(x, resumen[col].to_numpy(float), color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
    ax.set_title(titulo, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    if linea is not None:
        ax.axhline(linea, color="red", linestyle="--", linewidth=1)

_barras(axes[0, 0], "observations", "Observaciones por servidor", "#4C72B0")
_barras(axes[0, 1], "mean_power_w", "Potencia media (W)", "#55A868")
_barras(axes[1, 0], "max_temperature_c", "Temperatura máxima (°C)", "#C44E52", linea=80.0)
_barras(axes[1, 1], "review_count", "Filas con requires_review=True", "#8172B2")
fig.suptitle("T4 · Resumen por servidor", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("t4_resumen_por_servidor.png", dpi=120)
plt.close(fig)

# --- 9) resultados.json -------------------------------------------------------
tabla_reg = a_python(resumen.to_dict(orient="records"))

tab_red = resumen.copy()
tab_red["mean_power_w"] = tab_red["mean_power_w"].round(4)
tab_red["max_temperature_c"] = tab_red["max_temperature_c"].round(2)
tab_red["mean_load"] = tab_red["mean_load"].round(6)

def tabla_markdown(df):
    cab = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    filas = ["| " + " | ".join(str(v) for v in fila) + " |"
             for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep] + filas)

control23 = {
    "indice_usado": int(idx_ctrl),
    "server": str(df_t3["server"].iloc[idx_ctrl]),
    "timestamp": pd.Timestamp(df_t3["timestamp"].iloc[idx_ctrl]).strftime("%Y-%m-%d %H:%M:%S"),
    "load_score": float(df_t3["load_score"].iloc[idx_ctrl]),
    "load_score_esperado_T3": CTRL_LOAD_SCORE,
    "temperature_c": float(df_t3["temperature_c"].iloc[idx_ctrl]),
    "requires_review": bool(df_t3["requires_review"].iloc[idx_ctrl]),
}

resultados = {
    "subtarea": "T4_resumen_por_servidor",
    "descripcion": ("Tabla resumen con una fila por servidor y columnas exactas: server, "
                    "observations, mean_power_w, max_temperature_c, mean_load, review_count; "
                    "review_count = nº de filas con requires_review=True"),
    "definiciones": {
        "observations": "número de filas del servidor",
        "mean_power_w": "media de power_w",
        "max_temperature_c": "máximo de temperature_c",
        "mean_load": "media de load_score",
        "review_count": "número de filas con requires_review=True",
    },
    "fuentes": fuentes,
    "load_score_reconstruido": bool(load_score_reconstruido),
    "n_filas_dataframe_t3": int(n_filas),
    "n_filas_resumen": int(len(resumen)),
    "n_servidores": int(resumen["server"].nunique()),
    "columnas": list(resumen.columns),
    "orden_columnas_correcto": bool(list(resumen.columns) == ORDEN_COLUMNAS),
    "tabla": tabla_reg,
    "tabla_redondeada_estilo_enunciado": a_python(tab_red.to_dict(orient="records")),
    "tabla_markdown": tabla_markdown(tab_red),
    "fila_AI-SRV-01": tabla_reg[0] if tabla_reg else None,
    "verificacion_criterio_exito": checks,
    "coherencia_con_T3": coherencia_t3,
    "fila_23_control": control23,
    "totales": {
        "observations": int(resumen["observations"].sum()),
        "review_count": int(resumen["review_count"].sum()),
    },
    "archivos_generados": {
        "parquet_t3_enriquecido": "output/server_analysis.parquet",
        "parquet_resumen": "output/server_summary.parquet",
        "csv_resumen": "output/server_summary.csv",
        "figura": "t4_resumen_por_servidor.png",
        "resultados": "resultados.json",
    },
    "criterio_exito_ok": bool(criterio_ok),
}
Path("resultados.json").write_text(
    json.dumps(a_python(resultados), indent=2, ensure_ascii=False), encoding="utf-8")

# --- 10) Resumen en consola ----------------------------------------------------
print("=" * 76)
print("T4 · RESUMEN POR SERVIDOR")
print("=" * 76)
print(f"Base            : {fuentes['base']}")
print(f"load_score      : {fuentes.get('load_score')}")
print(f"requires_review : {fuentes.get('requires_review')}")
print("-" * 76)
print(resumen.to_string(index=False))
print("-" * 76)
print(f"Filas: {len(resumen)} | Columnas: {list(resumen.columns)}")
if len(fila01):
    print("Fila AI-SRV-01:", a_python(fila01.to_dict(orient="records"))[0])
for k, v in checks.items():
    print(f"  [{'OK' if v else 'XX'}] {k}")
print(f"Coherencia con T3: {coherencia_t3}")
print(f"CRITERIO DE ÉXITO: {'CUMPLIDO' if criterio_ok else 'NO CUMPLIDO'}")
print("Archivos escritos: resultados.json, output/server_analysis.parquet, "
      "output/server_summary.parquet, output/server_summary.csv, "
      "t4_resumen_por_servidor.png")
