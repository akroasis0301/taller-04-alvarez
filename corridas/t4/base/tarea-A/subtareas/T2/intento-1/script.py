# -*- coding: utf-8 -*-
"""
T2 — Parte 2 de la Tarea A (generativo contra discriminativo).
Entrena un Naive Bayes gaussiano y una regresión logística (atributos estandarizados
con un escalador ajustado SOLO con el entrenamiento) sobre la división 70/30
estratificada (random_state=42) fijada en T1, y reporta accuracy y F1 macro de cada
modelo sobre el conjunto de prueba en una tabla.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SEMILLA = 42
TEST_SIZE = 0.3

# ----------------------------------------------------------------------
# 0. Valores de referencia de la división de T1 (se afinan con su
#    resultados.json si está disponible)
# ----------------------------------------------------------------------
n_train_esp, n_test_esp = 398, 171
conteo_train_esp = {"malignant": 148, "benign": 250}
conteo_test_esp = {"malignant": 64, "benign": 107}
p_t1 = Path("entrada/T1/resultados.json")
if p_t1.exists():
    try:
        division_t1 = json.loads(p_t1.read_text(encoding="utf-8")).get("division", {})
        n_train_esp = int(division_t1.get("n_train", n_train_esp))
        n_test_esp = int(division_t1.get("n_test", n_test_esp))
        if isinstance(division_t1.get("conteo_por_clase_train"), dict):
            conteo_train_esp = {str(k).lower(): int(v) for k, v in division_t1["conteo_por_clase_train"].items()}
        if isinstance(division_t1.get("conteo_por_clase_test"), dict):
            conteo_test_esp = {str(k).lower(): int(v) for k, v in division_t1["conteo_por_clase_test"].items()}
    except Exception:
        pass

# ----------------------------------------------------------------------
# 1. Recuperar la división entrenamiento/prueba de T1
# ----------------------------------------------------------------------
CANDIDATOS_TARGET = ["target", "clase", "diagnosis", "diagnostico", "label", "class", "y", "objetivo"]
MAPAS_ETIQUETAS = ({"malignant": 0, "benign": 1}, {"m": 0, "b": 1}, {"0": 0, "1": 1})


def _limpiar_df(df):
    df = df.copy()
    utiles = [c for c in df.columns
              if str(c).strip().lower() != "__index_level_0__"
              and not str(c).strip().lower().startswith("unnamed")]
    return df[utiles]


def _separar_xy(df):
    col = next((c for c in df.columns if str(c).strip().lower() in CANDIDATOS_TARGET), df.columns[-1])
    y_raw = df[col]
    X_df = df.drop(columns=[col])
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int).to_numpy()
    else:
        y_str = y_raw.astype(str).str.strip().str.lower()
        y = None
        for mapa in MAPAS_ETIQUETAS:
            if set(y_str.unique()) <= set(mapa):
                y = y_str.map(mapa).astype(int).to_numpy()
                break
        if y is None:
            y = pd.factorize(y_str)[0].astype(int)
    X = X_df.to_numpy(dtype=float)
    return X, y, str(col), [str(c) for c in X_df.columns]


def _conteos_por_clase(y):
    unicos, conteos = np.unique(np.asarray(y), return_counts=True)
    return {int(u): int(c) for u, c in zip(unicos, conteos)}


def _division_valida(Xtr, ytr, Xte, yte):
    if Xtr.shape[0] != n_train_esp or Xte.shape[0] != n_test_esp:
        return False
    if Xtr.shape[1] != 30 or Xte.shape[1] != 30:
        return False
    pares = (
        ({0: conteo_train_esp.get("malignant"), 1: conteo_train_esp.get("benign")}, _conteos_por_clase(ytr)),
        ({0: conteo_test_esp.get("malignant"), 1: conteo_test_esp.get("benign")}, _conteos_por_clase(yte)),
    )
    for esperado, real in pares:
        for k, v in esperado.items():
            if v is not None and real.get(k) != v:
                return False
    return True


def cargar_division_t1():
    bases = [Path("datos"), Path("entrada/T1/datos"), Path("entrada/T1")]
    # a) tablas train/test (parquet o csv) generadas por T1
    for base in bases:
        for ext, lector in (("parquet", pd.read_parquet), ("csv", pd.read_csv)):
            f_tr, f_te = base / f"train.{ext}", base / f"test.{ext}"
            if f_tr.exists() and f_te.exists():
                try:
                    Xtr, ytr, col_tr, nombres = _separar_xy(_limpiar_df(lector(f_tr)))
                    Xte, yte, _, _ = _separar_xy(_limpiar_df(lector(f_te)))
                    if _division_valida(Xtr, ytr, Xte, yte):
                        return Xtr, ytr, Xte, yte, nombres, f"archivos de T1: {f_tr} y {f_te} (objetivo: '{col_tr}')"
                except Exception:
                    continue
    # b) split.npz de T1
    for base in bases:
        f_npz = base / "split.npz"
        if f_npz.exists():
            try:
                z = np.load(f_npz)
                claves = {k.lower(): k for k in z.files}

                def toma(*nombres):
                    for n in nombres:
                        if n in claves:
                            return z[claves[n]]
                    return None

                Xtr, Xte = toma("x_train", "xtrain", "xtr"), toma("x_test", "xtest", "xte")
                ytr, yte = toma("y_train", "ytrain", "ytr"), toma("y_test", "ytest", "yte")
                if all(v is not None for v in (Xtr, Xte, ytr, yte)):
                    Xtr, Xte = np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float)
                    ytr, yte = np.asarray(ytr).astype(int), np.asarray(yte).astype(int)
                    if _division_valida(Xtr, ytr, Xte, yte):
                        return Xtr, ytr, Xte, yte, None, f"archivos de T1: {f_npz}"
            except Exception:
                continue
    # c) reconstrucción determinista de la división de T1
    datos = load_breast_cancer()
    X, y = datos.data.astype(float), datos.target.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEMILLA)
    return (Xtr, ytr, Xte, yte, [str(n) for n in datos.feature_names],
            "reconstrucción determinista de la división de T1: load_breast_cancer + "
            "train_test_split(test_size=0.3, stratify=y, random_state=42)")


Xtr, ytr, Xte, yte, nombres_atributos, fuente_division = cargar_division_t1()

# ----------------------------------------------------------------------
# 2. Estandarización: el escalador se ajusta SOLO con el entrenamiento
# ----------------------------------------------------------------------
escalador = StandardScaler().fit(Xtr)
Xtr_s = escalador.transform(Xtr)
Xte_s = escalador.transform(Xte)

# ----------------------------------------------------------------------
# 3. Entrenamiento de los dos clasificadores y evaluación en prueba
# ----------------------------------------------------------------------
modelos = {
    "Naive Bayes gaussiano": GaussianNB(),
    "Regresión logística": LogisticRegression(max_iter=5000, random_state=SEMILLA),
}

filas = []
for nombre, modelo in modelos.items():
    modelo.fit(Xtr_s, ytr)
    pred = modelo.predict(Xte_s)
    filas.append({
        "modelo": nombre,
        "accuracy_prueba": float(accuracy_score(yte, pred)),
        "f1_macro_prueba": float(f1_score(yte, pred, average="macro")),
    })

tabla = pd.DataFrame(filas)
tabla.to_csv("t2_tabla_metricas.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------
# 4. Figura comparativa
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xpos = np.arange(len(tabla))
ancho = 0.35
barras_acc = ax.bar(xpos - ancho / 2, tabla["accuracy_prueba"], ancho, label="Accuracy", color="#4C72B0")
barras_f1 = ax.bar(xpos + ancho / 2, tabla["f1_macro_prueba"], ancho, label="F1 macro", color="#DD8452")
lim_inf = max(0.0, tabla[["accuracy_prueba", "f1_macro_prueba"]].to_numpy().min() - 0.05)
ax.set_ylim(lim_inf, 1.0)
for grupo in (barras_acc, barras_f1):
    for b in grupo:
        ax.annotate(f"{b.get_height():.4f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(tabla["modelo"])
ax.set_ylabel("Puntaje en el conjunto de prueba")
ax.set_title("Parte 2 — Accuracy y F1 macro en prueba (división 70/30 de T1, random_state=42)")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.savefig("t2_comparacion_modelos.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5. resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": ("Naive Bayes gaussiano y regresión logística (atributos estandarizados con "
                    "escalador ajustado solo con el entrenamiento) evaluados con accuracy y F1 "
                    "macro sobre el conjunto de prueba de la división de T1"),
    "division_usada": {
        "origen": fuente_division,
        "test_size": TEST_SIZE,
        "estratificada_por_clase": True,
        "random_state": SEMILLA,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_atributos": int(Xtr.shape[1]),
        "conteo_por_clase_train": {"malignant": int((ytr == 0).sum()), "benign": int((ytr == 1).sum())},
        "conteo_por_clase_test": {"malignant": int((yte == 0).sum()), "benign": int((yte == 1).sum())},
    },
    "preprocesamiento": {
        "escalador": "StandardScaler",
        "ajustado_solo_con_entrenamiento": True,
        "aplicado_a": list(modelos.keys()),
    },
    "modelos": {
        "naive_bayes_gaussiano": {"clase": "sklearn.naive_bayes.GaussianNB", "hiperparametros": {}},
        "regresion_logistica": {"clase": "sklearn.linear_model.LogisticRegression",
                                 "hiperparametros": {"max_iter": 5000, "random_state": SEMILLA}},
    },
    "tabla_metricas_prueba": filas,
    "accuracy_prueba": {
        "naive_bayes_gaussiano": filas[0]["accuracy_prueba"],
        "regresion_logistica": filas[1]["accuracy_prueba"],
    },
    "f1_macro_prueba": {
        "naive_bayes_gaussiano": filas[0]["f1_macro_prueba"],
        "regresion_logistica": filas[1]["f1_macro_prueba"],
    },
    "archivos_generados": {
        "tabla_csv": "t2_tabla_metricas.csv",
        "figura": "t2_comparacion_modelos.png",
    },
}
if nombres_atributos is not None:
    resultados["division_usada"]["nombres_atributos"] = nombres_atributos

Path("resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

# ----------------------------------------------------------------------
# 6. Resumen
# ----------------------------------------------------------------------
print("T2 — Parte 2: generativo (Naive Bayes gaussiano) contra discriminativo (regresión logística)")
print(f"División usada: {fuente_division}")
print(f"n_train={Xtr.shape[0]}  n_test={Xte.shape[0]}  n_atributos={Xtr.shape[1]}")
print("Escalador: StandardScaler ajustado solo con el conjunto de entrenamiento")
print("\nTabla de métricas sobre el conjunto de prueba:")
print(tabla.to_string(index=False))
print("\nArchivos generados: resultados.json, t2_tabla_metricas.csv, t2_comparacion_modelos.png")
