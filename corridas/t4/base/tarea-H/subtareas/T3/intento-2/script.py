#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: Naive Bayes gaussiano (generativo) vs regresión logística
(discriminativo) sobre la división 70/30 estratificada de T1 (Breast Cancer Wisconsin).

- Submuestras estratificadas del 5%, 10%, 25%, 50% y 100% del entrenamiento (random_state=42).
- Evaluación SIEMPRE sobre el mismo conjunto de prueba de T1 (nunca se usa para ajustar nada).
- El escalador se ajusta únicamente con la submuestra de entrenamiento de cada corrida.
- Figura: curva_aprendizaje.png ; cifras: resultados.json
"""

import json
from pathlib import Path

import numpy as np
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
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
ETIQUETAS = ["5%", "10%", "25%", "50%", "100%"]
NOMBRES_MODELOS = ["GaussianNB", "LogisticRegression"]
NOMBRE_FIGURA = "curva_aprendizaje.png"

# ------------------------------------------------------------------
# 1) División de T1 (la única división de toda la tarea)
# ------------------------------------------------------------------
with open(Path("entrada") / "T1" / "resultados.json", "r", encoding="utf-8") as fh:
    t1 = json.load(fh)

idx_train = np.asarray(t1["indices_entrenamiento"], dtype=int)
idx_test = np.asarray(t1["indices_prueba"], dtype=int)

# ------------------------------------------------------------------
# 2) Datos
# ------------------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)  # y: 0 = malignant, 1 = benign
n_total = X.shape[0]

X_train_all = X[idx_train]
X_test = X[idx_test]          # conjunto de prueba fijo de T1 (nunca se usa para ajustar)
y_train_all = y[idx_train]
y_test = y[idx_test]

# Verificación: la división de T1 coincide con train_test_split(test_size=0.3, stratify, rs=42)
idx_ref = np.arange(n_total)
tr_ref, te_ref = train_test_split(idx_ref, test_size=0.3, stratify=y, random_state=SEMILLA)
division_coincide_t1 = bool(
    np.array_equal(np.sort(tr_ref), np.sort(idx_train))
    and np.array_equal(np.sort(te_ref), np.sort(idx_test))
)

# ------------------------------------------------------------------
# 3) Entrenamiento con submuestras estratificadas y evaluación en prueba fija
# ------------------------------------------------------------------
def entrenar_y_evaluar(X_sub, y_sub, X_prueba, y_prueba):
    """Escalador ajustado SOLO con la submuestra de entrenamiento; modelos de T2."""
    scaler = StandardScaler().fit(X_sub)
    X_sub_s = scaler.transform(X_sub)
    X_prueba_s = scaler.transform(X_prueba)
    modelos = {
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=SEMILLA),
    }
    acc, f1m = {}, {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_sub_s, y_sub)
        pred = modelo.predict(X_prueba_s)
        acc[nombre] = float(accuracy_score(y_prueba, pred))
        f1m[nombre] = float(f1_score(y_prueba, pred, average="macro"))
    return acc, f1m


filas = []
for frac, etiqueta in zip(FRACCIONES, ETIQUETAS):
    if frac >= 1.0:
        sub_idx = idx_train.copy()  # 100%: entrenamiento completo
    else:
        pos = np.arange(len(idx_train))
        pos_sub, _ = train_test_split(
            pos, train_size=frac, stratify=y_train_all, random_state=SEMILLA
        )
        sub_idx = idx_train[pos_sub]

    X_sub, y_sub = X[sub_idx], y[sub_idx]
    acc, f1m = entrenar_y_evaluar(X_sub, y_sub, X_test, y_test)
    filas.append(
        {
            "fraccion": frac,
            "fraccion_etiqueta": etiqueta,
            "n_entrenamiento_submuestra": int(len(sub_idx)),
            "conteo_por_clase_submuestra": {
                "malignant": int(np.sum(y_sub == 0)),
                "benign": int(np.sum(y_sub == 1)),
            },
            "indices_submuestra": [int(i) for i in sub_idx],
            "accuracy_prueba": acc,
            "f1_macro_prueba": f1m,
        }
    )

# ------------------------------------------------------------------
# 4) Figura: exactitud vs número de ejemplos de entrenamiento (una curva por modelo)
# ------------------------------------------------------------------
n_vals = [f["n_entrenamiento_submuestra"] for f in filas]
colores = {"GaussianNB": "tab:blue", "LogisticRegression": "tab:orange"}
marcas = {"GaussianNB": "o", "LogisticRegression": "s"}
offsets = {"GaussianNB": (0, -14), "LogisticRegression": (0, 9)}

todas_acc = [f["accuracy_prueba"][m] for f in filas for m in NOMBRES_MODELOS]

plt.figure(figsize=(8.5, 5.5))
for nombre in NOMBRES_MODELOS:
    accs = [f["accuracy_prueba"][nombre] for f in filas]
    tipo = "generativo" if nombre == "GaussianNB" else "discriminativo"
    plt.plot(
        n_vals, accs,
        marker=marcas[nombre], color=colores[nombre],
        label=f"{nombre} ({tipo})",
    )
    for n, a in zip(n_vals, accs):
        plt.annotate(
            f"{a:.3f}", (n, a), textcoords="offset points",
            xytext=offsets[nombre], ha="center", fontsize=8,
        )

plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("Exactitud (accuracy) en el conjunto de prueba")
plt.title("Curva de aprendizaje — Breast Cancer (división 70/30 de T1, prueba fija)")
plt.xticks(n_vals)
plt.ylim(max(0.0, min(todas_acc) - 0.06), 1.01)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(NOMBRE_FIGURA, dpi=120)
plt.close()

# ------------------------------------------------------------------
# 5) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
accuracy_por_modelo = {m: [f["accuracy_prueba"][m] for f in filas] for m in NOMBRES_MODELOS}
f1_por_modelo = {m: [f["f1_macro_prueba"][m] for f in filas] for m in NOMBRES_MODELOS}

# Verificación opcional: la corrida al 100% debe reproducir los resultados de T2
verif_t2 = {"comparado": False}
try:
    with open(Path("entrada") / "T2" / "resultados.json", "r", encoding="utf-8") as fh:
        t2 = json.load(fh)
    acc100 = filas[-1]["accuracy_prueba"]
    verif_t2 = {
        "comparado": True,
        "nota": "La corrida con 100% del entrenamiento debe reproducir los resultados de T2.",
        "modelos": {
            m: {
                "accuracy_100pct_T3": acc100[m],
                "accuracy_T2": t2["accuracy_prueba"][m],
                "coincide": bool(abs(acc100[m] - t2["accuracy_prueba"][m]) < 1e-12),
            }
            for m in NOMBRES_MODELOS
        },
    }
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    pass

salida = {
    "subtarea": "T3",
    "descripcion": (
        "Curva de aprendizaje: GaussianNB (generativo) y LogisticRegression (discriminativo) "
        "entrenados con submuestras estratificadas del 5%, 10%, 25%, 50% y 100% del entrenamiento "
        "de la división 70/30 de T1 (random_state=42); evaluación siempre sobre el mismo conjunto "
        "de prueba."
    ),
    "division_fuente": "entrada/T1/resultados.json (indices_entrenamiento / indices_prueba)",
    "n_entrenamiento_total": int(len(idx_train)),
    "n_prueba": int(len(idx_test)),
    "conteo_por_clase_prueba": {
        "malignant": int(np.sum(y_test == 0)),
        "benign": int(np.sum(y_test == 1)),
    },
    "fracciones": FRACCIONES,
    "random_state_submuestreo": SEMILLA,
    "metodo_submuestreo": (
        "train_test_split(train_size=fraccion, stratify=y_entrenamiento, random_state=42) "
        "aplicado al conjunto de entrenamiento de T1; para 100% se usa el entrenamiento completo. "
        "n_submuestra = floor(fraccion * n_entrenamiento) (redondeo de sklearn)."
    ),
    "preprocesamiento": (
        "StandardScaler ajustado únicamente con la submuestra de entrenamiento de cada corrida "
        "(nunca con prueba); el mismo escalador transforma submuestra y prueba."
    ),
    "config_modelos": {
        "GaussianNB": "GaussianNB() con parámetros por defecto, sobre atributos estandarizados",
        "LogisticRegression": "LogisticRegression(max_iter=5000, random_state=42), sobre atributos estandarizados",
    },
    "n_entrenamiento_por_fraccion": n_vals,
    "resultados_por_fraccion": filas,
    "accuracy_prueba_por_modelo": accuracy_por_modelo,
    "f1_macro_prueba_por_modelo": f1_por_modelo,
    "figura_png": NOMBRE_FIGURA,
    "verificaciones": {
        "division_coincide_con_T1": division_coincide_t1,
        "reproduccion_T2_al_100pct": verif_t2,
    },
}

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(salida, fh, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print(f"T3 — Curva de aprendizaje (división 70/30 de T1; prueba fija de {len(idx_test)} ejemplos)")
print(f"{'fracción':>9} | {'n_train':>7} | {'Acc GaussianNB':>14} | {'Acc LogReg':>10} | "
      f"{'F1m GaussianNB':>14} | {'F1m LogReg':>10}")
for fila in filas:
    print(
        f"{fila['fraccion_etiqueta']:>9} | {fila['n_entrenamiento_submuestra']:>7} | "
        f"{fila['accuracy_prueba']['GaussianNB']:>14.4f} | "
        f"{fila['accuracy_prueba']['LogisticRegression']:>10.4f} | "
        f"{fila['f1_macro_prueba']['GaussianNB']:>14.4f} | "
        f"{fila['f1_macro_prueba']['LogisticRegression']:>10.4f}"
    )
print(f"División de T1 verificada: {division_coincide_t1}")
print(f"Figura guardada: {NOMBRE_FIGURA} (existe: {Path(NOMBRE_FIGURA).exists()})")
print("Cifras escritas en: resultados.json")
