#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Entrenar sobre la división de T1:
  - Naive Bayes gaussiano
  - Regresión logística con atributos estandarizados (escalador ajustado SOLO con entrenamiento)
y medir accuracy y F1 macro de cada modelo sobre el conjunto de prueba.
División: la única de la tarea (70/30 estratificada, random_state=42), tomada de entrada/T1/.
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

SEED = 42

# ------------------------------------------------------------------
# 1) Cargar la división de T1 (contrato: entrada/T1/resultados.json)
# ------------------------------------------------------------------
t1_path = Path("entrada") / "T1" / "resultados.json"
with t1_path.open("r", encoding="utf-8") as f:
    t1 = json.load(f)

idx_all = np.arange(load_breast_cancer().data.shape[0])

if "indices_entrenamiento" in t1 and "indices_prueba" in t1:
    idx_train = np.asarray(t1["indices_entrenamiento"], dtype=int)
    idx_test = np.asarray(t1["indices_prueba"], dtype=int)
    fuente_division = "entrada/T1/resultados.json (indices_entrenamiento / indices_prueba)"
else:  # respaldo defensivo: reproducir la división de T1
    idx_train, idx_test = train_test_split(
        idx_all, train_size=0.7, test_size=0.3,
        stratify=load_breast_cancer().target, random_state=SEED
    )
    fuente_division = "recalculada con train_test_split(0.7/0.3, stratify, random_state=42)"

# ------------------------------------------------------------------
# 2) Cargar el dataset y reconstruir las particiones de T1
# ------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
class_names = list(data.target_names)  # ['malignant', 'benign']

# Sanidad de la partición
assert len(idx_train) == len(np.unique(idx_train)), "índices de entrenamiento repetidos"
assert len(idx_test) == len(np.unique(idx_test)), "índices de prueba repetidos"
assert set(idx_train.tolist()).isdisjoint(set(idx_test.tolist())), "solapamiento train/test"
assert len(idx_train) + len(idx_test) == len(y), "la partición no cubre todo el dataset"

X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

# Verificación informativa (no cambia la partición usada): ¿coincide con la división canónica?
tr_ref, te_ref = train_test_split(idx_all, train_size=0.7, test_size=0.3,
                                  stratify=y, random_state=SEED)
split_match = (set(tr_ref.tolist()) == set(idx_train.tolist()) and
               set(te_ref.tolist()) == set(idx_test.tolist()))

# ------------------------------------------------------------------
# 3) Estandarización: el escalador se ajusta SOLO con el entrenamiento
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)   # nada de información de prueba aquí
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4) Entrenar y evaluar los dos clasificadores
# ------------------------------------------------------------------
modelos = {
    "GaussianNB": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=SEED),
}

filas = []
detalle = {}
for nombre, modelo in modelos.items():
    modelo.fit(Xtr, y_train)             # solo entrenamiento
    y_pred = modelo.predict(Xte)         # evaluación en prueba
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    rep = classification_report(y_test, y_pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    filas.append({"modelo": nombre, "accuracy_prueba": acc, "f1_macro_prueba": f1m})
    detalle[nombre] = {
        "accuracy_prueba": acc,
        "f1_macro_prueba": f1m,
        "n_errores_prueba": int((y_pred != y_test).sum()),
        "matriz_confusion_prueba": {
            "clases_orden": class_names,  # filas = real, columnas = predicho
            "matriz": cm.tolist(),
        },
        "precision_por_clase": {c: float(rep[c]["precision"]) for c in class_names},
        "recall_por_clase": {c: float(rep[c]["recall"]) for c in class_names},
    }

tabla = pd.DataFrame(filas)

# ------------------------------------------------------------------
# 5) Escribir resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "descripcion": ("Naive Bayes gaussiano y regresión logística (atributos estandarizados) "
                    "entrenados con la división 70/30 estratificada de T1; accuracy y F1 macro "
                    "medidos sobre el conjunto de prueba."),
    "division_fuente": fuente_division,
    "n_entrenamiento": int(len(idx_train)),
    "n_prueba": int(len(idx_test)),
    "conteo_por_clase_prueba": {c: int((y_test == i).sum()) for i, c in enumerate(class_names)},
    "preprocesamiento": ("StandardScaler ajustado únicamente con el conjunto de entrenamiento; "
                         "el mismo escalador transforma entrenamiento y prueba."),
    "config_modelos": {
        "GaussianNB": "GaussianNB() con parámetros por defecto, sobre atributos estandarizados",
        "LogisticRegression": "LogisticRegression(max_iter=5000, random_state=42), sobre atributos estandarizados",
    },
    "tabla": filas,  # TABLA exigida: accuracy y F1 macro en prueba para los dos modelos
    "accuracy_prueba": {r["modelo"]: r["accuracy_prueba"] for r in filas},
    "f1_macro_prueba": {r["modelo"]: r["f1_macro_prueba"] for r in filas},
    "detalle_por_modelo": detalle,
    "verificacion_division_t1": {
        "coincide_con_train_test_split_recalculado": bool(split_match),
        "train_size": 0.7,
        "test_size": 0.3,
        "estratificada_por_clase": True,
        "random_state": SEED,
    },
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# Tablas de apoyo para el reporte
tabla.to_csv("tabla_t2_metricas.csv", index=False)
md = ["| modelo | accuracy_prueba | f1_macro_prueba |",
      "|---|---|---|"]
for r in filas:
    md.append(f"| {r['modelo']} | {r['accuracy_prueba']:.4f} | {r['f1_macro_prueba']:.4f} |")
with open("tabla_t2_metricas.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

# ------------------------------------------------------------------
# 6) Figuras (PNG)
# ------------------------------------------------------------------
# Figura 1: barras agrupadas de métricas
fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = np.arange(len(filas))
w = 0.35
accs = [r["accuracy_prueba"] for r in filas]
f1s = [r["f1_macro_prueba"] for r in filas]
b1 = ax.bar(xpos - w / 2, accs, w, label="Accuracy (prueba)", color="#4C72B0")
b2 = ax.bar(xpos + w / 2, f1s, w, label="F1 macro (prueba)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.annotate(f"{b.get_height():.4f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels([r["modelo"] for r in filas])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Valor en el conjunto de prueba")
ax.set_title("T2: accuracy y F1 macro en prueba\n(división 70/30 estratificada de T1, breast_cancer)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("t2_metricas_por_modelo.png", dpi=120)
plt.close(fig)

# Figura 2: la tabla como imagen
fig2, ax2 = plt.subplots(figsize=(6.5, 2.3))
ax2.axis("off")
celdas = [[f"{r['accuracy_prueba']:.4f}", f"{r['f1_macro_prueba']:.4f}"] for r in filas]
tbl = ax2.table(cellText=celdas,
                rowLabels=[r["modelo"] for r in filas],
                colLabels=["Accuracy (prueba)", "F1 macro (prueba)"],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.7)
ax2.set_title("T2: métricas en el conjunto de prueba (división de T1)")
fig2.tight_layout()
fig2.savefig("t2_tabla_metricas.png", dpi=120)
plt.close(fig2)

# ------------------------------------------------------------------
# 7) Resumen
# ------------------------------------------------------------------
print("=== T2: Generativo (GaussianNB) vs discriminativo (Regresión Logística) ===")
print(f"División usada: {fuente_division}")
print(f"n_entrenamiento={len(idx_train)}  n_prueba={len(idx_test)}")
print(f"Escalador: StandardScaler ajustado solo con entrenamiento "
      f"(medias train usadas; prueba nunca usada para ajustar).")
print(f"Verificación: la división coincide con train_test_split(0.7/0.3, stratify, seed=42): {split_match}")
print("\nTabla de métricas en PRUEBA:")
print(tabla.to_string(index=False))
for nombre, d in detalle.items():
    print(f"\n{nombre}: errores en prueba = {d['n_errores_prueba']}")
    print(f"  matriz de confusión (filas=real [malignant, benign], cols=predicho): {d['matriz_confusion_prueba']['matriz']}")
print("\nArchivos generados: resultados.json, tabla_t2_metricas.csv, tabla_t2_metricas.md, "
      "t2_metricas_por_modelo.png, t2_tabla_metricas.png")
