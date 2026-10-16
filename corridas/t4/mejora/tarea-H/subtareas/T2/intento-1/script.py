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

# ------------------------------------------------------------------
# 1) Cargar la división 70/30 estratificada (random_state=42) de T1
# ------------------------------------------------------------------
t1_path = Path("entrada/T1/resultados.json")
n_train_t1, n_test_t1 = 398, 171
split_file = "T1_division_indices.parquet"
if t1_path.exists():
    with open(t1_path, encoding="utf-8") as f:
        t1 = json.load(f)
    n_train_t1 = int(t1.get("n_train", n_train_t1))
    n_test_t1 = int(t1.get("n_test", n_test_t1))
    split_file = t1.get("archivo_division", split_file)
split_path = Path("entrada/T1") / split_file

data = load_breast_cancer()
X, y = data.data, data.target
n_total = X.shape[0]

def _valid(tr, te):
    """Comprueba que tr/te forman una partición posicional válida (398/171)."""
    try:
        tr = np.asarray(tr, dtype=int)
        te = np.asarray(te, dtype=int)
    except (TypeError, ValueError):
        return False
    if len(tr) != n_train_t1 or len(te) != n_test_t1:
        return False
    if len(np.intersect1d(tr, te)) != 0:
        return False
    return np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(n_total))

train_idx, test_idx = None, None
if split_path.exists():
    try:
        df = pd.read_parquet(split_path)
        cols = list(df.columns)

        def col_with(*claves):
            for c in cols:
                if any(k in c.lower() for k in claves):
                    return c
            return None

        def interpretar(col):
            s = df[col]
            if s.dtype == bool:
                return np.flatnonzero(s.to_numpy())
            num = pd.to_numeric(s, errors="coerce")
            if num.isna().any():
                return num.dropna().to_numpy().astype(int)
            vals = num.to_numpy()
            if set(np.unique(vals)).issubset({0.0, 1.0}):
                return np.flatnonzero(vals != 0)
            return vals.astype(int)

        # Formato largo: una fila por ejemplo + columna que indica el conjunto
        split_col = col_with("conjunto", "split", "particion", "partición", "tipo", "rol", "role")
        if split_col is not None:
            lab = df[split_col].astype(str).str.lower()
            id_col = col_with("indice", "index", "idx") or cols[0]
            tr = df.loc[lab.str.contains("train|entrena"), id_col].to_numpy()
            te = df.loc[lab.str.contains("test|prueba"), id_col].to_numpy()
            if _valid(tr, te):
                train_idx, test_idx = tr.astype(int), te.astype(int)

        # Formato ancho: columnas 'train'/'test' con listas, índices o máscaras
        if train_idx is None:
            tr_col, te_col = col_with("train", "entrena"), col_with("test", "prueba")
            if tr_col is not None and te_col is not None:
                tr_cell, te_cell = df[tr_col].iloc[0], df[te_col].iloc[0]
                if isinstance(tr_cell, (list, tuple, np.ndarray)):
                    tr, te = np.asarray(tr_cell), np.asarray(te_cell)
                    if _valid(tr, te):
                        train_idx, test_idx = tr.astype(int), te.astype(int)
                else:
                    tr, te = interpretar(tr_col), interpretar(te_col)
                    if _valid(tr, te):
                        train_idx, test_idx = tr, te
    except Exception:
        train_idx, test_idx = None, None

if train_idx is None:
    # Respaldo determinístico: reproduce exactamente la división de T1
    train_idx, test_idx = train_test_split(
        np.arange(n_total), test_size=0.3, stratify=y, random_state=42
    )
    fuente = "recalculada de forma determinística (test_size=0.3, stratify=y, random_state=42), idéntica a la de T1"
else:
    fuente = f"índices cargados de {split_path.as_posix()}"

assert _valid(train_idx, test_idx), "La partición cargada/generada no es una partición válida"

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ------------------------------------------------------------------
# 2) Estandarización ajustada SOLO con el conjunto de entrenamiento
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3) Modelos: generativo (GaussianNB) y discriminativo (reg. logística)
# ------------------------------------------------------------------
nb = GaussianNB().fit(X_train_s, y_train)
lr = LogisticRegression(max_iter=5000, random_state=42).fit(X_train_s, y_train)

modelos = {"Naive Bayes gaussiano": nb, "Regresión logística": lr}
tabla = []
metricas = {}
for nombre, modelo in modelos.items():
    pred = modelo.predict(X_test_s)
    acc = float(accuracy_score(y_test, pred))
    f1m = float(f1_score(y_test, pred, average="macro"))
    tabla.append({"modelo": nombre, "accuracy_prueba": acc, "f1_macro_prueba": f1m})
    metricas[nombre] = (acc, f1m)

# ------------------------------------------------------------------
# 4) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T2",
    "dataset": "load_breast_cancer",
    "fuente_division": fuente,
    "n_train": int(len(train_idx)),
    "n_test": int(len(test_idx)),
    "tabla_metricas_prueba": tabla,
    "accuracy_prueba_naive_bayes_gaussiano": metricas["Naive Bayes gaussiano"][0],
    "f1_macro_prueba_naive_bayes_gaussiano": metricas["Naive Bayes gaussiano"][1],
    "accuracy_prueba_regresion_logistica": metricas["Regresión logística"][0],
    "f1_macro_prueba_regresion_logistica": metricas["Regresión logística"][1],
    "supuestos": [
        "Se usa exactamente la división 70/30 estratificada con random_state=42 producida por T1 (índices de entrada/T1/T1_division_indices.parquet).",
        "La estandarización (StandardScaler ajustado solo con el entrenamiento) se aplica a ambos modelos, incluido el Naive Bayes gaussiano, según la relación del material del curso; en todo caso las predicciones del NB gaussiano son invariantes a esta transformación afín por atributo.",
        "La regresión logística usa los parámetros por defecto de scikit-learn (C=1.0, solver lbfgs); max_iter=5000 solo garantiza convergencia y no se ajustó ningún hiperparámetro con el conjunto de prueba.",
    ],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 5) Figura de barras con las cuatro cifras
# ------------------------------------------------------------------
nombres = list(modelos.keys())
accs = [metricas[n][0] for n in nombres]
f1s = [metricas[n][1] for n in nombres]
pos = np.arange(len(nombres))

fig, ax = plt.subplots(figsize=(6.5, 4))
b1 = ax.bar(pos - 0.2, accs, 0.4, label="Accuracy (prueba)", color="#4C72B0")
b2 = ax.bar(pos + 0.2, f1s, 0.4, label="F1 macro (prueba)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
ax.set_xticks(pos)
ax.set_xticklabels(["Naive Bayes\ngaussiano", "Regresión\nlogística"])
ax.set_ylim(0, 1.12)
ax.set_ylabel("Valor en el conjunto de prueba")
ax.set_title("T2: generativo (NB gaussiano) vs. discriminativo (reg. logística)\n"
             "breast cancer · división 70/30 de T1 · evaluación en prueba")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("T2_metricas_prueba.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("T2 — Métricas en el conjunto de prueba (división de T1):")
for fila in tabla:
    print(f"  {fila['modelo']:<24} accuracy={fila['accuracy_prueba']:.4f}  F1 macro={fila['f1_macro_prueba']:.4f}")
print(f"División usada: {fuente}")
print("Archivos escritos: resultados.json, T2_metricas_prueba.png")
