# T3 — Curva de aprendizaje: NB gaussiano vs regresión logística
# Sobre la MISMA división 70/30 estratificada de T1 (índices en entrada/T1/),
# submuestras estratificadas del 5/10/25/50/100 % del entrenamiento (random_state=42),
# evaluación siempre sobre el mismo conjunto de prueba.

import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------ datos
data = load_breast_cancer()
X, y = data.data, data.target

# Tamaños esperados de la división de T1 (leídos de su resultados.json si existe)
n_train_esp, n_test_esp = 398, 171
try:
    with open("entrada/T1/resultados.json", encoding="utf-8") as f:
        t1 = json.load(f)
    n_train_esp = int(t1["n_train"])
    n_test_esp = int(t1["n_test"])
except Exception:
    pass


def _col_a_indices(df, col):
    vals = df[col].dropna().tolist()
    if vals and isinstance(vals[0], (list, tuple, np.ndarray)):
        return np.concatenate([np.asarray(v, dtype=int).ravel() for v in vals])
    return df[col].to_numpy(dtype=int)


def extraer_indices(df):
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}
    tr_col = next((c for c in cols if "train" in low[c] or "entrenam" in low[c]), None)
    te_col = next((c for c in cols if "test" in low[c] or "prueba" in low[c]), None)
    if tr_col is not None and te_col is not None:
        return _col_a_indices(df, tr_col), _col_a_indices(df, te_col)
    # formato largo: una columna de etiquetas (train/test) y una de índices
    for c in cols:
        vals = df[c].astype(str).str.lower()
        m_tr = vals.str.contains("train|entrenam").to_numpy()
        m_te = vals.str.contains("test|prueba").to_numpy()
        if m_tr.any() and m_te.any():
            idx_cols = [k for k in cols if k != c]
            if not idx_cols:
                idx = np.asarray(df.index)
                return idx[m_tr].astype(int), idx[m_te].astype(int)
            ic = next((k for k in idx_cols
                       if any(t in str(k).lower() for t in ("index", "indice", "idx"))), idx_cols[0])
            return df.loc[m_tr, ic].to_numpy(dtype=int), df.loc[m_te, ic].to_numpy(dtype=int)
    raise ValueError("No se pudieron identificar los índices de la división en el parquet.")


# ------------------------------------------------- misma división de la Parte 1
tr_idx, te_idx = None, None
ruta_div = Path("entrada/T1/T1_division_indices.parquet")
if ruta_div.exists():
    try:
        df_div = pd.read_parquet(ruta_div)
        tr_idx, te_idx = extraer_indices(df_div)
    except Exception as e:
        print(f"Aviso: no se pudo interpretar {ruta_div} ({e}); se recalcula la división.")
if tr_idx is None:
    # Reproduce exactamente la división de T1: 70/30 estratificada, random_state=42
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=42, stratify=y)

tr_idx = np.sort(np.asarray(tr_idx, dtype=int))
te_idx = np.sort(np.asarray(te_idx, dtype=int))

assert len(tr_idx) == n_train_esp, f"n_train {len(tr_idx)} != esperado {n_train_esp}"
assert len(te_idx) == n_test_esp, f"n_test {len(te_idx)} != esperado {n_test_esp}"
assert np.intersect1d(tr_idx, te_idx).size == 0, "train y prueba se solapan"
assert np.union1d(tr_idx, te_idx).size == len(y), "la división no cubre todos los ejemplos"

X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]

# --------------------------------------------- curva de aprendizaje (T3)
fracciones = [0.05, 0.10, 0.25, 0.50, 1.00]
filas = []
for p in fracciones:
    if p >= 1.0:
        pos_sub = np.arange(len(y_tr))  # 100 % = entrenamiento completo
    else:
        pos_sub, _ = train_test_split(np.arange(len(y_tr)), train_size=p,
                                      stratify=y_tr, random_state=42)
    Xs, ys = X_tr[pos_sub], y_tr[pos_sub]

    # El escalador se ajusta SOLO con la submuestra de entrenamiento de este punto
    scaler = StandardScaler().fit(Xs)
    Xs_s = scaler.transform(Xs)
    Xte_s = scaler.transform(X_te)

    nb = GaussianNB().fit(Xs_s, ys)
    lr = LogisticRegression(max_iter=5000, random_state=42).fit(Xs_s, ys)

    acc_nb = float(accuracy_score(y_te, nb.predict(Xte_s)))
    acc_lr = float(accuracy_score(y_te, lr.predict(Xte_s)))
    conteo = np.bincount(ys.astype(int), minlength=2)

    filas.append({
        "porcentaje": p,
        "n_ejemplos": int(len(ys)),
        "n_malignant": int(conteo[0]),
        "n_benign": int(conteo[1]),
        "accuracy_naive_bayes": acc_nb,
        "accuracy_regresion_logistica": acc_lr,
    })
    print(f"{p*100:5.0f}%  n={len(ys):3d}  "
          f"acc NB={acc_nb:.4f}  acc LR={acc_lr:.4f}")

tabla = pd.DataFrame(filas)
n_ejemplos = tabla["n_ejemplos"].to_numpy()

# ------------------------------------------------------------- figura PNG
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_ejemplos, tabla["accuracy_naive_bayes"], "o-",
        label="Naive Bayes gaussiano (generativo)", color="tab:blue")
ax.plot(n_ejemplos, tabla["accuracy_regresion_logistica"], "s-",
        label="Regresión logística (discriminativo)", color="tab:orange")
ax.set_xlabel("Número de ejemplos de entrenamiento")
ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
ax.set_title("Curva de aprendizaje: generativo vs discriminativo\n"
             "(Breast Cancer, misma división 70/30, prueba fija de 171 ejemplos)")
ax.set_xticks(n_ejemplos)
ax.set_ylim(0.85, 1.01)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
plt.savefig("T3_curva_aprendizaje.png", dpi=120)
plt.close(fig)

# --------------------------------------------------------- resultados.json
claves_pct = ["5%", "10%", "25%", "50%", "100%"]
resultados = {
    "subtarea": "T3",
    "dataset": "load_breast_cancer",
    "fuente_division": "índices cargados de entrada/T1/T1_division_indices.parquet",
    "n_train_total": int(len(y_tr)),
    "n_test": int(len(y_te)),
    "porcentajes_entrenamiento": fracciones,
    "n_ejemplos_entrenamiento": [int(v) for v in n_ejemplos],
    "n_ejemplos_entrenamiento_por_porcentaje": {
        c: int(n) for c, n in zip(claves_pct, n_ejemplos)
    },
    "accuracy_naive_bayes": [float(v) for v in tabla["accuracy_naive_bayes"]],
    "accuracy_regresion_logistica": [float(v) for v in tabla["accuracy_regresion_logistica"]],
    "accuracy_naive_bayes_por_porcentaje": {
        c: float(v) for c, v in zip(claves_pct, tabla["accuracy_naive_bayes"])
    },
    "accuracy_regresion_logistica_por_porcentaje": {
        c: float(v) for c, v in zip(claves_pct, tabla["accuracy_regresion_logistica"])
    },
    "curva_aprendizaje": filas,
    "archivo_figura": "T3_curva_aprendizaje.png",
    "supuestos": [
        "La submuestra del 100 % es el conjunto de entrenamiento completo (398 ejemplos) sin nueva división, porque train_test_split no admite train_size=1.0.",
        "En cada punto de la curva el StandardScaler se reajusta solo con la submuestra de entrenamiento de ese punto; el conjunto de prueba se transforma con ese escalador y nunca se usa para ajustar nada.",
        "Los tamaños reales de las submuestras son ceil(fracción × 398): 20, 40, 100, 199 y 398 ejemplos (comportamiento de scikit-learn con train_size fraccional).",
        "Las submuestras son estratificadas por clase con random_state=42, y la regresión logística mantiene los parámetros de T2 (por defecto, max_iter=5000, random_state=42); no se ajustó ningún hiperparámetro con el conjunto de prueba.",
        "Los índices de la división se cargan de entrada/T1/T1_division_indices.parquet; si el archivo no pudiera interpretarse, se recalcula la misma división (70/30 estratificada, random_state=42), que es idéntica.",
    ],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\nResumen T3 (exactitud en el mismo conjunto de prueba de 171 ejemplos):")
print(tabla[["porcentaje", "n_ejemplos", "accuracy_naive_bayes",
             "accuracy_regresion_logistica"]].to_string(index=False))
print("\nFigura guardada en T3_curva_aprendizaje.png y cifras en resultados.json")
