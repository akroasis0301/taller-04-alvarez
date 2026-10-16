# T3 — Curva de aprendizaje: GaussianNB (generativo) vs Regresión logística (discriminativo)
# Reutiliza la división 70/30 estratificada de T1, entrena con submuestras estratificadas
# del 5/10/25/50/100 % del entrenamiento (random_state=42) y evalúa siempre sobre el MISMO
# conjunto de prueba. Guarda resultados.json y la figura t3_curva_aprendizaje.png.

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
# 1) Cargar la división de T1 (parquet si es interpretable; si no,
#    reconstrucción determinista con los parámetros documentados por T1,
#    mismo criterio que ya usó y validó T2).
# ------------------------------------------------------------------
data = load_breast_cancer()
X_all, y_all = data.data, data.target

# Conteos documentados por T1 (para validar cualquier fuente)
N_TR, N_TE = 398, 171
TR_CLASES = {0: 148, 1: 250}   # malignant=0, benign=1
TE_CLASES = {0: 64, 1: 107}

def _es_train(v):
    s = str(v).lower()
    return ("train" in s) or ("entren" in s) or (s in ("tr", "0"))

def _es_test(v):
    s = str(v).lower()
    return ("test" in s) or ("prueb" in s) or (s in ("te", "1"))

X_train = X_test = y_train = y_test = None
origen = None

ruta_parquet = Path("entrada/T1/division_train_test.parquet")
if ruta_parquet.exists():
    try:
        df = pd.read_parquet(ruta_parquet)
        cols = list(df.columns)
        low = {c: str(c).lower() for c in cols}
        target_col = next((c for c in cols if low[c] in
                           ("target", "y", "clase", "label", "diagnosis", "class")), None)
        split_col = next((c for c in cols if low[c] in
                          ("conjunto", "split", "set", "division", "particion", "partition")), None)
        if target_col is not None and split_col is not None:
            m_tr = df[split_col].map(_es_train)
            m_te = df[split_col].map(_es_test)
            feats = [c for c in cols if c not in (target_col, split_col)]
            if m_tr.any() and m_te.any() and not (m_tr & m_te).any() and len(feats) >= 1:
                Xtr = df.loc[m_tr, feats].to_numpy(dtype=float)
                Xte = df.loc[m_te, feats].to_numpy(dtype=float)
                ytr = df.loc[m_tr, target_col].to_numpy()
                yte = df.loc[m_te, target_col].to_numpy()
                if (Xtr.shape[0] == N_TR and Xte.shape[0] == N_TE
                        and int((ytr == 0).sum()) == TR_CLASES[0]
                        and int((ytr == 1).sum()) == TR_CLASES[1]
                        and int((yte == 0).sum()) == TE_CLASES[0]
                        and int((yte == 1).sum()) == TE_CLASES[1]):
                    X_train, X_test = Xtr, Xte
                    y_train, y_test = ytr.astype(int), yte.astype(int)
                    origen = (f"entrada/T1/division_train_test.parquet "
                              f"(columnas '{target_col}' y '{split_col}')")
    except Exception:
        pass

if X_train is None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.30, stratify=y_all, random_state=42)
    origen = ("reconstrucción determinista de la división de T1 "
              "(test_size=0.30, stratify=y, random_state=42), mismo criterio que T2")

# Verificación contra lo documentado por T1
assert X_train.shape[0] == N_TR and X_test.shape[0] == N_TE
assert int((y_train == 0).sum()) == TR_CLASES[0] and int((y_train == 1).sum()) == TR_CLASES[1]
assert int((y_test == 0).sum()) == TE_CLASES[0] and int((y_test == 1).sum()) == TE_CLASES[1]

# ------------------------------------------------------------------
# 2) Curva de aprendizaje: submuestras estratificadas del entrenamiento
# ------------------------------------------------------------------
fracciones = [0.05, 0.10, 0.25, 0.50, 1.00]
filas = []

for frac in fracciones:
    if frac >= 1.0:
        Xs, ys = X_train, y_train
    else:
        Xs, _, ys, _ = train_test_split(
            X_train, y_train, train_size=frac, stratify=y_train, random_state=42)

    n_ej = int(Xs.shape[0])

    # El escalador se ajusta SOLO con la submuestra de entrenamiento de este tamaño;
    # el conjunto de prueba nunca participa en ningún ajuste.
    scaler = StandardScaler().fit(Xs)
    Xs_z = scaler.transform(Xs)
    Xte_z = scaler.transform(X_test)

    nb = GaussianNB().fit(Xs_z, ys)
    lr = LogisticRegression(max_iter=5000).fit(Xs_z, ys)

    pred_nb = nb.predict(Xte_z)
    pred_lr = lr.predict(Xte_z)

    filas.append({
        "fraccion": frac,
        "n_ejemplos_train": n_ej,
        "accuracy_gaussian_nb": float(accuracy_score(y_test, pred_nb)),
        "accuracy_regresion_logistica": float(accuracy_score(y_test, pred_lr)),
        "f1_macro_gaussian_nb": float(f1_score(y_test, pred_nb, average="macro")),
        "f1_macro_regresion_logistica": float(f1_score(y_test, pred_lr, average="macro")),
    })

n_vals = [r["n_ejemplos_train"] for r in filas]
acc_nb = [r["accuracy_gaussian_nb"] for r in filas]
acc_lr = [r["accuracy_regresion_logistica"] for r in filas]

# ------------------------------------------------------------------
# 3) Figura: exactitud en prueba contra número de ejemplos de entrenamiento.
#    El eje Y se ajusta al rango REAL de los datos (sin límites fijos a mano)
#    para que ambas curvas y sus anotaciones queden dentro del área visible.
# ------------------------------------------------------------------
todas_acc = acc_nb + acc_lr
y_min = min(todas_acc) - 0.05
y_max = max(max(todas_acc) + 0.05, 1.0)  # deja ver el techo 1.0 si las curvas se acercan

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(n_vals, acc_nb, "o-", color="tab:blue", label="GaussianNB (generativo)")
ax.plot(n_vals, acc_lr, "s-", color="tab:red", label="Regresión logística (discriminativo)")

etiquetas = [f"{int(f*100)}%\n(n={n})" for f, n in zip(fracciones, n_vals)]
ax.set_xticks(n_vals)
ax.set_xticklabels(etiquetas)

# Anotaciones: se colocan hacia el lado con espacio dentro del rango visible
for n, a in zip(n_vals, acc_nb):
    dy = 7 if (a + 0.03) <= y_max else -14
    ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=8, color="tab:blue")
for n, a in zip(n_vals, acc_lr):
    dy = 7 if (a + 0.03) <= y_max else -14
    ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=8, color="tab:red")

ax.set_xlabel("Número de ejemplos de entrenamiento (5%, 10%, 25%, 50%, 100%)")
ax.set_ylabel("Exactitud en el conjunto de prueba")
ax.set_title("Curva de aprendizaje: generativo (GaussianNB) vs discriminativo (reg. logística)\n"
             "Breast Cancer Wisconsin — división 70/30 de T1, evaluación siempre sobre el mismo test")
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
plt.savefig("t3_curva_aprendizaje.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 4) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
salida = {
    "subtarea": "T3 — Curva de aprendizaje (5/10/25/50/100 % del entrenamiento, evaluación en el mismo test)",
    "origen_division": origen,
    "n_train_total": int(N_TR),
    "n_test": int(N_TE),
    "fracciones": fracciones,
    "n_ejemplos_train": n_vals,
    "accuracy_gaussian_nb_por_tamano": acc_nb,
    "accuracy_regresion_logistica_por_tamano": acc_lr,
    "tabla_curva_aprendizaje": filas,
    "figura": "t3_curva_aprendizaje.png",
    "rango_eje_y_figura": [float(y_min), float(y_max)],
    "supuestos": [
        "La división 70/30 estratificada de T1 se carga de entrada/T1/division_train_test.parquet; "
        "si el parquet no puede interpretarse, se reconstruye de forma determinista con los parámetros "
        "documentados por T1 (test_size=0.30, stratify=y, random_state=42) y se valida contra sus conteos "
        "por clase (398/171; 148/250 en train; 64/107 en test), igual que hizo T2.",
        "Cada submuestra se obtiene con train_test_split(train_size=fracción, stratify=y_train, "
        "random_state=42) sobre el entrenamiento completo de T1; para el 100 % se usa el entrenamiento completo.",
        "En cada tamaño, el StandardScaler se ajusta únicamente con la submuestra de entrenamiento "
        "correspondiente y se aplica al conjunto de prueba sin reajustarlo; el conjunto de prueba es "
        "siempre el mismo y no se usa para ajustar nada (ni escaladores ni hiperparámetros).",
        "GaussianNB usa los valores por defecto y la regresión logística los hiperparámetros por defecto "
        "con max_iter=5000, igual que en T2.",
        "El eje Y de la figura se ajusta automáticamente al rango real de las exactitudes "
        "(mínimo - 0.05 a máximo + 0.05, sin exceder 1.0 por arriba) para que ambas curvas y sus "
        "valores anotados queden dentro del área visible del PNG.",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(salida, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 5) Resumen
# ------------------------------------------------------------------
print("T3 — Curva de aprendizaje (evaluación siempre sobre el mismo test de T1)")
print(f"Origen de la división: {origen}")
print(f"{'fracción':>9} {'n_train':>8} {'acc NB':>9} {'acc LR':>9}")
for r in filas:
    print(f"{r['fraccion']*100:>8.0f}% {r['n_ejemplos_train']:>8} "
          f"{r['accuracy_gaussian_nb']:>9.4f} {r['accuracy_regresion_logistica']:>9.4f}")
print(f"Eje Y de la figura: [{y_min:.3f}, {y_max:.3f}] (ajustado al rango real de los datos)")
print("Figura guardada: t3_curva_aprendizaje.png")
print("Resultados escritos en resultados.json")
