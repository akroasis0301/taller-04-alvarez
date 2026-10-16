import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---------- 1. Cargar resultados y datos producidos por T1 ----------
T1 = Path("entrada/T1")
meta_t1 = json.loads((T1 / "resultados.json").read_text(encoding="utf-8"))
nombres_clases = meta_t1.get("nombres_clases", ["malignant", "benign"])
n_train_esp = int(meta_t1["n_train"])
n_test_esp = int(meta_t1["n_test"])
conteo_tr_esp = {k: int(v) for k, v in meta_t1["conteo_train_por_clase"].items()}
conteo_te_esp = {k: int(v) for k, v in meta_t1["conteo_test_por_clase"].items()}

full = pd.read_parquet(T1 / "breast_cancer_completo.parquet")


def detectar_target(df):
    for cand in ["target", "diagnosis", "diagnostico", "clase", "class", "label", "y"]:
        if cand in df.columns:
            return cand
    return df.columns[-1]


def a_enteros(y):
    y = np.asarray(y)
    if y.dtype.kind in "OUS":
        mapa = {str(n).strip().lower(): i for i, n in enumerate(nombres_clases)}
        return np.array([mapa[str(v).strip().lower()] for v in y], dtype=int)
    return y.astype(int)


target_col = detectar_target(full)
feature_cols = [c for c in full.columns
                if c != target_col and pd.api.types.is_numeric_dtype(full[c])]
X_all = full[feature_cols].to_numpy(dtype=float)
y_all = a_enteros(full[target_col].to_numpy())
assert set(np.unique(y_all).tolist()) == {0, 1}


def conteo_por_clase(y):
    vals, cnts = np.unique(np.asarray(y), return_counts=True)
    return {nombres_clases[int(v)]: int(c) for v, c in zip(vals, cnts)}


def coincide_con_t1(ytr, yte):
    if ytr is None or yte is None:
        return False
    return (conteo_por_clase(ytr) == conteo_tr_esp
            and conteo_por_clase(yte) == conteo_te_esp)


# ---------- 2. Recuperar la división única de T1 ----------
X_train = X_test = y_train = y_test = None
origen_division = None

div_path = T1 / "division_train_test.parquet"
if div_path.exists():
    try:
        div = pd.read_parquet(div_path)
        target_div = detectar_target(div)
        split_col = None
        for c in div.columns:
            if c == target_div:
                continue
            vals = set(div[c].astype(str).str.lower().unique())
            if (len(vals) == 2
                    and any(("train" in v or "entren" in v) for v in vals)
                    and any(("test" in v or "prueb" in v) for v in vals)):
                split_col = c
                break
        if split_col is not None and set(feature_cols).issubset(div.columns):
            v = div[split_col].astype(str).str.lower()
            m_tr = v.str.contains("train|entren").to_numpy()
            m_te = ~m_tr
            X_train = div.loc[m_tr, feature_cols].to_numpy(dtype=float)
            y_train = a_enteros(div.loc[m_tr, target_div].to_numpy())
            X_test = div.loc[m_te, feature_cols].to_numpy(dtype=float)
            y_test = a_enteros(div.loc[m_te, target_div].to_numpy())
            origen_division = "entrada/T1/division_train_test.parquet (atributos y marcador train/test)"
    except Exception:
        X_train = X_test = y_train = y_test = None
        origen_division = None

if X_train is None or not coincide_con_t1(y_train, y_test):
    # Reconstrucción determinista con los parámetros EXACTOS documentados por T1
    # (train_test_split es determinista con random_state=42 y stratify=y):
    # reproduce la misma partición, verificada contra los conteos por clase de T1.
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.30, stratify=y_all, random_state=42
    )
    origen_division = ("reconstrucción determinista de la división de T1 con sus parámetros "
                       "documentados (test_size=0.30, stratify=y, random_state=42)")

assert len(X_train) == n_train_esp == 398
assert len(X_test) == n_test_esp == 171
assert coincide_con_t1(y_train, y_test)

# ---------- 3. Estandarización ajustada SOLO con el entrenamiento ----------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------- 4. Entrenar y evaluar los dos clasificadores ----------
modelos = [
    ("GaussianNB (generativo)", GaussianNB()),
    ("Regresión logística (discriminativo)", LogisticRegression(max_iter=5000, random_state=42)),
]
tabla = []
for nombre, modelo in modelos:
    modelo.fit(X_train_s, y_train)
    y_pred = modelo.predict(X_test_s)
    tabla.append({
        "modelo": nombre,
        "accuracy_prueba": float(accuracy_score(y_test, y_pred)),
        "f1_macro_prueba": float(f1_score(y_test, y_pred, average="macro")),
    })

acc_nb, f1_nb = tabla[0]["accuracy_prueba"], tabla[0]["f1_macro_prueba"]
acc_lr, f1_lr = tabla[1]["accuracy_prueba"], tabla[1]["f1_macro_prueba"]

# ---------- 5. Tabla (JSON y Markdown) y figura ----------
filas_md = ["| modelo | accuracy (prueba) | F1 macro (prueba) |", "|---|---|---|"]
for r in tabla:
    filas_md.append(f"| {r['modelo']} | {r['accuracy_prueba']:.6f} | {r['f1_macro_prueba']:.6f} |")

fig, ax = plt.subplots(figsize=(6.5, 4))
x = np.arange(2)
w = 0.35
accs = [acc_nb, acc_lr]
f1s = [f1_nb, f1_lr]
ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4C72B0")
ax.bar(x + w / 2, f1s, w, label="F1 macro", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(["GaussianNB", "Regresión logística"])
ax.set_ylim(0.0, 1.05)
ax.set_ylabel("Puntaje en el conjunto de prueba")
ax.set_title("T2 · Desempeño en prueba (división 70/30 estratificada de T1)")
for i, (a, f) in enumerate(zip(accs, f1s)):
    ax.text(i - w / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + w / 2, f + 0.015, f"{f:.3f}", ha="center", fontsize=9)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("t2_metricas_prueba.png", dpi=120)
plt.close(fig)

resultados = {
    "subtarea": "T2 — GaussianNB y regresión logística sobre la división 70/30 de T1",
    "dataset": meta_t1.get("dataset", "Breast Cancer Wisconsin (Diagnostic)"),
    "origen_division": origen_division,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "escalador": "StandardScaler ajustado únicamente con el conjunto de entrenamiento",
    "tabla_prueba": tabla,
    "accuracy_gaussian_nb": acc_nb,
    "f1_macro_gaussian_nb": f1_nb,
    "accuracy_regresion_logistica": acc_lr,
    "f1_macro_regresion_logistica": f1_lr,
    "tabla_prueba_markdown": "\n".join(filas_md),
    "archivos_generados": ["t2_metricas_prueba.png", "t2_tabla_metricas.csv"],
    "supuestos": [
        "La división 70/30 estratificada (random_state=42) se toma de entrada/T1; si el parquet de la división no contiene los atributos, se reconstruye de forma determinista con los parámetros exactos documentados por T1, lo que reproduce la misma partición (verificada contra los conteos por clase reportados por T1).",
        "Los atributos se estandarizan para AMBOS modelos (GaussianNB y regresión logística), según la relación del material del curso; el StandardScaler se ajusta solo con el entrenamiento y se aplica al conjunto de prueba sin reajustarlo.",
        "La regresión logística usa los hiperparámetros por defecto de scikit-learn (solver lbfgs, C=1.0) con max_iter=5000 para garantizar convergencia; nada se ajusta con el conjunto de prueba.",
    ],
}

texto = json.dumps(resultados, indent=2, ensure_ascii=False)
Path("entrada/T2").mkdir(parents=True, exist_ok=True)
Path("entrada/T2/resultados.json").write_text(texto, encoding="utf-8")
Path("resultados.json").write_text(texto, encoding="utf-8")
pd.DataFrame(tabla).to_csv("t2_tabla_metricas.csv", index=False, encoding="utf-8")

# ---------- 6. Resumen ----------
print("T2 — Dos clasificadores sobre la división 70/30 estratificada de T1")
print(f"División usada: {origen_division}")
print(f"n_train={len(X_train)}  n_test={len(X_test)}")
print(pd.DataFrame(tabla).to_string(index=False))
print("Archivos generados: resultados.json, entrada/T2/resultados.json, "
      "t2_tabla_metricas.csv, t2_metricas_prueba.png")
