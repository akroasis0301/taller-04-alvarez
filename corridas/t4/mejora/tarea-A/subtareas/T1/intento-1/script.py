# T1: Cargar Breast Cancer Wisconsin (Diagnostic), reportar cifras básicas
# y crear la única división train/test 70/30 estratificada (random_state=42).

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1) Carga del dataset
# ------------------------------------------------------------------
data = load_breast_cancer()
X = data.data          # matriz (n_ejemplos, n_atributos)
y = data.target        # 0 = malignant, 1 = benign
clases = data.target_names
nombres_atributos = list(data.feature_names)

n_ejemplos, n_atributos = X.shape

# Conteo de casos por clase (índice de target -> nombre de clase)
casos_por_clase = {
    str(clases[i]): int(np.sum(y == i)) for i in range(len(clases))
}

# ------------------------------------------------------------------
# 2) Única división train/test 70/30 estratificada por clase, random_state=42
#    (el conjunto de prueba no se usa para ajustar nada en toda la tarea)
# ------------------------------------------------------------------
indices = np.arange(n_ejemplos)
idx_train, idx_test, y_train, y_test = train_test_split(
    indices, y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

n_train = int(len(idx_train))
n_test = int(len(idx_test))

# Verificaciones internas (garantizadas por la propia construcción)
assert n_train + n_test == n_ejemplos
assert abs(n_train / n_ejemplos - 0.70) < 0.01
assert abs(n_test / n_ejemplos - 0.30) < 0.01
# Estratificación: proporciones de clase casi idénticas en train y test
prop_train = {str(clases[i]): float(np.mean(y_train == i)) for i in range(len(clases))}
prop_test = {str(clases[i]): float(np.mean(y_test == i)) for i in range(len(clases))}

# ------------------------------------------------------------------
# 3) Guardar la división para reutilizarla en las demás subtareas
# ------------------------------------------------------------------
df_split = pd.DataFrame({
    "indice": np.concatenate([idx_train, idx_test]),
    "conjunto": ["train"] * n_train + ["test"] * n_test,
    "target": np.concatenate([y_train, y_test]),
})
df_split.to_parquet("division_train_test.parquet", index=False)

# También guardamos X e y completas con los nombres de atributos, para que
# las subtareas siguientes carguen desde aquí sin recalcular nada.
df_datos = pd.DataFrame(X, columns=nombres_atributos)
df_datos["target"] = y
df_datos.to_parquet("breast_cancer_completo.parquet", index=False)

# ------------------------------------------------------------------
# 4) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic) - sklearn.datasets.load_breast_cancer",
    "n_ejemplos": int(n_ejemplos),
    "n_atributos": int(n_atributos),
    "casos_por_clase": casos_por_clase,
    "nombres_clases": [str(c) for c in clases],
    "n_train": n_train,
    "n_test": n_test,
    "proporcion_train": n_train / n_ejemplos,
    "proporcion_test": n_test / n_ejemplos,
    "conteo_train_por_clase": {
        str(clases[i]): int(np.sum(y_train == i)) for i in range(len(clases))
    },
    "conteo_test_por_clase": {
        str(clases[i]): int(np.sum(y_test == i)) for i in range(len(clases))
    },
    "proporcion_clase_train": prop_train,
    "proporcion_clase_test": prop_test,
    "parametros_division": {
        "test_size": 0.30,
        "estratificado_por": "target",
        "random_state": 42,
    },
    "archivos_generados": [
        "division_train_test.parquet",
        "breast_cancer_completo.parquet",
    ],
    "supuestos": [
        "La división se crea una sola vez con train_test_split(test_size=0.30, stratify=y, random_state=42) y se guarda en division_train_test.parquet para que las subtareas siguientes la reutilicen sin recalcularla.",
        "El conjunto de prueba (30%) queda reservado y no se usa para ajustar nada en ninguna subtarea.",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 5) Resumen breve
# ------------------------------------------------------------------
print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
print(f"Ejemplos totales : {n_ejemplos}")
print(f"Atributos        : {n_atributos}")
for nombre, conteo in casos_por_clase.items():
    print(f"Clase {nombre:<10}: {conteo} casos")
print(f"Train (70%)      : {n_train}  -> {prop_train}")
print(f"Test  (30%)      : {n_test}  -> {prop_test}")
print("División guardada en division_train_test.parquet; resultados en resultados.json")
