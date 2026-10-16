# T1: Carga de Breast Cancer Wisconsin, reporte de cifras y división 70/30 estratificada.
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# 1) Cargar datos
data = load_breast_cancer()
X = data.data
y = data.target
n_ejemplos, n_atributos = X.shape
clases = data.target_names  # ['malignant', 'benign']
conteo_clases = {str(name): int(np.sum(y == i)) for i, name in enumerate(clases)}

# 2) División única 70/30 estratificada por clase, random_state=42
idx = np.arange(n_ejemplos)
idx_train, idx_test = train_test_split(
    idx, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
idx_train = np.sort(idx_train)
idx_test = np.sort(idx_test)

y_train = y[idx_train]
y_test = y[idx_test]

# 3) Verificaciones (el propio script garantiza estas condiciones)
assert len(idx_train) + len(idx_test) == n_ejemplos
assert len(np.intersect1d(idx_train, idx_test)) == 0
assert abs(len(idx_train) / n_ejemplos - 0.70) < 0.01
assert abs(len(idx_test) / n_ejemplos - 0.30) < 0.01
# Estratificación: proporción de cada clase igual (a redondeo entero) en train y test
for c in np.unique(y):
    prop_total = np.mean(y == c)
    prop_train = np.mean(y_train == c)
    prop_test = np.mean(y_test == c)
    assert abs(prop_train - prop_total) < 0.02
    assert abs(prop_test - prop_total) < 0.02

# 4) Guardar la división para reutilizarla en subtareas siguientes (índices + etiquetas)
df_split = pd.DataFrame({
    "indice": np.concatenate([idx_train, idx_test]),
    "conjunto": ["train"] * len(idx_train) + ["test"] * len(idx_test),
    "clase": np.concatenate([y_train, y_test]),
})
df_split.to_parquet("T1_division_indices.parquet", index=False)

# 5) Resultados (contrato)
resultados = {
    "subtarea": "T1",
    "n_ejemplos_total": int(n_ejemplos),
    "n_atributos": int(n_atributos),
    "conteo_por_clase": conteo_clases,
    "n_train": int(len(idx_train)),
    "n_test": int(len(idx_test)),
    "proporcion_train": float(len(idx_train) / n_ejemplos),
    "proporcion_test": float(len(idx_test) / n_ejemplos),
    "conteo_por_clase_train": {
        str(name): int(np.sum(y_train == i)) for i, name in enumerate(clases)
    },
    "conteo_por_clase_test": {
        str(name): int(np.sum(y_test == i)) for i, name in enumerate(clases)
    },
    "random_state": RANDOM_STATE,
    "estratificado": True,
    "archivo_division": "T1_division_indices.parquet",
    "supuestos": [
        "La división se guarda como índices (T1_division_indices.parquet) para que las "
        "subtareas siguientes carguen exactamente los mismos conjuntos sin recalcular.",
        "Clase 0 = malignant, clase 1 = benign (codificación nativa de load_breast_cancer).",
    ],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# 6) Resumen
print("=== T1: Carga y división de Breast Cancer Wisconsin ===")
print(f"Ejemplos totales: {n_ejemplos}")
print(f"Atributos: {n_atributos}")
print(f"Conteo por clase: {conteo_clases}")
print(f"Train: {len(idx_train)} ({len(idx_train)/n_ejemplos:.4f}) | "
      f"Test: {len(idx_test)} ({len(idx_test)/n_ejemplos:.4f})")
print(f"Train por clase: {resultados['conteo_por_clase_train']}")
print(f"Test por clase:  {resultados['conteo_por_clase_test']}")
print("División guardada en T1_division_indices.parquet; cifras en resultados.json")
