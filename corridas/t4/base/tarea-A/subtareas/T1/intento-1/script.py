# T1 — Carga de Breast Cancer Wisconsin (Diagnostic), reporte de composición
# y división única 70/30 estratificada (random_state=42) reutilizable.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def conteo_por_clase(y: np.ndarray, target_names) -> dict:
    clases, conteos = np.unique(y, return_counts=True)
    return {str(target_names[int(c)]): int(n) for c, n in zip(clases, conteos)}


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Carga del conjunto de datos
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = [str(f) for f in data.feature_names]
    target_names = [str(t) for t in data.target_names]

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])
    conteo_total = conteo_por_clase(y, target_names)

    # ------------------------------------------------------------------
    # 2) División ÚNICA 70/30 estratificada por clase, random_state=42
    #    (el conjunto de prueba no se usa para ajustar nada)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    conteo_train = conteo_por_clase(y_train, target_names)
    conteo_test = conteo_por_clase(y_test, target_names)

    # ------------------------------------------------------------------
    # 3) Guardar la división de forma reutilizable para las demás subtareas
    # ------------------------------------------------------------------
    Path("datos").mkdir(parents=True, exist_ok=True)

    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train.insert(0, "target", y_train.astype(int))
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test.insert(0, "target", y_test.astype(int))

    df_train.to_parquet("datos/train.parquet", index=False)
    df_test.to_parquet("datos/test.parquet", index=False)
    df_train.to_csv("datos/train.csv", index=False)
    df_test.to_csv("datos/test.csv", index=False)

    np.savez(
        "datos/split.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=np.array(feature_names, dtype=object),
        target_names=np.array(target_names, dtype=object),
    )

    # ------------------------------------------------------------------
    # 4) Figura: composición por clase (total / train / test)
    # ------------------------------------------------------------------
    etiquetas = target_names
    tot_vals = [conteo_total.get(e, 0) for e in etiquetas]
    tr_vals = [conteo_train.get(e, 0) for e in etiquetas]
    te_vals = [conteo_test.get(e, 0) for e in etiquetas]

    x = np.arange(len(etiquetas))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - ancho, tot_vals, ancho, label=f"Total (n={n_ejemplos})")
    ax.bar(x, tr_vals, ancho, label=f"Entrenamiento 70% (n={len(y_train)})")
    ax.bar(x + ancho, te_vals, ancho, label=f"Prueba 30% (n={len(y_test)})")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Número de casos")
    ax.set_title("Breast Cancer Wisconsin (Diagnostic): casos por clase")
    ax.legend()
    fig.tight_layout()
    plt.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "breast_cancer (sklearn.datasets.load_breast_cancer)",
        "n_ejemplos_totales": n_ejemplos,
        "n_atributos": n_atributos,
        "nombres_atributos": feature_names,
        "clases": target_names,
        "conteo_por_clase_total": conteo_total,
        "division": {
            "test_size": 0.30,
            "train_size": 0.70,
            "estratificada_por_clase": True,
            "random_state": 42,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "conteo_por_clase_train": conteo_train,
            "conteo_por_clase_test": conteo_test,
        },
        "archivos_generados": {
            "train_parquet": "datos/train.parquet",
            "test_parquet": "datos/test.parquet",
            "train_csv": "datos/train.csv",
            "test_csv": "datos/test.csv",
            "split_npz": "datos/split.npz",
            "figura": "t1_distribucion_clases.png",
        },
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6) Resumen
    # ------------------------------------------------------------------
    print("=== T1: Carga y división de Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"Ejemplos totales : {n_ejemplos}")
    print(f"Atributos        : {n_atributos}")
    print(f"Clases           : {target_names}")
    print(f"Casos por clase (total): {conteo_total}")
    print(
        f"División 70/30 estratificada (random_state=42): "
        f"train={len(y_train)} ({conteo_train}), test={len(y_test)} ({conteo_test})"
    )
    print("Archivos reutilizables guardados en datos/ (parquet, csv, npz).")
    print("Contrato escrito en resultados.json; figura en t1_distribucion_clases.png.")


if __name__ == "__main__":
    main()
