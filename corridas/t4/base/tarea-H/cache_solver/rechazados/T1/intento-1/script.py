#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subtarea T1 — Carga de Breast Cancer Wisconsin (Diagnostic), reporte descriptivo
y división única 70/30 estratificada con random_state=42.

Salidas (carpeta actual):
  - resultados.json            -> contrato con todas las cifras pedidas
  - t1_division_7030.npz       -> X/y de train y test + índices (para reutilizar
                                  la MISMA división en subtareas posteriores)
  - t1_distribucion_clases.png -> figura de verificación de la estratificación
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
TEST_SIZE = 0.30


def conteos_por_clase(y: np.ndarray, target_names) -> dict:
    clases, counts = np.unique(y, return_counts=True)
    return {str(target_names[int(c)]): int(n) for c, n in zip(clases, counts)}


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Carga del dataset
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=int)
    feature_names = [str(f) for f in data.feature_names]
    target_names = [str(t) for t in data.target_names]

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])
    conteo_total = conteos_por_clase(y, target_names)

    # ------------------------------------------------------------------
    # 2) División única 70/30 estratificada por clase, random_state=42
    #    (se guardan los índices para que toda la tarea use esta misma
    #     división; el conjunto de prueba no se usa para ajustar nada)
    # ------------------------------------------------------------------
    indices = np.arange(n_ejemplos)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    conteo_train = conteos_por_clase(y_train, target_names)
    conteo_test = conteos_por_clase(y_test, target_names)

    frac_train = n_train / n_ejemplos
    frac_test = n_test / n_ejemplos

    # Verificación de consistencia con 70/30 estratificado
    consistente = bool(
        abs(frac_train - TRAIN_SIZE) < 1e-9
        and abs(frac_test - TEST_SIZE) < 1e-9
        and (n_train + n_test) == n_ejemplos
        and all(
            abs((conteo_train[c] / n_train) - (conteo_total[c] / n_ejemplos)) < 0.01
            and abs((conteo_test[c] / n_test) - (conteo_total[c] / n_ejemplos)) < 0.01
            for c in conteo_total
        )
    )
    assert consistente, "La división no es consistente con 70/30 estratificado."

    # ------------------------------------------------------------------
    # 3) Guardar la división para subtareas posteriores (misma partición)
    # ------------------------------------------------------------------
    np.savez_compressed(
        "t1_division_7030.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        idx_train=idx_train,
        idx_test=idx_test,
        feature_names=np.array(feature_names, dtype=object),
        target_names=np.array(target_names, dtype=object),
    )

    # ------------------------------------------------------------------
    # 4) Figura de verificación (backend Agg, PNG dpi=120)
    # ------------------------------------------------------------------
    clases_orden = list(target_names)  # [malignant, benign]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    axes[0].bar(clases_orden, [conteo_total[c] for c in clases_orden],
                color=["#c44e52", "#4c72b0"])
    axes[0].set_title(f"Total (n={n_ejemplos})")
    axes[0].set_ylabel("Número de casos")

    axes[1].bar(clases_orden, [conteo_train[c] for c in clases_orden],
                color=["#c44e52", "#4c72b0"])
    axes[1].set_title(f"Entrenamiento 70% (n={n_train})")

    axes[2].bar(clases_orden, [conteo_test[c] for c in clases_orden],
                color=["#c44e52", "#4c72b0"])
    axes[2].set_title(f"Prueba 30% (n={n_test})")

    for ax in axes:
        ax.set_ylim(0, max(conteo_total.values()) * 1.15)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)

    fig.suptitle("Breast Cancer Wisconsin (Diagnostic) — división 70/30 estratificada "
                 f"(random_state={RANDOM_STATE})")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("t1_distribucion_clases.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) resultados.json (contrato de la subtarea)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1",
        "dataset": "breast_cancer_wisconsin_diagnostic (sklearn.datasets.load_breast_cancer)",
        "n_ejemplos_total": n_ejemplos,
        "n_atributos": n_atributos,
        "nombres_clases": clases_orden,
        "conteo_por_clase_total": conteo_total,
        "proporcion_por_clase_total": {
            c: conteo_total[c] / n_ejemplos for c in clases_orden
        },
        "division": {
            "train_size": TRAIN_SIZE,
            "test_size": TEST_SIZE,
            "estratificada_por_clase": True,
            "random_state": RANDOM_STATE,
            "n_entrenamiento": n_train,
            "n_prueba": n_test,
            "fraccion_entrenamiento": frac_train,
            "fraccion_prueba": frac_test,
            "conteo_por_clase_entrenamiento": conteo_train,
            "conteo_por_clase_prueba": conteo_test,
            "proporcion_por_clase_entrenamiento": {
                c: conteo_train[c] / n_train for c in clases_orden
            },
            "proporcion_por_clase_prueba": {
                c: conteo_test[c] / n_test for c in clases_orden
            },
            "consistente_con_70_30_estratificado": consistente,
        },
        "indices_entrenamiento": [int(i) for i in idx_train],
        "indices_prueba": [int(i) for i in idx_test],
        "archivos_generados": {
            "division_reutilizable": "t1_division_7030.npz",
            "figura": "t1_distribucion_clases.png",
        },
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 6) Resumen breve
    # ------------------------------------------------------------------
    print("=" * 62)
    print("T1 — Breast Cancer Wisconsin (Diagnostic): resumen")
    print("=" * 62)
    print(f"Ejemplos totales : {n_ejemplos}")
    print(f"Atributos        : {n_atributos}")
    print(f"Clases           : {conteo_total}")
    print(f"División 70/30 estratificada (random_state={RANDOM_STATE}):")
    print(f"  Entrenamiento  : {n_train} ({frac_train:.4f}) -> {conteo_train}")
    print(f"  Prueba         : {n_test} ({frac_test:.4f}) -> {conteo_test}")
    print("Consistencia 70/30 estratificada:", "OK" if consistente else "FALLÓ")
    print("Archivos: resultados.json, t1_division_7030.npz, t1_distribucion_clases.png")


if __name__ == "__main__":
    main()
