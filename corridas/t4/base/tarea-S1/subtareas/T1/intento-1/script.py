"""Subtarea T1: cargar data/server_measurements.csv y representar las mediciones.

Implementa:
  - clase ServerMeasurements: matriz NumPy 2D (values) + tupla (feature_names),
    con validaciones ValueError (bidimensionalidad, columnas vs nombres,
    datos numéricos) y __repr__ con forma y nombres.
  - build_measurements(df): selecciona gpu_utilization, cpu_utilization y
    memory_gb en ese orden, las convierte a matriz float y retorna el objeto.

El CSV de entrada solo se LEE (nunca se modifica). Escribe resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data/server_measurements.csv")
RESULTS_PATH = Path("resultados.json")
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
EXPECTED_REPR = (
    "ServerMeasurements(shape=(300, 3), "
    "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
)


class ServerMeasurements:
    """Mediciones de servidores: matriz NumPy 2D + nombres de características."""

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional.
        if arr.ndim != 2:
            raise ValueError(
                f"values debe ser una matriz bidimensional; se recibió un array "
                f"con ndim={arr.ndim}"
            )

        names = tuple(feature_names)

        # Validación 2: el número de columnas debe coincidir con los nombres.
        if arr.shape[1] != len(names):
            raise ValueError(
                f"el número de columnas ({arr.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(names)})"
            )

        # Validación 3: los datos deben ser numéricos.
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(
                f"los datos deben ser numéricos; dtype recibido: {arr.dtype}"
            )

        self.values = arr
        self.feature_names = names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df):
    """Selecciona las tres características, las convierte a matriz float y
    retorna un objeto ServerMeasurements."""
    faltantes = [c for c in FEATURES if c not in df.columns]
    if faltantes:
        raise ValueError(f"faltan columnas en el DataFrame: {faltantes}")
    subset = df.loc[:, list(FEATURES)]  # orden fijo: gpu, cpu, memory
    values = subset.to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def lanza_value_error(func):
    """Retorna True si func() lanza ValueError (y no otra excepción)."""
    try:
        func()
    except ValueError:
        return True
    except Exception:
        return False
    return False


def main():
    # Lectura del CSV con parse_dates (solo lectura; nunca se escribe sobre él).
    crudo_antes = CSV_PATH.read_bytes()
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

    measurements = build_measurements(df)
    print(repr(measurements))

    repr_ok = repr(measurements) == EXPECTED_REPR

    # Casos de prueba propios para las tres validaciones.
    casos = {
        "no_bidimensional_1d": lambda: ServerMeasurements(
            np.array([1.0, 2.0, 3.0]), ("x",)
        ),
        "no_bidimensional_3d": lambda: ServerMeasurements(
            np.zeros((2, 2, 2)), ("a", "b")
        ),
        "columnas_menos_que_nombres": lambda: ServerMeasurements(
            np.zeros((4, 2)), ("a", "b", "c")
        ),
        "columnas_mas_que_nombres": lambda: ServerMeasurements(
            np.zeros((4, 3)), ("a", "b")
        ),
        "no_numericos_texto": lambda: ServerMeasurements(
            np.array([["a", "b", "c"], ["d", "e", "f"]]), ("a", "b", "c")
        ),
        "no_numericos_mixto": lambda: ServerMeasurements(
            np.array([[1.0, "x", 3.0]], dtype=object), ("a", "b", "c")
        ),
    }
    validaciones = {nombre: lanza_value_error(caso) for nombre, caso in casos.items()}
    validaciones["todas_lanzan_valueerror"] = all(validaciones.values())

    crudo_despues = CSV_PATH.read_bytes()

    # Estadísticas descriptivas de las tres características seleccionadas.
    values = measurements.values
    stats = {}
    for i, nombre in enumerate(measurements.feature_names):
        col = values[:, i]
        stats[nombre] = {
            "media": float(col.mean()),
            "desviacion_estandar": float(col.std()),
            "minimo": float(col.min()),
            "maximo": float(col.max()),
        }

    resultados = {
        "subtarea": "T1_representar_las_mediciones",
        "csv_leido": str(CSV_PATH),
        "csv_no_modificado": bool(crudo_antes == crudo_despues),
        "timestamp_parseado_como_fecha": bool(
            pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        ),
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "feature_names": list(measurements.feature_names),
        "dtype_values": str(values.dtype),
        "repr": repr(measurements),
        "repr_coincide_con_esperado": bool(repr_ok),
        "estadisticas_por_feature": stats,
        "validaciones_valueerror": validaciones,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # Resumen breve
    print("Resumen T1")
    print(f"  repr: {measurements!r}")
    print(f"  repr coincide con el esperado: {repr_ok}")
    print(f"  shape: {values.shape}, dtype: {values.dtype}")
    print(f"  timestamp como datetime: {resultados['timestamp_parseado_como_fecha']}")
    print(f"  CSV sin modificar: {resultados['csv_no_modificado']}")
    for nombre, ok in validaciones.items():
        print(f"  validacion {nombre}: {'OK (ValueError)' if ok else 'FALLO'}")
    for nombre, s in stats.items():
        print(
            f"  {nombre}: media={s['media']:.4f}, std={s['desviacion_estandar']:.4f}, "
            f"min={s['minimo']:.4f}, max={s['maximo']:.4f}"
        )
    print(f"  resultados escritos en: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
