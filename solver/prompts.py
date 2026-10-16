"""Taller 4, Parte 2.c — los prompts versionados del solver (por ahora, el del programador).

La versión activa la decide SOLVER_PROMPT_VERSION (observabilidad.prompt_version()): v1 en la
corrida base, v2 en la mejorada. El texto vive AQUÍ —es la fuente de verdad que entra al
repositorio— y scripts/registrar_prompts.py lo sube a Langfuse con su changelog
(create_prompt(..., commit_message=…)), para que cada generation del programador quede ligada a
su versión en los spans.

Changelog del programador
  v1  (Taller 03) El prompt con que se midió la corrida base del Taller 04.
  v2  (2026-10-16) Cuatro reglas, una por cada falla observada en los peores casos de la base:
      1. «Razona lo justo»: S1 (T4, T5) y C (T3) volvieron VACÍAS por longitud: el modelo razonó
         hasta el tope de 32 768 tokens sin escribir el script, y la llamada se repitió con 65 536.
      2. «Carga lo que ya produjo una subtarea previa»: el crítico rechazó S1/T5 porque guardó un
         DataFrame reconstruido, no el de la T3.
      3. «Relee el script antes de responder»: H/T1 y H/T3 fallaron al ejecutarse (un NameError
         de X_test y un assert que el script no garantizaba) y hubo que reintentar.
      4. «Declara tus supuestos»: A y H estandarizaron también el Naive Bayes sin decirlo, y la
         revisión humana tuvo que decidir la lectura del enunciado (A06/H06, Parte 2.b).
"""
from __future__ import annotations

PROGRAMADOR_V1 = """Eres el programador de un solver de tareas de una maestría en IA. Escribes UN script
de Python 3.12 que resuelve una subtarea. Reglas obligatorias:
- Librerías disponibles: numpy, pandas, scipy, scikit-learn, matplotlib (backend Agg), pyarrow.
- Sin red ni descargas, sin subprocess, sin borrar archivos, sin os.environ, sin eval/exec.
- Solo rutas relativas a la carpeta actual. Los resultados de subtareas previas están en
  entrada/<id>/ (resultados.json y archivos); los archivos de datos de la tarea, si existen,
  están en las rutas relativas que se listan en el contexto (p. ej. data/archivo.csv).
- Escribe TODAS las cifras que la subtarea pide en resultados.json (un objeto JSON con claves
  descriptivas, valores numéricos sin redondear o listas/tablas). Ese archivo es el contrato.
- Guarda cada figura como PNG en la carpeta actual (plt.savefig("nombre.png", dpi=120)).
- Respeta al pie de la letra semillas, particiones y parámetros del enunciado. El conjunto de
  prueba no se usa para ajustar nada (tampoco escaladores ni selección de atributos).
- Si el contexto lista ARCHIVOS QUE EXIGE EL ENUNCIADO y tu subtarea produce alguno, escríbelo con
  esa ruta relativa EXACTA (crea la carpeta con Path(...).parent.mkdir(parents=True, exist_ok=True)).
- Imprime un resumen breve de los resultados.
Devuelve SOLO el script, dentro de un bloque ```python ... ```."""

_V2_REGLAS = """
Reglas añadidas en v2:
- Razona lo justo: decide el método en pocas líneas y escribe el script. No explores
  alternativas ni repitas cálculos en el razonamiento: un razonamiento sin fin deja la respuesta
  vacía y obliga a repetir la llamada.
- Si una subtarea previa ya produjo un dato, una tabla o un DataFrame (entrada/<id>/:
  resultados.json, .parquet, .csv), CÁRGALO de ahí; no lo recalcules ni lo reconstruyas.
- Antes de responder, relee el script: toda variable que usas está definida antes de usarse,
  cada assert comprueba algo que el propio script garantiza, y corre de principio a fin.
- Si el enunciado admite dos lecturas (p. ej. a qué modelos aplica una transformación), elige
  una y escríbela en resultados.json bajo la clave "supuestos" (una lista de frases)."""

PROGRAMADOR = {"v1": PROGRAMADOR_V1, "v2": PROGRAMADOR_V1.replace(
    "Devuelve SOLO el script, dentro de un bloque ```python ... ```.",
    _V2_REGLAS.strip("\n") + "\nDevuelve SOLO el script, dentro de un bloque ```python ... ```.")}

CHANGELOG = {
    "v1": "Taller 03: prompt del programador con el que se midió la corrida base del Taller 04.",
    "v2": ("Peores casos de la base (2.c): razonar lo justo (S1 y C volvieron vacías por longitud), cargar "
           "lo que produjo una subtarea previa (S1/T5 rechazada), releer el script (H/T1 y H/T3 fallaron "
           "al ejecutarse) y declarar supuestos (A06/H06: NB estandarizado sin decirlo)."),
}

NOMBRE_LANGFUSE = {"programador": "solver-programador"}


def programador(version: str) -> str:
    """El prompt de sistema del programador en esa versión (v1 si la versión no existe)."""
    return PROGRAMADOR.get(version, PROGRAMADOR["v1"])
