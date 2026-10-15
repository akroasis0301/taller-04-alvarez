"""Taller 4, Parte 1 — la observabilidad del solver con Langfuse, en un solo lugar.

El solver ya escribía su bitácora (traza.jsonl, C6). Langfuse añade lo que una bitácora plana
no tiene: un ÁRBOL de spans con su duración, que se consulta sin abrir el código y se exporta a
archivo (taller-04-llmops/Lab-04-LLMOps/exportar_trazas.py):

    corrida (raíz, Solver.solve)                         → span «corrida» del export
    ├── guardrail   entrada / salida / guarda_ast        → «guardrail»
    ├── agent       lector, indexador, …, redactor       → uno por rol (opción B)
    │   ├── generation  <agente>  (modelo, tokens)        → «llm»
    │   ├── tool        leer_pdf, sandbox                 → «herramienta»
    │   ├── retriever   seccion_literal, busqueda_local
    │   └── embedding   embeddings
    └── chain       orquestador (siguiente, cerrar), procedencia

Las tres correcciones de la Parte 0.c viven aquí y no dependen de que alguien las recuerde:

  1. Redactar antes de escribir: el cliente se crea con Langfuse(mask=mascara), y la máscara
     es la misma de la traza local (secretos + datos personales), recorriendo dicts, listas y
     TUPLAS (así llegan los argumentos).
  2. El cronómetro incluye los guardrails: la raíz se abre antes que cualquier otra cosa y
     los guardrails de entrada y salida son hijos suyos.
  3. El span existe aunque el código falle: cada observation es un `with`, que se cierra al
     salir por cualquier camino; si sale por una excepción queda en nivel ERROR con su mensaje.

Sin claves de Langfuse (o con SOLVER_LANGFUSE=0) todo esto no hace nada y no importa langfuse:
las pruebas y una corrida sin servidor siguen funcionando igual. Ningún fallo de Langfuse
detiene el solver: lo peor que pasa es que falte un span.

Variables (.env):
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL   el proyecto
  SOLVER_LANGFUSE=0          apaga la instrumentación
  SOLVER_SESION              session_id de las corridas (p. ej. t4-base, t4-mejora)
  SOLVER_PROMPT_VERSION      la versión de los prompts que queda en cada corrida (v1, v2…)
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import sys
import threading
from typing import Any, Callable, Iterator

from solver.traza import _limpiar

ETIQUETA = "taller-04"
_candado = threading.Lock()
_cliente: Any = None
_opciones: dict = {}
_avisado = False


# ============================================================================ configuración
def activo() -> bool:
    """¿Hay a dónde mandar los spans? Se lee en cada llamada: una prueba puede cambiarlo."""
    if os.environ.get("SOLVER_LANGFUSE", "1") == "0":
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")) \
        or "span_exporter" in _opciones


def sesion() -> str:
    return os.environ.get("SOLVER_SESION", "t4-dev")


def prompt_version() -> str:
    return os.environ.get("SOLVER_PROMPT_VERSION", "v1")


def mascara(*, data: Any, **_kw) -> Any:
    """La máscara de Langfuse: la misma limpieza que la traza local, recorriendo tuplas."""
    return _limpiar(data)


def configurar(**opciones) -> None:
    """Para las pruebas: rehace el cliente con otras opciones (p. ej. span_exporter en memoria)."""
    global _cliente, _opciones
    with _candado:
        _cliente, _opciones = None, dict(opciones)


def cliente():
    """El cliente de Langfuse con la máscara, o None si la observabilidad está apagada."""
    global _cliente, _avisado
    if not activo():
        return None
    if _cliente is None:
        with _candado:
            if _cliente is None:
                try:
                    from langfuse import Langfuse
                    _cliente = Langfuse(mask=mascara, **_opciones)
                except Exception as err:                      # nunca detiene el solver
                    if not _avisado:
                        print(f"[observabilidad] Langfuse no disponible: {type(err).__name__}: {err}",
                              file=sys.stderr)
                        _avisado = True
                    return None
    return _cliente


# ============================================================================ observations
class _Obs:
    """Envoltorio de una observation: update() nunca lanza (un span no vale una corrida)."""

    def __init__(self, real: Any = None):
        self._real = real

    def update(self, **campos) -> None:
        if self._real is None:
            return
        if campos.get("status_message"):
            # La máscara de Langfuse cubre input, output y metadata, NO el mensaje de estado
            # (comprobado con langfuse 4.17.0): un error con un teléfono llegaba tal cual.
            campos["status_message"] = _limpiar(str(campos["status_message"]))
        try:
            self._real.update(**campos)
        except Exception:                                     # pragma: no cover
            pass

    def error(self, err: BaseException | str, nivel: str = "ERROR") -> None:
        texto = f"{type(err).__name__}: {err}" if isinstance(err, BaseException) else str(err)
        self.update(level=nivel, status_message=texto[:1000])


@contextlib.contextmanager
def observar(nombre: str, tipo: str = "span", **campos) -> Iterator[_Obs]:
    """Una observation hija de la actual: `with observar("sandbox", "tool") as o: … o.update(…)`.

    tipo: span, agent, tool, chain, retriever, evaluator, guardrail, generation, embedding.
    Si el bloque lanza, el span queda en ERROR con el mensaje y la excepción sigue su camino."""
    c = cliente()
    if c is None:
        yield _Obs()
        return
    try:
        cm = c.start_as_current_observation(name=nombre, as_type=tipo, **campos)
        real = cm.__enter__()
    except Exception:                                         # pragma: no cover
        yield _Obs()
        return
    o = _Obs(real)
    info = (None, None, None)
    try:
        yield o
    except BaseException as err:
        o.error(err)
        info = sys.exc_info()
        raise
    finally:
        try:
            cm.__exit__(*info)
        except Exception:                                     # pragma: no cover
            pass


@contextlib.contextmanager
def corrida(nombre: str, *, entrada: Any = None, metadata: dict | None = None,
            etiquetas: list[str] | None = None) -> Iterator[_Obs]:
    """La raíz de una corrida, con su sesión, etiquetas y prompt_version para todos sus hijos."""
    c = cliente()
    if c is None:
        yield _Obs()
        return
    meta = {"prompt_version": prompt_version(), **{k: str(v) for k, v in (metadata or {}).items()}}
    with observar(nombre, "agent", input=entrada, metadata=meta) as raiz:
        try:
            from langfuse import propagate_attributes
            atributos = propagate_attributes(session_id=sesion(), tags=[ETIQUETA, *(etiquetas or [])],
                                             metadata=meta, trace_name=nombre)
        except Exception:                                     # pragma: no cover
            atributos = contextlib.nullcontext()
        with atributos:
            yield raiz


def traza_id() -> str | None:
    """El id de la traza actual (dentro de la raíz), para ir de una fila del CSV a sus spans."""
    c = cliente()
    if c is None:
        return None
    try:
        return c.get_current_trace_id()
    except Exception:                                         # pragma: no cover
        return None


def flush() -> None:
    """Envía lo pendiente. Se llama al terminar cada corrida."""
    c = cliente()
    if c is not None:
        try:
            c.flush()
        except Exception:                                     # pragma: no cover
            pass


def en_contexto(fn: Callable) -> Callable:
    """Para los hilos de GraphRAG: el contexto (y con él el span padre) no pasa solo a otro hilo.

    Se captura aquí, en el hilo que crea el pool, y cada tarea corre en su propia copia (un mismo
    Context no puede estar activo en dos hilos a la vez)."""
    padre = contextvars.copy_context()

    def en_el_hilo(*args, **kwargs):
        return padre.copy().run(fn, *args, **kwargs)
    return en_el_hilo
