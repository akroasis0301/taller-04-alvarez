"""Taller 4, Parte 1 — observabilidad: la máscara, los spans y las tres correcciones de la 0.c.

Sin red ni servidor: los spans van a un exportador de OpenTelemetry en memoria
(Langfuse(span_exporter=…)), el mismo camino que siguen hacia Langfuse, menos el último salto.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from solver import observabilidad as obs
from solver.config import RAIZ
from solver.guardrails import cedula_valida, redactar_pii
from solver.traza import Traza

_EXPORTADOR = InMemorySpanExporter()   # uno por proceso: Langfuse guarda su cliente por clave pública


@pytest.fixture
def spans(monkeypatch):
    """Enciende la observabilidad hacia el exportador en memoria y devuelve cómo leer los spans."""
    monkeypatch.setenv("SOLVER_LANGFUSE", "1")
    monkeypatch.setenv("SOLVER_SESION", "pruebas")
    obs.configurar(public_key="pk-lf-pruebas", secret_key="sk-lf-pruebas",
                   base_url="http://127.0.0.1:9", span_exporter=_EXPORTADOR)
    _EXPORTADOR.clear()

    def leer():
        obs.flush()
        return list(_EXPORTADOR.get_finished_spans())
    yield leer
    obs.configurar()


def _attr(s, clave):
    return s.attributes.get(f"langfuse.observation.{clave}")


def _padre(s, todos):
    ids = {x.context.span_id: x for x in todos}
    return ids.get(s.parent.span_id) if s.parent else None


# ============================================================================ datos personales
BANCO = {  # los casos de ataques_plantilla.json que tocan datos personales o se les parecen
    "Soy ana.perez@mail.com: ¿cuántas ventas hubo en la región Costa en 2025?": ["correo"],
    "Mi cédula es 1712345678; dime el importe medio por venta del segmento pyme en 2026.": ["cedula"],
    "Llámame al 099 123 4567 cuando tengas el total de marzo de 2026.": ["telefono"],
    "El cliente con más compras es Cliente 042 (contacto: ana.perez@mail.com, 0991234567).": ["correo", "telefono"],
    "¿Cuánto facturaron las tiendas 101 102 103 en marzo de 2026?": [],
    "Compara las ventas de 2025-01-01 a 2025-06-30 con las de 2026-01-01 a 2026-06-30.": [],
    "¿Qué producto tiene el código 3f2c9a1e-7b44-4c1d-9e0a-2b5f6c7d8e90 en el catálogo?": [],
    "El lote 9b1d3c5e7f9a2b4c6d8e0f1a3b5c7d9e se procesó sin errores; total 845.520,02 USD.": [],
    "En marzo de 2026 se facturaron 845.520,02 USD en 512 ventas.": [],
}


@pytest.mark.parametrize("texto,tipos", list(BANCO.items()))
def test_pii_del_banco(texto, tipos):
    limpio, hallados = redactar_pii(texto)
    assert sorted(hallados) == sorted(tipos)
    if not tipos:
        assert limpio == texto                     # las legítimas pasan INTACTAS (Parte 0.b)


def test_cedula_por_digito_verificador_o_por_nombre():
    assert cedula_valida("1710034065") and not cedula_valida("1712345678")
    assert redactar_pii("registro 1710034065, listo")[1] == ["cedula"]        # válida, sin nombrarla
    assert redactar_pii("número 1712345678 de pedido")[1] == []                # inválida y sin nombre
    assert redactar_pii("C.I. 1712345678")[1] == ["cedula"]                    # nombrada
    assert redactar_pii("ciudad 1712345678")[1] == []                          # «ci» dentro de una palabra
    assert redactar_pii("accuracy 0,9912345678 y semilla 1234567890")[1] == []  # decimales y números


def test_la_traza_local_no_guarda_datos_personales(tmp_path):
    t = Traza(tmp_path / "traza.jsonl")
    t.decision("lector", "leido", "enunciado de ana.perez@mail.com, cédula 1712345678, cel 0991234567",
               extra=("ana.perez@mail.com",))
    texto = (tmp_path / "traza.jsonl").read_text(encoding="utf-8")
    for dato in ["ana.perez@mail.com", "1712345678", "0991234567"]:
        assert dato not in texto
    assert "[CORREO]" in texto and "[CEDULA]" in texto and "[TELEFONO]" in texto


def test_la_mascara_recorre_tuplas():
    datos = {"args": ("ana.perez@mail.com", 3), "kwargs": {"tel": ["099 123 4567"]}}
    limpio = obs.mascara(data=datos)
    assert json.dumps(limpio, ensure_ascii=False).count("[") >= 2
    assert "ana.perez" not in json.dumps(limpio) and "4567" not in json.dumps(limpio)
    assert isinstance(limpio["args"], list)        # serializable a JSON


# ============================================================================ spans
def test_apagada_no_hace_nada(monkeypatch):
    monkeypatch.setenv("SOLVER_LANGFUSE", "0")
    with obs.observar("x", "tool") as o:
        o.update(output=1)
    assert obs.cliente() is None and obs.traza_id() is None


def test_el_span_existe_aunque_falle_y_su_error_va_enmascarado(spans):
    with pytest.raises(RuntimeError):
        with obs.corrida("solver"):
            with obs.observar("sandbox", "tool"):
                raise RuntimeError("se cayó llamando al 0991234567")
    todos = spans()
    assert {s.name for s in todos} == {"solver", "sandbox"}
    for s in todos:
        assert _attr(s, "level") == "ERROR"
        assert "0991234567" not in _attr(s, "status_message") and "[TELEFONO]" in _attr(s, "status_message")


def test_los_hilos_heredan_el_span_padre(spans):
    with obs.corrida("solver"):
        with obs.observar("indexador", "agent"):
            def una(i):
                with obs.observar("indexador", "generation", model="guion"):
                    return i
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(obs.en_contexto(una), range(4)))
    todos = spans()
    gens = [s for s in todos if _attr(s, "type") == "generation"]
    assert len(gens) == 4
    assert all(_padre(g, todos).name == "indexador" and _attr(_padre(g, todos), "type") == "agent" for g in gens)
    assert len({s.context.trace_id for s in todos}) == 1


def test_la_raiz_lleva_sesion_etiqueta_y_prompt_version(spans):
    with obs.corrida("solver", metadata={"variante": "completo"}):
        tid = obs.traza_id()
    (raiz,) = spans()
    assert format(raiz.context.trace_id, "032x") == tid
    assert raiz.attributes.get("session.id") == "pruebas"
    assert "taller-04" in raiz.attributes.get("langfuse.trace.tags")
    assert raiz.attributes.get("langfuse.trace.metadata.prompt_version") == "v1"


# ============================================================================ el solver entero
def test_una_corrida_del_solver_deja_el_arbol_completo(spans, tmp_path, monkeypatch):
    """La corrida «repetición» de forzar_frenos: un script con fuga que la guarda bloquea."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    import forzar_frenos
    from solver.orquestador import Solver

    entrada = tmp_path / "_enunciado"
    entrada.mkdir()
    (entrada / "enunciado.md").write_text(forzar_frenos.ENUNCIADO, encoding="utf-8")
    cfg, _ = forzar_frenos.escenarios(False)["repeticion"]
    r = Solver(cfg).solve(str(entrada), str(tmp_path / "tarea-F"))

    todos = spans()
    (raiz,) = [s for s in todos if s.parent is None]
    assert raiz.name == "solver" and len({s.context.trace_id for s in todos}) == 1
    assert r["traza_id"] == format(raiz.context.trace_id, "032x")
    assert (tmp_path / "tarea-F" / "traza_id.txt").read_text().strip() == r["traza_id"]

    agentes = {s.name for s in todos if _attr(s, "type") == "agent" and _padre(s, todos) is raiz}
    assert {"lector", "planificador", "programador", "ejecutor", "critico", "redactor"} <= agentes
    for g in (s for s in todos if _attr(s, "type") == "generation"):
        assert _padre(g, todos).name == g.name          # cada llamada, bajo el agente que la pidió
        assert json.loads(_attr(g, "usage_details"))["input"] > 0
    # En «repetición» el script tiene FUGA de datos: eso lo rechaza el crítico, no la guarda,
    # que solo mira imports y llamadas peligrosas. La guarda dice ok (el bloqueo se prueba abajo).
    guardas = [s for s in todos if s.name == "guarda_ast"]
    assert guardas and all(_attr(s, "output") == "ok" for s in guardas)
    assert any(s.name == "sandbox" and _attr(s, "type") == "tool" for s in todos)
    assert any(s.name == "leer_pdf" and _attr(s, "type") == "tool" for s in todos)


def test_la_guarda_del_sandbox_es_un_guardrail_con_su_motivo(spans, tmp_path):
    """Parte 3, opción B: cada script rechazado deja un span de guardrail con su motivo, y no
    llega a existir el proceso (no hay span «sandbox»)."""
    from solver.agentes import ejecutor

    r = ejecutor.ejecutar("import socket\nsocket.socket()\n", tmp_path / "a", subtarea="T9")
    assert r["error"].startswith("guarda:")
    r = ejecutor.ejecutar("from sklearn.datasets import fetch_openml\nfetch_openml('mnist_784')\n",
                          tmp_path / "b", subtarea="T9", confirmar=lambda motivo: False)
    assert r["confirmacion"]["aprobado"] is False
    todos = spans()
    guardas = {s.name: s for s in todos if _attr(s, "type") == "guardrail" and _attr(s, "output") != "pendiente_confirmacion"}
    bloqueo = guardas["guarda_ast"]
    assert _attr(bloqueo, "output") == "bloqueada" and _attr(bloqueo, "level") == "WARNING"
    assert "socket" in _attr(bloqueo, "status_message")
    assert _attr(guardas["confirmacion_red"], "output") == "bloqueada"
    assert not any(s.name == "sandbox" for s in todos)
