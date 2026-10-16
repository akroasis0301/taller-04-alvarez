# Taller 04 — Evaluar, observar y proteger el solver multiagente

MMIA 6013 IA Generativa y Agentes · USFQ · Miguel Álvarez (individual) · Opción **B** del Taller 3

El sistema es el **solver multiagente con GraphRAG del Taller 03 v2** (ocho agentes en LangGraph: recibe
el enunciado de una tarea en PDF y entrega el reporte, PDF o notebook que pide, sin publicar una cifra
que no salió de una ejecución). Este taller lo deja listo para operar:

- **observable**: cada corrida deja en **Langfuse** un árbol de spans —la corrida, un `agent` por rol,
  cada llamada al modelo (`generation`), cada herramienta y cada guardrail— que se exporta a archivo;
- **evaluado**: un golden set de 58 comprobaciones sobre 6 tareas, dos verificadores determinísticos y un
  **juez de otra familia** calibrado contra ellos y contra etiquetas humanas;
- **protegido**: guardrails de inyección (por intención), datos personales y el sandbox, medidos como
  clasificadores y demostrados en el golden.

La historia de git conserva el Taller 03; su README original está en `docs/README_taller03.md`.

---

## 1. Requisitos

| Qué | Para qué |
|---|---|
| [uv](https://docs.astral.sh/uv/) | entorno e instalación (Python 3.12) |
| VPN GlobalProtect de la USFQ | el agente (vLLM de la H200, puerto 12555), el juez y los embeddings (Ollama, 11434) |
| Docker Desktop | Langfuse autoalojado (servidor 4.53.0, `localhost:3000`) |
| macOS o Linux | el sandbox mata al grupo de procesos con `killpg` |

Modelos, leídos del servidor el 2026-10-15 (`taller04/modelos/`): **agente** `zai-org/GLM-5.3-Flash`
(vLLM) y **juez** `gemma3:27b` (Ollama), de otra familia (Zheng et al., 2023, §3.3). Costo real: cero
(H200). Costo de referencia: fila `propietario_economico` (gpt-4o-mini, 0,15/0,60 USD por 1M,
verificada el 2026-08-26), transcrita del anexo del enunciado en `fuentes/modelos/modelos-2026-1.json`.

## 2. Instalación y configuración

```bash
git clone https://github.com/akroasis0301/taller-04-alvarez.git && cd taller-04-alvarez
uv sync                          # o: pip install -r requirements.txt
cp .env.example .env             # y completar el bloque «Taller 04»
uv run python scripts/generar_enunciados_t4.py   # PDF de las tareas H y P del golden
```

**Langfuse.** Cualquier Langfuse 4.x sirve. El kit trae `taller-04-llmops/langfuse/levantar_langfuse.sh`
(Docker, versión fijada, proyecto y claves sin pasar por la interfaz); en este taller se usó el servidor
de la clase (`docker compose up -d` en el repositorio de Langfuse, imagen `langfuse:4` → 4.53.0) con un
proyecto propio, `taller-04-alvarez`. Sus claves van en `.env` (`LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`); **un solo `.env`** en la raíz sirve al solver y a las
herramientas del kit. Sin claves (o con `SOLVER_LANGFUSE=0`) la instrumentación no hace nada.

```bash
uv run python -c "from dotenv import load_dotenv; load_dotenv(); from langfuse import get_client; print(get_client().auth_check())"
uv run pytest -q                 # ≈160 pruebas, sin red ni H200
```

## 3. Reproducir el taller, paso a paso

Todo desde la raíz, con la VPN y Langfuse arriba. `scripts/corridas_t4.sh` deja cada versión en su
sesión de Langfuse (`t4-<versión>`), su carpeta (`corridas/t4/<versión>/`) y sus CSV (`resultados/t4/`).

| # | Parte | Orden | Tiempo |
|---|---|---|---|
| 1 | 0 | `cd taller-04-llmops/Lab-04-LLMOps && uv run python guion.py ceros\|guardrails\|traza` (sin VPN) | 1 min |
| 2 | 1, 2.b | `bash scripts/corridas_t4.sh base` — solver sobre el golden, prohibidos, export de spans | ≈60 min |
| 3 | 2.b | `bash scripts/corridas_t4.sh reevaluar base` — con el golden corregido, sin volver a correr | seg. |
| 4 | 2.b | `bash scripts/corridas_t4.sh juez base` — juez + calibración contra el verificador y `etiquetas.csv` | ≈50 min |
| 5 | 2.c | `uv run python scripts/registrar_prompts.py` — v1 y v2 del prompt del programador en Langfuse | seg. |
| 6 | 2.c | `bash scripts/corridas_t4.sh mejora && bash scripts/corridas_t4.sh juez mejora` | ≈110 min |
| 7 | 3 | `probar_guardrails.py` con `--entrada solver.guardrails:validar_entrada --salida solver.guardrails:validar_salida` sobre `taller04/parte3/ataques.json` y `fuera_del_banco.json`, y sin argumentos para los del ejemplo | seg. |
| 8 | todas | `uv run python scripts/tablas_t4.py` → `taller04/tablas/tablas.md` y un CSV por tabla | seg. |
| 9 | — | `uv run python scripts/verificar_prohibidos.py --corridas corridas/t4/<v>` antes de cada commit | seg. |

Un ensayo que no ensucia las sesiones que se entregan: `TAREAS="H P" bash scripts/corridas_t4.sh prueba`
(`corridas/t4/prueba/` está en `.gitignore`).

## 4. Qué se entrega y dónde está

| Pide el enunciado | Archivo |
|---|---|
| Parte 0: las tres salidas y su análisis | `taller04/parte0/` |
| Agente instrumentado | `solver/observabilidad.py` (el único punto de contacto con Langfuse), `orquestador.py`, `cliente_llm.py`, `agentes/ejecutor.py`, `graphrag.py` |
| Spans exportados de Langfuse | `taller04/parte1/spans_base.jsonl`, `spans_mejora.jsonl` (y su `resumen_*.txt` de `resumir_trazas.py`) |
| Golden set (2.a) | `golden/golden_t4.json` (A, B, C, S1, H, P; correcciones en los campos `_correccion`) |
| Verificador automático (opción B) | `resultados/t4/resultados_solver_*.csv`, `resumen_solver_*.csv` (`evaluar_solver.py` del kit) |
| Tercer verificador: lo que no puede aparecer | `scripts/verificar_prohibidos.py` → `resultados/t4/prohibidos_*.csv` |
| Juez y calibración (2.b) | `resultados/t4/juez_solver_*.csv`, `salida_juez_*.txt`, `taller04/parte2/etiquetas.csv` |
| Peores casos y mejora (2.c) | `solver/prompts.py` (v1, v2 y changelog), sección 2.c de `taller04/tablas/tablas.md` |
| Guardrails (3) | `solver/guardrails.py`; `taller04/parte3/ataques.json`, `guardrails_nuestros.csv`, `guardrails_ejemplo.csv`, `fuera_del_banco.json`, `guardrails_fuera_del_banco.csv` |
| Ajuste fino y alertas (4) | `taller04/parte4/parte4.md` |
| Carpetas de corridas | `corridas/t4/base/`, `corridas/t4/mejora/` |
| Todas las tablas del informe | `taller04/tablas/` (generadas por `scripts/tablas_t4.py`) |

## 5. Arquitectura y spans

| # | Agente | Hace | Span |
|---|---|---|---|
| 1 | Lector | PDF → secciones, tablas y restricciones | `agent` + `tool leer_pdf` |
| 2 | Indexador | esqueleto del grafo + GraphRAG capa 2 (entidades, notas del curso, bge-m3) | `agent` + `generation` (en hilos) + `embedding` |
| 3 | Planificador | subtareas en un DAG validado por código | `agent` + `generation` |
| 4 | Investigador | sección literal, dependencias y búsqueda local/global | `agent` + `retriever` |
| 5 | Programador | un script por subtarea (prompt versionado: `solver/prompts.py`) | `agent` + `generation` ligada a su versión |
| 6 | Ejecutor | sandbox: guarda AST, entorno vacío, `killpg` | `agent` + `guardrail guarda_ast` + `tool sandbox` |
| 7 | Crítico | comprobaciones de código y LLM | `agent` + `generation` |
| 8 | Redactor | entregable en el formato del enunciado | `agent` + `generation` |

Guardrails de entrada (`entrada_inyeccion`, `entrada_pii`) y de salida (`salida_secretos`, `salida_pii`)
son hijos de la raíz `corrida`. Las tres correcciones de la Parte 0.c: cada span es un `with` (existe
aunque el código falle, en nivel ERROR); la raíz se abre antes que los guardrails; y la máscara
(`Langfuse(mask=…)`, que recorre tuplas, más el `status_message`, que la máscara del SDK no cubre)
redacta secretos y datos personales antes de que nada salga del proceso.

## 6. Credenciales y datos personales

Ninguna clave en el código, las trazas, los scripts generados ni el repositorio (público): `.env` está
fuera de git y los scripts del sandbox corren con el entorno vacío. Los datos personales de la tarea P
son **sintéticos**; `verificar_prohibidos.py` comprueba que no aparezcan en el entregable, la traza local,
la carpeta de la corrida ni Langfuse. Los ensayos anteriores a la regla de nombres no se versionan.
El informe declara el uso de un asistente de IA (redacción de código y de las etiquetas, revisadas por
el autor).
