#!/usr/bin/env bash
# Taller 04 — las corridas medidas, en orden. Desde la RAÍZ del repo, con la VPN conectada y el
# Langfuse local arriba (docker compose up -d en la carpeta de clase).
#
#   bash scripts/corridas_t4.sh base            # el solver sobre el golden (≈50 min) + prohibidos + spans
#   bash scripts/corridas_t4.sh juez base       # el juez (gemma3) sobre esa corrida (≈20-30 min)
#   bash scripts/corridas_t4.sh mejora          # la versión mejorada de la 2.c (mismo golden, juez y modelo)
#   bash scripts/corridas_t4.sh juez mejora
#   TAREAS="H P" bash scripts/corridas_t4.sh prueba   # ensayo con algunas tareas, en su propia sesión
#                                                     (no ensucia la sesión t4-base que se entrega)
#
# Cada versión tiene su sesión de Langfuse (t4-<version>), su carpeta (corridas/t4/<version>/) y sus
# CSV (resultados/t4/*_<version>.csv): nada se mezcla ni se sobrescribe.
set -euo pipefail
cd "$(dirname "$0")/.."
GOLDEN=golden/golden_t4.json
TAREAS="${TAREAS:-A B C S1 H P}"
JUEZ="${JUEZ_MODEL:-gemma3:27b}"
export PYTHONUNBUFFERED=1
mkdir -p resultados/t4 corridas/t4 taller04/parte1
DESPIERTO=""; command -v caffeinate >/dev/null && DESPIERTO="caffeinate -i"

corrida() {  # version prompt_version
  export SOLVER_SESION="t4-$1" SOLVER_PROMPT_VERSION="$2"
  echo "== corrida $1 · sesión $SOLVER_SESION · prompts $SOLVER_PROMPT_VERSION · tareas $TAREAS"
  $DESPIERTO uv run python solver-v2/evaluar_solver.py --golden "$GOLDEN" \
    --solver solver.orquestador:Solver --ruta . --corridas "corridas/t4/$1" \
    --solo $TAREAS --salida "resultados/t4/resultados_solver_$1.csv" | tee "resultados/t4/salida_$1.txt"
  uv run python scripts/verificar_prohibidos.py --golden "$GOLDEN" --corridas "corridas/t4/$1" \
    --salida "resultados/t4/prohibidos_$1.csv" || echo "⚠ hay datos prohibidos: revisa resultados/t4/prohibidos_$1.csv"
  sleep 15                                   # lo recién enviado a Langfuse tarda en ser consultable
  (cd taller-04-llmops/Lab-04-LLMOps && \
   uv run python exportar_trazas.py --sesion "t4-$1" -o "../../taller04/parte1/spans_$1.jsonl" && \
   uv run python resumir_trazas.py "../../taller04/parte1/spans_$1.jsonl" --precio propietario_economico \
     > "../../taller04/parte1/resumen_$1.txt")
  echo "✓ corrida $1: resultados/t4/ y taller04/parte1/spans_$1.jsonl"
}

case "${1:-}" in
  base)   corrida base v1 ;;
  prueba) corrida prueba v1 ;;
  mejora) corrida mejora "${SOLVER_PROMPT_VERSION_MEJORA:-v2}" ;;
  juez)
    v="${2:?uso: corridas_t4.sh juez <base|mejora>}"
    uv run python taller-04-llmops/Lab-04-LLMOps/juez_solver.py --golden "$GOLDEN" --corridas "corridas/t4/$v" \
      --resultados "resultados/t4/resultados_solver_$v.csv" --juez "$JUEZ" \
      --salida "resultados/t4/juez_solver_$v.csv" | tee "resultados/t4/salida_juez_$v.txt" ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
