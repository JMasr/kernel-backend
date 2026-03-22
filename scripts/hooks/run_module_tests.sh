#!/usr/bin/env bash
# PostToolUse hook: ejecuta tests unitarios del módulo modificado.
# No puede bloquear retroactivamente, pero su output alimenta el contexto de Claude Code.
set -uo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty' 2>/dev/null || true)

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# No ejecutar si el archivo modificado ya es un test
if [[ "$FILE_PATH" == *"tests/"* ]]; then
  exit 0
fi

# Mapping módulo → keyword de test
declare -A MODULE_MAP=(
  ["engine/audio/"]="audio"
  ["engine/video/"]="video"
  ["engine/codec/"]="codec or reed_solomon or spread_spectrum or hopping"
  ["core/services/"]="service"
  ["core/domain/"]="domain"
  ["core/ports/"]="storage or port"
  ["infrastructure/db/"]="repository or db"
  ["infrastructure/storage/"]="storage"
  ["infrastructure/email/"]="email"
  ["infrastructure/queue/"]="queue or job"
  ["api/"]="api or endpoint or router"
)

TEST_FILTER=""
for module in "${!MODULE_MAP[@]}"; do
  if [[ "$FILE_PATH" == *"$module"* ]]; then
    TEST_FILTER="${MODULE_MAP[$module]}"
    break
  fi
done

if [ -z "$TEST_FILTER" ]; then
  CMD="uv run pytest tests/unit/ -x --tb=short -q"
else
  CMD="uv run pytest tests/unit/ -k \"$TEST_FILTER\" -x --tb=short -q"
fi

echo "🧪 $CMD"
eval "$CMD"
STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo "❌ Tests fallaron tras modificar: $FILE_PATH" >&2
  echo "Corrige los errores antes de continuar con el siguiente archivo." >&2
fi

exit 0
