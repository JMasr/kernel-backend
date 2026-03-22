#!/usr/bin/env bash
# PreToolUse hook: bloquea escritura en zonas inmutables del backend.
# Recibe JSON por stdin con el input de la herramienta.
# Exit 2 = bloquea y devuelve stderr como feedback a Claude Code.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty' 2>/dev/null || true)

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

IMMUTABLE_PATTERNS=(
  "alembic/versions/"
  "tests/fixtures/audio_signals.py"
  "tests/fixtures/video_signals.py"
  ".env"
)

for pattern in "${IMMUTABLE_PATTERNS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    echo "🚫 ZONA INMUTABLE: $FILE_PATH" >&2
    echo "" >&2
    echo "Este archivo requiere autorización explícita en el sprint .md." >&2
    echo "Si el sprint lo requiere, la sección 'Notas de implementación'" >&2
    echo "debe incluir: AUTORIZADO: modificar $pattern" >&2
    exit 2
  fi
done

exit 0
