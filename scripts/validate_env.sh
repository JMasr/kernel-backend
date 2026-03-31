#!/usr/bin/env bash
# Validate that credentials in .env meet minimum security requirements.
# Called automatically by: make services-up / make services-up-dev
set -euo pipefail

ERRORS=0

check_length() {
  local name="$1" min="${2:-32}"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "ERROR: $name is not set." >&2
    ERRORS=$((ERRORS + 1))
  elif [[ ${#value} -lt $min ]]; then
    echo "ERROR: $name is too short (${#value} chars — minimum $min)." >&2
    ERRORS=$((ERRORS + 1))
  fi
}

check_not_default() {
  local name="$1" forbidden="$2"
  local value="${!name:-}"
  if [[ "$value" == "$forbidden" ]]; then
    echo "ERROR: $name must not be the default value '$forbidden'." >&2
    ERRORS=$((ERRORS + 1))
  fi
}

# Load .env if present so the script works without pre-exporting variables
if [[ -f .env ]]; then
  set -o allexport
  # shellcheck source=/dev/null
  source .env
  set +o allexport
fi

check_length     REDIS_PASSWORD        32
check_length     MINIO_ROOT_PASSWORD   32
check_not_default MINIO_ROOT_USER      "minioadmin"

if [[ $ERRORS -gt 0 ]]; then
  echo "" >&2
  echo "Fix the above issues in .env before starting services." >&2
  echo "Generate a strong password with: openssl rand -base64 32" >&2
  exit 1
fi

echo "Environment credentials validated."
