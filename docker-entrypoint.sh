#!/usr/bin/env bash
set -euo pipefail

MODE="${APP_MODE:-serve}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
BASE_MODEL="${BASE_MODEL:-Chunjiang-Intelligence/DeepSeek-v4-Fable}"
MODEL_PATH="${MODEL_PATH:-}"
LORA_PATH="${LORA_PATH:-}"
API_KEY_ENV="${API_KEY_ENV:-DEEPSEEK_FABLE_API_KEY}"

case "${MODE}" in
  serve)
    args=( -m deepseek_fable_lab.cli serve --base-model "${BASE_MODEL}" --host "${HOST}" --port "${PORT}" --device "${DEVICE}" --dtype "${DTYPE}" --api-key-env "${API_KEY_ENV}" )
    if [[ -n "${MODEL_PATH}" ]]; then
      args+=( --model-path "${MODEL_PATH}" )
    fi
    if [[ -n "${LORA_PATH}" ]]; then
      args+=( --lora-path "${LORA_PATH}" )
    fi
    exec python "${args[@]}"
    ;;
  audit)
    exec python -m deepseek_fable_lab.cli audit --host "${HOST}" --port "${PORT}" --api-key-env "${API_KEY_ENV}"
    ;;
  telegram)
    exec python -m deepseek_fable_lab.cli telegram --host "${HOST}" --port "${PORT}" --api-key-env "${API_KEY_ENV}"
    ;;
  train)
    exec python -m deepseek_fable_lab.cli train "$@"
    ;;
  merge)
    exec python -m deepseek_fable_lab.cli merge "$@"
    ;;
  *)
    echo "Unknown APP_MODE: ${MODE}" >&2
    exit 1
    ;;
esac
