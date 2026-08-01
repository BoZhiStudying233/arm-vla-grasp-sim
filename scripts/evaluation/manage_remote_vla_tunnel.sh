#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
SERVER="${2:-}"
LOCAL_PORT="${LOCAL_PORT:-10093}"
REMOTE_PORT="${REMOTE_PORT:-10093}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/pct-vla-${UID}-${LOCAL_PORT}.sock}"

if [[ ! "${ACTION}" =~ ^(start|stop|check)$ ]] || [[ -z "${SERVER}" ]]; then
  echo "用法: $0 <start|stop|check> <ssh-host>" >&2
  echo "示例: LOCAL_PORT=10093 REMOTE_PORT=10093 $0 start zju-server" >&2
  exit 2
fi

case "${ACTION}" in
  start)
    if ssh -S "${CONTROL_SOCKET}" -O check "${SERVER}" >/dev/null 2>&1; then
      echo "SSH tunnel 已运行: ws://127.0.0.1:${LOCAL_PORT}"
      exit 0
    fi
    rm -f "${CONTROL_SOCKET}"
    ssh \
      -M -S "${CONTROL_SOCKET}" -fNT \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
      "${SERVER}"
    echo "SSH tunnel 已启动: ws://127.0.0.1:${LOCAL_PORT} -> ${SERVER}:127.0.0.1:${REMOTE_PORT}"
    ;;
  stop)
    ssh -S "${CONTROL_SOCKET}" -O exit "${SERVER}"
    ;;
  check)
    ssh -S "${CONTROL_SOCKET}" -O check "${SERVER}"
    ;;
esac
