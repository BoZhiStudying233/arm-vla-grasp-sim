#!/usr/bin/env bash
set -euo pipefail

# 仿真 VLA 全链路：从 YAML 启动 Isaac 场景 + 交互 client。
#
# 用法（评测机）：
#   bash scripts/evaluation/run_vla_sim_all.sh --config configs/vla_eval/sim_liangzhu.yaml
#
# 前提：推理服务已启动（starVLA_sc scripts/evaluation/start_vla_inference.sh），
# 本机 127.0.0.1:10093 可访问（SSH 本地转发）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  echo "用法: $0 --config <sim.yaml> [--override key=value ...]" >&2
  exit 2
}

CONFIG=""
OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --override)
      OVERRIDES+=("${2:?--override 需要 key=value}")
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done
[[ -n "${CONFIG}" ]] || usage
CONFIG_PATH="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
[[ -f "${CONFIG_PATH}" ]] || {
  echo "config yaml 不存在: ${CONFIG_PATH}" >&2
  exit 2
}

# 可选：指定 Isaac Python 解释器（缺省使用当前 python，若含 isaac 环境可覆盖）
ISAAC_PYTHON="${ISAAC_PYTHON:-}"
if [[ -n "${ISAAC_PYTHON}" ]]; then
  PY_CMD="${ISAAC_PYTHON}"
else
  PY_CMD="$(command -v python3 || echo python)"
fi

CMD=(
  "${PY_CMD}"
  "${REPO_ROOT}/scripts/evaluation/run_vla_sim_interactive.py"
  --config "${CONFIG_PATH}"
)
for ov in "${OVERRIDES[@]:-}"; do
  CMD+=(--override "${ov}")
done

echo "[vla] 启动仿真评测:" "${CMD[@]}"
exec "${CMD[@]}"
