#!/usr/bin/env python3
"""仿真 VLA 交互评测入口：一条 YAML 命令启动场景、重置位置并运行交互 client。

用法：
  python scripts/evaluation/run_vla_sim_interactive.py \
    --config configs/vla_eval/sim_liangzhu.yaml

该脚本复用 run_full_physics_pipeline.py 的 Isaac 启动路径，并在
remote_vla_eval 链路上注入交互行为：
  - NAV 只执行第一个 waypoint，到位后请求下一次推理；
  - GRASP/PLACE 收到决策后终端阻塞，等待操作者输入数字（1=执行 0=跳过）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    raw = Path(path).expanduser().resolve()
    if not raw.is_file():
        raise FileNotFoundError(f"config yaml does not exist: {raw}")
    with raw.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"config yaml must be a mapping: {raw}")
    return data


def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> None:
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"--override 需要 key=value 格式，got {raw!r}")
        key, value = raw.split("=", 1)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)


def _coerce(value: str) -> Any:
    lowered = value.strip()
    if lowered in {"null", "None", "~"}:
        return None
    if lowered in {"true", "True"}:
        return True
    if lowered in {"false", "False"}:
        return False
    try:
        return int(lowered)
    except ValueError:
        pass
    try:
        return float(lowered)
    except ValueError:
        pass
    return value


def _project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="启动仿真场景、重置物体/机器狗并运行远程 VLA 交互 client。"
    )
    parser.add_argument("--config", required=True, help="YAML 评测配置路径")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="点分路径覆盖，如 --override task.instruction='x'（可多次）",
    )
    return parser.parse_args()


def _arm_gate_factory(cfg: dict[str, Any]):
    """构造人工门控回调：终端阻塞输入数字，1=执行，0=跳过。"""

    prompt_template = str(
        cfg.get("arm_confirm_prompt", "是否执行 {route}？输入 1=执行 0=跳过: ")
    )

    def gate(decision) -> bool:
        route = str(decision.route)
        prompt = prompt_template.format(route=route, subtask=decision.subtask)
        while True:
            try:
                raw = input(prompt).strip()
                choice = int(raw)
            except (EOFError, KeyboardInterrupt):
                print("[vla] 输入中断，按跳过处理")
                return False
            except ValueError:
                print("[vla] 输入无效，请输入 0 或 1")
                continue
            if choice == 1:
                return True
            if choice == 0:
                print(f"[vla] {route} 被操作者跳过")
                return False
            print("[vla] 请输入 0 或 1")

    return gate


def main() -> int:
    args = _parse_args()
    cfg = _load_yaml(args.config)
    _apply_overrides(cfg, args.override)

    server_cfg = cfg.get("server") or {}
    task_cfg = cfg.get("task") or {}
    sim_cfg = cfg.get("sim") or {}
    interactive_cfg = cfg.get("interactive") or {}

    pipeline_script = PROJECT_ROOT / "scripts/pipeline/run_full_physics_pipeline.py"
    cmd = [
        sys.executable,
        str(pipeline_script),
        "--mode",
        "remote_vla_eval",
        "--scene-profile",
        str(task_cfg.get("scene_profile", "liangzhu")),
        "--task-json",
        str(_project_path(str(task_cfg.get("task_json")))),
        "--vla-endpoint",
        str(server_cfg.get("endpoint", "ws://127.0.0.1:10093")),
        "--vla-connect-timeout-s",
        str(float(server_cfg.get("connect_timeout_s", 10.0))),
        "--vla-response-timeout-s",
        str(float(server_cfg.get("response_timeout_s", 120.0))),
        "--vla-jpeg-quality",
        str(int(sim_cfg.get("jpeg_quality", 90))),
        "--vla-max-replans",
        str(int(interactive_cfg.get("nav_max_replans", 64))),
        "--vla-max-chunk-steps",
        str(int(interactive_cfg.get("nav_max_steps_per_waypoint", 500))),
        "--vla-arm-mode",
        "shadow",
        "--seed",
        str(int(sim_cfg.get("seed", 42))),
        "--num-episodes",
        str(int(sim_cfg.get("num_episodes", 1))),
    ]
    if sim_cfg.get("headless", False):
        cmd.append("--headless")
    if task_cfg.get("scene_usd"):
        cmd += ["--scene-usd", str(_project_path(str(task_cfg["scene_usd"])))]
    if interactive_cfg.get("first_waypoint_only", True):
        cmd.append("--vla-first-waypoint-only")

    # 注入人工门控：通过环境变量传给 pipeline 侧 gate 工厂。
    os.environ["VLA_ARM_GATE_YAML"] = args.config
    print("[vla] 启动仿真评测:", " ".join(cmd), flush=True)
    os.execv(sys.executable, cmd)
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
