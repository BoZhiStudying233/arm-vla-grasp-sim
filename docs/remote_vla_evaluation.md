# 远程 StarVLA 仿真评测

## 模块边界

- `starVLA_sc/deployment/go2_remote`：加载完整 QwenPI checkpoint，输出 `route/subtask/nav_waypoints`，服务只监听服务器回环地址。
- `pct_scene/source/evaluation`：编码 front/wrist 图像与机体系速度，校验协议和 waypoint，将机体系短轨迹转到世界系。
- 现有 `DwaNavExecutor`：把世界系 waypoint 转为底盘速度，再由 Isaac Lab locomotion policy 执行。
- 现有 full-physics state machine：`GRASP/PLACE` 仅作为语义门控，机械臂仍使用 cuRobo 在线规划与分段执行。

协议版本为 `starvla-go2-eval/v1`。每次请求携带 `request_id/episode_id/frame_index/phase`；每次响应记录 route、subtask、延迟、checkpoint 和 waypoint。网络失败、协议不匹配、越界 waypoint 或重规划超限都会发送零速度并终止该 episode。

## 启动顺序

以下示例使用 `zju-server`。每次启动服务和 smoke test 前仍需确认实际服务器与 GPU 归属。

1. 在推理服务器启动模型服务：

```bash
cd /hdd4/MaTianran/pct_workspace/starVLA_sc
CHECKPOINT=/hdd4/MaTianran/pct_workspace/starVLA_sc/results/Checkpoints/go2_n200_curriculum_a40_b12_0801_pick_place/final_model/pytorch_model.pt \
GPU=0 PORT=10093 \
scripts/evaluation/start_go2_vla_server_tmux.sh
```

2. 在评测机建立 SSH 本地转发：

```bash
cd /home/natural/Desktop/mtr/pct_scene
scripts/evaluation/manage_remote_vla_tunnel.sh start zju-server
scripts/evaluation/manage_remote_vla_tunnel.sh check zju-server
```

3. 在评测机启动单 episode：

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_remote_vla_eval" \
  --seed 3000 \
  --remote-vla-eval \
  --vla-endpoint ws://127.0.0.1:10093 \
  --headless
```

批量评测将入口换成 `run_full_physics_batch.py`，增加 `--num-episodes N --continue-on-failure`，其余 VLA 参数相同。默认不采集 overview，也不制作视频。

4. 结束后关闭隧道：

```bash
scripts/evaluation/manage_remote_vla_tunnel.sh stop zju-server
```

## 本地协议验证

远端模型未就绪时，可在 `starVLA_sc` 启动 `--mock-route nav` 服务，再运行 `pct_scene/tests/evaluation`。Mock 只验证网络与合同，不代表任务可完成，因为固定 NAV 不会切换到 GRASP/PLACE/DONE。
