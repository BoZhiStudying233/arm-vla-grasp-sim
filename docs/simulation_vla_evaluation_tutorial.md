# StarVLA 仿真测评教程

## 1. 当前可测范围

当前闭环由两台机器协作：推理服务器运行 `starVLA_sc`，评测机运行 `pct_scene` 和 Isaac Sim。远程协议为 `starvla-go2-eval/v2`，只允许通过 SSH 隧道访问服务器回环端口。

### 为什么入口仍是 full-physics pipeline

`run_full_physics_pipeline.py` 和 `run_full_physics_batch.py` 最初用于数据采集，但它们同时拥有 Isaac Sim 启停、场景随机化、任务状态机、locomotion 和 cuRobo 执行器。测评没有复制这套重型运行时，而是通过专用 mode 复用它：

- 不传 `--remote-vla-eval`：普通 full-physics/数采流程，使用确定性规划器。
- 传 `--remote-vla-eval`：仍由同一入口创建 Isaac runtime，但 pipeline factory 切换为 `source.evaluation.create_remote_vla_evaluation_pipeline`，将远程 VLA session 注入导航规划栈和 route gate。
- `run_full_physics_batch.py` 只负责按 seed 重复启动上述单 episode 测评并汇总失败，不是另一套模型测评逻辑。

测评命令应显式使用 `--no-record-dataset --no-record-video`，只保留 summary、远程决策、延迟、执行状态和失败证据；需要人工复核时再单独开启视频。

当前能力边界如下：

- VLM 输出 `route/subtask`。
- NAV 输出机体系稀疏 `[dx,dy,dyaw]`，pct_scene 转到世界系后交给 DWA 和机器狗 locomotion policy。
- 协议能够携带七维 `arm_targets_base`。
- 默认 full-physics 状态机尚未执行模型机械臂目标；GRASP/PLACE 仍门控已有 cuRobo 确定性抓放。

因此，当前仿真闭环可以评价 route/subtask、导航轨迹和完整任务成功率，但完整任务成功率不能单独证明机械臂 action head 有效。机械臂动作质量应先做离线误差评测，待显式执行 adapter 通过 smoke test 后再做模型控制的物理评测。

随机化任务会在送入远程策略前生成逐 episode 全局 instruction：box1 方位以机器狗初始位姿为参考，box2 方位以抓取完成后的 `pick_base_goal` 为参考，均量化为八方向。该文本格式与 StarVLA 训练 loader 完全一致。

旧 3 维 NAV checkpoint 与新 10 维动作头不兼容。启动 v2 完整模型服务必须使用重新训练后的 10 维 checkpoint；旧 checkpoint 只能用于独立 VLM route/subtask 评测。

## 2. 测评前检查

在推理服务器检查 GPU 和代码，不得占用或终止他人进程：

```bash
nvidia-smi
cd /hdd4/MaTianran/pct_workspace/starVLA_sc
git status --short --branch
```

在评测机检查分支、环境和协议测试：

```bash
cd /home/natural/Desktop/mtr/pct_scene
git status --short --branch
conda activate isaac_locomani
export ISAAC_PYTHON="$(command -v python)"
export PCT_SCENE_OUTPUT="${PCT_SCENE_OUTPUT:-$HOME/pct_scene_outputs}"
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B -m pytest -q tests/evaluation
```

## 3. 启动远程模型服务

在用户确认的推理服务器选择空闲 GPU，并使用独立 tmux session：

```bash
cd /hdd4/MaTianran/pct_workspace/starVLA_sc
CHECKPOINT=/absolute/path/to/10d/final_model/pytorch_model.pt \
GPU=0 PORT=10093 TMUX_SESSION=go2_vla_eval \
scripts/evaluation/start_go2_vla_server_tmux.sh
```

检查 tmux、日志、进程和监听地址。服务必须只监听 `127.0.0.1`：

```bash
tmux list-sessions
tmux capture-pane -pt go2_vla_eval:0 -S -120
ss -lntp | grep 10093
```

## 4. 建立 SSH 隧道

在评测机执行，主机名替换为本次确认的服务器：

```bash
cd /home/natural/Desktop/mtr/pct_scene
LOCAL_PORT=10093 REMOTE_PORT=10093 \
  scripts/evaluation/manage_remote_vla_tunnel.sh start zju-server
LOCAL_PORT=10093 REMOTE_PORT=10093 \
  scripts/evaluation/manage_remote_vla_tunnel.sh check zju-server
```

不要把 endpoint 改为服务器公网地址。pct_scene 会拒绝非回环 WebSocket。

## 5. 单 episode 验证

先运行无 overview、无视频的 headless 单 episode：

```bash
cd /home/natural/Desktop/mtr/pct_scene
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_vla_eval_seed3000" \
  --seed 3000 \
  --remote-vla-eval \
  --vla-endpoint ws://127.0.0.1:10093 \
  --vla-connect-timeout-s 10 \
  --vla-response-timeout-s 120 \
  --vla-max-replans 64 \
  --no-record-dataset \
  --no-record-video \
  --headless
```

需要人工观察时把 `--headless` 改为 `--no-headless --keep-window-open`。只有显式添加 `--overview` 才采集 overview 并制作对应视频。

## 6. 批量评测

单 episode 通过后再扩大 batch：

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_vla_eval_n20" \
  --num-episodes 20 \
  --seed 3000 \
  --remote-vla-eval \
  --vla-endpoint ws://127.0.0.1:10093 \
  --vla-response-timeout-s 120 \
  --vla-max-replans 64 \
  --no-record-dataset \
  --no-record-video \
  --headless \
  --continue-on-failure
```

在线 batch 汇总任务成功率、NAV 到达率、抓取成功率、放置成功率、平均重规划次数、推理延迟、超时率和失败原因分布。状态机在关键阶段会记录 expected/actual route，并在 route 错误时终止 episode。

局部 instruction exact accuracy 不能由无逐帧 GT 的在线仿真 summary 凭空得到，应使用 `starVLA_sc/scripts/evaluate_go2_vlm_subtasks.py` 在 held-out LeRobot episode 上单独评测。模型、协议、seed、checkpoint、scene profile 和代码 commit 必须随结果保存。

## 7. 安全退出与故障判断

先停止新 episode，再关闭评测进程、隧道和属于本项目的模型 tmux。不要终止其他用户的进程。

```bash
LOCAL_PORT=10093 REMOTE_PORT=10093 \
  scripts/evaluation/manage_remote_vla_tunnel.sh stop zju-server
```

- `protocol_version` 不一致：两仓库代码未同步，停止评测后对齐版本。
- `NAV response has no nav_waypoints`：模型 route 与动作输出合同不一致。
- 网络超时：机器人应发送零速度并终止 episode，不得复用旧动作。
- cuRobo 抓放失败：属于当前确定性 manipulation 路径，不能归因于机械臂 action head。
