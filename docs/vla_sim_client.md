# 仿真 VLA 交互 Client 启动指南

本文描述如何在仿真侧（pct_scene）启动 Isaac 场景并运行远程 StarVLA 交互 client。
所有参数集中在 YAML，一条命令启动。

仿真侧跑在远程仿真主机上（pct_scene 仓库所在机器）。启动前先登录仿真主机，
确认仓库已同步（`git pull` 或 rsync）以及本机 `127.0.0.1:10093` 隧道已建立。

## 1. 配置（YAML）

`configs/vla_eval/sim_liangzhu.yaml`：

```yaml
server:
  endpoint: ws://127.0.0.1:10093
  connect_timeout_s: 10.0
  response_timeout_s: 120.0

task:
  instruction: "Pick up the coke can on box1 and place it on box2."
  episode_id: "sim_vla_liangzhu_0001"
  scene_profile: liangzhu
  scene_usd: null
  task_json: tasks/nav_pick_place_cola_liangzhu_pct.json

sim:
  seed: 42
  robot_start_override: {x: -1.4849, y: 1.5886, yaw: 0.0}
  object_pose_override: null
  headless: false
  num_episodes: 1
  jpeg_quality: 90

interactive:
  nav_waypoint_tolerance_m: 0.12
  nav_yaw_tolerance_rad: 0.14
  nav_max_steps_per_waypoint: 500
  nav_max_replans: 64
  arm_confirm_prompt: "是否执行 {route}？输入 1=执行 0=跳过: "
  arm_confirm_timeout_s: null
  show_raw_text: true
  show_timing: true
  jsonl_out: results/vla_eval/sim_vla_interactive.jsonl
```

可调项说明：

- `task.scene_profile` / `task.scene_usd`：场景 profile 或直接指定 USD 文件；
- `sim.robot_start_override`：覆盖机器狗起始位姿（x/y/yaw）；
- `sim.object_pose_override`：覆盖物体初始位姿（x/y/z/roll/pitch/yaw）；
- `sim.headless`：`false` 显示 GUI，`true` 无头；
- `interactive.*`：NAV 到达容差、人工门控提示、JSONL 输出路径。

## 2. 启动前提

1. 推理服务已启动（见 `starVLA_sc/docs/vla_remote_inference_server.md`）；
   - 注意服务端代码需与 checkpoint 架构匹配：2025-08 后新训练产物使用
     **按 step 索引归一化**（`dataset_statistics.json` 为 `[horizon,10]`），
     双 expert（NAV/ARM 两个 head）用 `robodog_two_heads` 分支；
     client 协议不变，仍按 route 取 `nav_waypoints` / `arm_targets_base`。
2. 本机 `127.0.0.1:10093` 可访问（SSH 本地转发已建立）；
3. 机器上有可运行的 Isaac Sim 环境（需设置 `ISAAC_PYTHON` 指向含 Isaac 的
   Python，缺省用当前 `python3`）。

## 3. 启动

```bash
cd /path/to/pct_scene          # 仿真主机上的仓库（例如 /home/natural/Desktop/mtr/pct_scene）
bash scripts/evaluation/run_vla_sim_all.sh --config configs/vla_eval/sim_liangzhu.yaml
```

临时覆盖参数（例如改 instruction）：

```bash
bash scripts/evaluation/run_vla_sim_all.sh \
  --config configs/vla_eval/sim_liangzhu.yaml \
  --override task.instruction="Move the cola to the other box."
```

指定 Isaac Python：

```bash
ISAAC_PYTHON=/path/to/isaac/python \
  bash scripts/evaluation/run_vla_sim_all.sh --config configs/vla_eval/sim_liangzhu.yaml
```

## 4. 交互行为

- NAV：只执行模型 chunk 的**第一个 waypoint**，到位后请求下一次推理；
- GRASP/PLACE：终端实时打印 route/subtask/TCP 目标，**阻塞等待数字输入**
  （`1` 执行、`0` 跳过），阻塞期间不发起新推理；
- DONE/RECOVER：停止。

每次决策写入 `interactive.jsonl_out` 指定的 JSONL。

## 5. 停止

在终端 `Ctrl+C` 结束 client；Isaac 场景随进程退出（或按 GUI 关闭）。
