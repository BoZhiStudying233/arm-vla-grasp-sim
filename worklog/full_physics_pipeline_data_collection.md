# run_full_physics_pipeline.py 数采流程与自动化采集说明

本文说明 `scripts/pipeline/run_full_physics_pipeline.py` 涉及的 full-physics 数据采集流程，以及在 `pct_scene` codebase 下如何设计轨迹、生成任务并批量采集。

## 1. 入口脚本做了什么

`run_full_physics_pipeline.py` 是单进程、单 Isaac World 的 nav-pick-place 入口。它本身不直接实现导航、抓取或数据编码，而是负责把 CLI、场景 profile、任务 JSON 和 recorder 配置解析成 `FullPhysicsConfig`，然后创建具体 pipeline。

主流程如下：

1. 解析 CLI 参数，并通过 `--scene-profile` 加载场景默认配置。
2. 加载 `--task-json` 指向的任务文件，转换成 `EpisodeSpec`。
3. 根据场景 profile 绑定 runtime USD 资产，例如良渚 NuRec/Gaussian visual 和 collision USD。
4. 启动 cuRobo planner server。full-physics 和 pick-smoke 都要求在线规划，禁止使用离线 `--pick-plan-json` / `--place-plan-json`。
5. 启动 Isaac Sim / Isaac Lab runtime，创建 Go2-X5、场景、相机、PCT 导航组件、机械臂执行器、夹爪和 verifier。
6. 对每个 episode 按 `seed + episode_index` 调用 `prepare_episode_spec`，必要时执行任务随机化和 base_goal 随机化。
7. 创建 `FullPhysicsPipeline`，进入唯一的仿真 step loop。
8. episode 结束后写 `summary.json`、`events.jsonl`、`frames.jsonl`、LeRobot 原始样本和可选视频。

`liangzhu` profile 默认已经配置：

- `task_json`: `tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json`
- `global_planner`: `pct`
- `policy_profile`: `pct_multifloor`
- `locomotion_checkpoint`: `checkpoints/go2_x5/pct_multifloor/model_26000.pt`
- `randomize_task`: `true`
- `randomize_base_goal`: `true`
- `navigation_visual_mode`: `collision`
- `video_mode`: `composite`

因此良渚场景的常规批采通常只需要指定输出目录、episode 数和 seed。

## 2. 单 episode 内的数据流

真实采集循环在 `source/pipeline/full_physics_pipeline.py` 的 `FullPhysicsPipeline.run_episode()` 中。

每个 tick 的顺序是：

1. `simulation.read()` 读取当前仿真状态：机器人 root pose/velocity、关节、TCP、物体、相机图像和 runtime metadata。
2. `FullPhysicsStateMachine.tick(observation)` 根据当前状态决定下一步 action 和事件。
3. `simulation.apply(decision.action)` 下发底盘速度、机械臂关节目标、夹爪命令或调试/重置 action。
4. 如果 action 没有声明 `skip_physics_step`，执行 `simulation.step(render=config.render)`。
5. 再次 `simulation.read()` 得到 step 后状态。
6. `OverviewVideoRecorder.add_frame()` 可选保存展示视频帧。
7. `JsonlEpisodeRecorder.record_step()` 保存诊断帧，并把同步后的训练样本交给 LeRobot writer。

这个设计的关键点是：pipeline 拥有唯一 physics step，导航、机械臂和夹爪 executor 只产生命令，不各自推进仿真。这样数据里的 observation、action、post-step observation 能保持严格的控制步对应关系。

## 3. 状态机阶段

full-physics 的典型阶段是：

- `build_stage`: 构建或复用 Isaac stage。
- `reset_episode`: 按任务设置机器人、物体、place target 等初始状态。
- `plan_nav_to_pick`: 用 PCT/A* 规划到 pick handoff base_goal。
- `exec_nav_to_pick`: DWA/locomotion policy 执行导航。
- `verify_pick_reachable`: 到达 pick 点后检查机械臂可达、底盘稳定等条件。
- `plan_pick`: 基于当前仿真状态在线生成 cuRobo pick 分段轨迹。
- `exec_pick`: 执行 pregrasp、approach、close、lift、return-home 等分段动作。
- `verify_pick_success`: 检查抓取是否成立。
- `plan_nav_to_place`: 携物状态下规划到 place handoff base_goal。
- `exec_nav_to_place`: 携物导航，必要时用 PCT 多楼层/楼梯相关逻辑。
- `verify_place_reachable`: 检查 place 可达。
- `plan_place`: 在线生成 place 分段轨迹。
- `exec_place`: 执行 pre-place、place、open、settle、retreat、return-home。
- `verify_place_success`: 检查放置成功和训练质量门禁。
- `export_lerobot`: 导出/验证 LeRobot 数据集。
- `cleanup_episode`: 关闭 recorder、写 summary。

smoke 模式会截断或替换其中一部分阶段。例如 `--navigation-smoke` 只验证导航，`--pick-smoke` 到 pick 成功即停止，`--pct-plan-preview` 只规划并显示 PCT 路线。

## 4. 采集输出

每个 episode 默认写到：

```text
<output-dir>/episode_000000/
```

重要文件包括：

- `task.json`: 本 episode 实际使用的任务配置，含随机化后的 pick/place/base_goal。
- `events.jsonl`: 状态机事件、成功/失败原因和阶段 metadata。
- `frames.jsonl`: 每个 pipeline step 的轻量诊断帧，不把像素直接写入 JSONL。
- `data.csv`: DWA 兼容的人工可读采样表，包含 base、TCP、关节、action、图片路径和 pipeline_state。
- `samples.jsonl`: LeRobot 原始样本，每条含 state、action、object_state、tcp_pose、camera frame metadata、action semantics。
- `images/<camera_key>/*.jpg`: 原始图片，默认相机为 `front`、`wrist`；传 `--overview` 时追加 `overview`。
- `recording_videos/*.mp4`: LeRobot video feature 的 staging 视频。
- `summary.json`: 最终成功、失败、训练质量门禁、LeRobot export、性能报告和关键路径汇总。
- `overview_videos/*.mp4`: 传 `--overview` 时保存展示视频；默认不制作视频。

full-physics 实际使用的是 `source.recording.JsonlEpisodeRecorder` 和 `source.recording.lerobot_dataset.DwaEpisodeWriter`。`source.data.episode_recorder.EpisodeRecorder` 是更早的 phase CSV/image recorder，不是当前 full-physics 主采集路径。

LeRobot 采样频率由 `RecordingSettings.dataset_fps` 控制，默认 5 FPS；仿真控制步是 0.02 s。writer 会按固定 dataset 时间栅格采样，并检查相机帧与状态 step/timestamp 同步。

## 5. 在 pct_scene 下如何设计轨迹

这里的“轨迹”不是手写一整条 root 和关节序列，而是通过任务目标和规划约束间接设计：

- 导航轨迹由 `start`、`pick.base_goal`、`place.base_goal`、场景 nav/PCT 地图和 DWA 参数决定。
- 抓取轨迹由 `pick.object_pose_world`、`pick.object_prim_path`、`grasp_mode`、runtime mesh/bbox target 和 cuRobo 当前状态规划决定。
- 放置轨迹由 `place.place_pose_world`、receptacle/support 配置、当前抓取姿态和 cuRobo 当前状态规划决定。
- 数据集中的 action 是 pipeline 实际下发的底盘速度、机械臂关节目标和夹爪目标，而不是离线计划文件的静态拷贝。

设计一条稳定轨迹时，建议按这个顺序做：

1. 选场景 profile。良渚单层用 `--scene-profile liangzhu`，多楼层用 `multi_floor` 或对应 profile。
2. 复制一个已有任务 JSON，例如 `tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json`。
3. 修改 `instruction`、`start`、`pick.base_goal`、`pick.object_pose_world`、`pick.object_prim_path`。
4. 如果要放置，设置 `place.enabled=true`，并修改 `place.base_goal`、`place.place_pose_world`、目标 receptacle/support 字段。
5. 确保 base_goal 满足机械臂工作空间：目标在机器人前方，目标到 arm base 的 XY 半径大致在配置允许范围内；良渚默认会随机化并校验 handoff goal。
6. 用 `--pct-plan-preview --no-headless --show-planned-trajectories` 看 PCT 路线是否穿墙、绕路或跨层失败。
7. 用 `--navigation-smoke` 或 `--pick-smoke` 分阶段验收。
8. full-physics 采集前检查 `summary.json` 里的 `training_quality_gate_passed`，只有通过门禁的数据才适合训练。

任务 JSON 的核心字段结构：

```json
{
  "task_id": 2001,
  "episode_id": 1,
  "instruction": "Pick up ... and place ...",
  "scene_usd": "source/scene/liangzhu/liangzhu.usda",
  "nav_map": "",
  "scene_profile": "liangzhu_single_floor",
  "start": {"x": -1.48, "y": 5.12, "z": 0.29, "yaw": 0.0},
  "pick": {
    "base_goal": {"x": -0.64, "y": 5.22, "z": 0.29, "yaw": 1.57},
    "object_prim_path": "/World/cola",
    "object_pose_world": {"x": -0.64, "y": 5.60, "z": -0.08, "yaw": 0.0},
    "grasp_mode": "top_down"
  },
  "place": {
    "enabled": true,
    "base_goal": {"x": -0.44, "y": 4.73, "z": 0.29, "yaw": 1.57},
    "place_pose_world": {"x": -0.44, "y": 5.11, "z": -0.097, "yaw": 0.0}
  },
  "recording": {
    "dataset_dir": "episodes"
  }
}
```

注意：full-physics 会优先使用 CLI 的 `--output-dir` 作为 episode 输出根目录；任务里的 `recording.dataset_dir` 更多是旧 schema 兼容字段。

## 6. 单条采集命令

良渚默认配置下，运行一条 full-physics episode：

```bash
cd /home/natural/Desktop/mtr/pct_scene
python -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir outputs/liangzhu_manual_run \
  --num-episodes 1 \
  --seed 5000 \
  --headless \
  --record-dataset \
  --overview
```

如果只想快速验证流程，不保存训练数据：

```bash
python -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir outputs/liangzhu_smoke \
  --pick-smoke \
  --seed 5000 \
  --headless \
  --no-record-dataset \
  --no-record-video
```

可视化 PCT 规划路线：

```bash
python -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir outputs/liangzhu_pct_preview \
  --pct-plan-preview \
  --no-headless \
  --show-planned-trajectories
```

## 7. 自动化采集方法

### 方法 A：单进程多 episode

`run_full_physics_pipeline.py` 支持在 headless full-physics 下复用同一个 Isaac stage：

```bash
python -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir outputs/liangzhu_collect_seed5000 \
  --num-episodes 50 \
  --seed 5000 \
  --reuse-isaac-stage \
  --headless \
  --record-dataset \
  --no-record-video
```

优点是少启停 Isaac，速度更好；限制是只支持 headless full_physics 多 episode。

### 方法 B：batch launcher

`scripts/pipeline/run_full_physics_batch.py` 是更适合批采的入口。它会构造 episode 子进程或复用 Isaac 进程，打印进度心跳，最后汇总每条 episode 的成功、训练质量门禁、随机化坐标和 LeRobot 路径。

```bash
python -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir outputs/liangzhu_batch_seed5000 \
  --num-episodes 100 \
  --seed 5000 \
  --headless \
  --record-dataset \
  --no-record-video \
  --continue-on-failure
```

如果需要展示视频抽检，可以开少量 episode：

```bash
python -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir outputs/liangzhu_batch_video_check \
  --num-episodes 5 \
  --seed 7000 \
  --headless \
  --record-dataset \
  --overview \
  --video-mode composite
```

### 方法 C：任务随机化

良渚 profile 默认启用任务随机化和 base_goal 随机化。每个 episode 使用 `seed + episode_index`，所以同一命令可复现，同一 seed 对应同一布局。

相关开关：

- `--randomize-task` / `--no-randomize-task`: 控制物体、放置点或场景布局随机化。
- `--randomize-base-goal` / `--no-randomize-base-goal`: 控制 pick/place handoff base_goal 随机化。
- `--show-randomization-debug`: 在 GUI 中显示采样区域和采样点。

良渚已有两类随机化实现：

- `robot_forward_sector_v1`: 在机器人前方扇形内采样物体和地垫目标。
- `liangzhu_box_pair_xy_v1`: 对 box1/box2 支撑台、物体、机器人和 place 区域做联合随机化。

随机化结果会写入 episode 的 `task.json` 和 `summary.json`，batch 表格也会显示 pick/place XY 和 base_goal 相对目标偏移。

### 方法 D：先生成任务 JSON，再逐条采集

仓库里有 `scripts/tasks/generate_random_pick_task.py` 和 `source.data.random_task`，可以生成静态任务 JSON。适合需要人工筛选、保存任务清单或固定 benchmark 的场景。

基本思路：

1. 用脚本生成 `tasks/generated/*.json`。
2. 对每个 JSON 运行 `run_full_physics_pipeline.py --task-json ...`。
3. 把每条输出目录与 task JSON 绑定保存。

这种方式的优点是任务文件本身就是采集清单；缺点是没有 `run_full_physics_batch.py` 那样完整的实时进度和训练质量汇总。

### 方法 E：分阶段 smoke 后再 full-physics

新轨迹或新目标不建议直接大规模 full-physics。更稳的自动化顺序是：

```bash
# 1. 只看 PCT 路线
python -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --pct-plan-preview \
  --no-headless \
  --output-dir outputs/check_pct

# 2. 真实导航 smoke
python -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --navigation-smoke \
  --headless \
  --output-dir outputs/check_nav

# 3. pick smoke
python -B scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --pick-smoke \
  --headless \
  --output-dir outputs/check_pick

# 4. 小批量 full-physics
python -B scripts/pipeline/run_full_physics_batch.py \
  --scene-profile liangzhu \
  --output-dir outputs/check_full \
  --num-episodes 5 \
  --seed 9000 \
  --headless \
  --record-dataset
```

## 8. 质量门禁与筛选

采集完成后优先看每个 episode 的 `summary.json`：

- `success`: pipeline 是否成功到达终态。
- `training_quality_gate_passed`: 是否通过训练数据质量门禁。
- `failure_reason` / `failure_metadata`: 失败原因和失败阶段。
- `lerobot_export`: LeRobot 导出路径、样本数和验证信息。
- `data_output_path`: episode 原始数据路径。

`run_full_physics_batch.py` 的汇总表会把 `Pipeline 成功` 和 `训练质量门禁` 分开显示。训练集应只接收两者都通过的 episode。

验证 LeRobot episode 可用：

```bash
python -B scripts/pipeline/validate_lerobot_episode.py \
  --episode-dir outputs/liangzhu_batch_seed5000/episode_000000
```

也可以用 `scripts/pipeline/validate_full_pipeline_acceptance.py` 做 full-pipeline acceptance 检查。

## 9. 常见调参入口

相机与数据：

- `--overview`
- `--dataset-camera-keys front wrist`
- `--record-dataset` / `--no-record-dataset`
- `--record-video` / `--no-record-video`
- `--video-mode overview|front|wrist|composite|all`
- `--overview-camera-mode fixed|auto`
- `--overview-camera-schedule configs/recording/*.json`

导航与 PCT：

- `--global-planner pct|astar`
- `--pct-no-fallback` / `--pct-allow-fallback`
- `--pct-collision-ply-path`
- `--pct-tomogram-path`
- `--pct-walkable-path`
- `--goal-z-tolerance`
- `--navigation-visual-mode collision|full|auto`

随机化：

- `--seed`
- `--num-episodes`
- `--randomize-task`
- `--randomize-base-goal`
- `--show-randomization-debug`

运行模式：

- `--dry-run`
- `--simulation-smoke`
- `--navigation-smoke`
- `--navigation-carry-smoke`
- `--stair-locomotion-smoke`
- `--pct-plan-preview`
- `--pick-smoke`
- `--manipulation-smoke`
- `--manipulation-apply-smoke`
- 默认不传模式即 `full_physics`

## 10. 推荐采集工作流

1. 从已有 task JSON 复制一个新任务，先只改 `instruction`、`start`、`pick` 和 `place` 的核心 pose。
2. 用 `--pct-plan-preview --no-headless` 检查路线。
3. 用 `--pick-smoke` 检查导航到 pick、在线 cuRobo 规划和抓取。
4. 用 3 到 5 条 full-physics 小批量检查 `summary.json`、视频和 `samples.jsonl`。
5. 固定通过的小批配置，使用 `run_full_physics_batch.py --num-episodes N --seed S` 批采。
6. 只把 `success=true` 且 `training_quality_gate_passed=true` 的 episode 纳入训练。
