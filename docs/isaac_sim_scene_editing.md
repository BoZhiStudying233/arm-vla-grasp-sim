# Isaac Sim 场景编辑

## 启动编辑器

本项目使用 `isaac_locomani` conda 环境中的 pip 版 Isaac Sim。进入仓库并启动完整 GUI 编辑器：

```bash
cd /home/natural/Desktop/mtr/pct_scene
conda activate isaac_locomani
export ISAAC_PYTHON="$(command -v python)"
isaacsim isaacsim.exp.full
```

`isaacsim` 不显式指定 experience 时也默认使用 `isaacsim.exp.full`。启动前可确认入口和 Python 来自同一环境：

```bash
command -v python
command -v isaacsim
python -c "import isaacsim; print(isaacsim.__file__)"
```

## 编辑良渚场景

任务当前加载 `source/scene/liangzhu/liangzhu.usda`。该文件是薄封装层，组合了高斯可视场景、物理碰撞场景、Go2-X5、箱子、可乐罐和相机。为避免直接修改基准场景，先复制封装层：

```bash
cp source/scene/liangzhu/liangzhu.usda \
  source/scene/liangzhu/liangzhu_obstacles.usda
```

在 Isaac Sim 中执行：

1. 通过 `File -> Open` 打开 `source/scene/liangzhu/liangzhu_obstacles.usda`。
2. 在 `/World/TaskObstacles` 下创建几何体或引用新的 USD 资产。
3. 设置世界坐标位置、旋转和尺寸。
4. 静态障碍物添加 `Physics -> Collider`，不要添加 Rigidbody。
5. 可移动障碍物同时添加 `Rigid Body`、Collider 和 Mass。
6. 保存场景，并复制任务 JSON，将 `scene_usd` 指向新的 USD。

建议复制 `tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json` 后修改，保留原任务作为回归基线。

## 同步碰撞与导航

场景中的可见几何、PhysX 碰撞和 PCT 导航地图是三套不同信息：

- 只有可见 Mesh：相机能够看到，但机器人可能穿过它。
- 添加 Collider：PhysX 会发生碰撞，但 PCT/DWA 仍可能使用旧地图规划穿过障碍物。
- 更新导航资产：固定障碍物应并入 collision PLY，再重新生成 tomogram 和 walkable。

少量临时障碍物可在任务 JSON 中添加 DWA 局部 keepout：

```json
"navigation_dynamic_keepouts": [
  {
    "id": "obstacle_01",
    "center_xy": [1.0, 2.0],
    "radius_m": 0.45,
    "phases": ["nav_to_pick", "nav_to_place"]
  }
]
```

该字段只提供局部避障约束。正式固定场景应把障碍物导出并合并到 collision PLY，然后运行：

```bash
python scripts/navigation/build_pct_multifloor_assets.py \
  --collision-ply /path/to/updated_collision.ply \
  --output-tomogram /path/to/updated_tomogram.pickle \
  --output-walkable /path/to/updated_walkable.npy \
  --report-output /path/to/build_report.json
```

随后在 `configs/scenes/<profile>.json` 中更新 `pct_collision_ply_path`、`pct_tomogram_path` 和 `pct_walkable_path`。如果障碍物进入机械臂工作空间，还必须把它加入 cuRobo 的碰撞世界；PhysX Collider 不会自动成为 cuRobo 障碍物。

## 检查运行时场景

需要查看 pipeline 实际组合和随机化后的 stage 时，可使用非 headless smoke。该方式适合检查，不建议把运行时 stage 直接保存成正式场景：

```bash
PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PYTHON" -B \
  scripts/pipeline/run_full_physics_pipeline.py \
  --scene-profile liangzhu \
  --output-dir "$PCT_SCENE_OUTPUT/liangzhu_scene_inspection" \
  --seed 3000 \
  --simulation-smoke \
  --navigation-visual-mode full \
  --no-headless \
  --keep-window-open
```

正式修改完成后，依次执行 preflight、simulation smoke、navigation smoke，再运行单 episode full-physics，确认可视效果、物理碰撞、导航绕障和机械臂规划均使用同一版场景资产。
