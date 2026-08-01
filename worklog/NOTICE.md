# 项目工作约定

- 本地仓库：`/home/natural/Desktop/mtr/pct_scene`
- 本地环境：`isaac_locomani`
- 当前分支：`mtr_dev`
- Fork：`git@github.com:Natural-Horse/arm-vla-grasp-sim.git`（`origin`）
- 上游：`https://github.com/yagami-light7/arm-vla-grasp-sim.git`（`upstream`）
- 本地采集输出：`/home/natural/pct_scene_outputs`
- 模型仓库：`/home/natural/Desktop/mtr/starVLA_sc`，分支 `robodog`

工作时保留用户已有改动。先在本地修改、测试并提交到 `mtr_dev`；远端评测机的仓库路径和同步方式必须在首次写入前确认，不可从训练服务器路径推断。

仿真模型评测采用明确分层：`starVLA_sc` 负责模型加载、版本化协议和机体系稀疏 waypoint 输出；`pct_scene` 负责观测编码、waypoint 世界系变换、DWA/RL 速度适配、Isaac 状态机及 cuRobo 抓放。两仓库不复制模型业务代码，机械臂不直接执行模型文本或远端关节指令。

评测机只连接回环 WebSocket，经 `scripts/evaluation/manage_remote_vla_tunnel.sh` 建立 SSH 本地转发。远端服务必须绑定 `127.0.0.1`。网络超时、协议不匹配和 waypoint 越界均安全停车并记录失败。

所有日志与说明文档使用中文。每天只写 `worklog/YYYY-MM-DD.md`，结束时刷新精简的 `## 全日总结`。
