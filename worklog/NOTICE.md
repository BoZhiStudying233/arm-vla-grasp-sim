# 项目工作约定

- 本地仓库：`/home/natural/Desktop/mtr/pct_scene`
- 本地环境：`isaac_locomani`
- 当前分支：`mtr_dev`
- Fork：`git@github.com:Natural-Horse/arm-vla-grasp-sim.git`（`origin`）
- 上游：`https://github.com/yagami-light7/arm-vla-grasp-sim.git`（`upstream`）
- 本地采集输出：`/home/natural/pct_scene_outputs`
- 模型仓库：`/home/natural/Desktop/mtr/starVLA_sc`，分支 `robodog`
- 真机仓库：`/home/natural/Desktop/mtr/gx-real`，分支 `mtr_dev`
- 真机 Fork：`https://github.com/Natural-Horse/gx-real.git`（`origin`）
- 真机上游：`https://github.com/lemonoscar/gx-real.git`（`upstream`）

涉及 `gx-real` 时，提交、分支推送和 PR 只面向用户 Fork `Natural-Horse/gx-real`（`origin`）；`upstream` 仅供只读参考，不得向其推送或创建 PR。推送前必须核对 `git remote get-url --push origin`。

工作时保留用户已有改动。先在本地修改、测试并提交到 `mtr_dev`；远端评测机的仓库路径和同步方式必须在首次写入前确认，不可从训练服务器路径推断。

仿真模型评测采用明确分层：`starVLA_sc` 负责模型加载、版本化协议和机体系稀疏 waypoint 输出；`pct_scene` 负责观测编码、waypoint 世界系变换、DWA/RL 速度适配、Isaac 状态机及 cuRobo 抓放。两仓库不复制模型业务代码，机械臂不直接执行模型文本或远端关节指令。

统一动作协议为 10 维 `[dx_body,dy_body,dyaw,tcp_x_base,tcp_y_base,tcp_z_base,roll_base,pitch_base,yaw_base,gripper]`。NAV 只使用前三维；GRASP/PLACE 只使用后七维，并由 pct_scene 将机体系 TCP 目标送入本地 cuRobo、安全检查和仿真执行器。真机侧 gx-real 使用同一语义，但保留独立标定和硬件安全层。

评测机只连接回环 WebSocket，经 `scripts/evaluation/manage_remote_vla_tunnel.sh` 建立 SSH 本地转发。远端服务必须绑定 `127.0.0.1`。网络超时、协议不匹配和 waypoint 越界均安全停车并记录失败。

所有日志与说明文档使用中文。每天只写 `worklog/YYYY-MM-DD.md`，结束时刷新精简的 `## 全日总结`。
