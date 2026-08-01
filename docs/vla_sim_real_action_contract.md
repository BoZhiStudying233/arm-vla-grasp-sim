# VLA 仿真与真机动作接口

`starVLA_sc`、`pct_scene` 与 `gx-real` 统一使用 10 维 route-conditioned 动作协议：

```text
[dx_body, dy_body, dyaw,
 tcp_x_base, tcp_y_base, tcp_z_base,
 roll_base, pitch_base, yaw_base, gripper]
```

- NAV 只使用前三维。pct_scene 将机体系 waypoint 转到世界系，再交给现有 DWA 和机器狗 locomotion policy。
- GRASP/PLACE 只使用后七维。它们是机体系 TCP 目标，不是机械臂关节角。
- 仿真执行端负责把 TCP 目标送入 cuRobo、碰撞检查、工作空间/速率限制和 Isaac 执行器。
- 真机执行端由 gx-real 做自己的标定、IK/规划、安全限幅、急停、看门狗及唯一 CAN owner 管理。

远程协议 `starvla-go2-eval/v2` 可携带 `nav_waypoints` 和 `arm_targets_base`。当前默认 pipeline 仍保留已有确定性 cuRobo 抓放；只有显式启用并通过仿真 smoke test 的机械臂动作适配器才可消费模型 TCP 目标。协议数据不得绕过本地规划器直接转换为关节或 CAN 指令。
