"""共享的机器人本体八方向与全局任务 instruction 构造。"""

from __future__ import annotations

import math


RELATIVE_DIRECTION_LABELS = (
    "front",
    "front-left",
    "left",
    "back-left",
    "back",
    "back-right",
    "right",
    "front-right",
)


def relative_direction_label(relative_bearing_rad: float) -> str:
    """把机器人局部目标方位量化为稳定的八方向英文标签。"""

    angle = math.atan2(
        math.sin(float(relative_bearing_rad)),
        math.cos(float(relative_bearing_rad)),
    )
    if not math.isfinite(angle):
        raise ValueError("relative_bearing_rad 必须是有限数值")
    index = int(math.floor((angle + math.pi / 8.0) / (math.pi / 4.0))) % 8
    return RELATIVE_DIRECTION_LABELS[index]


def target_direction_from_pose(
    *,
    anchor_x: float,
    anchor_y: float,
    anchor_yaw: float,
    target_x: float,
    target_y: float,
) -> tuple[str, float]:
    """返回目标相对 anchor 位姿的八方向标签和局部 bearing。"""

    values = tuple(
        float(value)
        for value in (anchor_x, anchor_y, anchor_yaw, target_x, target_y)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("anchor/target pose 必须是有限数值")
    target_bearing = math.atan2(values[4] - values[1], values[3] - values[0])
    relative_bearing = math.atan2(
        math.sin(target_bearing - values[2]),
        math.cos(target_bearing - values[2]),
    )
    return relative_direction_label(relative_bearing), relative_bearing


def build_box_pair_global_instruction(
    base_instruction: str,
    *,
    box1_direction: str,
    box2_direction: str,
) -> str:
    """构造训练与在线评测共用的 box1/box2 方位任务描述。"""

    instruction = str(base_instruction).strip()
    if not instruction:
        raise ValueError("base_instruction 不能为空")
    for field_name, direction in (
        ("box1_direction", box1_direction),
        ("box2_direction", box2_direction),
    ):
        if direction not in RELATIVE_DIRECTION_LABELS:
            raise ValueError(f"{field_name} 不是有效八方向: {direction!r}")
    return (
        f"{instruction} "
        f"Box1 is to the robot's {box1_direction} from its initial pose. "
        f"Box2 is to the robot's {box2_direction} from its first pose after grasping."
    )


__all__ = [
    "RELATIVE_DIRECTION_LABELS",
    "build_box_pair_global_instruction",
    "relative_direction_label",
    "target_direction_from_pose",
]
