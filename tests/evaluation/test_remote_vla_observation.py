from __future__ import annotations

import pytest

from source.evaluation.observation import ObservationEncoder
from source.interfaces import SimulationState


def test_arm_state_is_encoded_in_robot_base_frame() -> None:
    state = SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0,) * 6,
        joint_positions=(0.02, 0.02),
        tcp_pose=(1.3, 2.0, 0.7, 1.0, 0.0, 0.0, 0.0),
        metadata={"joint_names": ("left_gripper", "right_gripper")},
    )
    arm = ObservationEncoder()._arm_tcp_base(state)
    assert arm == pytest.approx((0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.5))
