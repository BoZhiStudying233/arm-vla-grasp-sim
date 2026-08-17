# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ARM_VLA_ROOT = Path(os.environ.get("ARM_VLA_ROOT", Path(__file__).resolve().parents[4]))
GO2_X5_URDF = ARM_VLA_ROOT / "source/robot/go2_x5/urdf/go2_x5.urdf"
GRIPPER_MIMIC_NATURAL_FREQUENCY_HZ = 0.0
GRIPPER_MIMIC_DAMPING_RATIO = 0.0
# With soft_joint_pos_limit_factor=0.9, the effective lower limit of the
# 0..44 mm finger joints is 2.2 mm.  Seed both coordinates consistently so
# the hard mimic constraint starts satisfied instead of correcting a one-frame
# master-only clamp during articulation initialization.
GRIPPER_INITIAL_POSITION_M = 0.0022


def _spawn_go2_x5_with_hard_gripper_mimic(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Import the URDF and make joint8 a hard, drive-free mimic follower."""

    prim = sim_utils.spawn_from_urdf(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    from pxr import Sdf, UsdPhysics

    stage = sim_utils.get_current_stage()
    followers = [
        candidate
        for candidate in stage.Traverse()
        if candidate.GetName() == "arm_joint8"
    ]
    patched = 0
    for follower in followers:
        mimic_schemas = [
            str(schema)
            for schema in follower.GetAppliedSchemas()
            if str(schema).startswith("PhysxMimicJointAPI:")
        ]
        for mimic_schema in mimic_schemas:
            instance_name = mimic_schema.split(":", 1)[1]
            for suffix, value in (
                ("naturalFrequency", GRIPPER_MIMIC_NATURAL_FREQUENCY_HZ),
                ("dampingRatio", GRIPPER_MIMIC_DAMPING_RATIO),
            ):
                attribute_name = f"physxMimicJoint:{instance_name}:{suffix}"
                attribute = follower.GetAttribute(attribute_name)
                if not attribute:
                    attribute = follower.CreateAttribute(
                        attribute_name,
                        Sdf.ValueTypeNames.Float,
                    )
                attribute.Set(float(value))
            # The follower must never receive a second drive in parallel with
            # the two-way mimic constraint.
            follower.RemoveAPI(UsdPhysics.DriveAPI, "linear")
            patched += 1
    if patched != 1:
        raise RuntimeError(
            "X5 gripper requires exactly one arm_joint8 PhysX mimic API, "
            f"patched={patched}"
        )
    return prim

##
# Configuration
##

GO2_X5_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=_spawn_go2_x5_with_hard_gripper_mimic,
        fix_base=False,
        merge_fixed_joints=True,
        # IsaacLab 2.x uses this flag to pass set_parse_mimic(True) to the
        # Isaac Sim importer.  joint8 remains a dependent PhysX joint.
        convert_mimic_joints_to_normal_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=str(GO2_X5_URDF),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            # 机器狗关节
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            # 机械臂关节（全零默认姿态）
            "arm_joint1": 0.0,
            "arm_joint2": 0.0,
            "arm_joint3": 0.0,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
            "arm_joint7": GRIPPER_INITIAL_POSITION_M,
            "arm_joint8": GRIPPER_INITIAL_POSITION_M,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        # manipulation baseline 通过 Isaac Sim implicit drive 跟踪 ArticulationAction。
        # 这里不要用 DCMotorCfg 的速度-力矩曲线限幅机械臂，否则 arm_joint4 等腕部
        # 关节在抓取轨迹中会明显滞后，表现为 TCP 低头并扫到桌面。
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1000.0,
            damping=50.0,
            friction=0.0,
        ),
        "gripper": DCMotorCfg(
            # joint8 is physically driven by the hard mimic constraint only.
            joint_names_expr=["arm_joint7"],
            effort_limit=20.0,
            saturation_effort=20.0,
            velocity_limit=1.0,
            stiffness=1000.0,
            damping=50.0,
            friction=0.0,
        ),
    },
)


# 该配置只服务 pct_multifloor checkpoint；机器人资产继续复用 URDF，避免 stale USD 覆盖材质和关节结构。
GO2_X5_PCT_DOG_ONLY_CFG = GO2_X5_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            "FR_hip_joint": 0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "FL_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "RR_hip_joint": 0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
            "RL_hip_joint": -0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "arm_joint1": 0.0,
            "arm_joint2": 0.3,
            "arm_joint3": 0.5,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
            "arm_joint7": GRIPPER_INITIAL_POSITION_M,
            "arm_joint8": GRIPPER_INITIAL_POSITION_M,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs_hip_thigh": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=35.278,
            saturation_effort=35.278,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=44.4,
            saturation_effort=44.4,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        # DogOnly 策略没有机械臂 action 槽位，因此机械臂仍走 pipeline
        # 独立 position target；这里必须复用 baseline 的 implicit drive，
        # 避免 DCMotor 速度/力矩曲线让 arm_joint4 等腕部关节滞后。
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1000.0,
            damping=50.0,
            friction=0.0,
        ),
        # 主 pipeline 后续仍要抓取，保留本地两指夹爪 actuator。
        "gripper": DCMotorCfg(
            joint_names_expr=["arm_joint7"],
            effort_limit=20.0,
            saturation_effort=20.0,
            velocity_limit=1.0,
            stiffness=1000.0,
            damping=50.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 using DC motor.
"""
