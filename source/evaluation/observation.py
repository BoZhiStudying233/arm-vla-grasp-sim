"""Encode simulation observations for StarVLA without coupling to Isaac APIs."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from source.interfaces import SimulationState
from source.navigation.adapters.frame_utils import world_velocity_to_body

from .protocol import RemotePolicyError


def _yaw_from_wxyz(quaternion: tuple[float, ...]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_conjugate(quaternion: tuple[float, ...]) -> tuple[float, float, float, float]:
    w, x, y, z = quaternion
    return w, -x, -y, -z


def _quat_multiply(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_to_rpy(quaternion: tuple[float, ...]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = _yaw_from_wxyz(quaternion)
    return roll, pitch, yaw


def _rotate_vector(
    quaternion: tuple[float, ...], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    pure = (0.0, *vector)
    rotated = _quat_multiply(_quat_multiply(quaternion, pure), _quat_conjugate(quaternion))
    return rotated[1], rotated[2], rotated[3]


@dataclass(frozen=True)
class ObservationEncoder:
    jpeg_quality: int = 90
    front_camera_key: str = "front"
    wrist_camera_key: str = "wrist"
    gripper_closed_m: float = 0.0
    gripper_open_m: float = 0.04

    def build_payload(
        self,
        state: SimulationState,
        *,
        instruction: str,
        episode_id: str,
        phase: str,
    ) -> dict[str, Any]:
        front = state.camera_images.get(self.front_camera_key)
        if front is None:
            raise RemotePolicyError(
                f"simulation observation has no {self.front_camera_key!r} camera image"
            )
        images = {"front": self._encode_jpeg(front)}
        wrist = state.camera_images.get(self.wrist_camera_key)
        if wrist is not None:
            images["wrist"] = self._encode_jpeg(wrist)
        return {
            "episode_id": str(episode_id),
            "frame_index": int(state.step_index),
            "phase": str(phase),
            "instruction": str(instruction),
            "images": images,
            "state": {
                "base_velocity_body": list(self._body_velocity(state)),
                "base_world_xyyaw": list(self._pose_xyyaw(state)),
                "arm_tcp_base": list(self._arm_tcp_base(state)),
            },
        }

    def _encode_jpeg(self, image: Any) -> dict[str, str]:
        try:
            import cv2
        except ImportError as exc:
            raise RemotePolicyError("opencv-python is required for JPEG encoding") from exc
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise RemotePolicyError(
                f"camera image must have shape [H,W,3|4], got {array.shape}"
            )
        if np.issubdtype(array.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        color_code = cv2.COLOR_RGBA2BGR if array.shape[2] == 4 else cv2.COLOR_RGB2BGR
        bgr = cv2.cvtColor(array, color_code)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            raise RemotePolicyError("failed to encode camera image as JPEG")
        return {
            "encoding": "jpeg_base64",
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    @staticmethod
    def _pose_xyyaw(state: SimulationState) -> tuple[float, float, float]:
        pose = state.robot_root_pose
        return float(pose[0]), float(pose[1]), _yaw_from_wxyz(tuple(pose[3:7]))

    @classmethod
    def _body_velocity(cls, state: SimulationState) -> tuple[float, float, float]:
        measured = state.metadata.get("body_velocity")
        if isinstance(measured, (list, tuple)) and len(measured) >= 3:
            return tuple(float(value) for value in measured[:3])
        yaw = cls._pose_xyyaw(state)[2]
        vx, vy = world_velocity_to_body(
            float(state.robot_root_velocity[0]),
            float(state.robot_root_velocity[1]),
            yaw,
        )
        return vx, vy, float(state.robot_root_velocity[5])

    def _arm_tcp_base(
        self, state: SimulationState
    ) -> tuple[float, float, float, float, float, float, float]:
        if state.tcp_pose is None:
            return (0.0,) * 7
        root_position = tuple(float(value) for value in state.robot_root_pose[:3])
        root_quaternion = tuple(float(value) for value in state.robot_root_pose[3:7])
        tcp_position = tuple(float(value) for value in state.tcp_pose[:3])
        tcp_quaternion = tuple(float(value) for value in state.tcp_pose[3:7])
        delta_world = tuple(tcp_position[i] - root_position[i] for i in range(3))
        base_inverse = _quat_conjugate(root_quaternion)
        tcp_position_base = _rotate_vector(base_inverse, delta_world)
        tcp_quaternion_base = _quat_multiply(base_inverse, tcp_quaternion)
        tcp_rpy_base = _quat_to_rpy(tcp_quaternion_base)

        names = tuple(str(name) for name in state.metadata.get("joint_names", ()))
        gripper_values = [
            float(state.joint_positions[index])
            for index, name in enumerate(names)
            if index < len(state.joint_positions) and "gripper" in name.lower()
        ]
        gripper_m = sum(gripper_values) / len(gripper_values) if gripper_values else 0.0
        span = self.gripper_open_m - self.gripper_closed_m
        gripper = 0.0 if span <= 0.0 else (gripper_m - self.gripper_closed_m) / span
        gripper = max(0.0, min(1.0, gripper))
        return (*tcp_position_base, *tcp_rpy_base, gripper)
