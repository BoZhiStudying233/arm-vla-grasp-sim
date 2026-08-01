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


@dataclass(frozen=True)
class ObservationEncoder:
    jpeg_quality: int = 90
    front_camera_key: str = "front"
    wrist_camera_key: str = "wrist"

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
