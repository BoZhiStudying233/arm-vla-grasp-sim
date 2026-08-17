"""Client-side types for the StarVLA Go2 evaluation protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "starvla-go2-eval/v2"
ROUTES = frozenset({"nav", "grasp", "place", "done", "recover"})


class RemotePolicyError(RuntimeError):
    """The remote service failed or violated the protocol."""


@dataclass(frozen=True)
class PolicyDecision:
    route: str
    subtask: str | None = None
    nav_waypoints: tuple[tuple[float, float, float], ...] = ()
    arm_targets_base: tuple[
        tuple[float, float, float, float, float, float, float], ...
    ] = ()
    route_confidence: float | None = None
    route_probs: dict[str, float | None] | None = None
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(
        cls,
        data: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "PolicyDecision":
        if not isinstance(data, dict):
            raise RemotePolicyError("inference response data must be an object")
        route = str(data.get("route", "")).strip().lower()
        if route not in ROUTES:
            raise RemotePolicyError(f"unsupported route={route!r}")
        raw_waypoints = data.get("nav_waypoints")
        waypoints: list[tuple[float, float, float]] = []
        if route == "nav":
            if not isinstance(raw_waypoints, list) or not raw_waypoints:
                raise RemotePolicyError("NAV response has no nav_waypoints")
            if len(raw_waypoints) > 32:
                raise RemotePolicyError("NAV response exceeds 32 waypoints")
            for index, point in enumerate(raw_waypoints):
                if not isinstance(point, list) or len(point) != 3:
                    raise RemotePolicyError(
                        f"nav_waypoints[{index}] must contain [dx,dy,dyaw]"
                    )
                values = tuple(float(value) for value in point)
                if not all(math.isfinite(value) for value in values):
                    raise RemotePolicyError(f"nav_waypoints[{index}] is non-finite")
                waypoints.append(values)
        raw_arm_targets = data.get("arm_targets_base")
        arm_targets: list[tuple[float, float, float, float, float, float, float]] = []
        if raw_arm_targets is not None:
            if not isinstance(raw_arm_targets, list) or not raw_arm_targets:
                raise RemotePolicyError("arm_targets_base must be a non-empty list")
            if len(raw_arm_targets) > 32:
                raise RemotePolicyError("arm_targets_base exceeds 32 targets")
            for index, target in enumerate(raw_arm_targets):
                if not isinstance(target, list) or len(target) != 7:
                    raise RemotePolicyError(
                        f"arm_targets_base[{index}] must contain [x,y,z,roll,pitch,yaw,gripper]"
                    )
                values = tuple(float(value) for value in target)
                if not all(math.isfinite(value) for value in values):
                    raise RemotePolicyError(f"arm_targets_base[{index}] is non-finite")
                if not 0.0 <= values[-1] <= 1.0:
                    raise RemotePolicyError(
                        f"arm_targets_base[{index}] gripper must be in [0,1]"
                    )
                arm_targets.append(values)
        confidence = data.get("route_confidence")
        if confidence is not None and not math.isfinite(float(confidence)):
            raise RemotePolicyError("route_confidence is non-finite")
        return cls(
            route=route,
            subtask=None if data.get("subtask") is None else str(data["subtask"]),
            nav_waypoints=tuple(waypoints),
            arm_targets_base=tuple(arm_targets),
            route_confidence=None if confidence is None else float(confidence),
            route_probs=data.get("route_probs"),
            raw_text=str(data.get("raw_text", "")),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "subtask": self.subtask,
            "nav_waypoints": [list(point) for point in self.nav_waypoints],
            "arm_targets_base": [list(target) for target in self.arm_targets_base],
            "route_confidence": self.route_confidence,
            "route_probs": self.route_probs,
            "raw_text": self.raw_text,
            **self.metadata,
        }
