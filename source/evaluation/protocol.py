"""Client-side types for the StarVLA Go2 evaluation protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "starvla-go2-eval/v1"
ROUTES = frozenset({"nav", "grasp", "place", "done", "recover"})


class RemotePolicyError(RuntimeError):
    """The remote service failed or violated the protocol."""


@dataclass(frozen=True)
class PolicyDecision:
    route: str
    subtask: str | None = None
    nav_waypoints: tuple[tuple[float, float, float], ...] = ()
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
        confidence = data.get("route_confidence")
        if confidence is not None and not math.isfinite(float(confidence)):
            raise RemotePolicyError("route_confidence is non-finite")
        return cls(
            route=route,
            subtask=None if data.get("subtask") is None else str(data["subtask"]),
            nav_waypoints=tuple(waypoints),
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
            "route_confidence": self.route_confidence,
            "route_probs": self.route_probs,
            "raw_text": self.raw_text,
            **self.metadata,
        }
