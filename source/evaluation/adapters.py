"""Typed route gate and receding-horizon adapter for the existing nav stack."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from source.interfaces import (
    EpisodeSpec,
    NavExecutor,
    NavGoal,
    NavPlan,
    RobotAction,
    SimulationState,
)
from source.navigation.adapters.frame_utils import wrap_yaw

from .client import RemotePolicyClient
from .observation import ObservationEncoder
from .protocol import PolicyDecision


@dataclass(frozen=True)
class WaypointSafetyConfig:
    max_segment_translation_m: float = 0.80
    max_segment_yaw_rad: float = math.radians(45.0)
    max_replans_per_navigation: int = 64


class RemotePolicySession:
    """Share one client and one observation contract across route and NAV calls."""

    def __init__(
        self,
        *,
        client: RemotePolicyClient,
        encoder: ObservationEncoder,
        episode_spec: EpisodeSpec,
    ) -> None:
        self.client = client
        self.encoder = encoder
        self.episode_spec = episode_spec
        self.inference_count = 0
        self.last_decision: PolicyDecision | None = None
        self.server_health: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        health = self.client.health()
        self.server_health = dict(health)
        self.client.reset(str(self.episode_spec.episode_id))
        return health

    def predict(self, state: SimulationState, *, phase: str) -> PolicyDecision:
        payload = self.encoder.build_payload(
            state,
            instruction=self.episode_spec.instruction,
            episode_id=str(self.episode_spec.episode_id),
            phase=phase,
        )
        decision = self.client.infer(payload)
        decision = replace(
            decision,
            metadata={
                **decision.metadata,
                "server_health": self.server_health,
                "inference_index": self.inference_count + 1,
            },
        )
        self.inference_count += 1
        self.last_decision = decision
        return decision

    def predict_route(self, state: SimulationState, *, phase: str) -> dict[str, Any]:
        return self.predict(state, phase=phase).to_dict()

    def close(self) -> None:
        self.client.close()


class RemoteVLANavPlanner:
    def __init__(
        self,
        session: RemotePolicySession,
        episode_spec: EpisodeSpec,
        safety: WaypointSafetyConfig | None = None,
    ) -> None:
        self.session = session
        self.episode_spec = episode_spec
        self.safety = safety or WaypointSafetyConfig()

    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        phase = self._phase_for_goal(goal)
        decision = self.session.predict(state, phase=phase)
        return self.plan_from_decision(state, goal, decision, phase=phase)

    def plan_from_decision(
        self,
        state: SimulationState,
        task_goal: NavGoal,
        decision: PolicyDecision,
        *,
        phase: str,
    ) -> NavPlan:
        current_x, current_y, current_yaw = ObservationEncoder._pose_xyyaw(state)
        metadata = {
            "planner": "remote_starvla",
            "remote_vla_route": decision.route,
            "remote_vla_subtask": decision.subtask,
            "remote_vla_route_confidence": decision.route_confidence,
            "remote_vla_raw_text": decision.raw_text,
            "remote_vla_transport": decision.metadata,
            "remote_vla_phase": phase,
            "evaluation_task_goal": {
                "x": task_goal.x,
                "y": task_goal.y,
                "yaw": task_goal.yaw,
                "z": task_goal.z,
                "floor_id": task_goal.floor_id,
                "slice_id": task_goal.slice_id,
            },
        }
        if decision.route != "nav":
            return NavPlan(
                goal=NavGoal(current_x, current_y, current_yaw),
                waypoints=((current_x, current_y), (current_x, current_y)),
                metadata=metadata,
            )

        world_points: list[tuple[float, float, float]] = []
        cosine = math.cos(current_yaw)
        sine = math.sin(current_yaw)
        for dx, dy, dyaw in decision.nav_waypoints:
            point = (
                current_x + cosine * dx - sine * dy,
                current_y + sine * dx + cosine * dy,
                wrap_yaw(current_yaw + dyaw),
            )
            previous = world_points[-1] if world_points else (current_x, current_y, current_yaw)
            if (
                math.hypot(point[0] - previous[0], point[1] - previous[1]) < 1.0e-3
                and abs(wrap_yaw(point[2] - previous[2])) < 1.0e-3
            ):
                continue
            world_points.append(point)
        if not world_points:
            raise ValueError("remote NAV decision contains no executable waypoint")
        self._validate_segments((current_x, current_y, current_yaw), world_points)
        final_x, final_y, final_yaw = world_points[-1]
        return NavPlan(
            goal=NavGoal(final_x, final_y, final_yaw),
            waypoints=((current_x, current_y), *((x, y) for x, y, _ in world_points)),
            metadata={
                **metadata,
                "remote_vla_waypoints_body": [
                    list(point) for point in decision.nav_waypoints
                ],
            },
        )

    def _phase_for_goal(self, goal: NavGoal) -> str:
        pick = self.episode_spec.pick_goal
        if math.hypot(goal.x - pick.x, goal.y - pick.y) < 1.0e-5:
            return "nav_to_pick"
        return "nav_to_place"

    def _validate_segments(
        self,
        start: tuple[float, float, float],
        points: list[tuple[float, float, float]],
    ) -> None:
        previous = start
        for index, point in enumerate(points):
            translation = math.hypot(point[0] - previous[0], point[1] - previous[1])
            yaw_delta = abs(wrap_yaw(point[2] - previous[2]))
            if translation > self.safety.max_segment_translation_m:
                raise ValueError(
                    f"remote waypoint segment {index} translation {translation:.3f} m exceeds "
                    f"{self.safety.max_segment_translation_m:.3f} m"
                )
            if yaw_delta > self.safety.max_segment_yaw_rad:
                raise ValueError(
                    f"remote waypoint segment {index} yaw {yaw_delta:.3f} rad exceeds "
                    f"{self.safety.max_segment_yaw_rad:.3f} rad"
                )
            previous = point


class RecedingHorizonNavExecutor:
    """Refresh a sparse model chunk whenever the existing executor finishes it."""

    def __init__(self, planner: RemoteVLANavPlanner, executor: NavExecutor) -> None:
        self.planner = planner
        self.executor = executor
        self.task_goal: NavGoal | None = None
        self.plan_metadata: dict[str, Any] = {}
        self.terminal_route: str | None = None
        self.failed = False
        self.failure_reason = ""
        self.replan_count = 0
        self.last_replan_report: dict[str, Any] = {}

    def reset(self, plan: NavPlan) -> None:
        raw_goal = plan.metadata.get("evaluation_task_goal") or {}
        self.task_goal = NavGoal(
            float(raw_goal["x"]),
            float(raw_goal["y"]),
            float(raw_goal["yaw"]),
            z=raw_goal.get("z"),
            floor_id=raw_goal.get("floor_id"),
            slice_id=raw_goal.get("slice_id"),
        )
        self.plan_metadata = dict(plan.metadata)
        self.terminal_route = None
        self.failed = False
        self.failure_reason = ""
        self.replan_count = 0
        self.last_replan_report = {}
        self._accept_plan(plan)

    def compute_action(self, state: SimulationState) -> RobotAction:
        if self.failed or self.terminal_route is not None:
            return RobotAction(
                source="remote_vla_navigation_stop",
                metadata={"remote_vla_terminal_route": self.terminal_route},
            )
        if not self.executor.is_done(state):
            return self.executor.compute_action(state)
        if self.replan_count >= self.planner.safety.max_replans_per_navigation:
            self.failed = True
            self.failure_reason = "remote_vla_replan_limit"
            return RobotAction.idle(source=self.failure_reason)
        assert self.task_goal is not None
        try:
            refreshed = self.planner.plan(state, self.task_goal)
            refreshed = replace(
                refreshed,
                metadata={**refreshed.metadata, **self._execution_metadata()},
            )
            self.replan_count += 1
            self._accept_plan(refreshed)
            self.last_replan_report = {
                "replan_count": self.replan_count,
                "route": refreshed.metadata.get("remote_vla_route"),
                "subtask": refreshed.metadata.get("remote_vla_subtask"),
                "waypoints_body": refreshed.metadata.get("remote_vla_waypoints_body"),
                "transport": refreshed.metadata.get("remote_vla_transport"),
            }
        except Exception as exc:
            self.failed = True
            self.failure_reason = f"remote_vla_replan_failed: {exc}"
        return RobotAction(
            source="remote_vla_replan",
            metadata={**self.last_replan_report, "failed": self.failed},
        )

    def is_done(self, state: SimulationState) -> bool:
        return self.terminal_route is not None

    def status(self) -> dict[str, Any]:
        underlying = self.executor.status()
        return {
            **underlying,
            "failed": self.failed or bool(underlying.get("failed")),
            "failure_reason": (
                self.failure_reason or underlying.get("failure_reason", "")
            ),
            "remote_vla_replan_count": self.replan_count,
            "remote_vla_terminal_route": self.terminal_route,
            "remote_vla_inference_count": self.planner.session.inference_count,
            "remote_vla_last_replan": dict(self.last_replan_report),
        }

    def _accept_plan(self, plan: NavPlan) -> None:
        route = str(plan.metadata.get("remote_vla_route", ""))
        if route == "nav":
            self.executor.reset(plan)
        else:
            self.terminal_route = route or "recover"
            expected_route = {
                "nav_to_pick": "grasp",
                "nav_to_place": "place",
            }.get(str(plan.metadata.get("remote_vla_phase", "")))
            if expected_route is not None and self.terminal_route != expected_route:
                self.failed = True
                self.failure_reason = (
                    "remote_vla_unexpected_navigation_exit_route: "
                    f"expected={expected_route} actual={self.terminal_route}"
                )

    def _execution_metadata(self) -> dict[str, Any]:
        keys = (
            "execution_phase",
            "carry_departure",
            "require_yaw_alignment",
            "yaw_tolerance",
        )
        return {key: self.plan_metadata[key] for key in keys if key in self.plan_metadata}
