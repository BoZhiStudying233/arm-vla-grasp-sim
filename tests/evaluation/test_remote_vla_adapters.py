from __future__ import annotations

import math

import pytest

from source.evaluation.adapters import (
    RecedingHorizonNavExecutor,
    RemoteVLANavPlanner,
)
from source.evaluation.protocol import PolicyDecision
from source.interfaces import EpisodeSpec, NavGoal, RobotAction, SimulationState


class _Session:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.inference_count = 0

    def predict(self, state, *, phase):
        self.inference_count += 1
        return next(self.decisions)


class _Executor:
    def __init__(self):
        self.plans = []
        self.done = False

    def reset(self, plan):
        self.plans.append(plan)
        self.done = False

    def compute_action(self, state):
        self.done = True
        return RobotAction(base_velocity=(0.1, 0.0, 0.0), source="fake")

    def is_done(self, state):
        return self.done

    def status(self):
        return {"success": self.done, "failed": False}


def _episode() -> EpisodeSpec:
    return EpisodeSpec(
        task_id=0,
        episode_id=7,
        instruction="Move the cola from box one to box two.",
        scene_usd="scene.usd",
        nav_map="map.json",
        start=NavGoal(0.0, 0.0, math.pi / 2),
        pick_goal=NavGoal(0.0, 1.0, math.pi / 2),
        place_goal=NavGoal(0.0, 2.0, math.pi / 2),
        object_prim_path=None,
        object_initial_pose=None,
        place_target_pose=None,
    )


def _state(yaw: float = math.pi / 2) -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(
            0.0,
            0.0,
            0.5,
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        ),
        robot_root_velocity=(0.0,) * 6,
    )


def test_body_waypoints_are_transformed_to_world_plan() -> None:
    session = _Session(
        [
            PolicyDecision(
                route="nav",
                nav_waypoints=((0.25, 0.0, 0.0), (0.50, 0.0, 0.1)),
            )
        ]
    )
    planner = RemoteVLANavPlanner(session, _episode())
    plan = planner.plan(_state(), _episode().pick_goal)
    assert plan.waypoints[0] == (0.0, 0.0)
    assert plan.waypoints[1][0] == pytest.approx(0.0, abs=1e-6)
    assert plan.waypoints[1][1] == pytest.approx(0.25)
    assert plan.goal.y == pytest.approx(0.50)
    assert plan.goal.yaw == pytest.approx(math.pi / 2 + 0.1)


def test_executor_replans_then_stops_on_grasp_route() -> None:
    session = _Session(
        [
            PolicyDecision(route="nav", nav_waypoints=((0.25, 0.0, 0.0),)),
            PolicyDecision(route="grasp"),
        ]
    )
    planner = RemoteVLANavPlanner(session, _episode())
    base = _Executor()
    executor = RecedingHorizonNavExecutor(planner, base)
    executor.reset(planner.plan(_state(), _episode().pick_goal))
    executor.compute_action(_state())
    assert not executor.is_done(_state())
    executor.compute_action(_state())
    assert executor.is_done(_state())
    assert executor.status()["remote_vla_terminal_route"] == "grasp"


def test_zero_nav_chunk_is_rejected_instead_of_reusing_stale_motion() -> None:
    session = _Session(
        [PolicyDecision(route="nav", nav_waypoints=((0.0, 0.0, 0.0),))]
    )
    planner = RemoteVLANavPlanner(session, _episode())
    with pytest.raises(ValueError, match="no executable waypoint"):
        planner.plan(_state(), _episode().pick_goal)


def test_wrong_terminal_route_fails_before_manipulation() -> None:
    session = _Session([PolicyDecision(route="place")])
    planner = RemoteVLANavPlanner(session, _episode())
    executor = RecedingHorizonNavExecutor(planner, _Executor())
    executor.reset(planner.plan(_state(), _episode().pick_goal))

    status = executor.status()
    assert status["failed"] is True
    assert "expected=grasp actual=place" in status["failure_reason"]
