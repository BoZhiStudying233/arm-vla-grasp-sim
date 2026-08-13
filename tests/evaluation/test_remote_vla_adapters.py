from __future__ import annotations

import math

import pytest

from source.evaluation.adapters import (
    RemoteVLAArmTargetShadowValidator,
    RecedingHorizonNavExecutor,
    RemoteVLANavPlanner,
    RemotePolicySession,
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


def test_first_waypoint_only_keeps_single_body_waypoint() -> None:
    session = _Session(
        [
            PolicyDecision(
                route="nav",
                nav_waypoints=((0.25, 0.0, 0.0), (0.50, 0.0, 0.1)),
            )
        ]
    )
    planner = RemoteVLANavPlanner(session, _episode(), first_waypoint_only=True)
    plan = planner.plan(_state(), _episode().pick_goal)
    assert len(plan.waypoints) == 2  # start + first waypoint
    assert plan.metadata["remote_vla_waypoints_body"] == [[0.25, 0.0, 0.0]]
    assert plan.goal.y == pytest.approx(0.25)


def test_remote_policy_session_arm_gate_skips_grasp() -> None:
    class _Client:
        def health(self):
            return {"ok": True}

        def reset(self, episode_id):
            return {"ok": True}

        def infer(self, payload):
            return PolicyDecision(
                route="grasp",
                subtask="Pick up the can.",
                arm_targets_base=((0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),),
            )

    class _Encoder:
        def build_payload(self, state, *, instruction, episode_id, phase):
            return {"phase": phase}

    gate_calls = []

    def gate(decision):
        gate_calls.append(decision.route)
        return False

    session = RemotePolicySession(
        client=_Client(),
        encoder=_Encoder(),
        episode_spec=_episode(),
        arm_mode="shadow",
        arm_gate=gate,
    )
    decision = session.predict(_state(), phase="nav_to_pick")

    assert gate_calls == ["grasp"]
    assert decision.route == "recover"
    assert decision.subtask is not None
    assert decision.subtask.startswith("operator_skipped_grasp")
    assert decision.arm_targets_base == ()
    assert session.locked_route is None


def test_remote_policy_session_reuses_and_updates_route_lock() -> None:
    class _Client:
        def __init__(self):
            self.payloads = []
            self.decisions = iter(
                (
                    PolicyDecision(
                        route="nav",
                        subtask="Move toward box1.",
                        nav_waypoints=((0.25, 0.0, 0.0),),
                    ),
                    PolicyDecision(
                        route="nav",
                        subtask="Move toward box1.",
                        nav_waypoints=((0.20, 0.0, 0.0),),
                    ),
                    PolicyDecision(
                        route="grasp",
                        subtask="Pick up the can.",
                        arm_targets_base=((0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),),
                    ),
                    PolicyDecision(route="done", subtask="Task complete."),
                )
            )

        def infer(self, payload):
            self.payloads.append(dict(payload))
            return next(self.decisions)

    class _Encoder:
        def build_payload(self, state, *, instruction, episode_id, phase):
            return {"phase": phase}

    client = _Client()
    session = RemotePolicySession(
        client=client,
        encoder=_Encoder(),
        episode_spec=_episode(),
    )

    session.predict(_state(), phase="nav_to_pick")
    session.predict(_state(), phase="nav_to_pick")
    session.predict(_state(), phase="nav_to_pick")
    session.predict(_state(), phase="after_place")

    assert "locked_route" not in client.payloads[0]
    assert client.payloads[1]["locked_route"] == "nav"
    assert client.payloads[1]["locked_subtask"] == "Move toward box1."
    assert client.payloads[2]["locked_route"] == "nav"
    assert client.payloads[3]["locked_route"] == "grasp"
    assert session.locked_route is None
    assert session.locked_subtask is None


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


def test_arm_shadow_validator_accepts_small_base_frame_chunk() -> None:
    state = SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0,) * 6,
        tcp_pose=(0.30, 0.0, 0.70, 1.0, 0.0, 0.0, 0.0),
    )
    decision = PolicyDecision(
        route="grasp",
        arm_targets_base=((0.34, 0.0, 0.22, 0.0, 0.1, 0.0, 0.4),),
    )
    report = RemoteVLAArmTargetShadowValidator().validate(state, decision)
    assert report["validated"] is True
    assert report["target_count"] == 1


def test_arm_shadow_validator_rejects_missing_targets() -> None:
    with pytest.raises(ValueError, match="no arm_targets_base"):
        RemoteVLAArmTargetShadowValidator().validate(
            _state(), PolicyDecision(route="place")
        )


def test_nav_executor_expires_a_stale_chunk() -> None:
    session = _Session(
        [PolicyDecision(route="nav", nav_waypoints=((0.25, 0.0, 0.0),))]
    )
    planner = RemoteVLANavPlanner(session, _episode())
    planner.safety = type(planner.safety)(max_chunk_execution_steps=1)
    base = _Executor()
    base.done = False
    executor = RecedingHorizonNavExecutor(planner, base)
    executor.reset(planner.plan(_state(), _episode().pick_goal))
    executor.compute_action(_state())
    late = SimulationState(
        step_index=2,
        timestamp=0.04,
        robot_root_pose=_state().robot_root_pose,
        robot_root_velocity=(0.0,) * 6,
    )
    executor.compute_action(late)
    assert executor.status()["failed"] is True
    assert executor.status()["failure_reason"] == "remote_vla_waypoint_chunk_expired"
