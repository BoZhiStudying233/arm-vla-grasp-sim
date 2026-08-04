import unittest

from source.evaluation.interactive import (
    InteractiveVLAConfig,
    InteractiveVLAController,
    waypoint_reached,
)


class _FakeDecision:
    def __init__(
        self,
        route,
        subtask=None,
        nav_waypoints=(),
        arm_targets_base=(),
        raw_text="",
    ):
        self.route = route
        self.subtask = subtask
        self.nav_waypoints = nav_waypoints
        self.arm_targets_base = arm_targets_base
        self.raw_text = raw_text


class _FakeClient:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.health_calls = 0
        self.reset_calls = []

    def health(self):
        self.health_calls += 1
        return {"ok": True}

    def reset(self, episode_id):
        self.reset_calls.append(episode_id)
        return {"ok": True}

    def infer(self, payload):
        return self.decisions.pop(0)


class _FakeObservation:
    def __init__(self):
        self.calls = []

    def build_payload(self, *, phase, frame_index):
        self.calls.append((phase, frame_index))
        return {"phase": phase, "frame_index": frame_index}


class _FakeWaypointExecutor:
    def __init__(self, reached=True):
        self.reached = reached
        self.calls = []

    def execute_waypoint(
        self, waypoint, *, tolerance_m, yaw_tolerance_rad, max_steps
    ):
        self.calls.append(
            (waypoint, tolerance_m, yaw_tolerance_rad, max_steps)
        )
        return {"reached": self.reached, "steps": 10}


class _FakeArmExecutor:
    def __init__(self):
        self.calls = []

    def execute_arm_target(self, target):
        self.calls.append(target)
        return {"accepted": True}


class InteractiveControllerTest(unittest.TestCase):
    def _config(self, **overrides):
        base = dict(
            instruction="Pick up the coke can.",
            endpoint="ws://127.0.0.1:10093",
            jsonl_out=None,
        )
        base.update(overrides)
        return InteractiveVLAConfig(**base)

    def test_nav_executes_first_waypoint_then_replans(self):
        client = _FakeClient(
            [
                _FakeDecision("nav", nav_waypoints=((1.0, 0.2, 0.0),)),
                _FakeDecision("nav", nav_waypoints=((0.5, 0.0, 0.0),)),
                _FakeDecision("done", subtask="done"),
            ]
        )
        obs = _FakeObservation()
        wp = _FakeWaypointExecutor()
        arm = _FakeArmExecutor()
        controller = InteractiveVLAController(
            client=client,
            observation=obs,
            waypoint_executor=wp,
            arm_executor=arm,
            config=self._config(),
            display=lambda _: None,
            input_fn=lambda _: "1",
        )
        summary = controller.run()

        self.assertEqual(summary["inference_count"], 3)
        self.assertEqual(summary["replan_count"], 2)
        self.assertEqual(len(wp.calls), 2)
        self.assertEqual(wp.calls[0][0], (1.0, 0.2, 0.0))
        self.assertEqual(wp.calls[1][0], (0.5, 0.0, 0.0))
        self.assertEqual(len(arm.calls), 0)
        self.assertEqual(client.reset_calls, ["vla_interactive"])

    def test_grasp_gated_by_numeric_input(self):
        client = _FakeClient(
            [
                _FakeDecision("grasp", arm_targets_base=((0.3, 0.0, 0.2, 0, 0, 0, 1.0),)),
                _FakeDecision("done", subtask="done"),
            ]
        )
        obs = _FakeObservation()
        wp = _FakeWaypointExecutor()
        arm = _FakeArmExecutor()
        controller = InteractiveVLAController(
            client=client,
            observation=obs,
            waypoint_executor=wp,
            arm_executor=arm,
            config=self._config(),
            display=lambda _: None,
            input_fn=lambda _: "1",
        )
        summary = controller.run()

        self.assertEqual(summary["inference_count"], 2)
        self.assertEqual(len(arm.calls), 1)
        self.assertEqual(arm.calls[0], (0.3, 0.0, 0.2, 0, 0, 0, 1.0))
        # grasp 后 phase 应进入 nav_place
        self.assertEqual(controller.phase, "nav_place")

    def test_grasp_skipped_when_operator_chooses_zero(self):
        client = _FakeClient(
            [
                _FakeDecision("grasp", arm_targets_base=((0.3, 0.0, 0.2, 0, 0, 0, 1.0),)),
                _FakeDecision("done", subtask="done"),
            ]
        )
        obs = _FakeObservation()
        wp = _FakeWaypointExecutor()
        arm = _FakeArmExecutor()
        controller = InteractiveVLAController(
            client=client,
            observation=obs,
            waypoint_executor=wp,
            arm_executor=arm,
            config=self._config(),
            display=lambda _: None,
            input_fn=lambda _: "0",
        )
        controller.run()

        self.assertEqual(len(arm.calls), 0)

    def test_waypoint_reached_check(self):
        self.assertTrue(
            waypoint_reached(
                (0.05, 0.03, 0.0),
                (0.0, 0.0, 0.0),
                tolerance_m=0.12,
                yaw_tolerance_rad=0.14,
            )
        )
        self.assertFalse(
            waypoint_reached(
                (0.5, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                tolerance_m=0.12,
                yaw_tolerance_rad=0.14,
            )
        )


if __name__ == "__main__":
    unittest.main()
