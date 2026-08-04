"""Interactive VLA evaluation controller shared by sim and real clients.

The controller owns the inference cadence and the human gate:
- NAV: execute only the first predicted waypoint, then request the next inference.
- GRASP/PLACE: print the decision and block on a numeric terminal prompt
  (1 = execute, 0 = skip). No new inference is issued while blocked.
- DONE/RECOVER: stop.

This module has no Isaac or ROS imports so it can be unit-tested and reused
by both ``pct_scene`` and ``gx-real`` entry points.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


ARM_ROUTES = frozenset({"grasp", "place"})
TERMINAL_ROUTES = frozenset({"done", "recover"})


@dataclass(frozen=True)
class InteractiveVLAConfig:
    instruction: str
    episode_id: str = "vla_interactive"
    endpoint: str = "ws://127.0.0.1:10093"
    connect_timeout_s: float = 10.0
    response_timeout_s: float = 120.0
    nav_waypoint_tolerance_m: float = 0.12
    nav_yaw_tolerance_rad: float = 0.14
    nav_max_steps_per_waypoint: int = 500
    nav_max_replans: int = 64
    arm_confirm_prompt: str = "是否执行 {route}？输入 1=执行 0=跳过: "
    arm_confirm_timeout_s: float | None = None
    show_raw_text: bool = True
    show_timing: bool = True
    jsonl_out: str | None = None

    def __post_init__(self) -> None:
        if not str(self.instruction).strip():
            raise ValueError("instruction must be non-empty")
        if self.nav_max_steps_per_waypoint <= 0:
            raise ValueError("nav_max_steps_per_waypoint must be positive")
        if self.nav_max_replans <= 0:
            raise ValueError("nav_max_replans must be positive")
        if self.arm_confirm_timeout_s is not None and self.arm_confirm_timeout_s <= 0:
            raise ValueError("arm_confirm_timeout_s must be positive or null")


@dataclass
class InferenceRecord:
    frame_index: int
    phase: str
    route: str
    subtask: str | None
    nav_waypoints: tuple[tuple[float, float, float], ...] = ()
    arm_targets_base: tuple[
        tuple[float, float, float, float, float, float, float], ...
    ] = ()
    raw_text: str = ""
    timing_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "phase": self.phase,
            "route": self.route,
            "subtask": self.subtask,
            "nav_waypoints": [list(p) for p in self.nav_waypoints],
            "arm_targets_base": [list(t) for t in self.arm_targets_base],
            "raw_text": self.raw_text,
            "timing_ms": self.timing_ms,
            **self.extra,
        }


class RemoteClient(Protocol):
    def health(self) -> dict[str, Any]: ...

    def reset(self, episode_id: str) -> dict[str, Any]: ...

    def infer(self, payload: dict[str, Any]) -> Any: ...


class ObservationProvider(Protocol):
    def build_payload(self, *, phase: str, frame_index: int) -> dict[str, Any]: ...


class WaypointExecutor(Protocol):
    def execute_waypoint(
        self,
        waypoint: tuple[float, float, float],
        *,
        tolerance_m: float,
        yaw_tolerance_rad: float,
        max_steps: int,
    ) -> dict[str, Any]:
        """Execute one body-frame waypoint until reached; return execution report."""


class ArmExecutor(Protocol):
    def execute_arm_target(self, target: tuple[float, ...]) -> dict[str, Any]:
        """Execute one base-frame TCP target; return execution report."""


class InteractiveVLAController:
    """Human-gated inference loop with first-waypoint NAV execution."""

    def __init__(
        self,
        *,
        client: RemoteClient,
        observation: ObservationProvider,
        waypoint_executor: WaypointExecutor,
        arm_executor: ArmExecutor,
        config: InteractiveVLAConfig,
        display: Callable[[str], None] = print,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.client = client
        self.observation = observation
        self.waypoint_executor = waypoint_executor
        self.arm_executor = arm_executor
        self.config = config
        self.display = display
        self.input_fn = input_fn
        self.phase = "nav_pick"
        self.frame_index = 0
        self.inference_count = 0
        self.replan_count = 0
        self.records: list[InferenceRecord] = []
        self.log_path = Path(config.jsonl_out) if config.jsonl_out else None

    def run(self) -> dict[str, Any]:
        self._append_jsonl(
            {
                "event": "start",
                "episode_id": self.config.episode_id,
                "endpoint": self.config.endpoint,
            }
        )
        health = self.client.health()
        self.display(f"[vla] server health: {health}")
        self.client.reset(self.config.episode_id)

        while True:
            decision = self._infer_once()
            if decision is None:
                break
            if decision.route == "nav":
                self._execute_nav(decision)
                continue
            if decision.route in ARM_ROUTES:
                self._gate_arm(decision)
                self.phase = _next_phase(self.phase, decision.route)
                continue
            if decision.route in TERMINAL_ROUTES:
                self.display(
                    f"[vla] terminal route={decision.route} subtask={decision.subtask}"
                )
                self._append_jsonl(
                    {
                        "event": "terminal",
                        "route": decision.route,
                        "subtask": decision.subtask,
                    }
                )
                break
            self.display(f"[vla] unknown route={decision.route}; stopping")
            break

        summary = {
            "episode_id": self.config.episode_id,
            "inference_count": self.inference_count,
            "replan_count": self.replan_count,
            "records": [record.to_dict() for record in self.records],
        }
        self._append_jsonl({"event": "summary", **summary})
        return summary

    def _infer_once(self) -> InferenceRecord | None:
        self.frame_index += 1
        payload = self.observation.build_payload(
            phase=self.phase,
            frame_index=self.frame_index,
        )
        started = time.perf_counter()
        try:
            decision = self.client.infer(payload)
        except Exception as exc:
            self.display(f"[vla] inference failed: {exc}")
            self._append_jsonl({"event": "infer_error", "error": str(exc)})
            return None
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        self.inference_count += 1

        waypoints = getattr(decision, "nav_waypoints", ()) or ()
        arm_targets = getattr(decision, "arm_targets_base", ()) or ()
        record = InferenceRecord(
            frame_index=self.frame_index,
            phase=self.phase,
            route=str(getattr(decision, "route", "")),
            subtask=getattr(decision, "subtask", None),
            nav_waypoints=tuple(tuple(float(v) for v in p) for p in waypoints),
            arm_targets_base=tuple(tuple(float(v) for v in t) for t in arm_targets),
            raw_text=str(getattr(decision, "raw_text", "")),
            timing_ms=elapsed_ms,
        )
        self.records.append(record)
        self._print_decision(record)
        self._append_jsonl({"event": "decision", **record.to_dict()})
        return record

    def _execute_nav(self, record: InferenceRecord) -> None:
        if not record.nav_waypoints:
            self.display("[vla] NAV decision has no waypoint; stopping")
            return
        if self.replan_count >= self.config.nav_max_replans:
            self.display("[vla] NAV replan limit reached; stopping")
            return
        self.replan_count += 1
        self.display(
            f"[vla] NAV waypoint #{self.replan_count}: "
            f"{[round(v, 3) for v in record.nav_waypoints[0]]}"
        )
        report = self.waypoint_executor.execute_waypoint(
            record.nav_waypoints[0],
            tolerance_m=self.config.nav_waypoint_tolerance_m,
            yaw_tolerance_rad=self.config.nav_yaw_tolerance_rad,
            max_steps=self.config.nav_max_steps_per_waypoint,
        )
        self._append_jsonl(
            {
                "event": "nav_exec",
                "frame_index": record.frame_index,
                "replan": self.replan_count,
                "report": report,
            }
        )
        if not report.get("reached"):
            self.display(
                f"[vla] NAV waypoint not reached: {report.get('reason', 'timeout')}"
            )

    def _gate_arm(self, record: InferenceRecord) -> None:
        if not record.arm_targets_base:
            self.display(f"[vla] {record.route} decision has no arm target; skipping")
            return
        prompt = self.config.arm_confirm_prompt.format(route=record.route)
        self.display(
            f"[vla] {record.route} subtask={record.subtask} "
            f"target={[round(v, 3) for v in record.arm_targets_base[0]]}"
        )
        choice = self._ask_number(prompt)
        if choice != 1:
            self.display(f"[vla] {record.route} skipped by operator")
            self._append_jsonl(
                {
                    "event": "arm_gate",
                    "frame_index": record.frame_index,
                    "route": record.route,
                    "choice": choice,
                }
            )
            return
        report = self.arm_executor.execute_arm_target(record.arm_targets_base[0])
        self.display(f"[vla] {record.route} executed: {report}")
        self._append_jsonl(
            {
                "event": "arm_exec",
                "frame_index": record.frame_index,
                "route": record.route,
                "choice": 1,
                "report": report,
            }
        )

    def _ask_number(self, prompt: str) -> int:
        while True:
            try:
                raw = self.input_fn(prompt)
                value = int(str(raw).strip())
                if value in (0, 1):
                    return value
                self.display("[vla] 请输入 0 或 1")
            except (EOFError, KeyboardInterrupt):
                self.display("[vla] 输入中断，按跳过处理")
                return 0
            except ValueError:
                self.display("[vla] 输入无效，请输入 0 或 1")

    def _print_decision(self, record: InferenceRecord) -> None:
        line = (
            f"[vla] #{record.frame_index} phase={record.phase} "
            f"route={record.route} subtask={record.subtask}"
        )
        if record.nav_waypoints:
            line += f" waypoints={[list(round(v, 3) for v in p) for p in record.nav_waypoints]}"
        if record.arm_targets_base:
            line += f" arm_targets={[list(round(v, 3) for v in t) for t in record.arm_targets_base]}"
        if self.config.show_raw_text and record.raw_text:
            line += f" raw={record.raw_text!r}"
        if self.config.show_timing and record.timing_ms is not None:
            line += f" {record.timing_ms:.0f}ms"
        self.display(line)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def _next_phase(current: str, route: str) -> str:
    if route == "grasp":
        return "nav_place"
    if route == "place":
        return "place"
    return current


def waypoint_reached(
    waypoint: tuple[float, float, float],
    current_xyyaw: tuple[float, float, float],
    *,
    tolerance_m: float,
    yaw_tolerance_rad: float,
) -> bool:
    """Body-frame waypoint reached check (reused by sim and real executors)."""
    dx = waypoint[0] - 0.0
    dy = waypoint[1] - 0.0
    distance = math.hypot(dx, dy)
    dyaw = abs(_wrap_pi(float(waypoint[2])))
    return distance <= tolerance_m and dyaw <= yaw_tolerance_rad


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
