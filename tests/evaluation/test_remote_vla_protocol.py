from __future__ import annotations

import json
from threading import Thread

import pytest

from source.evaluation.client import RemotePolicyClient
from source.evaluation.client import RemotePolicyClientConfig
from source.evaluation.protocol import (
    PROTOCOL_VERSION,
    PolicyDecision,
    RemotePolicyError,
)


def test_policy_decision_requires_waypoints_for_nav() -> None:
    with pytest.raises(RemotePolicyError, match="no nav_waypoints"):
        PolicyDecision.from_response({"route": "nav", "nav_waypoints": None})


def test_policy_decision_accepts_base_frame_arm_targets() -> None:
    decision = PolicyDecision.from_response(
        {
            "route": "grasp",
            "arm_targets_base": [[0.35, 0.0, 0.2, 0.0, 0.1, 0.0, 1.0]],
        }
    )
    assert decision.arm_targets_base == ((0.35, 0.0, 0.2, 0.0, 0.1, 0.0, 1.0),)


def test_client_rejects_stale_request_id() -> None:
    raw = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "type": "health_result",
            "request_id": "old",
            "ok": True,
            "data": {},
        }
    )
    with pytest.raises(RemotePolicyError, match="request_id mismatch"):
        RemotePolicyClient._validate_response(raw, "new")


def test_client_config_rejects_non_loopback_endpoint() -> None:
    with pytest.raises(RemotePolicyError, match="loopback"):
        RemotePolicyClientConfig(endpoint="ws://10.0.0.4:10093")


def test_loopback_websocket_health_reset_and_infer_roundtrip() -> None:
    from websockets.sync.server import serve

    seen_types = []

    def handler(connection):
        request = json.loads(connection.recv())
        seen_types.append(request["type"])
        data = {
            "health": {"backend": "mock"},
            "reset": {"reset": True},
            "infer": {
                "route": "nav",
                "subtask": "Move toward box1.",
                "nav_waypoints": [[0.2, 0.0, 0.0]],
            },
        }[request["type"]]
        connection.send(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "type": f"{request['type']}_result",
                    "request_id": request["request_id"],
                    "ok": True,
                    "data": data,
                    "timing": {"server_total_ms": 1.0},
                }
            )
        )

    with serve(handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RemotePolicyClient(
                RemotePolicyClientConfig(endpoint=f"ws://127.0.0.1:{port}")
            )
            assert client.health()["backend"] == "mock"
            assert client.reset("episode-1")["reset"] is True
            decision = client.infer(
                {
                    "instruction": "test",
                    "images": {"front": {"encoding": "jpeg_base64", "data": "AA=="}},
                }
            )
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=2.0)
    assert seen_types == ["health", "reset", "infer"]
    assert decision.nav_waypoints == ((0.2, 0.0, 0.0),)
    assert decision.metadata["remote_timing"]["server_total_ms"] == 1.0
    assert decision.metadata["remote_timing"]["client_roundtrip_ms"] >= 0.0
