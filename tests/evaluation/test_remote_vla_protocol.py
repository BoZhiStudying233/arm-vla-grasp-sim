from __future__ import annotations

import json

import pytest

from source.evaluation.client import RemotePolicyClient
from source.evaluation.protocol import (
    PROTOCOL_VERSION,
    PolicyDecision,
    RemotePolicyError,
)


def test_policy_decision_requires_waypoints_for_nav() -> None:
    with pytest.raises(RemotePolicyError, match="no nav_waypoints"):
        PolicyDecision.from_response({"route": "nav", "nav_waypoints": None})


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
