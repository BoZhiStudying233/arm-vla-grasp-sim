"""Synchronous, reconnecting WebSocket client used by the Isaac control loop."""

from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any

from .protocol import PROTOCOL_VERSION, PolicyDecision, RemotePolicyError


@dataclass(frozen=True)
class RemotePolicyClientConfig:
    endpoint: str = "ws://127.0.0.1:10093"
    connect_timeout_s: float = 10.0
    response_timeout_s: float = 120.0
    max_message_bytes: int = 20 * 1024 * 1024
    reconnect_attempts: int = 1

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(str(self.endpoint))
        if parsed.scheme != "ws" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise RemotePolicyError(
                "remote VLA endpoint must be a loopback ws:// URL reached through SSH"
            )
        if parsed.port is None:
            raise RemotePolicyError("remote VLA endpoint must include an explicit port")
        if self.connect_timeout_s <= 0.0 or self.response_timeout_s <= 0.0:
            raise RemotePolicyError("remote VLA timeouts must be positive")
        if self.reconnect_attempts < 0:
            raise RemotePolicyError("remote VLA reconnect_attempts must be non-negative")


class RemotePolicyClient:
    def __init__(self, config: RemotePolicyClientConfig | None = None) -> None:
        self.config = config or RemotePolicyClientConfig()
        self._connection = None

    def close(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        except Exception:
            pass
        finally:
            self._connection = None

    def health(self) -> dict[str, Any]:
        return self._request("health", {})

    def reset(self, episode_id: str) -> dict[str, Any]:
        return self._request("reset", {"episode_id": str(episode_id)})

    def infer(self, payload: dict[str, Any]) -> PolicyDecision:
        response = self._request("infer", payload)
        timing = response.get("remote_timing")
        timing = dict(timing) if isinstance(timing, dict) else {}
        roundtrip_ms = response.get("client_roundtrip_ms")
        if isinstance(roundtrip_ms, (int, float)):
            timing["client_roundtrip_ms"] = float(roundtrip_ms)
        return PolicyDecision.from_response(
            response,
            metadata={
                "remote_endpoint": self.config.endpoint,
                "remote_timing": timing,
            },
        )

    def _connect(self):
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RemotePolicyError(
                "websockets.sync is unavailable; install websockets>=12"
            ) from exc
        self._connection = connect(
            self.config.endpoint,
            open_timeout=self.config.connect_timeout_s,
            close_timeout=2.0,
            compression=None,
            max_size=self.config.max_message_bytes,
        )
        return self._connection

    def _request(self, request_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        message = {
            "protocol_version": PROTOCOL_VERSION,
            "type": request_type,
            "request_id": request_id,
            "payload": payload,
        }
        attempts = self.config.reconnect_attempts + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            started_at = time.perf_counter()
            try:
                connection = self._connection or self._connect()
                connection.send(json.dumps(message, ensure_ascii=True))
                raw_response = connection.recv(timeout=self.config.response_timeout_s)
                data = self._validate_response(raw_response, request_id)
                data["client_roundtrip_ms"] = round(
                    (time.perf_counter() - started_at) * 1000.0,
                    3,
                )
                return data
            except Exception as exc:
                last_error = exc
                self.close()
        raise RemotePolicyError(
            f"remote policy request failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _validate_response(raw_response: Any, request_id: str) -> dict[str, Any]:
        if not isinstance(raw_response, str):
            raise RemotePolicyError("remote policy returned a binary frame")
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise RemotePolicyError("remote policy returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RemotePolicyError("remote policy response must be an object")
        if response.get("protocol_version") != PROTOCOL_VERSION:
            raise RemotePolicyError("remote policy protocol version mismatch")
        if response.get("request_id") != request_id:
            raise RemotePolicyError("remote policy request_id mismatch")
        if response.get("ok") is not True:
            error = response.get("error") or {}
            raise RemotePolicyError(str(error.get("message") or "remote inference failed"))
        data = response.get("data")
        if not isinstance(data, dict):
            raise RemotePolicyError("remote policy response has no data object")
        timing = response.get("timing")
        if isinstance(timing, dict):
            data = {**data, "remote_timing": timing}
        return data
