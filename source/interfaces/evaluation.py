"""Contracts for optional high-level VLA evaluation policies."""

from __future__ import annotations

from typing import Any, Protocol

from .simulation import SimulationState


class SemanticRoutePolicy(Protocol):
    """Predict a typed high-level route without applying robot actions."""

    def predict_route(self, state: SimulationState, *, phase: str) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...
