"""Remote StarVLA evaluation adapters for the full-physics pipeline."""

from .client import RemotePolicyClient, RemotePolicyClientConfig
from .protocol import PROTOCOL_VERSION, PolicyDecision, RemotePolicyError

__all__ = [
    "PROTOCOL_VERSION",
    "PolicyDecision",
    "RemotePolicyClient",
    "RemotePolicyClientConfig",
    "RemotePolicyError",
]
