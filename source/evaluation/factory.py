"""Compose the remote model adapters with the existing full-physics pipeline."""

from __future__ import annotations

from pathlib import Path

from source.interfaces import EpisodeSpec
from source.pipeline.config import FullPhysicsConfig
from source.pipeline.factory import create_full_physics_pipeline
from source.simulation import IsaacLabNavigationRuntime

from .adapters import (
    RecedingHorizonNavExecutor,
    RemotePolicySession,
    RemoteVLANavPlanner,
    WaypointSafetyConfig,
)
from .client import RemotePolicyClient, RemotePolicyClientConfig
from .observation import ObservationEncoder


def create_remote_vla_evaluation_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacLabNavigationRuntime,
    endpoint: str,
    connect_timeout_s: float,
    response_timeout_s: float,
    jpeg_quality: int,
    max_replans_per_navigation: int,
    close_simulation_on_exit: bool = True,
):
    client = RemotePolicyClient(
        RemotePolicyClientConfig(
            endpoint=endpoint,
            connect_timeout_s=connect_timeout_s,
            response_timeout_s=response_timeout_s,
        )
    )
    session = RemotePolicySession(
        client=client,
        encoder=ObservationEncoder(jpeg_quality=jpeg_quality),
        episode_spec=episode_spec,
    )
    try:
        session.start()
    except Exception:
        session.close()
        raise

    def transform_navigation_stack(_base_planner, base_executor):
        planner = RemoteVLANavPlanner(
            session,
            episode_spec,
            safety=WaypointSafetyConfig(
                max_replans_per_navigation=max_replans_per_navigation,
            ),
        )
        return planner, RecedingHorizonNavExecutor(planner, base_executor)

    try:
        return create_full_physics_pipeline(
            config=config,
            episode_spec=episode_spec,
            episode_seed=episode_seed,
            episode_dir=episode_dir,
            simulation=simulation,
            close_simulation_on_exit=close_simulation_on_exit,
            navigation_stack_transform=transform_navigation_stack,
            semantic_route_policy=session,
        )
    except Exception:
        session.close()
        raise
