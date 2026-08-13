"""双箱任务的二维约束采样与禁止区域几何检查。"""

from __future__ import annotations

import math
import random
from typing import Any

from .forward_sector_randomization import (
    _finite_float,
    _positive_range,
    _vector,
)


BOX_PAIR_LAYOUT_MODE = "central_annulus_opposite_halfplane_v1"


def _polygon_xy(value: Any, *, field_name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"{field_name} 必须包含至少三个 XY 顶点")
    points = tuple(
        tuple(
            _vector(point, field_name=f"{field_name}[{index}]", length=2)
        )
        for index, point in enumerate(value)
    )
    if len(set(points)) < 3:
        raise ValueError(f"{field_name} 必须包含至少三个不同顶点")
    return points


def _layout_sampling_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """解析中央环形与禁止区约束；未配置时保留旧矩形偏移行为。"""

    raw = config.get("layout_sampling")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("box_pair.layout_sampling 必须是对象")
    mode = str(raw.get("mode") or "").strip()
    if mode != BOX_PAIR_LAYOUT_MODE:
        raise ValueError(
            "box_pair.layout_sampling.mode 不支持: "
            f"{mode!r}，期望 {BOX_PAIR_LAYOUT_MODE!r}"
        )
    center_xy = _vector(
        raw.get("field_center_xy"),
        field_name="box_pair.layout_sampling.field_center_xy",
        length=2,
    )
    box1_radius = _positive_range(
        raw.get("box1_radius_range_m"),
        field_name="box_pair.layout_sampling.box1_radius_range_m",
    )
    box2_radius = _positive_range(
        raw.get("box2_radius_range_m", box1_radius),
        field_name="box_pair.layout_sampling.box2_radius_range_m",
    )
    minimum_distance = _finite_float(
        raw.get("box2_min_distance_from_box1_m"),
        field_name=(
            "box_pair.layout_sampling.box2_min_distance_from_box1_m"
        ),
    )
    if minimum_distance <= 0.0:
        raise ValueError(
            "box_pair.layout_sampling.box2_min_distance_from_box1_m "
            "必须大于零"
        )
    footprint_margin = _finite_float(
        raw.get("table_forbidden_region_margin_m", 0.0),
        field_name=(
            "box_pair.layout_sampling.table_forbidden_region_margin_m"
        ),
    )
    if footprint_margin < 0.0:
        raise ValueError(
            "box_pair.layout_sampling.table_forbidden_region_margin_m "
            "必须非负"
        )
    raw_regions = raw.get("forbidden_regions_xy", [])
    if not isinstance(raw_regions, list):
        raise ValueError(
            "box_pair.layout_sampling.forbidden_regions_xy 必须是数组"
        )
    forbidden_regions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, region in enumerate(raw_regions):
        if not isinstance(region, dict):
            raise ValueError(
                "box_pair.layout_sampling.forbidden_regions_xy"
                f"[{index}] 必须是对象"
            )
        region_id = str(region.get("id") or "").strip()
        if not region_id or region_id in seen_ids:
            raise ValueError(
                "box_pair.layout_sampling.forbidden_regions_xy "
                "要求非空且唯一的 id"
            )
        seen_ids.add(region_id)
        forbidden_regions.append(
            {
                "id": region_id,
                "polygon_xy": _polygon_xy(
                    region.get("polygon_xy"),
                    field_name=(
                        "box_pair.layout_sampling.forbidden_regions_xy"
                        f"[{index}].polygon_xy"
                    ),
                ),
            }
        )
    return {
        "mode": mode,
        "field_center_xy": center_xy,
        "box1_radius_range_m": box1_radius,
        "box2_radius_range_m": box2_radius,
        "box2_min_distance_from_box1_m": minimum_distance,
        "table_forbidden_region_margin_m": footprint_margin,
        "forbidden_regions_xy": forbidden_regions,
    }


def _point_on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    cross = (point[0] - first[0]) * (second[1] - first[1]) - (
        point[1] - first[1]
    ) * (second[0] - first[0])
    if abs(cross) > 1.0e-9:
        return False
    return (
        min(first[0], second[0]) - 1.0e-9
        <= point[0]
        <= max(first[0], second[0]) + 1.0e-9
        and min(first[1], second[1]) - 1.0e-9
        <= point[1]
        <= max(first[1], second[1]) + 1.0e-9
    )


def _point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    """边界也视为区域内部，避免候选刚好压在禁止区边缘。"""

    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        if (current[1] > point[1]) != (previous[1] > point[1]):
            intersection_x = (
                (previous[0] - current[0])
                * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
            if point[0] < intersection_x:
                inside = not inside
        previous = current
    return inside


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _segments_intersect(
    first_a: tuple[float, float],
    first_b: tuple[float, float],
    second_a: tuple[float, float],
    second_b: tuple[float, float],
) -> bool:
    values = (
        _orientation(first_a, first_b, second_a),
        _orientation(first_a, first_b, second_b),
        _orientation(second_a, second_b, first_a),
        _orientation(second_a, second_b, first_b),
    )
    if values[0] * values[1] < 0.0 and values[2] * values[3] < 0.0:
        return True
    return any(
        abs(value) <= 1.0e-9 and _point_on_segment(point, segment_a, segment_b)
        for value, point, segment_a, segment_b in (
            (values[0], second_a, first_a, first_b),
            (values[1], second_b, first_a, first_b),
            (values[2], first_a, second_a, second_b),
            (values[3], first_b, second_a, second_b),
        )
    )


def _polygons_overlap(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
) -> bool:
    if any(_point_in_polygon(point, second) for point in first):
        return True
    if any(_point_in_polygon(point, first) for point in second):
        return True
    first_edges = tuple(zip(first, (*first[1:], first[0])))
    second_edges = tuple(zip(second, (*second[1:], second[0])))
    return any(
        _segments_intersect(first_a, first_b, second_a, second_b)
        for first_a, first_b in first_edges
        for second_a, second_b in second_edges
    )


def _table_at_center(
    table: dict[str, Any],
    center_xy: tuple[float, float],
) -> dict[str, Any]:
    nominal_center = table["support_center_xy"]
    offset_xy = (
        float(center_xy[0]) - nominal_center[0],
        float(center_xy[1]) - nominal_center[1],
    )
    nominal_root = table["root_translate_xyz"]
    root_xyz = (
        nominal_root[0] + offset_xy[0],
        nominal_root[1] + offset_xy[1],
        nominal_root[2],
    )
    dims = table["support_dims_xyz"]
    top_z = float(table["support_top_z"])
    return {
        **table,
        "offset_xy_m": list(offset_xy),
        "support_center_xy_sampled": list(center_xy),
        "root_translate_xyz_sampled": list(root_xyz),
        "support_bbox_min_xyz": [
            center_xy[0] - 0.5 * dims[0],
            center_xy[1] - 0.5 * dims[1],
            top_z - dims[2],
        ],
        "support_bbox_max_xyz": [
            center_xy[0] + 0.5 * dims[0],
            center_xy[1] + 0.5 * dims[1],
            top_z,
        ],
    }


def _sample_annulus_center(
    rng: random.Random,
    *,
    center_xy: tuple[float, float],
    radius_range_m: tuple[float, float],
    angle_range_rad: tuple[float, float],
) -> tuple[tuple[float, float], float, float]:
    """按面积均匀采样圆环，避免样本在内圈过密。"""

    radius = math.sqrt(
        rng.uniform(radius_range_m[0] ** 2, radius_range_m[1] ** 2)
    )
    angle = rng.uniform(*angle_range_rad)
    return (
        (
            center_xy[0] + radius * math.cos(angle),
            center_xy[1] + radius * math.sin(angle),
        ),
        radius,
        math.atan2(math.sin(angle), math.cos(angle)),
    )


def _table_footprint_polygon(
    table: dict[str, Any],
    *,
    margin_m: float,
) -> tuple[tuple[float, float], ...]:
    center = tuple(float(value) for value in table["support_center_xy_sampled"])
    dims = tuple(float(value) for value in table["support_dims_xyz"][:2])
    half_x = 0.5 * dims[0] + margin_m
    half_y = 0.5 * dims[1] + margin_m
    yaw = float(table["fixed_world_yaw_rad"])
    output: list[tuple[float, float]] = []
    for local_xy in (
        (-half_x, -half_y),
        (half_x, -half_y),
        (half_x, half_y),
        (-half_x, half_y),
    ):
        dx = local_xy[0] * math.cos(yaw) - local_xy[1] * math.sin(yaw)
        dy = local_xy[0] * math.sin(yaw) + local_xy[1] * math.cos(yaw)
        output.append((center[0] + dx, center[1] + dy))
    return tuple(output)


def _table_forbidden_region_report(
    table: dict[str, Any],
    *,
    layout_sampling: dict[str, Any],
) -> dict[str, Any]:
    footprint = _table_footprint_polygon(
        table,
        margin_m=float(layout_sampling["table_forbidden_region_margin_m"]),
    )
    overlaps = [
        str(region["id"])
        for region in layout_sampling["forbidden_regions_xy"]
        if _polygons_overlap(footprint, region["polygon_xy"])
    ]
    if overlaps:
        raise RuntimeError(
            f"{table['name']}_overlaps_forbidden_regions: {overlaps}"
        )
    return {
        "expanded_footprint_polygon_xy": [list(point) for point in footprint],
        "footprint_margin_m": float(
            layout_sampling["table_forbidden_region_margin_m"]
        ),
        "forbidden_region_ids": [
            str(region["id"])
            for region in layout_sampling["forbidden_regions_xy"]
        ],
        "overlaps": [],
        "geometry_verified": True,
    }



