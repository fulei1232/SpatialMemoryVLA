"""State-level LIBERO interventions shared by data generation and evaluation.

These helpers mutate MuJoCo state, never policy memory tensors.  They are kept
independent of the rollout driver so the same intervention can be applied while
collecting demonstrations and while evaluating a checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class TemporalOcclusionConfig:
    partial_start_ratio: float = 0.30
    full_start_ratio: float = 0.50
    recovery_start_ratio: float = 0.75
    partial_max_fraction: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.partial_start_ratio < self.full_start_ratio < self.recovery_start_ratio <= 1.0:
            raise ValueError("Expected partial_start < full_start < recovery_start in [0, 1]")
        if not 0.0 < self.partial_max_fraction <= 1.0:
            raise ValueError("partial_max_fraction must be in (0, 1]")


def apply_temporal_occlusion(
    image: np.ndarray,
    timestep: int,
    horizon: int,
    cfg: TemporalOcclusionConfig,
) -> tuple[np.ndarray, str, float]:
    """Return a visible -> partial -> full -> recovered RGB observation."""
    progress = max(0.0, min(1.0, timestep / max(1, horizon - 1)))
    if progress < cfg.partial_start_ratio:
        return image, "visible_history", 0.0
    if progress >= cfg.recovery_start_ratio:
        return image, "recovered_visible", 0.0

    result = np.asarray(image, dtype=np.uint8).copy()
    if progress >= cfg.full_start_ratio:
        result[...] = 0
        return result, "full_occlusion", 1.0

    ramp = (progress - cfg.partial_start_ratio) / (cfg.full_start_ratio - cfg.partial_start_ratio)
    severity = max(1.0 / max(result.shape[:2]), cfg.partial_max_fraction * ramp)
    height, width = result.shape[:2]
    mask_h = max(1, int(round(height * np.sqrt(severity))))
    mask_w = max(1, int(round(width * np.sqrt(severity))))
    y0, x0 = (height - mask_h) // 2, (width - mask_w) // 2
    result[y0 : y0 + mask_h, x0 : x0 + mask_w] = 0
    return result, "partial_occlusion", float(severity)


def _core_env(env: Any) -> Any:
    return env.env if hasattr(env, "env") else env


def _refresh_observation(env: Any) -> dict[str, Any]:
    core = _core_env(env)
    core.sim.forward()
    if hasattr(core, "_post_process"):
        core._post_process()
    if hasattr(core, "_update_observables"):
        core._update_observables(force=True)
    return core._get_observations()


def get_object_free_joint(env: Any, object_name: str) -> str:
    core = _core_env(env)
    if object_name not in core.objects_dict:
        raise KeyError(f"Unknown movable object {object_name!r}; choices={sorted(core.objects_dict)}")
    joints = core.objects_dict[object_name].joints
    if not joints:
        raise ValueError(f"Object {object_name!r} has no movable joint")
    return joints[-1]


def get_object_pose(env: Any, object_name: str) -> np.ndarray:
    joint = get_object_free_joint(env, object_name)
    pose = np.asarray(_core_env(env).sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    if pose.shape != (7,):
        raise ValueError(f"Expected free-joint pose [xyz, quaternion] for {object_name}, got {pose.shape}")
    return pose


def set_object_pose(env: Any, object_name: str, pose: np.ndarray) -> dict[str, Any]:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid free-joint pose for {object_name}: {pose}")
    joint = get_object_free_joint(env, object_name)
    _core_env(env).sim.data.set_joint_qpos(joint, pose)
    return _refresh_observation(env)


@dataclass(frozen=True)
class RelocationConfig:
    object_name: str
    timestep: int
    translation: tuple[float, float, float]
    yaw_radians: float = 0.0
    workspace_x_bounds: tuple[float, float] | None = None
    workspace_y_bounds: tuple[float, float] | None = None


def relocate_object(env: Any, cfg: RelocationConfig) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    old_pose = get_object_pose(env, cfg.object_name)
    new_pose = old_pose.copy()
    new_pose[:3] += np.asarray(cfg.translation, dtype=np.float64)
    if cfg.workspace_x_bounds is not None and not (
        cfg.workspace_x_bounds[0] <= new_pose[0] <= cfg.workspace_x_bounds[1]
    ):
        raise ValueError(
            f"Relocation x={new_pose[0]:.4f} is outside workspace bounds {cfg.workspace_x_bounds}"
        )
    if cfg.workspace_y_bounds is not None and not (
        cfg.workspace_y_bounds[0] <= new_pose[1] <= cfg.workspace_y_bounds[1]
    ):
        raise ValueError(
            f"Relocation y={new_pose[1]:.4f} is outside workspace bounds {cfg.workspace_y_bounds}"
        )
    if cfg.yaw_radians:
        half = cfg.yaw_radians / 2.0
        yaw_quaternion = np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)])
        w1, x1, y1, z1 = yaw_quaternion
        w2, x2, y2, z2 = new_pose[3:]
        new_pose[3:] = [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    observation = set_object_pose(env, cfg.object_name, new_pose)
    return observation, old_pose, new_pose


def get_object_contacts(env: Any, object_name: str) -> list[tuple[str, str]]:
    """Return current contact pairs involving an object's collision geoms."""
    core = _core_env(env)
    if object_name not in core.objects_dict:
        raise KeyError(f"Unknown object {object_name!r}")
    target_geoms = set(core.objects_dict[object_name].contact_geoms)
    contacts: list[tuple[str, str]] = []
    for index in range(core.sim.data.ncon):
        contact = core.sim.data.contact[index]
        first = core.sim.model.geom_id2name(contact.geom1) or f"geom-{contact.geom1}"
        second = core.sim.model.geom_id2name(contact.geom2) or f"geom-{contact.geom2}"
        if first in target_geoms or second in target_geoms:
            contacts.append((first, second))
    return contacts


def unexpected_object_contacts(contacts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Exclude support-surface contacts while retaining robot/object collisions."""
    support_markers = ("table", "floor")
    return [
        pair for pair in contacts
        if not any(marker in geom.lower() for geom in pair for marker in support_markers)
    ]


@dataclass(frozen=True)
class CameraShiftConfig:
    timestep: int
    camera_name: str = "agentview"
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_radians: float = 0.7


def shift_camera_out_of_view(
    env: Any, cfg: CameraShiftConfig
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    sim = _core_env(env).sim
    camera_id = sim.model.camera_name2id(cfg.camera_name)
    old_position = sim.model.cam_pos[camera_id].copy()
    old_quaternion = sim.model.cam_quat[camera_id].copy()
    sim.model.cam_pos[camera_id] = old_position + np.asarray(cfg.translation)
    half = cfg.yaw_radians / 2.0
    rotation = np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)])
    w1, x1, y1, z1 = rotation
    w2, x2, y2, z2 = old_quaternion
    sim.model.cam_quat[camera_id] = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
    return _refresh_observation(env), (old_position, old_quaternion)


def restore_camera(
    env: Any, camera_name: str, state: tuple[np.ndarray, np.ndarray]
) -> dict[str, Any]:
    sim = _core_env(env).sim
    camera_id = sim.model.camera_name2id(camera_name)
    sim.model.cam_pos[camera_id], sim.model.cam_quat[camera_id] = state
    return _refresh_observation(env)


RelationLabel = Literal["left", "right", "front", "behind"]


def set_counterfactual_relation(
    env: Any,
    moving_object: str,
    anchor_object: str,
    relation: RelationLabel,
    distance_m: float = 0.10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    moving_pose = get_object_pose(env, moving_object)
    anchor_pose = get_object_pose(env, anchor_object)
    offsets = {
        "left": np.asarray([0.0, distance_m, 0.0]),
        "right": np.asarray([0.0, -distance_m, 0.0]),
        "front": np.asarray([-distance_m, 0.0, 0.0]),
        "behind": np.asarray([distance_m, 0.0, 0.0]),
    }
    moving_pose[:3] = anchor_pose[:3] + offsets[relation]
    observation = set_object_pose(env, moving_object, moving_pose)
    metadata = {
        "relation_type": "left_right" if relation in {"left", "right"} else "front_behind",
        "relation_label": relation,
        "moving_object": moving_object,
        "anchor_object": anchor_object,
        "distance_m": distance_m,
    }
    return observation, metadata


def intervention_metadata(config: RelocationConfig | CameraShiftConfig) -> dict[str, Any]:
    return {"intervention": type(config).__name__, **asdict(config)}
