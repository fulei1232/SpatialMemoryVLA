"""Safety-first adapter from a 7-D MemoryVLA action to a RealMan RM65.

This module deliberately does not connect to a robot during import and it
never enables motion by default.  Hardware motion needs both configured
workspace bounds and an explicit ``allow_motion=True`` at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import math
import sys


@dataclass(frozen=True)
class RM65SafetyConfig:
    """Limits for one policy-control step.

    The action convention is ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``:
    translation is in metres, rotation is in radians, and the gripper scalar
    is positive for open and negative for close.  Bounds are Cartesian target
    bounds in the controller's configured work frame, in metres.
    """

    max_translation_m: float = 0.02
    max_rotation_rad: float = math.radians(10.0)
    velocity_percent: int = 10
    blend_percent: int = 0
    frame_mode: int = 0  # 0 = work frame, 1 = tool frame in RM_API2
    gripper_speed: int = 200
    gripper_force: int = 200
    gripper_timeout_s: int = 5
    workspace_bounds_m: Optional[tuple[tuple[float, float], ...]] = None

    def __post_init__(self) -> None:
        if self.max_translation_m <= 0 or self.max_rotation_rad <= 0:
            raise ValueError("Per-step translation and rotation limits must be positive.")
        if not 1 <= self.velocity_percent <= 100:
            raise ValueError("velocity_percent must be in [1, 100].")
        if not 0 <= self.blend_percent <= 100:
            raise ValueError("blend_percent must be in [0, 100].")
        if self.frame_mode not in (0, 1):
            raise ValueError("frame_mode must be 0 (work) or 1 (tool).")
        if not 1 <= self.gripper_speed <= 1000:
            raise ValueError("gripper_speed must be in [1, 1000].")
        if not 1 <= self.gripper_force <= 1000:
            raise ValueError("gripper_force must be in [1, 1000].")
        if self.gripper_timeout_s <= 0:
            raise ValueError("gripper_timeout_s must be positive.")
        if self.workspace_bounds_m is not None:
            if len(self.workspace_bounds_m) != 3:
                raise ValueError("workspace_bounds_m must contain (min, max) for x, y, z.")
            if any(len(axis) != 2 or axis[0] >= axis[1] for axis in self.workspace_bounds_m):
                raise ValueError("Every workspace axis needs an ordered (min, max) pair.")


class RM65Client:
    """Thin RM_API2 client with validation before every physical command.

    ``connect`` is intentionally separate from construction.  It may be used
    to read status, but executing a motion further requires configured bounds
    and explicit per-call consent.
    """

    def __init__(self, safety: RM65SafetyConfig = RM65SafetyConfig()) -> None:
        self.safety = safety
        self._robot: Any = None

    @staticmethod
    def _default_sdk_root() -> Path:
        return Path(__file__).resolve().parents[1] / "third_libs" / "RM_API2" / "Python"

    def connect(self, ip: str, port: int = 8080, sdk_root: Optional[Path] = None) -> None:
        """Create an RM_API2 connection.  This does not issue a motion command."""
        if not ip:
            raise ValueError("A controller IP address is required.")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in [1, 65535].")
        root = (sdk_root or self._default_sdk_root()).resolve()
        if not (root / "Robotic_Arm").is_dir():
            raise FileNotFoundError(
                f"RM_API2 Python SDK was not found at {root}. "
                "Clone RealManRobot/RM_API2 under third_libs or pass sdk_root."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from Robotic_Arm.rm_ctypes_wrap import rm_thread_mode_e
        from Robotic_Arm.rm_robot_interface import RoboticArm

        robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = robot.rm_create_robot_arm(ip, port)
        if getattr(handle, "id", -1) == -1:
            robot.rm_delete_robot_arm()
            raise ConnectionError(f"Could not connect to RM65 controller at {ip}:{port}.")
        self._robot = robot

    def close(self) -> None:
        """Close the current SDK connection, if one exists."""
        if self._robot is not None:
            self._robot.rm_delete_robot_arm()
            self._robot = None

    def read_state(self) -> dict[str, Any]:
        """Read the SDK's current arm state; this is a non-motion operation."""
        robot = self._require_connection()
        status, state = robot.rm_get_current_arm_state()
        if status != 0:
            raise RuntimeError(f"RM_API2 state read failed with status {status}.")
        return state

    def validate_action(self, action: Sequence[float]) -> tuple[float, ...]:
        """Validate and normalize a 7-D policy action without contacting hardware."""
        if len(action) != 7:
            raise ValueError("Expected exactly 7 action values: dx dy dz dRx dRy dRz gripper.")
        values = tuple(float(item) for item in action)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("Action values must be finite.")
        if any(abs(item) > self.safety.max_translation_m for item in values[:3]):
            raise ValueError(f"Translation exceeds {self.safety.max_translation_m} m per policy step.")
        if any(abs(item) > self.safety.max_rotation_rad for item in values[3:6]):
            raise ValueError(
                f"Rotation exceeds {self.safety.max_rotation_rad:.4f} rad per policy step."
            )
        return values

    def execute_action(self, action: Sequence[float], *, allow_motion: bool = False) -> dict[str, Any]:
        """Execute one validated action only after explicit physical-motion consent.

        The controller pose returned by RM_API2 uses metres/radians.  RM_API2's
        ``rm_algo_pose_move`` uniquely expects delta rotations in degrees, so
        this function performs that conversion before issuing ``rm_movel``.
        """
        if not allow_motion:
            raise PermissionError("Motion is disabled; call with allow_motion=True only after a safety check.")
        if self.safety.workspace_bounds_m is None:
            raise RuntimeError("Set calibrated workspace_bounds_m before enabling RM65 motion.")
        values = self.validate_action(action)
        robot = self._require_connection()
        state = self.read_state()
        current_pose = state.get("pose")
        if not isinstance(current_pose, (list, tuple)) or len(current_pose) != 6:
            raise RuntimeError(f"Unexpected RM_API2 pose format: {current_pose!r}")

        delta_for_sdk = [*values[:3], *(math.degrees(item) for item in values[3:6])]
        target_pose = robot.rm_algo_pose_move(list(current_pose), delta_for_sdk, self.safety.frame_mode)
        self._validate_target(target_pose)
        move_status = robot.rm_movel(
            target_pose,
            self.safety.velocity_percent,
            self.safety.blend_percent,
            0,
            1,
        )
        if move_status != 0:
            raise RuntimeError(f"RM_API2 rm_movel failed with status {move_status}.")

        if values[6] >= 0:
            gripper_status = robot.rm_set_gripper_release(
                self.safety.gripper_speed, True, self.safety.gripper_timeout_s
            )
        else:
            gripper_status = robot.rm_set_gripper_pick(
                self.safety.gripper_speed,
                self.safety.gripper_force,
                True,
                self.safety.gripper_timeout_s,
            )
        if gripper_status != 0:
            raise RuntimeError(f"RM_API2 gripper command failed with status {gripper_status}.")
        return {"target_pose": target_pose, "move_status": move_status, "gripper_status": gripper_status}

    def _require_connection(self) -> Any:
        if self._robot is None:
            raise RuntimeError("Not connected. Call connect(ip) before requesting robot state or motion.")
        return self._robot

    def _validate_target(self, pose: Sequence[float]) -> None:
        if len(pose) != 6 or not all(math.isfinite(float(item)) for item in pose):
            raise RuntimeError(f"RM_API2 returned an invalid target pose: {pose!r}")
        assert self.safety.workspace_bounds_m is not None
        for axis, (minimum, maximum) in zip(pose[:3], self.safety.workspace_bounds_m):
            if not minimum <= float(axis) <= maximum:
                raise RuntimeError(
                    f"Target position {list(pose[:3])} is outside calibrated workspace bounds "
                    f"{self.safety.workspace_bounds_m}."
                )
