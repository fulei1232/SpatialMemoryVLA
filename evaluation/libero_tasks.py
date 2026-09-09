"""Stable LIBERO task selection without conflating two index namespaces."""

from __future__ import annotations

SMOKE_TASK_NAME = "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
SPATIAL_VALIDATION_TASK_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
LIBERO10_SPATIAL_MEMORY_TASK_NAME = (
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
)


def resolve_benchmark_task_index(task_name: str, suite_name: str = "libero_spatial") -> int:
    """Return the installed LIBERO API's 0-based index for ``task_name``."""
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    try:
        return suite.get_task_names().index(task_name)
    except ValueError as exc:
        raise ValueError(f"Task {task_name!r} is not present in suite {suite_name!r}") from exc
