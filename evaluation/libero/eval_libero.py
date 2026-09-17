import os
import json
import csv
from dataclasses import dataclass
from typing import List, Union
import draccus
import numpy as np
import tqdm

os.environ.setdefault("MUJOCO_GL", "osmesa")
# apt install -y libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev

from libero.libero import benchmark

from libero_utils import (
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from robot_utils import DATE_TIME, set_seed_everywhere
from hard_cases import (
    CameraShiftConfig,
    RelocationConfig,
    TemporalOcclusionConfig,
    apply_temporal_occlusion,
    relocate_object,
    set_counterfactual_relation,
    shift_camera_out_of_view,
)

# let tensorflow only see CPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


@dataclass
class GenerateConfig:
    # fmt: off
    task_suite_name: str = "libero_spatial" # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10 # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50 # Number of rollouts per task
    repeat_initial_states: bool = False # Allow cycling official states when trials exceed the available count
    spcial_task_id: Union[List[int], int, None] = None # List of task IDs to evaluate (default: None, evaluates all tasks in the suite)
    run_id_note: str = ""
    local_log_dir: str = "./logs/eval_libero" # Local directory for eval logs
    seed: int = 7 # Random Seed (for reproducibility)
    resolution: Union[int, tuple] = 256 # Image resolution for model input
    port: int = 6800
    model: str = ""
    episode_offset: int = 0
    episode_stride: int = 1
    hard_case: str = "normal" # normal | relocation | out_of_view | counterfactual
    intervention_timestep: int = 80
    target_object: str = ""
    anchor_object: str = ""
    relocation_translation: tuple = (0.10, 0.0, 0.0)
    camera_name: str = "agentview"
    camera_translation: tuple = (0.0, 0.0, 0.0)
    camera_yaw_radians: float = 0.7
    relation_label: str = "left"
    relation_distance_m: float = 0.10
    occlusion_partial_start_ratio: float = 0.30
    occlusion_full_start_ratio: float = 0.50
    occlusion_recovery_start_ratio: float = 0.75
    occlusion_partial_max_fraction: float = 0.80
    occlusion_schedule_steps: int = 16 # policy observations; aligned with memory capacity
    save_rollout_videos: bool = True
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    if cfg.hard_case not in {"normal", "temporal_occlusion", "relocation", "out_of_view", "counterfactual"}:
        raise ValueError(f"Unsupported hard_case={cfg.hard_case!r}")
    if cfg.spcial_task_id is not None and isinstance(cfg.spcial_task_id, int):
        cfg.spcial_task_id = [cfg.spcial_task_id]

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize local logging
    run_id = f"{cfg.task_suite_name}-{cfg.num_trials_per_task}trials-seed{cfg.seed}-{cfg.run_id_note}-{DATE_TIME}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    trace_filepath = os.path.join(cfg.local_log_dir, "failure_trace.jsonl")
    trace_file = open(trace_filepath, "a")
    rollout_jsonl_path = os.path.join(cfg.local_log_dir, "rollouts.jsonl")
    rollout_csv_path = os.path.join(cfg.local_log_dir, "rollouts.csv")
    rollout_fields = [
        "model",
        "task_name",
        "runtime_task_index",
        "initial_state_id",
        "hard_case",
        "success",
        "policy_calls",
        "timeout",
        "success_phase",
        "reached_full_occlusion",
        "reached_recovery",
        "recovery_success",
    ]
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = cfg.resolution # TODO need to be done in the cfg

    ################################################################
    ### import Policy
    from vla_policy import LLaVAClient
    policy = LLaVAClient(base_url=f'http://localhost:{cfg.port}')
    ################################################################

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Skip tasks if specified
        if cfg.spcial_task_id is not None and task_id not in cfg.spcial_task_id:
            print(f"Skipping task {task_id}...")
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        if cfg.num_trials_per_task > len(initial_states) and not cfg.repeat_initial_states:
            raise ValueError(
                f"Requested {cfg.num_trials_per_task} trials but task {task_id} has only "
                f"{len(initial_states)} official initial states. Set --repeat_initial_states True to cycle them."
            )

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for local_episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            episode_idx = cfg.episode_offset + local_episode_idx * cfg.episode_stride
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            policy.reset()
            episode_first_frame = 'True'

            # Set initial states
            initial_state_idx = episode_idx % len(initial_states)
            log_file.write(f"Official initial state index: {initial_state_idx}\n")
            obs = env.set_init_state(initial_states[initial_state_idx])
            intervention_record = None
            intervention_applied = False
            occlusion_phase_counts = {
                "visible_history": 0,
                "partial_occlusion": 0,
                "full_occlusion": 0,
                "recovered_visible": 0,
            }
            temporal_occlusion_cfg = TemporalOcclusionConfig(
                partial_start_ratio=cfg.occlusion_partial_start_ratio,
                full_start_ratio=cfg.occlusion_full_start_ratio,
                recovery_start_ratio=cfg.occlusion_recovery_start_ratio,
                partial_max_fraction=cfg.occlusion_partial_max_fraction,
            )
            if cfg.hard_case == "temporal_occlusion":
                intervention_record = {
                    "type": "temporal_occlusion",
                    "partial_start_ratio": cfg.occlusion_partial_start_ratio,
                    "full_start_ratio": cfg.occlusion_full_start_ratio,
                    "recovery_start_ratio": cfg.occlusion_recovery_start_ratio,
                    "partial_max_fraction": cfg.occlusion_partial_max_fraction,
                    "schedule_steps": cfg.occlusion_schedule_steps,
                }
                intervention_applied = True
            if cfg.hard_case == "counterfactual":
                if not cfg.target_object or not cfg.anchor_object:
                    raise ValueError("counterfactual evaluation requires target_object and anchor_object")
                obs, intervention_record = set_counterfactual_relation(
                    env,
                    cfg.target_object,
                    cfg.anchor_object,
                    cfg.relation_label,
                    cfg.relation_distance_m,
                )
                intervention_applied = True

            # Setup
            t = 0
            done = False
            exception_text = None
            policy_calls = 0
            replay_images = []
            current_occlusion_phase = "normal"
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1} (global index {episode_idx})...")
            log_file.write(f"Starting episode {task_episodes+1} (global index {episode_idx})...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                        t += 1
                        continue

                    if not intervention_applied and t >= cfg.intervention_timestep:
                        if cfg.hard_case == "relocation":
                            if not cfg.target_object:
                                raise ValueError("relocation evaluation requires target_object")
                            obs, old_pose, new_pose = relocate_object(
                                env,
                                RelocationConfig(
                                    object_name=cfg.target_object,
                                    timestep=t,
                                    translation=tuple(cfg.relocation_translation),
                                ),
                            )
                            intervention_record = {
                                "type": "relocation", "timestep": t,
                                "object": cfg.target_object,
                                "old_pose": old_pose.tolist(), "new_pose": new_pose.tolist(),
                            }
                            intervention_applied = True
                        elif cfg.hard_case == "out_of_view":
                            obs, _camera_state = shift_camera_out_of_view(
                                env,
                                CameraShiftConfig(
                                    timestep=t,
                                    camera_name=cfg.camera_name,
                                    translation=tuple(cfg.camera_translation),
                                    yaw_radians=cfg.camera_yaw_radians,
                                ),
                            )
                            intervention_record = {
                                "type": "out_of_view", "timestep": t,
                                "camera": cfg.camera_name,
                                "yaw_radians": cfg.camera_yaw_radians,
                            }
                            intervention_applied = True

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    if cfg.hard_case == "temporal_occlusion":
                        img, occlusion_phase, _severity = apply_temporal_occlusion(
                            img, policy_calls, cfg.occlusion_schedule_steps, temporal_occlusion_cfg
                        )
                        current_occlusion_phase = occlusion_phase
                        occlusion_phase_counts[occlusion_phase] += 1

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    observation = {
                        "base_cam": img,
                        "states": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    ###############################################################################
                    ### here we need to use flask to get the action from the model
                    ### we need a policy to get the action
                    action = policy.process_frame(text=task_description,
                                                  episode_first_frame=episode_first_frame,
                                                  **observation)
                    policy_calls += 1

                    if ';' in action:
                        action = action.replace(';', ' ')

                    # str to np array
                    action = action.split(' ')
                    action = [float(x) for x in action]
                    action = np.array(action, dtype=float)

                    episode_first_frame = 'False'
                    action_dim = 7

                    # Adjust gripper action values according to your data's gripper definition
                    for i in range(len(action)):
                        if i % action_dim == action_dim - 1:
                            if action[i] == 1.0:
                                action[i] = -1.0
                            elif action[i] == 0.0:
                                action[i] = 1.0

                    done_flag = False
                    chunk_size = len(action) // action_dim
                    for i in range(chunk_size):
                        action_chunk = action[i * action_dim:(i + 1) * action_dim]

                        # Execute action in environment
                        obs, reward, done, info = env.step(action_chunk)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            done_flag = True
                            break
                        t += 1

                    if done_flag:
                        break

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    exception_text = repr(e)
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if cfg.save_rollout_videos:
                rollout_dir = os.path.join(cfg.local_log_dir, run_id + "_videos")
                save_rollout_video(
                    replay_images, total_episodes,
                    success=done, task_description=task_description,
                    log_file=log_file,
                    rollout_dir=rollout_dir,
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()
            trace_file.write(json.dumps({
                "suite": cfg.task_suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "episode": task_episodes,
                "global_episode_index": episode_idx,
                "official_initial_state_index": initial_state_idx,
                "success": bool(done),
                "timeout": bool((not done) and exception_text is None and t >= max_steps),
                "steps": int(t),
                "replay_frames": len(replay_images),
                "exception": exception_text,
                "hard_case": cfg.hard_case,
                "intervention": intervention_record,
                "occlusion_phase_counts": occlusion_phase_counts,
            }, ensure_ascii=False) + "\n")
            trace_file.flush()
            rollout_record = {
                "model": cfg.model,
                "task_name": task_description,
                "runtime_task_index": task_id,
                "initial_state_id": initial_state_idx,
                "hard_case": cfg.hard_case,
                "success": bool(done),
                "policy_calls": policy_calls,
                "timeout": bool((not done) and exception_text is None and t >= max_steps),
                "success_phase": current_occlusion_phase if done else None,
                "reached_full_occlusion": occlusion_phase_counts["full_occlusion"] > 0,
                "reached_recovery": occlusion_phase_counts["recovered_visible"] > 0,
                "recovery_success": bool(done and occlusion_phase_counts["recovered_visible"] > 0),
            }
            with open(rollout_jsonl_path, "a") as rollout_jsonl:
                rollout_jsonl.write(json.dumps(rollout_record, ensure_ascii=False) + "\n")
            csv_exists = os.path.exists(rollout_csv_path)
            with open(rollout_csv_path, "a", newline="") as rollout_csv:
                writer = csv.DictWriter(rollout_csv, fieldnames=rollout_fields)
                if not csv_exists:
                    writer.writeheader()
                writer.writerow(rollout_record)

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        env.close()

    log_file.close()
    trace_file.close()

if __name__ == "__main__":
    eval_libero()
