# SpatialMemoryVLA step-10000 — rollout sanity and geometry report

Date: 2026-09-04

## Evaluation setup

- Checkpoint: `/media/fulei/jlu/spatial_memory_libero_spatial_functional--image_aug/checkpoints/step-010000-epoch-48-loss=0.0210.pt`
- Suite: `libero_spatial`
- Trials: 10 official initial states per task, seed 7
- Inference: bfloat16, DDIM defaults, CFG scale 1.5, action chunking window 8
- Episode horizon: 220 simulator steps plus 10 stabilization steps
- Action normalization key: `libero_spatial_no_noops`
- Only checkpoint C (SpatialMemoryVLA) was evaluated; no A/B training was started.

## Quantitative results

| Task | Runtime task index | Successes | Success rate | Policy calls per successful episode |
| --- | ---: | ---: | ---: | --- |
| `next_to_the_plate` | 8 | 9/10 | 90% | 11–13 |
| `between_the_plate_and_the_ramekin` | 0 | 10/10 | 100% | 10–11 |
| Combined | — | 19/20 | 95% | — |

The single failure used all 28 policy calls (approximately the full 220-step
horizon with action chunks), whereas successful episodes terminated much
earlier.

## Behavioral review

### `next_to_the_plate` sanity check

- Correct target motion: yes. Reviewed rollouts approach the black bowl next to
  the plate rather than a distractor.
- Grasp: normal in successful rollouts. The end effector reaches the target,
  closes the gripper, and lifts the bowl.
- Transport: yes. The grasped bowl is moved toward the pink-rimmed plate.
- Gripper: both close and release phases are visibly functional.
- Placement: 9/10 satisfy the simulator success predicate.
- Failure mode (episode 9): target selection, grasp, lift, and transport all
  occur. The bowl is released near/on the plate edge, but the placement does not
  satisfy the task predicate. The policy then remains near the terminal pose
  until the horizon. This is a placement-precision failure, not a target-choice
  or gripper failure.

### `between_the_plate_and_the_ramekin` geometry check

- All 10 official initial states succeeded.
- Reviewed trajectories approach the black bowl defined by the between-relation,
  grasp it, carry it to the plate, and release it successfully.
- No observed confusion with the neighboring ramekin or other bowls.
- Successful trajectories are compact and consistent: 10–11 policy calls.

## Interpretation

Checkpoint C passes the rollout sanity check and gives positive initial evidence
for the geometry hypothesis. The relational task is not merely attempted: it
achieves 10/10 with consistent short trajectories. This is evidence that the
trained policy can use the relevant spatial relation under these official
initial states.

This small evaluation does not by itself prove that the spatial-memory component
caused the result. A causal claim still requires the same evaluation protocol on
matched A/B baselines or an ablation. The 90% sanity score also indicates a
remaining placement-precision tail failure worth tracking at larger sample size.

## Artifacts

- Sanity log and videos:
  `/media/fulei/jlu/spatial_memory_libero_spatial_functional--image_aug/eval_step10000_C/next_to_the_plate_valid/`
- Geometry log and videos:
  `/media/fulei/jlu/spatial_memory_libero_spatial_functional--image_aug/eval_step10000_C/between_the_plate_and_the_ramekin_valid/`

Earlier directories without the `_valid` suffix contain environment/proxy setup
attempts and must not be included in metrics.

## Evaluation environment fixes

- Pinned `mujoco==2.3.7` for compatibility with `robosuite==1.4.0`.
- Made the SimplerEnv-only policy import lazy so LIBERO deployment does not
  require unrelated `transforms3d` dependencies.
- Allowed the renderer backend to be selected externally and used EGL.
- Forced the local inference client to bypass the cluster HTTP proxy.
