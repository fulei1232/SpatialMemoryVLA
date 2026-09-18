# SpatialMemoryVLA progress report

Date: 2026-09-18 (UTC)

## Executive summary

SpatialMemoryVLA has completed the core Geometry-Aligned Temporal Memory
implementation, distributed training validation, a matched LIBERO temporal-
occlusion study, a corrected 8-GPU RoboMME A/B/C training run, and the first
dynamic-object-relocation data/evaluation pipeline.

The strongest validated result is the controlled memory-length comparison on
LIBERO-Spatial: with 8 fully black policy observations, C16 outperforms C1 by
18 percentage points (38% versus 20%, exact paired McNemar p=0.007916). The
result remains significant after Bonferroni correction across the four tested
occlusion durations (adjusted p=0.03166). This supports a temporal-retention
claim for longer history within SpatialMemoryVLA. It does not yet establish a
significant advantage of C16 over the spatial-forcing B model.

## Method status

Implemented components:

- A frozen, training-only VGGT-1B teacher.
- Alignment of VLA layer-24 visual tokens from `[B,256,4096]` into the
  `[B,256,2048]` VGGT feature space with cosine loss.
- Compression of aligned features into `[B,256,256]` perception-memory tokens.
- Episode memory with temporal encoding, cross-attention retrieval, gated
  fusion, and FIFO or token-merge consolidation.
- Teacher-free inference: VGGT is excluded from the policy checkpoint and is
  not called during action prediction.
- Deterministic episode-aware visible/partial/full/recovered occlusion
  curricula and gate diagnostics.
- State-level LIBERO interventions for temporal occlusion, object relocation,
  camera shift, and counterfactual spatial relations.
- Relocation safety auditing, refreshed-observation checks, successful-policy-
  trajectory export, and an NPZ training-data loader.

The matched ablation remains:

| Group | Spatial alignment | Aligned spatial tokens used by memory |
| --- | ---: | ---: |
| A: MemoryVLA | No | No |
| B: Spatial Forcing | Yes | No |
| C: SpatialMemoryVLA | Yes | Yes |

## Completed training

### LIBERO-Spatial 10k functional run

The 10,000-step C checkpoint completed and passed a small normal-condition
rollout sanity test:

| Task | Success |
| --- | ---: |
| `next_to_the_plate` | 9/10 (90%) |
| `between_the_plate_and_the_ramekin` | 10/10 (100%) |
| Combined | 19/20 (95%) |

This confirms that the full policy is executable and can solve the tested
spatial-relation tasks. Because this run did not include matched A/B policies,
it is not causal evidence for the spatial-memory component.

### Corrected RoboMME A/B/C 8-GPU runs

All three groups completed 5,000 optimizer steps with the intended full
configuration: 8 GPUs, global batch 256, per-device batch 8, learning rate
2e-5, BF16 FSDP full-shard, action dimension 8, action horizon 16, and seed 42.
Unlike the earlier fast run, these corrected runs use four repeated diffusion
steps and the full trainable-backbone configuration.

| Group | Final action loss | Final spatial loss | Final total loss | Last-100 mean total loss |
| --- | ---: | ---: | ---: | ---: |
| A | 0.013010 | 0 | 0.013010 | 0.012438 |
| B | 0.013743 | 0.001386 | 0.014436 | 0.014077 |
| C | 0.013346 | 0.001758 | 0.014225 | 0.016026 |

All runs produced 5,000 JSONL metric records and final model/optimizer
checkpoints. These are optimization results only; RoboMME policy-level A/B/C
evaluation is still required before making a performance claim.

## Completed evaluation

### Temporal-occlusion retention curve

Protocol: matched B/C1/C16 500-step checkpoints; LIBERO-Spatial tasks 0 and 8;
50 official initial states per task/model/condition; inference seed 7; five
visible calls, three partially occluded calls, followed by 4/8/12/16 fully
black calls; 1,200 occluded rollouts in total.

| Fully black calls | B | C1 | C16 | C16 - C1 | Exact McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 27% | 29% | 35% | +6 pp | 0.3616 |
| 8 | 29% | 20% | 38% | +18 pp | 0.007916 |
| 12 | 19% | 13% | 27% | +14 pp | 0.02882 |
| 16 | 3% | 4% | 6% | +2 pp | 0.7266 |

The supported conclusion is that longer retained history improves robustness
to medium-duration missing visual observations, particularly at 8 calls. The
16-call condition shows a floor effect. The current scope is two tasks and one
training seed.

### Dynamic relocation smoke

The target bowl was moved by 0.10 m before policy call 5 on task 8. Ten paired
official initial states were run for each model.

| Model | Success | Safe relocation | Refreshed observation | Exceptions |
| --- | ---: | ---: | ---: | ---: |
| B | 3/10 (30%) | 10/10 | 10/10 | 0 |
| C1 | 2/10 (20%) | 10/10 | 10/10 | 0 |
| C16 | 4/10 (40%) | 10/10 | 10/10 | 0 |

This smoke test validates the intervention and logging pipeline. The sample is
too small for a performance conclusion.

## Relocation dataset

A successful-policy-rollout relocation dataset has been materialized:

- 50 episodes selected from 52 valid candidates;
- 6,120 transitions;
- four collection seeds: 7, 17, 27, and 37;
- per-transition RGB image, action, robot state, joint state, timestep, and
  relocation flag;
- per-episode instruction, task/state identity, source index, intervention
  timestep, and old/new target poses;
- audited relocation-flag lifecycle and finite tensor contents;
- total size approximately 973 MB.

The next planned experiment is C16 relocation fine-tuning/overfitting followed
by matched pre/post-relocation evaluation. A stronger method-level extension
would explicitly detect current-observation versus memory conflicts and learn
when to retain or overwrite stale spatial state.

## Verification performed

- `git diff --check`: passed.
- Shell syntax checks for relocation collection, evaluation, and training
  entry points: passed.
- `tests/test_libero_hard_cases.py` and `tests/test_memory_curriculum.py`:
  8/8 tests passed.
- No active `torchrun`, `train.py`, `deploy.py`, or `eval_libero.py` process at
  report time.

## Claim boundaries

- Training loss is not a policy-success metric.
- The 19/20 normal-condition result lacks matched A/B controls.
- C16 significantly outperforms C1 at the key medium-duration occlusion point,
  but has not significantly outperformed B at an individual duration.
- The relocation result is a 10-state smoke test, not a statistical study.
- Current results cover two LIBERO-Spatial tasks and one main training seed.

## Backup scope

GitHub is the source-of-truth backup for code, launch scripts, tests, and human-
readable reports. Hugging Face stores final model checkpoints, full metric
logs/configurations, evaluation traces/statistics, and generated datasets.

Intermediate checkpoints and optimizer states from the corrected RoboMME runs
occupy approximately 1.4 TB locally. They are redundant recovery snapshots,
not distinct experimental results, and are intentionally excluded from the
portable Hub backup. The backup includes the three final step-5000 model
checkpoints and complete training logs/configuration needed to identify the
runs. Local optimizer states remain available under the run directories.
