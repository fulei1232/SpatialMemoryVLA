# RoboMME corrected 8-GPU A/B/C training

Date: 2026-09-18

All matched A/B/C runs completed 5,000 optimizer steps.

## Shared protocol

- Initialization: `memvla-libero-spatial.pt`
- Hardware: 8 GPUs
- Global/per-device batch: 256/8
- Learning rate: 2e-5, constant
- Precision/strategy: BF16, FSDP full-shard
- Diffusion training repeats: 4
- Action dimension/horizon: 8/16
- Memory length/group size: 16/16
- Seed: 42

## Final metrics

| Group | Action loss | Spatial loss | Weighted ratio | Grad norm | Total loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.013010 | 0 | 0 | 0.406501 | 0.013010 |
| B | 0.013743 | 0.001386 | 0.050441 | 0.364619 | 0.014436 |
| C | 0.013346 | 0.001758 | 0.065874 | 0.460708 | 0.014225 |

Last-100-step mean total losses are 0.012438, 0.014077, and 0.016026 for A, B,
and C, respectively. All JSONL logs contain exactly 5,000 records.

These values verify completion and numerical stability. They do not establish
policy performance; matched RoboMME downstream evaluation remains pending.
