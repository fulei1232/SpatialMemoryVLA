# RoboMME A/B/C Fast Ablation (5k Steps)

This artifact contains the completed A/B/C runs from the parameter-efficient
RoboMME ablation on four NVIDIA A100 80GB GPUs.

## Shared training configuration

- Initialization: `memvla-libero-spatial.pt`
- Steps: 5,000 optimizer steps per group
- Per-device batch size: 8
- Global batch size: 32 (no gradient accumulation)
- Learning rate: 2e-5, constant schedule
- Precision/strategy: BF16, FSDP full-shard
- Vision backbone: frozen
- Llama 2 7B backbone: frozen except for the final layer
- Action model: DiT-L
- Diffusion repeats during training: 1
- Action dimension/horizon: 8 / 16
- Memory length/group size: 16 / 16
- Seed: 42

## Ablations

- A (`memoryvla`): spatial forcing disabled, spatial memory disabled.
- B (`spatial_forcing`): spatial forcing enabled with a frozen VGGT-1B
  teacher, spatial memory disabled.
- C (`spatial_memory`): spatial forcing and spatial memory enabled with a
  frozen VGGT-1B teacher.

## Completed outputs

| Group | Final step | Final total loss | Final checkpoint |
| --- | ---: | ---: | --- |
| A | 5,000 | 0.014218 | `A/checkpoints/step-005000-epoch-00-loss=0.0142.pt` |
| B | 5,000 | 0.014952 | `B/checkpoints/step-005000-epoch-00-loss=0.0150.pt` |
| C | 5,000 | 0.016509 | `C/checkpoints/step-005000-epoch-00-loss=0.0165.pt` |

Each group includes its final optimizer state, configuration, dataset
statistics, complete JSONL metrics, and checkpoint event log. Intermediate
checkpoints are intentionally omitted from the Hub artifact because they would
add hundreds of gigabytes of redundant snapshots.

The implementation and launch scripts are maintained in
`fulei1232/SpatialMemoryVLA`.
