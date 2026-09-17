# LIBERO temporal-occlusion memory evaluation (2026-09-17)

## Result

Longer memory history improves robustness to medium-duration visual occlusion in
the matched two-task LIBERO-Spatial evaluation.  C16 significantly outperforms
the direct C1 history ablation at 8 fully occluded policy calls and retains a
positive advantage at 12 calls.  Four calls are too weak to separate the
models, while 16 calls produce a floor effect.

| Fully black policy calls | B success | C1 success | C16 success | C16 - C1 | Exact McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 27% | 29% | 35% | +6 pp | 0.3616 |
| 8 | 29% | 20% | 38% | +18 pp | 0.007916 |
| 12 | 19% | 13% | 27% | +14 pp | 0.02882 |
| 16 | 3% | 4% | 6% | +2 pp | 0.7266 |

The 8-call C16-vs-C1 result remains significant after Bonferroni correction
across the four duration-wise comparisons (adjusted p=0.03166).  Descriptive
trapezoidal mean success over 4--16 calls is 21.0% for B, 16.5% for C1, and
28.5% for C16.  C16 does not significantly outperform B at an individual
duration, so the supported claim is specifically about the C16-vs-C1 history
length ablation.  The current scope is two tasks and one training seed.

## Protocol

- Models: matched 500-step B, C1, and C16 checkpoints.
- Tasks: LIBERO-Spatial task 0 and task 8.
- Episodes: 50 official initial states per task, model, and condition.
- Inference seed: 7 for all paired runs.
- Occlusion schedule: 5 visible policy calls, 3 partially occluded calls, then
  4, 8, 12, or 16 fully black calls before vision recovers.
- Action chunking: 8 environment actions per policy call.
- Total duration-curve sample size: 1,200 occluded rollouts.
- Runtime exceptions: 0.

## Reproduction

Training:

```bash
script/train/libero/train_libero_spatial_occlusion_b_c1_c16.sh
```

Matched 4-call normal/occlusion evaluation:

```bash
script/eval/libero/run_b_c1_c16_500step_formal.sh
```

Long 16-call evaluation:

```bash
script/eval/libero/run_b_c1_c16_500step_long_occlusion.sh
```

Intermediate 8/12-call sweep:

```bash
script/eval/libero/run_b_c1_c16_occlusion_sweep_8_12.sh
```

The result aggregation entry points are
`evaluation/libero/summarize_b_c1_c16_rollouts.py` and
`evaluation/libero/build_occlusion_length_curve.py`.

## Published artifacts

- Checkpoints: <https://huggingface.co/fulei1232/SpatialMemoryVLA-checkpoints/tree/main/libero_spatial_memory_occlusion_500step>
- Rollouts and statistics: <https://huggingface.co/datasets/fulei1232/spatialmemoryvla-libero-abc-rollouts/tree/main/temporal_occlusion_memory_20260917>
