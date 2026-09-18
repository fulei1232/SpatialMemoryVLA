# LIBERO dynamic-relocation pipeline and smoke test

Date: 2026-09-18

## Intervention protocol

- Suite/task: LIBERO-Spatial task 8, `next_to_the_plate`
- Initial states: official states 0-9, paired across B/C1/C16
- Inference seed: 7
- Intervention: before policy call 5
- Target translation: +0.0954 m X, +0.03 m Y, 0 m Z (0.10 m total)
- Safety checks: pose tolerance, workspace bounds, unchanged height,
  observation refresh, and absence of unexpected contact

## Smoke result

| Model | Episodes | Success | Safe pose | Observation refreshed | Exceptions |
| --- | ---: | ---: | ---: | ---: | ---: |
| B | 10 | 30% | 10/10 | 10/10 | 0 |
| C1 | 10 | 20% | 10/10 | 10/10 | 0 |
| C16 | 10 | 40% | 10/10 | 10/10 | 0 |

The result validates the intervention implementation but is not large enough
for model comparison.

## Collected training data

Fifty successful C16 relocation episodes were selected from 52 valid
candidates, yielding 6,120 transitions and approximately 973 MB of compressed
NPZ data. Collection used seeds 7, 17, 27, and 37. The preparation audit found
no rejected candidates and verified a single relocation-flag transition per
episode.

The dataset is consumed through the `libero_relocation_npz` data mix. The
provided training entry point runs a C16 relocation adaptation experiment and
keeps the existing VGGT training-only constraint.
