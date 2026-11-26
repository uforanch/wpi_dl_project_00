# Probe stability report

## Collapse rate and scores per setting

9 of 64 runs collapsed overall.

| setting | experiment | labels | runs | collapsed | diverged | val mAP50 (all) | val mAP50 (healthy) | Vehicle AP (healthy) | train mAP50 (all) |
|---|---|---|---|---|---|---|---|---|---|
| lowlabel | a_imagenet | 10% | 10 | 3 | 0 | 0.061 ± 0.040 | 0.086 ± 0.010 | 0.244 ± 0.031 | 0.671 ± 0.332 |
| lowlabel | a_imagenet | 25% | 10 | 0 | 0 | 0.148 ± 0.033 | 0.148 ± 0.033 | 0.380 ± 0.043 | 0.716 ± 0.091 |
| lowlabel | a_imagenet | 50% | 3 | 0 | 0 | 0.233 ± 0.038 | 0.233 ± 0.038 | 0.469 ± 0.036 | 0.667 ± 0.022 |
| lowlabel | a_imagenet | 100% | 3 | 0 | 0 | 0.272 ± 0.004 | 0.272 ± 0.004 | 0.535 ± 0.005 | 0.501 ± 0.012 |
| lowlabel | a_random_init | 10% | 3 | 1 | 0 | 0.026 ± 0.024 | 0.040 ± 0.008 | 0.116 ± 0.026 | 0.546 ± 0.473 |
| lowlabel | a_random_init | 25% | 3 | 1 | 0 | 0.045 ± 0.034 | 0.064 ± 0.005 | 0.185 ± 0.024 | 0.440 ± 0.276 |
| lowlabel | a_random_init | 50% | 3 | 3 | 0 | 0.093 ± 0.014 | - | - | 0.397 ± 0.030 |
| lowlabel | a_random_init | 100% | 3 | 0 | 0 | 0.119 ± 0.009 | 0.119 ± 0.009 | 0.292 ± 0.005 | 0.254 ± 0.030 |
| lowlabel | e2d_weighted | 10% | 10 | 1 | 0 | 0.089 ± 0.033 | 0.098 ± 0.017 | 0.270 ± 0.031 | 0.818 ± 0.171 |
| lowlabel | e2d_weighted | 25% | 10 | 0 | 0 | 0.150 ± 0.024 | 0.150 ± 0.024 | 0.384 ± 0.037 | 0.756 ± 0.069 |
| lowlabel | e2d_weighted | 50% | 3 | 0 | 0 | 0.232 ± 0.025 | 0.232 ± 0.025 | 0.464 ± 0.030 | 0.712 ± 0.030 |
| lowlabel | e2d_weighted | 100% | 3 | 0 | 0 | 0.265 ± 0.005 | 0.265 ± 0.005 | 0.528 ± 0.002 | 0.523 ± 0.012 |

## Difference to a_imagenet (same seed and labels, healthy runs only)

Positive = better than the reference. *p* = two-sided sign test on the pairs (below 0.05: unlikely to be chance; needs about 6+ pairs to be possible at all).

| setting | experiment | labels | pairs | Δ val mAP50 | Δ Vehicle AP | beats ref | p (mAP) | p (Vehicle) |
|---|---|---|---|---|---|---|---|---|
| lowlabel | a_random_init | 10% | 2 | -0.042 ± 0.010 | -0.117 ± 0.039 | 0/2 | 0.500 | 0.500 |
| lowlabel | a_random_init | 25% | 2 | -0.073 ± 0.010 | -0.162 ± 0.043 | 0/2 | 0.500 | 0.500 |
| lowlabel | a_random_init | 50% | 0 | - | - | 0/0 | - | - |
| lowlabel | a_random_init | 100% | 3 | -0.153 ± 0.005 | -0.243 ± 0.009 | 0/3 | 0.250 | 0.250 |
| lowlabel | e2d_weighted | 10% | 7 | 0.009 ± 0.008 | 0.021 ± 0.014 | 7/7 | 0.016 | 0.016 |
| lowlabel | e2d_weighted | 25% | 10 | 0.002 ± 0.012 | 0.004 ± 0.020 | 7/10 | 0.344 | 1.000 |
| lowlabel | e2d_weighted | 50% | 3 | -0.001 ± 0.015 | -0.005 ± 0.010 | 2/3 | 1.000 | 1.000 |
| lowlabel | e2d_weighted | 100% | 3 | -0.008 ± 0.007 | -0.007 ± 0.005 | 0/3 | 0.250 | 0.250 |
