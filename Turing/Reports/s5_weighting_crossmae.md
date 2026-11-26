# Probe stability report

## Collapse rate and scores per setting

1 of 21 runs collapsed overall.

| setting | experiment | labels | runs | collapsed | diverged | val mAP50 (all) | val mAP50 (healthy) | Vehicle AP (healthy) | train mAP50 (all) |
|---|---|---|---|---|---|---|---|---|---|
| weighting_enc0.1 | a_imagenet | 10% | 3 | 1 | 0 | 0.057 ± 0.044 | 0.081 ± 0.016 | 0.229 ± 0.067 | 0.671 ± 0.354 |
| weighting_enc0.1 | a_imagenet | 100% | 3 | 0 | 0 | 0.277 ± 0.001 | 0.277 ± 0.001 | 0.532 ± 0.005 | 0.504 ± 0.004 |
| weighting_enc0.1 | b_plain | 100% | 3 | 0 | 0 | 0.254 ± 0.005 | 0.254 ± 0.005 | 0.523 ± 0.005 | 0.503 ± 0.025 |
| weighting_enc0.1 | c_count | 100% | 3 | 0 | 0 | 0.249 ± 0.003 | 0.249 ± 0.003 | 0.513 ± 0.004 | 0.486 ± 0.030 |
| weighting_enc0.1 | c_ego | 100% | 3 | 0 | 0 | 0.249 ± 0.010 | 0.249 ± 0.010 | 0.515 ± 0.004 | 0.475 ± 0.029 |
| weighting_enc0.1 | c_floor | 100% | 3 | 0 | 0 | 0.242 ± 0.009 | 0.242 ± 0.009 | 0.529 ± 0.013 | 0.471 ± 0.029 |
| weighting_enc0.1 | c_recency | 100% | 3 | 0 | 0 | 0.246 ± 0.004 | 0.246 ± 0.004 | 0.511 ± 0.002 | 0.501 ± 0.013 |

## Difference to a_imagenet (same seed and labels, healthy runs only)

Positive = better than the reference.

| setting | experiment | labels | pairs | Δ val mAP50 | Δ Vehicle AP | beats ref |
|---|---|---|---|---|---|---|
| weighting_enc0.1 | b_plain | 100% | 3 | -0.023 ± 0.005 | -0.009 ± 0.008 | 0/3 |
| weighting_enc0.1 | c_count | 100% | 3 | -0.028 ± 0.003 | -0.019 ± 0.006 | 0/3 |
| weighting_enc0.1 | c_ego | 100% | 3 | -0.028 ± 0.010 | -0.017 ± 0.003 | 0/3 |
| weighting_enc0.1 | c_floor | 100% | 3 | -0.034 ± 0.008 | -0.003 ± 0.008 | 0/3 |
| weighting_enc0.1 | c_recency | 100% | 3 | -0.031 ± 0.004 | -0.021 ± 0.007 | 0/3 |
