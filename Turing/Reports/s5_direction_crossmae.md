# Probe stability report

## Collapse rate and scores per setting

2 of 42 runs collapsed overall.

| setting | experiment | labels | runs | collapsed | diverged | val mAP50 (all) | val mAP50 (healthy) | Vehicle AP (healthy) | train mAP50 (all) |
|---|---|---|---|---|---|---|---|---|---|
| enc0.1_all_pretrain_seeds | a_imagenet | 10% | 3 | 1 | 0 | 0.057 ± 0.044 | 0.081 ± 0.016 | 0.229 ± 0.067 | 0.671 ± 0.354 |
| enc0.1_all_pretrain_seeds | a_imagenet | 100% | 3 | 0 | 0 | 0.277 ± 0.001 | 0.277 ± 0.001 | 0.532 ± 0.005 | 0.504 ± 0.004 |
| enc0.1_all_pretrain_seeds | b_plain | 100% | 3 | 0 | 0 | 0.254 ± 0.005 | 0.254 ± 0.005 | 0.523 ± 0.005 | 0.503 ± 0.025 |
| enc0.1_all_pretrain_seeds | c_ego | 100% | 9 | 1 | 0 | 0.227 ± 0.075 | 0.251 ± 0.018 | 0.516 ± 0.010 | 0.437 ± 0.112 |
| enc0.1_all_pretrain_seeds | d2e_plain | 100% | 3 | 0 | 0 | 0.244 ± 0.015 | 0.244 ± 0.015 | 0.509 ± 0.009 | 0.447 ± 0.010 |
| enc0.1_all_pretrain_seeds | d2e_weighted | 100% | 9 | 0 | 0 | 0.247 ± 0.016 | 0.247 ± 0.016 | 0.517 ± 0.010 | 0.479 ± 0.026 |
| enc0.1_all_pretrain_seeds | e2d_plain | 100% | 3 | 0 | 0 | 0.253 ± 0.006 | 0.253 ± 0.006 | 0.526 ± 0.005 | 0.501 ± 0.018 |
| enc0.1_all_pretrain_seeds | e2d_weighted | 100% | 9 | 0 | 0 | 0.257 ± 0.009 | 0.257 ± 0.009 | 0.525 ± 0.009 | 0.504 ± 0.022 |

## Difference to a_imagenet (same seed and labels, healthy runs only)

Positive = better than the reference.

| setting | experiment | labels | pairs | Δ val mAP50 | Δ Vehicle AP | beats ref |
|---|---|---|---|---|---|---|
| enc0.1_all_pretrain_seeds | b_plain | 100% | 3 | -0.023 ± 0.005 | -0.009 ± 0.008 | 0/3 |
| enc0.1_all_pretrain_seeds | c_ego | 100% | 8 | -0.026 ± 0.017 | -0.016 ± 0.008 | 1/8 |
| enc0.1_all_pretrain_seeds | d2e_plain | 100% | 3 | -0.033 ± 0.015 | -0.023 ± 0.013 | 0/3 |
| enc0.1_all_pretrain_seeds | d2e_weighted | 100% | 9 | -0.029 ± 0.016 | -0.015 ± 0.009 | 0/9 |
| enc0.1_all_pretrain_seeds | e2d_plain | 100% | 3 | -0.024 ± 0.006 | -0.005 ± 0.009 | 0/3 |
| enc0.1_all_pretrain_seeds | e2d_weighted | 100% | 9 | -0.019 ± 0.009 | -0.007 ± 0.008 | 0/9 |
