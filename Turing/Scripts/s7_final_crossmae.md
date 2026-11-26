# Probe stability report

## Collapse rate and scores per setting

4 of 46 runs collapsed overall.

| setting | experiment | labels | runs | collapsed | diverged | val mAP50 (all) | val mAP50 (healthy) | Vehicle AP (healthy) | train mAP50 (all) |
|---|---|---|---|---|---|---|---|---|---|
| crossmae/s5_curve_base | a_imagenet | 10% | 3 | 1 | 0 | 0.057 ± 0.044 | 0.081 ± 0.018 | 0.233 ± 0.066 | 0.672 ± 0.350 |
| crossmae/s5_curve_base | a_imagenet | 25% | 3 | 0 | 0 | 0.169 ± 0.056 | 0.169 ± 0.056 | 0.382 ± 0.077 | 0.762 ± 0.029 |
| crossmae/s5_curve_base | a_imagenet | 50% | 3 | 0 | 0 | 0.233 ± 0.038 | 0.233 ± 0.038 | 0.469 ± 0.036 | 0.667 ± 0.022 |
| crossmae/s5_curve_base | a_imagenet | 100% | 3 | 0 | 0 | 0.272 ± 0.004 | 0.272 ± 0.004 | 0.535 ± 0.005 | 0.501 ± 0.012 |
| crossmae/s6_lowlabel | a_imagenet | 10% | 7 | 2 | 0 | 0.063 ± 0.042 | 0.088 ± 0.008 | 0.249 ± 0.016 | 0.671 ± 0.353 |
| crossmae/s6_lowlabel | a_imagenet | 25% | 7 | 0 | 0 | 0.139 ± 0.015 | 0.139 ± 0.015 | 0.379 ± 0.027 | 0.696 ± 0.103 |
| crossmae/s7_pseed1 | e2d_weighted | 10% | 10 | 0 | 0 | 0.095 ± 0.015 | 0.095 ± 0.015 | 0.269 ± 0.034 | 0.850 ± 0.054 |
| crossmae/s7_pseed2 | e2d_weighted | 10% | 10 | 1 | 0 | 0.090 ± 0.034 | 0.100 ± 0.013 | 0.272 ± 0.027 | 0.783 ± 0.280 |

## Difference to a_imagenet (same seed and labels, healthy runs only)

Positive = better than the reference. *p* = two-sided sign test on the pairs (below 0.05: unlikely to be chance; needs about 6+ pairs to be possible at all).

| setting | experiment | labels | pairs | Δ val mAP50 | Δ Vehicle AP | beats ref | p (mAP) | p (Vehicle) |
|---|---|---|---|---|---|---|---|---|
| crossmae/s7_pseed1 | e2d_weighted | 10% | 7 | 0.004 ± 0.008 | 0.015 ± 0.023 | 4/7 | 1.000 | 1.000 |
| crossmae/s7_pseed2 | e2d_weighted | 10% | 7 | 0.012 ± 0.009 | 0.019 ± 0.012 | 7/7 | 0.016 | 0.016 |
