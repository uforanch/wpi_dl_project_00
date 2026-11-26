# Probe stability report

## Collapse rate and scores per setting

1 of 18 runs collapsed overall.

| setting | experiment | labels | runs | collapsed | diverged | val mAP50 (all) | val mAP50 (healthy) | Vehicle AP (healthy) | train mAP50 (all) |
|---|---|---|---|---|---|---|---|---|---|
| crossmae/fix_grid | a_imagenet | 10% | 3 | 1 | 0 | 0.057 ± 0.044 | 0.081 ± 0.016 | 0.229 ± 0.067 | 0.671 ± 0.354 |
| crossmae/fix_grid | a_imagenet | 100% | 3 | 0 | 0 | 0.277 ± 0.001 | 0.277 ± 0.001 | 0.532 ± 0.005 | 0.504 ± 0.004 |
| crossmae/fix_grid | b_plain | 10% | 3 | 0 | 0 | 0.076 ± 0.017 | 0.076 ± 0.017 | 0.216 ± 0.038 | 0.811 ± 0.026 |
| crossmae/fix_grid | b_plain | 100% | 3 | 0 | 0 | 0.222 ± 0.006 | 0.222 ± 0.006 | 0.486 ± 0.007 | 0.440 ± 0.009 |
| crossmae/fix_grid | c_ego | 10% | 3 | 0 | 0 | 0.103 ± 0.047 | 0.103 ± 0.047 | 0.244 ± 0.045 | 0.845 ± 0.023 |
| crossmae/fix_grid | c_ego | 100% | 3 | 0 | 0 | 0.234 ± 0.013 | 0.234 ± 0.013 | 0.511 ± 0.010 | 0.457 ± 0.040 |

## Loss behaviour per run (from the logs)

Labels are a heuristic: *diverged* = NaN/inf loss; *never started* = loss never fell below 85% of its first value; *spiked* = loss jumped to over 2× its lowest value and ended >30% above it; otherwise *learned*. Check the numbers before quoting a label.

| setting | experiment | labels | seed | behaviour | first loss | min loss | last loss | classification loss | csv |
|---|---|---|---|---|---|---|---|---|---|
| crossmae/fix_grid_seed0 | a_imagenet | 10% | 0 | learned | 2.965 | 0.619 | 0.686 | 1.274 → 0.009 | ok |
| crossmae/fix_grid_seed0 | a_imagenet | 100% | 0 | learned | 2.958 | 0.987 | 1.056 | 1.262 → 0.149 | ok |
| crossmae/fix_grid_seed0 | b_plain | 10% | 0 | learned | 2.875 | 0.603 | 0.687 | 1.199 → 0.009 | ok |
| crossmae/fix_grid_seed0 | b_plain | 100% | 0 | learned | 2.905 | 0.951 | 1.092 | 1.209 → 0.171 | ok |
| crossmae/fix_grid_seed0 | c_ego | 10% | 0 | learned | 2.906 | 0.629 | 0.698 | 1.221 → 0.010 | ok |
| crossmae/fix_grid_seed0 | c_ego | 100% | 0 | learned | 2.913 | 0.949 | 1.078 | 1.243 → 0.156 | ok |
| crossmae/fix_grid_seed1 | a_imagenet | 10% | 1 | learned | 2.765 | 0.630 | 0.652 | 1.021 → 0.002 | ok |
| crossmae/fix_grid_seed1 | a_imagenet | 100% | 1 | learned | 2.817 | 0.991 | 1.109 | 1.091 → 0.158 | ok |
| crossmae/fix_grid_seed1 | b_plain | 10% | 1 | learned | 2.784 | 0.642 | 0.653 | 1.057 → 0.003 | ok |
| crossmae/fix_grid_seed1 | b_plain | 100% | 1 | learned | 2.731 | 1.019 | 1.107 | 1.018 → 0.148 | ok |
| crossmae/fix_grid_seed1 | c_ego | 10% | 1 | learned | 2.792 | 0.637 | 0.658 | 1.082 → 0.004 | ok |
| crossmae/fix_grid_seed1 | c_ego | 100% | 1 | learned | 2.770 | 1.005 | 1.110 | 1.047 → 0.156 | ok |
| crossmae/fix_grid_seed2 | a_imagenet | 10% | 2 | learned | 2.902 | 1.123 | 1.149 | 1.194 → 0.003 | COLLAPSED |
| crossmae/fix_grid_seed2 | a_imagenet | 100% | 2 | learned | 2.896 | 0.924 | 1.081 | 1.196 → 0.154 | ok |
| crossmae/fix_grid_seed2 | b_plain | 10% | 2 | learned | 2.910 | 0.664 | 0.719 | 1.216 → 0.007 | ok |
| crossmae/fix_grid_seed2 | b_plain | 100% | 2 | learned | 2.897 | 0.937 | 1.074 | 1.194 → 0.134 | ok |
| crossmae/fix_grid_seed2 | c_ego | 10% | 2 | learned | 2.921 | 0.668 | 0.719 | 1.199 → 0.007 | ok |
| crossmae/fix_grid_seed2 | c_ego | 100% | 2 | learned | 2.943 | 0.964 | 1.081 | 1.229 → 0.137 | ok |

### Behaviour counts

| setting | experiment | labels | diverged | never started | spiked | learned |
|---|---|---|---|---|---|---|
| crossmae/fix_grid | a_imagenet | 10% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | a_imagenet | 100% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | b_plain | 10% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | b_plain | 100% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | c_ego | 10% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | c_ego | 100% | 0 | 0 | 0 | 3 |

## Difference to a_imagenet (same seed and labels, healthy runs only)

Positive = better than the reference.

| setting | experiment | labels | pairs | Δ val mAP50 | Δ Vehicle AP | beats ref |
|---|---|---|---|---|---|---|
| crossmae/fix_grid | b_plain | 10% | 2 | -0.011 ± 0.002 | -0.024 ± 0.019 | 0/2 |
| crossmae/fix_grid | b_plain | 100% | 3 | -0.055 ± 0.006 | -0.045 ± 0.004 | 0/3 |
| crossmae/fix_grid | c_ego | 10% | 2 | -0.004 ± 0.001 | -0.000 ± 0.016 | 0/2 |
| crossmae/fix_grid | c_ego | 100% | 3 | -0.043 ± 0.013 | -0.020 ± 0.009 | 0/3 |
