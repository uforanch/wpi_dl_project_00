# Probe stability report

## Collapse rate and scores per setting

3 of 33 runs collapsed overall.

| setting | experiment | labels | runs | collapsed | diverged | val mAP50 (all) | val mAP50 (healthy) | Vehicle AP (healthy) | train mAP50 (all) |
|---|---|---|---|---|---|---|---|---|---|
| videomae/fix_grid | a_imagenet | 10% | 3 | 0 | 0 | 0.131 ± 0.026 | 0.131 ± 0.026 | 0.367 ± 0.056 | 0.876 ± 0.042 |
| videomae/fix_grid | a_imagenet | 100% | 3 | 0 | 0 | 0.251 ± 0.009 | 0.251 ± 0.009 | 0.567 ± 0.004 | 0.434 ± 0.015 |
| videomae/fix_grid | b_plain | 10% | 3 | 0 | 0 | 0.079 ± 0.018 | 0.079 ± 0.018 | 0.235 ± 0.052 | 0.808 ± 0.039 |
| videomae/fix_grid | b_plain | 100% | 3 | 0 | 0 | 0.184 ± 0.011 | 0.184 ± 0.011 | 0.459 ± 0.005 | 0.349 ± 0.009 |
| videomae/fix_grid | c_ego | 10% | 3 | 0 | 0 | 0.068 ± 0.009 | 0.068 ± 0.009 | 0.202 ± 0.025 | 0.735 ± 0.049 |
| videomae/fix_grid | c_ego | 100% | 3 | 0 | 0 | 0.150 ± 0.009 | 0.150 ± 0.009 | 0.391 ± 0.008 | 0.312 ± 0.014 |
| videomae/q_baseline | a_imagenet | 100% | 4 | 1 | 0 | 0.191 ± 0.101 | 0.241 ± 0.011 | 0.562 ± 0.017 | 0.277 ± 0.139 |
| videomae/q_lr | a_imagenet | 100% | 4 | 0 | 0 | 0.253 ± 0.007 | 0.253 ± 0.007 | 0.564 ± 0.010 | 0.451 ± 0.016 |
| videomae/s1_baseline | a_imagenet | 100% | 3 | 1 | 0 | 0.174 ± 0.118 | 0.241 ± 0.019 | 0.566 ± 0.001 | 0.265 ± 0.174 |
| videomae/s1_baseline | b_plain | 100% | 2 | 1 | 0 | 0.100 ± 0.097 | 0.169 | 0.448 | 0.159 ± 0.118 |
| videomae/s1_baseline | c_ego | 100% | 2 | 0 | 0 | 0.147 ± 0.013 | 0.147 ± 0.013 | 0.398 ± 0.001 | 0.220 ± 0.044 |

## Loss behaviour per run (from the logs)

Labels are a heuristic: *diverged* = NaN/inf loss; *never started* = loss never fell below 85% of its first value; *spiked* = loss jumped to over 2× its lowest value and ended >30% above it; otherwise *learned*. Check the numbers before quoting a label.

| setting | experiment | labels | seed | behaviour | first loss | min loss | last loss | classification loss | csv |
|---|---|---|---|---|---|---|---|---|---|
| videomae/fix_grid_seed0 | a_imagenet | 10% | 0 | learned | 2.768 | 0.598 | 0.598 | 1.071 → 0.003 | ok |
| videomae/fix_grid_seed0 | a_imagenet | 100% | 0 | learned | 2.777 | 1.007 | 1.107 | 1.087 → 0.161 | ok |
| videomae/fix_grid_seed0 | b_plain | 10% | 0 | learned | 2.842 | 0.685 | 0.711 | 1.138 → 0.022 | ok |
| videomae/fix_grid_seed0 | b_plain | 100% | 0 | learned | 2.842 | 1.050 | 1.085 | 1.148 → 0.164 | ok |
| videomae/fix_grid_seed0 | c_ego | 10% | 0 | learned | 2.859 | 0.677 | 0.677 | 1.160 → 0.022 | ok |
| videomae/fix_grid_seed0 | c_ego | 100% | 0 | learned | 2.853 | 1.134 | 1.147 | 1.155 → 0.208 | ok |
| videomae/fix_grid_seed1 | a_imagenet | 10% | 1 | learned | 2.789 | 0.652 | 0.668 | 1.084 → 0.004 | ok |
| videomae/fix_grid_seed1 | a_imagenet | 100% | 1 | learned | 2.781 | 0.991 | 1.141 | 1.087 → 0.230 | ok |
| videomae/fix_grid_seed1 | b_plain | 10% | 1 | learned | 2.793 | 0.692 | 0.743 | 1.093 → 0.020 | ok |
| videomae/fix_grid_seed1 | b_plain | 100% | 1 | learned | 2.765 | 0.965 | 1.283 | 1.068 → 0.247 | ok |
| videomae/fix_grid_seed1 | c_ego | 10% | 1 | learned | 2.842 | 0.700 | 0.772 | 1.129 → 0.026 | ok |
| videomae/fix_grid_seed1 | c_ego | 100% | 1 | learned | 2.839 | 1.019 | 1.390 | 1.111 → 0.298 | ok |
| videomae/fix_grid_seed2 | a_imagenet | 10% | 2 | learned | 2.905 | 0.636 | 0.636 | 1.175 → 0.005 | ok |
| videomae/fix_grid_seed2 | a_imagenet | 100% | 2 | learned | 2.879 | 0.920 | 1.133 | 1.171 → 0.179 | ok |
| videomae/fix_grid_seed2 | b_plain | 10% | 2 | learned | 2.911 | 0.670 | 0.743 | 1.217 → 0.013 | ok |
| videomae/fix_grid_seed2 | b_plain | 100% | 2 | learned | 2.872 | 1.048 | 1.307 | 1.184 → 0.284 | ok |
| videomae/fix_grid_seed2 | c_ego | 10% | 2 | learned | 2.902 | 0.674 | 0.750 | 1.204 → 0.014 | ok |
| videomae/fix_grid_seed2 | c_ego | 100% | 2 | learned | 2.937 | 1.080 | 1.325 | 1.240 → 0.280 | ok |
| videomae/q_baseline | a_imagenet | 100% | 0 | learned | 2.777 | 1.404 | 1.489 | 1.087 → 0.176 | COLLAPSED |
| videomae/q_baseline | a_imagenet | 100% | 1 | learned | 2.781 | 1.002 | 1.182 | 1.087 → 0.256 | ok |
| videomae/q_baseline | a_imagenet | 100% | 2 | learned | 2.879 | 0.989 | 1.246 | 1.171 → 0.218 | ok |
| videomae/q_baseline | a_imagenet | 100% | 3 | learned | 2.730 | 0.941 | 1.243 | 1.038 → 0.238 | ok |
| videomae/q_lr | a_imagenet | 100% | 0 | learned | 2.777 | 1.011 | 1.092 | 1.087 → 0.159 | ok |
| videomae/q_lr | a_imagenet | 100% | 1 | learned | 2.781 | 0.980 | 1.150 | 1.087 → 0.229 | ok |
| videomae/q_lr | a_imagenet | 100% | 2 | learned | 2.879 | 0.935 | 1.131 | 1.171 → 0.170 | ok |
| videomae/q_lr | a_imagenet | 100% | 3 | learned | 2.730 | 0.886 | 1.157 | 1.038 → 0.216 | ok |
| videomae/s1_baseline | a_imagenet | 100% | 0 | learned | 2.777 | 1.399 | 1.489 | 1.087 → 0.174 | COLLAPSED |
| videomae/s1_baseline | b_plain | 100% | 0 | learned | 2.842 | 1.433 | 1.478 | 1.148 → 0.166 | COLLAPSED |
| videomae/s1_baseline | c_ego | 100% | 0 | learned | 2.853 | 1.207 | 1.207 | 1.155 → 0.215 | ok |
| videomae/s1_baseline | a_imagenet | 100% | 1 | learned | 2.781 | 1.024 | 1.200 | 1.087 → 0.267 | ok |
| videomae/s1_baseline | b_plain | 100% | 1 | learned | 2.765 | 0.991 | 1.371 | 1.068 → 0.299 | ok |
| videomae/s1_baseline | c_ego | 100% | 1 | learned | 2.839 | 1.058 | 1.472 | 1.111 → 0.367 | ok |
| videomae/s1_baseline | a_imagenet | 100% | 2 | learned | 2.879 | 0.983 | 1.232 | 1.171 → 0.209 | ok |
| videomae/s1_baseline | b_plain | 100% | 2 | learned | 2.872 | 1.097 | 1.281 | 1.184 → 0.248 | not in csv |

### Behaviour counts

| setting | experiment | labels | diverged | never started | spiked | learned |
|---|---|---|---|---|---|---|
| videomae/fix_grid | a_imagenet | 10% | 0 | 0 | 0 | 3 |
| videomae/fix_grid | a_imagenet | 100% | 0 | 0 | 0 | 3 |
| videomae/fix_grid | b_plain | 10% | 0 | 0 | 0 | 3 |
| videomae/fix_grid | b_plain | 100% | 0 | 0 | 0 | 3 |
| videomae/fix_grid | c_ego | 10% | 0 | 0 | 0 | 3 |
| videomae/fix_grid | c_ego | 100% | 0 | 0 | 0 | 3 |
| videomae/q_baseline | a_imagenet | 100% | 0 | 0 | 0 | 4 |
| videomae/q_lr | a_imagenet | 100% | 0 | 0 | 0 | 4 |
| videomae/s1_baseline | a_imagenet | 100% | 0 | 0 | 0 | 3 |
| videomae/s1_baseline | b_plain | 100% | 0 | 0 | 0 | 3 |
| videomae/s1_baseline | c_ego | 100% | 0 | 0 | 0 | 2 |

## Difference to a_imagenet (same seed and labels, healthy runs only)

Positive = better than the reference.

| setting | experiment | labels | pairs | Δ val mAP50 | Δ Vehicle AP | beats ref |
|---|---|---|---|---|---|---|
| videomae/fix_grid | b_plain | 10% | 3 | -0.052 ± 0.020 | -0.132 ± 0.023 | 0/3 |
| videomae/fix_grid | b_plain | 100% | 3 | -0.068 ± 0.008 | -0.108 ± 0.001 | 0/3 |
| videomae/fix_grid | c_ego | 10% | 3 | -0.063 ± 0.020 | -0.166 ± 0.032 | 0/3 |
| videomae/fix_grid | c_ego | 100% | 3 | -0.101 ± 0.011 | -0.176 ± 0.008 | 0/3 |
| videomae/s1_baseline | b_plain | 100% | 1 | -0.059 | -0.118 | 0/1 |
| videomae/s1_baseline | c_ego | 100% | 1 | -0.072 | -0.168 | 0/1 |
