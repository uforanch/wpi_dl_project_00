# Probe stability report

## Collapse rate and scores per setting

1 of 54 runs collapsed overall.

| setting | experiment | labels | runs | collapsed | diverged | val mAP50 (all) | val mAP50 (healthy) | Vehicle AP (healthy) | train mAP50 (all) |
|---|---|---|---|---|---|---|---|---|---|
| crossmae/fix_grid | a_imagenet | 10% | 3 | 1 | 0 | 0.057 ± 0.044 | 0.081 ± 0.016 | 0.229 ± 0.067 | 0.671 ± 0.354 |
| crossmae/fix_grid | a_imagenet | 100% | 3 | 0 | 0 | 0.277 ± 0.001 | 0.277 ± 0.001 | 0.532 ± 0.005 | 0.504 ± 0.004 |
| crossmae/fix_grid | b_plain | 10% | 3 | 0 | 0 | 0.076 ± 0.017 | 0.076 ± 0.017 | 0.216 ± 0.038 | 0.811 ± 0.026 |
| crossmae/fix_grid | b_plain | 100% | 3 | 0 | 0 | 0.222 ± 0.006 | 0.222 ± 0.006 | 0.486 ± 0.007 | 0.440 ± 0.009 |
| crossmae/fix_grid | c_ego | 10% | 3 | 0 | 0 | 0.103 ± 0.047 | 0.103 ± 0.047 | 0.244 ± 0.045 | 0.845 ± 0.023 |
| crossmae/fix_grid | c_ego | 100% | 3 | 0 | 0 | 0.234 ± 0.013 | 0.234 ± 0.013 | 0.511 ± 0.010 | 0.457 ± 0.040 |
| crossmae/s3_enc0.1 | b_plain | 100% | 3 | 0 | 0 | 0.254 ± 0.005 | 0.254 ± 0.005 | 0.523 ± 0.005 | 0.503 ± 0.025 |
| crossmae/s3_enc0.1 | c_ego | 100% | 3 | 0 | 0 | 0.249 ± 0.010 | 0.249 ± 0.010 | 0.515 ± 0.004 | 0.475 ± 0.029 |
| crossmae/s3_step2000 | b_plain | 100% | 3 | 0 | 0 | 0.258 ± 0.005 | 0.258 ± 0.005 | 0.502 ± 0.000 | 0.453 ± 0.030 |
| crossmae/s3_step2000 | c_ego | 100% | 3 | 0 | 0 | 0.237 ± 0.013 | 0.237 ± 0.013 | 0.491 ± 0.008 | 0.467 ± 0.028 |
| crossmae/s4_d2e_enc0.1 | d2e_plain | 100% | 3 | 0 | 0 | 0.244 ± 0.015 | 0.244 ± 0.015 | 0.509 ± 0.009 | 0.447 ± 0.010 |
| crossmae/s4_d2e_enc0.1 | d2e_weighted | 100% | 3 | 0 | 0 | 0.246 ± 0.020 | 0.246 ± 0.020 | 0.519 ± 0.004 | 0.491 ± 0.025 |
| crossmae/s4_d2e_full | d2e_plain | 100% | 3 | 0 | 0 | 0.236 ± 0.004 | 0.236 ± 0.004 | 0.515 ± 0.007 | 0.466 ± 0.010 |
| crossmae/s4_d2e_full | d2e_weighted | 100% | 3 | 0 | 0 | 0.221 ± 0.016 | 0.221 ± 0.016 | 0.492 ± 0.006 | 0.488 ± 0.023 |
| crossmae/s4_e2d_enc0.1 | e2d_plain | 100% | 3 | 0 | 0 | 0.253 ± 0.006 | 0.253 ± 0.006 | 0.526 ± 0.005 | 0.501 ± 0.018 |
| crossmae/s4_e2d_enc0.1 | e2d_weighted | 100% | 3 | 0 | 0 | 0.264 ± 0.006 | 0.264 ± 0.006 | 0.528 ± 0.004 | 0.522 ± 0.012 |
| crossmae/s4_e2d_full | e2d_plain | 100% | 3 | 0 | 0 | 0.242 ± 0.002 | 0.242 ± 0.002 | 0.507 ± 0.010 | 0.453 ± 0.019 |
| crossmae/s4_e2d_full | e2d_weighted | 100% | 3 | 0 | 0 | 0.249 ± 0.012 | 0.249 ± 0.012 | 0.500 ± 0.006 | 0.478 ± 0.021 |

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
| crossmae/s3_enc0.1 | b_plain | 100% | 0 | learned | 2.839 | 0.940 | 1.097 | 1.169 → 0.181 | ok |
| crossmae/s3_enc0.1 | c_ego | 100% | 0 | learned | 2.930 | 0.950 | 1.096 | 1.232 → 0.168 | ok |
| crossmae/s3_enc0.1 | b_plain | 100% | 1 | learned | 2.763 | 0.974 | 1.063 | 1.057 → 0.127 | ok |
| crossmae/s3_enc0.1 | c_ego | 100% | 1 | learned | 2.770 | 0.975 | 1.076 | 1.061 → 0.135 | ok |
| crossmae/s3_enc0.1 | b_plain | 100% | 2 | learned | 2.876 | 0.940 | 1.055 | 1.183 → 0.148 | ok |
| crossmae/s3_enc0.1 | c_ego | 100% | 2 | learned | 2.892 | 0.940 | 1.025 | 1.175 → 0.125 | ok |
| crossmae/s3_step2000 | b_plain | 100% | 0 | learned | 2.908 | 0.960 | 1.079 | 1.213 → 0.164 | ok |
| crossmae/s3_step2000 | c_ego | 100% | 0 | learned | 2.928 | 0.959 | 1.108 | 1.260 → 0.179 | ok |
| crossmae/s3_step2000 | b_plain | 100% | 1 | learned | 2.766 | 0.989 | 1.100 | 1.044 → 0.142 | ok |
| crossmae/s3_step2000 | c_ego | 100% | 1 | learned | 2.782 | 1.005 | 1.126 | 1.054 → 0.164 | ok |
| crossmae/s3_step2000 | b_plain | 100% | 2 | learned | 2.915 | 0.941 | 1.074 | 1.209 → 0.131 | ok |
| crossmae/s3_step2000 | c_ego | 100% | 2 | learned | 2.846 | 0.918 | 1.079 | 1.159 → 0.133 | ok |
| crossmae/s4_d2e_enc0.1 | d2e_plain | 100% | 0 | learned | 2.910 | 0.957 | 1.075 | 1.229 → 0.149 | ok |
| crossmae/s4_d2e_enc0.1 | d2e_weighted | 100% | 0 | learned | 2.962 | 0.950 | 1.088 | 1.270 → 0.170 | ok |
| crossmae/s4_d2e_enc0.1 | d2e_plain | 100% | 1 | learned | 2.730 | 0.966 | 1.069 | 1.030 → 0.126 | ok |
| crossmae/s4_d2e_enc0.1 | d2e_weighted | 100% | 1 | learned | 2.733 | 0.967 | 1.057 | 1.038 → 0.124 | ok |
| crossmae/s4_d2e_enc0.1 | d2e_plain | 100% | 2 | learned | 2.879 | 0.972 | 1.098 | 1.163 → 0.148 | ok |
| crossmae/s4_d2e_enc0.1 | d2e_weighted | 100% | 2 | learned | 2.909 | 0.955 | 1.052 | 1.184 → 0.137 | ok |
| crossmae/s4_d2e_full | d2e_plain | 100% | 0 | learned | 2.925 | 0.934 | 1.065 | 1.239 → 0.156 | ok |
| crossmae/s4_d2e_full | d2e_weighted | 100% | 0 | learned | 2.950 | 0.969 | 1.099 | 1.292 → 0.158 | ok |
| crossmae/s4_d2e_full | d2e_plain | 100% | 1 | learned | 2.765 | 0.995 | 1.100 | 1.046 → 0.149 | ok |
| crossmae/s4_d2e_full | d2e_weighted | 100% | 1 | learned | 2.738 | 1.018 | 1.051 | 1.036 → 0.129 | ok |
| crossmae/s4_d2e_full | d2e_plain | 100% | 2 | learned | 2.894 | 0.977 | 1.101 | 1.183 → 0.148 | ok |
| crossmae/s4_d2e_full | d2e_weighted | 100% | 2 | learned | 2.880 | 0.972 | 1.043 | 1.176 → 0.132 | ok |
| crossmae/s4_e2d_enc0.1 | e2d_plain | 100% | 0 | learned | 2.904 | 0.930 | 1.060 | 1.219 → 0.150 | ok |
| crossmae/s4_e2d_enc0.1 | e2d_weighted | 100% | 0 | learned | 2.936 | 0.939 | 1.083 | 1.243 → 0.151 | ok |
| crossmae/s4_e2d_enc0.1 | e2d_plain | 100% | 1 | learned | 2.764 | 0.984 | 1.081 | 1.055 → 0.137 | ok |
| crossmae/s4_e2d_enc0.1 | e2d_weighted | 100% | 1 | learned | 2.752 | 0.973 | 1.089 | 1.057 → 0.130 | ok |
| crossmae/s4_e2d_enc0.1 | e2d_plain | 100% | 2 | learned | 2.888 | 0.947 | 1.048 | 1.182 → 0.132 | ok |
| crossmae/s4_e2d_enc0.1 | e2d_weighted | 100% | 2 | learned | 2.861 | 0.949 | 1.053 | 1.175 → 0.146 | ok |
| crossmae/s4_e2d_full | e2d_plain | 100% | 0 | learned | 2.905 | 0.960 | 1.097 | 1.221 → 0.155 | ok |
| crossmae/s4_e2d_full | e2d_weighted | 100% | 0 | learned | 2.944 | 0.932 | 1.084 | 1.252 → 0.148 | ok |
| crossmae/s4_e2d_full | e2d_plain | 100% | 1 | learned | 2.762 | 0.961 | 1.073 | 1.048 → 0.137 | ok |
| crossmae/s4_e2d_full | e2d_weighted | 100% | 1 | learned | 2.781 | 0.968 | 1.067 | 1.058 → 0.125 | ok |
| crossmae/s4_e2d_full | e2d_plain | 100% | 2 | learned | 2.863 | 0.932 | 1.074 | 1.174 → 0.137 | ok |
| crossmae/s4_e2d_full | e2d_weighted | 100% | 2 | learned | 2.918 | 0.927 | 1.066 | 1.233 → 0.126 | ok |

### Behaviour counts

| setting | experiment | labels | diverged | never started | spiked | learned |
|---|---|---|---|---|---|---|
| crossmae/fix_grid | a_imagenet | 10% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | a_imagenet | 100% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | b_plain | 10% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | b_plain | 100% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | c_ego | 10% | 0 | 0 | 0 | 3 |
| crossmae/fix_grid | c_ego | 100% | 0 | 0 | 0 | 3 |
| crossmae/s3_enc0.1 | b_plain | 100% | 0 | 0 | 0 | 3 |
| crossmae/s3_enc0.1 | c_ego | 100% | 0 | 0 | 0 | 3 |
| crossmae/s3_step2000 | b_plain | 100% | 0 | 0 | 0 | 3 |
| crossmae/s3_step2000 | c_ego | 100% | 0 | 0 | 0 | 3 |
| crossmae/s4_d2e_enc0.1 | d2e_plain | 100% | 0 | 0 | 0 | 3 |
| crossmae/s4_d2e_enc0.1 | d2e_weighted | 100% | 0 | 0 | 0 | 3 |
| crossmae/s4_d2e_full | d2e_plain | 100% | 0 | 0 | 0 | 3 |
| crossmae/s4_d2e_full | d2e_weighted | 100% | 0 | 0 | 0 | 3 |
| crossmae/s4_e2d_enc0.1 | e2d_plain | 100% | 0 | 0 | 0 | 3 |
| crossmae/s4_e2d_enc0.1 | e2d_weighted | 100% | 0 | 0 | 0 | 3 |
| crossmae/s4_e2d_full | e2d_plain | 100% | 0 | 0 | 0 | 3 |
| crossmae/s4_e2d_full | e2d_weighted | 100% | 0 | 0 | 0 | 3 |

## Difference to a_imagenet (same seed and labels, healthy runs only)

Positive = better than the reference.

| setting | experiment | labels | pairs | Δ val mAP50 | Δ Vehicle AP | beats ref |
|---|---|---|---|---|---|---|
| crossmae/fix_grid | b_plain | 10% | 2 | -0.011 ± 0.002 | -0.024 ± 0.019 | 0/2 |
| crossmae/fix_grid | b_plain | 100% | 3 | -0.055 ± 0.006 | -0.045 ± 0.004 | 0/3 |
| crossmae/fix_grid | c_ego | 10% | 2 | -0.004 ± 0.001 | -0.000 ± 0.016 | 0/2 |
| crossmae/fix_grid | c_ego | 100% | 3 | -0.043 ± 0.013 | -0.020 ± 0.009 | 0/3 |
| crossmae/s3_enc0.1 | b_plain | 100% | 3 | -0.023 ± 0.005 | -0.009 ± 0.008 | 0/3 |
| crossmae/s3_enc0.1 | c_ego | 100% | 3 | -0.028 ± 0.010 | -0.017 ± 0.003 | 0/3 |
| crossmae/s3_step2000 | b_plain | 100% | 3 | -0.019 ± 0.005 | -0.030 ± 0.005 | 0/3 |
| crossmae/s3_step2000 | c_ego | 100% | 3 | -0.040 ± 0.014 | -0.041 ± 0.008 | 0/3 |
| crossmae/s4_d2e_enc0.1 | d2e_plain | 100% | 3 | -0.033 ± 0.015 | -0.023 ± 0.013 | 0/3 |
| crossmae/s4_d2e_enc0.1 | d2e_weighted | 100% | 3 | -0.031 ± 0.020 | -0.013 ± 0.008 | 0/3 |
| crossmae/s4_d2e_full | d2e_plain | 100% | 3 | -0.041 ± 0.004 | -0.017 ± 0.009 | 0/3 |
| crossmae/s4_d2e_full | d2e_weighted | 100% | 3 | -0.056 ± 0.016 | -0.039 ± 0.010 | 0/3 |
| crossmae/s4_e2d_enc0.1 | e2d_plain | 100% | 3 | -0.024 ± 0.006 | -0.005 ± 0.009 | 0/3 |
| crossmae/s4_e2d_enc0.1 | e2d_weighted | 100% | 3 | -0.013 ± 0.005 | -0.004 ± 0.002 | 0/3 |
| crossmae/s4_e2d_full | e2d_plain | 100% | 3 | -0.034 ± 0.002 | -0.025 ± 0.010 | 0/3 |
| crossmae/s4_e2d_full | e2d_weighted | 100% | 3 | -0.028 ± 0.012 | -0.031 ± 0.009 | 0/3 |
