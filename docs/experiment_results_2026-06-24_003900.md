# DyGFormer Social Feature Experiment Results

- Report generated: 2026-06-24 00:39:00 CST (Asia/Shanghai)
- Experiment completed: 2026-06-07 06:29:37 CST (Asia/Shanghai)
- Task: Step 3 GPU smoke validation followed by 5-seed link prediction (LP)
- Datasets: `wikipedia`, `reddit`
- Variants: `Base`, `v1_trainfit`, `v2_trainfit`
- Device: `cuda:0` on NVIDIA GeForce RTX 4090

## Smoke Validation

All six Step 3 smoke runs completed on GPU with finite metrics. Channel loading was verified from run logs.

| Dataset | Variant | DyGFormer Channels | Extra Feature Dim | Test AP | New-node Test AP |
|---|---:|---:|---:|---:|---:|
| wikipedia | Base | 4 | - | 0.9886 | 0.9832 |
| wikipedia | v1_trainfit | 5 | social=10 | 0.9902 | 0.9858 |
| wikipedia | v2_trainfit | 5 | social=20 | 0.9905 | 0.9861 |
| reddit | Base | 4 | - | 0.9907 | 0.9859 |
| reddit | v1_trainfit | 5 | social=10 | 0.9923 | 0.9883 |
| reddit | v2_trainfit | 5 | social=20 | 0.9922 | 0.9881 |

Notes:

- `Base` uses 4 DyGFormer channels.
- `v1_trainfit` and `v2_trainfit` use `4 + 1 = 5` DyGFormer channels.
- The social feature vector dimensions are 10 for v1 and 20 for v2. The channel count is therefore 5, not 14.
- All smoke metrics were finite; the single-run standard deviation is `nan` only because each smoke used one run.

## 5-Seed LP Results

All 5-seed LP result JSON files were verified as present and finite.

| Dataset | Variant | Runs | Test AP | Test AUC | New-node Test AP | New-node Test AUC |
|---|---:|---:|---:|---:|---:|---:|
| wikipedia | Base | 5 | 0.9904 +/- 0.0002 | 0.9894 +/- 0.0002 | 0.9858 +/- 0.0002 | 0.9842 +/- 0.0001 |
| wikipedia | v1_trainfit | 5 | 0.9918 +/- 0.0002 | 0.9912 +/- 0.0001 | 0.9878 +/- 0.0003 | 0.9867 +/- 0.0002 |
| wikipedia | v2_trainfit | 5 | 0.9918 +/- 0.0002 | 0.9912 +/- 0.0002 | 0.9875 +/- 0.0003 | 0.9866 +/- 0.0004 |
| reddit | Base | 5 | 0.9922 +/- 0.0001 | 0.9915 +/- 0.0001 | 0.9882 +/- 0.0002 | 0.9869 +/- 0.0002 |
| reddit | v1_trainfit | 5 | 0.9930 +/- 0.0000 | 0.9925 +/- 0.0001 | 0.9895 +/- 0.0001 | 0.9884 +/- 0.0002 |
| reddit | v2_trainfit | 5 | 0.9929 +/- 0.0001 | 0.9924 +/- 0.0001 | 0.9894 +/- 0.0003 | 0.9884 +/- 0.0001 |

## Paired v2-v1 Gate

Gate criterion: v2 should improve over v1 on same-seed paired test AP, with broad positive-seed support and no systematic new-node degradation.

| Dataset | Test AP Mean Delta | Positive Seeds | New-node AP Mean Delta | Positive New-node Seeds | Gate |
|---|---:|---:|---:|---:|---:|
| wikipedia | -0.000040 | 2/5 | -0.000260 | 1/5 | Fail |
| reddit | -0.000100 | 0/5 | -0.000100 | 2/5 | Fail |

Per-seed paired deltas:

| Dataset | Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|---:|
| wikipedia | Test AP | +0.0003 | +0.0000 | +0.0001 | -0.0001 | -0.0005 |
| wikipedia | New-node Test AP | -0.0003 | -0.0001 | -0.0005 | +0.0005 | -0.0009 |
| reddit | Test AP | -0.0002 | +0.0000 | -0.0002 | -0.0001 | +0.0000 |
| reddit | New-node Test AP | -0.0003 | +0.0002 | -0.0004 | -0.0001 | +0.0001 |

## Artifact Locations

- Base results: `experiments/feature_base_ref/saved_results/DyGFormer/`
- v1 results: `experiments/feature_social_v1_trainfit/saved_results/DyGFormer/`
- v2 results: `experiments/feature_social_v2_trainfit/saved_results/DyGFormer/`
- Smoke logs:
  - `experiments/feature_base_ref/analysis/smoke_<dataset>_base.stdout.log`
  - `experiments/feature_social_v1_trainfit/analysis/smoke_<dataset>_v1_trainfit.stdout.log`
  - `experiments/feature_social_v2_trainfit/analysis/smoke_<dataset>_v2_trainfit.stdout.log`
- 5-seed reddit parallel logs:
  - `experiments/feature_base_ref/analysis/lp5_reddit_base_seed<seed>.stdout.log`
  - `experiments/feature_social_v1_trainfit/analysis/lp5_reddit_v1_trainfit_seed<seed>.stdout.log`
  - `experiments/feature_social_v2_trainfit/analysis/lp5_reddit_v2_trainfit_seed<seed>.stdout.log`

## Conclusion

The smoke validation and 5-seed LP experiments completed successfully on `cuda:0`, and all checked metrics were finite. However, `v2_trainfit` did not beat `v1_trainfit` under the paired 5-seed LP gate on either `wikipedia` or `reddit`.
