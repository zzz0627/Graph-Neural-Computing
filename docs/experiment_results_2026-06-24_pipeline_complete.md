# DyGFormer Social Feature Pipeline Results

- Report generated: 2026-06-24 11:46 CST (Asia/Shanghai)
- Command sequence: `pipeline.sh features`, `pipeline.sh smoke`, verified complete LP artifacts, `pipeline.sh nc`, `pipeline.sh analyze`, `pipeline.sh status`
- Environment: Ubuntu 22.04 server, NVIDIA GeForce RTX 4090, conda env `dyg`
- Datasets: `wikipedia`, `reddit`
- Variants:
  - `base`: DyGFormer baseline
  - `v1_trainfit`: DyGFormer + Social v1 train-fit features
  - `v2_trainfit`: DyGFormer + Social v2 train-fit features

## Artifact Completeness

Final `pipeline.sh status` confirmed all expected 5-seed result files are present.

| Variant | Dataset | LP JSON files | NC JSON files |
|---|---|---:|---:|
| base | wikipedia | 5 | 5 |
| base | reddit | 5 | 5 |
| v1_trainfit | wikipedia | 5 | 5 |
| v1_trainfit | reddit | 5 | 5 |
| v2_trainfit | wikipedia | 5 | 5 |
| v2_trainfit | reddit | 5 | 5 |

## Link Prediction

| Dataset | Variant | Runs | Test AP | Test AUC | New-node Test AP | New-node Test AUC |
|---|---|---:|---:|---:|---:|---:|
| wikipedia | base | 5 | 0.9904 +/- 0.0002 | 0.9894 +/- 0.0002 | 0.9858 +/- 0.0002 | 0.9842 +/- 0.0001 |
| wikipedia | v1_trainfit | 5 | 0.9915 +/- 0.0007 | 0.9908 +/- 0.0009 | 0.9873 +/- 0.0009 | 0.9862 +/- 0.0010 |
| wikipedia | v2_trainfit | 5 | 0.9915 +/- 0.0006 | 0.9907 +/- 0.0009 | 0.9872 +/- 0.0007 | 0.9861 +/- 0.0010 |
| reddit | base | 5 | 0.9919 +/- 0.0007 | 0.9910 +/- 0.0009 | 0.9877 +/- 0.0010 | 0.9863 +/- 0.0014 |
| reddit | v1_trainfit | 5 | 0.9928 +/- 0.0003 | 0.9923 +/- 0.0004 | 0.9892 +/- 0.0005 | 0.9881 +/- 0.0007 |
| reddit | v2_trainfit | 5 | 0.9928 +/- 0.0003 | 0.9923 +/- 0.0004 | 0.9892 +/- 0.0006 | 0.9881 +/- 0.0006 |

## Node Classification

| Dataset | Variant | Runs | Test ROC-AUC |
|---|---|---:|---:|
| wikipedia | base | 5 | 0.8714 +/- 0.0060 |
| wikipedia | v1_trainfit | 5 | 0.8840 +/- 0.0118 |
| wikipedia | v2_trainfit | 5 | 0.8788 +/- 0.0209 |
| reddit | base | 5 | 0.6845 +/- 0.0094 |
| reddit | v1_trainfit | 5 | 0.7002 +/- 0.0194 |
| reddit | v2_trainfit | 5 | 0.7181 +/- 0.0086 |

## v2 vs v1 Paired Delta

| Dataset | Task | Metric | Mean Delta | Positive Seeds | 95% CI | p-value |
|---|---|---|---:|---:|---:|---:|
| wikipedia | LP | Test AP | -0.000040 | 2/5 | [-0.000408, 0.000328] | 0.77805 |
| wikipedia | LP | New-node Test AP | -0.000140 | 2/5 | [-0.000851, 0.000571] | 0.613709 |
| wikipedia | NC | Test ROC-AUC | -0.005180 | 2/5 | [-0.026641, 0.016281] | 0.539466 |
| reddit | LP | Test AP | -0.000080 | 0/5 | [-0.000184, 0.000024] | 0.0993007 |
| reddit | LP | New-node Test AP | -0.000080 | 2/5 | [-0.000376, 0.000216] | 0.495354 |
| reddit | NC | Test ROC-AUC | 0.017920 | 4/5 | [-0.008694, 0.044534] | 0.13491 |

## Notes

- Social v1/v2 improve LP over the baseline on both datasets, but v2 is effectively tied with v1 in LP.
- For NC, v1 improves over base on both datasets. v2 is slightly below v1 on wikipedia but improves reddit NC to `0.7181 +/- 0.0086`.
- The repository stores lightweight JSON results and manifests. Large generated artifacts such as checkpoints, `.npy` feature banks, and verbose stdout logs remain ignored and reproducible through `pipeline.sh`.

