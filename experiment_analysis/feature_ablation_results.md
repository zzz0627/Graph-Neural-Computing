# Experiment Results

## Link Prediction

| dataset | model | feature_setting | num_runs | seed | val_AP | val_AUC | new_node_val_AP | new_node_val_AUC | test_AP | test_AUC | new_node_test_AP | new_node_test_AUC | notes |
|---------|-------|-----------------|----------|------|--------|---------|-----------------|------------------|---------|----------|------------------|-------------------|-------|
| wikipedia | DyGFormer | Base | 1 | 0 | 0.9895 | 0.9882 | 0.9843 | 0.9827 | 0.9874 | 0.9858 | 0.9809 | 0.9783 | 1 epoch smoke test |
| wikipedia | DyGFormer | Base+Style | 1 | 0 | 0.9892 | 0.9881 | 0.9842 | 0.9823 | 0.9877 | 0.9860 | 0.9812 | 0.9784 | 1 epoch smoke test, style_dim=32 |
| wikipedia | DyGFormer | Base+Personality | 1 | 0 | 0.9892 | 0.9879 | 0.9851 | 0.9827 | 0.9867 | 0.9851 | 0.9804 | 0.9777 | 1 epoch smoke test, personality_dim=5 |
| wikipedia | DyGFormer | Base+Style+Personality | 1 | 0 | 0.9897 | 0.9884 | 0.9858 | 0.9835 | 0.9874 | 0.9855 | 0.9817 | 0.9786 | 1 epoch, style=32d+personality=5d |
| wikipedia | DyGFormer | Base+Topic | 1 | 0 | | | | | 0.9868 | 0.9852 | 0.9808 | 0.9781 | 1 epoch, NMF topic_dim=10 |
| wikipedia | DyGFormer | Base+Style+Pers+Topic | 1 | 0 | | | | | 0.9873 | 0.9856 | 0.9814 | 0.9784 | 1 epoch, 6ch+sidecar |
| wikipedia | DyGFormer | Base+Social | 1 | 0 | | | | | 0.9883 | 0.9872 | 0.9835 | 0.9814 | 1 epoch, social_dim=10 |
| wikipedia | DyGFormer | Full | 1 | 0 | | | | | 0.9888 | 0.9871 | 0.9842 | 0.9815 | 1 epoch, 7ch+sidecar, all features |
| reddit | DyGFormer | Base+Style | 1 | 0 | 0.9891 | 0.9877 | 0.9828 | 0.9799 | 0.9896 | 0.9881 | 0.9845 | 0.9819 | 1 epoch smoke test, style_dim=32 |
| wikipedia | DyGFormer | Base | 5 | 0-4 | | | | | | | | | |
| wikipedia | DyGFormer | Base+Style | 5 | 0-4 | | | | | | | | | |
| reddit | DyGFormer | Base | 5 | 0-4 | | | | | | | | | |
| reddit | DyGFormer | Base+Style | 5 | 0-4 | | | | | | | | | |
| wikipedia | DyGFormer | Base+Personality | 5 | 0-4 | | | | | | | | | |
| wikipedia | DyGFormer | Base+Topic | 5 | 0-4 | | | | | | | | | |
| wikipedia | DyGFormer | Base+Social | 5 | 0-4 | | | | | | | | | |
| wikipedia | DyGFormer | Base+Style+Personality | 5 | 0-4 | | | | | | | | | |
| wikipedia | DyGFormer | Base+Style+Personality+Topic | 5 | 0-4 | | | | | | | | | |
| wikipedia | DyGFormer | Full | 5 | 0-4 | | | | | | | | | |

## Node Classification

| dataset | model | feature_setting | num_runs | seed | val_roc_auc | test_roc_auc | notes |
|---------|-------|-----------------|----------|------|-------------|--------------|-------|
| wikipedia | DyGFormer | Base | 5 | 0-4 | | | |
| wikipedia | DyGFormer | Base+Style | 5 | 0-4 | | | |
| reddit | DyGFormer | Base | 5 | 0-4 | | | |
| reddit | DyGFormer | Base+Style | 5 | 0-4 | | | |

## Feature Configuration Summary

| feature | type | proxy_method | input_source | output_dim | variance_explained |
|---------|------|-------------|--------------|------------|-------------------|
| style | proxy (PCA) | PCA on 172-d LIWC edge features | ml_{dataset}.npy | 32 | wiki=78.1%, reddit=69.1% |
| personality | proxy (PCA) | per-node mean of LIWC edge features -> PCA | ml_{dataset}.npy + ml_{dataset}.csv | 5 | wiki=74.7%, reddit=69.1% |
| topic | proxy (NMF) | NMF on shifted LIWC edge features, row-normalized | ml_{dataset}.npy | 10 | N/A (reconstruction err: wiki=3239, reddit=7671) |
| social | proxy (structural) | time-aware per-edge structural stats (degree, diversity, recency, rate, repeat), z-normalized | ml_{dataset}.csv | 10 (5 src + 5 dst) | N/A |

## Command Reference

```bash
# Generate style features
python -m features.style_extractor --dataset_name wikipedia --style_dim 32
python -m features.style_extractor --dataset_name reddit --style_dim 32

# Link Prediction: Base
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0

# Link Prediction: Base + Style
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0 --use_style

# Link Prediction: Base + Personality
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0 --use_personality

# Link Prediction: Base + Style + Personality
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0 --use_style --use_personality

# Node Classification: Base + Style (requires pretrained LP model with --use_style)
python train_node_classification.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0 --use_style
```
