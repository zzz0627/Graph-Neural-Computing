# Feature Proxy Improvement Handoff

Last updated: 2026-04-28

This document is intended for the next Codex instance. It summarizes the
Wikipedia/Reddit Style, Personality, Topic, and Social feature ablation state,
the current diagnosis, and the concrete next steps. Read this file before
touching code.

## 1. Scope

The relevant experiment line is the DyGFormer multi-feature extension on
Wikipedia and Reddit, not the newer mooc/temfin baseline experiments.

Relevant files:

- `experiment_analysis/feature_ablation_analysis.md`: main result analysis.
- `experiment_analysis/feature_ablation_all_results.csv`: per-seed result table.
- `experiment_analysis/feature_ablation_results.md`: early/manual result notes.
- `features/`: feature extraction code.
- `models/DyGFormer.py`: feature-channel and sidecar fusion integration.
- `train_link_prediction.py`, `train_node_classification.py`: training entry points.
- `evaluate_link_prediction.py`, `evaluate_node_classification.py`: evaluation entry points.
- `saved_results/DyGFormer/{wikipedia,reddit}/`: result JSON files.
- `saved_models/DyGFormer/{wikipedia,reddit}/`: checkpoints.

Do not assume the feature names are semantically faithful. In particular,
Style, Personality, and Topic are proxy labels built from LIWC-derived arrays,
not direct measurements from raw text.

## 2. Current Experimental Facts

All numbers below are 5-run means from `feature_ablation_analysis.md` and
`feature_ablation_all_results.csv`.

### Link Prediction

Wikipedia:

| Setting | Test AP | Test AUC | New-node Test AP | Delta AP vs Base |
|---|---:|---:|---:|---:|
| Base | 0.9904 +/- 0.0002 | 0.9894 +/- 0.0002 | 0.9858 +/- 0.0002 | -- |
| +Style | 0.9902 +/- 0.0003 | 0.9890 +/- 0.0002 | 0.9853 +/- 0.0002 | -0.0002 |
| +Personality | 0.9904 +/- 0.0001 | 0.9895 +/- 0.0002 | 0.9856 +/- 0.0002 | +0.0000 |
| +Topic | 0.9903 +/- 0.0003 | 0.9893 +/- 0.0003 | 0.9858 +/- 0.0003 | -0.0001 |
| +Social | 0.9918 +/- 0.0001 | 0.9912 +/- 0.0001 | 0.9878 +/- 0.0004 | +0.0014 |
| Full | 0.9917 +/- 0.0002 | 0.9909 +/- 0.0002 | 0.9875 +/- 0.0004 | +0.0013 |

Reddit:

| Setting | Test AP | Test AUC | New-node Test AP | Delta AP vs Base |
|---|---:|---:|---:|---:|
| Base | 0.9922 +/- 0.0001 | 0.9915 +/- 0.0001 | 0.9882 +/- 0.0002 | -- |
| +Style | 0.9920 +/- 0.0001 | 0.9912 +/- 0.0001 | 0.9879 +/- 0.0002 | -0.0002 |
| +Personality | 0.9924 +/- 0.0001 | 0.9917 +/- 0.0001 | 0.9886 +/- 0.0001 | +0.0002 |
| +Topic | 0.9921 +/- 0.0001 | 0.9914 +/- 0.0001 | 0.9882 +/- 0.0002 | -0.0001 |
| +Social | 0.9930 +/- 0.0001 | 0.9925 +/- 0.0001 | 0.9895 +/- 0.0003 | +0.0008 |
| Full | 0.9930 +/- 0.0001 | 0.9925 +/- 0.0001 | 0.9895 +/- 0.0002 | +0.0008 |

LP conclusion: Social explains essentially all Full-model improvement.
Style and Topic are neutral to slightly negative. Personality is neutral on
Wikipedia and mildly positive on Reddit.

### Node Classification

Wikipedia:

| Setting | Test ROC-AUC | Delta vs Base |
|---|---:|---:|
| Base | 0.8714 +/- 0.0060 | -- |
| +Style | 0.8720 +/- 0.0109 | +0.0006 |
| +Personality | 0.8552 +/- 0.0400 | -0.0162 |
| +Topic | 0.8776 +/- 0.0217 | +0.0062 |
| +Social | 0.8806 +/- 0.0197 | +0.0092 |
| Full | 0.8755 +/- 0.0119 | +0.0041 |

Reddit:

| Setting | Test ROC-AUC | Delta vs Base |
|---|---:|---:|
| Base | 0.6837 +/- 0.0090 | -- |
| +Style | 0.6703 +/- 0.0257 | -0.0134 |
| +Personality | 0.6431 +/- 0.0147 | -0.0406 |
| +Topic | 0.6701 +/- 0.0455 | -0.0136 |
| +Social | 0.6983 +/- 0.0304 | +0.0146 |
| Full | 0.6453 +/- 0.0463 | -0.0384 |

NC conclusion: variance is much larger than LP. Still, Social is the best
single feature in both datasets. Personality is harmful, especially on Reddit.
Full is weaker than Social alone because noisy proxy features dilute or
override the useful Social signal.

## 3. Current Feature Definitions

### Style

Implementation: `features/style_extractor.py`

- Input: `processed_data/{dataset}/ml_{dataset}.npy`, the existing 172-d LIWC
  edge feature matrix.
- Method: PCA on all actual edges, default `style_dim=32`.
- Output: `style_v1.npy`, shape `(num_edges + 1, 32)`.
- Model use: extra edge channel.

Problem: this is a linear projection of information already present in the
baseline edge channel. It is a redundant view, not a new style signal from raw
text.

### Personality

Implementation: `features/personality_extractor.py`

- Input: same 172-d LIWC edge features plus edge list.
- Method: aggregate mean LIWC vector per node, then PCA to 5 dimensions.
- Output: `personality_v1.npy`, shape `(num_nodes + 1, 5)`.
- Model use: node sidecar fused after DyGFormer output layer.

Problems:

- It is not a validated Big Five feature.
- It is static and computed from all interactions, including future edges.
- It is injected late, after the backbone has already produced embeddings.
- In node classification, the backbone is frozen, so noise goes directly into
  the MLP classifier.

### Topic

Implementation: `features/topic_extractor.py`

- Input: same 172-d LIWC edge features.
- Method: shift columns nonnegative, run NMF, row-normalize to topic-like
  proportions, default `topic_dim=10`.
- Output: `topic_v1.npy`, shape `(num_edges + 1, 10)`.
- Model use: extra edge channel.

Problem: this is another decomposition of the same LIWC feature matrix. It is
not true topic modeling from raw documents. It should be described as
NMF-based content decomposition, not as semantic topic modeling.

### Social

Implementation: `features/social_extractor.py`

- Input: edge list only.
- Method: for each edge `(u, v, t)`, compute only from interactions before
  `t`: degree, unique-neighbor count, recency, activity rate, and repeat ratio
  for src and dst.
- Output: `social_v1.npy`, shape `(num_edges + 1, 10)`.
- Model use: extra edge channel.

Strength: this is time-aware dynamic structural signal. It is not redundant
with LIWC, and it aligns with both tasks:

- Link prediction predicts future edges, so local historical structure helps.
- Node classification labels are plausibly correlated with behavior/activity
  patterns.

## 4. Diagnosis

The poor behavior of Style, Topic, and Personality is not primarily explained
by an obvious indexing or shape bug.

Observed sanity checks:

- Feature files exist for both datasets:
  `style_v1.npy`, `personality_v1.npy`, `topic_v1.npy`, `social_v1.npy`.
- Row 0 padding is zero.
- Edge-level features have `num_edges + 1` rows.
- Personality has `num_nodes + 1` rows.
- No obvious NaNs or constant columns were found in the feature arrays.

The likely issue is proxy quality and redundancy:

- Style and Topic are derived from the same LIWC edge features that the
  baseline already consumes. They add parameters more than they add
  information.
- Personality is a weak static node proxy and has a risky late-fusion path.
- Social is the only feature family that contributes an independent source of
  signal.

Lightweight diagnostic statistics from the current workspace support this:

| Dataset | Feature | Simple label AUC diagnostic |
|---|---|---:|
| Wikipedia | Style | 0.5675 |
| Wikipedia | Topic | 0.6486 |
| Wikipedia | Social | 0.8848 |
| Wikipedia | Personality | 0.6903 |
| Reddit | Style | 0.4958 |
| Reddit | Topic | 0.5376 |
| Reddit | Social | 0.6562 |
| Reddit | Personality | 0.6844 |

This diagnostic is not a formal temporal evaluation. It is only evidence that
Social carries stronger label-related signal than Style/Topic, and that
Personality can be label-related while still hurting the current frozen-backbone
NC pipeline.

## 5. Important Caveats Before New Experiments

Several current proxy extractors use full-dataset preprocessing:

- Style PCA is fit on all edges.
- Topic NMF is fit on all edges.
- Personality aggregates all edges for each node.
- Social uses causal historical counts per edge, but its final z-score
  normalization uses all edges.

For a stricter paper-quality pipeline, fit preprocessing only on training
edges, then transform validation/test edges with the fitted transform and
train-derived normalization statistics. This may reduce absolute performance.
Do not compare train-only preprocessing runs directly against old full-data
preprocessing runs without labeling the protocol change.

## 6. Recommended Next Steps

### Step 1: Freeze the current conclusion in writing

Goal: avoid over-claiming weak proxy features.

Action:

- Treat Social as the main useful feature.
- Treat Style, Topic, and Personality as proxy ablation baselines.
- Avoid claims that they are faithful language style, true personality, or
  semantic topic features.

Suggested wording:

- Style: "PCA-based linguistic variation channel".
- Personality: "user-level aggregated linguistic proxy".
- Topic: "NMF-based content decomposition".
- Social: "time-aware dynamic structural features" or "dynamic social context".

Verify:

- Any paper/report text should say Social is the stable contribution.
- Any claim about Style/Topic/Personality should include the word "proxy" or
  otherwise make the limitation explicit.

### Step 2: Add train-only feature preprocessing

Goal: remove preprocessing leakage and make future experiments cleaner.

Action:

- Add a new feature version, e.g. `v2_trainfit`, rather than overwriting
  `*_v1.npy`.
- For Style: fit PCA on training edges only, then transform all edges.
- For Topic: fit NMF on training edges only. Use train-set column shifts and
  fitted NMF to transform all edges.
- For Social: keep causal edge statistics, but compute normalization mean/std
  on training edges only.
- For Personality: avoid all-time node aggregation. Either compute train-only
  node profiles or make a time-aware per-edge user-profile feature.

Verify:

- Output files keep row 0 zero.
- Edge-level files have shape `(num_edges + 1, dim)`.
- Node-level files have shape at least `(max_node_id + 1, dim)`.
- Re-run a one-seed smoke test before launching 5-run sweeps.

### Step 3: Prioritize Social enrichment over more LIWC projections

Goal: build on the only consistently positive signal.

Action candidates:

- Add rolling-window versions of the current Social stats.
- Split stats by src-side and dst-side behavior more explicitly.
- Add interaction burstiness or inter-event-time summary features.
- Add time-decayed degree and time-decayed unique neighbor count.
- Add per-node historical positive-label exposure only if it is causal and
  label availability is justified.

Avoid expensive graph-global features first. PageRank, betweenness, and
community detection may be useful later, but they introduce complexity and
potential temporal leakage.

Verify:

- New Social features must be computed with `timestamp < t`.
- Compare `Base`, `Social v1`, and `Social v2` first, before running all
  feature combinations.
- Primary success metric: LP AP/AUC and NC ROC-AUC improve over Social v1,
  not just over Base.

### Step 4: Rework Personality only if raw text or a better temporal proxy is available

Goal: avoid spending effort on a weak static proxy.

Preferred options:

1. If raw text is available, use a real text encoder or a validated personality
   predictor.
2. If raw text is not available, rename the feature and make it time-aware:
   for each edge, aggregate only historical LIWC of the node before `t`.
3. Inject the resulting profile earlier as a node/edge channel, not only as a
   post-output sidecar.

Verify:

- Start with LP one-seed and NC one-seed smoke tests.
- For NC, compare frozen-backbone vs partially unfrozen backbone. The current
  frozen setting may exaggerate sidecar noise.
- Do not call the feature "personality" in final claims unless it is validated
  against an external personality signal.

### Step 5: Deprioritize Style and Topic unless raw text appears

Goal: avoid repeated experiments on redundant projections.

Action:

- Keep current Style/Topic results as negative/neutral ablations.
- Do not spend the next experiment cycle on PCA/NMF dimension sweeps unless
  there is a specific hypothesis.
- If raw text becomes available, replace these proxies with real text-derived
  features:
  - Style: learned style encoder or stylometric feature extractor.
  - Topic: BERTopic, LDA/NMF on actual text, or supervised content encoder.

Verify:

- Any new Style/Topic experiment must define what new information it adds
  beyond the original 172-d LIWC edge feature.

## 7. Suggested Experiment Order

Run experiments in this order to avoid wasting GPU time:

1. `Social v2` one-seed LP smoke test on Wikipedia.
2. `Social v2` one-seed LP smoke test on Reddit.
3. If both are sane, run 5-seed LP for `Base`, `Social v1`, and `Social v2`.
4. Run NC only after LP checkpoints exist for the exact same feature tag.
5. Only after Social v2 is understood, test a time-aware user profile feature.
6. Do not run Full combinations until each individual feature has a positive
   standalone reason to exist.

Success criteria:

- LP: `Social v2` should beat `Social v1` or at least match it with cleaner
  train-only preprocessing.
- NC: improvements must be interpreted with seed variance. Prefer changes that
  improve the mean and do not rely on one outlier seed.
- Reporting: every table must identify the feature version and preprocessing
  protocol.

## 8. Command References

Existing v1 feature generation examples:

```bash
python -m features.style_extractor --dataset_name wikipedia --style_dim 32
python -m features.personality_extractor --dataset_name wikipedia --personality_dim 5
python -m features.topic_extractor --dataset_name wikipedia --topic_dim 10
python -m features.social_extractor --dataset_name wikipedia
```

Existing LP training examples:

```bash
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0 --use_social
python train_link_prediction.py --dataset_name reddit --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0 --use_social
```

Existing NC training examples:

```bash
python train_node_classification.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0 --use_social
python train_node_classification.py --dataset_name reddit --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0 --use_social
```

Important: NC loads the matching LP checkpoint. The feature flags and
`feature_tag` must match between LP and NC.

## 9. Next Codex Checklist

Before editing code:

- Read this file and `experiment_analysis/feature_ablation_analysis.md`.
- Confirm whether the user wants paper/report wording only, or actual feature
  implementation changes.
- If implementing changes, create new versioned feature outputs; do not
  overwrite `*_v1.npy`.
- Keep changes surgical. Start with Social or train-only preprocessing, not a
  full rewrite.

Do not:

- Claim Style/Topic/Personality are validated semantic features.
- Compare train-only preprocessing results against v1 without noting protocol
  differences.
- Run Full combinations before individual features are justified.
- Modify mooc/temfin baseline code paths unless explicitly requested.
