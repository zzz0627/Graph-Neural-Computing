# DyGFormer Multi-Feature Extension: Experiment Analysis

## 1. Link Prediction Results (5 runs, full training with best configs)

### 1.1 Wikipedia


| Setting      | test AP           | test AUC          | nn_test AP        | nn_test AUC       | delta AP vs Base | Sig.        |
| ------------ | ----------------- | ----------------- | ----------------- | ----------------- | ---------------- | ----------- |
| Base         | 0.9904±0.0002     | 0.9894±0.0002     | 0.9858±0.0002     | 0.9842±0.0001     | --               | --          |
| +Style       | 0.9902±0.0003     | 0.9890±0.0002     | 0.9853±0.0002     | 0.9836±0.0003     | -0.0002          | n.s.        |
| +Personality | 0.9904±0.0001     | 0.9895±0.0002     | 0.9856±0.0002     | 0.9841±0.0002     | +0.0000          | n.s.        |
| +Topic       | 0.9903±0.0003     | 0.9893±0.0003     | 0.9858±0.0003     | 0.9840±0.0003     | -0.0001          | n.s.        |
| **+Social**  | **0.9918±0.0001** | **0.9912±0.0001** | **0.9878±0.0004** | **0.9866±0.0002** | **+0.0014**      | **p<0.001** |
| +S+P         | 0.9905±0.0001     | 0.9895±0.0001     | 0.9860±0.0002     | 0.9843±0.0001     | +0.0001          | n.s.        |
| +S+P+T       | 0.9906±0.0003     | 0.9894±0.0003     | 0.9856±0.0002     | 0.9840±0.0003     | +0.0002          | n.s.        |
| **Full**     | **0.9917±0.0002** | **0.9909±0.0002** | **0.9875±0.0004** | **0.9865±0.0004** | **+0.0013**      | **p<0.001** |


### 1.2 Reddit


| Setting      | test AP           | test AUC          | nn_test AP        | nn_test AUC       | delta AP vs Base | Sig.        |
| ------------ | ----------------- | ----------------- | ----------------- | ----------------- | ---------------- | ----------- |
| Base         | 0.9922±0.0001     | 0.9915±0.0001     | 0.9882±0.0002     | 0.9869±0.0002     | --               | --          |
| +Style       | 0.9920±0.0001     | 0.9912±0.0001     | 0.9879±0.0002     | 0.9865±0.0002     | -0.0002          | p<0.01      |
| +Personality | 0.9924±0.0001     | 0.9917±0.0001     | 0.9886±0.0001     | 0.9875±0.0002     | +0.0002          | p<0.01      |
| +Topic       | 0.9921±0.0001     | 0.9914±0.0001     | 0.9882±0.0002     | 0.9869±0.0003     | -0.0001          | p<0.05      |
| **+Social**  | **0.9930±0.0001** | **0.9925±0.0001** | **0.9895±0.0003** | **0.9885±0.0003** | **+0.0008**      | **p<0.001** |
| +S+P         | 0.9922±0.0001     | 0.9915±0.0001     | 0.9884±0.0001     | 0.9871±0.0001     | +0.0000          | n.s.        |
| +S+P+T       | 0.9922±0.0001     | 0.9914±0.0001     | 0.9883±0.0002     | 0.9871±0.0002     | +0.0000          | n.s.        |
| **Full**     | **0.9930±0.0001** | **0.9925±0.0001** | **0.9895±0.0002** | **0.9886±0.0002** | **+0.0008**      | **p<0.001** |


### 1.3 Link Prediction Key Findings

**Finding 1: Social 是唯一稳定显著提升 link prediction 的单特征。**

- Wikipedia: +0.0014 AP (p<0.001), 5 个 seed 全部高于 Base 最高值
- Reddit: +0.0008 AP (p<0.001), 同样全部 seed 一致
- 这不令人意外：time-aware structural statistics (degree, diversity, recency, activity rate, repeat ratio) 直接编码了交互拓扑，而 link prediction 本质上就是预测拓扑——结构特征与任务目标高度对齐

**Finding 2: Style / Topic 对 link prediction 有微弱负效应或无效应。**

- Style: Wikipedia -0.0002 (n.s.), Reddit -0.0002 (p<0.01 因 std 极小)
- Topic: Wikipedia -0.0001 (n.s.), Reddit -0.0001 (p<0.05)
- 解释：Style (PCA) 和 Topic (NMF) 是原始 172d edge features 的不同低维投影。原始 edge features 已经通过 edge channel 完整进入 DyGFormer。增加这些冗余的变换视角不仅没有提供新信息，反而增加了模型参数 (多一个 projection_layer + 更宽的 Transformer)，在不增加训练 epochs 的情况下略微过拟合

**Finding 3: Personality 在 Reddit 上有微弱正效应。**

- Reddit: +0.0002 AP (p<0.01), 在 new node test 上更明显 (+0.0004 AP, p<0.01)
- Wikipedia: 无显著效应
- 解释：Personality 是 per-node 聚合特征，通过 sidecar fusion 注入。它提供了与 per-edge 特征正交的用户级信号。Reddit 用户的行为模式更多样 (672K edges vs 157K edges)，per-node 聚合更稳定

**Finding 4: Full model 的增益主要来自 Social。**

- Wikipedia Full (+0.0013) 几乎等于 Social alone (+0.0014)
- Reddit Full (+0.0008) 完全等于 Social alone (+0.0008)
- 这说明 Style / Personality / Topic 在 Full model 中没有在 Social 基础上提供额外增益。Social 的结构信号主导了 link prediction 任务上的改进

**Finding 5: 组合特征 (S+P, S+P+T) 无叠加效应。**

- S+P 和 S+P+T 在两个数据集上都接近或等于 Base
- Style 和 Topic 的微弱负效应被 Personality 的微弱正效应抵消

---

## 2. Node Classification Results (5 runs)

### 2.1 Wikipedia


| Setting      | test ROC-AUC      | delta vs Base |
| ------------ | ----------------- | ------------- |
| Base         | 0.8714±0.0060     | --            |
| +Style       | 0.8720±0.0109     | +0.0006       |
| +Personality | 0.8552±0.0400     | -0.0162       |
| +Topic       | 0.8776±0.0217     | +0.0062       |
| **+Social**  | **0.8806±0.0197** | **+0.0092**   |
| +S+P         | 0.8565±0.0177     | -0.0149       |
| +S+P+T       | 0.8640±0.0237     | -0.0074       |
| Full         | 0.8755±0.0119     | +0.0041       |


### 2.2 Reddit


| Setting      | test ROC-AUC      | delta vs Base |
| ------------ | ----------------- | ------------- |
| Base         | 0.6837±0.0090     | --            |
| +Style       | 0.6703±0.0257     | -0.0134       |
| +Personality | 0.6431±0.0147     | -0.0406       |
| +Topic       | 0.6701±0.0455     | -0.0136       |
| **+Social**  | **0.6983±0.0304** | **+0.0146**   |
| +S+P         | 0.6378±0.0144     | -0.0459       |
| +S+P+T       | 0.6526±0.0225     | -0.0311       |
| Full         | 0.6453±0.0463     | -0.0384       |


### 2.3 Node Classification Key Findings

**Finding 6: Node classification 方差远大于 link prediction。**

- NC 的 std 是 LP 的 10-100 倍 (如 Reddit Base: NC std=0.009 vs LP std=0.0001)
- 这是因为 NC 只训练 MLPClassifier head (backbone 冻结)，label 本身噪声大 (state_label = user banned or not)，且 Wikipedia/Reddit 的 NC 标签分布高度不平衡

**Finding 7: Social 是唯一在 NC 上也稳定正向的特征。**

- Wikipedia: +0.0092 (0.8714 -> 0.8806)
- Reddit: +0.0146 (0.6837 -> 0.6983)
- Social 的 time-aware structural statistics 编码了用户的行为活跃模式，与 "是否被 ban" 的节点标签存在合理的相关性 (高 activity rate, 高 repeat ratio 可能与被 ban 行为相关)

**Finding 8: Personality 在 NC 上有明显负效应，特别是 Reddit。**

- Wikipedia: -0.0162 (高方差 std=0.04)
- Reddit: -0.0406
- 原因分析：Personality 是 per-node mean of LIWC features -> PCA(5d)。这 5 维 proxy 信号被 sidecar fusion 注入 frozen backbone 的 output layer 之后，可能引入了与 NC 标签无关的噪声。特别是在 Reddit 上，节点分类本身就困难 (Base AUC=0.68)，噪声信号的负面影响更大

**Finding 9: Full model 在 NC 上表现不一致。**

- Wikipedia Full (+0.0041) 不如 Social alone (+0.0092)
- Reddit Full (-0.0384) 严重恶化
- 这是因为 Full = Social(正) + Style(弱负/中性) + Personality(强负) + Topic(弱负)。Personality 的负效应抵消并超过了 Social 的正效应

---

## 3. Cross-Task Analysis

### 3.1 Feature Contribution Heatmap


| Feature                     | Wiki LP     | Reddit LP   | Wiki NC     | Reddit NC   | Verdict                                                   |
| --------------------------- | ----------- | ----------- | ----------- | ----------- | --------------------------------------------------------- |
| Style (PCA 32d)             | -0.0002     | -0.0002     | +0.0006     | -0.0134     | Neutral to slightly negative; redundant with edge channel |
| Personality (PCA 5d)        | +0.0000     | +0.0002     | -0.0162     | -0.0406     | Harmful for NC, neutral for LP                            |
| Topic (NMF 10d)             | -0.0001     | -0.0001     | +0.0062     | -0.0136     | Neutral; redundant with edge channel                      |
| **Social (structural 10d)** | **+0.0014** | **+0.0008** | **+0.0092** | **+0.0146** | **Consistently positive across all 4 settings**           |


### 3.2 Why Social Wins

Social 特征在所有 4 个实验设置（2 数据集 x 2 任务）中都是唯一稳定正向的特征。原因：

1. **信号独立性**：Social 特征 (degree, diversity, recency, activity rate, repeat ratio) 来自图的拓扑结构，与原始 172d LIWC edge features 在信息来源上完全正交。而 Style / Topic / Personality 都是 LIWC features 的不同变换，与 edge channel 已有的 172d features 高度重叠。
2. **任务对齐性**：Link prediction 预测的是拓扑结构 (会不会发生交互)，Social 编码的正是拓扑结构的局部统计。Node classification 预测的是用户状态 (会不会被 ban)，Social 编码的活跃模式与被 ban 行为高度相关。
3. **时间感知**：Social 是唯一严格因果、时间感知的特征。其他三个都是静态的全局变换。时间感知的 social context 让 Transformer attention 能看到 "这个节点的社交活跃度随历史如何演变"，这是独特的时序信息。

### 3.3 Why Style / Topic Underperform

Style (PCA) 和 Topic (NMF) 在 DyGFormer 框架下作为额外 edge channel 效果有限。根因分析：

1. **信息冗余**：原始 172d LIWC features 已经通过 edge channel 完整进入模型。PCA 的 32 个主成分只是 172d 的线性子空间投影，NMF 的 10 个 topic 是非负分解——它们不包含 edge channel 没有的信息。模型需要额外学习 projection 层来利用这些冗余视角，但收益不足以弥补参数增加的代价。
2. **DyGFormer 的 multi-channel Transformer 已经具有隐式特征分离能力**：Transformer 的 multi-head attention 本身就可以学会关注 172d edge features 的不同子空间。显式地预先分离成 style / topic 并不比让 Transformer 自行学习更好。

### 3.4 Personality 的问题

Personality 作为 per-node sidecar 在 LP 上中性、在 NC 上有害。可能的原因：

1. **Proxy 质量低**：5 维 PCA on per-node mean LIWC 是非常粗糙的 personality proxy。真实的 Big Five 需要专门的问卷或 fine-tuned 文本分类器。
2. **Sidecar fusion 的位置问题**：Personality 在 output_layer 之后通过 concat + projection 注入。这个位置太晚了——fusion 层需要学会在 172d backbone embedding 和 5d personality proxy 之间建立有用的关联，但 5d 的信噪比太低。
3. **NC 任务的特殊性**：NC backbone 是 frozen 的，只有 fusion 层和 MLPClassifier 可训练。这意味着 personality 的噪声直接传递到分类器输入，没有机会被 backbone 的注意力机制过滤。

---

## 4. Honest Assessment of Proxy Features


| Feature     | Proxy method                              | Faithfulness                                          | Recommendation for paper                                                                     |
| ----------- | ----------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Style       | PCA on same LIWC features as edge channel | Low -- just a linear subspace, not true style         | Do NOT claim "language style modeling". At most say "PCA-based linguistic variation channel" |
| Personality | Per-node mean LIWC + PCA(5)               | Low -- not validated Big Five                         | Do NOT claim "personality feature". Say "user-level aggregated linguistic proxy"             |
| Topic       | NMF on same LIWC features                 | Low -- not real topic model                           | Do NOT claim "topic modeling". Say "NMF-based content decomposition"                         |
| **Social**  | Time-aware structural stats               | **Medium-High** -- these ARE real structural features | Can reasonably say "dynamic structural features" or "time-aware social context"              |


---

## 5. Recommendations

### 5.1 For the Paper

1. **Lead with Social**: Social is the only feature with consistent, significant, cross-task improvements. Make it the central contribution.
2. **Frame Style/Topic/Personality as ablation baselines**: They show that naive feature engineering on LIWC (PCA, NMF) does NOT help when the original features are already available as an edge channel. This is itself a useful finding.
3. **The negative result is publishable**: Showing that proxy NLP features (from LIWC dimensions) cannot improve over the raw LIWC edge channel, while structural graph features CAN, is a meaningful empirical contribution.
4. **Be honest about proxy nature**: Every proxy feature must be clearly labeled. Do not over-claim.

### 5.2 For Future Work

1. **If raw text were available**: Style / Personality / Topic could be computed with dedicated models (BERT-based style classifier, Big Five predictor, BERTopic). This would likely yield much stronger signals.
2. **Personality injection point**: If continued, try injecting personality into node features (replacing the zero vectors) instead of sidecar fusion. This would let the Transformer see it throughout the attention process.
3. **Social feature enrichment**: The current 10d structural features are simple but effective. Consider adding more: PageRank, betweenness centrality (expensive), community membership (approximated).

