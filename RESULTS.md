# DyGFormer 四特征引入实验结果概览

本文档用于快速掌握本代码库在 DyGFormer 上围绕四类新增特征所做的改进、接入方式和实验结论。相关实现主要位于 `features/`、`models/DyGFormer.py`、训练/评估入口脚本，以及 `experiment_analysis/feature_ablation_analysis.md`。

## 1. 项目改进概述

原始 DyGFormer 使用节点特征、边特征、时间编码和邻居共现编码四个通道。当前代码库在此基础上增加了一个统一的特征扩展机制：

- `FeatureBank` 从 `processed_data/{dataset}/` 读取预计算特征文件，例如 `style_v1.npy`、`social_v1.npy`。
- `FeatureAssembler` 根据命令行开关 `--use_style`、`--use_personality`、`--use_topic`、`--use_social` 选择要启用的特征。
- `DyGFormer` 对边级特征新增独立 channel，并为每个新增边级特征建立独立 projection layer。
- 节点级 `personality` 特征通过 `FeatureFusion` 在 DyGFormer 输出层之后与节点 embedding 融合。
- 结果文件名通过 `feature_tag` 区分不同实验设置，例如 `base`、`S`、`P`、`T`、`So`、`full`。

这套改动让同一训练/评估脚本可以直接跑 baseline、单特征消融、组合特征和 full model。

## 2. 四个新增特征

| 特征 | 粒度 | 维度 | 构造方法 | 模型接入方式 | 主要判断 |
|---|---:|---:|---|---|---|
| Style | edge | 32 | 对原始 172 维 LIWC 边特征做 PCA | 额外边级 channel | 与原始边特征高度冗余，收益很弱 |
| Personality | node | 5 | 按节点聚合交互中的 LIWC 均值，再 PCA | 输出后 sidecar fusion | LP 基本中性，NC 上明显不稳定/有害 |
| Topic | edge | 10 | 对平移为非负的 LIWC 边特征做 NMF，并行归一化 | 额外边级 channel | 与原始边特征同源，整体无稳定收益 |
| Social | edge | 10 | 对每条边按时间顺序计算 src/dst 的历史结构统计 | 额外边级 channel | 唯一跨数据集、跨任务稳定正向的特征 |

需要注意：Style、Personality、Topic 都是从已有 LIWC 特征派生出来的代理特征，不是基于原始文本的真实风格、人格或主题建模。Social 是动态图结构统计，信息来源与 LIWC 边特征正交。

Social 的 10 维由 src 和 dst 各 5 个统计组成：历史交互次数、历史唯一邻居数、距离上次交互的时间、活跃率、重复交互比例。计算时每条边只使用当前时间之前的交互，因此局部统计本身是时间因果的。

## 3. Link Prediction 结果

实验对象为 Wikipedia 和 Reddit，均为 5 个 seed 的完整训练结果。核心指标使用 test AP。

| 数据集 | Base | +Style | +Personality | +Topic | +Social | Full |
|---|---:|---:|---:|---:|---:|---:|
| Wikipedia | 0.9904±0.0002 | 0.9902±0.0003 | 0.9904±0.0001 | 0.9903±0.0003 | **0.9918±0.0001** | **0.9917±0.0002** |
| Reddit | 0.9922±0.0001 | 0.9920±0.0001 | 0.9924±0.0001 | 0.9921±0.0001 | **0.9930±0.0001** | **0.9930±0.0001** |

相对 Base 的提升：

- Wikipedia: `+Social` 提升约 +0.0014 AP，Full 提升约 +0.0013 AP。
- Reddit: `+Social` 提升约 +0.0008 AP，Full 提升约 +0.0008 AP。
- Style 和 Topic 在两个数据集上基本持平或轻微下降。
- Personality 在 Reddit 上有很小正效应，在 Wikipedia 上近似无变化。

结论：LP 上 Full model 的收益几乎完全来自 Social。Style/Topic 是原始 172 维 LIWC 边特征的低维变换，新增参数多于新增信息；DyGFormer 原本的 edge channel 已经能直接利用这些 LIWC 信息。

## 4. Node Classification 结果

节点分类同样在 Wikipedia 和 Reddit 上做 5 个 seed，核心指标为 test ROC-AUC。

| 数据集 | Base | +Style | +Personality | +Topic | +Social | Full |
|---|---:|---:|---:|---:|---:|---:|
| Wikipedia | 0.8714±0.0060 | 0.8720±0.0109 | 0.8552±0.0400 | 0.8776±0.0217 | **0.8806±0.0197** | 0.8755±0.0119 |
| Reddit | 0.6837±0.0090 | 0.6703±0.0257 | 0.6431±0.0147 | 0.6701±0.0455 | **0.6983±0.0304** | 0.6453±0.0463 |

相对 Base 的变化：

- Wikipedia: `+Social` 提升约 +0.0092 ROC-AUC，是最佳单特征；Full 只提升约 +0.0041。
- Reddit: `+Social` 提升约 +0.0146 ROC-AUC，是唯一明显正向设置；Full 下降约 -0.0384。
- Personality 在 NC 上负效应明显，尤其 Reddit 下降约 -0.0406。
- NC 的方差明显大于 LP，结论应比 LP 更谨慎。

结论：NC 上 Social 仍是最可靠的正向特征。Full model 在 Reddit 上表现差，原因是 Personality、Style、Topic 的噪声或冗余信号抵消了 Social 的贡献。

## 5. 总体结论

1. 最有价值的改进是 Social 动态结构特征。它在 Wikipedia/Reddit、LP/NC 四个设置中都带来正向结果，是当前四特征扩展里最值得保留和重点阐述的部分。
2. Full model 并不总是优于 `+Social`。LP 上 Full 的提升基本等同于 Social；NC 上 Full 甚至可能被弱代理特征拖累。
3. Style 和 Topic 的实验价值主要在消融对照：它们证明“对已有 LIWC 边特征再做 PCA/NMF 分解”不能稳定提升 DyGFormer。
4. Personality 当前实现是粗糙的节点级 LIWC 聚合代理，不应被表述为真实 Big Five 人格特征；在当前后置 fusion 和冻结 backbone 的 NC 流程下尤其容易引入噪声。
5. 如果撰写论文或报告，应把贡献表述为“动态结构特征对 DyGFormer 的有效增强”，而不是泛化地宣称四类特征全部有效。

## 6. 实验与协议注意事项

- 当前结论来自 `experiment_analysis/feature_ablation_analysis.md` 和 `experiment_analysis/feature_ablation_all_results.csv` 中的 5-run 消融结果。
- Style PCA、Topic NMF、Personality 聚合当前是在全量边上拟合/统计；Social 的逐边统计是因果的，但最终 z-score 标准化使用了全量边统计。
- 如果需要更严格的论文级协议，应只在训练边上拟合 PCA/NMF/归一化统计，再 transform 验证集和测试集。新协议下的结果不能直接与当前全量预处理结果混用。
- 推荐后续优先改进 Social，而不是继续堆叠 LIWC 派生代理特征；若要继续做 Style/Topic/Personality，最好引入原始文本或更可靠的外部文本模型。
