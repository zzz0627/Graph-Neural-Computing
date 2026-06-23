# 项目介绍

本文档根据仓库源码、`README.md`、`HANDOFF.md`、`experiment_analysis/` 以及 `experiment_logs/` 中的训练评估日志整理，用于快速说明本项目正在做什么、已经完成了哪些实验，以及从现有结果中可以得到什么判断。

## 一句话概括

本项目基于 DyGLib 动态图学习框架，围绕 DyGFormer 在连续时间动态图上的链路预测和节点分类任务展开实验；当前仓库还扩展了多类外部特征通道，并在 `mooc`、`temfin`、`wikipedia`、`reddit` 等数据集上做了 baseline 与特征消融分析。

## 项目背景

DyGLib 原始目标是为动态图学习提供统一、可复现的训练和评估框架。动态图中的一条边通常表示为 `(source_node, destination_node, timestamp, edge_label, edge_features)`，模型需要基于历史交互序列学习节点表示，并完成后续预测。

仓库保留了 DyGLib 的主干能力：

- 支持连续时间动态图数据预处理，处理后数据位于 `processed_data/<dataset>/`，包括 `ml_<dataset>.csv`、边特征 `.npy` 和节点特征 `.npy`。
- 支持动态链路预测，评估 transductive 与 inductive/new-node 场景，并支持 random、historical、inductive 等负采样策略。
- 支持动态节点分类，通过已有动态图编码器输出节点表示，再训练分类头。
- 集成多种动态图模型，包括 TGAT、JODIE、DyRep、TGN、CAWN、TCL、GraphMixer、EdgeBank 和 DyGFormer。

当前仓库在原版基础上做了两类主要扩展：

- 数据与实验扩展：加入或适配了 `mooc`、`temfin` 数据集，并完成 DyGFormer baseline 的训练和评估。
- 特征扩展：加入 `style`、`personality`、`topic`、`social` 四类预计算特征，并通过 `FeatureBank`、`FeatureAssembler` 和 DyGFormer 的额外通道/sidecar fusion 接入模型。

## 核心任务

### 动态链路预测

入口脚本是 `train_link_prediction.py` 和 `evaluate_link_prediction.py`。任务是在给定历史时序交互后，判断某个时间点的源节点和目标节点是否会发生连接。训练中会采样负边，评估时会分别报告普通测试集和 new-node 测试集的 Average Precision 与 ROC-AUC。

当前 `experiment_logs` 中的 `mooc` 与 `temfin` 链路预测实验均使用 DyGFormer baseline：

- `model_name=DyGFormer`
- `max_input_sequence_length=256`
- `patch_size=8`
- `dropout=0.1`
- `num_runs=5`
- `negative_sample_strategy=random`
- 未启用额外特征，日志中显示 `FeatureAssembler: running in baseline mode (no extra features)`

### 动态节点分类

入口脚本是 `train_node_classification.py` 和 `evaluate_node_classification.py`。任务是利用动态图编码器得到节点表示后预测节点状态标签。当前仓库已经把节点分类数据集选择扩展到 `wikipedia`、`reddit`、`mooc`、`temfin`。

在 `experiment_logs` 中，`mooc` 和 `temfin` 的节点分类实验同样使用 DyGFormer baseline，主要指标是 ROC-AUC。

## 代码结构

关键目录和文件如下：

| 路径 | 作用 |
|---|---|
| `train_link_prediction.py` | 动态链路预测训练入口 |
| `evaluate_link_prediction.py` | 动态链路预测 checkpoint 评估入口 |
| `train_node_classification.py` | 动态节点分类训练入口 |
| `evaluate_node_classification.py` | 动态节点分类 checkpoint 评估入口 |
| `models/` | TGAT、TGN、CAWN、TCL、GraphMixer、DyGFormer 等模型实现 |
| `utils/DataLoader.py` | 读取处理后的动态图数据并划分训练/验证/测试集 |
| `utils/load_configs.py` | 命令行参数、best config 加载、特征开关配置 |
| `features/` | Style、Personality、Topic、Social 特征抽取与加载 |
| `processed_data/` | 已预处理的数据集文件 |
| `experiment_logs/` | 当前 mooc/temfin baseline 训练和评估日志 |
| `experiment_analysis/` | Wikipedia/Reddit 特征消融、mooc/temfin 评估分析和结果汇总脚本 |

## 特征扩展在做什么

仓库新增的多特征机制集中在 `features/` 和 `models/DyGFormer.py`。

| 特征 | 粒度 | 来源与方法 | 当前判断 |
|---|---|---|---|
| `style` | edge | 对原始 172 维 LIWC 边特征做 PCA，默认 32 维 | 与原始边特征信息重叠，收益很弱 |
| `personality` | node | 按节点聚合 LIWC 均值后 PCA 到 5 维 | 只是用户级语言 proxy，节点分类中可能引入噪声 |
| `topic` | edge | 对 LIWC 边特征做 NMF，默认 10 维 | 更像内容分解 proxy，不是真正文本 topic |
| `social` | edge | 基于历史交互计算 degree、邻居多样性、recency、activity rate、repeat ratio 等 10 维结构统计 | 最稳定有效，属于时间感知动态图结构信号 |

`FeatureBank` 负责从 `processed_data/<dataset>/<feature>_v1.npy` 懒加载预计算特征；`FeatureAssembler` 根据命令行 `--use_style`、`--use_personality`、`--use_topic`、`--use_social` 组织启用特征；DyGFormer 将 edge-level 特征作为额外 edge channel，将 node-level 特征通过 sidecar fusion 融入输出节点表示。

## experiment_logs 反映出的实验事实

`experiment_logs/` 主要记录了 `mooc` 和 `temfin` 上 DyGFormer baseline 的训练与评估。日志文件包括：

- `train_mooc_dygformer.log`
- `train_mooc_node_classification.log`
- `train_temfin_dygformer.log`
- `train_temfin_dygformer_seed4.log`
- `train_temfin_node_classification.log`
- `eval_mooc_lp.log`
- `eval_mooc_nc.log`
- `eval_temfin_lp.log`
- `eval_temfin_nc.log`

### 汇总结果

| 数据集 | 任务 | 阶段 | Test 指标 | New-node Test 指标 | 结论 |
|---|---|---|---|---|---|
| `mooc` | Link Prediction | evaluation | AP 0.8656±0.0049, AUC 0.8719±0.0045 | AP 0.8614±0.0041, AUC 0.8705±0.0036 | 普通测试和新节点测试接近，泛化差距小 |
| `mooc` | Node Classification | train/evaluation | ROC-AUC 0.7860±0.0062 | 不适用 | 分类性能中等且 seed 间稳定 |
| `temfin` | Link Prediction | evaluation | AP 0.9887±0.0004, AUC 0.9841±0.0007 | AP 0.9155±0.0008, AUC 0.8787±0.0011 | 普通链路预测极高，但新节点泛化差距明显 |
| `temfin` | Node Classification | train/evaluation | ROC-AUC 0.6068±0.0606 | 不适用 | 分类效果弱且方差大 |

### 对日志结果的解释

`mooc` 上，DyGFormer 的链路预测 Test AP/AUC 约为 0.866/0.872，new-node 指标几乎不下降。这说明该数据集中训练期间出现过的节点与新节点之间的行为模式差异较小，模型能较好迁移到未见节点。

`temfin` 上，普通链路预测 Test AP/AUC 达到 0.989/0.984，说明金融交互图中存在很强的可预测结构或重复交易模式。但 new-node Test AP 降到 0.9155，AUC 降到 0.8787，说明新交易方或新节点的行为模式更难从历史已知节点泛化。

节点分类方面，`mooc` 的 Test ROC-AUC 为 0.7860±0.0062，较稳定；`temfin` 的 Test ROC-AUC 为 0.6068±0.0606，波动明显。`HANDOFF.md` 中给出的解释是 `temfin` 标签高度不平衡，正样本率约 0.71%，而当前训练使用的 BCELoss 没有 class weighting，因此分类头很难稳定学习少数类信号。

`train_temfin_dygformer.log` 中 seed4 曾中断，`train_temfin_dygformer_seed4.log` 是单独补跑结果；最终 `eval_temfin_lp.log` 对五个 checkpoint 的评估显示 5 runs 结果完整且稳定。

## experiment_analysis 反映出的特征消融结论

`experiment_analysis/feature_ablation_analysis.md` 和 `feature_proxy_improvement_handoff.md` 记录了 Wikipedia/Reddit 上 DyGFormer 多特征扩展实验。核心结论是：

- `social` 是唯一在链路预测和节点分类中都稳定正向的特征。
- `style` 和 `topic` 主要来自原始 LIWC 边特征的 PCA/NMF 变换，与 baseline 已输入的 172 维边特征高度冗余。
- `personality` 是静态的用户级 proxy，不是经过验证的人格测量；在节点分类中尤其容易引入噪声。
- Full model 的提升主要来自 `social`，并不说明四类特征都有贡献。

关键结果摘要：

| 数据集 | 任务 | Base | +Social | Full | 主要观察 |
|---|---|---:|---:|---:|---|
| Wikipedia | Link Prediction AP | 0.9904±0.0002 | 0.9918±0.0001 | 0.9917±0.0002 | Social 显著提升 |
| Reddit | Link Prediction AP | 0.9922±0.0001 | 0.9930±0.0001 | 0.9930±0.0001 | Full 收益基本等同 Social |
| Wikipedia | Node Classification ROC-AUC | 0.8714±0.0060 | 0.8806±0.0197 | 0.8755±0.0119 | Social 单特征最好 |
| Reddit | Node Classification ROC-AUC | 0.6837±0.0090 | 0.6983±0.0304 | 0.6453±0.0463 | Personality 等噪声特征拖累 Full |

因此，如果后续要写论文或报告，比较稳妥的表述是：本项目验证了时间感知结构特征可以稳定增强 DyGFormer；而 Style、Topic、Personality 当前只是基于 LIWC 的 proxy 特征，不能过度声称它们是真正的语言风格、语义主题或人格特征。

## 如何运行

本项目使用 conda 环境 `dyg`。运行任何 Python 脚本前应先激活环境：

```bash
conda activate dyg
```

示例命令：

```bash
# mooc 链路预测训练
python train_link_prediction.py \
  --dataset_name mooc --model_name DyGFormer \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 600 --num_runs 5 --gpu 0

# temfin 链路预测评估
python evaluate_link_prediction.py \
  --dataset_name temfin --model_name DyGFormer \
  --negative_sample_strategy random \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 2000 --num_runs 5 --gpu 0

# Wikipedia/Reddit 多特征链路预测实验
bash run_experiments.sh --dataset wikipedia --tag So

# Wikipedia/Reddit 多特征节点分类实验
bash run_experiments.sh --nc --dataset reddit --tag So
```

注意：`temfin` 不应直接使用 `--load_best_configs`，因为当前 `utils/load_configs.py` 中 DyGFormer 的 best config 分支没有专门覆盖 `temfin`，会落入不匹配的默认 `seq_len=32, patch_size=1` 配置，导致 checkpoint 结构不一致。

## 当前项目状态

从现有文件看，项目已经完成以下工作：

- 原版 DyGLib/DyGFormer 训练、评估流程仍可用。
- `mooc` 数据已转换并完成 DyGFormer baseline 的链路预测与节点分类。
- `temfin` 数据已适配并完成 DyGFormer baseline 的链路预测与节点分类。
- Wikipedia/Reddit 上的四类 proxy 特征消融实验已完成，并形成分析文档。
- 当前最可靠的实验结论是：动态图结构类 `social` 特征最值得保留和强调；其余基于 LIWC 变换的 proxy 特征应谨慎使用和描述。

后续如果继续推进，优先方向应是围绕 `social` 特征做更严格的因果预处理、训练集内归一化和跨数据集验证；如果继续研究 Style/Topic/Personality，应先获得更高质量的原始文本或外部标注，而不是只在已有 LIWC 特征上做线性或矩阵分解变换。
