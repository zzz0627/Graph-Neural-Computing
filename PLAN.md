# Social v2 与严格特征协议优化计划

## Summary

目标是围绕当前唯一稳定正向的 `Social` 特征做下一轮优化，同时修正实验管理、统计功效和预处理协议问题。核心原则是：不覆盖 v1、先 Smoke Test、科学对比中固定同一组超参、用 5-seed paired delta 判断小幅 LP 改进、NC 作为最终价值判断的主要证据。

本轮不继续优化 `Style/Topic/Personality`。它们保留为 v1 全量预处理下的负/中性消融结果，并在报告中明确标注 proxy 与 preprocessing caveat；除非后续获得 raw text 或外部标注，不做 PCA/NMF 维度 sweep 或 Full 组合。

## Review Assessment

Claude Opus 4.8 的审查意见总体有效，确实指出了原 Plan 的主要不足：

- 保存路径风险成立：当前 `save_model_name/save_result_name` 只含 `feature_tag`，不含 `feature_version`，所以 `social_v1`、`social_v1_trainfit`、`social_v2_trainfit` 都会落到 `So` 产物名，必须隔离。
- 3-seed gate 不足成立：Wikipedia/Reddit LP 的 Social 增益只有约 `0.0008-0.0014` AP，必须用 5 个相同 seed 的 paired delta，而不是 3-seed 均值粗判。
- batch size sweep 混杂成立：Base/v1/v2 若使用不同 batch size，特征效果和优化超参效果无法区分。
- LP gate 不是 NC 充分条件成立：LP 只能证明特征没有明显破坏链路预测，NC 才是高方差但更能反映行为特征价值的证据。
- v2 特征规格不足成立：需要固定窗口、衰减参数和严格 `< t` 实现方式。
- 但有一点需修正：LP 训练边不只是 `ts <= val_time`，还包含 inductive 设置下的 `observed_edges_mask`。因此 train-only normalization 应使用 `get_link_prediction_data` 中真实 LP `train_data.edge_ids`，而不是仅用时间阈值。

## Key Changes

- 新增实验隔离参数 `--experiment_dir`，默认空值保持现有 legacy 行为；设置后 `saved_models/`、`saved_results/`、`logs/` 都写入该目录下。
- 严格要求每个 feature protocol 使用独立目录。由于文件名仍可为 `DyGFormer_So_seed*`，目录隔离是防覆盖的硬约束。
- `--experiment_dir` 必须贯穿所有硬编码路径，特别是 NC **读取** LP checkpoint 的路径（`train_node_classification.py` 的 `load_model_folder`），否则 NC 会在实验目录里找不到 LP checkpoint 而失败。需要一并改写的至少有：`train_link_prediction.py` 的 `save_model_folder` 与 `save_result_folder`、`train_node_classification.py` 的 load / save / result 路径、`evaluate_link_prediction.py` 与 `evaluate_node_classification.py`、以及 `collect_results.py` 的 `--results_root`。实现上应把路径拼接收敛到单一 helper（如 `build_experiment_paths(args)`），避免逐处漏改。
- 不新增运行时 `--social_variant` 或 `--normalization_scope`。Social v1/v2 作为离线 `.npy` 特征生成协议实现，训练时继续使用现有 `--feature_bank_dir` 和 `--feature_version`。
- 更新结果收集脚本支持 `--results_root`，用于扫描独立实验目录；汇总输出必须包含 `experiment_dir`、`feature_version`、`preprocessing_protocol`。
- 推荐目录结构：

```text
experiments/
  feature_base_ref/
    saved_models/
    saved_results/
    logs/
    analysis/
  feature_social_v1_trainfit/
    feature_bank/{wikipedia,reddit}/social_v1_trainfit.npy
    saved_models/
    saved_results/
    logs/
    manifests/
    analysis/
  feature_social_v2_trainfit/
    feature_bank/{wikipedia,reddit}/social_v2_trainfit.npy
    saved_models/
    saved_results/
    logs/
    manifests/
    analysis/
```

## Social Feature Protocol

所有 Social 特征仍是 edge-level feature，文件路径满足：

```text
{feature_bank_dir}/{dataset}/social_{feature_version}.npy
```

`v1_trainfit`：

- 维度保持 10，即每端 5 维：`log_degree`、`log_unique_neighbors`、`log_recency`、`activity_rate`、`repeat_ratio`。
- 每条边仍按现有 `social_extractor.py` 的因果模式计算：先读取节点历史状态，再更新当前边，确保只使用 `timestamp < t`。
- z-score normalization 的 mean/std 只用 LP 真实训练边拟合：`get_link_prediction_data(...).train_data.edge_ids`，包含 `ts <= val_time` 和 inductive `observed_edges_mask`。

`v2_trainfit`：

- 维度为 20，即 src/dst 各 10 维：先包含 v1 的 5 维，再追加 5 维动态结构统计。
- 追加统计定义固定为：
  - `short_count = log1p(count(t_i in [t - 0.01 * train_time_span, t)))`
  - `long_count = log1p(count(t_i in [t - 0.10 * train_time_span, t)))`
  - `short_repeat_ratio = (short_count_raw - unique_neighbors_short) / max(short_count_raw, 1)`
  - `decayed_degree = sum(exp(-(t - t_i) / tau))`，`tau = 0.05 * train_time_span`
  - `burstiness = std(last_inter_event_gaps) / (mean(last_inter_event_gaps) + 1e-8)`，最多使用最近 20 个正间隔，少于 2 个间隔时为 0
- `train_time_span = val_time - min_timestamp + 1e-8`，按数据集分别计算。Wikipedia/Reddit 时间戳范围接近，但仍使用各自训练时间跨度以避免硬编码绝对窗口。
- 所有追加统计也必须使用 `timestamp < t`，实现上复用“读状态后更新”的单次 chronological scan。
- normalization 同 `v1_trainfit`，只用 LP 真实训练边拟合 mean/std，然后 transform 全量边。

`v2_trainfit` 设计假设（必须在分析阶段验证，避免重蹈 Style/Topic 冗余覆辙）：

- v2 新增 5 维中，`short_count`/`long_count`/`decayed_degree` 本质是 v1 `log_degree` 的时间窗/时间加权重投影，`short_repeat_ratio` 是 v1 `repeat_ratio` 的窗口版本，因此**大概率与 v1 高度冗余**；预期真正提供正交新信号的主要是 `burstiness`（交互间隔规律性）、其次是 `decayed_degree` 的时间加权视角。
- 若 v2 相对 v1 无稳定增益，结论应明确归因到“这些时间加权统计与 v1 已有结构信号冗余”，而不是含糊地说 v2 无效。

实现复杂度说明：

- `short_repeat_ratio` 需要“时间窗内唯一邻居数”，比 v1 的全量 `set` 更复杂，不能简单复用全量去重。实现上为每个节点维护一个按时间排序的 `deque`，存近期 `(neighbor, timestamp)`；扫描到每条边时先从队首逐出早于 `t - window` 的记录，再读取当前窗口内的计数与唯一邻居数（唯一邻居用一个随逐出/插入增删的 `Counter` 多重集维护，计数归零即视为窗口内消失）。该实现按时间窗口、严格 `< t`、单次 chronological scan 摊还 O(1)~O(log n)，不引入跨数据集绝对窗口硬编码。
- `decayed_degree` 按连续时间 Hawkes 式增量维护，`burstiness` 取最近 ≤20 个正间隔；二者无需窗口去重，成本低。

## Experiment Protocol

所有 Python 命令必须使用：

```bash
source /home/zyh/anaconda3/etc/profile.d/conda.sh && conda activate dyg && <python command>
```

执行顺序：

1. 生成 `v1_trainfit` 和 `v2_trainfit` Social 特征，覆盖 Wikipedia/Reddit，但只写入 `experiments/*/feature_bank/`，不覆盖 `processed_data/*/social_v1.npy`。
2. 做特征 sanity check：row 0 全零、shape 正确、无 NaN/Inf、LP 训练边 normalization 均值接近 0/std 接近 1。
3. Smoke Test：Wikipedia/Reddit 各跑 `Base`、`Social v1_trainfit`、`Social v2_trainfit` 的 1 seed、3 epoch LP，只验证可执行性和产物路径，不据此判断效果。
4. 科学对比阶段固定完全相同超参：相同 seed、相同 `--load_best_configs`、相同 `batch_size`、相同 `num_epochs/patience/test_interval_epochs`。禁止只给 v2 做 batch size sweep。
5. 先跑 5-seed LP 确认：`Base`、`Social v1_trainfit`、`Social v2_trainfit`，每个数据集 seed 0-4。
6. LP 判定只作为“没破坏模型”的必要条件：以 same-seed paired delta 比较 `v2_trainfit - v1_trainfit`，主指标 test AP，辅助指标 test AUC/new-node AP。
7. 只有当两个数据集的 LP paired mean delta 均为正、至少 4/5 seed 的 test AP delta 为正、且 new-node AP 无系统退化时，才继续跑 NC。
8. NC 使用 5 seed，并作为 `Social v2` 是否值得保留的主要证据。NC 必须使用相同 `experiment_dir`、`feature_bank_dir`、`feature_version` 和 `--use_social` 加载对应 LP checkpoint。
9. 如果 LP 未通过 gate，不启动 NC 和完整后续实验；先分析 feature distribution、paired delta、per-seed 退化和 normalization 影响。

## Compute Policy

- 为避免混杂，Base/v1/v2 的正式对比不做 batch size 调参，沿用同一套 best config 和默认 batch size。
- 服务器资源利用方式：特征生成可并行跑 Wikipedia/Reddit；GPU 训练按 seed 队列串行执行，确保单个实验独占 RTX 4090，避免多进程抢显存影响稳定性。
- 若后续只为工程吞吐做 batch size probe，必须在所有配置上使用同一 batch size，并作为单独工程实验记录，不混入科学对比表。

## Test And Analysis Plan

- 静态检查：
  - 新 CLI 默认值不改变 legacy 路径。
  - 设置 `--experiment_dir` 后，模型、结果和日志都进入对应实验目录。
  - `feature_version` 不再被误认为能单独隔离产物。
- 特征检查：
  - `social_v1_trainfit.npy`: `(num_edges + 1, 10)`。
  - `social_v2_trainfit.npy`: `(num_edges + 1, 20)`。
  - row 0 全零，无 NaN/Inf。
  - LP 训练边归一化后每列 mean 约 0、std 约 1。
- Smoke Test 验收：
  - 训练不崩溃。
  - checkpoint 和 result JSON 写入对应 `experiments/*` 目录。
  - AP/AUC 非 NaN，无明显异常。
- LP 5-seed 分析：
  - 输出 per-seed paired delta 表。
  - 输出 mean/std、95% CI、paired t-test p-value；p-value 用作证据强弱，不作为唯一开关。
  - 主报告比较 `Base`、`Social v1_trainfit`、`Social v2_trainfit`，不与 v1 全量预处理结果混排。
- v2 逐维增量诊断（验证“是否冗余”假设，避免 v2 沦为更小号的 Style/Topic）：
  - 计算 v2 新增 5 维与 v1 对应 5 维的相关性，量化冗余程度。
  - 做 v2 逐维（或逐组：windowed-count 组 vs `burstiness`/`decayed_degree` 组）的简易 label-AUC 或留一消融诊断，判断增益来自哪几维。
  - 若 v2 增益主要来自 `burstiness`/`decayed_degree`，在结论中明确；若 windowed-count 组无独立贡献，明确归因为与 v1 冗余。
- NC 5-seed 分析：
  - 输出 ROC-AUC mean/std 和 per-seed delta。
  - 结论按均值和方差解释，禁止依赖单个 outlier seed。
  - `Social v2` 保留标准：两个数据集 NC paired mean delta 不为负，且至少一个数据集有清晰正向趋势；若 Reddit 退化超过 0.01 ROC-AUC，视为不通过。
  - 噪声地板提示：NC std 约 0.02–0.04、n=5，则均值标准误约 0.013，0.01 量级变化本就在噪声范围内。因此 NC 给出**统计不显著/不确定（inconclusive）是一个合法且需如实记录的结局**，不得为通过门槛而强行解读趋势或挑选有利 seed。inconclusive 时的处理：保留 `Social v2` 当且仅当它在 LP 上已稳定优于 v1，且 NC 至少未出现系统性退化。
- 报告规范：
  - 每张表必须标注 `feature_version`、`feature_bank_dir`、`experiment_dir`、`preprocessing_protocol`。
  - Style/Topic/Personality 只作为 v1 全量预处理 proxy 消融引用，并显式说明 full-data preprocessing caveat；不得与 train-only Social 结果直接等价比较。

## Assumptions

- 本轮主攻 Wikipedia/Reddit，因为四特征消融和 NC 标签主要集中在这两个数据集。
- `Social v2` 不做 Full 组合；Full 只有在单特征稳定正向后才值得重新测试。
- `v1_trainfit/v2_trainfit` 的 normalization 使用 LP 真实训练边定义，并同时用于 LP 和 NC，以保持 NC checkpoint 加载与特征分布一致。
- 如果运行中发现缺依赖，先执行 `conda activate dyg && proxy && pip install <package>`；当前计划应主要使用已有 numpy/pandas/sklearn/PyTorch 依赖。
