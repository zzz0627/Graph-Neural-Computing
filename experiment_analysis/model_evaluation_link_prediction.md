# Model Evaluation — Link Prediction (DyGFormer Baseline)

> 评估时间: 2026-04-28
> 评估脚本: `evaluate_link_prediction.py`
> 负采样策略: random
> 评估模式: 加载已训练 checkpoint，仅推理（无训练）

---

## 1. 实验配置

| 数据集 | 模型 | seq_len | patch_size | dropout | batch_size | num_runs | GPU | 使用 --load_best_configs |
|--------|------|---------|------------|---------|------------|----------|-----|--------------------------|
| mooc | DyGFormer | 256 | 8 | 0.1 | 200 (default) | 5 | 0 | 是 |
| temfin | DyGFormer | 256 | 8 | 0.1 | 2000 | 5 | 0 | 否（手动指定） |

**temfin 不使用 `--load_best_configs` 的原因:** `load_link_prediction_best_configs` 中 DyGFormer 分支未包含 temfin，会落入 else 分支（seq_len=32, patch_size=1），与训练时的 256/8 不匹配，导致 checkpoint 加载失败。

**命令:**

```bash
# mooc
python evaluate_link_prediction.py \
  --dataset_name mooc --model_name DyGFormer \
  --negative_sample_strategy random --load_best_configs --num_runs 5 --gpu 0

# temfin
python evaluate_link_prediction.py \
  --dataset_name temfin --model_name DyGFormer \
  --negative_sample_strategy random \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 2000 --num_runs 5 --gpu 0
```

---

## 2. mooc — 评估结果

### 2.1 汇总 (5 runs 均值±标准差)

| 指标 | Validate | New Node Validate | Test | New Node Test |
|------|----------|-------------------|------|---------------|
| Average Precision | 0.8733 ± 0.0045 | 0.8497 ± 0.0035 | **0.8656 ± 0.0049** | 0.8614 ± 0.0041 |
| ROC-AUC | 0.8817 ± 0.0040 | 0.8578 ± 0.0031 | **0.8719 ± 0.0045** | 0.8705 ± 0.0036 |

### 2.2 各 Seed 明细

| Seed | Val AP | Val AUC | NN Val AP | NN Val AUC | Test AP | Test AUC | NN Test AP | NN Test AUC |
|------|--------|---------|-----------|------------|---------|----------|------------|-------------|
| 0 | 0.8770 | 0.8851 | 0.8526 | 0.8614 | 0.8695 | 0.8746 | 0.8646 | 0.8732 |
| 1 | 0.8776 | 0.8852 | 0.8539 | 0.8601 | 0.8706 | 0.8769 | 0.8657 | 0.8738 |
| 2 | 0.8688 | 0.8766 | 0.8458 | 0.8536 | 0.8597 | 0.8654 | 0.8558 | 0.8646 |
| 3 | 0.8682 | 0.8782 | 0.8471 | 0.8569 | 0.8614 | 0.8694 | 0.8586 | 0.8701 |
| 4 | 0.8747 | 0.8833 | 0.8491 | 0.8570 | 0.8666 | 0.8731 | 0.8625 | 0.8707 |

### 2.3 与训练阶段结果对比

| 指标 | 训练阶段 Test | 评估阶段 Test | 差值 |
|------|---------------|---------------|------|
| Average Precision | 0.8660 ± 0.0039 | 0.8656 ± 0.0049 | -0.0004 |
| ROC-AUC | 0.8727 ± 0.0043 | 0.8719 ± 0.0045 | -0.0008 |

训练与评估结果高度一致（差值 < 0.001），验证了 checkpoint 正确性和评估流程的可靠性。微小差异来自评估阶段使用 `full_neighbor_sampler`（含全部数据的邻居信息），而训练阶段 test 使用的 neighbor sampler 可能有细微区别。

---

## 3. temfin — 评估结果

### 3.1 汇总 (5 runs 均值±标准差)

| 指标 | Validate | New Node Validate | Test | New Node Test |
|------|----------|-------------------|------|---------------|
| Average Precision | 0.9879 ± 0.0006 | 0.9337 ± 0.0027 | **0.9887 ± 0.0004** | 0.9155 ± 0.0008 |
| ROC-AUC | 0.9832 ± 0.0008 | 0.9064 ± 0.0037 | **0.9841 ± 0.0007** | 0.8787 ± 0.0011 |

### 3.2 各 Seed 明细

| Seed | Val AP | Val AUC | NN Val AP | NN Val AUC | Test AP | Test AUC | NN Test AP | NN Test AUC |
|------|--------|---------|-----------|------------|---------|----------|------------|-------------|
| 0 | 0.9869 | 0.9818 | 0.9291 | 0.9000 | 0.9881 | 0.9831 | 0.9146 | 0.8777 |
| 1 | 0.9879 | 0.9830 | 0.9358 | 0.9092 | 0.9885 | 0.9837 | 0.9152 | 0.8778 |
| 2 | 0.9883 | 0.9835 | 0.9356 | 0.9084 | 0.9889 | 0.9842 | 0.9168 | 0.8804 |
| 3 | 0.9882 | 0.9837 | 0.9341 | 0.9072 | 0.9891 | 0.9847 | 0.9156 | 0.8792 |
| 4 | 0.9882 | 0.9837 | 0.9341 | 0.9069 | 0.9890 | 0.9846 | 0.9151 | 0.8786 |

### 3.3 与训练阶段结果对比

| 指标 | 训练阶段 Test | 评估阶段 Test | 差值 |
|------|---------------|---------------|------|
| Average Precision | 0.9887 ± 0.0004 | 0.9887 ± 0.0004 | 0.0000 |
| ROC-AUC | 0.9841 ± 0.0007 | 0.9841 ± 0.0007 | 0.0000 |

训练与评估结果完全一致（差值 = 0），checkpoint 验证通过。

---

## 4. 跨数据集对比分析

| 数据集 | 交互数 | 节点数 | Test AP | Test AUC | NN Test AP | NN Test AUC | Transductive-Inductive Gap (AP) |
|--------|--------|--------|---------|----------|------------|-------------|-------------------------------|
| mooc | 411K | 7,144 | 0.8656 | 0.8719 | 0.8614 | 0.8705 | 0.0042 |
| temfin | 710K | 33,245 | 0.9887 | 0.9841 | 0.9155 | 0.8787 | 0.0732 |

### 关键观察

1. **temfin Transductive 性能远高于 mooc:** temfin Test AP (0.9887) 比 mooc (0.8656) 高 12+ 个百分点。temfin 的金融交易图具有更强的结构规律性（重复交易模式），使得 link prediction 更容易。

2. **temfin Transductive-Inductive Gap 显著更大:** temfin 的 gap 为 0.0732 AP，远大于 mooc 的 0.0042。这表明 temfin 中新节点（未在训练集出现的交易方）的交互模式与已知节点有较大差异，模型对新节点的泛化较弱。金融网络中新用户的行为模式确实难以从已有用户推断。

3. **mooc Inductive 性能接近 Transductive:** mooc 的 gap 仅 0.0042 AP，说明 MOOC 平台上新用户的课程选择模式与已有用户高度相似，DyGFormer 能很好地泛化。

4. **两个数据集的 seed 间方差特征不同:**
   - mooc std ≈ 0.004-0.005（中等方差），各 seed 间性能波动明显
   - temfin std ≈ 0.0004-0.0008（极低方差），5 个 seed 高度一致，模型收敛稳定

---

## 5. 产物清单

| 类型 | 路径 |
|------|------|
| 评估结果 JSON (mooc) | `saved_results/DyGFormer/mooc/random_negative_sampling_DyGFormer_base_seed{0..4}.json` |
| 评估结果 JSON (temfin) | `saved_results/DyGFormer/temfin/random_negative_sampling_DyGFormer_base_seed{0..4}.json` |
| 评估日志 (mooc) | `experiment_logs/eval_mooc_lp.log` |
| 评估日志 (temfin) | `experiment_logs/eval_temfin_lp.log` |
| 模型内部日志 (mooc) | `logs/DyGFormer/mooc/random_negative_sampling_DyGFormer_base_seed{0..4}/` |
| 模型内部日志 (temfin) | `logs/DyGFormer/temfin/random_negative_sampling_DyGFormer_base_seed{0..4}/` |
