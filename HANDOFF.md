# HANDOFF.md — DyGFormer 实验交接文档

> 最后更新: 2026-04-28 11:42 UTC+8

---

## 1. 当前进度总览

| 任务 | 状态 | 核心指标 |
|------|------|---------|
| mooc 数据集格式转换 | **已完成** | `processed_data/mooc/ml_mooc.csv` 等 |
| mooc LP 训练 (DyGFormer, 5 runs) | **已完成** | test AP 0.8660±0.0039, AUC 0.8727±0.0043 |
| mooc NC 训练 (DyGFormer, 5 runs) | **已完成** | test ROC-AUC 0.7860±0.0062 |
| mooc LP 评估 (evaluate_link_prediction.py) | **已完成** | test AP 0.8656±0.0049, AUC 0.8719±0.0045 |
| mooc NC 评估 (evaluate_node_classification.py) | **已完成** | test ROC-AUC 0.7860±0.0062 |
| temfin 数据集适配 | **已完成** | symlink + load_configs 扩展 |
| temfin LP 训练 (DyGFormer, 5 runs) | **已完成** | test AP 0.9887±0.0004, AUC 0.9841±0.0007 |
| temfin NC 训练 (DyGFormer, 5 runs) | **已完成** | test ROC-AUC 0.6068±0.0606 |
| temfin LP 评估 (evaluate_link_prediction.py) | **已完成** | test AP 0.9887±0.0004, AUC 0.9841±0.0007 |
| temfin NC 评估 (evaluate_node_classification.py) | **已完成** | test ROC-AUC 0.6068±0.0606 |
| DyGFormer GPU 优化 | **已完成** | 训练速度 4.0→8.9 it/s, GPU 利用率 38%→81% |

**目前 mooc 和 temfin 的 baseline DyGFormer 实验（训练+评估）已全部完成。**

---

## 2. 服务器环境

- Ubuntu 22.04, 32 核 CPU, 256GB RAM, 1x RTX 4090 (24GB)
- Conda 环境: `dyg`
- 激活命令: `conda activate dyg`
- 若需安装依赖: 先执行 `proxy` 再 `pip install`

---

## 3. 目录结构

```
DyGLib/
├── processed_data/
│   ├── mooc/              # ml_mooc.csv, ml_mooc.npy, ml_mooc_node.npy
│   └── temfin/            # ml_temfin.* (symlink → ml_Tem_Fin.*)
├── saved_models/DyGFormer/
│   ├── mooc/              # DyGFormer_base_seed{0..4}/, node_classification_*
│   └── temfin/            # DyGFormer_base_seed{0..4}/, node_classification_*
├── saved_results/DyGFormer/
│   ├── mooc/              # 见下方产物清单
│   └── temfin/            # 见下方产物清单
├── experiment_logs/       # nohup 监控日志（手动挂起的进程输出）
│   ├── train_mooc_dygformer.log
│   ├── train_mooc_node_classification.log
│   ├── train_temfin_dygformer.log
│   ├── train_temfin_dygformer_seed4.log
│   ├── train_temfin_node_classification.log
│   ├── eval_mooc_lp.log
│   ├── eval_temfin_lp.log
│   ├── eval_mooc_nc.log
│   └── eval_temfin_nc.log
├── experiment_analysis/   # 实验结果分析文档
│   ├── feature_ablation_analysis.md       # Wikipedia/Reddit 四特征消融分析
│   ├── feature_ablation_results.md        # 消融实验结果表
│   ├── feature_ablation_all_results.csv   # 消融实验全量 CSV
│   ├── collect_results.py                 # 结果收集脚本
│   └── model_evaluation_link_prediction.md  # mooc/temfin LP 评估分析
└── logs/DyGFormer/        # 模型内部自动生成的训练/评估日志（每 run 一个 .log 文件）
```

---

## 4. 数据集

### 4.1 mooc

```
processed_data/mooc/
├── ml_mooc.csv          # 411,749 条交互, 7,144 个节点
├── ml_mooc.npy          # 边特征 (411750, 172)
├── ml_mooc_node.npy     # 节点特征 (7145, 172), 全零
└── ml_mooc_meta.json    # 元数据
```

转换脚本: `preprocess_data/prepare_jodie_mooc_for_dyglib.py`

### 4.2 temfin

```
processed_data/temfin/
├── ml_Tem_Fin.csv       # 原始文件 (709,774 条交互)
├── ml_Tem_Fin.npy       # 原始边特征 (709775, 154)
├── ml_Tem_Fin_node.npy  # 原始节点特征 (34069, 172)
├── ml_temfin.csv        # symlink → ml_Tem_Fin.csv
├── ml_temfin.npy        # symlink → ml_Tem_Fin.npy
└── ml_temfin_node.npy   # symlink → ml_Tem_Fin_node.npy
```

数据集统计:
- 709,774 条交互，33,245 个唯一节点（src: 24,677，dst: 32,495）
- 节点 ID 范围: [1, 34,068]，节点特征文件形状 (34,069, 172) 已完全覆盖
- 时间戳范围: [2, 86,399]（单日内，秒级）
- 标签: 正样本率 0.71%（5,067 / 709,774），高度不平衡

---

## 5. 所有已完成实验的命令与结果

### 5.1 mooc — Link Prediction 训练

**命令:**
```bash
python train_link_prediction.py \
  --dataset_name mooc --model_name DyGFormer \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 600 --num_runs 5 --gpu 0
```

**结果 (5 runs):**

| Seed | Val AP | Val AUC | NN Val AP | NN Val AUC | Test AP | Test AUC | NN Test AP | NN Test AUC |
|------|--------|---------|-----------|------------|---------|----------|------------|-------------|
| 均值±std | — | — | — | — | **0.8660±0.0039** | **0.8727±0.0043** | 0.8641±0.0038 | 0.8722±0.0036 |

**产物:** `saved_models/DyGFormer/mooc/DyGFormer_base_seed{0..4}/`，`saved_results/DyGFormer/mooc/DyGFormer_base_seed{0..4}.json`

---

### 5.2 mooc — Node Classification 训练

**命令:**
```bash
python train_node_classification.py \
  --dataset_name mooc --model_name DyGFormer \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 600 --num_runs 5 --gpu 0
```

**结果 (5 runs):**

| 指标 | Validate | Test |
|------|----------|------|
| ROC-AUC | 0.8100±0.0034 | **0.7860±0.0062** |

**产物:** `saved_models/DyGFormer/mooc/node_classification_DyGFormer_base_seed{0..4}/`，`saved_results/DyGFormer/mooc/node_classification_DyGFormer_base_seed{0..4}.json`

---

### 5.3 mooc — Link Prediction 评估

**命令:**
```bash
python evaluate_link_prediction.py \
  --dataset_name mooc --model_name DyGFormer \
  --negative_sample_strategy random --load_best_configs \
  --num_runs 5 --gpu 0
```

注: `--load_best_configs` 对 mooc 有效（对应 seq_len=256, patch_size=8，与训练一致）。

**结果 (5 runs):**

| Seed | Val AP | Val AUC | NN Val AP | NN Val AUC | Test AP | Test AUC | NN Test AP | NN Test AUC |
|------|--------|---------|-----------|------------|---------|----------|------------|-------------|
| 0 | 0.8770 | 0.8851 | 0.8526 | 0.8614 | 0.8695 | 0.8746 | 0.8646 | 0.8732 |
| 1 | 0.8776 | 0.8852 | 0.8539 | 0.8601 | 0.8706 | 0.8769 | 0.8657 | 0.8738 |
| 2 | 0.8688 | 0.8766 | 0.8458 | 0.8536 | 0.8597 | 0.8654 | 0.8558 | 0.8646 |
| 3 | 0.8682 | 0.8782 | 0.8471 | 0.8569 | 0.8614 | 0.8694 | 0.8586 | 0.8701 |
| 4 | 0.8747 | 0.8833 | 0.8491 | 0.8570 | 0.8666 | 0.8731 | 0.8625 | 0.8707 |
| **均值±std** | 0.8733±0.0045 | 0.8817±0.0040 | 0.8497±0.0035 | 0.8578±0.0031 | **0.8656±0.0049** | **0.8719±0.0045** | 0.8614±0.0041 | 0.8705±0.0036 |

**产物:** `saved_results/DyGFormer/mooc/random_negative_sampling_DyGFormer_base_seed{0..4}.json`

---

### 5.4 mooc — Node Classification 评估

**命令:**
```bash
python evaluate_node_classification.py \
  --dataset_name mooc --model_name DyGFormer \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 600 --num_runs 5 --gpu 0
```

注: **禁止使用 `--load_best_configs`**，NC 的 best_configs 对非 reddit 数据集会设置 seq_len=32/patch_size=1，与训练时 256/8 不匹配会导致 checkpoint 加载失败。

**结果 (5 runs):**

| Seed | Val ROC-AUC | Test ROC-AUC |
|------|-------------|--------------|
| 0 | 0.8134 | 0.7951 |
| 1 | 0.8043 | 0.7860 |
| 2 | 0.8108 | 0.7826 |
| 3 | 0.8117 | 0.7785 |
| 4 | 0.8099 | 0.7879 |
| **均值±std** | **0.8100±0.0034** | **0.7860±0.0062** |

**产物:** `saved_results/DyGFormer/mooc/evaluate_node_classification_DyGFormer_base_seed{0..4}.json`

---

### 5.5 temfin — Link Prediction 训练

**命令:**
```bash
python train_link_prediction.py \
  --dataset_name temfin --model_name DyGFormer \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 2000 --num_runs 5 --gpu 0
```

**结果 (5 runs):**

| Seed | Test AP | Test AUC | NN Test AP | NN Test AUC |
|------|---------|----------|------------|-------------|
| 0 | 0.9881 | 0.9831 | 0.9146 | 0.8777 |
| 1 | 0.9885 | 0.9837 | 0.9152 | 0.8778 |
| 2 | 0.9889 | 0.9842 | 0.9168 | 0.8804 |
| 3 | 0.9891 | 0.9847 | 0.9156 | 0.8792 |
| 4 | 0.9890 | 0.9846 | 0.9151 | 0.8786 |
| **均值±std** | **0.9887±0.0004** | **0.9841±0.0007** | 0.9155±0.0008 | 0.8787±0.0011 |

**产物:** `saved_models/DyGFormer/temfin/DyGFormer_base_seed{0..4}/`，`saved_results/DyGFormer/temfin/DyGFormer_base_seed{0..4}.json`

备注: seed4 曾因服务器断电在 Epoch 7 中断，通过临时修改 `range(4, args.num_runs)` 单独重跑，最终结果与其余 seed 一致。

---

### 5.6 temfin — Node Classification 训练

**命令:**
```bash
python train_node_classification.py \
  --dataset_name temfin --model_name DyGFormer \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 4000 --num_runs 5 --gpu 0
```

**结果 (5 runs):**

| Seed | Val ROC-AUC | Test ROC-AUC |
|------|-------------|--------------|
| 0 | 0.6109 | 0.6361 |
| 1 | 0.5431 | 0.5434 |
| 2 | 0.5759 | 0.5621 |
| 3 | 0.5940 | 0.5976 |
| 4 | 0.6881 | 0.6948 |
| **均值±std** | 0.6024±0.0541 | **0.6068±0.0606** |

方差极大，原因：标签高度不平衡（正样本率 0.71%），BCELoss 未做 class weighting。

**产物:** `saved_models/DyGFormer/temfin/node_classification_DyGFormer_base_seed{0..4}/`，`saved_results/DyGFormer/temfin/node_classification_DyGFormer_base_seed{0..4}.json`

---

### 5.7 temfin — Link Prediction 评估

**命令:**
```bash
python evaluate_link_prediction.py \
  --dataset_name temfin --model_name DyGFormer \
  --negative_sample_strategy random \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 2000 --num_runs 5 --gpu 0
```

注: **禁止使用 `--load_best_configs`**，temfin 不在任何已知列表中，会落入 else 分支 (seq_len=32, patch_size=1)。

**结果 (5 runs):**

| Seed | Val AP | Val AUC | NN Val AP | NN Val AUC | Test AP | Test AUC | NN Test AP | NN Test AUC |
|------|--------|---------|-----------|------------|---------|----------|------------|-------------|
| 0 | 0.9869 | 0.9818 | 0.9291 | 0.9000 | 0.9881 | 0.9831 | 0.9146 | 0.8777 |
| 1 | 0.9879 | 0.9830 | 0.9358 | 0.9092 | 0.9885 | 0.9837 | 0.9152 | 0.8778 |
| 2 | 0.9883 | 0.9835 | 0.9356 | 0.9084 | 0.9889 | 0.9842 | 0.9168 | 0.8804 |
| 3 | 0.9882 | 0.9837 | 0.9341 | 0.9072 | 0.9891 | 0.9847 | 0.9156 | 0.8792 |
| 4 | 0.9882 | 0.9837 | 0.9341 | 0.9069 | 0.9890 | 0.9846 | 0.9151 | 0.8786 |
| **均值±std** | 0.9879±0.0006 | 0.9832±0.0008 | 0.9337±0.0027 | 0.9064±0.0037 | **0.9887±0.0004** | **0.9841±0.0007** | 0.9155±0.0008 | 0.8787±0.0011 |

**产物:** `saved_results/DyGFormer/temfin/random_negative_sampling_DyGFormer_base_seed{0..4}.json`

---

### 5.8 temfin — Node Classification 评估

**命令:**
```bash
python evaluate_node_classification.py \
  --dataset_name temfin --model_name DyGFormer \
  --max_input_sequence_length 256 --patch_size 8 \
  --dropout 0.1 --batch_size 4000 --num_runs 5 --gpu 0
```

**结果 (5 runs):**

| Seed | Val ROC-AUC | Test ROC-AUC |
|------|-------------|--------------|
| 0 | 0.6109 | 0.6361 |
| 1 | 0.5431 | 0.5434 |
| 2 | 0.5759 | 0.5621 |
| 3 | 0.5940 | 0.5976 |
| 4 | 0.6881 | 0.6948 |
| **均值±std** | 0.6024±0.0541 | **0.6068±0.0606** |

**产物:** `saved_results/DyGFormer/temfin/evaluate_node_classification_DyGFormer_base_seed{0..4}.json`

---

## 6. 完整产物清单

### saved_results/DyGFormer/mooc/（20 个 JSON）

| 文件模式 | 对应任务 |
|----------|----------|
| `DyGFormer_base_seed{0..4}.json` | LP 训练阶段 test 指标 |
| `node_classification_DyGFormer_base_seed{0..4}.json` | NC 训练阶段 test 指标 |
| `random_negative_sampling_DyGFormer_base_seed{0..4}.json` | LP 评估阶段（evaluate_link_prediction.py） |
| `evaluate_node_classification_DyGFormer_base_seed{0..4}.json` | NC 评估阶段（evaluate_node_classification.py） |

### saved_results/DyGFormer/temfin/（20 个 JSON，结构同上）

### saved_models/DyGFormer/

| 路径 | 内容 |
|------|------|
| `mooc/DyGFormer_base_seed{0..4}/` | mooc LP backbone checkpoint |
| `mooc/node_classification_DyGFormer_base_seed{0..4}/` | mooc NC (backbone+head) checkpoint |
| `temfin/DyGFormer_base_seed{0..4}/` | temfin LP backbone checkpoint |
| `temfin/node_classification_DyGFormer_base_seed{0..4}/` | temfin NC (backbone+head) checkpoint |

---

## 7. 代码改动（相对于原始 DyGLib）

### 7.1 `utils/load_configs.py`

1. LP 的 `--dataset_name` choices 添加 `'temfin'`（第 85 行）
2. NC 的 `--dataset_name` choices 扩展为 `['wikipedia','reddit','mooc','temfin']`（第 320 行）
3. NC 的 assert 同步扩展（第 360 行）

### 7.2 `models/DyGFormer.py` — GPU 性能优化

1. **`count_nodes_appearances` 向量化:** 原 Python for 循环改为 GPU 批量 `scatter_add_` 直方图。训练速度 4.0→8.9 it/s，GPU 利用率 38%→81%。
2. **新增 `get_padded_neighbors`:** 将 `get_all_first_hop_neighbors` + `pad_sequences` 合并为单次遍历。`compute_src_dst_node_temporal_embeddings` 已更新为调用此方法。两项优化数值验证正确（max diff = 0.0）。

### 7.3 Feature Extension 框架（更早，与本系列实验无关）

`train_link_prediction.py`、`evaluate_link_prediction.py`、`train_node_classification.py`、`evaluate_node_classification.py`、`utils/load_configs.py`、`utils/DataLoader.py`、`models/DyGFormer.py` 均添加了 Style/Personality/Topic/Social 特征融合框架。未启用 `--use_*` flag 时为 no-op，不影响 baseline。

---

## 8. 关键约束与注意事项（必读）

### 8.1 参数匹配约束（最重要）

NC 从 LP checkpoint 加载 backbone。以下场景会导致 **权重形状不匹配报错**：

| 场景 | 正确做法 |
|------|---------|
| mooc NC 训练/评估 | 必须指定 `--max_input_sequence_length 256 --patch_size 8`，与 LP 训练时一致 |
| temfin NC 训练/评估 | 必须指定 `--max_input_sequence_length 256 --patch_size 8`，与 LP 训练时一致 |
| temfin LP 评估 | 必须手动指定参数，**不能用 `--load_best_configs`** |
| mooc NC/temfin NC 评估 | 必须手动指定参数，**不能用 `--load_best_configs`**（会错误设置 seq_len=32/patch_size=1） |
| mooc LP 评估 | 可以用 `--load_best_configs`（mooc 在 best_configs 中对应 256/8） |

### 8.2 temfin NC 方差大

正样本率仅 0.71%，BCELoss 未做 class weighting，各 seed 方差 std ≈ 0.06。若需改善：在 `train_node_classification.py` 中将 `nn.BCELoss()` 改为 `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([140.0]))` 并去掉 sigmoid 调用。

### 8.3 GPU 显存估算

| 任务 | batch_size | 峰值显存 |
|------|------------|---------|
| mooc LP 训练 | 600 | ~8GB |
| temfin LP 训练 | 2000 | ~17GB |
| temfin NC 训练/评估 | 4000 | ~13GB（无反向传播） |
| temfin LP 评估 | 2000 | ~17GB |

`count_nodes_appearances` 分配 `batch_size × max_node_id × 2` GPU 张量。temfin（34K 节点, batch=2000）≈ 544MB。

### 8.4 temfin 文件命名

原始文件名含大写 `Tem_Fin`，DataLoader 以 `ml_{dataset_name}.*` 读取，需要 symlink。symlink 已建，无需重复操作。

---

## 9. 关键文件索引

| 路径 | 用途 |
|------|------|
| `train_link_prediction.py` | LP 训练入口 |
| `train_node_classification.py` | NC 训练入口（加载 LP backbone） |
| `evaluate_link_prediction.py` | LP 评估入口（加载已保存 checkpoint） |
| `evaluate_node_classification.py` | NC 评估入口（加载已保存 checkpoint） |
| `models/DyGFormer.py` | DyGFormer 模型（含 GPU 优化） |
| `utils/load_configs.py` | 参数解析 + 各数据集最佳配置 |
| `utils/DataLoader.py` | 数据加载 + train/val/test 划分 |
| `experiment_analysis/` | 所有实验分析文档 |
| `experiment_logs/` | 所有 nohup 监控日志 |
| `preprocess_data/prepare_jodie_mooc_for_dyglib.py` | mooc 数据转换脚本 |
| `features/` | Feature extension 框架（当前 baseline 实验未启用） |
