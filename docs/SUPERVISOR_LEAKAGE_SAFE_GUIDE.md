# 导师运行说明 — 无泄漏行为建模（carlo_dev 分支）

本文档面向在实验室服务器上复现 **Stage 8 行为预测模型** 的导师/审稿人。代码仓库：

**https://github.com/Summercoconutt/DAOsynthvoter/tree/carlo_dev**

请务必使用 **`carlo_dev` 分支**，不要使用 `main`（`main` 缺少 Stage 09 评估、Stage 10 检测模式及本次泄漏修复）。

---

## 1. 本次更新做了什么（为何必须重新训练）

此前版本存在 **数据泄漏**，会导致测试集指标虚高、论文结论不可信。本次在 `carlo_dev` 上已修复，主要变更如下：

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| `aligned_with_majority`、`vp_share` | 作为模型数值输入 | **已从模型输入中移除** |
| 滑动窗口分组 | 仅按 `voter` | 按 **`(voter, space)`** |
| DAO / 选民聚类 | Stage 6/7 全量数据 + 含标签统计特征 | **仅在训练集选民上拟合**，再分配到 val/test |
| Stage 7 输出 | 直接喂给行为模型 | **仅作探索性分析（EDA）**，不参与 Stage 8 |
| 模型数值特征维度 | 10（含泄漏列） | **8**（4 个数值 + 4 个时间编码） |

**重要：** 用旧代码训练的 checkpoint（`feat_dim=10`）与本次代码 **不兼容**，必须重新运行 Stage 08。

---

## 2. 环境准备

```bash
git clone https://github.com/Summercoconutt/DAOsynthvoter.git
cd DAOsynthvoter
git checkout carlo_dev
git pull

python3 -m pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src"    # Windows PowerShell: $env:PYTHONPATH = "src"
```

- Python **3.10+**
- GPU 可选；RoBERTa 训练建议 CUDA；可用 `configs/smoke_behaviour.yaml` 做快速冒烟
- Stage 08b（无 RoBERTa 数值基线）需要 **duckdb**（已在 `requirements.txt`）

---

## 3. 数据与路径配置

原始投票数据通常在服务器磁盘（例如 `D:/111111/Data`），**不要**把清洗结果写回原始目录。

1. 复制并编辑 `configs/example_server_raw.yaml`，将 `paths.master_votes_parquet` 设为 **绝对路径** 指向未清洗 parquet。
2. 运行 Stage 03 时合并配置：

```bash
python scripts/03_run_global_cleaning.py \
  --config configs/default.yaml \
  --extra-config configs/example_server_raw.yaml
```

清洗输出仅写入本仓库下的 `data/processed/votes_cleaned.parquet`（见 `configs/default.yaml` 中 `paths.cleaned_master_parquet`）。

### Stage 8 最低前置条件（不必跑完 Stage 5–7）

| 文件 | 产生阶段 | 用途 |
|------|----------|------|
| `data/processed/votes_cleaned.parquet` | Stage 03 | 行为数据集构建 |
| `data/processed/dao_feature_table.csv` | Stage 04 | 训练集上拟合 **结构型** DAO 聚类 |

Stage 6（DAO 聚类）、Stage 7（选民聚类）**可选**，仅用于论文中的聚类探索图/表，**不再**作为行为模型输入来源。

---

## 4. 推荐运行顺序（服务器）

### 4.1 检测模式（先跑，确认路径与列名正确）

```bash
# 只检查前置文件（不扫描原始盘）
python scripts/10_run_detection_mode.py --skip-raw-scan

# 可选：1 epoch 冒烟训练（RoBERTa + 数值特征）
python scripts/10_run_detection_mode.py --skip-raw-scan --smoke-train
```

报告：`outputs/reports/detection_mode_report.md`

### 4.2 若尚未清洗 / 建 DAO 特征表

```bash
python scripts/03_run_global_cleaning.py --config configs/default.yaml --extra-config configs/example_server_raw.yaml
python scripts/04_run_dao_metrics.py --config configs/default.yaml
```

### 4.3 行为模型训练（泄漏安全流程）

**RoBERTa 主模型：**

```bash
python scripts/08_run_behaviour_modelling.py --config configs/default.yaml
```

**无 RoBERTa 数值基线（对比实验）：**

```bash
python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml
```

详见 `docs/SUPERVISOR_NO_ROBERTA_RUNBOOK.md`。

### 4.4 训练后综合评估（Stage 09）

使用与训练相同的选民划分（`outputs/processed/split_manifest.json`）：

```bash
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split test
python scripts/09_run_behaviour_evaluation.py --config configs/default.yaml --split val
```

输出目录：`outputs/tables/eval/{test|val}/`（含 F1、ECE、混淆矩阵、按聚类分层指标等）。

### 4.5 自动化测试（验证泄漏修复逻辑）

```bash
python -m pytest tests/test_leakage_safe_pipeline.py -v
```

预期：**9 passed**（无需 GPU、无需真实数据）。

---

## 5. Stage 8 内部流程（供审稿核对）

```
构建 behaviour_dataset.csv（无聚类列）
    → 按 voter 划分 train / val / test
    → 仅在 train 上拟合数值预处理器（VP 截断等）
    → 仅在 train 上拟合结构型 DAO / 选民 KMeans
    → 将聚类 ID 分配到全部划分（transform only）
    → 构建 (voter, space) 滑动窗口
    → 训练 TimeSeriesClassifier（DistilRoBERTa + 数值 + 时间）
```

关键实现文件：

- `src/dao_governance/features/behaviour_pipeline.py` — 编排上述流程
- `src/dao_governance/features/causal_clusters.py` — 训练集专用聚类
- `src/dao_governance/modelling/preprocess.py` — 数值列：`voting_power`, `is_whale`, `dao_cluster`, `voter_cluster`
- `src/dao_governance/modelling/windows.py` — 窗口构建与标签历史

---

## 6. 主要输出文件

| 路径 | 说明 |
|------|------|
| `outputs/tables/behaviour_dataset.csv` | 票级数据，**不含**聚类 |
| `outputs/tables/behaviour_dataset_with_clusters.csv` | 含训练集拟合的聚类 ID（08b 缓存用） |
| `outputs/processed/split_manifest.json` | 选民三分划 |
| `outputs/models/predictive_clusters/cluster_bundle.pkl` | 聚类模型（仅 train 拟合） |
| `outputs/models/behaviour_agent2/` | RoBERTa 模型、`config.json`（含 `feat_dim=8`） |
| `outputs/reports/stage08_behaviour_modelling.md` | Stage 8 摘要 |
| `outputs/tables/eval/test/metrics_report.md` | Stage 9 测试集指标 |

---

## 7. 配置项（`configs/default.yaml`）

```yaml
behaviour_model:
  use_dao_clusters: true    # 设为 false 可做消融
  use_voter_clusters: true
  group_windows_by: [voter, space]
```

`predictive_cluster_artifacts_dir` 默认：`outputs/models/predictive_clusters`

---

## 8. 大数据集与缓存（08b）

首次运行 08 或 08b 会生成 split manifest 与 enriched CSV。之后可复用：

```bash
python scripts/08b_run_behaviour_modelling_no_roberta.py --config configs/default.yaml \
  --reuse-split-manifest --reuse-window-cache
```

**注意：** 更新泄漏修复相关代码后，应删除 `outputs/behaviour_modelling/window_cache_no_roberta/` 强制重建缓存。

---

## 9. 已知限制（非泄漏，但需在论文中说明）

1. **`is_whale`**：若上游导出使用全量 q99 阈值，可能存在轻微未来信息；当前保留作结构特征。
2. **DAO 结构指标（Stage 4）**：使用全历史聚合，属时间性局限，非标签泄漏。
3. **聚类 ID**：对每个 `(voter, space)` 在划分内为静态值（仅结构特征，不含投票标签统计）。
4. **文本中的 `[LABEL_k]` 前缀**：仅为同一选民历史投票的自回归输入，当前步标签不作为数值特征。

---

## 10. 故障排查

| 现象 | 处理 |
|------|------|
| `feat_dim` 不匹配 / 加载 `model.pt` 失败 | 删除 `outputs/models/behaviour_agent2/`，重新跑 Stage 08 |
| detection 报缺少 `votes_cleaned.parquet` | 先跑 Stage 03 |
| 报缺少 `dao_feature_table.csv` | 先跑 Stage 04 |
| 窗口数为 0 | 检查清洗后是否有足够 `(voter, space)` 投票；调低 `min_votes_per_pair`（聚类模块内） |
| pytest 失败 | 在仓库根目录执行，确认 `PYTHONPATH=src` |

---

## 11. 相关文档

- `docs/Feature_Specification.md` — 全流水线特征目录与 Stage 8 输入规范（`feat_dim=8`）
- `docs/Leakage_Audit.md` — 泄漏问题登记表、修复状态、严重性排序与验证清单
- `docs/LEAKAGE_SAFE_RUNBOOK.md` — 英文简明运行说明
- `docs/SUPERVISOR_NO_ROBERTA_RUNBOOK.md` — 08b 无 RoBERTa 基线
- `SERVER_CHECKLIST.txt` — 服务器检查清单（简版）

如有问题，请对照 `outputs/reports/` 下各 stage 的 markdown 报告与 `tests/test_leakage_safe_pipeline.py` 中的断言逻辑。
