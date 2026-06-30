# ASLAM 论文基线模型在本地数据集上的统一实验报告

## 1. 任务范围

本报告汇总 ASLAM 论文中列出的主要对比基线模型，在当前 `ASLAM-3` 根目录统一协议下的本地实验结果。
报告同时附带根目录已正式完成的 `ASLAM` 与 `ASLAM-SSAF` 结果，以便直接形成总对比表。

## 2. 统一 protocol

- 设备：`cuda:0`
- runs：`1`
- epochs：`3`
- patience：`2`
- seed 起点：`2`
- val/test：`0.05` / `0.1`
- neg sampling：`1.0`
- 数据集：`Citeseer, DBLP, PubMed, amz_Photo, CoRA, Twitch_EN`

## 3. 学术规范说明

1. 本报告只把本地固定数据、固定切分和固定随机种子协议下得到的结果，表述为 `local formal comparison`。
2. 若某 baseline 采用的是根目录重建实现或恢复出的作者代码片段，而不是独立官方仓库逐字运行，报告中会明确写出，不混淆为官方严格复现。
3. `ASLAM` 与 `ASLAM-SSAF` 行来自根目录已有 formal compare JSON，不与新跑的 baseline 混用不同 protocol。

## 4. 代码来源与实现口径

| 模型 | 实现口径 |
| --- | --- |
| CN | 本地闭式启发式实现；无外部代码依赖。 |
| AA | 本地闭式启发式实现；无外部代码依赖。 |
| Node2Vec | 基于 PyTorch Geometric 2.7.0 官方实现。 |
| GCN | 基于 PyTorch Geometric 2.7.0 官方组件。 |
| GAE | 基于 PyTorch Geometric 2.7.0 官方 GAE 实现。 |
| VGAE | 基于 PyTorch Geometric 2.7.0 官方 VGAE 实现。 |
| GraphSAGE | 基于 PyTorch Geometric 2.7.0 官方组件。 |
| GAT | 基于 PyTorch Geometric 2.7.0 官方组件。 |
| SuperGAT | 基于 PyTorch Geometric 2.7.0 组件，并参考官方 SuperGAT 项目。 |
| SEAL | 根目录下按公开 SEAL 范式重建的 1-hop DRNL + DGCNN 子图分类实现。 |
| BSAL | 根目录下依据 BSAL 论文与恢复出的双分量融合代码片段重建；未宣称为独立官方仓库逐字复现。 |
| ASLAM | 根目录正式 baseline，对应已有 formal compare 结果。 |
| ASLAM-SSAF | 根目录正式改进模型，对应已有 formal compare 结果。 |

## 5. 分数据集结果

### 5.1 Citeseer

| 模型 | AUC | AP | 最佳 epoch | 说明 |
| --- | ---: | ---: | ---: | --- |
| CN | 0.6591 +/- 0.0000 | 0.6587 +/- 0.0000 | 0.0 | 本地闭式启发式实现；无外部代码依赖。 |
| AA | 0.6593 +/- 0.0000 | 0.6600 +/- 0.0000 | 0.0 | 本地闭式启发式实现；无外部代码依赖。 |
| Node2Vec | 0.7749 +/- 0.0000 | 0.8028 +/- 0.0000 | 30.0 | 基于 PyTorch Geometric 2.7.0 官方实现。 |
| SEAL | 0.8612 +/- 0.0000 | 0.8879 +/- 0.0000 | 2.0 | 根目录下按公开 SEAL 范式重建的 1-hop DRNL + DGCNN 子图分类实现。 |
| ASLAM | 0.8469 +/- 0.0120 | 0.8540 +/- 0.0177 | 43.9 | 根目录正式 baseline，对应已有 formal compare 结果。 |
| ASLAM-SSAF | 0.8590 +/- 0.0161 | 0.8631 +/- 0.0191 | 42.2 | 根目录正式改进模型，对应已有 formal compare 结果。 |

### 5.2 DBLP

| 模型 | AUC | AP | 最佳 epoch | 说明 |
| --- | ---: | ---: | ---: | --- |
| ASLAM | 0.9013 +/- 0.0110 | 0.9085 +/- 0.0119 | 57.6 | 根目录正式 baseline，对应已有 formal compare 结果。 |
| ASLAM-SSAF | 0.9157 +/- 0.0135 | 0.9262 +/- 0.0092 | 57.2 | 根目录正式改进模型，对应已有 formal compare 结果。 |

### 5.3 PubMed

| 模型 | AUC | AP | 最佳 epoch | 说明 |
| --- | ---: | ---: | ---: | --- |
| ASLAM | 0.9767 +/- 0.0013 | 0.9759 +/- 0.0014 | 38.1 | 根目录正式 baseline，对应已有 formal compare 结果。 |
| ASLAM-SSAF | 0.9787 +/- 0.0011 | 0.9775 +/- 0.0015 | 35.7 | 根目录正式改进模型，对应已有 formal compare 结果。 |

### 5.4 amz_Photo

| 模型 | AUC | AP | 最佳 epoch | 说明 |
| --- | ---: | ---: | ---: | --- |
| ASLAM | 0.9810 +/- 0.0011 | 0.9806 +/- 0.0017 | 11.2 | 根目录正式 baseline，对应已有 formal compare 结果。 |
| ASLAM-SSAF | 0.9857 +/- 0.0010 | 0.9863 +/- 0.0014 | 7.2 | 根目录正式改进模型，对应已有 formal compare 结果。 |

### 5.5 CoRA

| 模型 | AUC | AP | 最佳 epoch | 说明 |
| --- | ---: | ---: | ---: | --- |
| ASLAM | 0.9474 +/- 0.0024 | 0.9544 +/- 0.0027 | 33.8 | 根目录正式 baseline，对应已有 formal compare 结果。 |
| ASLAM-SSAF | 0.9535 +/- 0.0032 | 0.9622 +/- 0.0022 | 30.3 | 根目录正式改进模型，对应已有 formal compare 结果。 |

### 5.6 Twitch_EN

| 模型 | AUC | AP | 最佳 epoch | 说明 |
| --- | ---: | ---: | ---: | --- |
| ASLAM | 0.9281 +/- 0.0030 | 0.9332 +/- 0.0036 | 41.1 | 根目录正式 baseline，对应已有 formal compare 结果。 |
| ASLAM-SSAF | 0.9290 +/- 0.0032 | 0.9341 +/- 0.0030 | 31.0 | 根目录正式改进模型，对应已有 formal compare 结果。 |

## 6. 跨数据集平均结果

| 模型 | 覆盖数据集数 | 平均 AUC | 平均 AP |
| --- | ---: | ---: | ---: |
| CN | 1 | 0.6591 | 0.6587 |
| AA | 1 | 0.6593 | 0.6600 |
| Node2Vec | 1 | 0.7749 | 0.8028 |
| SEAL | 1 | 0.8612 | 0.8879 |
| ASLAM | 6 | 0.9302 | 0.9344 |
| ASLAM-SSAF | 6 | 0.9369 | 0.9416 |

## 7. 结论建议

1. 这份总表可直接作为论文实验部分的本地 formal comparison 草稿。
2. 若其中个别 baseline 结果明显异常，应优先回看该模型的实现口径，而不是直接下结论。
3. 对外写作时，应把 `官方论文表格数值`、`本地 recovered baseline`、`根目录统一协议 formal comparison` 三层结果严格分开。
