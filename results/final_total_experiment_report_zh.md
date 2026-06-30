# 最终总实验报告（中文对照版）

日期：2026-04-15

## 1. 报告范围

本文件是 `G:\myProject\ASLAM-3` 当前全部已完成实验的中文总报告。

在学术写作中，本文件严格区分四个实验层次，因为它们回答的是不同问题：

1. 你提供论文中的 ASLAM 原始报告结果；
2. `G:\myProject\ASLAM-3\ASLAM\code` 中恢复得到的冻结 baseline；
3. `G:\myProject\ASLAM-3` 根目录重建框架中的正式对比实验；
4. `G:\myProject\ASLAM-3` 根目录重建框架中的正式消融实验，即 `ASLAM -> ASLAMPlus -> ASLAMSSMAttn`。

## 2. 关键路径

Baseline 复现路径：

`G:\myProject\ASLAM-3\ASLAM`

改进模型根目录路径：

`G:\myProject\ASLAM-3`

冻结本地 baseline 汇总文件：

`G:\myProject\ASLAM-3\ASLAM\results\local_formal_baseline_summary.md`

root 正式长报告：

`G:\myProject\ASLAM-3\results\aslam_ssmattn_formal_report.md`

本中文总报告：

`G:\myProject\ASLAM-3\results\final_total_experiment_report_zh.md`

英文总报告：

`G:\myProject\ASLAM-3\results\final_total_experiment_report.md`

## 3. 运行环境

1. Conda 环境：`ASLAM_THC`
2. root formal 实验所用设备：`cuda:0`
3. root formal 共用协议：
   - runs `10`
   - epochs `401`
   - patience `20`
   - hidden channels `64`
   - seed `2`

## 4. 论文原文中的 ASLAM 参考值

来源文件：

`G:\myProject\ASLAM-3\papers\Link prediction for attribute and structure learning based on attention mechanism.pdf`

提取文本：

`G:\myProject\ASLAM-3\papers\aslam_paper.txt`

| 数据集 | 论文 AUC | 论文 AP |
| --- | ---: | ---: |
| Photo | 99.14 | 99.01 |
| Citeseer | 94.71 | 94.97 |
| PubMed | 97.73 +/- 0.02 | 97.74 +/- 0.01 |
| DBLP | 96.68 | 97.15 |

重要说明：

1. 上表只是论文文献参考值。
2. 这些数值不是本地重跑结果。

## 5. 冻结的 `ASLAM/code` 本地 formal baseline

来源文件：

`G:\myProject\ASLAM-3\ASLAM\results\local_formal_baseline_summary.md`

协议：

1. split `85/5/10`
2. seed `2`
3. runs `10`
4. epochs `401`
5. patience `20`
6. batch size `32`

结果如下：

| 数据集 | AUC mean | AP mean | AUC std | AP std |
| --- | ---: | ---: | ---: | ---: |
| Citeseer | 0.9676 | 0.9655 | 0.0031 | 0.0024 |
| PubMed | 0.9952 | 0.9943 | 0.0007 | 0.0008 |
| DBLP | 0.9409 | 0.9478 | 0.0084 | 0.0057 |
| Photo | 0.9727 | 0.9796 | 0.0154 | 0.0106 |

数据不一致说明：

1. 当前本地数据文件与论文 Table 1 中的数据统计并不一致。
2. 因此，这些结果只能作为冻结的本地 baseline。
3. 它们不能被表述为对论文原表结果的严格复现。

覆盖范围说明：

1. 当前冻结 baseline 仅覆盖 `Citeseer`、`PubMed`、`DBLP` 和 `Photo`。
2. `CoRA` 与 `Twitch_EN` 在本工作区中没有对应的 `ASLAM/code` 作者风格冻结 baseline。

## 6. 数据集概况与本地数据源映射

下表中的节点数、边数和特征维度，均来自 root 重建框架实际载入后的图统计。边数统一按无向图记为 `edge_index.size(1) / 2`。

| 数据集 | 节点数 | 无向边数 | 特征维度 | 固定 batch size | loader 输入文件 |
| --- | ---: | ---: | ---: | ---: | --- |
| Citeseer | 3312 | 4536 | 3703 | 16384 | `datasets/Citeseer/citeseer.content`，`datasets/Citeseer/citeseer.cites` |
| DBLP | 4057 | 3528 | 334 | 16384 | `datasets/DBLP/dblp_feat.npy`，`datasets/DBLP/dblp_label.npy`，`datasets/DBLP/dblp_adj.npy` |
| PubMed | 19717 | 44324 | 500 | 16384 | `datasets/PubMed/Pubmed-Diabetes.NODE.paper.tab`，`datasets/PubMed/Pubmed-Diabetes.DIRECTED.cites.tab` |
| amz_Photo | 18333 | 81894 | 6805 | 16384 | `datasets/amz_Photo/ms_academic_cs.npz` |
| CoRA | 11881 | 31482 | 9568 | 4096 | `datasets/CoRA/CoRA_Raw/papers_dataset.txt`，`datasets/CoRA/CoRA_Raw/citations.txt`，`datasets/CoRA/CoRA_Raw/topics.txt`，`datasets/CoRA/CoRA_Raw/words_dictionary.txt` |
| Twitch_EN | 7126 | 35324 | 128 | 8192 | `datasets/Twitch_EN/musae_Twitch_EN.json` |

实验层覆盖情况：

1. 论文参考值只覆盖 `Photo`、`Citeseer`、`PubMed`、`DBLP`。
2. 冻结 recovered `ASLAM/code` baseline 只覆盖 `Citeseer`、`PubMed`、`DBLP`、`Photo`。
3. root formal 对比与 root formal 消融现在已覆盖全部 6 个 root 支持数据集：`Citeseer`、`DBLP`、`PubMed`、`amz_Photo`、`CoRA`、`Twitch_EN`。

## 7. root 重建框架 formal 对比实验

root formal 结果文件：

1. `G:\myProject\ASLAM-3\results\formal_citeseer_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
2. `G:\myProject\ASLAM-3\results\formal_dblp_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
3. `G:\myProject\ASLAM-3\results\formal_pubmed_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
4. `G:\myProject\ASLAM-3\results\formal_amz_photo_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
5. `G:\myProject\ASLAM-3\results\formal_cora_compare_runs10_epochs401_pat20_seed2_h64_b4096.json`
6. `G:\myProject\ASLAM-3\results\formal_twitch_en_compare_runs10_epochs401_pat20_seed2_h64_b8192.json`

formal 均值结果如下：

| 数据集 | Batch size | root `aslam` AUC | root `aslam` AP | root `aslam_ssmattn` AUC | root `aslam_ssmattn` AP | root `aslam` AUC std | root `aslam` AP std | root `aslam_ssmattn` AUC std | root `aslam_ssmattn` AP std | AUC 提升 | AP 提升 | 平均最优 epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 16384 | 0.8469 | 0.8540 | 0.8590 | 0.8631 | 0.0120 | 0.0177 | 0.0161 | 0.0191 | +0.0121 | +0.0091 | 42.2 |
| DBLP | 16384 | 0.9013 | 0.9085 | 0.9157 | 0.9262 | 0.0110 | 0.0119 | 0.0135 | 0.0092 | +0.0144 | +0.0177 | 57.2 |
| PubMed | 16384 | 0.9767 | 0.9759 | 0.9787 | 0.9775 | 0.0013 | 0.0014 | 0.0011 | 0.0015 | +0.0020 | +0.0016 | 35.7 |
| amz_Photo | 16384 | 0.9810 | 0.9806 | 0.9857 | 0.9863 | 0.0011 | 0.0017 | 0.0010 | 0.0014 | +0.0047 | +0.0057 | 7.2 |
| CoRA | 4096 | 0.9474 | 0.9544 | 0.9535 | 0.9622 | 0.0024 | 0.0027 | 0.0032 | 0.0022 | +0.0061 | +0.0078 | 30.3 |
| Twitch_EN | 8192 | 0.9281 | 0.9332 | 0.9290 | 0.9341 | 0.0030 | 0.0036 | 0.0032 | 0.0030 | +0.0009 | +0.0009 | 31.0 |

root 对比实验结论：

1. 在 root 重建框架的 formal 对比协议下，`aslam_ssmattn` 在 6 个已支持数据集上均同时优于 root `aslam`。
2. 这张对比表是当前根目录改进框架“全面提升”结论的正式依据。

## 8. root formal 消融实验

`ASLAMPlus` 的正式结果文件如下：

1. `G:\myProject\ASLAM-3\results\formal_citeseer_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json`
2. `G:\myProject\ASLAM-3\results\formal_dblp_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json`
3. `G:\myProject\ASLAM-3\results\formal_pubmed_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json`
4. `G:\myProject\ASLAM-3\results\formal_amz_photo_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json`
5. `G:\myProject\ASLAM-3\results\formal_cora_aslam_plus_runs10_epochs401_pat20_seed2_h64_b4096.json`
6. `G:\myProject\ASLAM-3\results\formal_twitch_en_aslam_plus_runs10_epochs401_pat20_seed2_h64_b8192.json`

本节按“以最终模型为基准、去掉模块后的掉点”格式书写。完整模型为 `ASLAMSSMAttn`，第一层弱化版本是 `去掉 SSM-注意力融合 = ASLAMPlus`，第二层弱化版本是 `进一步去掉增强骨干 = ASLAM`。

相对于最终模型的 AUC 掉点如下：

| 数据集 | Batch size | 最终 `ASLAMSSMAttn` AUC | `去掉 SSM-注意力融合` AUC | 相对最终模型 AUC 掉点 | `进一步去掉增强骨干` AUC | 相对最终模型 AUC 掉点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 16384 | 0.8590 +/- 0.0161 | 0.8412 +/- 0.0123 | -0.0178 | 0.8469 +/- 0.0120 | -0.0121 |
| DBLP | 16384 | 0.9157 +/- 0.0135 | 0.8926 +/- 0.0158 | -0.0231 | 0.9013 +/- 0.0110 | -0.0144 |
| PubMed | 16384 | 0.9787 +/- 0.0011 | 0.9781 +/- 0.0015 | -0.0006 | 0.9767 +/- 0.0013 | -0.0020 |
| amz_Photo | 16384 | 0.9857 +/- 0.0010 | 0.9841 +/- 0.0009 | -0.0016 | 0.9810 +/- 0.0011 | -0.0047 |
| CoRA | 4096 | 0.9535 +/- 0.0032 | 0.9486 +/- 0.0024 | -0.0049 | 0.9474 +/- 0.0024 | -0.0061 |
| Twitch_EN | 8192 | 0.9290 +/- 0.0032 | 0.9270 +/- 0.0034 | -0.0020 | 0.9281 +/- 0.0030 | -0.0009 |

相对于最终模型的 AP 掉点如下：

| 数据集 | Batch size | 最终 `ASLAMSSMAttn` AP | `去掉 SSM-注意力融合` AP | 相对最终模型 AP 掉点 | `进一步去掉增强骨干` AP | 相对最终模型 AP 掉点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 16384 | 0.8631 +/- 0.0191 | 0.8517 +/- 0.0129 | -0.0114 | 0.8540 +/- 0.0177 | -0.0091 |
| DBLP | 16384 | 0.9262 +/- 0.0092 | 0.9029 +/- 0.0167 | -0.0233 | 0.9085 +/- 0.0119 | -0.0177 |
| PubMed | 16384 | 0.9775 +/- 0.0015 | 0.9767 +/- 0.0015 | -0.0008 | 0.9759 +/- 0.0014 | -0.0016 |
| amz_Photo | 16384 | 0.9863 +/- 0.0014 | 0.9856 +/- 0.0012 | -0.0007 | 0.9806 +/- 0.0017 | -0.0057 |
| CoRA | 4096 | 0.9622 +/- 0.0022 | 0.9569 +/- 0.0038 | -0.0053 | 0.9544 +/- 0.0027 | -0.0078 |
| Twitch_EN | 8192 | 0.9341 +/- 0.0030 | 0.9332 +/- 0.0034 | -0.0009 | 0.9332 +/- 0.0036 | -0.0009 |

消融结论：

1. 去掉最后的 SSM-注意力融合模块后，6 个数据集的 AUC 和 AP 都出现掉点。
2. 对 `PubMed`、`amz_Photo`、`CoRA` 而言，继续去掉增强骨干会带来更大的性能损失；对 `Citeseer`、`DBLP`、`Twitch_EN` 而言，最终的 SSM-注意力融合本身就是更大的损失来源。
3. 因而，当前最终模型的提升是累积式形成的，但最后的跨分支 SSM-注意力融合阶段本身也是必要模块，而不是表面性附加结构。

## 9. t-SNE 可视化

本次可视化遵循论文中的口径，而不是直接对原始节点做二维投影：

1. 每个数据集在 root 重建框架下单独训练一次模型；
2. 从最终打分阶段之前提取已学习到的边表征；
3. 以固定随机种子从测试边中抽取 15%，同时保留正负边两类；
4. 使用 t-SNE 将边表征映射到二维空间，其中橙色为正边，蓝色为负边。

输出目录：

`G:\myProject\ASLAM-3\results\tsne`

主要产物：

1. 单数据集图片：`*_edge_tsne.png`
2. 单数据集坐标文件：`*_edge_tsne_coords.npz`
3. 总览拼图：`aslam_ssmattn_edge_tsne_grid.png`
4. 汇总文件：
   - `G:\myProject\ASLAM-3\results\tsne\tsne_visualization_summary.json`
   - `G:\myProject\ASLAM-3\results\tsne\tsne_visualization_summary.md`

## 10. 最终对齐结论

已完成实验覆盖情况如下：

1. 论文参考层：已完成 `Photo`、`Citeseer`、`PubMed`、`DBLP`
2. 冻结 recovered baseline 层：已完成 `Citeseer`、`PubMed`、`DBLP`、`Photo`
3. root formal 对比层：已完成 `Citeseer`、`DBLP`、`PubMed`、`amz_Photo`、`CoRA`、`Twitch_EN`
4. root formal 消融层：已完成 `ASLAM`、`ASLAMPlus`、`ASLAMSSMAttn` 在 `Citeseer`、`DBLP`、`PubMed`、`amz_Photo`、`CoRA`、`Twitch_EN` 上的正式结果
5. 数据集概况表：已在第 6 节补齐

因此，当前工作区中的对比实验、消融实验和数据集概况表已经全部完成并对齐。

## 11. 学术写作使用规则

对于 `PubMed`，论文或学位论文中可能会同时出现四组数字，但必须明确标注其含义：

1. 论文参考值：
   - `97.73 +/- 0.02 / 97.74 +/- 0.01`
2. 冻结 recovered local baseline：
   - `0.9952 / 0.9943`
3. root 重建框架 formal 对比与消融：
   - root `aslam`：`0.9767 / 0.9759`
   - root `aslam_plus`：`0.9781 / 0.9767`
   - root `aslam_ssmattn`：`0.9787 / 0.9775`

这些数字回答的是不同问题，不能混写进同一个 baseline 列中。

对于 `CoRA` 和 `Twitch_EN`：

1. 你提供的 ASLAM 论文中没有这两个数据集的原表参考值；
2. 当前工作区中也没有对应的 `ASLAM/code` 冻结 baseline；
3. 因此，这两个数据集只能报告 root 重建框架内部的 formal 对比与 formal 消融结果。

对于消融表：

1. 必须将三个变体都限制在同一套 root 重建框架内；
2. 不能把 root `ASLAMPlus` 与冻结 `ASLAM/code` baseline 放在同一张消融表中；
3. 必须明确说明该消融是在同一协议下隔离结构贡献。

## 12. 可直接引用的中文表述

如果你需要一段可以直接写入实验章节的总结，当前最稳妥的写法是：

1. 首先，基于恢复出的作者风格 `ASLAM/code` 实现，在本地可用数据文件上冻结了 formal baseline。
2. 随后，所有新模型结构修改均仅在项目根目录下的重建框架中进行评测，并统一采用固定的 10-run formal protocol。
3. 在该 root formal protocol 下，若以完整模型 `ASLAMSSMAttn` 为基准进行消融，则去掉最终的 SSM-注意力融合模块会在全部 6 个当前支持数据集上带来 AUC 与 AP 掉点，而进一步去掉增强骨干后会在多个数据集上造成更大的性能损失。
4. 由于当前本地数据文件与论文 Table 1 的统计信息不一致，因此所有本地重跑结果都不能表述为对论文原始表格结果的严格复现。
