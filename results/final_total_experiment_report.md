# Final Total Experiment Report

Date: 2026-04-15

## 1. Scope

This file is the single consolidated report for all currently completed experiments in `G:\myProject\ASLAM-3`.

It keeps four layers separate, because they answer different academic questions:

1. paper-reported ASLAM values from the provided paper;
2. frozen recovered baseline from `G:\myProject\ASLAM-3\ASLAM\code`;
3. root rebuilt-framework formal comparison in `G:\myProject\ASLAM-3`;
4. root rebuilt-framework formal ablation for `ASLAM -> ASLAMPlus -> ASLAMSSMAttn`.

## 2. Key paths

Baseline reproduction path:

`G:\myProject\ASLAM-3\ASLAM`

Improved root-code path:

`G:\myProject\ASLAM-3`

Frozen local baseline summary:

`G:\myProject\ASLAM-3\ASLAM\results\local_formal_baseline_summary.md`

Root formal report:

`G:\myProject\ASLAM-3\results\aslam_ssmattn_formal_report.md`

This total report:

`G:\myProject\ASLAM-3\results\final_total_experiment_report.md`

## 3. Environment

1. Conda environment: `ASLAM_THC`
2. Device used for root formal runs: `cuda:0`
3. Root formal shared protocol:
   - runs `10`
   - epochs `401`
   - patience `20`
   - hidden channels `64`
   - seed `2`

## 4. Paper-reported ASLAM reference values

Source:

`G:\myProject\ASLAM-3\papers\Link prediction for attribute and structure learning based on attention mechanism.pdf`

Extracted text source:

`G:\myProject\ASLAM-3\papers\aslam_paper.txt`

| Dataset | AUC (paper) | AP (paper) |
| --- | ---: | ---: |
| Photo | 99.14 | 99.01 |
| Citeseer | 94.71 | 94.97 |
| PubMed | 97.73 +/- 0.02 | 97.74 +/- 0.01 |
| DBLP | 96.68 | 97.15 |

Important note:

1. These are literature reference values only.
2. They are not local rerun values on this workspace.

## 5. Frozen `ASLAM/code` local formal baseline

Source:

`G:\myProject\ASLAM-3\ASLAM\results\local_formal_baseline_summary.md`

Protocol:

1. split `85/5/10`
2. seed `2`
3. runs `10`
4. epochs `401`
5. patience `20`
6. batch size `32`

Results:

| Dataset | AUC mean | AP mean | AUC std | AP std |
| --- | ---: | ---: | ---: | ---: |
| Citeseer | 0.9676 | 0.9655 | 0.0031 | 0.0024 |
| PubMed | 0.9952 | 0.9943 | 0.0007 | 0.0008 |
| DBLP | 0.9409 | 0.9478 | 0.0084 | 0.0057 |
| Photo | 0.9727 | 0.9796 | 0.0154 | 0.0106 |

Dataset mismatch note:

1. These local datasets do not match the paper Table 1 statistics.
2. Therefore, these values are valid as a frozen local baseline only.
3. They must not be claimed as a strict reproduction of the published table.

Coverage note:

1. This frozen local baseline exists only for `Citeseer`, `PubMed`, `DBLP`, and `Photo`.
2. It does not exist for `CoRA` or `Twitch_EN` in the recovered `ASLAM/code` pipeline on this workspace.

## 6. Dataset Overview And Local Source Mapping

The node count, edge count, and feature dimension below are the actual graph statistics used by the root rebuilt framework. Edge counts are reported as undirected edges after preprocessing, i.e. `edge_index.size(1) / 2`.

| Dataset | Nodes | Undirected edges | Features | Stable batch size | Loader input files |
| --- | ---: | ---: | ---: | ---: | --- |
| Citeseer | 3312 | 4536 | 3703 | 16384 | `datasets/Citeseer/citeseer.content`, `datasets/Citeseer/citeseer.cites` |
| DBLP | 4057 | 3528 | 334 | 16384 | `datasets/DBLP/dblp_feat.npy`, `datasets/DBLP/dblp_label.npy`, `datasets/DBLP/dblp_adj.npy` |
| PubMed | 19717 | 44324 | 500 | 16384 | `datasets/PubMed/Pubmed-Diabetes.NODE.paper.tab`, `datasets/PubMed/Pubmed-Diabetes.DIRECTED.cites.tab` |
| amz_Photo | 18333 | 81894 | 6805 | 16384 | `datasets/amz_Photo/ms_academic_cs.npz` |
| CoRA | 11881 | 31482 | 9568 | 4096 | `datasets/CoRA/CoRA_Raw/papers_dataset.txt`, `datasets/CoRA/CoRA_Raw/citations.txt`, `datasets/CoRA/CoRA_Raw/topics.txt`, `datasets/CoRA/CoRA_Raw/words_dictionary.txt` |
| Twitch_EN | 7126 | 35324 | 128 | 8192 | `datasets/Twitch_EN/musae_Twitch_EN.json` |

Reporting-layer coverage:

1. Paper reference values are available only for `Photo`, `Citeseer`, `PubMed`, and `DBLP`.
2. Frozen recovered `ASLAM/code` baselines are available only for `Citeseer`, `PubMed`, `DBLP`, and `Photo`.
3. Root formal comparison and root formal ablation are now available for all six root-supported datasets: `Citeseer`, `DBLP`, `PubMed`, `amz_Photo`, `CoRA`, and `Twitch_EN`.

## 7. Root Rebuilt-Framework Formal Comparison

Root formal result files:

1. `G:\myProject\ASLAM-3\results\formal_citeseer_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
2. `G:\myProject\ASLAM-3\results\formal_dblp_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
3. `G:\myProject\ASLAM-3\results\formal_pubmed_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
4. `G:\myProject\ASLAM-3\results\formal_amz_photo_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
5. `G:\myProject\ASLAM-3\results\formal_cora_compare_runs10_epochs401_pat20_seed2_h64_b4096.json`
6. `G:\myProject\ASLAM-3\results\formal_twitch_en_compare_runs10_epochs401_pat20_seed2_h64_b8192.json`

Formal mean results:

| Dataset | Batch size | Root `aslam` AUC | Root `aslam` AP | Root `aslam_ssmattn` AUC | Root `aslam_ssmattn` AP | Root `aslam` AUC std | Root `aslam` AP std | Root `aslam_ssmattn` AUC std | Root `aslam_ssmattn` AP std | Delta AUC | Delta AP | Mean best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 16384 | 0.8469 | 0.8540 | 0.8590 | 0.8631 | 0.0120 | 0.0177 | 0.0161 | 0.0191 | +0.0121 | +0.0091 | 42.2 |
| DBLP | 16384 | 0.9013 | 0.9085 | 0.9157 | 0.9262 | 0.0110 | 0.0119 | 0.0135 | 0.0092 | +0.0144 | +0.0177 | 57.2 |
| PubMed | 16384 | 0.9767 | 0.9759 | 0.9787 | 0.9775 | 0.0013 | 0.0014 | 0.0011 | 0.0015 | +0.0020 | +0.0016 | 35.7 |
| amz_Photo | 16384 | 0.9810 | 0.9806 | 0.9857 | 0.9863 | 0.0011 | 0.0017 | 0.0010 | 0.0014 | +0.0047 | +0.0057 | 7.2 |
| CoRA | 4096 | 0.9474 | 0.9544 | 0.9535 | 0.9622 | 0.0024 | 0.0027 | 0.0032 | 0.0022 | +0.0061 | +0.0078 | 30.3 |
| Twitch_EN | 8192 | 0.9281 | 0.9332 | 0.9290 | 0.9341 | 0.0030 | 0.0036 | 0.0032 | 0.0030 | +0.0009 | +0.0009 | 31.0 |

Root comparison conclusion:

1. Under the root rebuilt-framework formal comparison, `aslam_ssmattn` improves over root `aslam` on all six supported local datasets in both AUC and AP.
2. This comparison table is the formal basis for claiming consistent improvement in the rebuilt root framework.

## 8. Root Formal Ablation Study

Ablation result files for `ASLAMPlus`:

1. `G:\myProject\ASLAM-3\results\formal_citeseer_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json`
2. `G:\myProject\ASLAM-3\results\formal_dblp_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json`
3. `G:\myProject\ASLAM-3\results\formal_pubmed_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json`
4. `G:\myProject\ASLAM-3\results\formal_amz_photo_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json`
5. `G:\myProject\ASLAM-3\results\formal_cora_aslam_plus_runs10_epochs401_pat20_seed2_h64_b4096.json`
6. `G:\myProject\ASLAM-3\results\formal_twitch_en_aslam_plus_runs10_epochs401_pat20_seed2_h64_b8192.json`

This ablation is written in the requested drop-from-final format. The complete model is `ASLAMSSMAttn`, the first weakened variant is `w/o SSM-attention fusion = ASLAMPlus`, and the second weakened variant is `w/o enhanced backbone and w/o SSM-attention fusion = ASLAM`.

AUC drop from the final model:

| Dataset | Batch size | Final `ASLAMSSMAttn` AUC | `w/o SSM-attn fusion` AUC | AUC drop vs final | `w/o enhanced backbone` AUC | AUC drop vs final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 16384 | 0.8590 +/- 0.0161 | 0.8412 +/- 0.0123 | -0.0178 | 0.8469 +/- 0.0120 | -0.0121 |
| DBLP | 16384 | 0.9157 +/- 0.0135 | 0.8926 +/- 0.0158 | -0.0231 | 0.9013 +/- 0.0110 | -0.0144 |
| PubMed | 16384 | 0.9787 +/- 0.0011 | 0.9781 +/- 0.0015 | -0.0006 | 0.9767 +/- 0.0013 | -0.0020 |
| amz_Photo | 16384 | 0.9857 +/- 0.0010 | 0.9841 +/- 0.0009 | -0.0016 | 0.9810 +/- 0.0011 | -0.0047 |
| CoRA | 4096 | 0.9535 +/- 0.0032 | 0.9486 +/- 0.0024 | -0.0049 | 0.9474 +/- 0.0024 | -0.0061 |
| Twitch_EN | 8192 | 0.9290 +/- 0.0032 | 0.9270 +/- 0.0034 | -0.0020 | 0.9281 +/- 0.0030 | -0.0009 |

AP drop from the final model:

| Dataset | Batch size | Final `ASLAMSSMAttn` AP | `w/o SSM-attn fusion` AP | AP drop vs final | `w/o enhanced backbone` AP | AP drop vs final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 16384 | 0.8631 +/- 0.0191 | 0.8517 +/- 0.0129 | -0.0114 | 0.8540 +/- 0.0177 | -0.0091 |
| DBLP | 16384 | 0.9262 +/- 0.0092 | 0.9029 +/- 0.0167 | -0.0233 | 0.9085 +/- 0.0119 | -0.0177 |
| PubMed | 16384 | 0.9775 +/- 0.0015 | 0.9767 +/- 0.0015 | -0.0008 | 0.9759 +/- 0.0014 | -0.0016 |
| amz_Photo | 16384 | 0.9863 +/- 0.0014 | 0.9856 +/- 0.0012 | -0.0007 | 0.9806 +/- 0.0017 | -0.0057 |
| CoRA | 4096 | 0.9622 +/- 0.0022 | 0.9569 +/- 0.0038 | -0.0053 | 0.9544 +/- 0.0027 | -0.0078 |
| Twitch_EN | 8192 | 0.9341 +/- 0.0030 | 0.9332 +/- 0.0034 | -0.0009 | 0.9332 +/- 0.0036 | -0.0009 |

Ablation conclusion:

1. Removing the final SSM-attention fusion module causes AUC and AP drops on all six datasets.
2. Removing both the enhanced backbone and the final fusion module causes larger damage on `PubMed`, `amz_Photo`, and `CoRA`, while the final fusion removal is the larger damage source on `Citeseer`, `DBLP`, and `Twitch_EN`.
3. The final gains are therefore cumulative, but the last cross-branch SSM-attention fusion stage is independently validated as a necessary component rather than a cosmetic add-on.

## 9. t-SNE Visualization

The visualization follows the paper-style protocol instead of raw-node projection:

1. one single-run model is trained per dataset under the root rebuilt framework;
2. learned edge representations are extracted from the final scoring stage input;
3. 15% of test edges are sampled with a fixed seed while preserving positive and negative classes;
4. t-SNE maps the learned edge features into 2D space, with positive edges shown in orange and negative edges shown in blue.

Output path:

`G:\myProject\ASLAM-3\results\tsne`

Main artifacts:

1. per-dataset figures: `*_edge_tsne.png`
2. per-dataset coordinates: `*_edge_tsne_coords.npz`
3. grid overview: `aslam_ssmattn_edge_tsne_grid.png`
4. summary files:
   - `G:\myProject\ASLAM-3\results\tsne\tsne_visualization_summary.json`
   - `G:\myProject\ASLAM-3\results\tsne\tsne_visualization_summary.md`

## 10. Final Alignment Judgment

Completed experiment coverage:

1. Paper reference layer: completed for `Photo`, `Citeseer`, `PubMed`, `DBLP`
2. Frozen recovered local baseline layer: completed for `Citeseer`, `PubMed`, `DBLP`, `Photo`
3. Root formal comparison layer: completed for `Citeseer`, `DBLP`, `PubMed`, `amz_Photo`, `CoRA`, `Twitch_EN`
4. Root formal ablation layer: completed for `ASLAM`, `ASLAMPlus`, and `ASLAMSSMAttn` on `Citeseer`, `DBLP`, `PubMed`, `amz_Photo`, `CoRA`, and `Twitch_EN`
5. Dataset overview table: completed in Section 6

Therefore, the comparison experiments, ablation experiments, and dataset overview table are all complete and aligned in this workspace.

## 11. Academic Reporting Rule

For `PubMed`, three different numbers may appear in the dissertation or paper draft, but they must be labeled correctly:

1. paper reference value:
   - `97.73 +/- 0.02 / 97.74 +/- 0.01`
2. frozen recovered local baseline:
   - `0.9952 / 0.9943`
3. root rebuilt-framework formal comparison and ablation:
   - root `aslam`: `0.9767 / 0.9759`
   - root `aslam_plus`: `0.9781 / 0.9767`
   - root `aslam_ssmattn`: `0.9787 / 0.9775`

These numbers answer different questions and must not be merged into one baseline column.

For `CoRA` and `Twitch_EN`:

1. there is no paper-reference value from the provided ASLAM paper;
2. there is no frozen recovered `ASLAM/code` baseline in this workspace;
3. only the root rebuilt-framework comparison and ablation should be reported.

For the ablation tables:

1. keep all three variants inside the same rebuilt root framework;
2. do not mix root `ASLAMPlus` with frozen `ASLAM/code` in one ablation table;
3. explicitly state that the ablation isolates architectural contribution under one fixed protocol.

## 12. Recommended Citation-Ready Wording

If you need one short final statement for the experiment section, the safest wording is:

1. The recovered author-style `ASLAM/code` implementation was first frozen as the local formal baseline on the available local datasets.
2. All new architectural changes were then evaluated only in the rebuilt root framework under a fixed 10-run formal protocol on six supported datasets.
3. In the root-framework ablation, `ASLAMPlus` alone was not universally stronger than root `ASLAM`, whereas the full `ASLAMSSMAttn` model improved over root `ASLAM` on all six datasets in both AUC and AP.
4. Because the local dataset files do not match the paper Table 1 statistics, none of the local rerun results should be claimed as a strict reproduction of the paper's published table.
