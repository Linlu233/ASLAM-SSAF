# Formal All-Experiment Alignment Summary

## Scope

This summary aligns all currently completed experiments in `G:\myProject\ASLAM-3` under the user's final boundary:

1. `G:\myProject\ASLAM-3\ASLAM` is baseline-only and is used to freeze the recovered local baseline.
2. `G:\myProject\ASLAM-3` is the root rebuilt framework where all new model changes and new comparisons are performed.
3. The provided ASLAM paper remains the only paper-reference source for the original reported method.

## Three comparison layers

1. Paper reference values:
   - only available for `Photo`, `Citeseer`, `PubMed`, and `DBLP` from the provided ASLAM paper.
2. Frozen `ASLAM/code` local baseline:
   - only available for `Citeseer`, `PubMed`, `DBLP`, and `Photo`.
3. Root rebuilt framework formal comparison:
   - available for `Citeseer`, `DBLP`, `PubMed`, `amz_Photo`, `CoRA`, and `Twitch_EN`.

These layers answer different questions and must not be merged into one baseline column.

## Frozen `ASLAM/code` local baseline

Source:

`G:\myProject\ASLAM-3\ASLAM\results\local_formal_baseline_summary.md`

| Dataset | AUC mean | AP mean |
| --- | ---: | ---: |
| Citeseer | 0.9676 | 0.9655 |
| PubMed | 0.9952 | 0.9943 |
| DBLP | 0.9409 | 0.9478 |
| Photo | 0.9727 | 0.9796 |

Protocol:

1. split `85/5/10`
2. seed `2`
3. runs `10`
4. epochs `401`
5. patience `20`
6. batch size `32`

## Root formal comparison

Shared root protocol:

1. runs `10`
2. epochs `401`
3. patience `20`
4. hidden channels `64`
5. seed `2`
6. device `cuda:0`

Dataset-specific stable edge batch sizes:

1. `Citeseer`, `DBLP`, `PubMed`, `amz_Photo`: `16384`
2. `Twitch_EN`: `8192`
3. `CoRA`: `4096`

Formal root mean results:

| Dataset | Batch size | Root `aslam` AUC | Root `aslam` AP | Root `aslam_ssmattn` AUC | Root `aslam_ssmattn` AP | AUC gain | AP gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 16384 | 0.8469 | 0.8540 | 0.8590 | 0.8631 | +0.0121 | +0.0091 |
| DBLP | 16384 | 0.9013 | 0.9085 | 0.9157 | 0.9262 | +0.0144 | +0.0177 |
| PubMed | 16384 | 0.9767 | 0.9759 | 0.9787 | 0.9775 | +0.0020 | +0.0016 |
| amz_Photo | 16384 | 0.9810 | 0.9806 | 0.9857 | 0.9863 | +0.0047 | +0.0057 |
| CoRA | 4096 | 0.9474 | 0.9544 | 0.9535 | 0.9622 | +0.0061 | +0.0078 |
| Twitch_EN | 8192 | 0.9281 | 0.9332 | 0.9290 | 0.9341 | +0.0009 | +0.0009 |

Formal root result files:

1. `G:\myProject\ASLAM-3\results\formal_citeseer_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
2. `G:\myProject\ASLAM-3\results\formal_dblp_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
3. `G:\myProject\ASLAM-3\results\formal_pubmed_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
4. `G:\myProject\ASLAM-3\results\formal_amz_photo_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
5. `G:\myProject\ASLAM-3\results\formal_cora_compare_runs10_epochs401_pat20_seed2_h64_b4096.json`
6. `G:\myProject\ASLAM-3\results\formal_twitch_en_compare_runs10_epochs401_pat20_seed2_h64_b8192.json`

## Academic use rule

1. For `PubMed`, three values can coexist:
   - paper reference: `97.73 +/- 0.02 / 97.74 +/- 0.01`
   - frozen `ASLAM/code` local baseline: `0.9952 / 0.9943`
   - root rebuilt-framework formal comparison: `0.9767 / 0.9759` vs `0.9787 / 0.9775`
2. For `CoRA` and `Twitch_EN`, only the root formal comparison should be reported in this workspace.
3. The root formal table must be presented as a fair within-framework comparison, not as a strict reproduction of the original released author pipeline.
