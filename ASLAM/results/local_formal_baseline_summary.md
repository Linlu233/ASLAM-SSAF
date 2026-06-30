# Local Formal Baseline Summary

This report freezes the current recovered `ASLAM/code` implementation as the official local baseline.

## Academic note

The local datasets do not match the paper's reported Table 1 statistics, so these results are valid only as a local-dataset formal baseline.
They must not be presented as a strict reproduction of the published table.

## Protocol

- Split ratio: 85/5/10
- Fixed split seed: 2
- Batch size: 32
- Early-stop patience: 20
- Default formal setting: runs=10, epochs=401
- Comparison rule: all improved models must be compared against this frozen local baseline under the same local datasets and protocol

## Dataset mismatch check

| Dataset | Local nodes | Paper nodes | Local edges | Paper edges | Local features | Paper features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| citeseer | 3312 | 4230 | 4536 | 10674 | 3703 | 602 |
| pubmed | 19717 | 19717 | 44324 | 88648 | 500 | 500 |
| dblp | 4057 | 17716 | 3528 | 105734 | 334 | 1639 |
| photo | 18333 | 7650 | 81894 | 23862 | 6805 | 745 |

## Baseline results

| Dataset | Runs found | AUC mean | AP mean | AUC std | AP std | Log |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| citeseer | 10 | 0.9676 | 0.9655 | 0.0031 | 0.0024 | G:\myProject\ASLAM-3\ASLAM\results\train=0.9\Citeseer_1.0_feat_False_20260403_095405.txt |
| pubmed | 10 | 0.9952 | 0.9943 | 0.0007 | 0.0008 | G:\myProject\ASLAM-3\ASLAM\results\train=0.9\Pubmed_1.0_feat_False_20260403_120258.txt |
| dblp | 10 | 0.9409 | 0.9478 | 0.0084 | 0.0057 | G:\myProject\ASLAM-3\ASLAM\results\train=0.9\Dblp_1.0_feat_False_20260405_050147.txt |
| photo | 10 | 0.9727 | 0.9796 | 0.0154 | 0.0106 | G:\myProject\ASLAM-3\ASLAM\results\train=0.9\Photo_1.0_feat_False_20260405_095524.txt |
