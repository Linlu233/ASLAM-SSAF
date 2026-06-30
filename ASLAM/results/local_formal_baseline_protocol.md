# Local Formal Baseline Protocol

This file freezes the current recovered `ASLAM/code` implementation as the official local baseline for all later model improvements in this workspace.

## Frozen scope

- Baseline source root: `G:\myProject\ASLAM-3\ASLAM\code`
- Baseline model file: `G:\myProject\ASLAM-3\ASLAM\code\models\bsal.py`
- Baseline training entry: `G:\myProject\ASLAM-3\ASLAM\code\train.py`
- Baseline data pipeline: `G:\myProject\ASLAM-3\ASLAM\code\utils\data_utils.py`

## Protocol

- Task: link prediction
- Local datasets used: `Citeseer`, `PubMed`, `DBLP`, `Photo`
- Split ratio: `85% / 5% / 10%`
- Fixed split seed: `2`
- Batch size: `32`
- Early-stop patience: `20`
- Formal run setting: `runs=10`, `epochs=401`
- Comparison rule: every improved model must use the same local datasets, split policy, and evaluation protocol

## Academic compliance note

The local dataset versions in this workspace do not match the dataset statistics reported in the paper's Table 1. Therefore:

- It is academically valid to define and report a local formal baseline on the local datasets.
- It is not academically valid to claim that the local benchmark has already strictly reproduced the paper's published table.
- Any later improvement should be reported as outperforming the frozen local baseline on the local dataset versions, not as surpassing the paper's original benchmark unless the original paper datasets are exactly matched.

## Reproduction commands

Run the frozen local baseline inside the specified conda environment:

```powershell
conda run -n ASLAM_THC python G:\myProject\ASLAM-3\ASLAM\code\scripts\run_local_formal_baseline.py --cuda cuda:0
```

Collect the latest available logs into a unified summary:

```powershell
conda run -n ASLAM_THC python G:\myProject\ASLAM-3\ASLAM\code\scripts\collect_local_baseline.py
```
