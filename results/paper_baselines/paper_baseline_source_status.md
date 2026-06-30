# Paper Baseline Source And Environment Status

Date: 2026-04-17

## 1. Official source directories

Path:

`G:\myProject\ASLAM-3\external\official_baselines`

Current status:

| Source | Local path | Status | Note |
| --- | --- | --- | --- |
| PyTorch Geometric | `G:\myProject\ASLAM-3\external\official_baselines\pytorch_geometric` | available | Used as the official implementation source for `Node2Vec`, `GCN`, `GAE`, `VGAE`, `GraphSAGE`, `GAT`, and `SuperGATConv`-based reproduction. |
| SuperGAT | `G:\myProject\ASLAM-3\external\official_baselines\SuperGAT` | available | Official project source downloaded successfully through proxy. |
| SEAL_OGB | `G:\myProject\ASLAM-3\external\official_baselines\SEAL_OGB` | available | Public SEAL-family source downloaded successfully through proxy. |
| SEAL | `G:\myProject\ASLAM-3\external\official_baselines\SEAL` | partial checkout | Git clone succeeded, but Windows checkout failed because the repository contains invalid path names on NTFS such as `This_is_nauty26r7.` |

## 2. Conda environments

Conda env root:

`C:\Users\linlu\miniconda3\envs`

Created model-named environments:

`aa`, `bsal`, `cn`, `gae`, `gat`, `gcn`, `graphsage`, `node2vec`, `seal`, `supergat`, `vgae`

Validation status:

1. All created environments were verified to import `torch`.
2. CUDA is available in the cloned environments.
3. Minimal benchmark execution has been validated at least on `cn` and `gcn` environment clones.

## 3. Benchmark framework status

Main benchmark entry:

`G:\myProject\ASLAM-3\benchmark_paper_baselines.py`

Report compiler:

`G:\myProject\ASLAM-3\compile_paper_baseline_report.py`

Support scripts:

1. `G:\myProject\ASLAM-3\scripts\setup_paper_baseline_envs.ps1`
2. `G:\myProject\ASLAM-3\scripts\fetch_paper_baseline_sources.ps1`
3. `G:\myProject\ASLAM-3\scripts\run_paper_baseline_suite.ps1`

Smoke-test status:

1. `CN`, `AA`, `Node2Vec`, `GCN`, `GAE`, `VGAE`, `GraphSAGE`, `GAT`, `SuperGAT`, `SEAL`, and `BSAL` all passed minimal smoke testing on `Citeseer`.
2. Smoke result files are stored in:
   `G:\myProject\ASLAM-3\results\paper_baselines`

## 4. Formal results completed so far

Completed formal result JSON files:

1. `G:\myProject\ASLAM-3\results\paper_baselines\paper_baselines_cn.json`
2. `G:\myProject\ASLAM-3\results\paper_baselines\paper_baselines_aa.json`
3. `G:\myProject\ASLAM-3\results\paper_baselines\paper_baselines_node2vec.json`

Current partial consolidated report:

`G:\myProject\ASLAM-3\results\paper_baselines\paper_baseline_partial_report.md`

## 5. Academic compliance note

1. The current partial report is a local formal comparison under a fixed protocol.
2. It must not be written as a strict reproduction of the original ASLAM paper tables.
3. For `BSAL`, the current root implementation is explicitly documented as a local reproduction based on the paper description and recovered code fragments, not as a verified standalone official repository replay.
