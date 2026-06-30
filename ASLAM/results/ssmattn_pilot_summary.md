# ASLAM SSM-Attn Pilot Summary

This file records the first pilot results of the new `ssmattn` model branch.

## Model status

- Variant name: `ssmattn`
- Baseline preservation: the default `baseline` path in `G:\myProject\ASLAM-3\ASLAM\code\train.py` is unchanged
- Current goal: verify whether the new fusion head is consistently stronger than the frozen local baseline before launching a full 10-run formal benchmark

## Current pilot results

| Dataset | Baseline formal mean AUC | Baseline formal mean AP | `ssmattn` pilot AUC | `ssmattn` pilot AP | Pilot setting |
| --- | ---: | ---: | ---: | ---: | --- |
| Citeseer | 0.9676 | 0.9655 | 0.9786 | 0.9710 | runs=1, epochs=6 |
| PubMed | 0.9952 | 0.9943 | 0.9959 | 0.9950 | runs=1, epochs=6 |
| DBLP | 0.9409 | 0.9478 | 0.9980 | 0.9979 | runs=1, epochs=8 |
| Photo | 0.9727 | 0.9796 | 0.9836 | 0.9855 | runs=1, epochs=6 |

## Interpretation

- The pilot branch currently shows improvements on all four local datasets relative to the frozen local formal baseline mean.
- These are still pilot results rather than formal publication-level results because they are single-run checks.
- The next academically valid step is a full `runs=10`, `epochs=401`, `patience=20` benchmark under the same protocol.

## Formal run command

```powershell
conda run -n ASLAM_THC python G:\myProject\ASLAM-3\ASLAM\code\scripts\run_local_formal_baseline.py --model-variant ssmattn --cuda cuda:0
```
