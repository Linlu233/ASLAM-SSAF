from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_auc_ap(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    y_true = labels.detach().cpu().numpy()
    y_score = logits.detach().cpu().numpy()
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "AP": float(average_precision_score(y_true, y_score)),
    }


def summarize_results(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }
