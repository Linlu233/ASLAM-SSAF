from __future__ import annotations

from typing import Dict

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_auc_ap(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    y_true = labels.detach().cpu().numpy()
    y_score = logits.detach().cpu().numpy()
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "AP": float(average_precision_score(y_true, y_score)),
    }


def evaluate_hits(input_dict: Dict[str, torch.Tensor], k: int) -> Dict[str, float]:
    pos_pred = input_dict["y_pred_pos"].detach().cpu()
    neg_pred = input_dict["y_pred_neg"].detach().cpu()
    if pos_pred.numel() == 0 or neg_pred.numel() == 0:
        return {f"hits@{k}": 0.0}
    kth_neg = torch.topk(neg_pred, min(k, neg_pred.numel())).values[-1]
    hits = (pos_pred > kth_neg).float().mean().item()
    return {f"hits@{k}": hits}
