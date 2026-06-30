from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def edge_batch_iterator(
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    heuristics: Optional[torch.Tensor],
    batch_size: int,
    shuffle: bool,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    num_edges = edge_label_index.size(1)
    indices = torch.randperm(num_edges) if shuffle else torch.arange(num_edges)
    for start in range(0, num_edges, batch_size):
        batch_ids = indices[start : start + batch_size]
        batch_edges = edge_label_index[:, batch_ids]
        batch_labels = edge_label[batch_ids]
        batch_heuristics = heuristics[batch_ids] if heuristics is not None else None
        yield batch_edges, batch_labels, batch_heuristics


def save_json(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def format_metric(result: Dict[str, float]) -> str:
    return "AUC={:.4f}, AP={:.4f}".format(result["AUC"], result["AP"])


def to_numpy_int_pairs(edge_label_index: torch.Tensor) -> np.ndarray:
    return edge_label_index.detach().cpu().t().numpy().astype(np.int64, copy=False)


def glorot_init(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.dim() > 1:
            torch.nn.init.xavier_uniform_(parameter)


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))
