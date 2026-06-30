from __future__ import annotations

import argparse
import random
from typing import Any, Dict

import numpy as np
import torch
import xgboost as xgb
from shap import TreeExplainer
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec


def add_flags_from_config(parser: argparse.ArgumentParser, config_dict: Dict[str, Any]) -> argparse.ArgumentParser:
    for param, (default, param_type, help_text) in config_dict.items():
        if param_type is bool:
            parser.add_argument(
                f"--{param}",
                type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
                default=default,
                help=help_text,
            )
        else:
            parser.add_argument(f"--{param}", type=param_type, default=default, help=help_text)
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construct_knn_graph(data: Data, k: int = 10) -> Data:
    x = data.x.detach().cpu().to(torch.float32)
    x = torch.nan_to_num(x)
    num_nodes = x.size(0)
    pairwise = torch.cdist(x, x, p=2)
    pairwise.fill_diagonal_(float("inf"))
    k = max(1, min(k, num_nodes - 1))
    knn = torch.topk(pairwise, k=k, largest=False, dim=1).indices

    row = torch.arange(num_nodes).view(-1, 1).repeat(1, k).reshape(-1)
    col = knn.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return Data(edge_index=edge_index, x=x.clone(), num_nodes=num_nodes)


def train_node2vec_emb(knn_graph: Data, embedding_dim: int = 32, epochs: int = 30) -> torch.Tensor:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Node2Vec(
        knn_graph.edge_index,
        embedding_dim=embedding_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        sparse=True,
    ).to(device)
    loader = model.loader(batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        return model().detach().cpu()


def _aggregate_multiclass_shap(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        return np.mean(np.abs(shap_array), axis=1)
    return np.abs(shap_array)


def shxgb(emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x_np = emb.detach().cpu().numpy().astype(np.float32, copy=False)
    y_np = labels.detach().cpu().numpy()
    unique_labels = np.unique(y_np)
    if unique_labels.size <= 1:
        return emb

    importance_acc = np.zeros_like(x_np)
    for class_id in unique_labels:
        target = (y_np == class_id).astype(np.int32)
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=2,
        )
        model.fit(x_np, target)
        try:
            explainer = TreeExplainer(model)
            shap_values = explainer.shap_values(x_np)
            importance_acc += _aggregate_multiclass_shap(shap_values)
        except Exception:
            importance_acc += np.tile(np.abs(model.feature_importances_), (x_np.shape[0], 1))

    importance = importance_acc / float(unique_labels.size)
    return torch.from_numpy(importance).to(torch.float32) * emb.detach().cpu()


def sh(emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return shxgb(emb, labels)
