from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from shap import TreeExplainer
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import BCEWithLogitsLoss, BatchNorm1d, Conv1d, Linear, MaxPool1d, ModuleList
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GAE,
    VGAE,
    GATConv,
    GCNConv,
    Node2Vec,
    SAGEConv,
    SuperGATConv,
    global_sort_pool,
)
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_undirected

from utils.data_utils import load_data
from utils.eval_utils import evaluate_auc_ap, summarize_results
from utils.train_utils import resolve_device, save_json, seed_everything


DISPLAY_NAME = {
    "cn": "CN",
    "aa": "AA",
    "node2vec": "Node2Vec",
    "gcn": "GCN",
    "gae": "GAE",
    "vgae": "VGAE",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "supergat": "SuperGAT",
    "seal": "SEAL",
    "bsal": "BSAL",
}


SOURCES = {
    "cn": {
        "implementation": "Closed-form heuristic implemented locally under the shared local protocol.",
        "source": "Classical common-neighbor heuristic; no external code dependency required.",
    },
    "aa": {
        "implementation": "Closed-form heuristic implemented locally under the shared local protocol.",
        "source": "Classical Adamic-Adar heuristic; no external code dependency required.",
    },
    "node2vec": {
        "implementation": "PyTorch Geometric 2.7.0 official Node2Vec implementation.",
        "source": "https://github.com/pyg-team/pytorch_geometric",
    },
    "gcn": {
        "implementation": "PyTorch Geometric 2.7.0 official message-passing components.",
        "source": "https://github.com/pyg-team/pytorch_geometric",
    },
    "gae": {
        "implementation": "PyTorch Geometric 2.7.0 official GAE implementation.",
        "source": "https://github.com/pyg-team/pytorch_geometric",
    },
    "vgae": {
        "implementation": "PyTorch Geometric 2.7.0 official VGAE implementation.",
        "source": "https://github.com/pyg-team/pytorch_geometric",
    },
    "graphsage": {
        "implementation": "PyTorch Geometric 2.7.0 official GraphSAGE components.",
        "source": "https://github.com/pyg-team/pytorch_geometric",
    },
    "gat": {
        "implementation": "PyTorch Geometric 2.7.0 official GAT components.",
        "source": "https://github.com/pyg-team/pytorch_geometric",
    },
    "supergat": {
        "implementation": "PyTorch Geometric 2.7.0 SuperGATConv plus the official project release.",
        "source": "https://github.com/dongkwan-kim/SuperGAT",
    },
    "seal": {
        "implementation": "Local root reproduction adapted from the public SEAL architecture and DGCNN-based subgraph classification pipeline.",
        "source": "https://github.com/muhanzhang/SEAL",
    },
    "bsal": {
        "implementation": (
            "Local root reproduction using the recovered bi-component fusion code fragment in the ASLAM author package "
            "plus the BSAL paper description. A standalone public BSAL repository was not robustly verified in this workspace."
        ),
        "source": "https://doi.org/10.1145/3477495.3531804",
    },
}


ROOT_COMPARE_PATTERNS = {
    "Citeseer": "formal_citeseer_compare_runs10_epochs401_pat20_seed2_h64_b16384.json",
    "DBLP": "formal_dblp_compare_runs10_epochs401_pat20_seed2_h64_b16384.json",
    "PubMed": "formal_pubmed_compare_runs10_epochs401_pat20_seed2_h64_b16384.json",
    "amz_Photo": "formal_amz_photo_compare_runs10_epochs401_pat20_seed2_h64_b16384.json",
    "CoRA": "formal_cora_compare_runs10_epochs401_pat20_seed2_h64_b4096.json",
    "Twitch_EN": "formal_twitch_en_compare_runs10_epochs401_pat20_seed2_h64_b8192.json",
}


@dataclass
class SplitBundle:
    train_data: Data
    val_data: Data
    test_data: Data


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper baselines under the local ASLAM-3 protocol.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="Citeseer,DBLP,PubMed,amz_Photo,CoRA,Twitch_EN",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="cn,aa,node2vec,gcn,gae,vgae,graphsage,gat,supergat,seal,bsal",
        help="Comma-separated baseline names.",
    )
    parser.add_argument("--root", type=str, default="datasets", help="Dataset root directory.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs.")
    parser.add_argument("--seed", type=int, default=2, help="Base random seed.")
    parser.add_argument("--epochs", type=int, default=401, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden size for full-graph baselines.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers for full-graph baselines.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout ratio for full-graph baselines.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Test ratio.")
    parser.add_argument("--neg_sampling_ratio", type=float, default=1.0, help="Negative sampling ratio.")
    parser.add_argument("--normalize_features", action="store_true", help="Row-normalize node features.")
    parser.add_argument("--node2vec_dim", type=int, default=64, help="Node2Vec embedding dimension.")
    parser.add_argument("--node2vec_epochs", type=int, default=30, help="Node2Vec training epochs.")
    parser.add_argument("--node2vec_batch_size", type=int, default=256, help="Node2Vec loader batch size.")
    parser.add_argument("--supergat_lambda", type=float, default=1.0, help="Weight for SuperGAT attention loss.")
    parser.add_argument("--seal_batch_size", type=int, default=32, help="Mini-batch size for SEAL.")
    parser.add_argument("--seal_hidden_channels", type=int, default=32, help="Hidden size for SEAL/BSAL DGCNN.")
    parser.add_argument("--seal_num_layers", type=int, default=3, help="Number of DGCNN layers for SEAL/BSAL.")
    parser.add_argument("--bsal_batch_size", type=int, default=32, help="Mini-batch size for BSAL.")
    parser.add_argument("--bsal_semantic_dim", type=int, default=32, help="Semantic branch dimension for BSAL.")
    parser.add_argument("--knn_k", type=int, default=10, help="K for semantic kNN graph in BSAL.")
    parser.add_argument("--cache_dir", type=str, default="results/paper_baseline_cache", help="Cache directory.")
    parser.add_argument("--results_dir", type=str, default="results/paper_baselines", help="Results directory.")
    parser.add_argument("--summary_name", type=str, default="paper_baselines_summary.json", help="Summary JSON filename.")
    return parser


def create_link_splits(
    data: Data,
    val_ratio: float,
    test_ratio: float,
    neg_sampling_ratio: float,
    seed: int,
) -> SplitBundle:
    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=neg_sampling_ratio,
        split_labels=False,
    )
    train_data, val_data, test_data = transform(data)
    return SplitBundle(train_data=train_data, val_data=val_data, test_data=test_data)


def split_pos_neg(edge_label_index: torch.Tensor, edge_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_mask = edge_label.to(torch.bool)
    neg_mask = ~pos_mask
    return edge_label_index[:, pos_mask], edge_label_index[:, neg_mask]


def dot_decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_label_index
    return (z[src] * z[dst]).sum(dim=-1)


def build_neighbor_sets(edge_index: torch.Tensor, num_nodes: int) -> List[set[int]]:
    neighbors = [set() for _ in range(num_nodes)]
    src, dst = edge_index.detach().cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:
            neighbors[u].add(v)
    return neighbors


def score_cn_aa(
    neighbor_sets: List[set[int]],
    edge_label_index: torch.Tensor,
    method: str,
) -> torch.Tensor:
    degrees = np.asarray([len(nbrs) for nbrs in neighbor_sets], dtype=np.float32)
    pairs = edge_label_index.detach().cpu().t().numpy().astype(np.int64, copy=False)
    scores = []
    for u, v in pairs:
        common = neighbor_sets[int(u)] & neighbor_sets[int(v)]
        if method == "cn":
            scores.append(float(len(common)))
            continue
        aa_score = 0.0
        for w in common:
            deg_w = max(float(degrees[w]), 2.0)
            aa_score += 1.0 / math.log(deg_w)
        scores.append(aa_score)
    return torch.tensor(scores, dtype=torch.float32)


def extract_edge_metrics(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    return evaluate_auc_ap(scores.detach().cpu(), labels.detach().cpu())


class DotProductGNN(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.model_name = model_name
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev_channels = in_channels
        for layer in range(num_layers):
            is_last = layer == num_layers - 1
            if model_name == "gcn":
                conv = GCNConv(prev_channels, hidden_channels)
                out_channels = hidden_channels
            elif model_name == "graphsage":
                conv = SAGEConv(prev_channels, hidden_channels)
                out_channels = hidden_channels
            elif model_name == "gat":
                if is_last:
                    conv = GATConv(prev_channels, hidden_channels, heads=1, concat=False, dropout=dropout)
                else:
                    conv = GATConv(prev_channels, hidden_channels // 4, heads=4, concat=True, dropout=dropout)
                out_channels = hidden_channels
            elif model_name == "supergat":
                if is_last:
                    conv = SuperGATConv(
                        prev_channels,
                        hidden_channels,
                        heads=1,
                        concat=False,
                        dropout=dropout,
                        attention_type="MX",
                        neg_sample_ratio=0.5,
                        is_undirected=True,
                    )
                else:
                    conv = SuperGATConv(
                        prev_channels,
                        hidden_channels // 4,
                        heads=4,
                        concat=True,
                        dropout=dropout,
                        attention_type="MX",
                        neg_sample_ratio=0.5,
                        is_undirected=True,
                    )
                out_channels = hidden_channels
            else:
                raise ValueError(f"Unsupported GNN baseline: {model_name}")
            self.convs.append(conv)
            if not is_last:
                self.norms.append(BatchNorm1d(hidden_channels))
            prev_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        neg_edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x
        for layer, conv in enumerate(self.convs):
            if self.model_name == "supergat":
                h = conv(h, edge_index, neg_edge_index=neg_edge_index)
            else:
                h = conv(h, edge_index)
            if layer != len(self.convs) - 1:
                h = self.norms[layer](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def attention_loss(self) -> torch.Tensor:
        if self.model_name != "supergat":
            parameter = next(self.parameters())
            return parameter.new_zeros(())
        losses = [conv.get_attention_loss() for conv in self.convs if hasattr(conv, "get_attention_loss")]
        if not losses:
            parameter = next(self.parameters())
            return parameter.new_zeros(())
        return torch.stack(losses).sum()


class GAEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class VGAEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, hidden_channels)
        self.conv_logstd = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class SubgraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "target_nodes":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def _limit_split(
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    keep_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if keep_ratio >= 1.0:
        return edge_label_index, edge_label
    num_edges = edge_label.numel()
    keep = max(1, int(math.ceil(num_edges * keep_ratio)))
    perm = torch.randperm(num_edges)[:keep]
    return edge_label_index[:, perm], edge_label[perm]


def _extract_subgraph(
    full_data: Data,
    message_edge_index: torch.Tensor,
    src: int,
    dst: int,
    num_hops: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        [src, dst],
        num_hops=num_hops,
        edge_index=message_edge_index,
        relabel_nodes=True,
        num_nodes=full_data.num_nodes,
    )
    src_local, dst_local = mapping.tolist()
    keep_mask = ~(
        ((sub_edge_index[0] == src_local) & (sub_edge_index[1] == dst_local))
        | ((sub_edge_index[0] == dst_local) & (sub_edge_index[1] == src_local))
    )
    sub_edge_index = sub_edge_index[:, keep_mask]
    return subset, sub_edge_index, src_local, dst_local


def _drnl_node_labeling(edge_index: torch.Tensor, src: int, dst: int, num_nodes: int) -> torch.Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adjacency[edge_index[0], edge_index[1]] = True

    def bfs(start: int) -> torch.Tensor:
        distance = torch.full((num_nodes,), fill_value=num_nodes, dtype=torch.long)
        distance[start] = 0
        frontier = [start]
        while frontier:
            new_frontier = []
            for node in frontier:
                neighbors = adjacency[node].nonzero(as_tuple=False).view(-1).tolist()
                for nbr in neighbors:
                    if distance[nbr] > distance[node] + 1:
                        distance[nbr] = distance[node] + 1
                        new_frontier.append(nbr)
            frontier = new_frontier
        return distance

    dist_src = bfs(src)
    dist_dst = bfs(dst)
    total = dist_src + dist_dst
    total_half = torch.div(total, 2, rounding_mode="floor")
    total_mod = total % 2
    labels = 1 + torch.min(dist_src, dist_dst)
    labels = labels + total_half * (total_half + total_mod - 1)
    labels[src] = 1
    labels[dst] = 1
    labels[(dist_src >= num_nodes) | (dist_dst >= num_nodes)] = 0
    return labels.to(torch.long)


def _degree_labels(edge_index: torch.Tensor, num_nodes: int, max_degree: int) -> torch.Tensor:
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    return degree.clamp(max=max_degree)


def _one_hot_feature(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.clamp(min=0, max=num_classes - 1)
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32)


class LabeledSubgraphDataset(InMemoryDataset):
    def __init__(
        self,
        base_data: Data,
        split_bundle: SplitBundle,
        split: str,
        label_mode: str,
        num_hops: int = 1,
        keep_ratio: float = 1.0,
        subgraph_cache: Optional[Dict[Tuple[str, int, int], Tuple[torch.Tensor, int, int, int]]] = None,
        cache_path: Optional[Path] = None,
    ):
        self.base_data = base_data
        self.split_bundle = split_bundle
        self.split = split
        self.label_mode = label_mode
        self.num_hops = num_hops
        self.keep_ratio = keep_ratio
        self.subgraph_cache = {} if subgraph_cache is None else subgraph_cache
        self.cache_path = cache_path
        super().__init__(".")
        if self.cache_path is not None and self.cache_path.exists():
            payload = load_torch_cache(self.cache_path)
            self._data = payload["data"]
            self.slices = payload["slices"]
        else:
            data_list = self._build_graphs()
            self._data, self.slices = self.collate(data_list)
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"data": self._data, "slices": self.slices}, self.cache_path)

    def _select_split(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.split == "train":
            split_data = self.split_bundle.train_data
        elif self.split == "val":
            split_data = self.split_bundle.val_data
        else:
            split_data = self.split_bundle.test_data
        edge_label_index, edge_label = _limit_split(
            split_data.edge_label_index,
            split_data.edge_label,
            self.keep_ratio,
        )
        return self.split_bundle.train_data.edge_index, edge_label_index, edge_label

    def _build_graphs(self) -> List[SubgraphData]:
        message_edge_index, edge_label_index, edge_label = self._select_split()
        graphs = []
        for idx in range(edge_label.numel()):
            src = int(edge_label_index[0, idx])
            dst = int(edge_label_index[1, idx])
            pair_key = tuple(sorted((src, dst)))
            cache_key = (self.split, pair_key[0], pair_key[1])
            cached = self.subgraph_cache.get(cache_key)
            if cached is None:
                subset, sub_edge_index, src_local, dst_local = _extract_subgraph(
                    self.base_data,
                    message_edge_index,
                    src,
                    dst,
                    self.num_hops,
                )
                cached = (sub_edge_index, src_local, dst_local, int(subset.numel()))
                self.subgraph_cache[cache_key] = cached
            sub_edge_index, src_local, dst_local, num_nodes = cached
            if self.label_mode == "drnl":
                labels = _drnl_node_labeling(sub_edge_index, src_local, dst_local, num_nodes).clamp(max=50)
                x = _one_hot_feature(labels, 51)
            elif self.label_mode == "de":
                labels = _degree_labels(sub_edge_index, num_nodes, max_degree=32)
                x = _one_hot_feature(labels, 33)
            else:
                raise ValueError(f"Unsupported label mode: {self.label_mode}")
            graph = SubgraphData(
                x=x,
                edge_index=sub_edge_index,
                z=labels,
                y=edge_label[idx].view(1).to(torch.float32),
                target_nodes=torch.tensor([src, dst], dtype=torch.long),
                num_nodes=num_nodes,
            )
            graphs.append(graph)
        return graphs


class DGCNN(nn.Module):
    def __init__(
        self,
        train_dataset: Sequence[Data],
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        k: float = 0.6,
    ):
        super().__init__()
        if k < 1:
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, out_channels)

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs.append(torch.tanh(conv(xs[-1], edge_index)))
        x = torch.cat(xs[1:], dim=-1)
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.lin1(x)


class BSALStyleModel(nn.Module):
    def __init__(
        self,
        train_dataset_drnl: Sequence[Data],
        train_dataset_de: Sequence[Data],
        semantic_dim: int,
        hidden_channels: int,
        num_layers: int,
    ):
        super().__init__()
        self.dgcnn_drnl = DGCNN(
            train_dataset_drnl,
            in_channels=train_dataset_drnl[0].num_features,
            hidden_channels=hidden_channels,
            out_channels=16,
            num_layers=num_layers,
        )
        self.dgcnn_de = DGCNN(
            train_dataset_de,
            in_channels=train_dataset_de[0].num_features,
            hidden_channels=hidden_channels,
            out_channels=16,
            num_layers=num_layers,
        )
        self.semantic_proj = nn.Identity() if semantic_dim == 32 else Linear(semantic_dim, 32)
        self.lin1 = Linear(32, 1)
        self.lin2 = Linear(32, 1)
        self.lin3 = Linear(32, 1)
        self.att_1 = Linear(32, 16)
        self.att_2 = Linear(32, 16)
        self.query = Linear(16, 1)

    def forward(self, data_drnl: Batch, data_de: Batch, semantic_graph: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        structure_1 = self.dgcnn_drnl(data_drnl)
        structure_2 = self.dgcnn_de(data_de)
        structure = torch.cat([structure_1, structure_2], dim=-1)

        semantic = semantic_graph.x[data_drnl.target_nodes]
        semantic = semantic.view(-1, 2, semantic.size(-1)).sum(dim=1)
        semantic = self.semantic_proj(semantic)

        att_structure = self.query(torch.tanh(self.att_1(structure)))
        att_semantic = self.query(torch.tanh(self.att_2(semantic)))
        alpha = torch.softmax(torch.cat([att_structure, att_semantic], dim=-1), dim=-1)
        fused = alpha[:, :1] * structure + alpha[:, 1:] * semantic

        output_structure = self.lin1(structure)
        output_semantic = self.lin2(semantic)
        output_fused = self.lin3(fused)
        return output_structure, output_semantic, output_fused


def _aggregate_multiclass_shap(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        return np.mean(np.abs(shap_array), axis=1)
    return np.abs(shap_array)


def shap_weight_embeddings(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x_np = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    y_np = labels.detach().cpu().numpy()
    unique_labels = np.unique(y_np)
    if unique_labels.size <= 1:
        return embeddings.detach().cpu()

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
    return torch.from_numpy(importance).to(torch.float32) * embeddings.detach().cpu()


def construct_knn_graph(data: Data, k: int) -> Data:
    x_np = np.nan_to_num(data.x.detach().cpu().numpy().astype(np.float32, copy=False))
    n_neighbors = min(k + 1, x_np.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(x_np)
    indices = nbrs.kneighbors(return_distance=False)
    if indices.shape[1] > 1:
        indices = indices[:, 1:]
    else:
        indices = indices[:, :0]
    row = np.repeat(np.arange(x_np.shape[0], dtype=np.int64), indices.shape[1])
    col = indices.reshape(-1).astype(np.int64, copy=False)
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    edge_index = to_undirected(edge_index, num_nodes=x_np.shape[0])
    edge_index, _ = remove_self_loops(edge_index)
    return Data(edge_index=edge_index, x=torch.tensor(x_np, dtype=torch.float32), num_nodes=x_np.shape[0])


def train_node2vec_embeddings(
    edge_index: torch.Tensor,
    num_nodes: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model = Node2Vec(
        edge_index,
        num_nodes=num_nodes,
        embedding_dim=embedding_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        sparse=True,
    ).to(device)
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
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


def get_bsal_semantic_graph(
    args,
    dataset_name: str,
    run_seed: int,
    data: Data,
    device: torch.device,
) -> Data:
    cache_dir = Path(args.cache_dir) / "bsal_semantic"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = dataset_name.lower().replace("/", "_")
    cache_path = cache_dir / f"{dataset_key}_seed{run_seed}_k{args.knn_k}_dim{args.bsal_semantic_dim}.pt"
    if cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        return Data(edge_index=payload["edge_index"], x=payload["x"], num_nodes=payload["x"].size(0))

    knn_graph = construct_knn_graph(data, k=args.knn_k)
    embeddings = train_node2vec_embeddings(
        edge_index=knn_graph.edge_index,
        num_nodes=data.num_nodes,
        embedding_dim=args.bsal_semantic_dim,
        epochs=args.node2vec_epochs,
        batch_size=args.node2vec_batch_size,
        device=device,
    )
    weighted = shap_weight_embeddings(embeddings, data.y)
    payload = {"edge_index": knn_graph.edge_index.cpu(), "x": weighted.cpu()}
    torch.save(payload, cache_path)
    return Data(edge_index=payload["edge_index"], x=payload["x"], num_nodes=payload["x"].size(0))


def build_subgraph_dataset_cache_path(
    args,
    dataset_name: str,
    run_seed: int,
    split: str,
    label_mode: str,
    num_hops: int,
    keep_ratio: float,
) -> Path:
    dataset_key = dataset_name.lower().replace("/", "_")
    keep_tag = str(keep_ratio).replace(".", "p")
    filename = f"{dataset_key}_seed{run_seed}_{split}_{label_mode}_{num_hops}hop_keep{keep_tag}.pt"
    return Path(args.cache_dir) / "subgraphs" / filename


def load_torch_cache(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def paired_batch_iterator(
    dataset_a: Sequence[Data],
    dataset_b: Sequence[Data],
    batch_size: int,
    shuffle: bool,
) -> Iterator[Tuple[Batch, Batch]]:
    indices = torch.randperm(len(dataset_a)).tolist() if shuffle else list(range(len(dataset_a)))
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        batch_a = Batch.from_data_list([dataset_a[idx] for idx in batch_ids])
        batch_b = Batch.from_data_list([dataset_b[idx] for idx in batch_ids])
        yield batch_a, batch_b


@torch.no_grad()
def evaluate_fullbatch_model(
    model: DotProductGNN,
    x: torch.Tensor,
    train_edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    z = model(x, train_edge_index)
    logits = dot_decode(z, edge_label_index)
    return extract_edge_metrics(logits, edge_label)


@torch.no_grad()
def evaluate_autoencoder_model(
    model: GAE | VGAE,
    x: torch.Tensor,
    train_edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    z = model.encode(x, train_edge_index)
    logits = model.decoder(z, edge_label_index, sigmoid=False).view(-1)
    return extract_edge_metrics(logits, edge_label)


@torch.no_grad()
def evaluate_subgraph_model(loader: DataLoader, model: DGCNN, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_score = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).view(-1).cpu()
        y_score.append(logits)
        y_true.append(batch.y.view(-1).cpu())
    return extract_edge_metrics(torch.cat(y_score, dim=0), torch.cat(y_true, dim=0))


@torch.no_grad()
def evaluate_bsal_model(
    dataset_drnl: Sequence[Data],
    dataset_de: Sequence[Data],
    model: BSALStyleModel,
    semantic_graph: Data,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_score = []
    for batch_drnl, batch_de in paired_batch_iterator(dataset_drnl, dataset_de, batch_size, shuffle=False):
        batch_drnl = batch_drnl.to(device)
        batch_de = batch_de.to(device)
        _, _, logits = model(batch_drnl, batch_de, semantic_graph)
        y_score.append(logits.view(-1).cpu())
        y_true.append(batch_drnl.y.view(-1).cpu())
    return extract_edge_metrics(torch.cat(y_score, dim=0), torch.cat(y_true, dim=0))


def run_heuristic_baseline(
    model_name: str,
    data: Data,
    split_bundle: SplitBundle,
) -> Dict[str, object]:
    neighbor_sets = build_neighbor_sets(split_bundle.train_data.edge_index, num_nodes=data.num_nodes)
    val_scores = score_cn_aa(neighbor_sets, split_bundle.val_data.edge_label_index, method=model_name)
    test_scores = score_cn_aa(neighbor_sets, split_bundle.test_data.edge_label_index, method=model_name)
    return {
        "best_epoch": 0,
        "val": extract_edge_metrics(val_scores, split_bundle.val_data.edge_label),
        "test": extract_edge_metrics(test_scores, split_bundle.test_data.edge_label),
        "notes": "No trainable parameters; scores are computed directly from the training graph neighborhoods.",
    }


def run_node2vec_baseline(
    args,
    data: Data,
    split_bundle: SplitBundle,
    device: torch.device,
) -> Dict[str, object]:
    embeddings = train_node2vec_embeddings(
        edge_index=split_bundle.train_data.edge_index.to(device),
        num_nodes=data.num_nodes,
        embedding_dim=args.node2vec_dim,
        epochs=args.node2vec_epochs,
        batch_size=args.node2vec_batch_size,
        device=device,
    )
    val_scores = dot_decode(embeddings, split_bundle.val_data.edge_label_index.cpu())
    test_scores = dot_decode(embeddings, split_bundle.test_data.edge_label_index.cpu())
    return {
        "best_epoch": args.node2vec_epochs,
        "val": extract_edge_metrics(val_scores, split_bundle.val_data.edge_label),
        "test": extract_edge_metrics(test_scores, split_bundle.test_data.edge_label),
        "notes": "Unsupervised Node2Vec embeddings with dot-product edge decoding.",
    }


def run_message_passing_baseline(
    args,
    model_name: str,
    data: Data,
    split_bundle: SplitBundle,
    device: torch.device,
) -> Dict[str, object]:
    x = data.x.to(device)
    train_edge_index = split_bundle.train_data.edge_index.to(device)
    train_edge_label_index = split_bundle.train_data.edge_label_index.to(device)
    train_edge_label = split_bundle.train_data.edge_label.to(device).float()
    val_edge_label_index = split_bundle.val_data.edge_label_index.to(device)
    val_edge_label = split_bundle.val_data.edge_label.to(device)
    test_edge_label_index = split_bundle.test_data.edge_label_index.to(device)
    test_edge_label = split_bundle.test_data.edge_label.to(device)
    _, train_neg_edge_index = split_pos_neg(split_bundle.train_data.edge_label_index, split_bundle.train_data.edge_label)
    train_neg_edge_index = train_neg_edge_index.to(device)

    model = DotProductGNN(
        model_name=model_name,
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = BCEWithLogitsLoss()

    best_epoch = 0
    best_val = {"AUC": 0.0, "AP": 0.0}
    best_test = {"AUC": 0.0, "AP": 0.0}
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(
            x,
            train_edge_index,
            neg_edge_index=train_neg_edge_index if model_name == "supergat" else None,
        )
        logits = dot_decode(z, train_edge_label_index)
        loss = criterion(logits, train_edge_label)
        if model_name == "supergat":
            loss = loss + args.supergat_lambda * model.attention_loss()
        loss.backward()
        optimizer.step()

        val_metric = evaluate_fullbatch_model(
            model=model,
            x=x,
            train_edge_index=train_edge_index,
            edge_label_index=val_edge_label_index,
            edge_label=val_edge_label,
        )
        if val_metric["AUC"] > best_val["AUC"]:
            best_val = val_metric
            best_test = evaluate_fullbatch_model(
                model=model,
                x=x,
                train_edge_index=train_edge_index,
                edge_label_index=test_edge_label_index,
                edge_label=test_edge_label,
            )
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break

    return {
        "best_epoch": best_epoch,
        "val": best_val,
        "test": best_test,
        "notes": f"Full-batch {DISPLAY_NAME[model_name]} encoder with dot-product edge decoder.",
    }


def run_autoencoder_baseline(
    args,
    model_name: str,
    data: Data,
    split_bundle: SplitBundle,
    device: torch.device,
) -> Dict[str, object]:
    x = data.x.to(device)
    train_edge_index = split_bundle.train_data.edge_index.to(device)
    val_edge_label_index = split_bundle.val_data.edge_label_index.to(device)
    val_edge_label = split_bundle.val_data.edge_label.to(device)
    test_edge_label_index = split_bundle.test_data.edge_label_index.to(device)
    test_edge_label = split_bundle.test_data.edge_label.to(device)
    train_pos_edge_index, train_neg_edge_index = split_pos_neg(
        split_bundle.train_data.edge_label_index,
        split_bundle.train_data.edge_label,
    )
    train_pos_edge_index = train_pos_edge_index.to(device)
    train_neg_edge_index = train_neg_edge_index.to(device)

    if model_name == "gae":
        model = GAE(GAEEncoder(data.num_features, args.hidden_channels, args.dropout)).to(device)
    elif model_name == "vgae":
        model = VGAE(VGAEEncoder(data.num_features, args.hidden_channels, args.dropout)).to(device)
    else:
        raise ValueError(f"Unsupported autoencoder baseline: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_epoch = 0
    best_val = {"AUC": 0.0, "AP": 0.0}
    best_test = {"AUC": 0.0, "AP": 0.0}
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index, train_neg_edge_index)
        if model_name == "vgae":
            loss = loss + model.kl_loss() / max(data.num_nodes, 1)
        loss.backward()
        optimizer.step()

        val_metric = evaluate_autoencoder_model(
            model=model,
            x=x,
            train_edge_index=train_edge_index,
            edge_label_index=val_edge_label_index,
            edge_label=val_edge_label,
        )
        if val_metric["AUC"] > best_val["AUC"]:
            best_val = val_metric
            best_test = evaluate_autoencoder_model(
                model=model,
                x=x,
                train_edge_index=train_edge_index,
                edge_label_index=test_edge_label_index,
                edge_label=test_edge_label,
            )
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break

    return {
        "best_epoch": best_epoch,
        "val": best_val,
        "test": best_test,
        "notes": f"{DISPLAY_NAME[model_name]} with the standard inner-product decoder under the shared split protocol.",
    }


def run_seal_baseline(
    args,
    dataset_name: str,
    data: Data,
    split_bundle: SplitBundle,
    run_seed: int,
    device: torch.device,
) -> Dict[str, object]:
    subgraph_cache: Dict[Tuple[str, int, int], Tuple[torch.Tensor, int, int, int]] = {}
    train_dataset = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="train",
        label_mode="drnl",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "train", "drnl", 1, 1.0),
    )
    val_dataset = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="val",
        label_mode="drnl",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "val", "drnl", 1, 1.0),
    )
    test_dataset = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="test",
        label_mode="drnl",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "test", "drnl", 1, 1.0),
    )
    model = DGCNN(
        train_dataset=train_dataset,
        in_channels=train_dataset[0].num_features,
        hidden_channels=args.seal_hidden_channels,
        out_channels=1,
        num_layers=args.seal_num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = BCEWithLogitsLoss()
    train_loader = DataLoader(train_dataset, batch_size=args.seal_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.seal_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.seal_batch_size, shuffle=False)

    best_epoch = 0
    best_val = {"AUC": 0.0, "AP": 0.0}
    best_test = {"AUC": 0.0, "AP": 0.0}
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch).view(-1)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()

        val_metric = evaluate_subgraph_model(val_loader, model, device)
        if val_metric["AUC"] > best_val["AUC"]:
            best_val = val_metric
            best_test = evaluate_subgraph_model(test_loader, model, device)
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break

    return {
        "best_epoch": best_epoch,
        "val": best_val,
        "test": best_test,
        "notes": "1-hop DRNL-labeled subgraph classification with DGCNN, aligned to the SEAL paradigm.",
    }


def run_bsal_baseline(
    args,
    dataset_name: str,
    data: Data,
    split_bundle: SplitBundle,
    run_seed: int,
    device: torch.device,
) -> Dict[str, object]:
    semantic_graph = get_bsal_semantic_graph(args, dataset_name, run_seed, data, device).to(device)
    subgraph_cache: Dict[Tuple[str, int, int], Tuple[torch.Tensor, int, int, int]] = {}
    train_drnl = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="train",
        label_mode="drnl",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "train", "drnl", 1, 1.0),
    )
    val_drnl = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="val",
        label_mode="drnl",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "val", "drnl", 1, 1.0),
    )
    test_drnl = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="test",
        label_mode="drnl",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "test", "drnl", 1, 1.0),
    )
    train_de = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="train",
        label_mode="de",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "train", "de", 1, 1.0),
    )
    val_de = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="val",
        label_mode="de",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "val", "de", 1, 1.0),
    )
    test_de = LabeledSubgraphDataset(
        data,
        split_bundle,
        split="test",
        label_mode="de",
        subgraph_cache=subgraph_cache,
        cache_path=build_subgraph_dataset_cache_path(args, dataset_name, run_seed, "test", "de", 1, 1.0),
    )

    model = BSALStyleModel(
        train_dataset_drnl=train_drnl,
        train_dataset_de=train_de,
        semantic_dim=semantic_graph.x.size(-1),
        hidden_channels=args.seal_hidden_channels,
        num_layers=args.seal_num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = BCEWithLogitsLoss()

    best_epoch = 0
    best_val = {"AUC": 0.0, "AP": 0.0}
    best_test = {"AUC": 0.0, "AP": 0.0}
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_drnl, batch_de in paired_batch_iterator(train_drnl, train_de, args.bsal_batch_size, shuffle=True):
            batch_drnl = batch_drnl.to(device)
            batch_de = batch_de.to(device)
            optimizer.zero_grad()
            logits_structure, logits_semantic, logits_fused = model(batch_drnl, batch_de, semantic_graph)
            labels = batch_drnl.y.view(-1)
            loss = (
                criterion(logits_structure.view(-1), labels)
                + criterion(logits_semantic.view(-1), labels)
                + criterion(logits_fused.view(-1), labels)
            ) / 3.0
            loss.backward()
            optimizer.step()

        val_metric = evaluate_bsal_model(val_drnl, val_de, model, semantic_graph, args.bsal_batch_size, device)
        if val_metric["AUC"] > best_val["AUC"]:
            best_val = val_metric
            best_test = evaluate_bsal_model(test_drnl, test_de, model, semantic_graph, args.bsal_batch_size, device)
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break

    return {
        "best_epoch": best_epoch,
        "val": best_val,
        "test": best_test,
        "notes": "Two-component subgraph-and-semantic fusion benchmark following the recovered BSAL-style pipeline.",
    }


def run_single_model_on_dataset(
    args,
    dataset_name: str,
    model_name: str,
    run_seed: int,
    device: torch.device,
) -> Dict[str, object]:
    seed_everything(run_seed)
    data = load_data(dataset_name, root=args.root, normalize_features=args.normalize_features)
    split_bundle = create_link_splits(
        data=data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        neg_sampling_ratio=args.neg_sampling_ratio,
        seed=run_seed,
    )
    if model_name in {"cn", "aa"}:
        result = run_heuristic_baseline(model_name, data, split_bundle)
    elif model_name == "node2vec":
        result = run_node2vec_baseline(args, data, split_bundle, device)
    elif model_name in {"gcn", "graphsage", "gat", "supergat"}:
        result = run_message_passing_baseline(args, model_name, data, split_bundle, device)
    elif model_name in {"gae", "vgae"}:
        result = run_autoencoder_baseline(args, model_name, data, split_bundle, device)
    elif model_name == "seal":
        result = run_seal_baseline(args, dataset_name, data, split_bundle, run_seed, device)
    elif model_name == "bsal":
        result = run_bsal_baseline(args, dataset_name, data, split_bundle, run_seed, device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    result.update(
        {
            "dataset": dataset_name,
            "model": model_name,
            "seed": run_seed,
            "graph": {
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.edge_index.size(1) // 2),
                "num_features": int(data.num_features),
            },
        }
    )
    return result


def summarize_runs(runs: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "runs": runs,
        "test_auc": summarize_results([item["test"]["AUC"] for item in runs]),
        "test_ap": summarize_results([item["test"]["AP"] for item in runs]),
        "best_epoch": summarize_results([item["best_epoch"] for item in runs]),
        "val_auc": summarize_results([item["val"]["AUC"] for item in runs]),
        "val_ap": summarize_results([item["val"]["AP"] for item in runs]),
        "notes": runs[0].get("notes", "") if runs else "",
    }


def load_root_formal_results(results_root: Path) -> Dict[str, Dict[str, object]]:
    aggregated: Dict[str, Dict[str, object]] = {}
    for dataset_name, filename in ROOT_COMPARE_PATTERNS.items():
        path = results_root / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset_payload = payload.get("datasets", {}).get(dataset_name, {})
        if not dataset_payload:
            continue
        slim_payload = {}
        for model_name in ("aslam", "aslam_ssmattn"):
            model_payload = dataset_payload.get(model_name)
            if not model_payload:
                continue
            slim_payload[model_name] = {
                "test_auc": model_payload["test_auc"],
                "test_ap": model_payload["test_ap"],
                "best_epoch": model_payload["best_epoch"],
                "notes": "Loaded from existing root formal comparison JSON.",
            }
        aggregated[dataset_name] = {
            "aslam": slim_payload.get("aslam"),
            "aslam_ssmattn": slim_payload.get("aslam_ssmattn"),
        }
    return aggregated


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    device = resolve_device(args.device)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    output_path = Path(args.results_dir) / args.summary_name

    if output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        results = {"datasets": {}}
    results["args"] = vars(args)
    results["device"] = str(device)
    results["sources"] = SOURCES
    results.setdefault("datasets", {})
    results["root_formal_models"] = load_root_formal_results(Path("results"))
    start = time.time()
    for dataset_name in datasets:
        results["datasets"].setdefault(dataset_name, {})
        for model_name in models:
            print(f"\n=== dataset={dataset_name} model={model_name} ===")
            existing = results["datasets"][dataset_name].get(model_name, {})
            runs = list(existing.get("runs", []))
            completed_seeds = {int(item["seed"]) for item in runs}
            for run_id in range(args.runs):
                run_seed = args.seed + run_id
                if run_seed in completed_seeds:
                    print(f"Skipping seed={run_seed} (already saved)")
                    continue
                print(f"Running seed={run_seed}")
                run_result = run_single_model_on_dataset(args, dataset_name, model_name, run_seed, device)
                runs.append(run_result)
                runs.sort(key=lambda item: int(item["seed"]))
                print(
                    f"seed={run_seed} best_epoch={run_result['best_epoch']} "
                    f"val_auc={run_result['val']['AUC']:.4f} test_auc={run_result['test']['AUC']:.4f} "
                    f"test_ap={run_result['test']['AP']:.4f}"
                )
                results["datasets"][dataset_name][model_name] = summarize_runs(runs)
                results["elapsed_seconds"] = float(results.get("elapsed_seconds", 0.0)) + (time.time() - start)
                save_json(results, output_path)
                start = time.time()
            results["datasets"][dataset_name][model_name] = summarize_runs(runs)

    results["elapsed_seconds"] = float(results.get("elapsed_seconds", 0.0)) + (time.time() - start)
    save_json(results, output_path)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
