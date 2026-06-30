from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_undirected


SUPPORTED_DATASETS = {
    "citeseer",
    "pubmed",
    "dblp",
    "photo",
    "amz_photo",
    "amazon_photo",
}


class SubgraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "target_nodes":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


@dataclass
class SplitBundle:
    train_data: Data
    val_data: Data
    test_data: Data


def _label_encode(labels: List[str]) -> torch.Tensor:
    vocab = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return torch.tensor([vocab[label] for label in labels], dtype=torch.long)


def _build_undirected_edge_index(edges: Iterable[Tuple[int, int]], num_nodes: int) -> torch.Tensor:
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index


def _load_citeseer(root: Path) -> Data:
    content_path = root / "Citeseer" / "citeseer.content"
    cites_path = root / "Citeseer" / "citeseer.cites"

    id_to_idx: Dict[str, int] = {}
    features: List[List[float]] = []
    labels: List[str] = []
    with content_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            id_to_idx[parts[0]] = idx
            features.append([float(v) for v in parts[1:-1]])
            labels.append(parts[-1])

    edges: List[Tuple[int, int]] = []
    with cites_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] in id_to_idx and parts[1] in id_to_idx:
                edges.append((id_to_idx[parts[0]], id_to_idx[parts[1]]))

    x = torch.tensor(np.asarray(features, dtype=np.float32))
    y = _label_encode(labels)
    edge_index = _build_undirected_edge_index(edges, x.size(0))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=x.size(0))


def _load_pubmed(root: Path) -> Data:
    node_path = root / "PubMed" / "Pubmed-Diabetes.NODE.paper.tab"
    edge_path = root / "PubMed" / "Pubmed-Diabetes.DIRECTED.cites.tab"

    lines = node_path.read_text(encoding="utf-8").splitlines()
    header_tokens = lines[1].split("\t")
    feature_names = [token.split(":")[1] for token in header_tokens[1:-1]]
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    paper_ids: List[str] = []
    labels: List[int] = []
    features: List[List[float]] = []
    for line in lines[2:]:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        feat_values = np.zeros(len(feature_names), dtype=np.float32)
        for token in parts[2:-1]:
            if "=" not in token:
                continue
            feature_name, feature_value = token.split("=", 1)
            idx = feature_to_idx.get(feature_name)
            if idx is not None:
                feat_values[idx] = float(feature_value)
        paper_ids.append(parts[0])
        labels.append(int(parts[1].split("=")[-1]) - 1)
        features.append(feat_values.tolist())

    id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    edges: List[Tuple[int, int]] = []
    with edge_path.open("r", encoding="utf-8") as f:
        next(f)
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            src = parts[1].split(":")[-1]
            dst = parts[3].split(":")[-1]
            if src in id_to_idx and dst in id_to_idx:
                edges.append((id_to_idx[src], id_to_idx[dst]))

    x = torch.tensor(np.asarray(features, dtype=np.float32))
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = _build_undirected_edge_index(edges, x.size(0))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=x.size(0))


def _load_dblp(root: Path) -> Data:
    dblp_root = root / "DBLP"
    x = torch.tensor(np.load(dblp_root / "dblp_feat.npy"), dtype=torch.float32)
    y = torch.tensor(np.load(dblp_root / "dblp_label.npy"), dtype=torch.long)
    adj = np.load(dblp_root / "dblp_adj.npy")
    row, col = np.nonzero(np.triu(adj, k=1))
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))
    edge_index, _ = remove_self_loops(edge_index)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=x.size(0))


def _load_photo(root: Path) -> Data:
    npz_path = root / "amz_Photo" / "ms_academic_cs.npz"
    payload = np.load(npz_path, allow_pickle=True)
    adj = sp.csr_matrix(
        (payload["adj_data"], payload["adj_indices"], payload["adj_indptr"]),
        shape=tuple(payload["adj_shape"]),
    )
    attrs = sp.csr_matrix(
        (payload["attr_data"], payload["attr_indices"], payload["attr_indptr"]),
        shape=tuple(payload["attr_shape"]),
    )
    row, col = sp.triu(adj, k=1).nonzero()
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=adj.shape[0])
    edge_index, _ = remove_self_loops(edge_index)
    x = torch.tensor(attrs.toarray(), dtype=torch.float32)
    y = torch.tensor(payload["labels"], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=x.size(0))


def load_data(dataset_name: str, root: str = "../../datasets"):
    dataset_key = dataset_name.lower()
    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported now: {sorted(SUPPORTED_DATASETS)}")

    root_path = Path(root)
    if dataset_key == "citeseer":
        data = _load_citeseer(root_path)
    elif dataset_key == "pubmed":
        data = _load_pubmed(root_path)
    elif dataset_key == "dblp":
        data = _load_dblp(root_path)
    else:
        data = _load_photo(root_path)
    return [data]


def _make_split_bundle(data: Data, val_ratio: float, test_ratio: float, seed: int) -> SplitBundle:
    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        split_labels=False,
    )
    train_data, val_data, test_data = transform(data)
    return SplitBundle(train_data=train_data, val_data=val_data, test_data=test_data)


def make_split_bundle(data: Data, val_ratio: float, test_ratio: float, seed: int = 2) -> SplitBundle:
    return _make_split_bundle(data, val_ratio, test_ratio, seed)


def _limit_split(edge_label_index: torch.Tensor, edge_label: torch.Tensor, keep_ratio: float):
    if keep_ratio >= 1.0:
        return edge_label_index, edge_label
    num_edges = edge_label.numel()
    num_keep = max(1, int(math.ceil(num_edges * keep_ratio)))
    perm = torch.randperm(num_edges)[:num_keep]
    return edge_label_index[:, perm], edge_label[perm]


def _drnl_node_labeling(edge_index: torch.Tensor, src: int, dst: int, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True

    def bfs(start: int) -> torch.Tensor:
        dist = torch.full((num_nodes,), fill_value=num_nodes, dtype=torch.long)
        dist[start] = 0
        frontier = [start]
        while frontier:
            new_frontier = []
            for node in frontier:
                neighbors = adj[node].nonzero(as_tuple=False).view(-1).tolist()
                for nbr in neighbors:
                    if dist[nbr] > dist[node] + 1:
                        dist[nbr] = dist[node] + 1
                        new_frontier.append(nbr)
            frontier = new_frontier
        return dist

    dist_src = bfs(src)
    dist_dst = bfs(dst)
    total = dist_src + dist_dst
    total_half = torch.div(total, 2, rounding_mode="floor")
    total_mod = total % 2
    z = 1 + torch.min(dist_src, dist_dst)
    z = z + total_half * (total_half + total_mod - 1)
    z[src] = 1
    z[dst] = 1
    z[(dist_src >= num_nodes) | (dist_dst >= num_nodes)] = 0
    return z.to(torch.long)


def _degree_labels(edge_index: torch.Tensor, num_nodes: int, max_degree: int) -> torch.Tensor:
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    return degree.clamp(max=max_degree)


def _one_hot_feature(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.clamp(min=0, max=num_classes - 1)
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32)


def _extract_subgraph(
    full_data: Data,
    message_edge_index: torch.Tensor,
    src: int,
    dst: int,
    num_hops: int,
):
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


def _build_labeled_subgraph(
    sub_edge_index: torch.Tensor,
    src: int,
    dst: int,
    src_local: int,
    dst_local: int,
    num_nodes: int,
    node_label: str,
) -> SubgraphData:
    if node_label == "drnl":
        z = _drnl_node_labeling(sub_edge_index, src_local, dst_local, num_nodes).clamp(max=50)
        x = _one_hot_feature(z, 51)
    elif node_label == "de":
        z = _degree_labels(sub_edge_index, num_nodes, max_degree=32)
        x = _one_hot_feature(z, 33)
    else:
        raise ValueError(f"Unsupported node label mode: {node_label}")

    return SubgraphData(
        x=x,
        edge_index=sub_edge_index,
        z=z,
        target_nodes=torch.tensor([src, dst], dtype=torch.long),
        num_nodes=num_nodes,
    )


class ASLAM_Dataset(InMemoryDataset):
    def __init__(
        self,
        dataset,
        emb_shap: torch.Tensor,
        args,
        node_label: str = "drnl",
        num_hops: int = 1,
        split: str = "train",
        split_bundle: SplitBundle | None = None,
        subgraph_cache: Dict[Tuple[str, int, int], Tuple[torch.Tensor, int, int, int]] | None = None,
        transform=None,
        pre_transform=None,
    ):
        self.args = args
        self.node_label = node_label
        self.num_hops = num_hops
        self.split = split
        self._base_data = dataset[0]
        self.emb_shap = emb_shap.detach().cpu()
        self._global_degree = self._compute_global_degree(self._base_data)
        self._bundle = split_bundle or _make_split_bundle(
            self._base_data,
            float(args.val_ratio),
            float(args.test_ratio),
            seed=2,
        )
        self._subgraph_cache = subgraph_cache if subgraph_cache is not None else {}
        super().__init__(".", transform, pre_transform)
        data_list = self._build_graphs()
        self.data, self.slices = self.collate(data_list)

    @staticmethod
    def _compute_global_degree(data: Data) -> torch.Tensor:
        degree = torch.zeros(data.num_nodes, dtype=torch.long)
        degree.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), dtype=torch.long))
        return degree

    def _select_split(self):
        if self.split == "train":
            split_data = self._bundle.train_data
            keep_ratio = float(self.args.train_percent)
        elif self.split == "val":
            split_data = self._bundle.val_data
            keep_ratio = float(self.args.val_percent)
        else:
            split_data = self._bundle.test_data
            keep_ratio = float(self.args.test_percent)
        edge_label_index, edge_label = _limit_split(split_data.edge_label_index, split_data.edge_label, keep_ratio)
        return self._bundle.train_data.edge_index, edge_label_index, edge_label

    def _build_graphs(self):
        message_edge_index, edge_label_index, edge_label = self._select_split()
        data_list = []
        for idx in range(edge_label.size(0)):
            src = int(edge_label_index[0, idx])
            dst = int(edge_label_index[1, idx])
            pair_key = tuple(sorted((src, dst)))
            cache_key = (self.split, pair_key[0], pair_key[1])
            cached = self._subgraph_cache.get(cache_key)
            if cached is None:
                subset, sub_edge_index, src_local, dst_local = _extract_subgraph(
                    self._base_data,
                    message_edge_index,
                    src,
                    dst,
                    self.num_hops,
                )
                cached = (sub_edge_index, src_local, dst_local, int(subset.numel()))
                self._subgraph_cache[cache_key] = cached
            sub_edge_index, src_local, dst_local, num_nodes = cached
            sub_data = _build_labeled_subgraph(
                sub_edge_index,
                src,
                dst,
                src_local,
                dst_local,
                num_nodes,
                self.node_label,
            )
            sub_data.y = edge_label[idx].view(1).to(torch.float32)
            pair_degree = self._global_degree[torch.tensor([src, dst], dtype=torch.long)]
            sub_data.pair_degree = pair_degree.to(torch.float32)
            data_list.append(sub_data)
        return data_list
