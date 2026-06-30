from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import remove_self_loops, to_undirected


SUPPORTED_DATASETS = {
    "citeseer",
    "dblp",
    "pubmed",
    "amz_photo",
    "amazon_photo",
    "photo",
    "cora",
    "co_ra",
    "twitch_en",
    "twitch",
}


@dataclass
class SplitBundle:
    train_data: Data
    val_data: Data
    test_data: Data


def _row_normalize(x: torch.Tensor) -> torch.Tensor:
    denom = x.sum(dim=-1, keepdim=True).clamp_min_(1e-12)
    return x / denom


def _build_undirected_edge_index(edges: Iterable[Tuple[int, int]], num_nodes: int) -> torch.Tensor:
    edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        edge_set.add((u, v))
    if not edge_set:
        raise ValueError("No valid edges were parsed from the dataset.")
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index


def _label_encode(labels: List[str]) -> torch.Tensor:
    vocab = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return torch.tensor([vocab[label] for label in labels], dtype=torch.long)


def load_citeseer(root: Path) -> Data:
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
            paper_id = parts[0]
            id_to_idx[paper_id] = idx
            features.append([float(v) for v in parts[1:-1]])
            labels.append(parts[-1])

    edges: List[Tuple[int, int]] = []
    with cites_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            src, dst = parts
            if src in id_to_idx and dst in id_to_idx:
                edges.append((id_to_idx[src], id_to_idx[dst]))

    x = torch.tensor(np.asarray(features, dtype=np.float32))
    y = _label_encode(labels)
    edge_index = _build_undirected_edge_index(edges, x.size(0))
    return Data(x=x, edge_index=edge_index, y=y)


def load_dblp(root: Path) -> Data:
    dblp_root = root / "DBLP"
    x = torch.tensor(np.load(dblp_root / "dblp_feat.npy"), dtype=torch.float32)
    y = torch.tensor(np.load(dblp_root / "dblp_label.npy"), dtype=torch.long)
    adj = np.load(dblp_root / "dblp_adj.npy")
    row, col = np.nonzero(np.triu(adj, k=1))
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))
    return Data(x=x, edge_index=edge_index, y=y)


def load_pubmed(root: Path) -> Data:
    node_path = root / "PubMed" / "Pubmed-Diabetes.NODE.paper.tab"
    edge_path = root / "PubMed" / "Pubmed-Diabetes.DIRECTED.cites.tab"

    lines = node_path.read_text(encoding="utf-8").splitlines()
    header_tokens = lines[1].split("\t")
    feature_names = [token.split(":")[1] for token in header_tokens[1:-1]]
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    node_lines = lines[2:]

    paper_ids: List[str] = []
    labels: List[int] = []
    features: List[List[float]] = []

    for line in node_lines:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        paper_id = parts[0]
        label = int(parts[1].split("=")[-1]) - 1
        feat_values = np.zeros(len(feature_names), dtype=np.float32)
        for token in parts[2:-1]:
            if "=" not in token:
                continue
            feature_name, feature_value = token.split("=", 1)
            if feature_name in feature_to_idx:
                feat_values[feature_to_idx[feature_name]] = float(feature_value)
        paper_ids.append(paper_id)
        labels.append(label)
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
    return Data(x=x, edge_index=edge_index, y=y)


def load_amazon_photo(root: Path) -> Data:
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
    x = torch.tensor(attrs.toarray(), dtype=torch.float32)
    y = torch.tensor(payload["labels"], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def load_twitch_en(root: Path) -> Data:
    json_path = root / "Twitch_EN" / "musae_Twitch_EN.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    x = torch.tensor(np.asarray(payload["x"], dtype=np.float32))
    y = torch.tensor(np.asarray(payload["y"], dtype=np.int64))
    edge_index = torch.tensor(np.asarray(payload["edge_index"], dtype=np.int64))
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))
    edge_index, _ = remove_self_loops(edge_index)
    return Data(x=x, edge_index=edge_index, y=y)


def load_cora(root: Path) -> Data:
    cora_root = root / "CoRA" / "CoRA_Raw"
    papers_path = cora_root / "papers_dataset.txt"
    citations_path = cora_root / "citations.txt"
    topics_path = cora_root / "topics.txt"
    words_path = cora_root / "words_dictionary.txt"

    feature_to_idx: Dict[str, int] = {}
    with words_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                feature_to_idx[parts[1]] = idx

    paper_ids: List[str] = []
    id_to_idx: Dict[str, int] = {}
    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []
    with papers_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";", maxsplit=2)
            if len(parts) < 2:
                continue
            paper_id = parts[0]
            node_idx = len(paper_ids)
            paper_ids.append(paper_id)
            id_to_idx[paper_id] = node_idx
            if len(parts) != 3 or not parts[2]:
                continue
            for token_spec in parts[2].split(","):
                if ":" not in token_spec:
                    continue
                token_name, token_value = token_spec.split(":", maxsplit=1)
                feat_idx = feature_to_idx.get(token_name)
                if feat_idx is None:
                    continue
                row_indices.append(node_idx)
                col_indices.append(feat_idx)
                values.append(float(token_value))

    x = sp.csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, col_indices)),
        shape=(len(paper_ids), len(feature_to_idx)),
        dtype=np.float32,
    ).toarray()

    topic_map: Dict[str, List[str]] = {}
    with topics_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or "(" not in text or ")" not in text:
                continue
            topic = text[: text.index("(")]
            paper_id = text[text.index("(") + 1 : text.index(")")]
            topic_map.setdefault(paper_id, []).append(topic)

    primary_labels = []
    for paper_id in paper_ids:
        labels = sorted(set(topic_map.get(paper_id, ["__unlabeled__"])))
        primary_labels.append(labels[0])
    y = _label_encode(primary_labels)

    edges: List[Tuple[int, int]] = []
    with citations_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text.startswith("Cite(") or ")" not in text:
                continue
            pair = text[text.index("(") + 1 : text.index(")")]
            parts = pair.split(",", maxsplit=1)
            if len(parts) != 2:
                continue
            src, dst = parts
            if src in id_to_idx and dst in id_to_idx:
                edges.append((id_to_idx[src], id_to_idx[dst]))

    edge_index = _build_undirected_edge_index(edges, num_nodes=len(paper_ids))
    return Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index, y=y)


def load_data(dataset_name: str, root: str = "datasets", normalize_features: bool = False) -> Data:
    root_path = Path(root)
    dataset_key = dataset_name.lower()
    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {sorted(SUPPORTED_DATASETS)}")

    if dataset_key == "citeseer":
        data = load_citeseer(root_path)
    elif dataset_key == "dblp":
        data = load_dblp(root_path)
    elif dataset_key == "pubmed":
        data = load_pubmed(root_path)
    elif dataset_key in {"cora", "co_ra"}:
        data = load_cora(root_path)
    elif dataset_key in {"twitch_en", "twitch"}:
        data = load_twitch_en(root_path)
    else:
        data = load_amazon_photo(root_path)

    if normalize_features:
        data.x = _row_normalize(data.x)
    return data


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
