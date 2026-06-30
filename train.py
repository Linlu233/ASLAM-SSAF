from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from config import parser
from models.aslam import ASLAM, ASLAMPlus, ASLAMSSMAttn
from utils.data_utils import create_link_splits, load_data
from utils.eval_utils import evaluate_auc_ap, summarize_results
from utils.train_utils import (
    edge_batch_iterator,
    format_metric,
    resolve_device,
    save_json,
    seed_everything,
    to_numpy_int_pairs,
)


def build_neighbor_sets(edge_index: torch.Tensor, num_nodes: int) -> List[set[int]]:
    neighbors = [set() for _ in range(num_nodes)]
    src, dst = edge_index.cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:
            neighbors[u].add(v)
    return neighbors


def compute_edge_heuristics(
    neighbor_sets: List[set[int]],
    edge_label_index: torch.Tensor,
) -> torch.Tensor:
    pairs = to_numpy_int_pairs(edge_label_index)
    degrees = np.asarray([len(nbrs) for nbrs in neighbor_sets], dtype=np.float32)
    features = []
    for u, v in pairs:
        neigh_u = neighbor_sets[int(u)]
        neigh_v = neighbor_sets[int(v)]
        common = neigh_u & neigh_v
        union = neigh_u | neigh_v
        cn = float(len(common))
        jaccard = cn / max(len(union), 1)
        aa = 0.0
        ra = 0.0
        for w in common:
            deg_w = max(degrees[w], 2.0)
            aa += 1.0 / np.log(deg_w)
            ra += 1.0 / deg_w
        pref_attach = degrees[u] * degrees[v]
        deg_u = degrees[u]
        deg_v = degrees[v]
        cosine_deg = cn / max(np.sqrt(max(deg_u, 1.0) * max(deg_v, 1.0)), 1.0)
        features.append([cn, jaccard, aa, ra, pref_attach, deg_u, deg_v, cosine_deg])
    return torch.tensor(np.asarray(features, dtype=np.float32))


def normalize_heuristics(
    train_heuristics: torch.Tensor,
    *others: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    mean = train_heuristics.mean(dim=0, keepdim=True)
    std = train_heuristics.std(dim=0, keepdim=True).clamp_min_(1e-6)
    normalized = [((train_heuristics - mean) / std)]
    normalized.extend((tensor - mean) / std for tensor in others)
    return tuple(normalized)


def instantiate_model(args, model_name: str, in_channels: int, heuristic_dim: int) -> nn.Module:
    model_map = {
        "aslam": ASLAM,
        "aslam_plus": ASLAMPlus,
        "aslam_ssmattn": ASLAMSSMAttn,
    }
    model_cls = model_map[model_name]
    return model_cls(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heuristic_dim=heuristic_dim,
        dropout=args.dropout,
    )


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    heuristics: torch.Tensor,
    batch_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_edges = 0
    criterion = nn.BCEWithLogitsLoss()
    for batch_edges, batch_labels, batch_heuristics in edge_batch_iterator(
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        heuristics=heuristics,
        batch_size=batch_size,
        shuffle=True,
    ):
        optimizer.zero_grad()
        logits_attr, logits_struct, logits_fused = model(x, edge_index, batch_edges, batch_heuristics)
        labels = batch_labels.float()
        loss = (
            criterion(logits_attr, labels)
            + criterion(logits_struct, labels)
            + criterion(logits_fused, labels)
        ) / 3.0
        loss.backward()
        optimizer.step()
        batch_size_actual = batch_labels.numel()
        total_loss += loss.item() * batch_size_actual
        total_edges += batch_size_actual
    return total_loss / max(total_edges, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    heuristics: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    logits_all = []
    labels_all = []
    for batch_edges, batch_labels, batch_heuristics in edge_batch_iterator(
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        heuristics=heuristics,
        batch_size=batch_size,
        shuffle=False,
    ):
        _, _, logits = model(x, edge_index, batch_edges, batch_heuristics)
        logits_all.append(logits.cpu())
        labels_all.append(batch_labels.cpu())
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    return evaluate_auc_ap(logits, labels)


def run_single_dataset(args, dataset_name: str, model_name: str, run_seed: int, device: torch.device) -> Dict:
    data = load_data(
        dataset_name=dataset_name,
        root=args.root,
        normalize_features=args.normalize_features,
    )
    splits = create_link_splits(
        data=data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        neg_sampling_ratio=args.neg_sampling_ratio,
        seed=run_seed,
    )

    neighbor_sets = build_neighbor_sets(splits.train_data.edge_index, num_nodes=data.num_nodes)
    train_heuristics = compute_edge_heuristics(neighbor_sets, splits.train_data.edge_label_index)
    val_heuristics = compute_edge_heuristics(neighbor_sets, splits.val_data.edge_label_index)
    test_heuristics = compute_edge_heuristics(neighbor_sets, splits.test_data.edge_label_index)
    train_heuristics, val_heuristics, test_heuristics = normalize_heuristics(
        train_heuristics, val_heuristics, test_heuristics
    )

    x = data.x.to(device)
    train_edge_index = splits.train_data.edge_index.to(device)
    train_edge_label_index = splits.train_data.edge_label_index.to(device)
    train_edge_label = splits.train_data.edge_label.to(device)
    val_edge_label_index = splits.val_data.edge_label_index.to(device)
    val_edge_label = splits.val_data.edge_label.to(device)
    test_edge_label_index = splits.test_data.edge_label_index.to(device)
    test_edge_label = splits.test_data.edge_label.to(device)

    train_heuristics = train_heuristics.to(device)
    val_heuristics = val_heuristics.to(device)
    test_heuristics = test_heuristics.to(device)

    model = instantiate_model(
        args=args,
        model_name=model_name,
        in_channels=data.num_features,
        heuristic_dim=train_heuristics.size(1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = {"AUC": 0.0, "AP": 0.0}
    best_test = {"AUC": 0.0, "AP": 0.0}
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            x=x,
            edge_index=train_edge_index,
            edge_label_index=train_edge_label_index,
            edge_label=train_edge_label,
            heuristics=train_heuristics,
            batch_size=args.batch_size,
        )
        val_metric = evaluate(
            model=model,
            x=x,
            edge_index=train_edge_index,
            edge_label_index=val_edge_label_index,
            edge_label=val_edge_label,
            heuristics=val_heuristics,
            batch_size=args.batch_size,
        )
        history.append({"epoch": epoch, "loss": train_loss, "val": val_metric})

        if val_metric["AUC"] > best_val["AUC"]:
            best_val = val_metric
            best_test = evaluate(
                model=model,
                x=x,
                edge_index=train_edge_index,
                edge_label_index=test_edge_label_index,
                edge_label=test_edge_label,
                heuristics=test_heuristics,
                batch_size=args.batch_size,
            )
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        print(
            f"[{dataset_name}][{model_name}] epoch={epoch:03d} "
            f"loss={train_loss:.4f} val={format_metric(val_metric)} "
            f"best_test={format_metric(best_test)}"
        )
        if patience >= args.patience:
            break

    return {
        "dataset": dataset_name,
        "model": model_name,
        "best_epoch": best_epoch,
        "val": best_val,
        "test": best_test,
        "history": history,
        "graph": {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1) // 2),
            "num_features": int(data.num_features),
        },
    }


def compare_models(args, dataset_name: str, device: torch.device) -> Dict:
    seeds = [args.seed + run_id for run_id in range(args.runs)]
    compare_groups = {
        "compare": ["aslam", "aslam_ssmattn"],
        "compare_plus": ["aslam", "aslam_plus"],
        "compare_all": ["aslam", "aslam_plus", "aslam_ssmattn"],
    }
    model_names = compare_groups.get(args.model, [args.model])
    per_model = {}
    for model_name in model_names:
        runs = []
        for run_seed in seeds:
            print(f"Running dataset={dataset_name}, model={model_name}, seed={run_seed}")
            runs.append(run_single_dataset(args, dataset_name, model_name, run_seed, device))
        per_model[model_name] = {
            "runs": runs,
            "test_auc": summarize_results([item["test"]["AUC"] for item in runs]),
            "test_ap": summarize_results([item["test"]["AP"] for item in runs]),
            "best_epoch": summarize_results([item["best_epoch"] for item in runs]),
        }
    if "aslam" in per_model:
        for improved_model in ("aslam_plus", "aslam_ssmattn"):
            if improved_model in per_model:
                per_model[f"delta_{improved_model}"] = {
                    "auc_mean_gain": per_model[improved_model]["test_auc"]["mean"]
                    - per_model["aslam"]["test_auc"]["mean"],
                    "ap_mean_gain": per_model[improved_model]["test_ap"]["mean"]
                    - per_model["aslam"]["test_ap"]["mean"],
                }
    return per_model


def main():
    args = parser.parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    results = {"args": vars(args), "device": str(device), "datasets": {}}

    start_time = time.time()
    for dataset_name in datasets:
        print(f"\n=== Dataset: {dataset_name} ===")
        results["datasets"][dataset_name] = compare_models(args, dataset_name, device)

    results["elapsed_seconds"] = time.time() - start_time
    output_path = Path(args.results_dir) / args.summary_name
    save_json(results, output_path)
    print(f"\nSaved summary to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
