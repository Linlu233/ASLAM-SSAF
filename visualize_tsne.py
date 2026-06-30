from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from train import (
    build_neighbor_sets,
    compute_edge_heuristics,
    evaluate,
    instantiate_model,
    normalize_heuristics,
    train_epoch,
)
from utils.data_utils import create_link_splits, load_data
from utils.train_utils import edge_batch_iterator, resolve_device, save_json, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-style t-SNE visualizations from learned edge representations."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="Citeseer,DBLP,PubMed,amz_Photo,CoRA,Twitch_EN",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="aslam_ssmattn",
        choices=["aslam", "aslam_plus", "aslam_ssmattn"],
        help="Model variant used to produce edge representations.",
    )
    parser.add_argument("--root", type=str, default="datasets", help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=401, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout ratio.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of graph layers.")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Train edge ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation edge ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Test edge ratio.")
    parser.add_argument(
        "--neg_sampling_ratio",
        type=float,
        default=1.0,
        help="Negative sampling ratio for each positive edge.",
    )
    parser.add_argument("--seed", type=int, default=2, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device. Use 'auto', 'cpu', or e.g. 'cuda:0'.",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.15,
        help="Fraction of test edges used in t-SNE rendering.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/tsne",
        help="Directory for generated figures and summaries.",
    )
    return parser


def default_batch_size(dataset_name: str) -> int:
    mapping = {
        "Citeseer": 16384,
        "DBLP": 16384,
        "PubMed": 16384,
        "amz_Photo": 16384,
        "CoRA": 4096,
        "Twitch_EN": 8192,
    }
    return mapping[dataset_name]


def clone_state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def fit_single_run(args, dataset_name: str, device: torch.device) -> Dict:
    seed_everything(args.seed)
    data = load_data(
        dataset_name=dataset_name,
        root=args.root,
        normalize_features=False,
    )
    splits = create_link_splits(
        data=data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        neg_sampling_ratio=args.neg_sampling_ratio,
        seed=args.seed,
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
        model_name=args.model,
        in_channels=data.num_features,
        heuristic_dim=train_heuristics.size(1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_epoch = 0
    best_state = clone_state_dict_to_cpu(model)
    best_val = {"AUC": 0.0, "AP": 0.0}
    best_test = {"AUC": 0.0, "AP": 0.0}
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            x=x,
            edge_index=train_edge_index,
            edge_label_index=train_edge_label_index,
            edge_label=train_edge_label,
            heuristics=train_heuristics,
            batch_size=default_batch_size(dataset_name),
        )
        val_metric = evaluate(
            model=model,
            x=x,
            edge_index=train_edge_index,
            edge_label_index=val_edge_label_index,
            edge_label=val_edge_label,
            heuristics=val_heuristics,
            batch_size=default_batch_size(dataset_name),
        )
        if val_metric["AUC"] > best_val["AUC"]:
            best_val = val_metric
            best_test = evaluate(
                model=model,
                x=x,
                edge_index=train_edge_index,
                edge_label_index=test_edge_label_index,
                edge_label=test_edge_label,
                heuristics=test_heuristics,
                batch_size=default_batch_size(dataset_name),
            )
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)
            patience = 0
        else:
            patience += 1
        print(
            f"[t-SNE][{dataset_name}] epoch={epoch:03d} "
            f"loss={train_loss:.4f} val_auc={val_metric['AUC']:.4f} best_auc={best_val['AUC']:.4f}"
        )
        if patience >= args.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "data": data,
        "splits": splits,
        "model": model,
        "x": x,
        "train_edge_index": train_edge_index,
        "test_heuristics": test_heuristics,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_test": best_test,
    }


@torch.no_grad()
def collect_edge_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    heuristics: torch.Tensor,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    features = []
    labels = []
    for batch_edges, batch_labels, batch_heuristics in edge_batch_iterator(
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        heuristics=heuristics,
        batch_size=batch_size,
        shuffle=False,
    ):
        batch_features = model.encode_edge_features(x, edge_index, batch_edges, batch_heuristics)
        features.append(batch_features.detach().cpu().numpy())
        labels.append(batch_labels.detach().cpu().numpy())
    return {
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
    }


def stratified_sample(features: np.ndarray, labels: np.ndarray, ratio: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected_parts: List[np.ndarray] = []
    for cls in (0, 1):
        cls_ids = np.flatnonzero(labels == cls)
        if cls_ids.size == 0:
            continue
        take = max(1, int(round(cls_ids.size * ratio)))
        take = min(take, cls_ids.size)
        selected_parts.append(rng.choice(cls_ids, size=take, replace=False))
    selected = np.sort(np.concatenate(selected_parts))
    return {
        "features": features[selected],
        "labels": labels[selected],
        "indices": selected,
    }


def run_tsne(features: np.ndarray, seed: int) -> np.ndarray:
    n_samples = features.shape[0]
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    return tsne.fit_transform(features)


def plot_single_dataset(
    coords: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
    model_name: str,
    sample_count: int,
    metric_text: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    mask_neg = labels == 0
    mask_pos = labels == 1
    ax.scatter(coords[mask_neg, 0], coords[mask_neg, 1], s=20, c="#2563eb", alpha=0.75, label="Negative edge")
    ax.scatter(coords[mask_pos, 0], coords[mask_pos, 1], s=20, c="#f59e0b", alpha=0.75, label="Positive edge")
    ax.set_title(f"{dataset_name} | {model_name} edge t-SNE")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.02,
        0.02,
        f"sampled edges={sample_count}\n{metric_text}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, linewidth=0.5),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grid(results: List[Dict], model_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    axes = axes.ravel()
    for ax, item in zip(axes, results):
        coords = item["coords"]
        labels = item["labels"]
        mask_neg = labels == 0
        mask_pos = labels == 1
        ax.scatter(coords[mask_neg, 0], coords[mask_neg, 1], s=10, c="#2563eb", alpha=0.68)
        ax.scatter(coords[mask_pos, 0], coords[mask_pos, 1], s=10, c="#f59e0b", alpha=0.68)
        ax.set_title(item["dataset"])
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.text(
            0.02,
            0.02,
            f"AUC={item['best_test']['AUC']:.4f}\nAP={item['best_test']['AP']:.4f}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, linewidth=0.4),
        )
    for ax in axes[len(results) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#f59e0b", label="Positive edge"),
        plt.Line2D([], [], marker="o", linestyle="", color="#2563eb", label="Negative edge"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{model_name} t-SNE of learned edge representations", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(results: List[Dict], model_name: str, output_path: Path) -> None:
    lines = [
        "# t-SNE Visualization Summary",
        "",
        f"Model: `{model_name}`",
        "",
        "This summary follows the paper-style visualization protocol: the learned edge representations are projected by t-SNE, and positive/negative test edges are rendered with different colors.",
        "",
        "| Dataset | Best epoch | Test AUC | Test AP | Sample ratio | Sampled edges | Figure | Coordinates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in results:
        lines.append(
            "| {dataset} | {best_epoch} | {auc:.4f} | {ap:.4f} | {sample_ratio:.2f} | {sample_count} | `{figure}` | `{coords}` |".format(
                dataset=item["dataset"],
                best_epoch=item["best_epoch"],
                auc=item["best_test"]["AUC"],
                ap=item["best_test"]["AP"],
                sample_ratio=item["sample_ratio"],
                sample_count=item["sample_count"],
                figure=item["figure_path"].as_posix(),
                coords=item["coords_path"].as_posix(),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    all_results = []
    for offset, dataset_name in enumerate(datasets):
        run_result = fit_single_run(args=args, dataset_name=dataset_name, device=device)
        feature_bundle = collect_edge_features(
            model=run_result["model"],
            x=run_result["x"],
            edge_index=run_result["train_edge_index"],
            edge_label_index=run_result["splits"].test_data.edge_label_index.to(device),
            edge_label=run_result["splits"].test_data.edge_label.to(device),
            heuristics=run_result["test_heuristics"],
            batch_size=default_batch_size(dataset_name),
        )
        sampled = stratified_sample(
            features=feature_bundle["features"],
            labels=feature_bundle["labels"],
            ratio=args.sample_ratio,
            seed=args.seed + offset,
        )
        coords = run_tsne(sampled["features"], seed=args.seed + offset)

        figure_path = output_dir / f"{dataset_name}_{args.model}_edge_tsne.png"
        coords_path = output_dir / f"{dataset_name}_{args.model}_edge_tsne_coords.npz"
        np.savez(
            coords_path,
            coords=coords,
            labels=sampled["labels"],
            sampled_indices=sampled["indices"],
        )
        metric_text = "AUC={:.4f}, AP={:.4f}".format(
            run_result["best_test"]["AUC"],
            run_result["best_test"]["AP"],
        )
        plot_single_dataset(
            coords=coords,
            labels=sampled["labels"],
            dataset_name=dataset_name,
            model_name=args.model,
            sample_count=int(sampled["labels"].shape[0]),
            metric_text=metric_text,
            output_path=figure_path,
        )
        all_results.append(
            {
                "dataset": dataset_name,
                "best_epoch": run_result["best_epoch"],
                "best_val": run_result["best_val"],
                "best_test": run_result["best_test"],
                "sample_ratio": args.sample_ratio,
                "sample_count": int(sampled["labels"].shape[0]),
                "figure_path": figure_path,
                "coords_path": coords_path,
                "coords": coords,
                "labels": sampled["labels"],
            }
        )

    grid_path = output_dir / f"{args.model}_edge_tsne_grid.png"
    plot_grid(all_results, model_name=args.model, output_path=grid_path)

    serializable = {
        "model": args.model,
        "device": str(device),
        "sample_ratio": args.sample_ratio,
        "datasets": {
            item["dataset"]: {
                "best_epoch": item["best_epoch"],
                "best_val": item["best_val"],
                "best_test": item["best_test"],
                "sample_count": item["sample_count"],
                "figure_path": str(item["figure_path"]),
                "coords_path": str(item["coords_path"]),
            }
            for item in all_results
        },
        "grid_path": str(grid_path),
    }
    save_json(serializable, output_dir / "tsne_visualization_summary.json")
    write_markdown_summary(all_results, model_name=args.model, output_path=output_dir / "tsne_visualization_summary.md")


if __name__ == "__main__":
    main()
