from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_ablation_bars import (
    DATASET_COLORS,
    DATASET_HATCHES,
    DEFAULT_DATASET_ORDER,
    DISPLAY_LABELS,
    configure_publication_style,
    resolve_dataset_items,
    save_figure,
)
from utils.data_utils import load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot publication-style dataset statistics for the five formal ASLAM submission datasets."
    )
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--output_dir", type=Path, default=Path("Figure"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASET_ORDER),
        help="Comma-separated dataset keys to include.",
    )
    return parser.parse_args()


def collect_stats(root: Path, dataset_items: list[dict[str, str]]) -> list[dict[str, float]]:
    stats: list[dict[str, float]] = []
    for item in dataset_items:
        dataset_key = item["label"]
        data = load_data(dataset_key, root=str(root), normalize_features=False)
        stats.append(
            {
                "key": dataset_key,
                "label": DISPLAY_LABELS.get(dataset_key, dataset_key),
                "nodes": int(data.num_nodes),
                "edges": int(data.edge_index.size(1) // 2),
                "features": int(data.num_features),
            }
        )
    return stats


def plot_panel(stats: list[dict[str, float]], output_dir: Path, dpi: int) -> None:
    labels = [item["label"] for item in stats]
    x = np.arange(len(labels))
    metrics = [
        ("nodes", "Number of Nodes (log scale)", "(a) Nodes"),
        ("edges", "Number of Undirected Edges (log scale)", "(b) Edges"),
        ("features", "Feature Dimension (log scale)", "(c) Features"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.8), dpi=240)
    for ax, (metric_key, ylabel, title) in zip(axes, metrics):
        values = np.asarray([item[metric_key] for item in stats], dtype=np.float64)
        for idx, value in enumerate(values):
            ax.bar(
                x[idx],
                value,
                width=0.72,
                color=DATASET_COLORS[idx],
                edgecolor="#2B2B2B",
                linewidth=0.9,
                hatch=DATASET_HATCHES[idx],
                alpha=0.96,
            )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title, pad=10)
        ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.6, color="#C9CDD3", alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", length=4)

    legend_handles = [
        Patch(
            facecolor=DATASET_COLORS[idx],
            edgecolor="#2B2B2B",
            linewidth=0.9,
            hatch=DATASET_HATCHES[idx],
            label=labels[idx],
        )
        for idx in range(len(labels))
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(len(labels), 5),
        frameon=False,
        columnspacing=1.1,
        handlelength=1.6,
        handletextpad=0.5,
        fontsize=9.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    outputs = [
        output_dir / "dataset_statistics_panel.png",
        output_dir / "dataset_statistics_panel.pdf",
        output_dir / "figure_dataset_statistics_panel.tif",
        output_dir / "figure_dataset_statistics_panel.pdf",
    ]
    for output_path in outputs:
        save_figure(fig, output_path, dpi)
    plt.close(fig)


def write_summary(stats: list[dict[str, float]], output_dir: Path) -> None:
    lines = [
        "# Dataset Statistics Summary",
        "",
        "| Dataset | Nodes | Undirected edges | Features |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in stats:
        lines.append(
            f"| {item['label']} | {item['nodes']} | {item['edges']} | {item['features']} |"
        )
    (output_dir / "dataset_statistics_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    dataset_items = resolve_dataset_items(args.datasets)
    stats = collect_stats(args.root, dataset_items)
    plot_panel(stats, args.output_dir, args.dpi)
    write_summary(stats, args.output_dir)
    print(f"Saved: {args.output_dir / 'dataset_statistics_panel.png'}")


if __name__ == "__main__":
    main()
