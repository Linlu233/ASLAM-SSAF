from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_ablation_bars import DEFAULT_DATASET_ORDER, DISPLAY_LABELS, configure_publication_style, resolve_dataset_items, save_figure


NEGATIVE_COLOR = "#5B7DB1"
POSITIVE_COLOR = "#D98F4E"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot publication-style t-SNE grids from saved edge embedding coordinates."
    )
    parser.add_argument("--summary_json", type=Path, default=Path("results") / "tsne" / "tsne_visualization_summary.json")
    parser.add_argument("--output_dir", type=Path, default=Path("Figure"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASET_ORDER),
        help="Comma-separated dataset keys to include.",
    )
    return parser.parse_args()


def load_summary(summary_json: Path) -> dict:
    with summary_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def marker_size(count: int) -> float:
    return float(np.clip(260.0 / np.sqrt(max(count, 1)), 3.2, 9.0))


def plot_grid(summary: dict, output_dir: Path, dpi: int) -> None:
    datasets = list(summary["datasets"].items())
    ncols = 3 if len(datasets) > 3 else len(datasets)
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.8, 3.15 * nrows + 0.4), dpi=240)
    axes = np.atleast_1d(axes).flatten()

    for ax, (dataset_key, item) in zip(axes, datasets):
        coords_path = Path(item["coords_path"])
        payload = np.load(coords_path)
        coords = payload["coords"]
        labels = payload["labels"]
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        ms = marker_size(coords.shape[0])

        ax.scatter(
            coords[neg_mask, 0],
            coords[neg_mask, 1],
            s=ms,
            c=NEGATIVE_COLOR,
            alpha=0.76,
            edgecolors="none",
            label="Negative edge",
        )
        ax.scatter(
            coords[pos_mask, 0],
            coords[pos_mask, 1],
            s=ms,
            c=POSITIVE_COLOR,
            alpha=0.76,
            edgecolors="none",
            label="Positive edge",
        )
        ax.set_title(DISPLAY_LABELS.get(dataset_key, dataset_key), pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.text(
            0.02,
            0.03,
            "AUC={auc:.3f}\nAP={ap:.3f}\nn={n}".format(
                auc=item["best_test"]["AUC"],
                ap=item["best_test"]["AP"],
                n=item["sample_count"],
            ),
            transform=ax.transAxes,
            fontsize=8.8,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B8BCC3", "linewidth": 0.8, "alpha": 0.95},
        )

    for idx in range(len(datasets), len(axes)):
        axes[idx].axis("off")
        if idx == len(datasets):
            axes[idx].text(
                0.5,
                0.68,
                "Legend and Notes",
                ha="center",
                va="center",
                fontsize=12.0,
                weight="semibold",
                transform=axes[idx].transAxes,
            )
            axes[idx].text(
                0.5,
                0.42,
                "Blue: negative edges\nOrange: positive edges\nInset box: best test AUC, AP, and sampled edges",
                ha="center",
                va="center",
                fontsize=9.4,
                transform=axes[idx].transAxes,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#B8BCC3", "linewidth": 0.9},
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=2.0,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    outputs = [
        output_dir / "tsne_grid_publication.png",
        output_dir / "tsne_grid_publication.pdf",
        output_dir / "figure_tsne_grid_publication.tif",
        output_dir / "figure_tsne_grid_publication.pdf",
    ]
    for output_path in outputs:
        save_figure(fig, output_path, dpi)
    plt.close(fig)


def write_summary(summary: dict, output_dir: Path) -> None:
    lines = [
        "# Publication t-SNE Summary",
        "",
        "| Dataset | Best epoch | Test AUC | Test AP | Sampled edges |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset_key, item in summary["datasets"].items():
        lines.append(
            "| {dataset} | {epoch} | {auc:.4f} | {ap:.4f} | {count} |".format(
                dataset=DISPLAY_LABELS.get(dataset_key, dataset_key),
                epoch=item["best_epoch"],
                auc=item["best_test"]["AUC"],
                ap=item["best_test"]["AP"],
                count=item["sample_count"],
            )
        )
    (output_dir / "tsne_publication_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    summary = load_summary(args.summary_json)
    dataset_keys = {item["label"] for item in resolve_dataset_items(args.datasets)}
    summary["datasets"] = {key: value for key, value in summary["datasets"].items() if key in dataset_keys}
    plot_grid(summary, args.output_dir, args.dpi)
    write_summary(summary, args.output_dir)
    print(f"Saved: {args.output_dir / 'tsne_grid_publication.png'}")


if __name__ == "__main__":
    main()
