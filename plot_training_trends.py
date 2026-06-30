from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from plot_ablation_bars import (
    DATASETS,
    DEFAULT_DATASET_ORDER,
    apply_full_model_label,
    configure_publication_style,
    resolve_dataset_items,
    save_figure,
)


LINE_META = {
    "aslam": {
        "label": "ASLAM",
        "color": "#4C72B0",
        "linestyle": "-",
        "marker": "o",
    },
    "aslam_plus": {
        "label": "ASLAM+",
        "color": "#DD8452",
        "linestyle": "--",
        "marker": "s",
    },
    "aslam_ssmattn": {
        "label": "ASLAM-SSAF",
        "color": "#55A868",
        "linestyle": "-",
        "marker": "D",
    },
}

DISPLAY_LABELS = {
    "Citeseer": "Citeseer",
    "DBLP": "DBLP",
    "PubMed": "PubMed",
    "amz_Photo": "Amazon Photo",
    "CoRA": "Cora",
    "Twitch_EN": "Twitch-EN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot journal-style training trend line charts from formal ASLAM experiment histories."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the formal comparison and aslam_plus JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results") / "trend_figures",
        help="Directory to export the training trend figures.",
    )
    parser.add_argument(
        "--full_model_label",
        type=str,
        default="ASLAM-SSAF",
        help="Display name of the final model.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster export resolution.",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=0,
        help="Optional maximum epoch to display. Use 0 to keep all available epochs.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASET_ORDER),
        help="Comma-separated dataset keys to include.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_run_groups(results_dir: Path, dataset_items: list[dict[str, str]] | None = None) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for item in dataset_items or DATASETS:
        compare_payload = load_json(results_dir / item["compare_file"])
        plus_payload = load_json(results_dir / item["plus_file"])
        compare_block = compare_payload["datasets"][item["compare_key"]]
        plus_block = plus_payload["datasets"][item["plus_key"]]["aslam_plus"]

        grouped[item["label"]] = {
            "aslam": compare_block["aslam"]["runs"],
            "aslam_plus": plus_block["runs"],
            "aslam_ssmattn": compare_block["aslam_ssmattn"]["runs"],
        }
    return grouped


def aggregate_history(runs: list[dict], metric: str, max_epoch: int = 0) -> dict[str, np.ndarray]:
    history_maps = []
    epoch_max = 0
    for run in runs:
        per_epoch = {}
        for item in run["history"]:
            epoch = int(item["epoch"])
            if metric == "loss":
                value = float(item["loss"])
            elif metric == "val_auc":
                value = float(item["val"]["AUC"]) * 100.0
            elif metric == "val_ap":
                value = float(item["val"]["AP"]) * 100.0
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            per_epoch[epoch] = value
            epoch_max = max(epoch_max, epoch)
        history_maps.append(per_epoch)

    if max_epoch > 0:
        epoch_max = min(epoch_max, max_epoch)

    epochs = []
    means = []
    stds = []
    counts = []
    for epoch in range(1, epoch_max + 1):
        values = [history_map[epoch] for history_map in history_maps if epoch in history_map]
        if not values:
            continue
        epochs.append(epoch)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
        counts.append(len(values))

    return {
        "epoch": np.asarray(epochs, dtype=np.int32),
        "mean": np.asarray(means, dtype=np.float64),
        "std": np.asarray(stds, dtype=np.float64),
        "count": np.asarray(counts, dtype=np.int32),
    }


def collect_trend_table(results_dir: Path, max_epoch: int, dataset_items: list[dict[str, str]] | None = None) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    grouped = get_run_groups(results_dir, dataset_items)
    trend_table: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for dataset, model_runs in grouped.items():
        trend_table[dataset] = {}
        for model_key, runs in model_runs.items():
            trend_table[dataset][model_key] = {
                "loss": aggregate_history(runs, "loss", max_epoch=max_epoch),
                "val_auc": aggregate_history(runs, "val_auc", max_epoch=max_epoch),
                "val_ap": aggregate_history(runs, "val_ap", max_epoch=max_epoch),
            }
    return trend_table


def y_limits(series_blocks: list[dict[str, np.ndarray]], is_percentage: bool) -> tuple[float, float]:
    mins = []
    maxs = []
    for block in series_blocks:
        mins.append(float(np.min(block["mean"] - block["std"])))
        maxs.append(float(np.max(block["mean"] + block["std"])))
    lower = min(mins)
    upper = max(maxs)
    if is_percentage:
        lower = max(0.0, np.floor((lower - 0.5) / 2.0) * 2.0)
        upper = min(100.0, np.ceil((upper + 0.5) / 2.0) * 2.0)
    else:
        span = max(0.03, upper - lower)
        lower = max(0.0, lower - span * 0.08)
        upper = upper + span * 0.08
    return lower, upper


def build_legend_meta(full_model_label: str) -> dict[str, dict[str, str]]:
    styled = {}
    for model_key, meta in LINE_META.items():
        item = dict(meta)
        if model_key == "aslam_ssmattn":
            item["label"] = full_model_label
        styled[model_key] = item
    return styled


def pretty_dataset_label(dataset: str) -> str:
    return DISPLAY_LABELS.get(dataset, dataset)


def plot_metric_grid(
    trend_table: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    metric_key: str,
    panel_title: str,
    ylabel: str,
    output_paths: list[Path],
    full_model_label: str,
    dpi: int,
) -> None:
    legend_meta = build_legend_meta(full_model_label)
    datasets = list(trend_table.keys())
    ncols = 3 if len(datasets) > 3 else len(datasets)
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.8, 3.25 * nrows + 0.2), dpi=240, sharex=False)
    axes = np.atleast_1d(axes).flatten()
    is_percentage = metric_key != "loss"

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        blocks_for_limit = []
        for model_key in ("aslam", "aslam_plus", "aslam_ssmattn"):
            block = trend_table[dataset][model_key][metric_key]
            blocks_for_limit.append(block)
        lower, upper = y_limits(blocks_for_limit, is_percentage=is_percentage)

        for model_key in ("aslam", "aslam_plus", "aslam_ssmattn"):
            block = trend_table[dataset][model_key][metric_key]
            meta = legend_meta[model_key]
            epoch = block["epoch"]
            mean = block["mean"]
            std = block["std"]
            if epoch.size == 0:
                continue
            markevery = max(1, len(epoch) // 8)
            ax.plot(
                epoch,
                mean,
                color=meta["color"],
                linestyle=meta["linestyle"],
                linewidth=1.7,
                marker=meta["marker"],
                markersize=3.6,
                markevery=markevery,
                label=meta["label"],
            )
            ax.fill_between(epoch, mean - std, mean + std, color=meta["color"], alpha=0.12, linewidth=0.0)

        ax.set_title(pretty_dataset_label(dataset), pad=8)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.6, color="#C9CDD3", alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", length=4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        if is_percentage:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    for idx in range(len(datasets), len(axes)):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        columnspacing=1.8,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for output_path in output_paths:
        save_figure(fig, output_path, dpi)
    plt.close(fig)


def plot_combined_training_panel(
    trend_table: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    output_paths: list[Path],
    full_model_label: str,
    dpi: int,
) -> None:
    legend_meta = build_legend_meta(full_model_label)
    datasets = list(trend_table.keys())
    metric_rows = [
        ("val_auc", "(a)", "Validation AUC (%)", True),
        ("val_ap", "(b)", "Validation AP (%)", True),
        ("loss", "(c)", "Training Loss", False),
    ]

    fig, axes = plt.subplots(
        len(metric_rows),
        len(datasets),
        figsize=(18.2, 8.8),
        dpi=240,
        sharex=False,
        squeeze=False,
    )

    for row_idx, (metric_key, row_label, ylabel, is_percentage) in enumerate(metric_rows):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            blocks_for_limit = [
                trend_table[dataset][model_key][metric_key]
                for model_key in ("aslam", "aslam_plus", "aslam_ssmattn")
            ]
            lower, upper = y_limits(blocks_for_limit, is_percentage=is_percentage)

            for model_key in ("aslam", "aslam_plus", "aslam_ssmattn"):
                block = trend_table[dataset][model_key][metric_key]
                meta = legend_meta[model_key]
                epoch = block["epoch"]
                mean = block["mean"]
                std = block["std"]
                if epoch.size == 0:
                    continue
                markevery = max(1, len(epoch) // 8)
                ax.plot(
                    epoch,
                    mean,
                    color=meta["color"],
                    linestyle=meta["linestyle"],
                    linewidth=1.5,
                    marker=meta["marker"],
                    markersize=3.0,
                    markevery=markevery,
                    label=meta["label"],
                )
                ax.fill_between(epoch, mean - std, mean + std, color=meta["color"], alpha=0.10, linewidth=0.0)

            ax.set_ylim(lower, upper)
            ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.55, color="#C9CDD3", alpha=0.9)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", length=3.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            if row_idx == 0:
                ax.set_title(pretty_dataset_label(dataset), pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n{ylabel}")
            else:
                ax.set_ylabel("")
            if row_idx == len(metric_rows) - 1:
                ax.set_xlabel("Epoch")
            else:
                ax.set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.8,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for output_path in output_paths:
        save_figure(fig, output_path, dpi)
    plt.close(fig)


def write_summary(trend_table: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]], output_path: Path, full_model_label: str) -> None:
    legend_meta = build_legend_meta(full_model_label)
    lines = [
        "# Training Trend Summary",
        "",
        f"Full model label: `{full_model_label}`",
        "",
        "| Dataset | Model | Last epoch | Last val AUC | Last val AP | Last train loss |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, models in trend_table.items():
        for model_key in ("aslam", "aslam_plus", "aslam_ssmattn"):
            auc_block = models[model_key]["val_auc"]
            ap_block = models[model_key]["val_ap"]
            loss_block = models[model_key]["loss"]
            lines.append(
                "| {dataset} | {model} | {last_epoch} | {auc:.2f} | {ap:.2f} | {loss:.4f} |".format(
                    dataset=pretty_dataset_label(dataset),
                    model=legend_meta[model_key]["label"],
                    last_epoch=int(auc_block["epoch"][-1]),
                    auc=float(auc_block["mean"][-1]),
                    ap=float(ap_block["mean"][-1]),
                    loss=float(loss_block["mean"][-1]),
                )
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    dataset_items = resolve_dataset_items(args.datasets)
    trend_table = collect_trend_table(args.results_dir, args.max_epoch, dataset_items)

    auc_outputs = [
        args.output_dir / "training_val_auc_trends.png",
        args.output_dir / "training_val_auc_trends.pdf",
        args.output_dir / "figure_training_val_auc_trends.tif",
        args.output_dir / "figure_training_val_auc_trends.pdf",
    ]
    ap_outputs = [
        args.output_dir / "training_val_ap_trends.png",
        args.output_dir / "training_val_ap_trends.pdf",
        args.output_dir / "figure_training_val_ap_trends.tif",
        args.output_dir / "figure_training_val_ap_trends.pdf",
    ]
    loss_outputs = [
        args.output_dir / "training_loss_trends.png",
        args.output_dir / "training_loss_trends.pdf",
        args.output_dir / "figure_training_loss_trends.tif",
        args.output_dir / "figure_training_loss_trends.pdf",
    ]
    combined_outputs = [
        args.output_dir / "training_trends_combined.png",
        args.output_dir / "training_trends_combined.pdf",
        args.output_dir / "figure_training_trends_combined.tif",
        args.output_dir / "figure_training_trends_combined.pdf",
    ]
    summary_md = args.output_dir / "training_trend_summary.md"

    plot_metric_grid(
        trend_table,
        metric_key="val_auc",
        panel_title=f"Validation AUC trajectories of {args.full_model_label} and its variants on five datasets",
        ylabel="Validation AUC (%)",
        output_paths=auc_outputs,
        full_model_label=args.full_model_label,
        dpi=args.dpi,
    )
    plot_metric_grid(
        trend_table,
        metric_key="val_ap",
        panel_title=f"Validation AP trajectories of {args.full_model_label} and its variants on five datasets",
        ylabel="Validation AP (%)",
        output_paths=ap_outputs,
        full_model_label=args.full_model_label,
        dpi=args.dpi,
    )
    plot_metric_grid(
        trend_table,
        metric_key="loss",
        panel_title=f"Training loss trajectories of {args.full_model_label} and its variants on five datasets",
        ylabel="Training Loss",
        output_paths=loss_outputs,
        full_model_label=args.full_model_label,
        dpi=args.dpi,
    )
    plot_combined_training_panel(
        trend_table,
        output_paths=combined_outputs,
        full_model_label=args.full_model_label,
        dpi=args.dpi,
    )
    write_summary(trend_table, summary_md, args.full_model_label)

    print(f"Saved: {auc_outputs[0]}")
    print(f"Saved: {ap_outputs[0]}")
    print(f"Saved: {loss_outputs[0]}")
    print(f"Saved: {combined_outputs[0]}")
    print(f"Saved: {auc_outputs[2]}")
    print(f"Saved: {summary_md}")


if __name__ == "__main__":
    main()
