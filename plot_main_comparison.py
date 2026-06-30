from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_ablation_bars import (
    DATASETS,
    DEFAULT_DATASET_ORDER,
    DISPLAY_LABELS,
    configure_publication_style,
    resolve_dataset_items,
    save_figure,
)


SERIES = [
    {"key": "aslam", "label": "ASLAM", "color": "#4C72B0", "hatch": ""},
    {"key": "aslam_ssmattn", "label": "ASLAM-SSAF", "color": "#55A868", "hatch": "xx"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot journal-style main comparison figures for ASLAM and ASLAM-SSAF."
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("Figure"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--full_model_label", type=str, default="ASLAM-SSAF")
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


def collect_metric_table(results_dir: Path, dataset_items: list[dict[str, str]] | None = None) -> dict[str, dict[str, dict[str, float]]]:
    metric_table: dict[str, dict[str, dict[str, float]]] = {}
    for item in dataset_items or DATASETS:
        payload = load_json(results_dir / item["compare_file"])
        block = payload["datasets"][item["compare_key"]]
        metric_table[item["label"]] = {
            "aslam": {
                "auc_mean": block["aslam"]["test_auc"]["mean"],
                "auc_std": block["aslam"]["test_auc"]["std"],
                "ap_mean": block["aslam"]["test_ap"]["mean"],
                "ap_std": block["aslam"]["test_ap"]["std"],
            },
            "aslam_ssmattn": {
                "auc_mean": block["aslam_ssmattn"]["test_auc"]["mean"],
                "auc_std": block["aslam_ssmattn"]["test_auc"]["std"],
                "ap_mean": block["aslam_ssmattn"]["test_ap"]["mean"],
                "ap_std": block["aslam_ssmattn"]["test_ap"]["std"],
            },
        }
    return metric_table


def plot_panel(metric_table: dict[str, dict[str, dict[str, float]]], output_dir: Path, dpi: int, full_model_label: str) -> None:
    datasets = list(metric_table.keys())
    x = np.arange(len(datasets))
    width = 0.28
    display_labels = [DISPLAY_LABELS.get(item, item) for item in datasets]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=240)
    for panel_idx, metric_name in enumerate(("auc", "ap")):
        ax = axes[panel_idx]
        collected = []
        for series_idx, series in enumerate(SERIES):
            label = full_model_label if series["key"] == "aslam_ssmattn" else series["label"]
            means = np.asarray(
                [metric_table[dataset][series["key"]][f"{metric_name}_mean"] * 100.0 for dataset in datasets],
                dtype=np.float64,
            )
            stds = np.asarray(
                [metric_table[dataset][series["key"]][f"{metric_name}_std"] * 100.0 for dataset in datasets],
                dtype=np.float64,
            )
            collected.append(means)
            ax.bar(
                x + (series_idx - 0.5) * width,
                means,
                width=width,
                color=series["color"],
                edgecolor="#2B2B2B",
                linewidth=0.9,
                hatch=series["hatch"],
                alpha=0.96,
                yerr=stds,
                capsize=3,
                error_kw={"elinewidth": 0.85, "ecolor": "#2B2B2B"},
                label=label,
            )

        lower = max(0.0, np.floor((np.concatenate(collected).min() - 1.0) / 2.0) * 2.0)
        upper = min(100.0, np.ceil((np.concatenate(collected).max() + 0.8) / 2.0) * 2.0)
        ax.set_ylabel(f"Mean Test {metric_name.upper()} (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=12, ha="right")
        ax.set_ylim(lower, upper)
        ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.6, color="#C9CDD3", alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_title("(a) Test AUC" if metric_name == "auc" else "(b) Test AP", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", length=4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=2.0,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    outputs = [
        output_dir / "main_comparison_auc_ap_panel.png",
        output_dir / "main_comparison_auc_ap_panel.pdf",
        output_dir / "figure_main_comparison_auc_ap_panel.tif",
        output_dir / "figure_main_comparison_auc_ap_panel.pdf",
    ]
    for output_path in outputs:
        save_figure(fig, output_path, dpi)
    plt.close(fig)


def write_summary(metric_table: dict[str, dict[str, dict[str, float]]], output_dir: Path, full_model_label: str) -> None:
    lines = [
        "# Main Comparison Summary",
        "",
        f"Improved model label: `{full_model_label}`",
        "",
        "| Dataset | ASLAM AUC | ASLAM AP | ASLAM-SSAF AUC | ASLAM-SSAF AP | Delta AUC | Delta AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, payload in metric_table.items():
        delta_auc = payload["aslam_ssmattn"]["auc_mean"] - payload["aslam"]["auc_mean"]
        delta_ap = payload["aslam_ssmattn"]["ap_mean"] - payload["aslam"]["ap_mean"]
        lines.append(
            "| {dataset} | {a_auc:.4f} | {a_ap:.4f} | {b_auc:.4f} | {b_ap:.4f} | {d_auc:+.4f} | {d_ap:+.4f} |".format(
                dataset=DISPLAY_LABELS.get(dataset, dataset),
                a_auc=payload["aslam"]["auc_mean"],
                a_ap=payload["aslam"]["ap_mean"],
                b_auc=payload["aslam_ssmattn"]["auc_mean"],
                b_ap=payload["aslam_ssmattn"]["ap_mean"],
                d_auc=delta_auc,
                d_ap=delta_ap,
            )
        )
    (output_dir / "main_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    dataset_items = resolve_dataset_items(args.datasets)
    metric_table = collect_metric_table(args.results_dir, dataset_items)
    plot_panel(metric_table, args.output_dir, args.dpi, args.full_model_label)
    write_summary(metric_table, args.output_dir, args.full_model_label)
    print(f"Saved: {args.output_dir / 'main_comparison_auc_ap_panel.png'}")


if __name__ == "__main__":
    main()
