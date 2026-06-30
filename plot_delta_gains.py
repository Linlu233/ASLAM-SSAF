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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot absolute gain charts for ASLAM-SSAF over ASLAM."
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


def collect_gains(results_dir: Path, dataset_items: list[dict[str, str]] | None = None) -> list[dict[str, float]]:
    gains: list[dict[str, float]] = []
    for item in dataset_items or DATASETS:
        payload = load_json(results_dir / item["compare_file"])
        block = payload["datasets"][item["compare_key"]]
        gains.append(
            {
                "label": DISPLAY_LABELS.get(item["label"], item["label"]),
                "auc_gain": (block["aslam_ssmattn"]["test_auc"]["mean"] - block["aslam"]["test_auc"]["mean"]) * 100.0,
                "ap_gain": (block["aslam_ssmattn"]["test_ap"]["mean"] - block["aslam"]["test_ap"]["mean"]) * 100.0,
            }
        )
    return gains


def plot_panel(gains: list[dict[str, float]], output_dir: Path, dpi: int, full_model_label: str) -> None:
    labels = [item["label"] for item in gains]
    y = np.arange(len(labels))
    auc_values = np.asarray([item["auc_gain"] for item in gains], dtype=np.float64)
    ap_values = np.asarray([item["ap_gain"] for item in gains], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=240, sharey=True)
    payloads = [
        (axes[0], auc_values, "#6C8EBF", "", "(a) AUC gain", "Absolute AUC Gain (percentage points)"),
        (axes[1], ap_values, "#C97F4F", "//", "(b) AP gain", "Absolute AP Gain (percentage points)"),
    ]

    for ax, values, color, hatch, title, xlabel in payloads:
        ax.barh(
            y,
            values,
            height=0.64,
            color=color,
            edgecolor="#2B2B2B",
            linewidth=0.9,
            hatch=hatch,
            alpha=0.96,
        )
        ax.axvline(0.0, color="#2B2B2B", linewidth=0.9)
        for idx, value in enumerate(values):
            ax.text(value + 0.015, idx, f"{value:+.2f}", va="center", ha="left", fontsize=8.8)
        ax.set_title(title, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.grid(axis="x", linestyle=(0, (3, 2)), linewidth=0.6, color="#C9CDD3", alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", length=4)

    max_value = float(max(auc_values.max(), ap_values.max()))
    upper = max(0.25, np.ceil((max_value + 0.05) / 0.05) * 0.05)
    for ax in axes:
        ax.set_xlim(0.0, upper)

    axes[0].invert_yaxis()
    fig.tight_layout()

    outputs = [
        output_dir / "gain_delta_panel.png",
        output_dir / "gain_delta_panel.pdf",
        output_dir / "figure_gain_delta_panel.tif",
        output_dir / "figure_gain_delta_panel.pdf",
    ]
    for output_path in outputs:
        save_figure(fig, output_path, dpi)
    plt.close(fig)


def write_summary(gains: list[dict[str, float]], output_dir: Path, full_model_label: str) -> None:
    lines = [
        "# Gain Summary",
        "",
        f"Improved model label: `{full_model_label}`",
        "",
        "| Dataset | AUC gain (pp) | AP gain (pp) |",
        "| --- | ---: | ---: |",
    ]
    for item in gains:
        lines.append(
            f"| {item['label']} | {item['auc_gain']:+.2f} | {item['ap_gain']:+.2f} |"
        )
    (output_dir / "gain_delta_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    dataset_items = resolve_dataset_items(args.datasets)
    gains = collect_gains(args.results_dir, dataset_items)
    plot_panel(gains, args.output_dir, args.dpi, args.full_model_label)
    write_summary(gains, args.output_dir, args.full_model_label)
    print(f"Saved: {args.output_dir / 'gain_delta_panel.png'}")


if __name__ == "__main__":
    main()
