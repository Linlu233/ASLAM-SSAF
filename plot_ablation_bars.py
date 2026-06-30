from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


ALL_DATASETS = [
    {
        "label": "Citeseer",
        "compare_file": "formal_citeseer_compare_runs10_epochs401_pat20_seed2_h64_b16384.json",
        "plus_file": "formal_citeseer_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json",
        "compare_key": "Citeseer",
        "plus_key": "Citeseer",
    },
    {
        "label": "DBLP",
        "compare_file": "formal_dblp_compare_runs10_epochs401_pat20_seed2_h64_b16384.json",
        "plus_file": "formal_dblp_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json",
        "compare_key": "DBLP",
        "plus_key": "DBLP",
    },
    {
        "label": "PubMed",
        "compare_file": "formal_pubmed_compare_runs10_epochs401_pat20_seed2_h64_b16384.json",
        "plus_file": "formal_pubmed_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json",
        "compare_key": "PubMed",
        "plus_key": "PubMed",
    },
    {
        "label": "amz_Photo",
        "compare_file": "formal_amz_photo_compare_runs10_epochs401_pat20_seed2_h64_b16384.json",
        "plus_file": "formal_amz_photo_aslam_plus_runs10_epochs401_pat20_seed2_h64_b16384.json",
        "compare_key": "amz_Photo",
        "plus_key": "amz_Photo",
    },
    {
        "label": "CoRA",
        "compare_file": "formal_cora_compare_runs10_epochs401_pat20_seed2_h64_b4096.json",
        "plus_file": "formal_cora_aslam_plus_runs10_epochs401_pat20_seed2_h64_b4096.json",
        "compare_key": "CoRA",
        "plus_key": "CoRA",
    },
    {
        "label": "Twitch_EN",
        "compare_file": "formal_twitch_en_compare_runs10_epochs401_pat20_seed2_h64_b8192.json",
        "plus_file": "formal_twitch_en_aslam_plus_runs10_epochs401_pat20_seed2_h64_b8192.json",
        "compare_key": "Twitch_EN",
        "plus_key": "Twitch_EN",
    },
]

SERIES = [
    {
        "key": "aslam",
        "label": "ASLAM",
        "color": "#4C72B0",
        "hatch": "",
    },
    {
        "key": "aslam_plus",
        "label": "ASLAM+",
        "color": "#DD8452",
        "hatch": "//",
    },
    {
        "key": "aslam_ssmattn",
        "label": "ASLAM-SSAF",
        "color": "#55A868",
        "hatch": "xx",
    },
]

DISPLAY_LABELS = {
    "Citeseer": "Citeseer",
    "DBLP": "DBLP",
    "PubMed": "PubMed",
    "amz_Photo": "Amazon Photo",
    "CoRA": "Cora",
    "Twitch_EN": "Twitch-EN",
}

PUBLICATION_DATASETS = [item for item in ALL_DATASETS if item["label"] != "Citeseer"]
DATASETS = PUBLICATION_DATASETS
DEFAULT_DATASET_ORDER = [item["label"] for item in PUBLICATION_DATASETS]
DATASET_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C9B458", "#8C8C8C"]
DATASET_HATCHES = ["", "//", "xx", "\\\\", ".."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-style ablation bar charts from the formal ASLAM experiment JSON files."
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
        default=Path("results") / "ablation_figures",
        help="Directory to export the bar charts.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASET_ORDER),
        help="Comma-separated dataset keys to include, e.g. DBLP,PubMed,amz_Photo,CoRA,Twitch_EN.",
    )
    parser.add_argument(
        "--full_model_label",
        type=str,
        default="ASLAM-SSAF",
        help="Display name of the final model shown in the legends and titles.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster export resolution for publication-ready PNG/TIFF files.",
    )
    return parser.parse_args()


def configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11.2,
            "axes.titlesize": 12.6,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.6,
            "figure.titlesize": 13.8,
            "axes.linewidth": 0.95,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_items(dataset_text: str | None) -> list[dict[str, str]]:
    if not dataset_text:
        return list(DATASETS)
    requested = [item.strip() for item in dataset_text.split(",") if item.strip()]
    if not requested:
        return list(DATASETS)

    dataset_map = {item["label"]: item for item in ALL_DATASETS}
    missing = [name for name in requested if name not in dataset_map]
    if missing:
        raise ValueError(f"Unsupported datasets: {', '.join(missing)}")
    return [dataset_map[name] for name in requested]


def collect_metric_table(results_dir: Path, dataset_items: list[dict[str, str]] | None = None) -> dict[str, dict[str, dict[str, float]]]:
    metric_table: dict[str, dict[str, dict[str, float]]] = {}
    for item in dataset_items or DATASETS:
        compare_payload = load_json(results_dir / item["compare_file"])
        plus_payload = load_json(results_dir / item["plus_file"])

        compare_block = compare_payload["datasets"][item["compare_key"]]
        plus_block = plus_payload["datasets"][item["plus_key"]]["aslam_plus"]

        metric_table[item["label"]] = {
            "aslam": {
                "auc_mean": compare_block["aslam"]["test_auc"]["mean"],
                "auc_std": compare_block["aslam"]["test_auc"]["std"],
                "ap_mean": compare_block["aslam"]["test_ap"]["mean"],
                "ap_std": compare_block["aslam"]["test_ap"]["std"],
            },
            "aslam_plus": {
                "auc_mean": plus_block["test_auc"]["mean"],
                "auc_std": plus_block["test_auc"]["std"],
                "ap_mean": plus_block["test_ap"]["mean"],
                "ap_std": plus_block["test_ap"]["std"],
            },
            "aslam_ssmattn": {
                "auc_mean": compare_block["aslam_ssmattn"]["test_auc"]["mean"],
                "auc_std": compare_block["aslam_ssmattn"]["test_auc"]["std"],
                "ap_mean": compare_block["aslam_ssmattn"]["test_ap"]["mean"],
                "ap_std": compare_block["aslam_ssmattn"]["test_ap"]["std"],
            },
        }
    return metric_table


def build_series_arrays(metric_table: dict[str, dict[str, dict[str, float]]], metric_prefix: str):
    values = []
    errors = []
    for series in SERIES:
        series_values = []
        series_errors = []
        for dataset in metric_table:
            block = metric_table[dataset][series["key"]]
            series_values.append(block[f"{metric_prefix}_mean"] * 100.0)
            series_errors.append(block[f"{metric_prefix}_std"] * 100.0)
        values.append(np.array(series_values))
        errors.append(np.array(series_errors))
    return values, errors


def apply_full_model_label(full_model_label: str) -> list[dict[str, str]]:
    styled = []
    for series in SERIES:
        styled_item = dict(series)
        if styled_item["key"] == "aslam_ssmattn":
            styled_item["label"] = full_model_label
        styled.append(styled_item)
    return styled


def pretty_dataset_labels(datasets: list[str]) -> list[str]:
    return [DISPLAY_LABELS.get(dataset, dataset) for dataset in datasets]


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    common_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
    if suffix in {".png", ".tif", ".tiff"}:
        save_kwargs = dict(common_kwargs)
        save_kwargs["dpi"] = dpi
        if suffix in {".tif", ".tiff"}:
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(output_path, **save_kwargs)
    else:
        fig.savefig(output_path, **common_kwargs)


def metric_axis_limits(values: list[np.ndarray]) -> tuple[float, float]:
    all_values = np.concatenate(values)
    lower = max(0.0, np.floor((all_values.min() - 1.0) / 2.0) * 2.0)
    upper = min(100.0, np.ceil((all_values.max() + 0.8) / 2.0) * 2.0)
    return lower, upper


def style_axis(ax: plt.Axes, ylabel: str, xticks: np.ndarray, xticklabels: list[str], lower: float, upper: float) -> None:
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=12, ha="right")
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.6, color="#C9CDD3", alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=4)
    ax.tick_params(axis="both", which="minor", length=2.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_single_metric(
    datasets: list[str],
    values: list[np.ndarray],
    errors: list[np.ndarray],
    series_meta: list[dict[str, str]],
    metric_name: str,
    output_paths: list[Path],
    full_model_label: str,
    dpi: int,
) -> None:
    x = np.arange(len(datasets))
    width = 0.22
    display_datasets = pretty_dataset_labels(datasets)

    fig, ax = plt.subplots(figsize=(8.8, 4.7), dpi=220)
    for idx, series in enumerate(series_meta):
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            values[idx],
            width=width,
            label=series["label"],
            color=series["color"],
            edgecolor="#2B2B2B",
            linewidth=0.9,
            hatch=series["hatch"],
            alpha=0.96,
            yerr=errors[idx],
            capsize=3,
            error_kw={"elinewidth": 0.85, "ecolor": "#2B2B2B"},
        )

    lower, upper = metric_axis_limits(values)
    style_axis(ax, f"Mean Test {metric_name} (%)", x, display_datasets, lower, upper)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=2.0, columnspacing=1.8)
    ax.set_title(f"Ablation study on five datasets: Test {metric_name}", pad=18)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for output_path in output_paths:
        save_figure(fig, output_path, dpi)
    plt.close(fig)


def plot_combined_panel(
    datasets: list[str],
    auc_values: list[np.ndarray],
    auc_errors: list[np.ndarray],
    ap_values: list[np.ndarray],
    ap_errors: list[np.ndarray],
    series_meta: list[dict[str, str]],
    output_paths: list[Path],
    full_model_label: str,
    dpi: int,
) -> None:
    x = np.arange(len(datasets))
    width = 0.22
    display_datasets = pretty_dataset_labels(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=220, sharex=False)
    metric_payloads = [
        ("AUC", "(a) Test AUC", auc_values, auc_errors, axes[0]),
        ("AP", "(b) Test AP", ap_values, ap_errors, axes[1]),
    ]

    for metric_name, panel_title, values, errors, ax in metric_payloads:
        for idx, series in enumerate(series_meta):
            offset = (idx - 1) * width
            ax.bar(
                x + offset,
                values[idx],
                width=width,
                label=series["label"],
                color=series["color"],
                edgecolor="#2B2B2B",
                linewidth=0.9,
                hatch=series["hatch"],
                alpha=0.96,
                yerr=errors[idx],
                capsize=3,
                error_kw={"elinewidth": 0.85, "ecolor": "#2B2B2B"},
            )

        lower, upper = metric_axis_limits(values)

        style_axis(ax, f"Mean Test {metric_name} (%)", x, display_datasets, lower, upper)
        ax.set_title(panel_title, pad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.0,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for output_path in output_paths:
        save_figure(fig, output_path, dpi)
    plt.close(fig)


def write_summary(metric_table: dict[str, dict[str, dict[str, float]]], output_path: Path, full_model_label: str) -> None:
    lines = [
        "# Ablation Bar Chart Summary",
        "",
        f"Full model label: `{full_model_label}`",
        "",
        "| Dataset | ASLAM AUC | ASLAM+ AUC | Full AUC | ASLAM AP | ASLAM+ AP | Full AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, payload in metric_table.items():
        lines.append(
            "| {dataset} | {aslam_auc:.4f} | {plus_auc:.4f} | {full_auc:.4f} | {aslam_ap:.4f} | {plus_ap:.4f} | {full_ap:.4f} |".format(
                dataset=DISPLAY_LABELS.get(dataset, dataset),
                aslam_auc=payload["aslam"]["auc_mean"],
                plus_auc=payload["aslam_plus"]["auc_mean"],
                full_auc=payload["aslam_ssmattn"]["auc_mean"],
                aslam_ap=payload["aslam"]["ap_mean"],
                plus_ap=payload["aslam_plus"]["ap_mean"],
                full_ap=payload["aslam_ssmattn"]["ap_mean"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style()

    dataset_items = resolve_dataset_items(args.datasets)
    metric_table = collect_metric_table(args.results_dir, dataset_items)
    datasets = list(metric_table.keys())
    series_meta = apply_full_model_label(args.full_model_label)

    auc_values, auc_errors = build_series_arrays(metric_table, "auc")
    ap_values, ap_errors = build_series_arrays(metric_table, "ap")

    auc_outputs = [
        args.output_dir / "ablation_auc_bar.png",
        args.output_dir / "ablation_auc_bar.pdf",
        args.output_dir / "figure_ablation_auc_bar.tif",
        args.output_dir / "figure_ablation_auc_bar.pdf",
    ]
    ap_outputs = [
        args.output_dir / "ablation_ap_bar.png",
        args.output_dir / "ablation_ap_bar.pdf",
        args.output_dir / "figure_ablation_ap_bar.tif",
        args.output_dir / "figure_ablation_ap_bar.pdf",
    ]
    combined_outputs = [
        args.output_dir / "ablation_auc_ap_panel.png",
        args.output_dir / "ablation_auc_ap_panel.pdf",
        args.output_dir / "figure_ablation_auc_ap_panel.tif",
        args.output_dir / "figure_ablation_auc_ap_panel.pdf",
    ]
    summary_md = args.output_dir / "ablation_bar_summary.md"

    plot_single_metric(datasets, auc_values, auc_errors, series_meta, "AUC", auc_outputs, args.full_model_label, args.dpi)
    plot_single_metric(datasets, ap_values, ap_errors, series_meta, "AP", ap_outputs, args.full_model_label, args.dpi)
    plot_combined_panel(
        datasets,
        auc_values,
        auc_errors,
        ap_values,
        ap_errors,
        series_meta,
        combined_outputs,
        args.full_model_label,
        args.dpi,
    )
    write_summary(metric_table, summary_md, args.full_model_label)

    print(f"Saved: {auc_outputs[0]}")
    print(f"Saved: {ap_outputs[0]}")
    print(f"Saved: {combined_outputs[0]}")
    print(f"Saved: {combined_outputs[2]}")
    print(f"Saved: {summary_md}")


if __name__ == "__main__":
    main()
