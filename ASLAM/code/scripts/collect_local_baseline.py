import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.data_utils import load_data


PAPER_STATS = {
    "citeseer": {"nodes": 4230, "edges": 10674, "features": 602},
    "pubmed": {"nodes": 19717, "edges": 88648, "features": 500},
    "dblp": {"nodes": 17716, "edges": 105734, "features": 1639},
    "photo": {"nodes": 7650, "edges": 23862, "features": 745},
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect frozen local ASLAM baseline logs and dataset statistics."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["citeseer", "pubmed", "dblp", "photo"],
        help="Datasets to summarize.",
    )
    parser.add_argument(
        "--root",
        default=str((PROJECT_ROOT / "datasets").resolve()),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--results-dir",
        default=str((PROJECT_ROOT / "ASLAM" / "results" / "train=0.9").resolve()),
        help="Directory containing train.py output logs.",
    )
    parser.add_argument(
        "--json-out",
        default=str((PROJECT_ROOT / "ASLAM" / "results" / "local_formal_baseline_summary.json").resolve()),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--md-out",
        default=str((PROJECT_ROOT / "ASLAM" / "results" / "local_formal_baseline_summary.md").resolve()),
        help="Output Markdown summary path.",
    )
    return parser


def parse_latest_metrics(results_dir: Path, dataset: str):
    prefix = dataset.capitalize() if dataset != "dblp" else "Dblp"
    candidates = sorted(results_dir.glob(f"{prefix}_*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if "_ssmattn_" in path.name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        auc = re.search(r"The average AUC for test data is ([0-9.]+)", text)
        ap = re.search(r"The average AP for test data is ([0-9.]+)", text)
        auc_std = re.search(r"The std of AUC for test data is ([0-9.]+)", text)
        ap_std = re.search(r"The std of AP for test data is ([0-9.]+)", text)
        run_count = len(re.findall(r"Run:\s+\d+, Test AUC:", text))
        if auc and ap:
            return {
                "log_path": str(path.resolve()),
                "auc_mean": float(auc.group(1)),
                "ap_mean": float(ap.group(1)),
                "auc_std": float(auc_std.group(1)) if auc_std else None,
                "ap_std": float(ap_std.group(1)) if ap_std else None,
                "runs": run_count,
            }
    return None


def collect_dataset_stats(dataset: str, root: str):
    data = load_data(dataset, root=root)[0]
    return {
        "nodes": int(data.num_nodes),
        "edges": int(data.edge_index.size(1) // 2),
        "features": int(data.x.size(1)),
        "classes": int(data.y.max().item()) + 1,
    }


def render_markdown(summary):
    lines = [
        "# Local Formal Baseline Summary",
        "",
        "This report freezes the current recovered `ASLAM/code` implementation as the official local baseline.",
        "",
        "## Academic note",
        "",
        "The local datasets do not match the paper's reported Table 1 statistics, so these results are valid only as a local-dataset formal baseline.",
        "They must not be presented as a strict reproduction of the published table.",
        "",
        "## Protocol",
        "",
        "- Split ratio: 85/5/10",
        "- Fixed split seed: 2",
        "- Batch size: 32",
        "- Early-stop patience: 20",
        "- Default formal setting: runs=10, epochs=401",
        "- Comparison rule: all improved models must be compared against this frozen local baseline under the same local datasets and protocol",
        "",
        "## Dataset mismatch check",
        "",
        "| Dataset | Local nodes | Paper nodes | Local edges | Paper edges | Local features | Paper features |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in summary["datasets"]:
        local_stats = item["local_stats"]
        paper_stats = item["paper_stats"]
        lines.append(
            f"| {item['name']} | {local_stats['nodes']} | {paper_stats['nodes']} | "
            f"{local_stats['edges']} | {paper_stats['edges']} | {local_stats['features']} | {paper_stats['features']} |"
        )

    lines.extend(
        [
            "",
            "## Baseline results",
            "",
            "| Dataset | Runs found | AUC mean | AP mean | AUC std | AP std | Log |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for item in summary["datasets"]:
        metrics = item["metrics"]
        if metrics is None:
            lines.append(f"| {item['name']} | 0 | pending | pending | pending | pending | pending |")
            continue
        lines.append(
            f"| {item['name']} | {metrics['runs']} | {metrics['auc_mean']:.4f} | {metrics['ap_mean']:.4f} | "
            f"{metrics['auc_std'] if metrics['auc_std'] is not None else 'n/a'} | "
            f"{metrics['ap_std'] if metrics['ap_std'] is not None else 'n/a'} | {metrics['log_path']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    summary = {
        "protocol": {
            "split": {"train": 0.85, "val": 0.05, "test": 0.10},
            "seed": 2,
            "batch_size": 32,
            "patience": 20,
            "formal_runs": 10,
            "formal_epochs": 401,
        },
        "academic_note": (
            "Local dataset versions differ from the paper's Table 1, so this summary is a local formal baseline only "
            "and cannot be claimed as a strict reproduction of the paper's reported benchmark table."
        ),
        "datasets": [],
    }

    for dataset in args.datasets:
        summary["datasets"].append(
            {
                "name": dataset,
                "local_stats": collect_dataset_stats(dataset, args.root),
                "paper_stats": PAPER_STATS[dataset],
                "metrics": parse_latest_metrics(results_dir, dataset),
            }
        )

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_out.write_text(render_markdown(summary), encoding="utf-8")
    print(f"[local-baseline] wrote {json_out}")
    print(f"[local-baseline] wrote {md_out}")


if __name__ == "__main__":
    main()
