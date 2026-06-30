import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASLAM and ASLAM-Plus for link prediction on citation/coauthor graphs."
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="Citeseer,DBLP,PubMed,amz_Photo",
        help=(
            "Comma-separated dataset names. Supported: "
            "Citeseer, DBLP, PubMed, amz_Photo, CoRA, Twitch_EN"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="compare",
        choices=["aslam", "aslam_plus", "aslam_ssmattn", "compare", "compare_plus", "compare_all"],
        help=(
            "Model to train. 'compare' runs the formal baseline-vs-new model comparison, "
            "'compare_plus' keeps the legacy ASLAM-vs-ASLAMPlus comparison, "
            "and 'compare_all' runs all three models."
        ),
    )
    parser.add_argument(
        "--root",
        type=str,
        default="datasets",
        help="Dataset root directory.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=12, help="Early-stop patience.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout ratio.")
    parser.add_argument("--hidden_channels", type=int, default=96, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of graph layers.")
    parser.add_argument("--batch_size", type=int, default=8192, help="Edge batch size.")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Train edge ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation edge ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Test edge ratio.")
    parser.add_argument(
        "--neg_sampling_ratio",
        type=float,
        default=1.0,
        help="Negative sampling ratio for each positive edge.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device. Use 'auto', 'cpu', or e.g. 'cuda:0'.",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        help="Row-normalize node features before training.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory for JSON summaries.",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="aslam_benchmark_summary.json",
        help="JSON filename for the current experiment summary.",
    )
    return parser


parser = build_parser()
