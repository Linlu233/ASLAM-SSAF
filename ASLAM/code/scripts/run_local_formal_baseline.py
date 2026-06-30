import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ["citeseer", "pubmed", "dblp", "photo"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run ASLAM-family benchmarks with the paper-aligned protocol."
    )
    parser.add_argument(
        "--model-variant",
        default="baseline",
        help="Model variant passed to train.py. baseline keeps the frozen official baseline.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to benchmark. Supported here: citeseer pubmed dblp photo.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs.")
    parser.add_argument("--epochs", type=int, default=401, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--cuda", default="cuda:0", help="Torch device string.")
    parser.add_argument(
        "--root",
        default=str((Path(__file__).resolve().parents[3] / "datasets").resolve()),
        help="Dataset root directory.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    code_root = Path(__file__).resolve().parents[1]
    train_script = code_root / "train.py"

    for dataset in args.datasets:
        cmd = [
            sys.executable,
            str(train_script),
            "--dataset",
            dataset,
            "--model_variant",
            args.model_variant,
            "--runs",
            str(args.runs),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--bs",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--wd",
            str(args.wd),
            "--train_percent",
            "1.0",
            "--val_percent",
            "1.0",
            "--test_percent",
            "1.0",
            "--root",
            args.root,
            "--cuda",
            args.cuda,
        ]
        print(f"[benchmark] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(code_root), check=True)


if __name__ == "__main__":
    main()
