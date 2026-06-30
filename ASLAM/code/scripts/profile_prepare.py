import argparse
import time
from types import SimpleNamespace
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.data_utils import ASLAM_Dataset, load_data, make_split_bundle
from utils.train_utils import construct_knn_graph, shxgb, train_node2vec_emb


def build_args():
    parser = argparse.ArgumentParser(description="Profile ASLAM preprocessing stages.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main():
    cli_args = build_args()
    dataset_args = SimpleNamespace(
        val_ratio=0.05,
        test_ratio=0.10,
        train_percent=1.0,
        val_percent=1.0,
        test_percent=1.0,
    )

    t0 = time.time()
    dataset = load_data(cli_args.dataset, root=cli_args.root)
    print(f"load_data: {time.time() - t0:.2f}s", flush=True)

    t1 = time.time()
    knn_graph = construct_knn_graph(dataset[0])
    print(f"construct_knn_graph: {time.time() - t1:.2f}s", flush=True)

    t2 = time.time()
    emb = train_node2vec_emb(knn_graph)
    print(f"train_node2vec_emb: {time.time() - t2:.2f}s", flush=True)

    t3 = time.time()
    emb_shap = shxgb(emb, dataset[0].y)
    print(f"shxgb: {time.time() - t3:.2f}s", flush=True)

    t4 = time.time()
    split_bundle = make_split_bundle(dataset[0], 0.05, 0.10, seed=2)
    print(f"make_split_bundle: {time.time() - t4:.2f}s", flush=True)

    t5 = time.time()
    train_drnl = ASLAM_Dataset(
        dataset,
        emb_shap,
        dataset_args,
        node_label="drnl",
        num_hops=1,
        split="train",
        split_bundle=split_bundle,
    )
    print(f"train_drnl: {len(train_drnl)} graphs in {time.time() - t5:.2f}s", flush=True)


if __name__ == "__main__":
    main()
