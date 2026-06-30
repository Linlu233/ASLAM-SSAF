import datetime
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torch_geometric import seed_everything
from torch_geometric.data import Batch

from config import parser
from models.bsal import ASLAM
from models.aslam_ssm import ASLAMSSMAttn
from utils.data_utils import ASLAM_Dataset, load_data, make_split_bundle
from utils.eval_utils import evaluate_auc_ap, evaluate_hits
from utils.train_utils import construct_knn_graph, shxgb, train_node2vec_emb

warnings.filterwarnings("ignore", message=".*global_sort_pool.*deprecated.*")

args = parser.parse_args()
device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


class NullWriter:
    def add_scalars(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _paired_batch_iterator(dataset_a, dataset_b, batch_size, shuffle):
    indices = torch.randperm(len(dataset_a)).tolist() if shuffle else list(range(len(dataset_a)))
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        batch_a = Batch.from_data_list([dataset_a[idx] for idx in batch_ids])
        batch_b = Batch.from_data_list([dataset_b[idx] for idx in batch_ids])
        yield batch_a, batch_b


def train(model, train_dataset, train_dataset_label2, knn_graph, device, optimizer, batch_size):
    model.train()
    total_loss = 0.0
    iterator = _paired_batch_iterator(train_dataset, train_dataset_label2, batch_size, shuffle=True)
    for data, data_label2 in iterator:
        data = data.to(device)
        data_label2 = data_label2.to(device)
        optimizer.zero_grad()
        logits_1, logits_2, logits_3 = model(data, data_label2, knn_graph)
        labels = data.y.to(torch.float)
        loss_1 = BCEWithLogitsLoss()(logits_1.view(-1), labels)
        loss_2 = BCEWithLogitsLoss()(logits_2.view(-1), labels)
        loss_3 = BCEWithLogitsLoss()(logits_3.view(-1), labels)
        loss = (loss_1 + loss_2 + loss_3) / 3.0
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(args, dataset, dataset_label2, batch_size, knn_graph, model, device):
    model.eval()
    y_pred, y_true = [], []
    iterator = _paired_batch_iterator(dataset, dataset_label2, batch_size, shuffle=False)
    for data, data_label2 in iterator:
        data = data.to(device)
        data_label2 = data_label2.to(device)
        _, _, logits = model(data, data_label2, knn_graph)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    if args.metric == "hits":
        pos_pred = y_pred[y_true == 1]
        neg_pred = y_pred[y_true == 0]
        return evaluate_hits({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred}, args.K)
    return evaluate_auc_ap(y_pred, y_true)


def prepare_experiment(args):
    logger.info("prepare_experiment: load_data")
    dataset = load_data(args.dataset, root=args.root)
    logger.info("prepare_experiment: construct_knn_graph")
    knn_graph = construct_knn_graph(dataset[0])
    logger.info("prepare_experiment: train_node2vec_emb")
    emb = train_node2vec_emb(knn_graph)
    logger.info("prepare_experiment: shxgb")
    emb_shap = shxgb(emb, dataset[0].y)
    knn_graph.x = emb_shap.to(torch.float32)
    logger.info("prepare_experiment: make_split_bundle")
    split_bundle = make_split_bundle(dataset[0], float(args.val_ratio), float(args.test_ratio), seed=2)
    subgraph_cache = None if args.dataset.lower() == "pubmed" else {}

    logger.info("prepare_experiment: build train_dataset drnl")
    train_dataset = ASLAM_Dataset(
        dataset, emb_shap, args, node_label="drnl", num_hops=1, split="train", split_bundle=split_bundle, subgraph_cache=subgraph_cache
    )
    logger.info("prepare_experiment: build val_dataset drnl")
    val_dataset = ASLAM_Dataset(
        dataset, emb_shap, args, node_label="drnl", num_hops=1, split="val", split_bundle=split_bundle, subgraph_cache=subgraph_cache
    )
    logger.info("prepare_experiment: build test_dataset drnl")
    test_dataset = ASLAM_Dataset(
        dataset, emb_shap, args, node_label="drnl", num_hops=1, split="test", split_bundle=split_bundle, subgraph_cache=subgraph_cache
    )
    logger.info("prepare_experiment: build train_dataset de")
    train_dataset_label2 = ASLAM_Dataset(
        dataset, emb_shap, args, node_label="de", num_hops=1, split="train", split_bundle=split_bundle, subgraph_cache=subgraph_cache
    )
    logger.info("prepare_experiment: build val_dataset de")
    val_dataset_label2 = ASLAM_Dataset(
        dataset, emb_shap, args, node_label="de", num_hops=1, split="val", split_bundle=split_bundle, subgraph_cache=subgraph_cache
    )
    logger.info("prepare_experiment: build test_dataset de")
    test_dataset_label2 = ASLAM_Dataset(
        dataset, emb_shap, args, node_label="de", num_hops=1, split="test", split_bundle=split_bundle, subgraph_cache=subgraph_cache
    )
    logger.info("prepare_experiment: done")
    return {
        "dataset": dataset,
        "knn_graph": knn_graph,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_dataset_label2": train_dataset_label2,
        "val_dataset_label2": val_dataset_label2,
        "test_dataset_label2": test_dataset_label2,
    }


def run(args, prepared):
    dataset = prepared["dataset"]
    knn_graph = prepared["knn_graph"]
    train_dataset = prepared["train_dataset"]
    val_dataset = prepared["val_dataset"]
    test_dataset = prepared["test_dataset"]
    train_dataset_label2 = prepared["train_dataset_label2"]
    val_dataset_label2 = prepared["val_dataset_label2"]
    test_dataset_label2 = prepared["test_dataset_label2"]

    if args.model_variant == "baseline":
        model = ASLAM(
            train_dataset,
            train_dataset_label2,
            train_dataset[0].num_features,
            train_dataset_label2[0].num_features,
            hidden_channels=32,
            out_channels=32,
            num_layers=3,
        ).to(device)
    elif args.model_variant == "ssmattn":
        model = ASLAMSSMAttn(
            train_dataset,
            train_dataset_label2,
            train_dataset[0].num_features,
            train_dataset_label2[0].num_features,
            hidden_channels=args.ssm_hidden,
            out_channels=32,
            num_layers=3,
            semantic_dim=knn_graph.x.size(-1),
            attr_dim=dataset[0].x.size(-1),
            dropout=args.fusion_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_variant: {args.model_variant}")
    logger.info(model)

    knn_graph = knn_graph.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    if args.metric == "auc_ap":
        best_val_auc = best_val_ap = test_auc = test_ap = 0.0
    elif args.metric == "hits":
        best_val_hits = test_hits = 0.0
    else:
        raise ValueError("Invalid metric")

    patience = 0
    log_dir = Path("graphtrain") / args.model_variant / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = NullWriter()

    for epoch in range(1, args.epochs):
        loss = train(model, train_dataset, train_dataset_label2, knn_graph, device, optimizer, args.bs)
        scheduler.step()
        results = test(args, val_dataset, val_dataset_label2, args.bs, knn_graph, model, device)
        writer.add_scalars("Loss", {f"Train_Loss_{args.model_variant}_{args.train_percent}": loss}, epoch)

        if args.metric == "auc_ap":
            val_auc, val_ap = results["AUC"], results["AP"]
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_ap = val_ap
                test_results = test(args, test_dataset, test_dataset_label2, args.bs, knn_graph, model, device)
                test_auc, test_ap = test_results["AUC"], test_results["AP"]
                patience = 0
            else:
                patience += 1
            logger.info(
                "Epoch: %02d, Loss: %.4f, Val_AUC: %.4f, Val_AP: %.4f, Test_AUC: %.4f, Test_AP: %.4f",
                epoch,
                loss,
                val_auc,
                val_ap,
                test_auc,
                test_ap,
            )
            writer.add_scalars("AUC", {f"Test_AUC_{args.model_variant}_{args.train_percent}": test_auc}, epoch)
            writer.add_scalars("AP", {f"Test_AP_{args.model_variant}_{args.train_percent}": test_ap}, epoch)
            if patience >= args.patience:
                logger.info(
                    "Early Stop! Best Val AUC: %.4f, Best Val AP: %.4f, Test AUC: %.4f, Test AP: %.4f",
                    best_val_auc,
                    best_val_ap,
                    test_auc,
                    test_ap,
                )
                break
        else:
            val_hits = results[f"hits@{args.K}"]
            if val_hits > best_val_hits:
                best_val_hits = val_hits
                test_hits = test(args, test_dataset, test_dataset_label2, args.bs, knn_graph, model, device)[f"hits@{args.K}"]
                patience = 0
            else:
                patience += 1
            logger.info("Epoch: %02d, Loss: %.4f, Val Hits: %.4f, Test Hits: %.4f", epoch, loss, val_hits, test_hits)
            if patience >= args.patience:
                logger.info("Early Stop! Best Val Hits: %.4f, Test Hits: %.4f", best_val_hits, test_hits)
                break

    writer.close()
    return [test_auc, test_ap] if args.metric == "auc_ap" else [test_hits]


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    exp_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("../results/train=0.9")
    result_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        result_dir / f"{args.dataset.capitalize()}_{args.model_variant}_{float(args.train_percent)}_feat_{args.use_feat}_{exp_time}.txt"
    )
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)
    res = []
    seed_everything(2)
    prepared = prepare_experiment(args)
    for _ in range(args.runs):
        seed_everything(2)
        res.append(run(args, prepared))

    if args.metric == "auc_ap":
        for i, run_result in enumerate(res, start=1):
            logger.info("Run: %2d, Test AUC: %.4f, Test AP: %.4f", i, run_result[0], run_result[1])
        auc = np.mean([item[0] for item in res])
        ap = np.mean([item[1] for item in res])
        logger.info("The average AUC for test data is %.4f", auc)
        logger.info("The average AP for test data is %.4f", ap)
        logger.info("The std of AUC for test data is %.4f", np.std([item[0] for item in res]))
        logger.info("The std of AP for test data is %.4f", np.std([item[1] for item in res]))
    else:
        for i, run_result in enumerate(res, start=1):
            logger.info("Run: %2d, HITS@%02d: %.4f", i, args.K, run_result[0])
        hits = np.mean([item[0] for item in res])
        logger.info("The average HITS@%02d for test data is %.4f", args.K, hits)
        logger.info("The std of Hits@%02d for test data is %.4f", args.K, np.std([item[0] for item in res]))

    endtime = datetime.datetime.now()
    logger.info((endtime - starttime).seconds)
