from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List


DISPLAY_NAME = {
    "cn": "CN",
    "aa": "AA",
    "node2vec": "Node2Vec",
    "gcn": "GCN",
    "gae": "GAE",
    "vgae": "VGAE",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "supergat": "SuperGAT",
    "seal": "SEAL",
    "bsal": "BSAL",
    "aslam": "ASLAM",
    "aslam_ssmattn": "ASLAM-SSAF",
}


MODEL_ORDER = [
    "cn",
    "aa",
    "node2vec",
    "gcn",
    "gae",
    "vgae",
    "graphsage",
    "gat",
    "supergat",
    "seal",
    "bsal",
    "aslam",
    "aslam_ssmattn",
]


SOURCE_NOTES = {
    "cn": "本地闭式启发式实现，无需外部代码依赖。",
    "aa": "本地闭式启发式实现，无需外部代码依赖。",
    "node2vec": "基于 PyTorch Geometric 2.7.0 官方 Node2Vec 实现。",
    "gcn": "基于 PyTorch Geometric 2.7.0 官方组件实现。",
    "gae": "基于 PyTorch Geometric 2.7.0 官方 GAE 实现。",
    "vgae": "基于 PyTorch Geometric 2.7.0 官方 VGAE 实现。",
    "graphsage": "基于 PyTorch Geometric 2.7.0 官方 GraphSAGE 组件实现。",
    "gat": "基于 PyTorch Geometric 2.7.0 官方 GAT 组件实现。",
    "supergat": "基于 PyTorch Geometric 2.7.0 SuperGATConv，并参考官方 SuperGAT 项目。",
    "seal": "根目录下按公开 SEAL 范式重建的 1-hop DRNL + DGCNN 子图分类实现。",
    "bsal": "根目录下依据 BSAL 论文与恢复代码片段重建的双分量融合实现，非独立官方仓库逐字复现。",
    "aslam": "根目录正式 baseline，对应既有 formal compare 结果。",
    "aslam_ssmattn": "根目录正式改进模型，对应既有 formal compare 结果。",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile a Markdown report for the paper baseline suite.")
    parser.add_argument(
        "--inputs",
        type=str,
        default="results/paper_baselines/paper_baselines_*.json",
        help="Glob pattern for summary JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/paper_baselines/paper_baseline_partial_report.md",
        help="Output Markdown path.",
    )
    return parser


def merge_payloads(payloads: List[Dict]) -> Dict:
    merged = {
        "args": {},
        "device": None,
        "sources": {},
        "datasets": {},
        "root_formal_models": {},
    }
    for payload in payloads:
        merged["args"].update(payload.get("args", {}))
        merged["device"] = payload.get("device", merged["device"])
        merged["sources"].update(payload.get("sources", {}))
        for dataset_name, dataset_payload in payload.get("datasets", {}).items():
            merged["datasets"].setdefault(dataset_name, {})
            merged["datasets"][dataset_name].update(dataset_payload)
        for dataset_name, dataset_payload in payload.get("root_formal_models", {}).items():
            merged["root_formal_models"].setdefault(dataset_name, {})
            merged["root_formal_models"][dataset_name].update(dataset_payload)

    for dataset_name, dataset_payload in merged["root_formal_models"].items():
        merged["datasets"].setdefault(dataset_name, {})
        for model_name in ("aslam", "aslam_ssmattn"):
            if dataset_payload.get(model_name):
                merged["datasets"][dataset_name][model_name] = dataset_payload[model_name]
    return merged


def metric_text(metric: Dict[str, float]) -> str:
    return f"{metric['mean']:.4f} +/- {metric['std']:.4f}"


def build_dataset_table(dataset_payload: Dict[str, Dict]) -> str:
    rows = ["| 模型 | AUC | AP | 最优 epoch | 说明 |", "| --- | ---: | ---: | ---: | --- |"]
    for model_name in MODEL_ORDER:
        if model_name not in dataset_payload:
            continue
        model_payload = dataset_payload[model_name]
        auc = metric_text(model_payload["test_auc"])
        ap = metric_text(model_payload["test_ap"])
        epoch = model_payload["best_epoch"]["mean"] if isinstance(model_payload["best_epoch"], dict) else 0.0
        note = SOURCE_NOTES.get(model_name, "")
        rows.append(f"| {DISPLAY_NAME[model_name]} | {auc} | {ap} | {epoch:.1f} | {note} |")
    return "\n".join(rows)


def build_average_table(merged: Dict) -> str:
    model_values: Dict[str, Dict[str, List[float]]] = {}
    for dataset_payload in merged["datasets"].values():
        for model_name, model_payload in dataset_payload.items():
            model_values.setdefault(model_name, {"auc": [], "ap": []})
            model_values[model_name]["auc"].append(model_payload["test_auc"]["mean"])
            model_values[model_name]["ap"].append(model_payload["test_ap"]["mean"])

    rows = ["| 模型 | 覆盖数据集数 | 平均 AUC | 平均 AP |", "| --- | ---: | ---: | ---: |"]
    for model_name in MODEL_ORDER:
        if model_name not in model_values:
            continue
        auc_values = model_values[model_name]["auc"]
        ap_values = model_values[model_name]["ap"]
        rows.append(
            f"| {DISPLAY_NAME[model_name]} | {len(auc_values)} | {sum(auc_values) / len(auc_values):.4f} | {sum(ap_values) / len(ap_values):.4f} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    paths = sorted(glob.glob(args.inputs))
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not payloads:
        raise FileNotFoundError(f"No JSON files matched: {args.inputs}")

    merged = merge_payloads(payloads)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(merged["datasets"].keys())

    lines: List[str] = []
    lines.append("# ASLAM 论文对比基线的本地统一实验报告")
    lines.append("")
    lines.append("## 1. 任务范围")
    lines.append("")
    lines.append("本报告汇总 ASLAM 论文中主要对比基线在当前 `ASLAM-3` 根目录统一协议下的本地实验结果。")
    lines.append("报告同时纳入根目录既有正式 `ASLAM` 与 `ASLAM-SSAF` 结果，便于形成统一对比表。")
    lines.append("")
    lines.append("## 2. 统一 protocol")
    lines.append("")
    lines.append(f"- 设备：`{merged['device']}`")
    lines.append(f"- runs：`{merged['args'].get('runs')}`")
    lines.append(f"- epochs：`{merged['args'].get('epochs')}`")
    lines.append(f"- patience：`{merged['args'].get('patience')}`")
    lines.append(f"- seed 起点：`{merged['args'].get('seed')}`")
    lines.append(f"- val/test：`{merged['args'].get('val_ratio')}` / `{merged['args'].get('test_ratio')}`")
    lines.append(f"- neg sampling：`{merged['args'].get('neg_sampling_ratio')}`")
    lines.append(f"- 数据集：`{', '.join(datasets)}`")
    lines.append("")
    lines.append("## 3. 学术规范说明")
    lines.append("")
    lines.append("1. 本报告结果只代表当前本地固定数据、固定划分比例与固定随机种子协议下的 `local formal comparison`。")
    lines.append("2. 若某个 baseline 使用的是根目录重建实现或依据作者代码片段恢复的实现，而非独立官方仓库逐字运行，表中会明确标注，不混写为严格官方复现。")
    lines.append("3. `ASLAM` 与 `ASLAM-SSAF` 行来自根目录既有 formal compare JSON，与本轮基线统一纳入同一张总表，但应与原论文表格分层陈述。")
    lines.append("")
    lines.append("## 4. 代码来源与实现口径")
    lines.append("")
    lines.append("| 模型 | 实现口径 |")
    lines.append("| --- | --- |")
    for model_name in MODEL_ORDER:
        if model_name in SOURCE_NOTES:
            lines.append(f"| {DISPLAY_NAME[model_name]} | {SOURCE_NOTES[model_name]} |")
    lines.append("")
    lines.append("## 5. 分数据集结果")
    lines.append("")
    for idx, dataset_name in enumerate(datasets, start=1):
        lines.append(f"### 5.{idx} {dataset_name}")
        lines.append("")
        lines.append(build_dataset_table(merged["datasets"][dataset_name]))
        lines.append("")
    lines.append("## 6. 跨数据集平均结果")
    lines.append("")
    lines.append(build_average_table(merged))
    lines.append("")
    lines.append("## 7. 写作建议")
    lines.append("")
    lines.append("1. 对外写作时，应严格区分“原论文表格数值”“本地 recovered baseline”“根目录统一协议 formal comparison”三层结果。")
    lines.append("2. 若某个 baseline 方差明显偏大，应结合实现口径、训练稳定性与数据划分敏感性进行讨论，不宜仅依据单次最优值下结论。")
    lines.append("3. 若当前总表尚未覆盖全部模型或全部数据集，应在论文中明确标注为阶段性汇总。")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Markdown report to {output_path}")


if __name__ == "__main__":
    main()
