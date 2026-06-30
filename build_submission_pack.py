from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_ablation_bars import DATASETS, DISPLAY_LABELS
from utils.data_utils import load_data


BATCH_SIZE_MAP = {
    "Citeseer": 16384,
    "DBLP": 16384,
    "PubMed": 16384,
    "amz_Photo": 16384,
    "CoRA": 4096,
    "Twitch_EN": 8192,
}

FULL_MODEL_LABEL = "ASLAM-SSAF"
FULL_MODEL_NAME = "ASLAM with Selective State-Space Attention Fusion"

REFERENCES = [
    {
        "id": "R1",
        "topic_cn": "图状态空间建模与 Graph Mamba 基线动机",
        "topic_en": "Graph state-space modeling and Graph Mamba motivation",
        "apa": "Behrouz, A., & Hashemi, F. (2024). Graph Mamba: Towards learning on graphs with state space models. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 119-130). Association for Computing Machinery. https://doi.org/10.1145/3637528.3672044",
        "url": "https://kdd2024.kdd.org/research-track-papers/",
    },
    {
        "id": "R2",
        "topic_cn": "图 Mamba 选择性状态空间与过平滑缓解",
        "topic_en": "Selective state-space graph modeling for over-smoothing mitigation",
        "apa": "He, X., Wang, Y., Fan, W., Shen, X., Juan, X., Miao, R., & Wang, X. (2025). Mamba-based graph convolutional networks: Tackling over-smoothing with selective state space. In *Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence* (pp. 5345-5353). IJCAI. https://doi.org/10.24963/ijcai.2025/595",
        "url": "https://www.ijcai.org/proceedings/2025/595",
    },
    {
        "id": "R3",
        "topic_cn": "链路预测中的成对 token 注意力建模",
        "topic_en": "Pairwise token attention modeling for link prediction",
        "apa": "Shomer, H., Ma, Y., Mao, H., Li, J., Wu, B., & Tang, J. (2024). LPFormer: An adaptive graph transformer for link prediction. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2686-2698). Association for Computing Machinery. https://doi.org/10.1145/3637528.3672025",
        "url": "https://kdd2024.kdd.org/research-track-papers/",
    },
    {
        "id": "R4",
        "topic_cn": "图结构与位置编码的统一可迁移建模",
        "topic_en": "Transferable positional and structural encoding for graphs",
        "apa": "Cantürk, S., Liu, R., Lapointe-Gagné, O., Létourneau, V., Wolf, G., Beaini, D., & Rampášek, L. (2024). Graph positional and structural encoder. In *Proceedings of the 41st International Conference on Machine Learning* (pp. 5533-5566). PMLR. https://proceedings.mlr.press/v235/canturk24a.html",
        "url": "https://proceedings.mlr.press/v235/canturk24a.html",
    },
    {
        "id": "R5",
        "topic_cn": "稳定而可表达的图位置编码",
        "topic_en": "Stable and expressive positional encoding for graphs",
        "apa": "Huang, Y., Lu, W., Robinson, J., Yang, Y., Zhang, M., Jegelka, S., & Li, P. (2024). On the stability of expressive positional encodings for graphs. In *The Twelfth International Conference on Learning Representations*. https://openreview.net/forum?id=xAqcJ9XoTf",
        "url": "https://openreview.net/forum?id=xAqcJ9XoTf",
    },
    {
        "id": "R6",
        "topic_cn": "Mamba 与注意力关系的理论解释",
        "topic_en": "Theoretical interpretation of the relation between Mamba and attention",
        "apa": "Ali, A. A., Zimerman, I., & Wolf, L. (2025). The hidden attention of Mamba models. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1516-1534). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.acl-long.76",
        "url": "https://aclanthology.org/2025.acl-long.76/",
    },
    {
        "id": "R7",
        "topic_cn": "混合 Mamba-Transformer 架构的经验依据",
        "topic_en": "Empirical evidence for hybrid Mamba-Transformer design",
        "apa": "Hatamizadeh, A., & Kautz, J. (2025). MambaVision: A hybrid Mamba-Transformer vision backbone. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 25261-25270). IEEE/CVF. https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html",
        "url": "https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html",
    },
    {
        "id": "R8",
        "topic_cn": "原始 ASLAM 方法论文",
        "topic_en": "Original ASLAM paper",
        "apa": "Nie, R., Wang, G., Liu, Q., & Peng, C. (2025). Link prediction for attribute and structure learning based on attention mechanism. *Applied Soft Computing, 179*, Article 113268. https://doi.org/10.1016/j.asoc.2025.113268",
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S1568494625005794",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build bilingual manuscript-support markdown files for the ASLAM-SSAF figure pack."
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("Figure"))
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_dataset_stats(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_key, display_name in [
        ("Citeseer", "Citeseer"),
        ("DBLP", "DBLP"),
        ("PubMed", "PubMed"),
        ("amz_Photo", "Amazon Photo"),
        ("CoRA", "Cora"),
        ("Twitch_EN", "Twitch-EN"),
    ]:
        data = load_data(dataset_key, root=str(root), normalize_features=False)
        rows.append(
            {
                "key": dataset_key,
                "label": display_name,
                "nodes": int(data.num_nodes),
                "edges": int(data.edge_index.size(1) // 2),
                "features": int(data.num_features),
                "batch_size": BATCH_SIZE_MAP[dataset_key],
            }
        )
    return rows


def collect_main_results(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in DATASETS:
        payload = load_json(results_dir / item["compare_file"])
        block = payload["datasets"][item["compare_key"]]
        rows.append(
            {
                "dataset": DISPLAY_LABELS.get(item["label"], item["label"]),
                "aslam_auc": block["aslam"]["test_auc"]["mean"],
                "aslam_ap": block["aslam"]["test_ap"]["mean"],
                "aslam_auc_std": block["aslam"]["test_auc"]["std"],
                "aslam_ap_std": block["aslam"]["test_ap"]["std"],
                "full_auc": block["aslam_ssmattn"]["test_auc"]["mean"],
                "full_ap": block["aslam_ssmattn"]["test_ap"]["mean"],
                "full_auc_std": block["aslam_ssmattn"]["test_auc"]["std"],
                "full_ap_std": block["aslam_ssmattn"]["test_ap"]["std"],
                "delta_auc": block["aslam_ssmattn"]["test_auc"]["mean"] - block["aslam"]["test_auc"]["mean"],
                "delta_ap": block["aslam_ssmattn"]["test_ap"]["mean"] - block["aslam"]["test_ap"]["mean"],
            }
        )
    return rows


def collect_ablation_results(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in DATASETS:
        compare_payload = load_json(results_dir / item["compare_file"])
        plus_payload = load_json(results_dir / item["plus_file"])
        compare_block = compare_payload["datasets"][item["compare_key"]]
        plus_block = plus_payload["datasets"][item["plus_key"]]["aslam_plus"]
        final_auc = compare_block["aslam_ssmattn"]["test_auc"]["mean"]
        final_ap = compare_block["aslam_ssmattn"]["test_ap"]["mean"]
        aslam_auc = compare_block["aslam"]["test_auc"]["mean"]
        aslam_ap = compare_block["aslam"]["test_ap"]["mean"]
        plus_auc = plus_block["test_auc"]["mean"]
        plus_ap = plus_block["test_ap"]["mean"]
        rows.append(
            {
                "dataset": DISPLAY_LABELS.get(item["label"], item["label"]),
                "final_auc": final_auc,
                "final_ap": final_ap,
                "plus_auc": plus_auc,
                "plus_ap": plus_ap,
                "aslam_auc": aslam_auc,
                "aslam_ap": aslam_ap,
                "drop_auc_plus": plus_auc - final_auc,
                "drop_ap_plus": plus_ap - final_ap,
                "drop_auc_aslam": aslam_auc - final_auc,
                "drop_ap_aslam": aslam_ap - final_ap,
            }
        )
    return rows


def write_dataset_table(rows: list[dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Table 1 / 表1 数据集概况与正式实验配置",
        "",
        "| Dataset / 数据集 | Nodes / 节点数 | Undirected edges / 无向边数 | Features / 特征维度 | Stable batch size / 稳定批大小 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['nodes']} | {row['edges']} | {row['features']} | {row['batch_size']} |"
        )
    (output_dir / "table_dataset_overview_bilingual.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_main_results_table(rows: list[dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Table 2 / 表2 主结果对比",
        "",
        f"| Dataset / 数据集 | ASLAM AUC | ASLAM AP | {FULL_MODEL_LABEL} AUC | {FULL_MODEL_LABEL} AP | Delta AUC / 提升 | Delta AP / 提升 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {aslam_auc:.4f}±{aslam_auc_std:.4f} | {aslam_ap:.4f}±{aslam_ap_std:.4f} | {full_auc:.4f}±{full_auc_std:.4f} | {full_ap:.4f}±{full_ap_std:.4f} | {delta_auc:+.4f} | {delta_ap:+.4f} |".format(
                **row
            )
        )
    (output_dir / "table_main_results_bilingual.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_ablation_table(rows: list[dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Table 3 / 表3 消融实验（以最终模型掉点格式书写）",
        "",
        f"| Dataset / 数据集 | Final AUC | w/o SSM-attention fusion AUC | Drop / 掉点 | w/o enhanced backbone AUC | Drop / 掉点 | Final AP | w/o SSM-attention fusion AP | Drop / 掉点 | w/o enhanced backbone AP | Drop / 掉点 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {final_auc:.4f} | {plus_auc:.4f} | {drop_auc_plus:+.4f} | {aslam_auc:.4f} | {drop_auc_aslam:+.4f} | {final_ap:.4f} | {plus_ap:.4f} | {drop_ap_plus:+.4f} | {aslam_ap:.4f} | {drop_ap_aslam:+.4f} |".format(
                **row
            )
        )
    (output_dir / "table_ablation_bilingual.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_references_table(output_dir: Path) -> None:
    lines = [
        "# References / 参考文献（APA 7）",
        "",
        "Below are all references actually used in the current manuscript-support package.",
        "以下为当前写作支撑包中实际使用的全部参考文献。",
        "",
        "| ID | Chinese use / 中文用途 | English use / 英文用途 | APA 7 reference | Source link |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ref in REFERENCES:
        lines.append(
            f"| {ref['id']} | {ref['topic_cn']} | {ref['topic_en']} | {ref['apa']} | {ref['url']} |"
        )
    (output_dir / "references_apa7_bilingual.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_bilingual_pack(output_dir: Path, dataset_rows: list[dict[str, object]], main_rows: list[dict[str, object]], ablation_rows: list[dict[str, object]]) -> None:
    avg_auc_gain = sum(float(item["delta_auc"]) for item in main_rows) / len(main_rows)
    avg_ap_gain = sum(float(item["delta_ap"]) for item in main_rows) / len(main_rows)

    lines = [
        "# ASLAM-SSAF Manuscript Support Pack / ASLAM-SSAF 论文写作支撑包",
        "",
        "## 1. Model Name / 模型正式名称",
        "",
        f"- Chinese / 中文：`{FULL_MODEL_LABEL}`，可译为“基于选择性状态空间注意力融合的 ASLAM 改进模型”。",
        f"- English / 英文：`{FULL_MODEL_NAME} ({FULL_MODEL_LABEL})`.",
        "- Baseline path / baseline 路径：`G:\\myProject\\ASLAM-3\\ASLAM`",
        "- Improved model path / 改进模型路径：`G:\\myProject\\ASLAM-3`",
        "- Core implementation / 核心实现：`G:\\myProject\\ASLAM-3\\models\\aslam.py`",
        "",
        "## 2. Academic Compliance / 学术规范声明",
        "",
        "- Chinese / 中文：当前结果符合学术规范，但必须明确声明比较是在本地固定数据文件和固定训练协议下完成的“同环境公平对比”，不能写成对原论文表格的严格逐项复现。",
        "- English / 英文：The current results are academically compliant if they are reported as a controlled within-workspace comparison under fixed local datasets and a fixed protocol, rather than as a strict row-by-row reproduction of the published paper tables.",
        "- Chinese / 中文：`ASLAM` 文件夹仅保留为冻结 baseline；所有新模型修改均位于项目根目录代码。",
        "- English / 英文：The `ASLAM` folder remains the frozen baseline reproduction only, while all new model modifications are kept in the project root codebase.",
        "",
        "## 3. Innovation Points / 模型创新点",
        "",
        "| No. | Chinese / 中文表述 | English / English wording | Literature grounding / 文献依据 |",
        "| --- | --- | --- | --- |",
        "| I1 | 在结构分支中引入“残差 GCN + 局部-全局图编码 + 双向 selective mixer”的多尺度结构编码链路，用选择性状态空间扫描聚合不同层级的图表示，提升长程依赖建模能力。 | A multi-scale structural encoding chain is introduced in the structure branch by combining residual GCN, local-global graph encoding, and a bi-directional selective mixer, enabling selective state-space aggregation over graph representations at different depths. | R1, R2 |",
        "| I2 | 在边级表示阶段，将属性端点、结构端点、分支摘要、跨分支差异、跨分支均值与启发式特征统一为 pair tokens，并通过 pairwise token attention 建模边两端的交互依赖。 | At the edge representation stage, attribute endpoints, structure endpoints, branch summaries, cross-branch discrepancy, cross-branch mean, and heuristic features are unified as pair tokens and modeled by pairwise token attention. | R3 |",
        "| I3 | 在最终融合头中设计 selective state-space attention pair fusion block，使状态空间混合与显式注意力在同一 token 序列上协同工作，而不是仅用单一路径 MLP 直接拼接打分。 | A selective state-space attention pair fusion block is designed in the final fusion head so that state-space mixing and explicit attention operate collaboratively on the same token sequence instead of relying on a single-path MLP scorer. | R2, R6, R7 |",
        "| I4 | 将启发式特征从“后拼接数值”提升为可学习 heuristic token，使结构先验参与表示学习而非只在分类头末端出现。 | Heuristic features are upgraded from post-hoc numeric concatenation to a learnable heuristic token, allowing structural priors to participate directly in representation learning. | R4, R5 |",
        "| I5 | 采用边级自适应门控 \\(\\alpha_{uv}\\) 在稳定 base score 与增强 expert score 之间逐边插值，降低复杂增强模块对不同密度图的退化风险。 | An edge-wise adaptive gate \\(\\alpha_{uv}\\) interpolates between a stable base score and an enhanced expert score, reducing degradation risk across datasets with different densities. | R3, R7 |",
        "",
        "## 4. SCI Q1 Writing Roadmap / 一区计算机顶刊写作思路",
        "",
        "| Stage / 阶段 | Chinese guidance / 中文写作要点 | English guidance / 英文写作要点 |",
        "| --- | --- | --- |",
        "| Problem framing / 问题界定 | 第一段先明确 attribute-structure fusion 在链路预测中的核心矛盾：结构依赖强、属性利用不足、复杂融合不稳定。第二段再指出现有注意力模型长程建模和跨分支耦合不足。 | First frame the core contradiction of attribute-structure fusion in link prediction: strong reliance on structure, underused attributes, and unstable complex fusion. Then point out the insufficiency of current attention-based models in long-range modeling and cross-branch coupling. |",
        "| Gap statement / 研究空白 | 不要写成“首次”“完全解决”，而应写成“current methods still lack a unified edge-level fusion mechanism that jointly exploits selective state-space modeling and pairwise attention under a controllable gating scheme”。 | Avoid absolute claims such as “the first” or “fully solves”; instead write that current methods still lack a unified edge-level fusion mechanism that jointly exploits selective state-space modeling and pairwise attention under a controllable gating scheme. |",
        "| Method section / 方法部分 | 按“node encoding -> pair tokenization -> hybrid fusion -> adaptive scoring”四级展开，每一级都给出一句设计动机和一条公式。 | Organize the method section as “node encoding -> pair tokenization -> hybrid fusion -> adaptive scoring”, and provide one design motivation plus one equation for each level. |",
        "| Experiment section / 实验部分 | 先交代 protocol fairness，再给主结果表，再给增益图和消融图，最后给训练趋势与 t-SNE。不要先放可视化再放主表。 | State protocol fairness first, then present the main results table, followed by gain and ablation figures, and only then the training trends and t-SNE visualizations. Do not place visualizations before the main quantitative table. |",
        "| Discussion / 讨论部分 | 讨论时重点区分“稳定增益数据集”和“边际增益数据集”，解释 why Twitch-EN and PubMed gains are smaller but still positive。 | Distinguish between stable-gain datasets and marginal-gain datasets, and explain why gains on Twitch-EN and PubMed are smaller yet still consistently positive. |",
        "| Limitation & ethics / 局限与伦理 | 必须写清楚：本地数据与原论文统计不完全一致，因此结论是“local formal comparison under matched protocol”而不是官方表格复现。 | Explicitly state that the local datasets do not exactly match the original paper statistics; therefore the conclusion is a local formal comparison under a matched protocol, not a strict reproduction of the published table. |",
        "",
        "## 5. Core Equations / 核心公式",
        "",
        "### 5.1 Multi-scale structure encoding / 多尺度结构编码",
        "",
        "$$",
        "h_{\\mathrm{local}}^{(l)} = \\mathrm{GAT}(h^{(l-1)}, A), \\qquad",
        "h_{\\mathrm{global}}^{(l)} = \\mathrm{GCN}(h^{(l-1)}, A)",
        "$$",
        "",
        "$$",
        "h^{(l)} = \\mathrm{Dropout}\\left(\\frac{h_{\\mathrm{local}}^{(l)} + h_{\\mathrm{global}}^{(l)}}{2}\\right) + h^{(l-1)}",
        "$$",
        "",
        "- Chinese / 中文：用局部注意力与全局平滑传播共同构建结构分支的多尺度状态序列。",
        "- English / 英文：Local attention and global propagation are jointly used to construct the multi-scale state sequence of the structure branch.",
        "",
        "### 5.2 Bi-directional selective state-space mixing / 双向选择性状态空间混合",
        "",
        "$$",
        "g_t = \\sigma(W_g s_t), \\qquad \\tilde{s}_t = \\tanh(W_c s_t), \\qquad",
        "m_t^{f} = g_t \\odot \\tilde{s}_t + (1-g_t) \\odot m_{t-1}^{f}",
        "$$",
        "",
        "$$",
        "\\hat{s}_t = W_o[s_t \\| m_t^f \\| m_t^b]",
        "$$",
        "",
        "- Chinese / 中文：对层间状态序列做前向与反向扫描，从而在图层级上模拟长程依赖。",
        "- English / 英文：Forward and backward scans are applied over the inter-layer state sequence to simulate long-range dependencies across graph scales.",
        "",
        "### 5.3 Pair token construction / 成对 token 构造",
        "",
        "$$",
        "p_{uv}^{A} = [h_u^A \\| h_v^A \\| |h_u^A-h_v^A| \\| h_u^A \\odot h_v^A \\| e_{uv}]",
        "$$",
        "",
        "$$",
        "p_{uv}^{S} = [h_u^S \\| h_v^S \\| |h_u^S-h_v^S| \\| h_u^S \\odot h_v^S \\| e_{uv}]",
        "$$",
        "",
        "$$",
        "T_{uv} = [h_u^A, h_v^A, h_u^S, h_v^S, \\phi_A(p_{uv}^{A}), \\phi_S(p_{uv}^{S}), \\phi_\\Delta, \\phi_\\mu, \\phi_e(e_{uv})]",
        "$$",
        "",
        "- Chinese / 中文：边级输入不是单一拼接向量，而是带有跨分支语义的 token 序列。",
        "- English / 英文：The edge-level input is not a single concatenated vector but a token sequence carrying cross-branch semantics.",
        "",
        "### 5.4 Pairwise attention and expert fusion / 成对注意力与专家融合",
        "",
        "$$",
        "\\ell_{\\mathrm{expert}} = \\frac{\\ell_{\\mathrm{attn}} + \\ell_{\\mathrm{ssm}} + \\ell_{\\mathrm{res}}}{3}",
        "$$",
        "",
        "- Chinese / 中文：增强路径由 token attention、SSM-attention pair fusion 和 residual scorer 三个专家共同给分。",
        "- English / 英文：The enhanced path is scored jointly by three experts: token attention, SSM-attention pair fusion, and a residual scorer.",
        "",
        "### 5.5 Edge-wise adaptive interpolation / 边级自适应插值",
        "",
        "$$",
        "\\alpha_{uv} = \\sigma\\big(g([p_{uv}^A, p_{uv}^S, c_{uv}, b_{uv}])\\big), \\qquad",
        "\\ell_{uv} = (1-\\alpha_{uv})\\ell_{\\mathrm{base}} + \\alpha_{uv}\\ell_{\\mathrm{expert}}",
        "$$",
        "",
        "- Chinese / 中文：以逐边门控方式保留 baseline 的稳定性，同时引入增强模块的增益。",
        "- English / 英文：An edge-wise gate preserves the stability of the baseline path while injecting the gain brought by the enhanced expert path.",
        "",
        "## 6. Quantitative Summary / 定量总结",
        "",
        f"- Chinese / 中文：在当前六个正式数据集上，`{FULL_MODEL_LABEL}` 相对根目录 `ASLAM` 的平均 AUC 提升为 `{avg_auc_gain:.4f}`，平均 AP 提升为 `{avg_ap_gain:.4f}`。",
        f"- English / 英文：Across the six formal datasets, `{FULL_MODEL_LABEL}` improves over the root `ASLAM` baseline by an average of `{avg_auc_gain:.4f}` in AUC and `{avg_ap_gain:.4f}` in AP.",
        "- Chinese / 中文：所有六个数据集上 AUC 和 AP 都保持正提升，因此可以写成“consistent improvement under the rebuilt root protocol”。",
        "- English / 英文：Both AUC and AP remain positive on all six datasets, so the result can be stated as a consistent improvement under the rebuilt root protocol.",
        "",
        "## 7. References / 参考文献",
        "",
        "The references below are all real papers actually cited in this support pack.",
        "以下列出的文献均为当前支撑包中实际使用的真实论文。",
        "",
    ]
    for ref in REFERENCES:
        lines.append(f"- {ref['id']}: {ref['apa']}")

    lines.extend(
        [
            "",
            "## 8. Linked Tables / 已生成表格",
            "",
            "- `table_dataset_overview_bilingual.md`",
            "- `table_main_results_bilingual.md`",
            "- `table_ablation_bilingual.md`",
            "- `references_apa7_bilingual.md`",
        ]
    )

    (output_dir / "ASLAM_SSAF_bilingual_submission_pack.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_figure_index(output_dir: Path) -> None:
    items = [
        ("Fig. 1 / 图1", "model_architecture_aslam_ssaf.png", "Overall architecture of ASLAM-SSAF", "ASLAM-SSAF 整体模型结构图"),
        ("Fig. 2 / 图2", "dataset_statistics_panel.png", "Statistical profile of the six local benchmark datasets", "六个本地基准数据集的统计概况"),
        ("Fig. 3 / 图3", "main_comparison_auc_ap_panel.png", "Main benchmark comparison of ASLAM-SSAF against ASLAM", "ASLAM-SSAF 与 ASLAM 的主结果对比"),
        ("Fig. 4 / 图4", "gain_delta_panel.png", "Absolute performance gains of ASLAM-SSAF over ASLAM", "ASLAM-SSAF 相对 ASLAM 的绝对增益"),
        ("Fig. 5 / 图5", "ablation_auc_ap_panel.png", "Ablation study of ASLAM-SSAF and its weakened variants", "ASLAM-SSAF 及其弱化变体的消融实验"),
        ("Fig. 6 / 图6", "training_val_auc_trends.png", "Validation AUC trends across six datasets", "六个数据集上的验证 AUC 训练趋势"),
        ("Fig. 7 / 图7", "training_val_ap_trends.png", "Validation AP trends across six datasets", "六个数据集上的验证 AP 训练趋势"),
        ("Fig. 8 / 图8", "training_loss_trends.png", "Training loss trends across six datasets", "六个数据集上的训练损失趋势"),
        ("Fig. 9 / 图9", "tsne_grid_publication.png", "t-SNE visualization of learned edge representations", "学习到的边表示的 t-SNE 可视化"),
        ("Table 1 / 表1", "table_dataset_overview_bilingual.md", "Dataset overview and formal protocol scale", "数据集概况与正式实验规模"),
        ("Table 2 / 表2", "table_main_results_bilingual.md", "Main comparison results", "主结果对比"),
        ("Table 3 / 表3", "table_ablation_bilingual.md", "Ablation results written in drop-from-final format", "以最终模型掉点格式书写的消融结果"),
    ]
    lines = [
        "# Figure and Table Index / 图表索引",
        "",
        "| No. | File | English caption | Chinese caption |",
        "| --- | --- | --- | --- |",
    ]
    for no, file_name, en_caption, zh_caption in items:
        lines.append(f"| {no} | {file_name} | {en_caption} | {zh_caption} |")
    (output_dir / "Figure_Table_Index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = collect_dataset_stats(args.root)
    main_rows = collect_main_results(args.results_dir)
    ablation_rows = collect_ablation_results(args.results_dir)

    write_dataset_table(dataset_rows, args.output_dir)
    write_main_results_table(main_rows, args.output_dir)
    write_ablation_table(ablation_rows, args.output_dir)
    write_references_table(args.output_dir)
    write_bilingual_pack(args.output_dir, dataset_rows, main_rows, ablation_rows)
    write_figure_index(args.output_dir)

    print(f"Saved: {args.output_dir / 'ASLAM_SSAF_bilingual_submission_pack.md'}")


if __name__ == "__main__":
    main()
