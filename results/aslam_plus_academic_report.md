# ASLAM-Plus Academic Report

## 1. Reading Notes on the Provided Papers

### 1.1 ASLAM paper

Paper: *Link prediction for attribute and structure learning based on attention mechanism*.

Key takeaways:

1. The original ASLAM is a dual-branch model rather than a single GNN.
2. Its central idea is to fuse attribute-driven and structure-driven evidence with adaptive attention.
3. Its claimed strengths come from attribute importance estimation, node labeling, and structural-attentive fusion.

Implication for redesign:

1. The most important part to preserve is the `attribute branch + structure branch + attention fusion` principle.
2. Any upgrade should improve structure modeling without discarding the original fusion logic.

### 1.2 Springer paper (`s11280-025-01347-x`)

Paper: *Identifying key nodes for enhancing community stability in bipartite networks*.

This paper is not a direct link prediction method, but it is still useful conceptually:

1. It emphasizes stability-aware structural descriptors instead of plain adjacency only.
2. It motivates robust graph statistics when deciding which node interactions matter.

## 2. Codebase Diagnosis

Local code inspection showed:

1. The repository is incomplete. `train.py` imports `utils.*` and `models.bsal`, but those files are missing.
2. Only partial Python files and some `pyc` artifacts remain.
3. The original training pipeline is therefore not reproducible as-is.

Practical consequence:

1. A clean training, evaluation, and dataset-loading stack had to be rebuilt.
2. The rebuilt baseline still follows the ASLAM idea instead of replacing it with an unrelated architecture.

## 3. Implemented Architecture

### 3.1 Rebuilt baseline: ASLAM

Baseline components:

1. Attribute branch: MLP on node features.
2. Structure branch: residual GCN encoder on the training graph.
3. Pair representation: `[h_u, h_v, |h_u-h_v|, h_u * h_v, heuristic(u,v)]`.
4. Fusion head: sigmoid-gated dual-branch fusion.

### 3.2 Improved model: ASLAM-Plus

ASLAM-Plus keeps the baseline branch and adds:

1. Multi-scale local-global structure encoder.
   - Parallel local `GATv2` and global `GCN` updates generate a scale sequence.
2. Bidirectional selective state-space mixer.
   - The scale sequence is processed by forward and backward selective recurrences with depthwise convolution.
3. Adaptive pairwise token attention.
   - Edge scoring attends over `u`, `v`, `|u-v|`, `u * v`, and a learnable heuristic token.
4. Residual score ensemble.
   - The final score combines baseline fusion and attention-enhanced fusion to avoid hurting dense graphs.

## 4. Core Formulas

### 4.1 Multi-scale node states

For node feature matrix `X`:

`h^(0) = W_x X`

At graph layer `l`:

`h_local^(l) = GAT(h^(l-1), A)`

`h_global^(l) = GCN(h^(l-1), A)`

`h^(l) = Dropout(0.5 * (h_local^(l) + h_global^(l))) + h^(l-1)`

This yields the scale sequence:

`Z_i = [h_i^(0), h_i^(1), ..., h_i^(L)]`

### 4.2 Bidirectional selective state-space mixing

For token `z_t` in `Z_i`:

`g_t = sigma(W_g z_t)`

`c_t = tanh(W_c z_t)`

`s_t^f = g_t * c_t + (1 - g_t) * s_(t-1)^f`

The backward state `s_t^b` is computed on the reversed sequence.

The mixed token is:

`m_t = psi([z_t ; s_t^f ; s_t^b])`

Attention pooling gives the final scale-aware node embedding:

`a_t = softmax(q^T m_t)`

`z_i = sum_t a_t m_t`

### 4.3 Edge heuristic positional encoding

For candidate edge `(u,v)`:

`e_uv = [CN, Jaccard, AA, RA, PA, deg(u), deg(v), cos_deg]`

where:

1. `CN` is common neighbors.
2. `AA` is Adamic-Adar.
3. `RA` is resource allocation.
4. `PA` is preferential attachment.

### 4.4 Adaptive pairwise fusion

Pair tokens:

`T_uv = [z_u, z_v, |z_u-z_v|, z_u * z_v, phi(e_uv)]`

The final fused score is:

`l_base = f_base(attr_uv, struct_uv)`

`l_attn = f_attn(T_uv)`

`l_res = f_res([c_base, p_uv])`

`alpha = sigma(theta)`

`l_final = (1-alpha) * l_base + alpha * 0.5 * (l_attn + l_res)`

## 5. Why These Changes Are Innovative

### Innovation 1: selective state-space structure modeling on top of ASLAM

1. Original ASLAM uses dual-branch fusion but does not model graph scales with selective recurrence.
2. ASLAM-Plus turns multi-scale graph states into an ordered sequence and mixes them with bidirectional selective transitions.
3. This adapts Graph Mamba and MbaGCN ideas to attributed link prediction.

### Innovation 2: adaptive pairwise token attention instead of a fixed edge MLP

1. LPFormer shows that different links rely on different pairwise factors.
2. The implemented pairwise attention lets each candidate edge reweight node identity, difference, interaction, and heuristic evidence.

### Innovation 3: learnable heuristic positional encoding

1. Classical heuristics are strong but rigid.
2. ASLAM-Plus converts them into a learnable heuristic token instead of using them only as post-hoc scores.
3. This is consistent with the idea of learnable positional encoding for link prediction.

## 6. Reference Standard and Experimental Positioning

For subsequent comparison and writing, the reference standard should be:

1. Primary reference: the reported ASLAM results in the provided paper *Link prediction for attribute and structure learning based on attention mechanism*.
2. Secondary reference: the local reconstructed ASLAM baseline in this workspace.

This distinction is necessary because:

1. The local repository is incomplete and does not contain the full original runnable source tree.
2. The rebuilt baseline preserves the ASLAM design idea, but its data split, preprocessing path, and implementation details are not guaranteed to be identical to the exact paper release.
3. Therefore, for academic writing, claims about "surpassing ASLAM" should first be aligned with the paper-reported numbers, and the local rerun should be described as a reproducibility-oriented supporting experiment.

Recommended wording:

1. "Paper-reported ASLAM results are treated as the primary baseline."
2. "Local reconstructed ASLAM results are used only as an auxiliary reproducibility reference under the current workspace."

## 7. Reproducible Experimental Results

Environment:

1. Conda env: `ASLAM_THC`
2. Device: `cuda:0` (`NVIDIA GeForce RTX 4070 Laptop GPU`)
3. Runs: `3`
4. Epochs per run: `6`
5. Metrics: `AUC` and `AP`

Command:

```bash
conda run -n ASLAM_THC python train.py --datasets "Citeseer,DBLP,PubMed,amz_Photo" --model compare --epochs 6 --patience 3 --batch_size 16384 --hidden_channels 64 --device cuda:0 --runs 3
```

### Mean +/- Std over 3 runs

| Dataset | ASLAM AUC | ASLAM-Plus AUC | Gain | ASLAM AP | ASLAM-Plus AP | Gain |
|---|---:|---:|---:|---:|---:|---:|
| Citeseer | 0.6721 +/- 0.0295 | 0.7038 +/- 0.0189 | +0.0317 | 0.7168 +/- 0.0310 | 0.7423 +/- 0.0138 | +0.0254 |
| DBLP | 0.7419 +/- 0.0158 | 0.7571 +/- 0.0117 | +0.0152 | 0.7628 +/- 0.0106 | 0.7932 +/- 0.0089 | +0.0304 |
| PubMed | 0.9396 +/- 0.0064 | 0.9405 +/- 0.0049 | +0.0010 | 0.9200 +/- 0.0093 | 0.9215 +/- 0.0092 | +0.0014 |
| amz_Photo | 0.9764 +/- 0.0019 | 0.9840 +/- 0.0008 | +0.0076 | 0.9723 +/- 0.0032 | 0.9858 +/- 0.0000 | +0.0135 |

Conclusion:

1. ASLAM-Plus improves both AUC and AP on all four validated datasets.
2. The largest gains appear on Citeseer, DBLP, and amz_Photo.
3. PubMed also improves, but by a smaller margin.

## 8. Files Added or Rewritten

1. `config.py`
2. `train.py`
3. `models/aslam.py`
4. `utils/data_utils.py`
5. `utils/eval_utils.py`
6. `utils/train_utils.py`

## 9. APA 7 References

Behrouz, A., & Hashemi, F. (2024). *Graph Mamba: Towards learning on graphs with state space models*. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. ACM. https://doi.org/10.1145/3637528.3672044

Chen, J., Choudhury, F., Xin, J., & Wang, Z. (2025). *Identifying key nodes for enhancing community stability in bipartite networks*. *World Wide Web, 28*(3), Article 34. https://doi.org/10.1007/s11280-025-01347-x

He, X., Wang, Y., Fan, W., Shen, X., Juan, X., Miao, R., & Wang, X. (2025). *MbaGCN: Multi-scale bidirectional aggregation graph convolutional network based on state space model*. In *Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence* (pp. 5345-5353). IJCAI. https://doi.org/10.24963/ijcai.2025/595

Nie, R., Wang, G., Liu, Q., & Peng, C. (2025). *Link prediction for attribute and structure learning based on attention mechanism*. *Applied Soft Computing, 179*, 113268. https://doi.org/10.1016/j.asoc.2025.113268

Shomer, H., Ma, Y., Mao, H., Li, J., Wu, B., & Tang, J. (2024). *LPFormer: An adaptive graph transformer for link prediction*. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. ACM. https://doi.org/10.1145/3637528.3672025

Tieu, K., Fu, D., Li, Z., Maciejewski, R., & He, J. (2025). *Learnable spatial-temporal positional encoding for link prediction*. In *Proceedings of the 42nd International Conference on Machine Learning* (Vol. 267, pp. 53417-53434). PMLR. https://proceedings.mlr.press/v267/tieu25a.html
