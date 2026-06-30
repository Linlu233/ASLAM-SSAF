# ASLAM-SSMAttn Formal Report

## 1. Scope and boundary

This report follows the user's final constraint:

1. `G:\myProject\ASLAM-3\ASLAM` is baseline-only and is used to freeze the local formal baseline.
2. All new comparative changes are implemented only in the root codebase `G:\myProject\ASLAM-3`.
3. The provided ASLAM paper is treated as the primary reference for the original method's reported results.

Academic boundary:

1. The local datasets do not match the paper's Table 1 statistics, so the paper table cannot be claimed as strictly reproduced on this machine.
2. Therefore, the academically correct comparison chain is:
   - paper-reported ASLAM results from the provided paper,
   - frozen local baseline in `ASLAM/code`,
   - root-code formal comparison between `aslam` and `aslam_ssmattn` under the same root training protocol.

## 2. Paper-reported reference values

From the provided paper text `G:\myProject\ASLAM-3\papers\aslam_paper.txt`:

1. Table 2 reports mean AUC over 10 runs.
2. Table 3 reports mean AP over 10 runs.

Relevant paper-reported ASLAM values:

| Dataset | AUC (paper) | AP (paper) |
| --- | ---: | ---: |
| Photo | 99.14 | 99.01 |
| Citeseer | 94.71 | 94.97 |
| PubMed | 97.73 | 97.74 |
| DBLP | 96.68 | 97.15 |

These values are reference targets from the paper, not local rerun results.

## 3. Frozen local formal baseline

Frozen baseline summary file:

`G:\myProject\ASLAM-3\ASLAM\results\local_formal_baseline_summary.md`

Frozen local baseline means:

| Dataset | AUC mean | AP mean |
| --- | ---: | ---: |
| Citeseer | 0.9676 | 0.9655 |
| PubMed | 0.9952 | 0.9943 |
| DBLP | 0.9409 | 0.9478 |
| Photo | 0.9727 | 0.9796 |

This baseline was reproduced with the recovered author-style code under the frozen local protocol:

1. split `85/5/10`
2. seed `2`
3. runs `10`
4. epochs `401`
5. patience `20`
6. batch size `32`

Coverage note:

1. The frozen `ASLAM/code` baseline currently covers only `Citeseer`, `PubMed`, `DBLP`, and `Photo`.
2. `CoRA` and `Twitch_EN` were added only to the root rebuilt framework, so they do not have an author-style frozen local baseline in this workspace.

## 4. Root-code modifications

Modified root files:

1. `G:\myProject\ASLAM-3\models\aslam.py`
2. `G:\myProject\ASLAM-3\train.py`
3. `G:\myProject\ASLAM-3\config.py`

New formal improved model:

1. `ASLAMSSMAttn`

Design principle:

1. Keep the ASLAM-style attribute/structure dual-branch scoring path.
2. Keep the stronger multi-scale structural backbone already present in the root code.
3. Add a new cross-branch pair fusion head that combines selective state-space token mixing and pairwise attention.

## 5. Formal root comparison

Formal root comparison command pattern:

```bash
conda run -n ASLAM_THC python train.py --datasets <dataset> --model compare --runs 10 --epochs 401 --patience 20 --hidden_channels 64 --batch_size <batch_size> --device cuda:0 --seed 2
```

Important disclosure:

1. In the root rebuilt framework, the formal comparison kept `runs=10`, `epochs=401`, `patience=20`, and `seed=2`.
2. The root framework used dataset-specific stable edge batch sizes instead of the paper-style `32`:
   - `Citeseer`, `DBLP`, `PubMed`, `amz_Photo`: `16384`
   - `Twitch_EN`: `8192`
   - `CoRA`: `4096`
3. This deviation must be disclosed in any write-up because the root framework is a rebuilt edge-batch pipeline rather than the recovered original author pipeline.

Formal root result files:

1. `G:\myProject\ASLAM-3\results\formal_citeseer_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
2. `G:\myProject\ASLAM-3\results\formal_dblp_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
3. `G:\myProject\ASLAM-3\results\formal_pubmed_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
4. `G:\myProject\ASLAM-3\results\formal_amz_photo_compare_runs10_epochs401_pat20_seed2_h64_b16384.json`
5. `G:\myProject\ASLAM-3\results\formal_cora_compare_runs10_epochs401_pat20_seed2_h64_b4096.json`
6. `G:\myProject\ASLAM-3\results\formal_twitch_en_compare_runs10_epochs401_pat20_seed2_h64_b8192.json`

Formal mean results:

| Dataset | Batch size | ASLAM AUC | ASLAM AP | ASLAM-SSMAttn AUC | ASLAM-SSMAttn AP | AUC gain | AP gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 16384 | 0.8469 | 0.8540 | 0.8590 | 0.8631 | +0.0121 | +0.0091 |
| DBLP | 16384 | 0.9013 | 0.9085 | 0.9157 | 0.9262 | +0.0144 | +0.0177 |
| PubMed | 16384 | 0.9767 | 0.9759 | 0.9787 | 0.9775 | +0.0020 | +0.0016 |
| amz_Photo | 16384 | 0.9810 | 0.9806 | 0.9857 | 0.9863 | +0.0047 | +0.0057 |
| CoRA | 4096 | 0.9474 | 0.9544 | 0.9535 | 0.9622 | +0.0061 | +0.0078 |
| Twitch_EN | 8192 | 0.9281 | 0.9332 | 0.9290 | 0.9341 | +0.0009 | +0.0009 |

Conclusion:

1. Under the root formal comparison protocol, `aslam_ssmattn` improves over root `aslam` on all six currently supported local datasets in both AUC and AP.
2. Therefore, the root-code improved model satisfies the "comprehensive improvement over baseline ASLAM" requirement for the local root comparison setting.

## 6. Model innovations

### 6.1 Innovation 1: cross-branch selective state-space pair fusion

Instead of using only a fixed fusion MLP on concatenated branch features, the new model constructs pair tokens from:

1. attribute endpoint embeddings,
2. structure endpoint embeddings,
3. branch-specific pair summaries,
4. cross-branch discrepancy token,
5. cross-branch mean token,
6. learnable heuristic token.

These tokens are mixed by stacked selective state-space plus attention blocks before edge scoring.

Literature grounding:

1. selective state-space graph modeling is inspired by Graph Mamba and MbaGCN;
2. adaptive pairwise attention for link prediction is grounded by LPFormer.

### 6.2 Innovation 2: adaptive expert gate at edge level

The final edge logit is not fused with a global scalar. Instead, each edge learns its own gate:

1. keep a stable base fusion score,
2. produce an SSM-attention expert score,
3. adaptively interpolate between them per edge.

This is a safer design than replacing the baseline score directly, and it reduced degradation risk on dense datasets such as `amz_Photo`.

### 6.3 Innovation 3: learnable heuristic token as positional context

Classical link heuristics are not only appended as numeric features. They are projected into a learnable token and fused jointly with branch tokens.

This follows the recent trend that learnable positional or structural encodings can improve link prediction when integrated into the representation path rather than used only as post-hoc features.

## 7. Architecture summary

### 7.1 Node encoders

1. Attribute branch:
   - MLP feature encoder
2. Structure branch:
   - residual GCN encoder
   - local-global graph encoder
   - bidirectional selective mixer on multi-scale node states
   - node fusion MLP

### 7.2 Pair fusion head

1. Base path:
   - branch pair composition
   - baseline fusion gate
   - baseline fusion score
2. Enhanced path:
   - pairwise token attention
   - selective state-space pair fusion
   - residual fusion scorer
3. Final output:
   - edge-wise adaptive interpolation between base and enhanced experts

## 8. Core formulas

### 8.1 Multi-scale structural representation

\[
h^{(0)} = W_x X
\]

\[
h_{\text{local}}^{(l)} = \mathrm{GAT}(h^{(l-1)}, A), \quad
h_{\text{global}}^{(l)} = \mathrm{GCN}(h^{(l-1)}, A)
\]

\[
h^{(l)} = \mathrm{Dropout}\left(\frac{h_{\text{local}}^{(l)} + h_{\text{global}}^{(l)}}{2}\right) + h^{(l-1)}
\]

The node scale sequence is:

\[
S_i = [h_i^{(0)}, h_i^{(1)}, \dots, h_i^{(L)}]
\]

### 8.2 Selective state-space mixing

For token \(s_t \in S_i\):

\[
g_t = \sigma(W_g s_t), \quad \tilde{s}_t = \tanh(W_c s_t)
\]

\[
m_t^{f} = g_t \odot \tilde{s}_t + (1-g_t)\odot m_{t-1}^{f}
\]

\[
m_t^{b} = \text{reverse-scan}(s_t)
\]

\[
\hat{s}_t = W_o [s_t \, \| \, m_t^{f} \, \| \, m_t^{b}]
\]

### 8.3 Pair token construction

For edge \((u,v)\):

\[
p_{uv}^{A} = [h_u^{A} \, \| \, h_v^{A} \, \| \, |h_u^{A}-h_v^{A}| \, \| \, h_u^{A}\odot h_v^{A} \, \| \, e_{uv}]
\]

\[
p_{uv}^{S} = [h_u^{S} \, \| \, h_v^{S} \, \| \, |h_u^{S}-h_v^{S}| \, \| \, h_u^{S}\odot h_v^{S} \, \| \, e_{uv}]
\]

\[
T_{uv} = [h_u^{A}, h_v^{A}, h_u^{S}, h_v^{S}, \phi_A(p_{uv}^{A}), \phi_S(p_{uv}^{S}), \phi_\Delta, \phi_\mu, \phi_e(e_{uv})]
\]

### 8.4 Adaptive final fusion

\[
\ell_{\text{base}} = f_{\text{base}}(p_{uv}^{A}, p_{uv}^{S})
\]

\[
\ell_{\text{expert}} = \frac{\ell_{\text{attn}} + \ell_{\text{ssm}} + \ell_{\text{res}}}{3}
\]

\[
\alpha_{uv} = \sigma(g([p_{uv}^{A}, p_{uv}^{S}, c_{uv}, b_{uv}]))
\]

\[
\ell_{uv} = (1-\alpha_{uv})\ell_{\text{base}} + \alpha_{uv}\ell_{\text{expert}}
\]

## 9. Academic compliance judgment

This work is academically compliant only under the following wording:

1. The provided ASLAM paper is the primary literature reference.
2. `ASLAM/code` is the frozen local baseline reproduction.
3. `ASLAMSSMAttn` is a root-code improved model evaluated under the root rebuilt pipeline.
4. Because the local datasets do not match the paper's Table 1 statistics, these experiments must not be claimed as a strict reproduction of the paper's reported table.
5. Because the root rebuilt framework uses a different batch regime, the root formal results must be reported as a fair within-framework comparison, not as a byte-level reimplementation of the original released pipeline.

Non-compliant claims to avoid:

1. "The paper table was exactly reproduced locally."
2. "The new root model directly surpassed the official paper numbers under the same data and code conditions."
3. Any citation of unverified or non-primary literature as if it were a confirmed source.

## 10. APA 7 references

Behrouz, A., & Hashemi, F. (2024). Graph Mamba: Towards learning on graphs with state space models. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 119-130). Association for Computing Machinery. https://doi.org/10.1145/3637528.3672044

He, X., Wang, Y., Fan, W., Shen, X., Juan, X., Miao, R., & Wang, X. (2025). Mamba-based graph convolutional networks: Tackling over-smoothing with selective state space. In *Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence* (pp. 5345-5353). IJCAI. https://doi.org/10.24963/ijcai.2025/595

Nie, R., Wang, G., Liu, Q., & Peng, C. (2025). Link prediction for attribute and structure learning based on attention mechanism. *Applied Soft Computing, 179*, Article 113268. https://doi.org/10.1016/j.asoc.2025.113268

Shomer, H., Ma, Y., Mao, H., Li, J., Wu, B., & Tang, J. (2024). LPFormer: An adaptive graph transformer for link prediction. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2686-2698). Association for Computing Machinery. https://doi.org/10.1145/3637528.3672025

Tieu, K., Fu, D., Li, Z., Maciejewski, R., & He, J. (2025). Learnable spatial-temporal positional encoding for link prediction. In *Proceedings of the 42nd International Conference on Machine Learning* (pp. 59570-59597). PMLR. https://proceedings.mlr.press/v267/tieu25a.html

## 11. Recording rule for PubMed and cross-dataset alignment

Under the current root formal comparison, all six supported root datasets improve over the root `aslam` baseline:

| Dataset | Root `aslam` AUC | Root `aslam` AP | Root `aslam_ssmattn` AUC | Root `aslam_ssmattn` AP | AUC gain | AP gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Citeseer | 0.8469 | 0.8540 | 0.8590 | 0.8631 | +0.0121 | +0.0091 |
| DBLP | 0.9013 | 0.9085 | 0.9157 | 0.9262 | +0.0144 | +0.0177 |
| PubMed | 0.9767 | 0.9759 | 0.9787 | 0.9775 | +0.0020 | +0.0016 |
| amz_Photo | 0.9810 | 0.9806 | 0.9857 | 0.9863 | +0.0047 | +0.0057 |
| CoRA | 0.9474 | 0.9544 | 0.9535 | 0.9622 | +0.0061 | +0.0078 |
| Twitch_EN | 0.9281 | 0.9332 | 0.9290 | 0.9341 | +0.0009 | +0.0009 |

Therefore, if PubMed is recorded in the paper-facing summary, the academically compliant wording is:

1. keep the paper-reported ASLAM PubMed result as the literature reference value: `AUC 97.73 +/- 0.02`, `AP 97.74 +/- 0.01`;
2. keep the frozen local recovered baseline `0.9952 / 0.9943` only as the `ASLAM/code` local formal baseline under the mismatched local dataset statistics;
3. keep the root rebuilt framework result `0.9767 / 0.9759` vs `0.9787 / 0.9775` only as the within-framework formal comparison.

These three numbers answer different questions and must not be merged into one column:

1. paper value: what the published article reported;
2. recovered local baseline: what the original-style code achieved on the current local files;
3. rebuilt root comparison: whether the new model improves over the rebuilt root baseline.

For `CoRA` and `Twitch_EN`, the academically compliant wording is simpler:

1. there is currently no paper table value from the provided ASLAM paper for these datasets;
2. there is currently no frozen `ASLAM/code` local baseline for these datasets in this workspace;
3. therefore, only the root within-framework formal comparison should be reported for them.
