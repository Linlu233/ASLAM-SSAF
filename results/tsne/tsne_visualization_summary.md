# t-SNE Visualization Summary

Model: `aslam_ssmattn`

This summary follows the paper-style visualization protocol: the learned edge representations are projected by t-SNE, and positive/negative test edges are rendered with different colors.

| Dataset | Best epoch | Test AUC | Test AP | Sample ratio | Sampled edges | Figure | Coordinates |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Citeseer | 43 | 0.8665 | 0.8702 | 0.15 | 136 | `results/tsne/Citeseer_aslam_ssmattn_edge_tsne.png` | `results/tsne/Citeseer_aslam_ssmattn_edge_tsne_coords.npz` |
| DBLP | 37 | 0.9308 | 0.9297 | 0.15 | 106 | `results/tsne/DBLP_aslam_ssmattn_edge_tsne.png` | `results/tsne/DBLP_aslam_ssmattn_edge_tsne_coords.npz` |
| PubMed | 29 | 0.9793 | 0.9773 | 0.15 | 1330 | `results/tsne/PubMed_aslam_ssmattn_edge_tsne.png` | `results/tsne/PubMed_aslam_ssmattn_edge_tsne_coords.npz` |
| amz_Photo | 5 | 0.9864 | 0.9881 | 0.15 | 2456 | `results/tsne/amz_Photo_aslam_ssmattn_edge_tsne.png` | `results/tsne/amz_Photo_aslam_ssmattn_edge_tsne_coords.npz` |
| CoRA | 19 | 0.9475 | 0.9592 | 0.15 | 944 | `results/tsne/CoRA_aslam_ssmattn_edge_tsne.png` | `results/tsne/CoRA_aslam_ssmattn_edge_tsne_coords.npz` |
| Twitch_EN | 19 | 0.9291 | 0.9373 | 0.15 | 1060 | `results/tsne/Twitch_EN_aslam_ssmattn_edge_tsne.png` | `results/tsne/Twitch_EN_aslam_ssmattn_edge_tsne_coords.npz` |
