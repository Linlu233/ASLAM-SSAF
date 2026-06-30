from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_sort_pool


class DGCNN(nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers, k=0.6):
        super().__init__()
        if k < 1:
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = nn.Linear(dense_dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)
        x = F.gelu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.gelu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.lin1(x)


class SelectiveStateSpaceBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 4)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.a_log = nn.Parameter(torch.zeros(dim))
        self.skip = nn.Parameter(torch.ones(dim))

    def _scan(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        tokens = x.flip(1) if reverse else x
        batch_size, seq_len, dim = tokens.shape
        state = torch.zeros(batch_size, dim, device=tokens.device, dtype=tokens.dtype)
        a = -F.softplus(self.a_log).view(1, dim)
        outputs = []
        for idx in range(seq_len):
            u, delta, b_t, c_t = self.in_proj(tokens[:, idx]).chunk(4, dim=-1)
            delta = F.softplus(delta)
            decay = torch.exp(delta * a)
            state = decay * state + delta * torch.tanh(b_t) * u
            y = torch.tanh(c_t) * state + self.skip * u
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        return y.flip(1) if reverse else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        mixed = 0.5 * (self._scan(x, reverse=False) + self._scan(x, reverse=True))
        return residual + self.dropout(self.out_proj(mixed))


class ASLAMSSMAttn(nn.Module):
    def __init__(
        self,
        train_dataset,
        train_dataset_label2,
        in_channels,
        in_channels_label2,
        hidden_channels,
        out_channels,
        num_layers,
        semantic_dim: int,
        attr_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        del attr_dim
        self.dgcnn = DGCNN(train_dataset, in_channels, hidden_channels, 16, num_layers, k=0.6)
        self.dgcnn_2 = DGCNN(train_dataset_label2, in_channels_label2, hidden_channels, 16, num_layers, k=0.6)
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, out_channels),
            nn.GELU(),
        )
        self.degree_proj = nn.Sequential(
            nn.Linear(4, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.token_embedding = nn.Embedding(5, out_channels)
        self.ssm_blocks = nn.ModuleList(
            [SelectiveStateSpaceBlock(out_channels, dropout) for _ in range(2)]
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(out_channels, 1)
        self.reliability = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, 1),
        )
        self.lin1 = nn.Linear(out_channels, 1)
        self.lin2 = nn.Linear(out_channels, 1)
        self.lin3 = nn.Linear(out_channels, 1)

    def forward(self, data, data_label2, knn_graph):
        emb_label1 = self.dgcnn(data)
        emb_label2 = self.dgcnn_2(data_label2)
        structure = torch.cat([emb_label1, emb_label2], dim=-1)

        semantic_pair = knn_graph.x[data.target_nodes].view(-1, 2, knn_graph.x.size(-1))
        semantic = self.semantic_proj(semantic_pair.sum(dim=1))
        interaction = structure * semantic
        residual = torch.abs(structure - semantic)

        degree = torch.log1p(data.pair_degree.view(-1, 2))
        degree_token = self.degree_proj(
            torch.cat([degree, degree.sum(dim=-1, keepdim=True), (degree[:, :1] - degree[:, 1:]).abs()], dim=-1)
        )

        tokens = torch.stack([structure, semantic, interaction, residual, degree_token], dim=1)
        token_ids = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        tokens = tokens + self.token_embedding(token_ids)

        for block in self.ssm_blocks:
            tokens = block(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.attn_norm(tokens + self.dropout(attn_out))
        token_scores = torch.softmax(self.score(torch.tanh(tokens)).squeeze(-1), dim=1)
        fused_token = torch.sum(tokens * token_scores.unsqueeze(-1), dim=1)

        alpha = torch.sigmoid(self.reliability(torch.cat([structure, degree_token], dim=-1)))
        fused = alpha * fused_token + (1.0 - alpha) * (0.5 * (structure + semantic))

        output_structure = self.lin1(structure)
        output_semantic = self.lin2(semantic)
        output_fused = self.lin3(fused)
        return output_structure, output_semantic, output_fused
