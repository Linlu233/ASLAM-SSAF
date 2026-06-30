from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv


def pairwise_compose(node_repr: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_label_index
    lhs = node_repr[src]
    rhs = node_repr[dst]
    return torch.cat([lhs, rhs, torch.abs(lhs - rhs), lhs * rhs], dim=-1)


class MLP(nn.Module):
    def __init__(self, dims, dropout: float = 0.0):
        super().__init__()
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx != len(dims) - 2:
                layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualGCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.layers, self.norms):
            residual = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + residual
        return h


class LocalGlobalGraphEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.local_layers = nn.ModuleList()
        self.local_norms = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.local_layers.append(
                GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, dropout=dropout)
            )
            self.local_norms.append(nn.LayerNorm(hidden_channels))
            self.global_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.global_norms.append(nn.LayerNorm(hidden_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.input_proj(x)
        multi_scale = [h]
        for gat, gat_norm, gcn, gcn_norm in zip(
            self.local_layers, self.local_norms, self.global_layers, self.global_norms
        ):
            local = gat(h, edge_index)
            local = gat_norm(local)
            local = F.gelu(local)
            global_h = gcn(h, edge_index)
            global_h = gcn_norm(global_h)
            global_h = F.gelu(global_h)
            h = 0.5 * (local + global_h) + h
            h = F.dropout(h, p=self.dropout, training=self.training)
            multi_scale.append(h)
        return h, multi_scale


class BiDirectionalSelectiveMixer(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_channels)
        self.depthwise_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.in_proj = nn.Linear(hidden_channels, hidden_channels)
        self.gate_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels * 3, hidden_channels)
        self.attn_query = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def _scan(self, sequence: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(sequence.size(0), sequence.size(2), device=sequence.device)
        outputs = []
        for t in range(sequence.size(1)):
            token = sequence[:, t, :]
            gate = torch.sigmoid(self.gate_proj(token))
            candidate = torch.tanh(self.in_proj(token))
            state = gate * candidate + (1.0 - gate) * state
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(self, multi_scale: list[torch.Tensor]) -> torch.Tensor:
        sequence = torch.stack(multi_scale, dim=1)
        sequence = self.pre_norm(sequence)
        conv_out = self.depthwise_conv(sequence.transpose(1, 2)).transpose(1, 2)
        sequence = sequence + conv_out
        forward = self._scan(sequence)
        backward = torch.flip(self._scan(torch.flip(sequence, dims=[1])), dims=[1])
        mixed = torch.cat([sequence, forward, backward], dim=-1)
        mixed = self.out_proj(mixed)
        mixed = F.gelu(mixed)
        mixed = F.dropout(mixed, p=self.dropout, training=self.training)
        weights = torch.softmax(self.attn_query(mixed).squeeze(-1), dim=-1)
        return torch.sum(mixed * weights.unsqueeze(-1), dim=1)


class PairwiseTokenAttention(nn.Module):
    def __init__(self, hidden_channels: int, heuristic_dim: int, heads: int, dropout: float):
        super().__init__()
        self.heuristic_proj = MLP([heuristic_dim, hidden_channels, hidden_channels], dropout=dropout)
        self.token_attn = nn.MultiheadAttention(
            hidden_channels, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.score_head = MLP([hidden_channels, hidden_channels, 1], dropout=dropout)

    def forward(self, node_repr: torch.Tensor, edge_label_index: torch.Tensor, heuristics: torch.Tensor):
        src, dst = edge_label_index
        lhs = node_repr[src]
        rhs = node_repr[dst]
        heuristic_token = self.heuristic_proj(heuristics)
        tokens = torch.stack([lhs, rhs, torch.abs(lhs - rhs), lhs * rhs, heuristic_token], dim=1)
        attended, _ = self.token_attn(tokens, tokens, tokens, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.score_head(pooled).view(-1), pooled


class SelectiveStateSpaceAttentionBlock(nn.Module):
    def __init__(self, hidden_channels: int, heads: int, dropout: float):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_channels)
        self.depthwise_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.in_proj = nn.Linear(hidden_channels, hidden_channels)
        self.gate_proj = nn.Linear(hidden_channels, hidden_channels)
        self.ssm_out = nn.Linear(hidden_channels * 3, hidden_channels)
        self.attn = nn.MultiheadAttention(
            hidden_channels, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(hidden_channels)
        self.ff = MLP([hidden_channels, hidden_channels * 2, hidden_channels], dropout=dropout)
        self.dropout = dropout

    def _scan(self, sequence: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(sequence.size(0), sequence.size(2), device=sequence.device)
        outputs = []
        for token in sequence.unbind(dim=1):
            gate = torch.sigmoid(self.gate_proj(token))
            candidate = torch.tanh(self.in_proj(token))
            state = gate * candidate + (1.0 - gate) * state
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        tokens = self.pre_norm(tokens)
        conv_out = self.depthwise_conv(tokens.transpose(1, 2)).transpose(1, 2)
        mixed_tokens = tokens + conv_out
        forward_state = self._scan(mixed_tokens)
        backward_state = torch.flip(self._scan(torch.flip(mixed_tokens, dims=[1])), dims=[1])
        ssm_out = self.ssm_out(torch.cat([mixed_tokens, forward_state, backward_state], dim=-1))
        attn_out, _ = self.attn(mixed_tokens, mixed_tokens, mixed_tokens, need_weights=False)
        tokens = residual + F.dropout(ssm_out + attn_out, p=self.dropout, training=self.training)
        tokens = tokens + F.dropout(self.ff(self.ff_norm(tokens)), p=self.dropout, training=self.training)
        return tokens


class SSMAttentionPairFusion(nn.Module):
    def __init__(self, hidden_channels: int, heuristic_dim: int, dropout: float):
        super().__init__()
        heads = 4 if hidden_channels % 4 == 0 else 2
        pair_dim = hidden_channels * 4 + heuristic_dim
        self.attr_pair_proj = MLP([pair_dim, hidden_channels, hidden_channels], dropout=dropout)
        self.struct_pair_proj = MLP([pair_dim, hidden_channels, hidden_channels], dropout=dropout)
        self.heuristic_proj = MLP([heuristic_dim, hidden_channels, hidden_channels], dropout=dropout)
        self.cross_delta_proj = MLP([hidden_channels * 2, hidden_channels, hidden_channels], dropout=dropout)
        self.cross_mean_proj = MLP([hidden_channels * 2, hidden_channels, hidden_channels], dropout=dropout)
        self.blocks = nn.ModuleList(
            [SelectiveStateSpaceAttentionBlock(hidden_channels, heads=heads, dropout=dropout) for _ in range(2)]
        )
        self.token_score = nn.Linear(hidden_channels, 1)
        self.refine = MLP([hidden_channels * 3, hidden_channels, hidden_channels], dropout=dropout)
        self.score_head = MLP([hidden_channels, hidden_channels, 1], dropout=dropout)
        self.edge_gate = MLP([hidden_channels * 4, hidden_channels, 1], dropout=dropout)

    def forward(
        self,
        attr_repr: torch.Tensor,
        struct_repr: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
        base_fused: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_label_index
        attr_u = attr_repr[src]
        attr_v = attr_repr[dst]
        struct_u = struct_repr[src]
        struct_v = struct_repr[dst]

        attr_pair = torch.cat([pairwise_compose(attr_repr, edge_label_index), heuristics], dim=-1)
        struct_pair = torch.cat([pairwise_compose(struct_repr, edge_label_index), heuristics], dim=-1)
        attr_summary = self.attr_pair_proj(attr_pair)
        struct_summary = self.struct_pair_proj(struct_pair)
        heuristic_token = self.heuristic_proj(heuristics)
        cross_delta = self.cross_delta_proj(
            torch.cat([torch.abs(attr_u - struct_u), torch.abs(attr_v - struct_v)], dim=-1)
        )
        cross_mean = self.cross_mean_proj(
            torch.cat([0.5 * (attr_u + attr_v), 0.5 * (struct_u + struct_v)], dim=-1)
        )

        tokens = torch.stack(
            [
                attr_u,
                attr_v,
                struct_u,
                struct_v,
                attr_summary,
                struct_summary,
                cross_delta,
                cross_mean,
                heuristic_token,
            ],
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens)

        weights = torch.softmax(self.token_score(tokens).squeeze(-1), dim=-1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        refined = self.refine(torch.cat([base_fused, attr_summary + struct_summary, pooled], dim=-1)) + base_fused
        expert_logits = self.score_head(refined).view(-1)
        edge_alpha = torch.sigmoid(
            self.edge_gate(torch.cat([attr_summary, struct_summary, pooled, base_fused], dim=-1))
        ).view(-1)
        return expert_logits, pooled, edge_alpha


class ASLAM(nn.Module):
    """Baseline ASLAM-style dual-branch fusion."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heuristic_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attribute_encoder = MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)
        self.structure_encoder = ResidualGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        pair_dim = hidden_channels * 4 + heuristic_dim
        self.attribute_scorer = MLP([pair_dim, hidden_channels, 1], dropout=dropout)
        self.structure_scorer = MLP([pair_dim, hidden_channels, 1], dropout=dropout)
        self.fusion_gate = MLP([pair_dim * 2, hidden_channels, hidden_channels], dropout=dropout)
        self.fusion_scorer = MLP([hidden_channels, hidden_channels, 1], dropout=dropout)

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attr_repr = self.attribute_encoder(x)
        struct_repr = self.structure_encoder(x, edge_index)
        attr_pair = torch.cat([pairwise_compose(attr_repr, edge_label_index), heuristics], dim=-1)
        struct_pair = torch.cat([pairwise_compose(struct_repr, edge_label_index), heuristics], dim=-1)

        attr_logits = self.attribute_scorer(attr_pair).view(-1)
        struct_logits = self.structure_scorer(struct_pair).view(-1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([attr_pair, struct_pair], dim=-1)))
        fused = gate * struct_pair[:, : gate.size(-1)] + (1.0 - gate) * attr_pair[:, : gate.size(-1)]
        fused_logits = self.fusion_scorer(fused).view(-1)
        return attr_logits, struct_logits, fused_logits, fused

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attr_logits, struct_logits, fused_logits, _ = self._forward_impl(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            heuristics=heuristics,
        )
        return attr_logits, struct_logits, fused_logits

    def encode_edge_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            heuristics=heuristics,
        )[3]


class ASLAMPlus(nn.Module):
    """ASLAM with multi-scale selective state-space mixing and pairwise attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heuristic_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attribute_encoder = MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)
        self.base_structure_encoder = ResidualGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.local_global_encoder = LocalGlobalGraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.selective_mixer = BiDirectionalSelectiveMixer(hidden_channels, dropout=dropout)
        pair_dim = hidden_channels * 4 + heuristic_dim
        self.attribute_scorer = MLP([pair_dim, hidden_channels, 1], dropout=dropout)
        self.structure_scorer = MLP([pair_dim, hidden_channels, 1], dropout=dropout)
        self.node_fusion = MLP([hidden_channels * 3, hidden_channels, hidden_channels], dropout=dropout)
        self.base_fusion_gate = MLP([pair_dim * 2, hidden_channels, hidden_channels], dropout=dropout)
        self.base_fusion_scorer = MLP([hidden_channels, hidden_channels, 1], dropout=dropout)
        self.pair_attention = PairwiseTokenAttention(
            hidden_channels=hidden_channels,
            heuristic_dim=heuristic_dim,
            heads=4,
            dropout=dropout,
        )
        self.residual_fusion = MLP([hidden_channels * 2, hidden_channels, hidden_channels], dropout=dropout)
        self.attn_alpha = nn.Parameter(torch.tensor(0.25))

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attr_repr = self.attribute_encoder(x)
        base_struct_repr = self.base_structure_encoder(x, edge_index)
        local_global_repr, multi_scale = self.local_global_encoder(x, edge_index)
        scale_repr = self.selective_mixer(multi_scale)
        struct_repr = self.node_fusion(
            torch.cat([base_struct_repr, local_global_repr, scale_repr], dim=-1)
        ) + base_struct_repr

        attr_pair = torch.cat([pairwise_compose(attr_repr, edge_label_index), heuristics], dim=-1)
        struct_pair = torch.cat([pairwise_compose(struct_repr, edge_label_index), heuristics], dim=-1)
        attr_logits = self.attribute_scorer(attr_pair).view(-1)
        struct_logits = self.structure_scorer(struct_pair).view(-1)
        base_gate = torch.sigmoid(self.base_fusion_gate(torch.cat([attr_pair, struct_pair], dim=-1)))
        base_fused = base_gate * struct_pair[:, : base_gate.size(-1)] + (1.0 - base_gate) * attr_pair[:, : base_gate.size(-1)]
        base_logits = self.base_fusion_scorer(base_fused).view(-1)
        attn_logits, attn_pooled = self.pair_attention(struct_repr + attr_repr, edge_label_index, heuristics)
        fused_context = self.residual_fusion(torch.cat([base_fused, attn_pooled], dim=-1)) + base_fused
        fusion_gain = self.base_fusion_scorer(fused_context).view(-1)
        alpha = torch.sigmoid(self.attn_alpha)
        fused_logits = (1.0 - alpha) * base_logits + alpha * 0.5 * (attn_logits + fusion_gain)
        return attr_logits, struct_logits, fused_logits, fused_context

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attr_logits, struct_logits, fused_logits, _ = self._forward_impl(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            heuristics=heuristics,
        )
        return attr_logits, struct_logits, fused_logits

    def encode_edge_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            heuristics=heuristics,
        )[3]


class ASLAMSSMAttn(nn.Module):
    """ASLAM-Plus backbone with cross-branch SSM-attention fusion for formal comparison."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heuristic_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attribute_encoder = MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)
        self.base_structure_encoder = ResidualGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.local_global_encoder = LocalGlobalGraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.selective_mixer = BiDirectionalSelectiveMixer(hidden_channels, dropout=dropout)
        self.node_fusion = MLP([hidden_channels * 3, hidden_channels, hidden_channels], dropout=dropout)
        pair_dim = hidden_channels * 4 + heuristic_dim
        self.attribute_scorer = MLP([pair_dim, hidden_channels, 1], dropout=dropout)
        self.structure_scorer = MLP([pair_dim, hidden_channels, 1], dropout=dropout)
        self.base_fusion_gate = MLP([pair_dim * 2, hidden_channels, hidden_channels], dropout=dropout)
        self.base_fusion_scorer = MLP([hidden_channels, hidden_channels, 1], dropout=dropout)
        self.pair_attention = PairwiseTokenAttention(
            hidden_channels=hidden_channels,
            heuristic_dim=heuristic_dim,
            heads=4,
            dropout=dropout,
        )
        self.ssm_pair_fusion = SSMAttentionPairFusion(
            hidden_channels=hidden_channels,
            heuristic_dim=heuristic_dim,
            dropout=dropout,
        )
        self.residual_fusion = MLP([hidden_channels * 3, hidden_channels, hidden_channels], dropout=dropout)
        self.fusion_scorer = MLP([hidden_channels, hidden_channels, 1], dropout=dropout)

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attr_repr = self.attribute_encoder(x)
        base_struct_repr = self.base_structure_encoder(x, edge_index)
        local_global_repr, multi_scale = self.local_global_encoder(x, edge_index)
        scale_repr = self.selective_mixer(multi_scale)
        struct_repr = self.node_fusion(
            torch.cat([base_struct_repr, local_global_repr, scale_repr], dim=-1)
        ) + base_struct_repr

        attr_pair = torch.cat([pairwise_compose(attr_repr, edge_label_index), heuristics], dim=-1)
        struct_pair = torch.cat([pairwise_compose(struct_repr, edge_label_index), heuristics], dim=-1)
        attr_logits = self.attribute_scorer(attr_pair).view(-1)
        struct_logits = self.structure_scorer(struct_pair).view(-1)

        base_gate = torch.sigmoid(self.base_fusion_gate(torch.cat([attr_pair, struct_pair], dim=-1)))
        base_fused = base_gate * struct_pair[:, : base_gate.size(-1)] + (1.0 - base_gate) * attr_pair[:, : base_gate.size(-1)]
        base_logits = self.base_fusion_scorer(base_fused).view(-1)

        attn_logits, attn_pooled = self.pair_attention(attr_repr + struct_repr, edge_label_index, heuristics)
        expert_logits, pair_context, edge_alpha = self.ssm_pair_fusion(
            attr_repr=attr_repr,
            struct_repr=struct_repr,
            edge_label_index=edge_label_index,
            heuristics=heuristics,
            base_fused=base_fused,
        )
        fused_context = self.residual_fusion(
            torch.cat([base_fused, pair_context, attn_pooled], dim=-1)
        ) + base_fused
        fusion_logits = self.fusion_scorer(fused_context).view(-1)
        expert_mix = (attn_logits + expert_logits + fusion_logits) / 3.0
        fused_logits = (1.0 - edge_alpha) * base_logits + edge_alpha * expert_mix
        return attr_logits, struct_logits, fused_logits, fused_context

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attr_logits, struct_logits, fused_logits, _ = self._forward_impl(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            heuristics=heuristics,
        )
        return attr_logits, struct_logits, fused_logits

    def encode_edge_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        heuristics: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(
            x=x,
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            heuristics=heuristics,
        )[3]
