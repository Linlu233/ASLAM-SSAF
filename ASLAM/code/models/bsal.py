from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, Linear, MaxPool1d, ModuleList
from torch_geometric.nn import GCNConv, global_sort_pool


class DGCNN(torch.nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers, GNN=GCNConv, k=0.6):
        super().__init__()
        if k < 1:
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        self.convs = ModuleList()
        self.convs.append(GNN(in_channels, hidden_channels))
        for _ in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.lin1(x)


class ASLAM(torch.nn.Module):
    def __init__(self, train_dataset, train_dataset_label2, in_channels, in_channels_label2, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.dgcnn = DGCNN(train_dataset, in_channels, hidden_channels, 16, num_layers, GNN=GCNConv, k=0.6)
        self.dgcnn_2 = DGCNN(train_dataset_label2, in_channels_label2, hidden_channels, 16, num_layers, GNN=GCNConv, k=0.6)
        self.lin1 = Linear(out_channels, 1)
        self.lin2 = Linear(out_channels, 1)
        self.lin3 = Linear(out_channels, 1)
        self.att_1 = Linear(out_channels, 16)
        self.att_2 = Linear(out_channels, 16)
        self.query = Linear(16, 1)

    def forward(self, data, data_label2, knn_graph):
        emb_label1 = self.dgcnn(data)
        emb_label2 = self.dgcnn_2(data_label2)
        emb_structure = torch.cat([emb_label1, emb_label2], dim=-1)

        emb_semantic = knn_graph.x[data.target_nodes]
        emb_semantic = emb_semantic.view(-1, 2, emb_semantic.size(-1)).sum(dim=1)

        att_structure = self.query(torch.tanh(self.att_1(emb_structure)))
        att_semantic = self.query(torch.tanh(self.att_2(emb_semantic)))
        pair_logits = torch.cat([att_structure, att_semantic], dim=-1)
        pair_alpha = torch.softmax(pair_logits, dim=-1)
        fused = pair_alpha[:, :1] * emb_structure + pair_alpha[:, 1:] * emb_semantic

        output_structure = self.lin1(emb_structure)
        output_semantic = self.lin2(emb_semantic)
        output_fused = self.lin3(fused)
        return output_structure, output_semantic, output_fused
