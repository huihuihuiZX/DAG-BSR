
import model.common as common
from model.DAGBSR.SR import SR
from model.moco import MoCo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import  add_self_loops

class GNN(nn.Module):
    def __init__(self, in_channels, heads=8, dropout=0.1, add_self_loops=True):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels % heads == 0
        self.hidden_channels_per_head = in_channels // heads
        self.out_channels = in_channels
        self.add_self_loops = add_self_loops
        self.gat_conv = GATConv(in_channels=in_channels,
                                out_channels=self.hidden_channels_per_head,
                                heads=heads,
                                dropout=dropout,
                                concat=True,
                                add_self_loops=False)
        self.norm = nn.LayerNorm(self.out_channels)

    def build_grid_edge_index(self, H, W, device):
        num_nodes = H * W
        rows, cols = [], []
        for r in range(H):
            for c in range(W):
                idx = r * W + c
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            n_idx = nr * W + nc
                            rows.append(idx)
                            cols.append(n_idx)
        edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
        if self.add_self_loops:
             edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        return edge_index

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        assert C == self.in_channels
        edge_index = self.build_grid_edge_index(H, W, x.device)
        x_nodes = x.flatten(2).permute(0, 2, 1)
        x_nodes_batched = x_nodes.reshape(B * N, C)
        num_edges_per_graph = edge_index.size(1)
        offsets = torch.arange(0, B, device=x.device) * N
        batched_edge_index = edge_index.repeat(1, B)
        batch_offsets_row = offsets.repeat_interleave(num_edges_per_graph)
        batch_offsets_col = offsets.repeat_interleave(num_edges_per_graph)
        batched_edge_index[0, :] += batch_offsets_row
        batched_edge_index[1, :] += batch_offsets_col
        x_gat_out = self.gat_conv(x_nodes_batched, batched_edge_index)
        x_gat_out = F.elu(x_gat_out)
        x_gat_out_seq = x_gat_out.view(B, N, self.out_channels)
        x_gat_out_norm = self.norm(x_gat_out_seq)
        x_out = x_gat_out_norm.permute(0, 2, 1).view(B, self.out_channels, H, W)
        return x_out

class GNNEncoder(nn.Module):
    def __init__(self, gnn_in_channels=128, gnn_heads=8):
        super(GNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.1, True)
        self.resblocks1 = nn.Sequential(
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
        )
        self.resblocks2 = nn.Sequential(
            common.ResBlock(common.default_conv, 64, kernel_size=3),
            common.ResBlock(common.default_conv, 64, kernel_size=3),
        )
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.1, True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU(0.1, True)
        self.gnn = GNN(in_channels=gnn_in_channels,
                       heads=gnn_heads,
                       dropout=0.1)
        self.conv4_downsample = nn.Conv2d(gnn_in_channels, 256, kernel_size=3, stride=2, padding=1) # 输入通道是 GNN 输出通道
        self.relu4 = nn.LeakyReLU(0.1, True)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 投影头 MLP (不变)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.resblocks1(x)
        x = self.resblocks2(x)
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x_gnn = self.gnn(x)
        x = x + x_gnn
        x = self.relu4(self.conv4_downsample(x))
        feature = self.pool(x)
        feature = feature.squeeze(-1).squeeze(-1)
        out = self.mlp(feature)
        return feature, out

class DAGBSR(nn.Module):
    def __init__(self, base_encoder=GNNEncoder):
        super(DAGBSR, self).__init__()
        # Generator
        self.G = SR(
            upscale=4,
            in_chans=3,
            img_size=[64, 64],
            window_size=16,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=4,
            upsampler='pixelshuffle',
            resi_connection='1conv',
            graph_flags=[1, 1, 1, 1, 1, 1],
            stage_spec=[['L', 'G', 'L', 'G', 'L', 'G'], ['L', 'G', 'L', 'G', 'L', 'G'],
                        ['L', 'G', 'L', 'G', 'L', 'G'], ['L', 'G', 'L', 'G', 'L', 'G'],
                        ['L', 'G', 'L', 'G', 'L', 'G'], ['L', 'G', 'L', 'G', 'L', 'G']],
            dist_type='cossim',
            top_k=256,
            head_wise=0,
            sample_size=16,
            graph_switch=1,
            flex_type='interdiff_plain',
            FFNtype='basic-dwconv3',
            conv_scale=0.01,
            conv_type='dwconv3-gelu-conv1-ca',
            diff_scales=[10, 1.5, 1.5, 1.5, 1.5, 1.5],
            fast_graph=1
        )
        self.M = MoCo(base_encoder=base_encoder)

    def forward(self, x):
        if self.training:
            x_query = x[:, 0, ...]
            x_key = x[:, 1, ...]
            dp, logits, labels = self.M(x_query, x_key)
            sr = self.G(x_query, dp)
            return sr, logits, labels
        else:
            dp = self.M(x, x)
            sr = self.G(x, dp)
            return sr
