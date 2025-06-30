import math, os
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import numpy as np
import einops
from model.DAGBSR.graph_utils import flex, cossim, local_sampling, global_sampling
list_to_save = list()

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30, conv_type=''):
        super(CAB, self).__init__()
        self.num_feat, self.compress_ratio, self.squeeze_factor = num_feat, compress_ratio, squeeze_factor
        if conv_type == '':
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(*self.block_selection(conv_type))

    def block_selection(self, conv_type: str):
        self.conv_type = conv_type
        conv_types = conv_type.split('-')
        keep_dim = ('dw' in conv_type) or (conv_type.count('conv') < 2)

        dims = [self.num_feat, self.num_feat // (self.compress_ratio if not keep_dim else 1), self.num_feat]
        conv_num = 0
        blocks = list()
        for name in conv_types:
            if name == 'ca':
                break
            elif name == 'gelu':
                blocks.append(nn.GELU())
            elif name.startswith('conv'):
                blocks.append(nn.Conv2d(dims[conv_num], dims[conv_num + 1], int(name[-1]), 1, (int(name[-1]) - 1) // 2))
                conv_num += 1
            elif name.startswith('dwconv'):
                blocks.append(nn.Conv2d(dims[conv_num], dims[conv_num + 1], int(name[-1]), 1, (int(name[-1]) - 1) // 2,
                                        groups=dims[conv_num]))
                conv_num += 1
        blocks.append(ChannelAttention(self.num_feat, self.squeeze_factor))
        return blocks

    def forward(self, x):
        return self.cab(x)



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class dwconv(nn.Module):
    def __init__(self, hidden_features, tp='dwconv5'):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=int(tp[-1]), stride=1,
                      padding=(int(tp[-1]) - 1) // 2, dilation=1,
                      groups=hidden_features if tp.startswith('dw') else 1), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features, self.hidden_features = in_features, hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.before_add = nn.Identity()
        self.after_add = nn.Identity()
        if kwargs.get('FFNtype') is None:
            self.kernel_size = 5
            self.dwconv = dwconv(hidden_features=hidden_features)
        elif kwargs.get('FFNtype') == 'none':
            self.kernel_size = 0
            self.dwconv = nn.Identity()
        elif kwargs.get('FFNtype').startswith('basic'):
            self.kernel_size = int(kwargs.get('FFNtype')[-1])  # figure out kernel size
            self.dwconv = dwconv(hidden_features=hidden_features, tp=kwargs.get('FFNtype').split('-')[-1])
        else:
            raise NotImplementedError(f'FFNType {(kwargs.get("FFNtype"))} not implemented!')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = self.before_add(x)
        if self.kernel_size > 0:
            x = x + self.dwconv(x, x_size)
        x = self.after_add(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Aggregator(nn.Module):

    def __init__(self, dim, window_size, num_heads, bias=True, proj_drop=0.,
                 unfold_dict=None, head_wise=None, top_k=None, **kwargs):

        super().__init__()
        self.dim = dim
        self.group_size = window_size
        self.num_heads = num_heads
        self.unfold_dict = unfold_dict
        self.head_wise = head_wise
        self.top_k = top_k
        self.sample_size = unfold_dict['kernel_size']
        self.graph_switch = kwargs.get('graph_switch', True)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.proj_group = nn.Linear(dim, dim, bias=bias)
        self.proj_sample = nn.Linear(dim, dim * 2, bias=bias)

        self.proj = nn.Linear(dim, dim)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.sample_size[0] - 1), self.group_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.sample_size[1] - 1), self.group_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
        relative_coords_table[:, :, :, 0] /= (self.group_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.group_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        relative_position_index = self.get_rel_pos_index()
        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = None

    def get_rel_pos_index(self):
        group_size = self.group_size
        sample_size = self.unfold_dict['kernel_size']

        coords_grid = torch.stack(
            torch.meshgrid([torch.arange(group_size[0]), torch.arange(group_size[1])], indexing='ij'))
        coords_grid_flatten = torch.flatten(coords_grid, 1)

        coords_sample = torch.stack(
            torch.meshgrid([torch.arange(sample_size[0]), torch.arange(sample_size[1])], indexing='ij'))
        coords_sample_flatten = torch.flatten(coords_sample, 1)

        relative_coords = coords_sample_flatten[:, None, :] - coords_grid_flatten[:, :, None]

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += group_size[0] - sample_size[0] + 1
        relative_coords[:, :, 0] *= group_size[1] + sample_size[1] - 1
        relative_coords[:, :, 1] += group_size[1] - sample_size[1] + 1

        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def rel_pos_bias(self):
        if self.training and self.relative_position_bias_table is not None:
            self.relative_position_bias_table = None  # clear

        if not self.training and self.relative_position_bias_table is not None:
            relative_position_bias_table = self.relative_position_bias_table
        else:
            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        # store
        if not self.training and self.relative_position_bias_table is None:
            self.relative_position_bias_table = relative_position_bias_table

        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.group_size[0] * self.group_size[1], self.sample_size[0] * self.sample_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias.unsqueeze(0)

    def get_correlation(self, x1, x2, graph):
        scale = torch.exp(torch.clamp(self.logit_scale, max=4.6052))
        if self.graph_switch:
            assert (x1.size(-2) == graph.size(-2)) and (x2.size(-2) == graph.size(-1))

        sim = cossim(x1, x2, graph=graph if self.graph_switch else None)

        sim = sim * scale + self.rel_pos_bias()

        sim = F.softmax(sim, dim=-1)

        return sim

    def forward(self, x_complete, graph=None, sampling_method=0):

        if sampling_method == 0:
            x = local_sampling(x_complete, group_size=self.group_size, unfold_dict=None, output=0, tp='bhwc')
        else:
            x = global_sampling(x_complete, group_size=self.group_size, sample_size=None, output=0, tp='bhwc')

        b_, n, c = x.shape
        x1 = einops.rearrange(self.proj_group(x), 'b n (h c) -> b h n c', b=b_, n=n, h=self.num_heads)

        if sampling_method == 0:
            x_sampled = local_sampling(self.proj_sample(x_complete), group_size=self.group_size,
                                       unfold_dict=self.unfold_dict, output=1, tp='bhwc')
        else:
            x_sampled = global_sampling(self.proj_sample(x_complete), group_size=self.group_size,
                                        sample_size=self.sample_size, output=1, tp='bhwc')

        x2, feat = einops.rearrange(x_sampled, 'b n (div h c) -> div b h n c', div=2, h=self.num_heads,
                                    c=c // self.num_heads)
        corr = self.get_correlation(x1, x2, graph)
        x = (corr @ feat).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, top_k={self.top_k}, ' \
               f'sample_size={self.sample_size}'


class MGAU(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, sampling_method=0,
                 mlp_ratio=4., bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.sampling_method = sampling_method
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.aggregator = Aggregator(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            bias=bias, proj_drop=drop, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, **kwargs)
        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.conv_scale = kwargs.get('conv_scale') or 0
        compress_ratio = kwargs.get('compress_ratio') or 3
        squeeze_factor = kwargs.get('squeeze_factor') or 30
        conv_type = kwargs.get('conv_type') or ''
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
                              conv_type=conv_type) if self.conv_scale != 0 else None
    def forward(self, x, x_size, graph):
        H, W = x_size
        B, _, C = x.shape
        shortcut = x
        x = x.view(B, H, W, C)
        conv_x = self.conv_block(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous().view(B, H * W, C) if self.conv_scale != 0 else 0
        x = self.aggregator(x, graph=graph[0] if self.sampling_method == 0 else graph[1],
                            sampling_method=self.sampling_method)
        if self.sampling_method:
            x = einops.rearrange(x, '(b numh numw) (sh sw) c -> b (sh numh sw numw) c', numh=H // self.window_size,
                                 numw=W // self.window_size, sh=self.window_size, sw=self.window_size)
        else:
            x = einops.rearrange(x, '(b numh numw) (sh sw) c -> b (numh sh numw sw) c', numh=H // self.window_size,
                                 numw=W // self.window_size, sh=self.window_size, sw=self.window_size)

        x = shortcut + self.drop_path(self.norm1(x)) + conv_x * self.conv_scale  # Channel Attention
        x = x + self.drop_path(self.norm2(self.mlp(x, x_size)))
        return x
    def extra_repr(self) -> str:
        return f"dim={self.dim}, sampling_method={self.sampling_method}, mlp_ratio={self.mlp_ratio}"



class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'



class DRFfusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kernel = nn.Sequential(nn.Linear(256, dim * 2, bias=False),)

    def forward(self, x, cdp):
        B, N, C = x.shape
        cdp = self.kernel(cdp).view(-1, 1, C * 2)
        cdp1, cdp2 = cdp.chunk(2, dim=2)
        x = x * cdp1 + cdp2
        return x


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 bias=True,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, stage_idx=None, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        stages = kwargs.get('stage_spec')[stage_idx]

        blocks = []
        for i in range(depth):
            if stages[i] == 'L':
                block = MGAU(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    sampling_method=0,
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, **kwargs
                )
            elif stages[i] == 'G':
                block = MGAU(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    sampling_method=1,
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, **kwargs
                )

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(dim)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    def forward(self, x, x_size, graph, dp):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size, graph)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, depth={self.depth}'



class AGCU(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 bias=True,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv', stage_idx=None, **kwargs):
        super(AGCU, self).__init__()
        self.kwargs = kwargs

        self.dim = dim
        self.input_resolution = input_resolution
        self.dr_f_fusion = DRFfusion(dim)
        self.window_size = window_size
        self.sample_size = kwargs.get('sample_size')
        self.padding_size = (self.sample_size - self.window_size) // 2
        self.unfold_dict = dict(kernel_size=(self.sample_size, self.sample_size), stride=(window_size, window_size),
                                padding=(self.padding_size, self.padding_size))

        self.num_head = num_heads
        self.graph_flag = kwargs.get('graph_flags')[stage_idx]
        self.head_wise = kwargs.get('head_wise', 0)
        self.dist_type = kwargs.get('dist_type')

        self.fast_graph = kwargs.get('fast_graph', 1)

        self.dist = cossim
        self.top_k = kwargs.get('top_k')[stage_idx] if isinstance(kwargs.get('top_k'), list) else kwargs.get('top_k')
        self.flex_type = kwargs.get('flex_type')
        self.graph_switch = kwargs.get('graph_switch')

        self.stage_idx = stage_idx
        self.output_folder = kwargs.get('output_folder')
        self.diff_scale = kwargs.get('diff_scales')[stage_idx] if kwargs.get(
            'diff_scales') is not None else None

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            bias=bias,
            drop=drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint, stage_idx=stage_idx, unfold_dict=self.unfold_dict, **kwargs)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.tensors = None
        self.tolerance = kwargs.get('tolerance', 8)

    def DS(self, x, shape=(80, 80), scale=2, he=1):
        B, _, C = x.shape
        H, W = shape
        x_rs = x.view(B, H, W, C // he, he).mean(-1).permute(0, 3, 1, 2)
        ##
        return (x_rs - F.interpolate(
            F.interpolate(x_rs, (H // scale, W // scale), mode='bilinear', align_corners=False), (H, W),
            mode='bilinear', align_corners=False)).abs().sum(dim=1)

    @torch.no_grad()
    def calc_graph(self, x_, x_size):
        if self.output_folder is not None:
            list_to_save.append(x_.cpu())
        if not self.graph_switch:
            return None, None

        # prepare const tensors
        if self.fast_graph and self.tensors is None:
            self.tensors = (
                torch.tensor([
                    [0.5, 1., 0.],
                    [0., 0., 0.],
                    [0.5, 0., 1.],
                ], dtype=torch.float32).to(x_.device),
                torch.tensor([
                    [0.5, 0., 1.],
                    [0.5, 1., 0.],
                    [0., 0., 0.],
                ], dtype=torch.float32).to(x_.device)
            )

        ''' Added: x_diff for inter diff_plain'''
        ds = [None, None]
        if self.flex_type.startswith('interdiff'):
            ds = self.DS(x_, x_size)
            if (self.diff_scale is not None) and (self.diff_scale != 0):
                mu = ds.mean(dim=(-2, -1), keepdim=True)  # (b 1 1)
                ds = mu + (ds - mu) / self.diff_scale
            ds = [
                einops.rearrange(ds, 'b (numh wh) (numw ww)-> (b numh numw) (wh ww)', wh=self.window_size,
                                 ww=self.window_size),
                einops.rearrange(ds, 'b (sh numh) (sw numw) -> (b numh numw) (sh sw)', sh=self.window_size,
                                 sw=self.window_size)
            ]

        graph0 = self.calc_graph_(x_, x_size, sampling_method=0, X_diff=ds[0])
        graph1 = self.calc_graph_(x_, x_size, sampling_method=1, X_diff=ds[1])
        return graph0, graph1

    @torch.no_grad()
    def calc_graph_(self, x_, x_size, sampling_method=0, X_diff=None):
        he = self.num_head if self.head_wise else 1
        x = einops.rearrange(x_, 'b (h w) c -> b c h w', h=x_size[0], w=x_size[1])
        if sampling_method:
            X_sample, Y_sample = global_sampling(x, group_size=self.window_size, sample_size=self.sample_size, output=2,
                                                 tp='bchw')
        else:  # dense local
            X_sample, Y_sample = local_sampling(x, group_size=self.window_size, unfold_dict=self.unfold_dict, output=2,
                                                tp='bchw')
        assert X_sample.size(0) == Y_sample.size(0)
        D = self.dist(X_sample.unsqueeze(1), Y_sample.unsqueeze(1)).squeeze(1)  # (b m n)
        if self.fast_graph:
            maskarray = (X_diff / X_diff.sum(dim=-1, keepdim=True)) * D.size(1) * self.top_k
            maskarray = torch.clamp(maskarray, 1, D.size(-1))
            minbound = torch.min(D, dim=-1, keepdim=True)[0]
            maxbound = torch.ones_like(minbound)
            wall = D.mean(dim=-1, keepdim=True)
            MAT = torch.cat([wall, minbound, maxbound], dim=-1)

            for _ in range(self.tolerance):
                allocated = (D > MAT[..., 0:1]).sum(dim=-1)
                MAT = torch.where(
                    (allocated > maskarray).unsqueeze(-1),
                    MAT @ self.tensors[0],
                    MAT @ self.tensors[1],
                )

            graph = (D > MAT[..., 0:1]).unsqueeze(1)
        else:
            val, idx = D.sort(dim=-1, descending=True)
            b, m, n = idx.shape

            mask = flex(D, X_sample, idx, self.flex_type, self.top_k, self.kwargs['model'].current_iter,
                        self.kwargs['model'].total_iters, X_diff, fast=True)
            if not self.head_wise:
                idx = idx.unsqueeze(1).expand(b, 1, m, n)
                mask = mask.unsqueeze(1).expand(b, 1, m, n)
            else:
                idx = einops.rearrange(idx, '(b he) m n -> b he m n', he=he)
                mask = einops.rearrange(mask, '(b he) m n -> b he m n', he=he)
            original_shape = idx.shape
            b_coord = torch.arange(idx.size(0), device=idx.device).int().view(-1, 1, 1, 1) * np.prod(original_shape[1:])
            he_coord = torch.arange(idx.size(1), device=idx.device).int().view(1, -1, 1, 1) * np.prod(
                original_shape[2:])
            m_coord = torch.arange(idx.size(2), device=idx.device).int().view(1, 1, -1, 1) * original_shape[3]

            overall_coord = b_coord + he_coord + m_coord + idx
            selected_coord = torch.masked_select(overall_coord, mask)
            graph = torch.ones_like(idx).bool()
            graph.view(-1)[selected_coord] = False
            if self.output_folder is not None:
                list_to_save.append(graph.cpu())
        return graph

    def forward(self, x, x_size, prev_graph=None, dp=None):
        short_cut = x
        x = self.dr_f_fusion(self.norm(x), dp) + short_cut
        graph = self.calc_graph(x, x_size) if self.graph_flag else prev_graph
        return self.patch_embed(
            self.conv(self.patch_unembed(self.residual_group(x, x_size, graph, dp), x_size))) + x, graph


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)##完整图像尺寸
        patch_size = to_2tuple(patch_size)##图像块尺寸
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]##图像块在HW上的数量
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]##图像块的数量

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c ##将x展平，然后转置
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        self.scale = scale
        self.num_feat = num_feat
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class SR(nn.Module):

    def __init__(self,
                 img_size=None,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=180,
                 depths=[6, 6, 6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=16,
                 mlp_ratio=4.,
                 bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 **kwargs):
        super(SR, self).__init__()
        if img_size is None:
            img_size = [64, 64]
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)##norm_layer=nn.LayerNorm

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        ''' Intermediate outputs '''
        self.output_folder = kwargs.get('output_folder')

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = AGCU(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                bias=bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection, stage_idx=i_layer, **kwargs)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, dp):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        prev_graph = None
        for layer in self.layers:
            x, prev_graph = layer(x, x_size, prev_graph, dp)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, dp):
        if (self.output_folder is not None):
            global list_to_save
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder, exist_ok=True)
            if len(os.listdir(self.output_folder)) > 0:
                output_idx = max([int(i[:-4]) if i.endswith('.pkl') and i[:-4].isdecimal() else -1 for i in
                                  os.listdir(self.output_folder)]) + 1
            else:
                output_idx = 0
            list_to_save.append(x.cpu())
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, dp)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        if self.output_folder is not None:
            list_to_save.append(x.cpu())
            torch.save(list_to_save, os.path.join(self.output_folder, str(output_idx) + '.pkl'))
            list_to_save = list()

        return x

