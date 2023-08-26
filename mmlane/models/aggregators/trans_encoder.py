import torch
import torch.nn as nn
from mmlane.models import AGGREGATORS
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule
import math
import time


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        """
        :param mask: (B, H, W)
        :return:
            pos: (B, C=num_feats*2, H, W)
        """
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)   # (B, H, W)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)   # (B, H, W)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)    # (num_feats, )
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)   # (num_feats, )   [10000^(0/128), 10000^(0/128), 10000^(2/128), 10000^(2/128), ...]

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, num_feats)  [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
        pos_y = y_embed[:, :, :, None] / dim_t  # (B, H, W, num_feats)  [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)   # (B, H, W, num_feats/2, 2) --> (B, H, W, num_feats)  num_feats: [sin(pos_x/10000^0/128), cos(pos_x/10000^0/128), sin(pos_x/10000^2/128), cos(pos_x/10000^2/128), ...]
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)   # (B, H, W, num_feats/2, 2) --> (B, H, W, num_feats)  num_feats: [sin(pos_y/10000^0/128), cos(pos_y/10000^0/128), sin(pos_y/10000^2/128), cos(pos_y/10000^2/128), ...]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, H, W, num_feats*2) --> (B, num_feats*2, H, W)
        return pos


def build_position_encoding(hidden_dim, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs


class AttentionLayer(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim, out_dim, ratio=4, stride=1):
        super(AttentionLayer, self).__init__()
        self.chanel_in = in_dim
        norm_cfg = dict(type='BN', requires_grad=True)
        act_cfg = dict(type='ReLU')
        self.pre_conv = ConvModule(
            in_dim,
            out_dim,
            kernel_size=3,  # 3
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.query_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.final_conv = ConvModule(
            out_dim,
            out_dim,
            kernel_size=3,  # 3
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pos=None):
        """
            inputs:
                x: inpput feature maps(B, C, H, W)
                pos: (1, C, H, W)
            returns:
                out: attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x = self.pre_conv(x)
        m_batchsize, _, height, width = x.size()
        if pos is not None:
            x += pos
        proj_query = self.query_conv(x).view(m_batchsize, -1,
                                             width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        attention_t = attention.permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention_t)
        out = out.view(m_batchsize, -1, height, width)
        out_feat = self.gamma * out + x
        out_feat = self.final_conv(out_feat)

        # out_feat = self.final_conv(out)
        return out_feat, attention


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, r=4):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // r, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // r, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         """
#         :param x: (B, C, H, W)
#         :return:
#             ca: (B, C, 1, 1)
#         """
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))      # (B, C, 1, 1)
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))      # (B, C, 1, 1)
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=3):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         """
#         :param x: (B, C, H, W)
#         :return:
#             sa: (B, 1, H, W)
#         """
#         avg_out = torch.mean(x, dim=1, keepdim=True)    # (B, 1, H, W)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
#         x = torch.cat([avg_out, max_out], dim=1)        # (B, 2, H, W)
#         x = self.conv1(x)     # (B, 1, H, W)
#         return self.sigmoid(x)


@AGGREGATORS.register_module()
class TransEncoderModule(BaseModule):
    def __init__(self,
                 attn_in_dims,
                 attn_out_dims,
                 strides,
                 ratios,
                 pos_shape=None,
                 ):
        super(TransEncoderModule, self).__init__()
        attn_layers = []
        for dim1, dim2, stride, ratio in zip(attn_in_dims, attn_out_dims, strides, ratios):
            attn_layers.append(AttentionLayer(dim1, dim2, ratio, stride))

        if pos_shape is not None:
            self.attn_layers = nn.ModuleList(attn_layers)
        else:
            self.attn_layers = nn.Sequential(*attn_layers)

        self.pos_shape = pos_shape
        self.pos_embeds = []
        if pos_shape is not None:
            for dim in attn_out_dims:
                pos_embed = build_position_encoding(dim, pos_shape).cuda()
                self.pos_embeds.append(pos_embed)

        # self.ca = ChannelAttention(in_planes=attn_out_dims[-1], r=4)
        # self.sa = SpatialAttention(kernel_size=3)

    def forward(self, src):
        if self.pos_shape is None:
            src = self.attn_layers(src)
        else:
            for layer, pos in zip(self.attn_layers, self.pos_embeds):
                src, attn = layer(src, pos.to(src.device))

        # ca = self.ca(src)
        # src = ca * src
        # sa = self.sa(src)
        # src = sa * src
        return src
