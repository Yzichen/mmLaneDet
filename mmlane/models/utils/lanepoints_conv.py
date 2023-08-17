import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from ...ops.dcn import DeformConv1D


class LanePointsConv(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=7,
                 gradient_mul=0.1,  # reduce gradient
                 use_latern=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 ):
        super(LanePointsConv, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.use_latern = use_latern
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn_kernel = num_points
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd number.'

        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(0, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, 1)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))  # (14,) [0.0, -3.0,.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3 0.0, -2.0, 0.0]
        self.dcn_base_offset = torch.tensor(dcn_base_offset).reshape(1, -1, 1, 1)   # (1, num_points*2, 1, 1)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=False)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv1D(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)

        if self.use_latern:
            self.implicit_cls_add = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
            nn.init.normal_(self.implicit_cls_add, std=.02)
            self.implicit_pts_add = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
            nn.init.normal_(self.implicit_pts_add, std=.02)
            self.implicit_cls_mul = nn.Parameter(torch.ones(1 , self.in_channels, 1, 1))
            nn.init.normal_(self.implicit_cls_mul, mean=1., std=.02)
            self.implicit_pts_mul = nn.Parameter(torch.ones(1 , self.in_channels, 1, 1))
            nn.init.normal_(self.implicit_pts_mul, mean=1., std=.02)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)

    def forward(self, feats):
        """
        :param feats: (B, C_in, fH, fW)
        :return:

        """
        dcn_base_offset = self.dcn_base_offset.type_as(feats)   # torch.Size([1, 14, 1, 1])
        # reg from kernel grid
        points_init = 0     # reg from center
        if self.use_latern:
            cls_feat = feats*self.implicit_cls_mul+self.implicit_cls_add
            pts_feat = feats*self.implicit_pts_mul+self.implicit_pts_add
        else:
            cls_feat = feats
            pts_feat = feats

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)       # (B, C=256, fH, fW)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)       # (B, C=256, fH, fW)

        # initialize reppoints
        # (B, C=256, fH, fW) --> (B, C=64, fH, fW) --> (B, C=14=num_points*2, fH, fW)
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        # regress from center
        # get kernel position from center
        # (B, C=14=num_points*2, fH, fW) + 0 --> (B, C=14=num_points*2, fH, fW)
        pts_out_init = pts_out_init + points_init

        # reduce the gradient for pts_out_init
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init    # (B, C=14=num_points*2, fH, fW)

        # diff between real position and init position, as the input of deformable convolution
        # (B, C=14=num_points*2, fH, fW) - (1, 14, 1, 1) --> (B, C=14=num_points*2, fH, fW)
        dcn_offset = pts_out_init_grad_mul.contiguous() - dcn_base_offset.contiguous()
        # deformable convolution, feature aggregation from given points
        # (B, C=256, fH, fW) --> (B, C=64, fH, fW)
        feature_out = self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset.contiguous()))

        return feature_out, pts_out_init
