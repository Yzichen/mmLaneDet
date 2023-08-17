import torch
import torch.nn as nn
import torch.nn.functional as F
from mmlane.models import AGGREGATORS
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule


@AGGREGATORS.register_module()
class PyramidPoolingModule(BaseModule):
    def __init__(self, in_channels, out_channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(
            in_channels + len(sizes) * out_channels, out_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


@AGGREGATORS.register_module()
class PyramidPoolingModuleV2(BaseModule):
    def __init__(self, in_channels, out_channels=64, sizes=(1, 2, 3, 6)):
        super().__init__()
        # self.pre_conv = ConvModule(
        #     in_channels,
        #     out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     norm_cfg=dict(type='BN', requires_grad=True),
        #     act_cfg=dict(type='ReLU'),
        #     inplace=False)
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(
            in_channels + (len(sizes) * out_channels), in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # feats = self.pre_conv(feats)    # (B, C, H, W)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out



