import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class DilatedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 dilation=1,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super(DilatedBottleneck, self).__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            padding=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.conv3 = ConvModule(
            in_channels=mid_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, C, H, W)
        :return:
            out: (B, C, H, W)
        """
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity

        return out


class DilatedBlocks(nn.Module):
    def __init__(self,
                 in_channels=256,
                 mid_channels=64,
                 dilations=[4, 8]
                 ):
        super(DilatedBlocks, self).__init__()
        if isinstance(dilations, int):
            dilations = [dilations, ]

        blocks = []
        for dilation in dilations:
            dilate_bottleneck = DilatedBottleneck(in_channels, mid_channels, dilation)
            blocks.append(dilate_bottleneck)

        self.dilated_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
            out: (B, C, H, W)
        """
        return self.dilated_blocks(x)




