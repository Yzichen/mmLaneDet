import torch.nn as nn
import torch.nn.functional as F

from mmlane.models import AGGREGATORS
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule
import math


@AGGREGATORS.register_module()
class SpatialCNN(BaseModule):
    def __init__(self, in_channels=512, out_channels=128, mid_channels=None, kernel_size=9):
        super(SpatialCNN, self).__init__()
        self.kernel_size = kernel_size
        if mid_channels is not None:
            self.channel_reducer = nn.Sequential(
                ConvModule(in_channels=in_channels,
                           out_channels=mid_channels,
                           kernel_size=3,
                           stride=1,
                           padding=4,
                           dilation=4,
                           conv_cfg=dict(type='Conv2d'),
                           norm_cfg=dict(type='BN'),
                           act_cfg=dict(type='ReLU')
                           ),
                ConvModule(in_channels=mid_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           stride=1,
                           conv_cfg=dict(type='Conv2d'),
                           norm_cfg=dict(type='BN'),
                           act_cfg=dict(type='ReLU')
                           )
            )
        else:
            self.channel_reducer = ConvModule(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              conv_cfg=dict(type='Conv2d'),
                                              norm_cfg=dict(type='BN'),
                                              act_cfg=dict(type='ReLU')
                                              )

        self.conv_d = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2), bias=False)
        self.conv_u = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, kernel_size//2), bias=False)
        self.conv_r = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(kernel_size//2, 0), bias=False)
        self.conv_l = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(kernel_size//2, 0), bias=False)

        self.init_weights()

    def init_weights(self, num_channels=128):
        # https://github.com/XingangPan/SCNN/issues/82
        bound = math.sqrt(2.0 / (num_channels * self.kernel_size * 5))
        nn.init.uniform_(self.conv_d.weight, -bound, bound)
        nn.init.uniform_(self.conv_u.weight, -bound, bound)
        nn.init.uniform_(self.conv_r.weight, -bound, bound)
        nn.init.uniform_(self.conv_l.weight, -bound, bound)

    # @auto_fp16()
    def forward(self, inputs):
        """
        :param inputs: tuple((B, C2, H2, W2),  (B, C3, H3, W3), ...)
        :return:
        """
        x = inputs[-1]    # (B, C_in, H, W)
        x = self.channel_reducer(x).clone()
        for i in range(1, x.shape[2]):
            x[..., i:i+1, :].add_(F.relu(self.conv_d(x[..., i-1:i, :])))

        for i in range(x.shape[2] - 2, 0, -1):
            x[..., i:i+1, :].add_(F.relu(self.conv_u(x[..., i+1:i+2, :])))

        for i in range(1, x.shape[3]):
            x[..., i:i+1].add_(F.relu(self.conv_r(x[..., i-1:i])))

        for i in range(x.shape[3] - 2, 0, -1):
            x[..., i:i+1].add_(F.relu(self.conv_l(x[..., i+1:i+2])))
        return x
